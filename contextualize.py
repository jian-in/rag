"""Contextual Retrieval (Anthropic-style) for the local RAG knowledge base.

Two phases:
    python contextualize.py generate   # LLM-generate a situating context per chunk
                                       # (checkpointed to contexts.json, resumable)
    python contextualize.py apply      # re-embed "context + chunk" and write
                                       # metadata.context_prefix into Chroma

Rollback: chroma_db.bak-pre-contextual (created before first apply).

Notes:
- Stored document text stays ORIGINAL (display/citation unchanged); only the
  embedding gets the context prefix. BM25 picks the context up via metadata
  (see BM25Retriever._document_text).
- Prompt follows Anthropic's "situate this chunk within the document" recipe.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _sanitize_proxy_environment() -> None:
    """Same defense as mcp_http_server: keep localhost local, drop dead proxies."""
    existing = os.getenv("NO_PROXY") or os.getenv("no_proxy") or ""
    entries = [item.strip() for item in existing.split(",") if item.strip()]
    for item in ("localhost", "127.0.0.1", "::1"):
        if item not in entries:
            entries.append(item)
    os.environ["NO_PROXY"] = ",".join(entries)
    broken = {"http://127.0.0.1:9", "https://127.0.0.1:9", "http://localhost:9", "https://localhost:9"}
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        value = os.getenv(key, "").strip().rstrip("/").lower()
        if value in broken:
            os.environ.pop(key, None)


_sanitize_proxy_environment()

import config  # noqa: E402  (loads .env)

CONTEXTS_PATH = ROOT / "contexts.json"
DOC_CHAR_LIMIT = 12000  # whole doc sent if <= this, else head+tail+headings
WORKERS = 3  # gateway key sits in a rate-limited group; keep concurrency gentle
EMBED_BATCH = 20

PROMPT_TEMPLATE = """<document>
{document}
</document>

这是该文档中的一个片段：
<chunk>
{chunk}
</chunk>

请用一两句话简要说明这个片段在整篇文档中的位置与背景（例如属于哪一章/节、讨论什么主题、与上下文的关系），用于提升该片段的检索命中率。只输出这段背景说明本身，不要输出其他内容。若文档为英文则用英文回答，否则用中文回答。"""


def get_collection():
    import chromadb

    chroma_dir = (ROOT / config.CHROMA_PERSIST_DIR).resolve()
    client = chromadb.PersistentClient(path=str(chroma_dir))
    return client.get_collection(config.CHROMA_COLLECTION_NAME)


def load_contexts() -> dict:
    if CONTEXTS_PATH.exists():
        return json.loads(CONTEXTS_PATH.read_text(encoding="utf-8"))
    return {}


def save_contexts(contexts: dict) -> None:
    CONTEXTS_PATH.write_text(
        json.dumps(contexts, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def doc_excerpt(text: str) -> str:
    if len(text) <= DOC_CHAR_LIMIT:
        return text
    headings = [ln for ln in text.splitlines() if ln.startswith("#")][:40]
    return (
        text[:6000]
        + "\n\n...（中间略）...\n\n[全文标题大纲]\n"
        + "\n".join(headings)
        + "\n\n[文档结尾]\n"
        + text[-2000:]
    )


def read_source(source: str, cache: dict) -> str:
    if source not in cache:
        try:
            cache[source] = Path(source).read_text(encoding="utf-8", errors="replace")
        except Exception:
            cache[source] = ""
    return cache[source]


def llm_context(document: str, chunk: str, max_attempts: int = 4) -> str:
    base = (config.OPENAI_API_BASE or "https://api.openai.com/v1").rstrip("/")
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(
                f"{base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": config.LLM_MODEL_NAME,
                    "temperature": 0,
                    "messages": [
                        {"role": "user", "content": PROMPT_TEMPLATE.format(document=document, chunk=chunk)}
                    ],
                },
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # retry with backoff; gateway is rate-limit sensitive
            last_exc = exc
            time.sleep(min(2 ** attempt, 20))
    raise last_exc  # type: ignore[misc]


def generate() -> None:
    coll = get_collection()
    snap = coll.get(include=["documents", "metadatas"])
    ids, docs, metas = snap["ids"], snap["documents"], snap["metadatas"]
    contexts = load_contexts()
    todo = [i for i in range(len(ids)) if ids[i] not in contexts and docs[i]]
    print(f"chunks: {len(ids)}, already done: {len(contexts)}, todo: {len(todo)}")
    if not todo:
        return

    doc_cache: dict = {}
    done_since_save = 0
    started = time.time()

    def work(i: int) -> tuple[str, str]:
        source = (metas[i] or {}).get("source", "")
        document = doc_excerpt(read_source(source, doc_cache))
        if not document.strip():
            document = "(源文件不可读，仅根据文件名推断) 文件: " + source
        return ids[i], llm_context(document, docs[i])

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(work, i): i for i in todo}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                chunk_id, ctx = fut.result()
                contexts[chunk_id] = ctx
            except Exception as exc:
                print(f"  [warn] chunk {ids[i]} failed: {exc}")
            done_since_save += 1
            if done_since_save >= 25:
                save_contexts(contexts)
                done_since_save = 0
                rate = len(contexts) / max(time.time() - started, 1)
                print(f"  progress: {len(contexts)}/{len(ids)} ({rate:.1f}/s)")
    save_contexts(contexts)
    print(f"done: {len(contexts)}/{len(ids)} contexts saved to {CONTEXTS_PATH}")


def embed(texts: list[str]) -> list[list[float]]:
    base = config.EMBEDDING_API_BASE.rstrip("/")
    resp = requests.post(
        f"{base}/embeddings",
        headers={
            "Authorization": f"Bearer {config.EMBEDDING_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": config.EMBEDDING_MODEL_NAME, "input": texts, "encoding_format": "float"},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return [d["embedding"] for d in sorted(data["data"], key=lambda x: x["index"])]


def apply() -> None:
    backup = ROOT / "chroma_db.bak-pre-contextual"
    if not backup.exists():
        print("ABORT: backup chroma_db.bak-pre-contextual not found; back up first.")
        sys.exit(1)

    contexts = load_contexts()
    coll = get_collection()
    snap = coll.get(include=["documents", "metadatas"])
    ids, docs, metas = snap["ids"], snap["documents"], snap["metadatas"]
    missing = [cid for cid in ids if cid not in contexts]
    print(f"chunks: {len(ids)}, contexts: {len(contexts)}, missing: {len(missing)}")
    if missing:
        print("ABORT: run generate until missing = 0 (chunks without context stay un-updated).")
        sys.exit(1)

    for start in range(0, len(ids), EMBED_BATCH):
        batch_ids = ids[start : start + EMBED_BATCH]
        batch_docs = docs[start : start + EMBED_BATCH]
        batch_metas = metas[start : start + EMBED_BATCH]
        prefixed = [f"{contexts[cid]}\n\n{text}" for cid, text in zip(batch_ids, batch_docs)]
        vecs = embed(prefixed)
        new_metas = [
            {**(m or {}), "context_prefix": contexts[cid]}
            for cid, m in zip(batch_ids, batch_metas)
        ]
        coll.update(ids=batch_ids, embeddings=vecs, metadatas=new_metas)
        print(f"  applied {start + len(batch_ids)}/{len(ids)}")

    print(f"done: {coll.count()} chunks now contextually embedded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["generate", "apply"])
    args = parser.parse_args()
    if args.phase == "generate":
        generate()
    else:
        apply()
