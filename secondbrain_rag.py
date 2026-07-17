"""
Second Brain RAG — 基于现有 RAG 基础设施，为 Second Brain 建向量索引 + 搜索 CLI
用法: python secondbrain_rag.py build          # 建库
      python secondbrain_rag.py search "查询"   # 搜索
      python secondbrain_rag.py stats           # 统计
"""
import sys, os, json, re, hashlib
import requests
import chromadb
from dotenv import load_dotenv

load_dotenv()

# ====== 配置 ======
SECONDBRAIN_ROOT = r"C:\Desktop\SecondBrain"
SECONDBRAIN_DIRS = [
    r"C:\Desktop\SecondBrain\02-Areas",
    r"C:\Desktop\SecondBrain\03-Resources",
]
CHROMA_DIR = r"E:\Backup\rag-knowledge-base\chroma_db"
COLLECTION_NAME = "secondbrain"
MANIFEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".secondbrain_manifest.json")

EMBEDDING_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBEDDING_API_KEY = os.getenv("SILICONFLOW_API_KEY") or os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_MODEL = "BAAI/bge-m3"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 20


def embed(texts: list[str]) -> list[list[float]]:
    """批量文本 → 向量"""
    if not EMBEDDING_API_KEY:
        raise ValueError("Missing embedding API key. Set SILICONFLOW_API_KEY or EMBEDDING_API_KEY before running secondbrain_rag.py.")

    r = requests.post(
        EMBEDDING_URL,
        headers={"Authorization": f"Bearer {EMBEDDING_API_KEY}", "Content-Type": "application/json"},
        json={"model": EMBEDDING_MODEL, "input": texts, "encoding_format": "float"},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    return [d["embedding"] for d in sorted(data["data"], key=lambda x: x["index"])]


def chunk_markdown(text: str, source: str) -> list[dict]:
    """Markdown 按 ## 标题切块，再按长度细分"""
    sections = re.split(r"\n(?=## )", text)
    chunks = []
    for sec in sections:
        # 取标题做预览
        title_match = re.match(r"^#+\s*(.+)$", sec, re.MULTILINE)
        title = title_match.group(1) if title_match else source
        # 超过 CHUNK_SIZE 则按段落再切
        if len(sec) > CHUNK_SIZE:
            paras = sec.split("\n\n")
            buf = ""
            for p in paras:
                if len(buf) + len(p) > CHUNK_SIZE and buf:
                    chunks.append({"text": buf.strip(), "source": source, "title": title})
                    buf = p
                else:
                    buf += ("\n\n" + p) if buf else p
            if buf.strip():
                chunks.append({"text": buf.strip(), "source": source, "title": title})
        else:
            chunks.append({"text": sec.strip(), "source": source, "title": title})
    return chunks


def file_hash(path: str) -> str:
    """文件内容的 md5，用于判断是否发生变化"""
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def scan_files(base_dir: str) -> dict[str, str]:
    """递归扫描 .md 文件，返回 {相对路径: 内容hash}"""
    manifest = {}
    for root, dirs, files in os.walk(base_dir):
        # 跳过隐藏目录和非笔记目录
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if f.endswith(".md"):
                path = os.path.join(root, f)
                rel = os.path.relpath(path, base_dir)
                try:
                    manifest[rel] = file_hash(path)
                except Exception:
                    continue
    return manifest


def load_manifest() -> dict[str, str]:
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}
    return {}


def save_manifest(manifest: dict[str, str]):
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)


def load_file_chunks(base_dir: str, rel_path: str) -> list[dict]:
    """加载单个文件并分块"""
    path = os.path.join(base_dir, rel_path)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception:
        return []
    return chunk_markdown(text, rel_path)


def build(full: bool = False):
    """
    增量扫描 Second Brain → 分块 → 嵌入 → 存入 ChromaDB

    只对新增/修改过的文件重新 embed，未变化的文件跳过。
    传 full=True（或 CLI 加 --full）可强制全量重建。
    """
    print(f"📂 扫描 {SECONDBRAIN_DIR} ...")
    new_manifest = scan_files(SECONDBRAIN_DIR)
    old_manifest = {} if full else load_manifest()

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if full:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        coll = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    else:
        try:
            coll = client.get_collection(COLLECTION_NAME)
        except Exception:
            coll = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    changed = [rel for rel, h in new_manifest.items() if old_manifest.get(rel) != h]
    removed = [rel for rel in old_manifest if rel not in new_manifest]

    print(f"🔍 变更: {len(changed)} 个文件, 删除: {len(removed)} 个文件, 未变: {len(new_manifest) - len(changed)} 个文件")

    if not changed and not removed:
        print("✨ 没有变化，无需重新构建")
        return

    # 删除已移除文件和已变更文件的旧 chunk（按 source 前缀匹配 id）
    for rel in removed + changed:
        try:
            existing = coll.get(where={"source": rel})
            if existing["ids"]:
                coll.delete(ids=existing["ids"])
        except Exception:
            pass

    if changed:
        all_chunks = []
        for rel in changed:
            all_chunks.extend(load_file_chunks(SECONDBRAIN_DIR, rel))
        print(f"📄 {len(all_chunks)} 个新/改动 chunks 待嵌入")

        for i in range(0, len(all_chunks), BATCH_SIZE):
            batch = all_chunks[i : i + BATCH_SIZE]
            texts = [c["text"] for c in batch]
            print(f"  🔢 embedding {i+1}-{min(i+BATCH_SIZE, len(all_chunks))}/{len(all_chunks)} ...", end=" ")
            vecs = embed(texts)
            ids = [f"sb_{hashlib.md5((c['source'] + str(j)).encode()).hexdigest()}" for j, c in enumerate(batch, start=i)]
            metas = [{"source": c["source"], "title": c["title"]} for c in batch]
            coll.add(ids=ids, embeddings=vecs, documents=texts, metadatas=metas)
            print("✅")

    save_manifest(new_manifest)
    print(f"🎉 完成！{coll.count()} 条已入库")


def search(query: str, top_k: int = 8):
    """搜索 Second Brain"""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        coll = client.get_collection(COLLECTION_NAME)
    except Exception:
        print("❌ 知识库不存在，先运行 build")
        return []

    q_vec = embed([query])[0]
    results = coll.query(query_embeddings=[q_vec], n_results=top_k)

    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        score = 1 - dist  # cosine distance → similarity
        print(f"\n{'='*60}")
        print(f"#{i+1}  [{meta.get('title','?')}]  📁 {meta.get('source','?')}  相关度: {score:.3f}")
        print(f"{'='*60}")
        print(doc[:600])

    return results


def stats():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        coll = client.get_collection(COLLECTION_NAME)
        print(f"📊 {COLLECTION_NAME}: {coll.count()} 条")
    except Exception:
        print("❌ 知识库不存在")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "search"
    if cmd == "build":
        build(full="--full" in sys.argv)
    elif cmd == "search":
        q = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
        if not q:
            print("用法: python secondbrain_rag.py search <查询词>")
        else:
            search(q)
    elif cmd == "stats":
        stats()
    else:
        print("用法: python secondbrain_rag.py build|search|stats")
