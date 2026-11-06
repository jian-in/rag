"""Orphan audit for the local RAG knowledge base.

The incremental sync is *manifest-driven*: it only ever touches files that are
registered in `.secondbrain_sync_manifest.json` (the mirror ledger) and
`.rag_vector_manifest.json` (the vector ledger). Anything that entered Chroma
outside that pipeline is invisible to sync and will never be reclaimed.

Two failure classes this tool catches:

  [1] 死源 (dead source)
      A `source` in Chroma whose mirror file no longer exists under `data/`.
      The vectors are stale and keep polluting retrieval.
      Fix: `sync_knowledge_base` now reclaims dead sources under `data/`
      automatically (see `_remove_dead_mirror_sources`); `--purge` remains for
      dead sources outside the mirror or when the MCP server is not running.

  [2] 账外 (off-ledger)
      A `source` in Chroma that is NOT registered in the vector ledger.
      Sync can never add, update, or remove these vectors. If the file is later
      deleted, its vectors become invisible orphans forever -- this is exactly
      how the `408-考研（2）.md` ghost (55 chunks) survived a full sync.

Usage:
    python audit_orphans.py            # report only (read-only)
    python audit_orphans.py --purge    # also reclaim class-[1] vectors

Note: `--purge` writes to Chroma. Stop the MCP server first (stop_all.bat) so
the two processes do not fight over the SQLite store; the tool warns if the
server still looks alive.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

VECTOR_MANIFEST = ROOT / ".rag_vector_manifest.json"
MIRROR_ROOT = ROOT / "data"
MCP_HOST, MCP_PORT = "127.0.0.1", 8766


def _chroma_dir() -> Path:
    persist = Path(config.CHROMA_PERSIST_DIR)
    return persist if persist.is_absolute() else (ROOT / persist)


def _norm(path: str) -> str:
    """Normalize for set comparison: absolute, resolved, case-folded (Windows)."""
    try:
        return str(Path(path).resolve()).lower()
    except OSError:
        return str(path).lower()


def load_sources() -> Counter:
    """Return {source: chunk_count} across the whole collection."""
    import chromadb

    client = chromadb.PersistentClient(path=str(_chroma_dir()))
    coll = client.get_collection(config.CHROMA_COLLECTION_NAME)
    metas = coll.get(include=["metadatas"])["metadatas"]
    return Counter(((m or {}).get("source") or "<无 source>") for m in metas)


def load_ledger() -> set[str]:
    if not VECTOR_MANIFEST.exists():
        return set()
    data = json.loads(VECTOR_MANIFEST.read_text(encoding="utf-8"))
    return set((data.get("files") or {}).keys())


def classify(sources: Counter, ledger_keys: set[str]) -> tuple[list, list]:
    """Split sources into (dead, off_ledger). A source may be in both."""
    norm_ledger = {_norm(k) for k in ledger_keys}
    dead: list[tuple[str, int]] = []
    off: list[tuple[str, int, str]] = []
    for src, n in sorted(sources.items()):
        if src.startswith("<"):
            off.append((src, n, "元数据缺 source"))
            continue
        path = Path(src)
        if not path.is_absolute():
            path = MIRROR_ROOT / src
        if not path.exists():
            dead.append((src, n))
        if _norm(src) not in norm_ledger:
            off.append((src, n, "不在向量清单"))
    return dead, off


def purge(sources: Counter, dead: list[tuple[str, int]]) -> int:
    """Delete every chunk whose source is `src`. Returns chunks removed."""
    if not dead:
        return 0
    import chromadb

    client = chromadb.PersistentClient(path=str(_chroma_dir()))
    coll = client.get_collection(config.CHROMA_COLLECTION_NAME)
    removed = 0
    for src, _ in dead:
        matched = coll.get(where={"source": src}).get("ids", [])
        if matched:
            coll.delete(where={"source": src})
        removed += len(matched)
        print(f"  [-]{src}  → 回收 {len(matched)} 块")
    return removed


def _server_up() -> bool:
    try:
        with socket.create_connection((MCP_HOST, MCP_PORT), timeout=1):
            return True
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="审计 RAG 向量库的孤儿数据")
    parser.add_argument("--purge", action="store_true",
                        help="回收 [1] 类死源向量（需先停 MCP 服务）")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    sources = load_sources()
    ledger_keys = load_ledger()
    dead, off = classify(sources, ledger_keys)

    print("=" * 64)
    print("孤儿审计 · rag-knowledge-base")
    print("=" * 64)
    print(f"向量库     : {_chroma_dir()}\\{config.CHROMA_COLLECTION_NAME}")
    print(f"镜像根     : {MIRROR_ROOT}")
    print(f"向量清单   : {VECTOR_MANIFEST.name}（{len(ledger_keys)} 条）")
    print()
    print(f"chroma 总块数 : {sum(sources.values())}")
    print(f"不同 source   : {len(sources)}")
    print()

    print(f"[1] 死源（镜像里已不存在，向量应被回收）")
    if dead:
        for src, n in dead:
            print(f"  ! {src}   {n} 块")
    else:
        print("  无 —— 库与镜像一致")

    print()
    print("[2] 账外 source（不在向量清单，同步永远管不到）")
    if off:
        for src, n, why in off:
            print(f"  ~ {src}   {n} 块  ({why})")
    else:
        print("  无")

    if not args.purge:
        print()
        print("用 --purge 回收 [1] 类向量。" if dead else "")
        return 0

    print()
    if not dead:
        print("--purge: 没有 [1] 类死源，未做任何改动。")
        return 0
    if _server_up():
        print(f"警告: 端口 {MCP_PORT} 仍在监听，MCP 服务可能持有向量库。")
        print("      建议先 stop_all.bat 再跑 --purge。（继续执行…）")

    before = sum(sources.values())
    print(f"回收前总块数 : {before}")
    removed = purge(sources, dead)
    after = sum(load_sources().values())
    print(f"回收后总块数 : {after}")
    print(f"共回收 {removed} 块。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
