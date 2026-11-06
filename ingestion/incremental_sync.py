"""Incremental synchronization for the SecondBrain-backed knowledge base."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

import config
from ingestion.loader import DocumentLoader
from ingestion.splitter import TextSplitter


SUPPORTED_FORMATS = DocumentLoader.SUPPORTED_FORMATS
DEFAULT_SOURCE_ROOT = Path(config.SECONDBRAIN_ROOT)
DEFAULT_INCLUDE_DIRS = config.SECONDBRAIN_INCLUDE_DIRS


class SyncError(RuntimeError):
    """Raised when the source mirror cannot be synchronized safely."""


@dataclass
class MirrorReport:
    source_root: str
    mirror_root: str
    added_files: List[str] = field(default_factory=list)
    changed_files: List[str] = field(default_factory=list)
    removed_files: List[str] = field(default_factory=list)
    unchanged_files: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    managed_targets: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source_root": self.source_root,
            "mirror_root": self.mirror_root,
            "added_files": len(self.added_files),
            "changed_files": len(self.changed_files),
            "removed_files": len(self.removed_files),
            "unchanged_files": len(self.unchanged_files),
            "conflicts": self.conflicts,
            "errors": self.errors,
        }


@dataclass
class VectorReport:
    added_files: int = 0
    changed_files: int = 0
    removed_files: int = 0
    unchanged_files: int = 0
    added_chunks: int = 0
    removed_chunks: int = 0
    errors: List[str] = field(default_factory=list)
    doc_count: int = 0
    vector_count: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "added_files": self.added_files,
            "changed_files": self.changed_files,
            "removed_files": self.removed_files,
            "unchanged_files": self.unchanged_files,
            "added_chunks": self.added_chunks,
            "removed_chunks": self.removed_chunks,
            "errors": self.errors,
            "doc_count": self.doc_count,
            "vector_count": self.vector_count,
        }


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _relative_target(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise SyncError(f"同步目标越界: {relative}") from exc
    return target


class SecondBrainMirror:
    """Mirror selected SecondBrain folders into the RAG data directory."""

    def __init__(
        self,
        source_root: Path = DEFAULT_SOURCE_ROOT,
        mirror_root: Optional[Path] = None,
        include_dirs: Sequence[str] = DEFAULT_INCLUDE_DIRS,
        manifest_path: Optional[Path] = None,
    ) -> None:
        self.source_root = Path(source_root).expanduser().resolve()
        self.mirror_root = (mirror_root or Path(__file__).resolve().parents[1] / "data").resolve()
        self.include_dirs = tuple(include_dirs)
        self.manifest_path = manifest_path or self.mirror_root.parent / ".secondbrain_sync_manifest.json"

    def _scan(self) -> Dict[str, Dict[str, str]]:
        if not self.source_root.is_dir():
            raise SyncError(f"SecondBrain 源目录不存在: {self.source_root}")

        files: Dict[str, Dict[str, str]] = {}
        target_sources: Dict[str, str] = {}
        for include_dir in self.include_dirs:
            base = (self.source_root / include_dir).resolve()
            if not base.is_dir():
                continue
            for path in sorted(base.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in SUPPORTED_FORMATS:
                    continue
                if any(part.startswith(".") for part in path.relative_to(base).parts):
                    continue
                source_rel = path.relative_to(self.source_root).as_posix()
                target_rel = path.relative_to(base).as_posix()
                previous = target_sources.get(target_rel)
                if previous and previous != source_rel:
                    raise SyncError(
                        f"SecondBrain 文件映射冲突: {previous} 与 {source_rel} 都映射到 {target_rel}"
                    )
                target_sources[target_rel] = source_rel
                files[source_rel] = {
                    "target_rel": target_rel,
                    "sha256": file_hash(path),
                }
        return files

    def sync(self) -> MirrorReport:
        current = self._scan()
        raw_manifest = _load_json(self.manifest_path, {})
        old_files = raw_manifest.get("files", {}) if isinstance(raw_manifest, dict) else {}
        if not isinstance(old_files, dict):
            old_files = {}

        report = MirrorReport(
            source_root=str(self.source_root),
            mirror_root=str(self.mirror_root),
        )
        self.mirror_root.mkdir(parents=True, exist_ok=True)

        for source_rel, info in current.items():
            target = _relative_target(self.mirror_root, info["target_rel"])
            old = old_files.get(source_rel, {})
            try:
                if old.get("sha256") == info["sha256"] and target.exists():
                    if file_hash(target) == info["sha256"]:
                        report.unchanged_files.append(source_rel)
                    else:
                        # The mirror is derived state; restore it from the
                        # SecondBrain source before comparing vectors.
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(self.source_root / Path(source_rel), target)
                        report.changed_files.append(source_rel)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(self.source_root / Path(source_rel), target)
                    if old:
                        report.changed_files.append(source_rel)
                    else:
                        report.added_files.append(source_rel)
            except OSError as exc:
                report.errors.append(f"{source_rel}: {exc}")
                # A locked or temporarily unreadable source must not make the
                # previous mirrored copy look like a deletion.
                if target.exists() and old:
                    report.managed_targets.append(str(target))
                continue

            if target.exists() and file_hash(target) == info["sha256"]:
                report.managed_targets.append(str(target))
            else:
                report.errors.append(f"{source_rel}: 镜像文件校验失败")

        for source_rel, old in old_files.items():
            if source_rel in current:
                continue
            target_rel = old.get("target_rel") if isinstance(old, dict) else None
            if not target_rel:
                continue
            target = _relative_target(self.mirror_root, target_rel)
            try:
                if target.exists():
                    if old.get("sha256") and file_hash(target) != old["sha256"]:
                        report.conflicts.append(str(target))
                        report.managed_targets.append(str(target))
                        continue
                    target.unlink()
                report.removed_files.append(source_rel)
            except OSError as exc:
                report.errors.append(f"{source_rel}: {exc}")

        manifest = {
            "version": 1,
            "source_root": str(self.source_root),
            "include_dirs": list(self.include_dirs),
            "files": current,
        }
        _save_json(self.manifest_path, manifest)
        return report


def _store_snapshot(vectorstore: Any) -> Dict[str, List[Dict[str, Any]]]:
    result = vectorstore.get_snapshot()
    ids = result.get("ids", [])
    documents = result.get("documents", []) or []
    metadatas = result.get("metadatas", []) or []
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for index, doc_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) else {}
        content = documents[index] if index < len(documents) else ""
        source = str(metadata.get("source", ""))
        if source:
            grouped[source].append(
                {"id": doc_id, "content": content or "", "metadata": metadata}
            )
    return grouped


def _chunk_signature(items: Iterable[Dict[str, Any]]) -> List[Tuple[str, int, str]]:
    def chunk_index(item: Dict[str, Any]) -> int:
        try:
            return int(item.get("metadata", {}).get("chunk_index", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def sort_key(item: Dict[str, Any]) -> Tuple[str, int]:
        metadata = item.get("metadata", {})
        return (str(metadata.get("page", "")), chunk_index(item))

    return [
        (
            str(item.get("metadata", {}).get("page", "")),
            chunk_index(item),
            str(item.get("content", "")),
        )
        for item in sorted(items, key=sort_key)
    ]


def _chunk_ids(source: str, source_hash: str, chunks: Sequence[Document]) -> List[str]:
    ids = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata
        page = metadata.get("page", "")
        chunk_index = metadata.get("chunk_index", index)
        payload = f"{source}\0{page}\0{chunk_index}\0{source_hash}"
        ids.append(f"rag_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32]}")
    return ids


def _pipeline_signature(splitter: TextSplitter) -> str:
    return "|".join(
        (
            "incremental-sync-v1",
            splitter.__class__.__name__,
            f"chunk_size={getattr(splitter, 'chunk_size', '')}",
            f"chunk_overlap={getattr(splitter, 'chunk_overlap', '')}",
            f"embedding={config.EMBEDDING_MODEL_NAME}",
            f"dimension={config.EMBEDDING_DIMENSION}",
        )
    )


def sync_vectorstore(
    vectorstore: Any,
    managed_paths: Iterable[Path],
    loader: Optional[DocumentLoader] = None,
    splitter: Optional[TextSplitter] = None,
    manifest_path: Optional[Path] = None,
    managed_root: Optional[Path] = None,
) -> VectorReport:
    """Synchronize mirrored files into an existing Chroma collection."""

    loader = loader or DocumentLoader()
    splitter = splitter or TextSplitter()
    manifest_path = manifest_path or Path(__file__).resolve().parents[1] / ".rag_vector_manifest.json"
    raw_manifest = _load_json(manifest_path, {})
    if isinstance(raw_manifest, dict) and isinstance(raw_manifest.get("files"), dict):
        old_manifest = raw_manifest["files"]
        old_pipeline_signature = str(raw_manifest.get("pipeline_signature", ""))
    elif isinstance(raw_manifest, dict):
        # Accept the first-run legacy shape and upgrade it after a successful sync.
        old_manifest = raw_manifest
        old_pipeline_signature = ""
    else:
        old_manifest = {}
        old_pipeline_signature = ""
    pipeline_signature = _pipeline_signature(splitter)
    force_reparse = bool(old_pipeline_signature and old_pipeline_signature != pipeline_signature)
    managed_root = Path(managed_root).resolve() if managed_root else None

    existing = _store_snapshot(vectorstore)
    current: Dict[str, Tuple[Path, str]] = {}
    for path in sorted({Path(item).resolve() for item in managed_paths}):
        if path.is_file() and path.suffix.lower() in SUPPORTED_FORMATS:
            current[str(path)] = (path, file_hash(path))

    report = VectorReport()
    pending: Dict[str, Tuple[str, List[Document], List[Dict[str, Any]]]] = {}
    new_manifest: Dict[str, str] = {}

    for source, (path, source_hash) in current.items():
        old_hash = old_manifest.get(source)
        old_chunks = existing.get(source, [])
        if not force_reparse and old_hash == source_hash and old_chunks:
            report.unchanged_files += 1
            new_manifest[source] = source_hash
            continue

        try:
            raw_docs = loader.load_file(str(path))
            chunks = splitter.split_documents(raw_docs)
        except Exception as exc:
            report.errors.append(f"{source}: {exc}")
            if old_hash is not None:
                new_manifest[source] = old_hash
            continue

        if not force_reparse and old_chunks and _chunk_signature(old_chunks) == _chunk_signature(
            [{"content": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks]
        ):
            report.unchanged_files += 1
            new_manifest[source] = source_hash
            continue

        if old_chunks:
            report.changed_files += 1
        else:
            report.added_files += 1
        file_key = source
        if managed_root:
            try:
                file_key = Path(source).relative_to(managed_root).as_posix()
            except ValueError:
                pass
        for chunk in chunks:
            chunk.metadata = {
                **chunk.metadata,
                "source_kind": "filesystem",
                "file_key": file_key,
                "source_hash": source_hash,
                "pipeline_signature": pipeline_signature,
            }
        pending[source] = (source_hash, chunks, old_chunks)
        new_manifest[source] = source_hash

    for source, old_hash in old_manifest.items():
        if source in current or source in new_manifest:
            continue
        old_chunks = existing.get(source, [])
        if not old_chunks:
            continue
        try:
            vectorstore.delete_ids([item["id"] for item in old_chunks])
            report.removed_files += 1
            report.removed_chunks += len(old_chunks)
        except Exception as exc:
            report.errors.append(f"删除 {source}: {exc}")
            new_manifest[source] = old_hash

    write_failed = False
    for source, (_source_hash, chunks, old_chunks) in pending.items():
        old_ids = [item["id"] for item in old_chunks]
        desired_ids = _chunk_ids(source, _source_hash, chunks)
        # Remove deterministic IDs left by an interrupted prior run before adding.
        stale_desired = [item_id for item_id in old_ids if item_id in desired_ids]
        if stale_desired:
            vectorstore.delete_ids(stale_desired)
        try:
            if chunks:
                vectorstore.add_documents_with_ids(chunks, desired_ids)
                report.added_chunks += len(chunks)
            remaining_old = [item_id for item_id in old_ids if item_id not in desired_ids]
            if remaining_old:
                vectorstore.delete_ids(remaining_old)
                report.removed_chunks += len(remaining_old)
        except Exception as exc:
            report.errors.append(f"写入 {source}: {exc}")
            write_failed = True
            # 写入失败时不能记下新 hash：否则下次同步会因 hash 相同且旧 chunk
            # 仍在而判为“未变化”，旧内容就永久留在索引里。回退到旧 hash
            # （新增文件则不记录），保证下次同步重试。
            previous_hash = old_manifest.get(source)
            if previous_hash is None:
                new_manifest.pop(source, None)
            else:
                new_manifest[source] = previous_hash

    final = _store_snapshot(vectorstore)
    report.doc_count = sum(len(items) for items in final.values())
    report.vector_count = vectorstore.count()
    # 管线升级（force_reparse）途中有文件写入失败时保留旧签名，
    # 否则失败文件会以旧管线的 chunk 被下次同步误判为已完成。
    saved_signature = (
        old_pipeline_signature if force_reparse and write_failed else pipeline_signature
    )
    _save_json(
        manifest_path,
        {
            "version": 1,
            "pipeline_signature": saved_signature,
            "files": new_manifest,
        },
    )
    return report


def sync_secondbrain(
    vectorstore: Any,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    mirror_root: Optional[Path] = None,
    include_dirs: Sequence[str] = DEFAULT_INCLUDE_DIRS,
    loader: Optional[DocumentLoader] = None,
    splitter: Optional[TextSplitter] = None,
    mirror_manifest_path: Optional[Path] = None,
    vector_manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Mirror SecondBrain and incrementally update the existing vector collection."""

    mirror = SecondBrainMirror(
        source_root=source_root,
        mirror_root=mirror_root,
        include_dirs=include_dirs,
        manifest_path=mirror_manifest_path,
    )
    mirror_report = mirror.sync()
    vector_report = sync_vectorstore(
        vectorstore=vectorstore,
        managed_paths=[Path(path) for path in mirror_report.managed_targets],
        loader=loader,
        splitter=splitter,
        manifest_path=vector_manifest_path,
        managed_root=mirror.mirror_root,
    )
    return {
        "source_root": str(mirror.source_root),
        "include_dirs": list(mirror.include_dirs),
        "mirror": mirror_report.as_dict(),
        "vector": vector_report.as_dict(),
        "errors": mirror_report.errors + mirror_report.conflicts + vector_report.errors,
    }
