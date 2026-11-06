from __future__ import annotations

import hashlib
import math
import os
import re
import sqlite3
import unicodedata
from array import array
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Protocol, Sequence
from urllib.parse import quote

try:  # numpy is optional: it speeds up vector search but the hub runs without it
    import numpy as _np
except ImportError:  # pragma: no cover - exercised only without numpy installed
    _np = None


MEMORY_SCOPES = ("vault", "profile", "episodes")
RETRIEVAL_MODES = ("hybrid", "local")
INDEX_PIPELINE_VERSION = "memory-hub-v3-metadata-headings"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
ASCII_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._/-]*")
CJK_SPAN_RE = re.compile(r"[\u3400-\u9fff]+")
CJK_QUERY_STOP_TOKENS = frozenset(
    {
        "什么",
        "什么样",
        "怎么",
        "怎样",
        "如何",
        "是否",
        "喜欢",
        "我喜欢",
        "我想",
        "想要",
        "请问",
        "告诉我",
        "帮我",
        "可以",
        "有没有",
        "关于",
        "当前",
        "现在",
        "进展",
        "当前进展",
        "下一",
        "一步",
        "下一步",
        "状态",
        "继续",
        "上次",
        "之前",
        "以前",
        "记得",
        "记忆",
        "回忆",
    }
)


def _default_index_root() -> Path:
    local = Path(os.getenv("LOCALAPPDATA", str(Path.home())))
    return local / "FeiliMemoryHub" / "indexes"


@dataclass(frozen=True, slots=True)
class MemoryHubSettings:
    index_root: Path = _default_index_root()
    vault_root: Path = Path(r"C:\Desktop\SecondBrain")
    memory_root: Path = Path.home() / ".codex" / "memory"
    rollout_root: Path = Path.home() / ".codex" / "memories" / "rollout_summaries"
    vault_include_dirs: tuple[str, ...] = ("01-Projects", "02-Areas", "03-Resources", "Daily")
    chunk_size: int = 900
    chunk_overlap: int = 120

    def __post_init__(self) -> None:
        object.__setattr__(self, "index_root", Path(self.index_root).expanduser().resolve())
        object.__setattr__(self, "vault_root", Path(self.vault_root).expanduser().resolve())
        object.__setattr__(self, "memory_root", Path(self.memory_root).expanduser().resolve())
        object.__setattr__(self, "rollout_root", Path(self.rollout_root).expanduser().resolve())
        if self.chunk_size < 100:
            raise ValueError("chunk_size must be at least 100 characters")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")


@dataclass(frozen=True, slots=True)
class MemoryHit:
    scope: str
    chunk_id: str
    source_path: str
    title: str
    heading: str
    preview: str
    score: float
    locator: str
    obsidian_uri: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class EmbeddingProvider(Protocol):
    model_name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class RerankProvider(Protocol):
    model_name: str

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]: ...


@dataclass(frozen=True, slots=True)
class _SourceDocument:
    scope: str
    path: Path
    source_key: str
    title: str
    obsidian_uri: str | None


class MemoryHub:
    """Deep module for synchronizing and retrieving isolated personal memories."""

    def __init__(
        self,
        settings: MemoryHubSettings | None = None,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        rerank_provider: RerankProvider | None = None,
    ) -> None:
        self.settings = settings or MemoryHubSettings()
        self.embedding_provider = embedding_provider
        self.rerank_provider = rerank_provider
        self._last_provider_error: str | None = None
        # scope -> (version key, _VectorIndex); see _vector_index
        self._vector_cache: dict[str, tuple[tuple, _VectorIndex]] = {}
        from .review import MemoryReviewQueue

        self._review_queue = MemoryReviewQueue(self.settings.index_root.parent / "audit.sqlite3")

    def propose_memory(
        self,
        *,
        scope: str,
        content: str,
        reason: str = "",
        source: str = "agent",
        session_id: str | None = None,
    ) -> dict[str, object]:
        return self._review_queue.propose(
            scope=scope,
            content=content,
            reason=reason,
            source=source,
            session_id=session_id,
        )

    def review_memories(self, *, status: str = "pending") -> list[dict[str, object]]:
        return self._review_queue.list(status=status)

    def review_memory(
        self,
        candidate_id: str,
        action: str,
        *,
        scope: str | None = None,
        content: str | None = None,
        note: str = "",
    ) -> dict[str, object]:
        if action == "update":
            return self._review_queue.update_pending(
                candidate_id, scope=scope, content=content, note=note
            )
        if action == "reject":
            if scope is not None or content is not None:
                raise ValueError("reject does not accept scope or content changes")
            return self._review_queue.transition(
                candidate_id,
                expected_status="pending",
                status="rejected",
                note=note,
            )
        if action == "approve":
            candidate = self._review_queue.update_pending(
                candidate_id, scope=scope, content=content, note=note
            )
            target = self._review_target(str(candidate["scope"]))
            added = self._append_managed_memory(
                target,
                candidate_id=candidate_id,
                content=str(candidate["content"]),
            )
            try:
                self.sync((str(candidate["scope"]),))
            except Exception:
                if added:
                    self._remove_managed_memory(target, candidate_id=candidate_id)
                    try:
                        self.sync((str(candidate["scope"]),))
                    except Exception:
                        pass
                raise
            try:
                return self._review_queue.transition(
                    candidate_id,
                    expected_status="pending",
                    status="approved",
                    target_path=str(target),
                    note=note,
                )
            except Exception:
                current = self._review_queue.get(candidate_id)
                if current["status"] == "approved":
                    return current
                if added:
                    self._remove_managed_memory(target, candidate_id=candidate_id)
                    try:
                        self.sync((str(candidate["scope"]),))
                    except Exception:
                        pass
                raise
        if action == "forget":
            if scope is not None or content is not None:
                raise ValueError("forget does not accept scope or content changes")
            candidate = self._review_queue.get(candidate_id)
            if candidate["status"] != "approved":
                raise ValueError("only approved memory candidates can be forgotten")
            target_text = candidate.get("target_path")
            if not target_text:
                raise RuntimeError("approved memory candidate has no managed target")
            target = Path(str(target_text)).resolve()
            expected_target = self._review_target(str(candidate["scope"]))
            if target != expected_target:
                raise RuntimeError("managed memory target does not match its configured scope")
            removed = self._remove_managed_memory(target, candidate_id=candidate_id)
            try:
                self.sync((str(candidate["scope"]),))
            except Exception:
                if removed:
                    self._append_managed_memory(
                        target,
                        candidate_id=candidate_id,
                        content=str(candidate["content"]),
                    )
                    try:
                        self.sync((str(candidate["scope"]),))
                    except Exception:
                        pass
                raise
            try:
                return self._review_queue.transition(
                    candidate_id,
                    expected_status="approved",
                    status="forgotten",
                    note=note,
                )
            except Exception:
                current = self._review_queue.get(candidate_id)
                if current["status"] == "forgotten":
                    return current
                if removed:
                    self._append_managed_memory(
                        target,
                        candidate_id=candidate_id,
                        content=str(candidate["content"]),
                    )
                    try:
                        self.sync((str(candidate["scope"]),))
                    except Exception:
                        pass
                raise
        raise ValueError(f"unknown review action: {action}")

    def memory_review_history(self, candidate_id: str) -> list[dict[str, object]]:
        return self._review_queue.history(candidate_id)

    def sync(self, scopes: Sequence[str] | None = None) -> dict[str, object]:
        selected = self._normalize_scopes(scopes)
        self.settings.index_root.mkdir(parents=True, exist_ok=True)
        reports = {}
        for scope in selected:
            reports[scope] = self._sync_scope(scope)
            if self.embedding_provider is not None:
                try:
                    reports[scope]["vectors_added"] = self._sync_vectors(scope)
                except Exception as exc:
                    self._last_provider_error = str(exc)
                    reports[scope]["vector_error"] = str(exc)
        return {"scopes": reports, "status": self.status()}

    def search(
        self,
        query: str,
        scopes: Sequence[str] | None = None,
        top_k: int = 8,
        retrieval_mode: str = "hybrid",
    ) -> list[MemoryHit]:
        if not query.strip():
            return []
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")

        selected = self._normalize_scopes(scopes)
        mode = self._normalize_retrieval_mode(retrieval_mode)
        per_scope = max(top_k * 3, top_k)
        ranked_lists: list[list[MemoryHit]] = []
        for scope in selected:
            lexical = self._search_scope(scope, query, per_scope)
            if lexical:
                ranked_lists.append(lexical)

        if mode == "hybrid" and self.embedding_provider is not None:
            try:
                query_vector = self.embedding_provider.embed([query])[0]
                for scope in selected:
                    semantic = self._search_vector_scope(scope, query_vector, per_scope)
                    if semantic:
                        ranked_lists.append(semantic)
            except Exception as exc:
                self._last_provider_error = str(exc)

        fused: dict[str, tuple[MemoryHit, float]] = {}
        for ranked in ranked_lists:
            for rank, hit in enumerate(ranked, start=1):
                previous = fused.get(hit.chunk_id)
                score = (previous[1] if previous else 0.0) + 1.0 / (60 + rank)
                fused[hit.chunk_id] = (previous[0] if previous else hit, score)

        candidates = [replace(hit, score=score) for hit, score in fused.values()]

        candidates.sort(key=lambda item: (-item.score, item.scope, item.source_path, item.chunk_id))
        if mode == "hybrid" and self.rerank_provider is not None and candidates:
            try:
                reranked = self.rerank_provider.rerank(
                    query,
                    [hit.preview for hit in candidates],
                    min(top_k, len(candidates)),
                )
                selected_hits: list[MemoryHit] = []
                seen_indexes: set[int] = set()
                for index, score in reranked:
                    if index in seen_indexes or not 0 <= index < len(candidates):
                        continue
                    seen_indexes.add(index)
                    selected_hits.append(replace(candidates[index], score=float(score)))
                if selected_hits:
                    return selected_hits[:top_k]
            except Exception as exc:
                self._last_provider_error = str(exc)
        return candidates[:top_k]

    def context(
        self,
        query: str,
        scopes: Sequence[str] | None = None,
        top_k: int = 8,
        char_budget: int = 6000,
        retrieval_mode: str = "hybrid",
    ) -> dict[str, object]:
        mode = self._normalize_retrieval_mode(retrieval_mode)
        hits = self.search(query, scopes=scopes, top_k=top_k, retrieval_mode=mode)
        remaining = max(0, char_budget)
        parts: list[str] = []
        included: list[MemoryHit] = []
        for hit in hits:
            block = f"[{hit.scope}] {hit.locator}\n{hit.preview.strip()}"
            if len(block) > remaining:
                if not parts and remaining:
                    parts.append(block[:remaining])
                    included.append(hit)
                break
            parts.append(block)
            included.append(hit)
            remaining -= len(block) + 2
        return {
            "query": query,
            "context": "\n\n".join(parts),
            "hits": [hit.as_dict() for hit in included],
            "char_budget": char_budget,
            "retrieval_mode": mode,
        }

    def status(self) -> dict[str, object]:
        scopes: dict[str, object] = {}
        for scope in MEMORY_SCOPES:
            path = self._db_path(scope)
            count = 0
            last_sync = None
            if path.is_file():
                with self._connect(scope) as connection:
                    count = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
                    row = connection.execute(
                        "SELECT value FROM metadata WHERE key = 'last_sync'"
                    ).fetchone()
                    last_sync = row[0] if row else None
            scopes[scope] = {
                "index_path": str(path),
                "chunk_count": count,
                "last_sync": last_sync,
            }
        return {
            "ready": all(self._db_path(scope).is_file() for scope in MEMORY_SCOPES),
            "index_root": str(self.settings.index_root),
            "scopes": scopes,
            "embedding": {
                "configured": self.embedding_provider is not None,
                "model": getattr(self.embedding_provider, "model_name", None),
            },
            "reranker": {
                "configured": self.rerank_provider is not None,
                "model": getattr(self.rerank_provider, "model_name", None),
            },
            "provider_error": self._last_provider_error,
            "review": self._review_queue.counts(),
        }

    def _sync_scope(self, scope: str) -> dict[str, int]:
        documents = {item.source_key: item for item in self._iter_sources(scope)}
        added = changed = removed = unchanged = 0

        with self._connect(scope) as connection:
            pipeline_row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'pipeline_version'"
            ).fetchone()
            if pipeline_row is None or pipeline_row[0] != INDEX_PIPELINE_VERSION:
                connection.executescript(
                    """
                    DELETE FROM chunks_fts;
                    DELETE FROM vectors;
                    DELETE FROM chunks;
                    DELETE FROM sources;
                    """
                )
            existing = {
                row[0]: row[1]
                for row in connection.execute("SELECT source_key, source_hash FROM sources")
            }

            for source_key in sorted(existing.keys() - documents.keys()):
                self._delete_source(connection, source_key)
                removed += 1

            for source_key, document in documents.items():
                source_hash = _file_hash(document.path)
                if existing.get(source_key) == source_hash:
                    unchanged += 1
                    continue
                if source_key in existing:
                    self._delete_source(connection, source_key)
                    changed += 1
                else:
                    added += 1
                self._index_source(connection, document, source_hash)

            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('last_sync', ?)",
                (datetime.now(timezone.utc).isoformat(),),
            )
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('pipeline_version', ?)",
                (INDEX_PIPELINE_VERSION,),
            )
            connection.commit()

        return {
            "added": added,
            "changed": changed,
            "removed": removed,
            "unchanged": unchanged,
            "chunks": self._scope_count(scope),
        }

    def _iter_sources(self, scope: str) -> Iterator[_SourceDocument]:
        if scope == "vault":
            if not self.settings.vault_root.is_dir():
                return
            managed_inbox = self._review_target("vault")
            yielded_paths: set[Path] = set()
            if managed_inbox.is_file():
                yielded_paths.add(managed_inbox.resolve())
                note_key = managed_inbox.relative_to(self.settings.vault_root).with_suffix("").as_posix()
                vault_name = quote(self.settings.vault_root.name, safe="")
                yield _SourceDocument(
                    scope,
                    managed_inbox,
                    managed_inbox.relative_to(self.settings.vault_root).as_posix(),
                    managed_inbox.stem,
                    f"obsidian://open?vault={vault_name}&file=" + quote(note_key, safe="/"),
                )
            for include_dir in self.settings.vault_include_dirs:
                base = self.settings.vault_root / include_dir
                if not base.is_dir():
                    continue
                for path in sorted(base.rglob("*.md")):
                    if any(part.startswith(".") for part in path.relative_to(base).parts):
                        continue
                    if path.resolve() in yielded_paths:
                        continue
                    relative = path.relative_to(self.settings.vault_root).as_posix()
                    note_key = path.relative_to(self.settings.vault_root).with_suffix("").as_posix()
                    vault_name = quote(self.settings.vault_root.name, safe="")
                    obsidian_uri = (
                        f"obsidian://open?vault={vault_name}&file=" + quote(note_key, safe="/")
                    )
                    yield _SourceDocument(scope, path, relative, path.stem, obsidian_uri)
            return

        if scope == "profile":
            for name in ("USER_PROFILE.md", "FEILI_SOUL.md"):
                path = self.settings.memory_root / name
                if path.is_file():
                    yield _SourceDocument(scope, path, name, path.stem, None)
            return

        for name in ("CURRENT_CONTEXT.md", "MEMORY_LOG.md"):
            path = self.settings.memory_root / name
            if path.is_file():
                yield _SourceDocument(scope, path, name, path.stem, None)
        if self.settings.rollout_root.is_dir():
            for path in sorted(self.settings.rollout_root.glob("*.md")):
                source_key = f"rollout_summaries/{path.name}"
                yield _SourceDocument(scope, path, source_key, path.stem, None)

    def _review_target(self, scope: str) -> Path:
        if scope == "profile":
            target = self.settings.memory_root / "USER_PROFILE.md"
            root = self.settings.memory_root
        elif scope == "episodes":
            target = self.settings.memory_root / "CURRENT_CONTEXT.md"
            root = self.settings.memory_root
        elif scope == "vault":
            target = self.settings.vault_root / "00-Inbox" / "Agent Memory Inbox.md"
            root = self.settings.vault_root
        else:
            raise ValueError(f"unknown memory scope: {scope}")
        resolved = target.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError("managed memory target escapes its configured root")
        return resolved

    @staticmethod
    def _append_managed_memory(target: Path, *, candidate_id: str, content: str) -> bool:
        start = f"<!-- feili-memory:{candidate_id}:start -->"
        end = f"<!-- feili-memory:{candidate_id}:end -->"
        target.parent.mkdir(parents=True, exist_ok=True)
        existing = target.read_text(encoding="utf-8") if target.is_file() else ""
        if start in existing and end in existing:
            managed = existing.split(start, 1)[1].split(end, 1)[0]
            if content.strip() in managed:
                return False
            raise RuntimeError("managed memory block already exists with different content")
        if start in existing or end in existing:
            raise RuntimeError("managed memory block markers are incomplete")
        if not existing:
            existing = "# Agent Memory Inbox\n" if target.name == "Agent Memory Inbox.md" else ""
        block = f"{start}\n\n## 审核记忆 {candidate_id[:8]}\n\n{content.strip()}\n\n{end}"
        target.write_text(existing.rstrip() + "\n\n" + block + "\n", encoding="utf-8")
        return True

    @staticmethod
    def _remove_managed_memory(target: Path, *, candidate_id: str) -> bool:
        if not target.is_file():
            return False
        start = re.escape(f"<!-- feili-memory:{candidate_id}:start -->")
        end = re.escape(f"<!-- feili-memory:{candidate_id}:end -->")
        text = target.read_text(encoding="utf-8")
        start_text = f"<!-- feili-memory:{candidate_id}:start -->"
        end_text = f"<!-- feili-memory:{candidate_id}:end -->"
        if start_text not in text and end_text not in text:
            return False
        if start_text not in text or end_text not in text:
            raise RuntimeError("managed memory block markers are incomplete")
        updated, count = re.subn(
            rf"\n*{start}.*?{end}\n*",
            "\n\n",
            text,
            count=1,
            flags=re.DOTALL,
        )
        if count != 1:
            raise RuntimeError("managed memory block could not be found")
        target.write_text(updated.rstrip() + "\n", encoding="utf-8")
        return True

    def _index_source(
        self,
        connection: sqlite3.Connection,
        document: _SourceDocument,
        source_hash: str,
    ) -> None:
        text = document.path.read_text(encoding="utf-8-sig", errors="replace")
        connection.execute(
            "INSERT INTO sources(source_key, source_path, source_hash, title, obsidian_uri) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                document.source_key,
                str(document.path),
                source_hash,
                document.title,
                document.obsidian_uri,
            ),
        )
        for ordinal, (heading, content) in enumerate(
            _chunk_markdown(text, self.settings.chunk_size, self.settings.chunk_overlap)
        ):
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            payload = f"{document.scope}\0{document.source_key}\0{heading}\0{ordinal}\0{content_hash}"
            chunk_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
            lexical = " ".join(_lexical_tokens(f"{document.title} {heading} {content}"))
            cursor = connection.execute(
                "INSERT INTO chunks(chunk_id, source_key, heading, ordinal, content, lexical) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    chunk_id,
                    document.source_key,
                    heading,
                    ordinal,
                    content,
                    lexical,
                ),
            )
            connection.execute(
                "INSERT INTO chunks_fts(rowid, content, lexical) VALUES (?, ?, ?)",
                (cursor.lastrowid, content, lexical),
            )

    def _delete_source(self, connection: sqlite3.Connection, source_key: str) -> None:
        row_ids = [
            row[0]
            for row in connection.execute(
                "SELECT rowid FROM chunks WHERE source_key = ?", (source_key,)
            )
        ]
        for row_id in row_ids:
            connection.execute("DELETE FROM chunks_fts WHERE rowid = ?", (row_id,))
        connection.execute("DELETE FROM chunks WHERE source_key = ?", (source_key,))
        connection.execute("DELETE FROM sources WHERE source_key = ?", (source_key,))

    def _sync_vectors(self, scope: str) -> int:
        """Attach vectors to chunks that lack one, embedding only unseen text.

        Editing a note deletes and re-creates all of its chunks (and, by
        cascade, their vectors); chunk ids also shift with ordinals. Vectors are
        therefore cached by (model, content hash), so only genuinely new text is
        sent to the embedding provider. Cache rows whose text no longer exists
        in any chunk are pruned, so deleted notes do not linger.
        """
        assert self.embedding_provider is not None
        model_name = self.embedding_provider.model_name
        with self._connect(scope) as connection:
            rows = connection.execute(
                """
                SELECT c.chunk_id, c.content
                FROM chunks c
                LEFT JOIN vectors v ON v.chunk_id = c.chunk_id AND v.model = ?
                WHERE v.chunk_id IS NULL
                ORDER BY c.rowid
                """,
                (model_name,),
            ).fetchall()
            added = 0
            missing: dict[str, str] = {}
            pending: list[tuple[str, str]] = []
            for chunk_id, content in rows:
                content_hash = _content_hash(content)
                cached = connection.execute(
                    "SELECT dimension, vector FROM embedding_cache "
                    "WHERE model = ? AND content_hash = ?",
                    (model_name, content_hash),
                ).fetchone()
                if cached is not None:
                    connection.execute(
                        "INSERT OR REPLACE INTO vectors(chunk_id, model, dimension, vector) "
                        "VALUES (?, ?, ?, ?)",
                        (chunk_id, model_name, cached[0], cached[1]),
                    )
                    added += 1
                    continue
                missing.setdefault(content_hash, content)
                pending.append((chunk_id, content_hash))

            embedded: dict[str, bytes] = {}
            dimensions: dict[str, int] = {}
            unique = list(missing.items())
            for start in range(0, len(unique), 32):
                batch = unique[start : start + 32]
                vectors = self.embedding_provider.embed([content for _, content in batch])
                if len(vectors) != len(batch):
                    raise RuntimeError("embedding provider returned an unexpected vector count")
                for (content_hash, _), vector in zip(batch, vectors):
                    if not vector:
                        raise RuntimeError("embedding provider returned an empty vector")
                    packed = _pack_vector(vector)
                    embedded[content_hash] = packed
                    dimensions[content_hash] = len(vector)
                    connection.execute(
                        "INSERT OR REPLACE INTO embedding_cache(model, content_hash, dimension, vector) "
                        "VALUES (?, ?, ?, ?)",
                        (model_name, content_hash, len(vector), packed),
                    )
            for chunk_id, content_hash in pending:
                connection.execute(
                    "INSERT OR REPLACE INTO vectors(chunk_id, model, dimension, vector) "
                    "VALUES (?, ?, ?, ?)",
                    (chunk_id, model_name, dimensions[content_hash], embedded[content_hash]),
                )
                added += 1

            live = {_content_hash(row[0]) for row in connection.execute("SELECT content FROM chunks")}
            stale = [
                (row[0], row[1])
                for row in connection.execute("SELECT model, content_hash FROM embedding_cache")
                if row[1] not in live
            ]
            connection.executemany(
                "DELETE FROM embedding_cache WHERE model = ? AND content_hash = ?", stale
            )
            connection.commit()
        return added

    def _search_scope(self, scope: str, query: str, limit: int) -> list[MemoryHit]:
        if not self._db_path(scope).is_file():
            return []
        tokens = _query_tokens(query)
        if not tokens:
            return []
        expression = " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)
        sql = """
            SELECT c.chunk_id, s.source_path, s.title, c.heading, c.content,
                   s.source_key, s.obsidian_uri, bm25(chunks_fts) AS lexical_rank
            FROM chunks_fts
            JOIN chunks c ON c.rowid = chunks_fts.rowid
            JOIN sources s ON s.source_key = c.source_key
            WHERE chunks_fts MATCH ?
            ORDER BY lexical_rank ASC
            LIMIT ?
        """
        with self._connect(scope) as connection:
            rows = connection.execute(sql, (expression, limit)).fetchall()

        hits = []
        for rank, row in enumerate(rows, start=1):
            heading = row[3] or row[2]
            locator = row[5]
            if heading and heading != row[2]:
                locator = f"{locator}#{heading}"
            hits.append(
                MemoryHit(
                    scope=scope,
                    chunk_id=row[0],
                    source_path=row[1],
                    title=row[2],
                    heading=heading,
                    preview=_preview(row[4], query),
                    score=1.0 / rank,
                    locator=locator,
                    obsidian_uri=row[6],
                )
            )
        return hits

    def _search_vector_scope(
        self,
        scope: str,
        query_vector: Sequence[float],
        limit: int,
    ) -> list[MemoryHit]:
        if not self._db_path(scope).is_file() or self.embedding_provider is None:
            return []
        index = self._vector_index(scope, self.embedding_provider.model_name)

        hits = []
        for score, row in index.top(query_vector, limit):
            heading = row[3] or row[2]
            locator = row[5]
            if heading and heading != row[2]:
                locator = f"{locator}#{heading}"
            hits.append(
                MemoryHit(
                    scope=scope,
                    chunk_id=row[0],
                    source_path=row[1],
                    title=row[2],
                    heading=heading,
                    preview=_preview(row[4], ""),
                    score=score,
                    locator=locator,
                    obsidian_uri=row[6],
                )
            )
        return hits

    def _vector_index(self, scope: str, model_name: str) -> "_VectorIndex":
        """Return the in-memory vector matrix for a scope, rebuilding it only when stale.

        Unpacking every BLOB and scoring it in pure Python on each query was
        O(N·d) interpreter work. The version key is a cheap aggregate that
        changes on any vector insert/delete and on every sync, so a sync from
        another process (``python -m memory_hub --sync-only``) is picked up too.
        """
        with self._connect(scope) as connection:
            count, max_rowid = connection.execute(
                "SELECT COUNT(*), COALESCE(MAX(rowid), 0) FROM vectors WHERE model = ?",
                (model_name,),
            ).fetchone()
            last_sync = connection.execute(
                "SELECT value FROM metadata WHERE key = 'last_sync'"
            ).fetchone()
            version = (model_name, int(count), int(max_rowid), last_sync[0] if last_sync else None)
            cached = self._vector_cache.get(scope)
            if cached is not None and cached[0] == version:
                return cached[1]
            rows = connection.execute(
                """
                SELECT c.chunk_id, s.source_path, s.title, c.heading, c.content,
                       s.source_key, s.obsidian_uri, v.dimension, v.vector
                FROM vectors v
                JOIN chunks c ON c.chunk_id = v.chunk_id
                JOIN sources s ON s.source_key = c.source_key
                WHERE v.model = ?
                """,
                (model_name,),
            ).fetchall()
        index = _VectorIndex(rows)
        self._vector_cache[scope] = (version, index)
        return index

    @contextmanager
    def _connect(self, scope: str) -> Iterator[sqlite3.Connection]:
        path = self._db_path(scope)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sources (
                    source_key TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    title TEXT NOT NULL,
                    obsidian_uri TEXT
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source_key TEXT NOT NULL REFERENCES sources(source_key) ON DELETE CASCADE,
                    heading TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    lexical TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS vectors (
                    chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                    model TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    PRIMARY KEY(chunk_id, model)
                );
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    model TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    PRIMARY KEY(model, content_hash)
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    content,
                    lexical,
                    tokenize='unicode61'
                );
                """
            )
            yield connection
        finally:
            connection.close()

    def _scope_count(self, scope: str) -> int:
        with self._connect(scope) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def _db_path(self, scope: str) -> Path:
        return self.settings.index_root / f"{scope}.sqlite3"

    @staticmethod
    def _normalize_scopes(scopes: Sequence[str] | None) -> tuple[str, ...]:
        selected = tuple(scopes or MEMORY_SCOPES)
        unknown = sorted(set(selected) - set(MEMORY_SCOPES))
        if unknown:
            raise ValueError(f"unknown memory scopes: {', '.join(unknown)}")
        return tuple(dict.fromkeys(selected))

    @staticmethod
    def _normalize_retrieval_mode(retrieval_mode: str) -> str:
        if retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(
                f"unknown retrieval mode: {retrieval_mode}; "
                f"choose from {', '.join(RETRIEVAL_MODES)}"
            )
        return retrieval_mode


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _chunk_markdown(text: str, chunk_size: int, overlap: int) -> Iterator[tuple[str, str]]:
    heading = ""
    buffer: list[str] = []
    sections: list[tuple[str, str]] = []
    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            if any(part.strip() for part in buffer):
                sections.append((heading, "\n".join(buffer).strip()))
            heading = match.group(2).strip()
            buffer = []
            continue
        buffer.append(line)
    if any(part.strip() for part in buffer):
        sections.append((heading, "\n".join(buffer).strip()))

    step = max(1, chunk_size - overlap)
    for section_heading, content in sections:
        if not content:
            continue
        start = 0
        while start < len(content):
            end = min(len(content), start + chunk_size)
            yield section_heading, content[start:end].strip()
            if end == len(content):
                break
            start += step


def _lexical_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens = ASCII_TOKEN_RE.findall(normalized)
    for span in CJK_SPAN_RE.findall(normalized):
        tokens.append(span)
        tokens.extend(span[index : index + 2] for index in range(max(0, len(span) - 1)))
        tokens.extend(span[index : index + 3] for index in range(max(0, len(span) - 2)))
    return list(dict.fromkeys(token for token in tokens if token))


def _query_tokens(text: str) -> list[str]:
    tokens = _lexical_tokens(text)
    expanded_stop_tokens = {
        token
        for phrase in CJK_QUERY_STOP_TOKENS
        for token in _lexical_tokens(phrase)
    }
    filtered = [token for token in tokens if token not in expanded_stop_tokens]
    return filtered or tokens


def _preview(content: str, query: str, length: int = 360) -> str:
    compact = re.sub(r"\s+", " ", content).strip()
    if len(compact) <= length:
        return compact
    terms = [token for token in _query_tokens(query) if len(token) >= 2]
    positions = [compact.casefold().find(term) for term in terms]
    positions = [position for position in positions if position >= 0]
    center = min(positions) if positions else 0
    start = max(0, center - length // 3)
    end = min(len(compact), start + length)
    prefix = "…" if start else ""
    suffix = "…" if end < len(compact) else ""
    return prefix + compact[start:end] + suffix


def _pack_vector(values: Sequence[float]) -> bytes:
    return array("f", (float(value) for value in values)).tobytes()


def _unpack_vector(payload: bytes, dimension: int) -> list[float]:
    values = array("f")
    values.frombytes(payload)
    if len(values) != dimension:
        raise RuntimeError("stored vector dimension does not match its metadata")
    return list(values)


class _VectorIndex:
    """Immutable snapshot of a scope's vectors for fast cosine top-k.

    Rows are ``(chunk_id, source_path, title, heading, content, source_key,
    obsidian_uri, dimension, vector_blob)``. With numpy each dimension group is
    one L2-normalised matrix, so a query is a single mat-vec product; without
    numpy it falls back to the original pure-Python scan.
    """

    def __init__(self, rows: Sequence[Sequence[object]]) -> None:
        self._rows = [tuple(row[:7]) for row in rows]
        self._vectors: list[list[float]] = []
        self._groups: dict[int, tuple[list[int], object]] = {}
        by_dimension: dict[int, list[int]] = {}
        for position, row in enumerate(rows):
            dimension = int(row[7])
            by_dimension.setdefault(dimension, []).append(position)
            if _np is None:
                self._vectors.append(_unpack_vector(row[8], dimension))
        if _np is not None:
            for dimension, positions in by_dimension.items():
                matrix = _np.vstack(
                    [_np.frombuffer(rows[position][8], dtype=_np.float32) for position in positions]
                ).astype(_np.float64)
                if matrix.shape[1] != dimension:
                    raise RuntimeError("stored vector dimension does not match its metadata")
                norms = _np.linalg.norm(matrix, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self._groups[dimension] = (positions, matrix / norms)

    def __len__(self) -> int:
        return len(self._rows)

    def top(self, query_vector: Sequence[float], limit: int) -> list[tuple[float, tuple]]:
        """Return up to ``limit`` ``(score, row)`` pairs with positive cosine score."""
        if not self._rows or limit <= 0:
            return []
        if _np is None:
            scored = []
            for row, vector in zip(self._rows, self._vectors):
                score = _cosine_similarity(query_vector, vector)
                if score > 0:
                    scored.append((score, row))
            scored.sort(key=lambda item: item[0], reverse=True)
            return scored[:limit]

        group = self._groups.get(len(query_vector))
        if group is None:
            return []
        positions, matrix = group
        query = _np.asarray(query_vector, dtype=_np.float64)
        norm = float(_np.linalg.norm(query))
        if norm == 0:
            return []
        scores = matrix @ (query / norm)
        count = min(limit, len(scores))
        candidates = _np.argpartition(-scores, count - 1)[:count]
        ordered = candidates[_np.argsort(-scores[candidates], kind="stable")]
        return [
            (float(scores[index]), self._rows[positions[index]])
            for index in ordered
            if scores[index] > 0
        ]


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
