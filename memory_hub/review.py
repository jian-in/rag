from __future__ import annotations

import hashlib
import re
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from .core import MEMORY_SCOPES


REVIEW_STATUSES = ("pending", "approved", "rejected", "forgotten")
SECRET_PATTERNS = (
    re.compile(r"(?i)(?:api[_-]?key|token|password|secret)\s*[:=]\s*\S+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


class MemoryReviewQueue:
    """Local control-plane queue for human-reviewed memory proposals."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def propose(
        self,
        *,
        scope: str,
        content: str,
        reason: str,
        source: str,
        session_id: str | None,
    ) -> dict[str, object]:
        scope = self._validate_scope(scope)
        normalized = self._validate_content(content)
        reason = reason.strip()
        source = source.strip() or "agent"
        session_id = session_id.strip() if session_id else None
        if len(reason) > 1000 or len(source) > 200 or (session_id and len(session_id) > 200):
            raise ValueError("memory proposal metadata is too long")

        content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        now = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as connection:
            existing = connection.execute(
                "SELECT * FROM memory_candidates "
                "WHERE scope = ? AND content_hash = ? AND status = 'pending'",
                (scope, content_hash),
            ).fetchone()
            if existing is not None:
                self._insert_event(
                    connection,
                    candidate_id=existing["id"],
                    event_type="deduplicated",
                    note=source,
                    created_at=now,
                )
                connection.commit()
                return self._row(existing)

            candidate_id = uuid.uuid4().hex
            connection.execute(
                """
                INSERT INTO memory_candidates(
                    id, scope, content, content_hash, reason, source, session_id,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    candidate_id,
                    scope,
                    normalized,
                    content_hash,
                    reason,
                    source,
                    session_id,
                    now,
                    now,
                ),
            )
            self._insert_event(
                connection,
                candidate_id=candidate_id,
                event_type="proposed",
                note=reason,
                created_at=now,
            )
            connection.commit()
            row = connection.execute(
                "SELECT * FROM memory_candidates WHERE id = ?", (candidate_id,)
            ).fetchone()
        assert row is not None
        return self._row(row)

    def list(self, *, status: str = "pending") -> list[dict[str, object]]:
        if status not in REVIEW_STATUSES:
            raise ValueError(f"unknown review status: {status}")
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM memory_candidates WHERE status = ? "
                "ORDER BY created_at DESC, id DESC",
                (status,),
            ).fetchall()
        return [self._row(row) for row in rows]

    def get(self, candidate_id: str) -> dict[str, object]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM memory_candidates WHERE id = ?", (candidate_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown memory candidate: {candidate_id}")
        return self._row(row)

    def update_pending(
        self,
        candidate_id: str,
        *,
        scope: str | None = None,
        content: str | None = None,
        note: str = "",
    ) -> dict[str, object]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM memory_candidates WHERE id = ?", (candidate_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown memory candidate: {candidate_id}")
            if row["status"] != "pending":
                raise ValueError("only pending memory candidates can be updated")

            next_scope = self._validate_scope(scope if scope is not None else row["scope"])
            next_content = self._validate_content(
                content if content is not None else row["content"]
            )
            now = datetime.now(timezone.utc).isoformat()
            content_hash = hashlib.sha256(next_content.encode("utf-8")).hexdigest()
            connection.execute(
                "UPDATE memory_candidates SET scope = ?, content = ?, content_hash = ?, "
                "decision_note = ?, updated_at = ? WHERE id = ?",
                (next_scope, next_content, content_hash, note.strip(), now, candidate_id),
            )
            self._insert_event(
                connection,
                candidate_id=candidate_id,
                event_type="updated",
                note=note,
                created_at=now,
            )
            connection.commit()
            updated = connection.execute(
                "SELECT * FROM memory_candidates WHERE id = ?", (candidate_id,)
            ).fetchone()
        assert updated is not None
        return self._row(updated)

    def transition(
        self,
        candidate_id: str,
        *,
        expected_status: str,
        status: str,
        target_path: str | None = None,
        note: str = "",
    ) -> dict[str, object]:
        if status not in REVIEW_STATUSES:
            raise ValueError(f"unknown review status: {status}")
        now = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM memory_candidates WHERE id = ?", (candidate_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown memory candidate: {candidate_id}")
            if row["status"] != expected_status:
                raise ValueError(
                    f"memory candidate must be {expected_status} before becoming {status}"
                )
            connection.execute(
                "UPDATE memory_candidates SET status = ?, target_path = COALESCE(?, target_path), "
                "decision_note = ?, updated_at = ? WHERE id = ?",
                (status, target_path, note.strip(), now, candidate_id),
            )
            self._insert_event(
                connection,
                candidate_id=candidate_id,
                event_type=status,
                note=note,
                created_at=now,
            )
            connection.commit()
            updated = connection.execute(
                "SELECT * FROM memory_candidates WHERE id = ?", (candidate_id,)
            ).fetchone()
        assert updated is not None
        return self._row(updated)

    def history(self, candidate_id: str) -> list[dict[str, object]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT event_type, note, created_at FROM memory_review_events "
                "WHERE candidate_id = ? ORDER BY id",
                (candidate_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def counts(self) -> dict[str, int]:
        result = {status: 0 for status in REVIEW_STATUSES}
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) FROM memory_candidates GROUP BY status"
            ).fetchall()
        for status, count in rows:
            result[str(status)] = int(count)
        return result

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_candidates (
                id TEXT PRIMARY KEY,
                scope TEXT NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                reason TEXT NOT NULL,
                source TEXT NOT NULL,
                session_id TEXT,
                status TEXT NOT NULL,
                target_path TEXT,
                decision_note TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_candidates_status_created "
            "ON memory_candidates(status, created_at DESC)"
        )
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_review_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                note TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(candidate_id) REFERENCES memory_candidates(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_memory_review_events_candidate
            ON memory_review_events(candidate_id, id);
            """
        )
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection,
        *,
        candidate_id: str,
        event_type: str,
        note: str,
        created_at: str,
    ) -> None:
        connection.execute(
            "INSERT INTO memory_review_events(candidate_id, event_type, note, created_at) "
            "VALUES (?, ?, ?, ?)",
            (candidate_id, event_type, note.strip()[:1000], created_at),
        )

    @staticmethod
    def _validate_scope(scope: str) -> str:
        if scope not in MEMORY_SCOPES:
            raise ValueError(f"unknown memory scope: {scope}")
        return scope

    @staticmethod
    def _validate_content(content: str) -> str:
        normalized = content.strip()
        if not 3 <= len(normalized) <= 4000:
            raise ValueError("memory content must be between 3 and 4000 characters")
        if any(pattern.search(normalized) for pattern in SECRET_PATTERNS):
            raise ValueError("memory proposal appears to contain a secret")
        return normalized

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, object]:
        return {key: row[key] for key in row.keys() if key != "content_hash"}
