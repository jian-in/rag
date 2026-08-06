from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from ingestion.incremental_sync import SecondBrainMirror, sync_vectorstore
from ingestion.loader import DocumentLoader
from ingestion.splitter import TextSplitter
from mcp_server import MCPServer


class FakeCollection:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    def get(self, include=None):
        rows = list(self.rows.values())
        return {
            "ids": [row["id"] for row in rows],
            "documents": [row["document"] for row in rows],
            "metadatas": [row["metadata"] for row in rows],
        }

    def count(self) -> int:
        return len(self.rows)

    def delete(self, ids=None, where=None) -> None:
        if ids:
            for item_id in ids:
                self.rows.pop(item_id, None)
            return
        if where and "source" in where:
            source = where["source"]
            for item_id, row in list(self.rows.items()):
                if row["metadata"].get("source") == source:
                    del self.rows[item_id]


class FakeVectorStore:
    def __init__(self) -> None:
        self._collection = FakeCollection()

    def add_documents(self, documents, ids=None, **kwargs) -> None:
        ids = ids or [f"generated-{index}" for index, _ in enumerate(documents)]
        for item_id, document in zip(ids, documents):
            self._collection.rows[item_id] = {
                "id": item_id,
                "document": document.page_content,
                "metadata": dict(document.metadata),
            }

    def get_snapshot(self):
        return self._collection.get(include=["documents", "metadatas"])

    def add_documents_with_ids(self, documents, ids) -> None:
        self.add_documents(documents, ids=ids)

    def delete_ids(self, ids) -> None:
        self._collection.delete(ids=ids)

    def count(self) -> int:
        return self._collection.count()


class IncrementalSyncTests(unittest.TestCase):
    def test_mirror_is_incremental_and_removes_deleted_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "SecondBrain"
            mirror = root / "data"
            note = source / "02-Areas" / "topic" / "note.md"
            note.parent.mkdir(parents=True)
            note.write_text("# first\n\nalpha", encoding="utf-8")

            report = SecondBrainMirror(
                source_root=source,
                mirror_root=mirror,
                include_dirs=("02-Areas",),
                manifest_path=root / "mirror-manifest.json",
            ).sync()
            target = mirror / "topic" / "note.md"
            self.assertEqual(report.added_files, ["02-Areas/topic/note.md"])
            self.assertEqual(target.read_text(encoding="utf-8"), "# first\n\nalpha")

            second = SecondBrainMirror(
                source_root=source,
                mirror_root=mirror,
                include_dirs=("02-Areas",),
                manifest_path=root / "mirror-manifest.json",
            ).sync()
            self.assertEqual(second.unchanged_files, ["02-Areas/topic/note.md"])

            note.write_text("# changed\n\nbeta", encoding="utf-8")
            changed = SecondBrainMirror(
                source_root=source,
                mirror_root=mirror,
                include_dirs=("02-Areas",),
                manifest_path=root / "mirror-manifest.json",
            ).sync()
            self.assertEqual(changed.changed_files, ["02-Areas/topic/note.md"])
            self.assertEqual(target.read_text(encoding="utf-8"), "# changed\n\nbeta")

            note.unlink()
            removed = SecondBrainMirror(
                source_root=source,
                mirror_root=mirror,
                include_dirs=("02-Areas",),
                manifest_path=root / "mirror-manifest.json",
            ).sync()
            self.assertEqual(removed.removed_files, ["02-Areas/topic/note.md"])
            self.assertFalse(target.exists())

    def test_vector_sync_is_idempotent_and_updates_changed_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            note = root / "note.md"
            note.write_text("# first\n\nalpha", encoding="utf-8")
            manifest = root / "vector-manifest.json"
            store = FakeVectorStore()

            first = sync_vectorstore(
                store,
                [note],
                loader=DocumentLoader(),
                splitter=TextSplitter(chunk_size=100, chunk_overlap=10),
                manifest_path=manifest,
            )
            self.assertEqual(first.added_files, 1)
            first_count = store._collection.count()
            self.assertGreater(first_count, 0)

            second = sync_vectorstore(
                store,
                [note],
                loader=DocumentLoader(),
                splitter=TextSplitter(chunk_size=100, chunk_overlap=10),
                manifest_path=manifest,
            )
            self.assertEqual(second.unchanged_files, 1)
            self.assertEqual(store._collection.count(), first_count)

            note.write_text("# changed\n\nbeta", encoding="utf-8")
            changed = sync_vectorstore(
                store,
                [note],
                loader=DocumentLoader(),
                splitter=TextSplitter(chunk_size=100, chunk_overlap=10),
                manifest_path=manifest,
            )
            self.assertEqual(changed.changed_files, 1)
            self.assertEqual(store._collection.count(), first_count)
            self.assertIn(
                "beta",
                "\n".join(row["document"] for row in store._collection.rows.values()),
            )

            removed = sync_vectorstore(
                store,
                [],
                loader=DocumentLoader(),
                splitter=TextSplitter(chunk_size=100, chunk_overlap=10),
                manifest_path=manifest,
            )
            self.assertEqual(removed.removed_files, 1)
            self.assertEqual(store._collection.count(), 0)

    def test_mcp_exposes_sync_tool(self) -> None:
        names = {tool["name"] for tool in MCPServer()._tool_definitions()}
        self.assertIn("sync_knowledge_base", names)


if __name__ == "__main__":
    unittest.main()
