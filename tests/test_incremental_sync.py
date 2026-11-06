from __future__ import annotations

import json
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


class FailingVectorStore(FakeVectorStore):
    """Vector store whose writes fail on demand, like an embedding API outage."""

    def __init__(self) -> None:
        super().__init__()
        self.fail_writes = False

    def add_documents_with_ids(self, documents, ids) -> None:
        if self.fail_writes:
            raise RuntimeError("embedding API unavailable")
        super().add_documents_with_ids(documents, ids)


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

    def test_failed_write_is_retried_instead_of_marked_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            note = root / "note.md"
            manifest = root / "manifest.json"
            store = FailingVectorStore()
            splitter = TextSplitter(chunk_size=100, chunk_overlap=10)

            note.write_text("old content version one", encoding="utf-8")
            sync_vectorstore(store, [note], DocumentLoader(), splitter, manifest)

            note.write_text("new content version two", encoding="utf-8")
            store.fail_writes = True
            failed = sync_vectorstore(store, [note], DocumentLoader(), splitter, manifest)
            self.assertEqual(len(failed.errors), 1)

            store.fail_writes = False
            retried = sync_vectorstore(store, [note], DocumentLoader(), splitter, manifest)

            self.assertEqual(retried.errors, [])
            self.assertEqual(retried.changed_files, 1)
            self.assertEqual(retried.unchanged_files, 0)
            contents = store.get_snapshot()["documents"]
            self.assertEqual(contents, ["new content version two"])

    def test_failed_write_of_new_file_is_not_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            note = root / "note.md"
            manifest = root / "manifest.json"
            store = FailingVectorStore()
            store.fail_writes = True
            note.write_text("brand new note", encoding="utf-8")

            sync_vectorstore(
                store,
                [note],
                DocumentLoader(),
                TextSplitter(chunk_size=100, chunk_overlap=10),
                manifest,
            )

            saved = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertNotIn(str(note.resolve()), saved["files"])

    def test_pipeline_upgrade_keeps_old_signature_when_a_write_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            note = root / "note.md"
            manifest = root / "manifest.json"
            store = FailingVectorStore()
            note.write_text("stable content " * 20, encoding="utf-8")
            sync_vectorstore(
                store, [note], DocumentLoader(), TextSplitter(chunk_size=100, chunk_overlap=10), manifest
            )
            old_signature = json.loads(manifest.read_text(encoding="utf-8"))["pipeline_signature"]

            store.fail_writes = True
            sync_vectorstore(
                store, [note], DocumentLoader(), TextSplitter(chunk_size=80, chunk_overlap=10), manifest
            )

            saved = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(saved["pipeline_signature"], old_signature)

    def test_sync_reclaims_untracked_dead_mirror_sources_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mirror = root / "data"
            mirror.mkdir()
            live = mirror / "live.md"
            live.write_text("still here", encoding="utf-8")
            outside_dead = root / "elsewhere" / "gone.md"
            store = FakeVectorStore()
            from langchain_core.documents import Document

            untracked = {
                "orphan-abs": str((mirror / "gone.md").resolve()),
                "orphan-rel": "sub/gone-too.md",
                "kece": "kece::c1::d1",
                "upload": "upload::notes.pdf",
                "outside": str(outside_dead),
            }
            for item_id, source in untracked.items():
                store.add_documents_with_ids(
                    [Document(page_content=item_id, metadata={"source": source})], [item_id]
                )

            report = sync_vectorstore(
                store,
                [live],
                DocumentLoader(),
                TextSplitter(chunk_size=100, chunk_overlap=10),
                root / "manifest.json",
                managed_root=mirror,
            )

            remaining = set(store._collection.rows)
            self.assertNotIn("orphan-abs", remaining)
            self.assertNotIn("orphan-rel", remaining)
            self.assertTrue({"kece", "upload", "outside"} <= remaining)
            self.assertEqual(report.orphan_files, 2)
            self.assertEqual(report.added_files, 1)

    def test_orphan_cleanup_needs_a_managed_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = FakeVectorStore()
            from langchain_core.documents import Document

            store.add_documents_with_ids(
                [Document(page_content="x", metadata={"source": str(root / "gone.md")})], ["orphan"]
            )
            sync_vectorstore(
                store,
                [],
                DocumentLoader(),
                TextSplitter(chunk_size=100, chunk_overlap=10),
                root / "manifest.json",
            )
            self.assertIn("orphan", store._collection.rows)

    def test_mcp_exposes_sync_tool(self) -> None:
        names = {tool["name"] for tool in MCPServer()._tool_definitions()}
        self.assertIn("sync_knowledge_base", names)


if __name__ == "__main__":
    unittest.main()
