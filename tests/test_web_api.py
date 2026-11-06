"""Guards and upload handling of the RAG web/REST service (main.py + api.py)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from api import create_api


class _State(SimpleNamespace):
    def is_ready(self) -> bool:
        return False


def _client(*, base_url: str = "http://127.0.0.1:7860", **kwargs) -> tuple[TestClient, dict]:
    calls: dict = {"build": [], "clear": 0}

    def build(uploads):
        calls["build"].append(list(uploads))
        return {"added_chunks": 3, "doc_count": 3}

    def clear():
        calls["clear"] += 1
        return "cleared"

    kwargs.setdefault("allowed_hosts", "")
    kwargs.setdefault("api_token", "")
    app = create_api(
        _State(doc_count=0),
        build_documents_fn=build,
        clear_fn=clear,
        **kwargs,
    )
    return TestClient(app, base_url=base_url), calls


class WebApiGuardTests(unittest.TestCase):
    def test_foreign_host_is_rejected_everywhere(self) -> None:
        client, calls = _client(base_url="http://evil.example:7860")
        self.assertEqual(client.get("/api/status").status_code, 403)
        self.assertEqual(client.get("/docs").status_code, 403)
        self.assertEqual(client.post("/api/clear", json={"confirm": True}).status_code, 403)
        self.assertEqual(calls["clear"], 0)

    def test_clear_needs_json_and_explicit_confirmation(self) -> None:
        client, calls = _client()
        # A cross-site <form> or fetch(no-cors) can only send "simple" requests.
        self.assertEqual(client.post("/api/clear").status_code, 415)
        self.assertEqual(
            client.post("/api/clear", content="{}", headers={"Content-Type": "text/plain"}).status_code,
            415,
        )
        refused = client.post("/api/clear", json={})
        self.assertFalse(refused.json()["success"])
        self.assertEqual(calls["clear"], 0)

        cleared = client.post("/api/clear", json={"confirm": True})
        self.assertTrue(cleared.json()["success"])
        self.assertEqual(calls["clear"], 1)

    def test_build_requires_custom_header_and_keeps_original_names(self) -> None:
        client, calls = _client()
        files = {"files": ("课程笔记.md", "# 标题\n\n正文", "text/markdown")}

        self.assertEqual(client.post("/api/build", files=files).status_code, 403)
        self.assertEqual(calls["build"], [])

        accepted = client.post("/api/build", files=files, headers={"X-RAG-Client": "test"})
        self.assertEqual(accepted.status_code, 200)
        self.assertTrue(accepted.json()["success"])
        [(tmp_path, display_name)] = calls["build"][0]
        self.assertEqual(display_name, "课程笔记.md")
        # The temporary upload directory is removed after ingestion.
        self.assertFalse(Path(tmp_path).exists())

    def test_optional_bearer_token_and_extra_hosts(self) -> None:
        client, _ = _client(base_url="http://nas.lan:7860", allowed_hosts="nas.lan", api_token="s3cret")
        self.assertEqual(client.get("/api/status").status_code, 401)
        self.assertEqual(
            client.get("/api/status", headers={"Authorization": "Bearer wrong"}).status_code, 401
        )
        self.assertEqual(
            client.get("/api/status", headers={"Authorization": "Bearer s3cret"}).status_code, 200
        )
        # The token only guards /api/*; the UI shell is still reachable on a trusted host.
        self.assertNotEqual(client.get("/docs").status_code, 401)


class _FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [float(len(text) % 7 + 1), float(sum(map(ord, text)) % 11 + 1), 1.0]


class UploadIngestionTests(unittest.TestCase):
    def setUp(self) -> None:
        import main

        self.main = main
        # Chroma keeps its index files open on Windows, so cleanup may not delete them.
        self.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.root = root
        from vectorstore.store import VectorStoreManager

        self.manager = VectorStoreManager(
            persist_dir=str(root / "chroma"),
            collection_name="upload_test",
            embedding_model=_FakeEmbeddings(),
        )
        self.manager.open()
        # A chunk that came from SecondBrain sync must survive uploads.
        self.manager.add_documents_with_ids(
            [Document(page_content="secondbrain note", metadata={"source": "data/n.md", "source_kind": "filesystem"})],
            ["rag_existing"],
        )
        patches = [
            mock.patch.object(main.app_state, "vectorstore_manager", self.manager),
            mock.patch.object(main, "QAChain"),
            mock.patch.object(main, "RAGWorkflow"),
            mock.patch.object(main, "QueryRewriter"),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _write(self, name: str, text: str) -> str:
        path = self.root / "uploads" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_upload_is_tagged_replaces_same_name_and_rebuilds_full_bm25(self) -> None:
        first = self.main.ingest_uploaded_files([(self._write("0.md", "first version"), "a.md")])
        second = self.main.ingest_uploaded_files([(self._write("1.md", "second version"), "a.md")])

        self.assertEqual(first["added_chunks"], 1)
        self.assertEqual(second["removed_chunks"], 1)
        snapshot = self.manager.get_snapshot()
        uploads = [
            (doc, meta)
            for doc, meta in zip(snapshot["documents"], snapshot["metadatas"])
            if meta.get("origin") == "upload"
        ]
        self.assertEqual([doc for doc, _ in uploads], ["second version"])
        meta = uploads[0][1]
        self.assertEqual(meta["file_name"], "a.md")
        self.assertEqual(meta["source"], "upload::a.md")
        self.assertEqual(meta["course_id"], self.main.UPLOAD_COURSE_ID)

        # BM25 covers both the pre-existing SecondBrain chunk and the upload.
        contents = sorted(chunk.page_content for chunk in self.main.app_state._chunks)
        self.assertEqual(contents, ["second version", "secondbrain note"])
        self.assertEqual(self.main.app_state.doc_count, 2)

    def test_clear_empties_collection_without_removing_directory(self) -> None:
        self.main.clear_knowledge_base()
        self.assertEqual(self.manager.count(), 0)
        self.assertTrue((self.root / "chroma").exists())
        self.assertIsNone(self.main.app_state.qa_chain)


if __name__ == "__main__":
    unittest.main()
