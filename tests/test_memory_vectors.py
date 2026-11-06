"""Embedding reuse and in-memory vector index of the personal memory hub."""

from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

from memory_hub import MemoryHub, MemoryHubSettings
from memory_hub import core


class RecordingEmbeddingProvider:
    model_name = "recording-embedding"

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [float(byte) - 127.5 for byte in digest[:8]]


def _hub(root: Path, provider) -> MemoryHub:
    for name in ("vault/02-Areas", "memory", "rollouts"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return MemoryHub(
        MemoryHubSettings(
            index_root=root / "indexes",
            vault_root=root / "vault",
            memory_root=root / "memory",
            rollout_root=root / "rollouts",
            vault_include_dirs=("02-Areas",),
            chunk_size=100,
            chunk_overlap=0,
        ),
        embedding_provider=provider,
    )


SECTION_A = "# 甲\n\n" + "甲段落内容。" * 10
SECTION_B = "# 乙\n\n" + "乙段落内容。" * 10
SECTION_NEW = "# 新\n\n" + "新加的段落。" * 10


class EmbeddingReuseTests(unittest.TestCase):
    def test_editing_a_note_only_embeds_new_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            provider = RecordingEmbeddingProvider()
            hub = _hub(root, provider)
            note = root / "vault" / "02-Areas" / "笔记.md"

            note.write_text(SECTION_A + "\n\n" + SECTION_B, encoding="utf-8")
            hub.sync(scopes=("vault",))
            first_round = list(provider.texts)
            self.assertEqual(len(first_round), 2)

            # Prepending a section shifts every ordinal and chunk id.
            provider.texts.clear()
            note.write_text(SECTION_NEW + "\n\n" + SECTION_A + "\n\n" + SECTION_B, encoding="utf-8")
            report = hub.sync(scopes=("vault",))

            self.assertEqual(len(provider.texts), 1)
            self.assertIn("新加的段落", provider.texts[0])
            self.assertEqual(report["scopes"]["vault"]["vectors_added"], 3)

    def test_deleted_text_is_pruned_from_the_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hub = _hub(root, RecordingEmbeddingProvider())
            note = root / "vault" / "02-Areas" / "笔记.md"
            note.write_text(SECTION_A + "\n\n" + SECTION_B, encoding="utf-8")
            hub.sync(scopes=("vault",))

            note.write_text(SECTION_A, encoding="utf-8")
            hub.sync(scopes=("vault",))

            # sqlite3's own context manager commits but does not close.
            with closing(sqlite3.connect(root / "indexes" / "vault.sqlite3")) as connection:
                cached = connection.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0]
                vectors = connection.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
            self.assertEqual((cached, vectors), (1, 1))


class VectorIndexTests(unittest.TestCase):
    def _rows(self):
        vectors = {"a": [1.0, 0.0, 0.0], "b": [0.6, 0.8, 0.0], "c": [0.0, 0.0, 1.0], "d": [-1.0, 0.0, 0.0]}
        return [
            (key, f"/{key}.md", key, "", key, key, None, 3, core._pack_vector(vector))
            for key, vector in vectors.items()
        ]

    def test_numpy_and_pure_python_paths_agree(self) -> None:
        rows = self._rows()
        query = [0.9, 0.3, 0.1]
        fast = core._VectorIndex(rows).top(query, 3)
        with mock.patch.object(core, "_np", None):
            slow = core._VectorIndex(rows).top(query, 3)

        self.assertEqual([row[0] for _, row in fast], [row[0] for _, row in slow])
        for (fast_score, _), (slow_score, _) in zip(fast, slow):
            self.assertAlmostEqual(fast_score, slow_score, places=6)
        # Non-positive similarities are excluded, like the original scan.
        self.assertNotIn("d", [row[0] for _, row in fast])

    def test_dimension_mismatch_and_zero_query_return_nothing(self) -> None:
        index = core._VectorIndex(self._rows())
        self.assertEqual(index.top([1.0, 0.0], 3), [])
        self.assertEqual(index.top([0.0, 0.0, 0.0], 3), [])

    def test_index_is_cached_until_vectors_change(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hub = _hub(root, RecordingEmbeddingProvider())
            note = root / "vault" / "02-Areas" / "笔记.md"
            note.write_text(SECTION_A, encoding="utf-8")
            hub.sync(scopes=("vault",))

            first = hub._vector_index("vault", "recording-embedding")
            self.assertIs(hub._vector_index("vault", "recording-embedding"), first)

            note.write_text(SECTION_A + "\n\n" + SECTION_B, encoding="utf-8")
            hub.sync(scopes=("vault",))
            refreshed = hub._vector_index("vault", "recording-embedding")
            self.assertIsNot(refreshed, first)
            self.assertEqual(len(refreshed), 2)


if __name__ == "__main__":
    unittest.main()
