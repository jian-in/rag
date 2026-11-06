"""Parent-window expansion must merge overlapping hits without dropping any."""

from __future__ import annotations

import unittest

from langchain_core.documents import Document

from mcp_server import KnowledgeRuntime


def _chunks(source: str, count: int) -> list[Document]:
    return [
        Document(
            page_content=f"{source}-C{index}",
            metadata={"source": source, "page": "", "chunk_index": index},
        )
        for index in range(1, count + 1)
    ]


def _runtime(*chunk_lists: list[Document]) -> KnowledgeRuntime:
    runtime = KnowledgeRuntime()
    runtime._build_chunk_index([chunk for chunks in chunk_lists for chunk in chunks])
    return runtime


def _parts(doc: Document) -> list[str]:
    return doc.page_content.split("\n\n")


class WindowExpansionTests(unittest.TestCase):
    def test_later_hit_before_earlier_window_is_kept(self) -> None:
        a = _chunks("a.md", 14)
        runtime = _runtime(a)

        # Rank 1 hits chunk 10, rank 2 hits chunk 8 of the same file.
        expanded = runtime._expand_with_neighbors([a[9], a[7]], window=1)

        self.assertEqual(len(expanded), 1)
        self.assertEqual(
            _parts(expanded[0]),
            ["a.md-C7", "a.md-C8", "a.md-C9", "a.md-C10", "a.md-C11"],
        )
        self.assertEqual(expanded[0].metadata["window_start"], 7)
        self.assertEqual(expanded[0].metadata["window_end"], 11)
        # The merged window keeps the best-ranked hit's metadata.
        self.assertEqual(expanded[0].metadata["chunk_index"], 10)

    def test_non_adjacent_hits_of_same_file_merge_across_other_files(self) -> None:
        a = _chunks("a.md", 10)
        b = _chunks("b.md", 10)
        runtime = _runtime(a, b)

        expanded = runtime._expand_with_neighbors([a[2], b[4], a[3]], window=1)

        self.assertEqual([doc.metadata["source"] for doc in expanded], ["a.md", "b.md"])
        self.assertEqual(_parts(expanded[0]), ["a.md-C2", "a.md-C3", "a.md-C4", "a.md-C5"])
        self.assertEqual(_parts(expanded[1]), ["b.md-C4", "b.md-C5", "b.md-C6"])

    def test_bridging_hit_merges_two_windows_at_best_rank(self) -> None:
        a = _chunks("a.md", 20)
        b = _chunks("b.md", 5)
        runtime = _runtime(a, b)

        # Windows [1-3] and [7-9] are separate until the hit on chunk 5 ([4-6]) bridges them.
        expanded = runtime._expand_with_neighbors([a[1], b[0], a[7], a[4]], window=1)

        self.assertEqual([doc.metadata["source"] for doc in expanded], ["a.md", "b.md"])
        self.assertEqual(expanded[0].metadata["window_start"], 1)
        self.assertEqual(expanded[0].metadata["window_end"], 9)
        self.assertEqual(len(_parts(expanded[0])), 9)

    def test_disjoint_windows_stay_separate_and_inputs_are_not_mutated(self) -> None:
        a = _chunks("a.md", 20)
        runtime = _runtime(a)
        hits = [a[1], a[14]]
        original = [(doc.page_content, dict(doc.metadata)) for doc in hits]

        expanded = runtime._expand_with_neighbors(hits, window=1)

        self.assertEqual(len(expanded), 2)
        self.assertEqual(_parts(expanded[1]), ["a.md-C14", "a.md-C15", "a.md-C16"])
        self.assertEqual([(doc.page_content, doc.metadata) for doc in hits], original)

    def test_hits_without_chunk_index_pass_through(self) -> None:
        runtime = _runtime(_chunks("a.md", 3))
        loose = Document(page_content="loose", metadata={"source": "x.md"})

        expanded = runtime._expand_with_neighbors([loose], window=2)

        self.assertIs(expanded[0], loose)


if __name__ == "__main__":
    unittest.main()
