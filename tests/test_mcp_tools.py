"""MCP tool argument validation and multi-query fusion in KnowledgeRuntime."""

from __future__ import annotations

import unittest
from unittest import mock

from langchain_core.documents import Document

import mcp_server
from mcp_server import KnowledgeRuntime, MCPError, MCPServer


def _doc(source: str, index: int = 1) -> Document:
    return Document(page_content=f"{source}#{index}", metadata={"source": source, "page": "", "chunk_index": index})


class ToolArgumentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = MCPServer()
        self.search = mock.Mock(return_value={"results": []})
        self.ask = mock.Mock(return_value={"answer": ""})
        self.server.runtime.search = self.search
        self.server.runtime.ask = self.ask

    def _search(self, **arguments):
        self.server._dispatch_tool("search_knowledge", {"query": "q", **arguments})
        return self.search.call_args.kwargs

    def test_out_of_range_numbers_are_clamped_to_schema(self) -> None:
        self.assertEqual(self._search(window=500)["window"], mcp_server.MAX_WINDOW)
        self.assertEqual(self._search(window=-2)["window"], 0)
        self.assertEqual(self._search(top_k=10000)["top_k"], mcp_server.MAX_TOP_K)
        self.assertEqual(self._search(top_k=0)["top_k"], 1)
        self.assertEqual(self._search(top_k="7")["top_k"], 7)
        defaults = self._search()
        self.assertEqual((defaults["top_k"], defaults["window"], defaults["use_hybrid"]), (5, 0, True))

    def test_string_false_is_not_truthy(self) -> None:
        self.assertIs(self._search(use_hybrid="false")["use_hybrid"], False)
        self.assertIs(self._search(use_hybrid=False)["use_hybrid"], False)

    def test_invalid_arguments_are_rejected(self) -> None:
        for arguments in (
            {"top_k": "many"},
            {"top_k": True},
            {"window": [1]},
            {"use_hybrid": "maybe"},
            {"rewrite_strategy": "bogus"},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(MCPError):
                self.server._dispatch_tool("search_knowledge", {"query": "q", **arguments})
        for query in ("", "   ", None, 3):
            with self.subTest(query=query), self.assertRaises(MCPError):
                self.server._dispatch_tool("search_knowledge", {"query": query})
        with self.assertRaises(MCPError):
            self.server._dispatch_tool("ask_knowledge", {"question": ""})

    def test_ask_uses_the_same_validation(self) -> None:
        self.server._dispatch_tool("ask_knowledge", {"question": " why? ", "window": 9})
        kwargs = self.ask.call_args.kwargs
        self.assertEqual(kwargs["question"], "why?")
        self.assertEqual(kwargs["window"], mcp_server.MAX_WINDOW)


class MultiQueryFusionTests(unittest.TestCase):
    def _runtime(self, rewritten, recalls, reranker=None) -> KnowledgeRuntime:
        runtime = KnowledgeRuntime()
        runtime.loaded = True
        runtime.vector_retriever = mock.Mock(reranker=reranker)
        runtime.hybrid_retriever = mock.Mock()
        runtime.qa_chain = mock.Mock()
        runtime.qa_chain.rewriter.rewrite.return_value = rewritten
        self.recall_calls = []

        def recall(query, top_k, use_hybrid, *, use_reranker):
            self.recall_calls.append((query, use_reranker))
            return recalls[query]

        runtime._recall = recall
        return runtime

    def test_fusion_keeps_hits_from_later_queries(self) -> None:
        recalls = {
            "原问题": [_doc("shared"), _doc("orig-only")],
            "改写一": [_doc("a1"), _doc("a2"), _doc("shared")],
            "改写二": [_doc("b1"), _doc("shared")],
        }
        runtime = self._runtime(["改写一", "改写二"], recalls)

        with mock.patch.object(mcp_server.config, "ENABLE_RERANKER", False):
            docs = runtime._retrieve_docs("原问题", top_k=3, use_hybrid=True, rewrite_strategy="decompose")

        sources = [doc.metadata["source"] for doc in docs]
        # "shared" is hit by every query and wins; the other slots go to the
        # best-ranked hits of *different* queries, not just the first one's.
        self.assertEqual(sources[0], "shared")
        self.assertEqual(len(sources), 3)
        self.assertTrue({"orig-only", "a1", "b1"} & set(sources[1:]))
        self.assertNotEqual(sources[1:], ["a1", "a2"])
        self.assertEqual([query for query, _ in self.recall_calls], ["原问题", "改写一", "改写二"])
        self.assertTrue(all(use_reranker is False for _, use_reranker in self.recall_calls))
        self.assertIn("rrf_score", docs[0].metadata)

    def test_rerank_runs_once_on_fused_pool_and_degrades_on_failure(self) -> None:
        recalls = {"q": [_doc("x")], "r": [_doc("y"), _doc("z")]}
        reranker = mock.Mock()
        reranker.rerank.side_effect = RuntimeError("rerank 503")
        runtime = self._runtime(["r"], recalls, reranker=reranker)

        with mock.patch.object(mcp_server.config, "ENABLE_RERANKER", True):
            docs = runtime._retrieve_docs("q", top_k=2, use_hybrid=False, rewrite_strategy="expand")

        reranker.rerank.assert_called_once()
        self.assertEqual(reranker.rerank.call_args.args[0], "q")
        self.assertEqual(len(docs), 2)

    def test_duplicate_rewrites_of_the_original_are_not_recalled_twice(self) -> None:
        runtime = self._runtime(["q", ""], {"q": [_doc("x")]})
        with mock.patch.object(mcp_server.config, "ENABLE_RERANKER", False):
            runtime._retrieve_docs("q", top_k=2, use_hybrid=True, rewrite_strategy="expand")
        self.assertEqual(self.recall_calls, [("q", False)])


if __name__ == "__main__":
    unittest.main()
