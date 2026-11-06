"""检索层边界回归测试

覆盖三个已复现的缺陷：
  1. BM25Retriever.build_index 对空语料抛 ZeroDivisionError
     （导致 main.py:378 用 `and app_state._chunks` 守卫绕行，
      进而使「删除到空库」时 BM25 索引不更新、已删文档仍被召回）
  2. Reranker 三个后端写回输入 Document 的 metadata
     （违反 tests/test_retrieval_logic.py:8 声明的既有不变量）
  3. 空语料重建后检索必须返回空，而不是保留陈旧结果

运行方式：
  python -m unittest tests.test_retrieval_edges -v
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document

from retrieval.hybrid_retriever import BM25Retriever
from retrieval.reranker import Reranker


def _corpus():
    """12 篇语料，仅 physics.md 含「量子纠缠」，保证 idf > 0。"""
    docs = [
        Document(
            page_content="量子纠缠 是 一种 物理 现象",
            metadata={"source": "physics.md", "page": 1, "chunk_index": 1},
        )
    ]
    for i in range(11):
        docs.append(
            Document(
                page_content=f"无关 内容 第{i}篇 普通 文本",
                metadata={"source": f"other{i}.md", "page": 1, "chunk_index": 1},
            )
        )
    return docs


class BM25EmptyCorpusTests(unittest.TestCase):
    """缺陷 1：空语料不应让 build_index 崩溃。"""

    def test_build_index_accepts_empty_corpus(self):
        retriever = BM25Retriever()
        try:
            retriever.build_index([])
        except ZeroDivisionError as exc:  # 修复前的实际行为
            self.fail(f"build_index([]) 不应抛 ZeroDivisionError: {exc}")

    def test_empty_index_search_returns_empty(self):
        retriever = BM25Retriever()
        retriever.build_index([])
        self.assertEqual(retriever.search("量子纠缠", top_k=5), [])


class StaleIndexTests(unittest.TestCase):
    """缺陷 3：清空语料后重建，检索不得再召回已删除内容。"""

    def test_rebuild_with_empty_corpus_drops_previous_results(self):
        retriever = BM25Retriever()
        retriever.build_index(_corpus())
        self.assertEqual(
            [d.metadata["source"] for d in retriever.search("量子纠缠", top_k=5)],
            ["physics.md"],
        )

        # 模拟 main.py kb_delete_documents：删光全部文档后重建索引。
        # 修复前 build_index([]) 会崩溃，main.py:378 的守卫因此跳过重建，
        # 索引停留在 12 篇陈旧状态，physics.md 仍被召回。
        retriever.build_index([])
        self.assertEqual(
            retriever.search("量子纠缠", top_k=5),
            [],
            "空库重建后不应再召回已删除文档",
        )

    def test_rebuild_with_subset_drops_removed_source(self):
        retriever = BM25Retriever()
        corpus = _corpus()
        retriever.build_index(corpus)

        remaining = [d for d in corpus if d.metadata["source"] != "physics.md"]
        retriever.build_index(remaining)
        self.assertEqual(retriever.search("量子纠缠", top_k=5), [])


class RerankerPurityTests(unittest.TestCase):
    """缺陷 2：重排不得写回输入 Document 的 metadata。"""

    def _docs(self):
        return [
            Document(
                page_content=f"内容 {i}",
                metadata={"source": "a.md", "page": 1, "chunk_index": i},
            )
            for i in range(3)
        ]

    def _assert_inputs_untouched(self, provider, rerank_call, documents):
        before = [dict(d.metadata) for d in documents]
        results = rerank_call()
        after = [dict(d.metadata) for d in documents]
        self.assertEqual(
            before,
            after,
            f"{provider} 后端污染了输入 Document 的 metadata",
        )
        # 分数应出现在返回的副本上，而不是丢失
        self.assertTrue(
            all("rerank_score" in d.metadata for d in results),
            f"{provider} 返回结果应带 rerank_score",
        )

    def test_cross_encoder_does_not_mutate_inputs(self):
        reranker = Reranker(provider="cross-encoder", model_name="fake")
        reranker._model = mock.Mock()
        reranker._model.predict.side_effect = lambda pairs: [
            0.9 - 0.1 * i for i in range(len(pairs))
        ]

        docs = self._docs()
        self._assert_inputs_untouched(
            "cross-encoder", lambda: reranker.rerank("q", docs, top_n=2), docs
        )

    def test_cohere_does_not_mutate_inputs(self):
        reranker = Reranker(provider="cohere", model_name="fake")
        item = mock.Mock(index=1, relevance_score=0.77)
        reranker._model = mock.Mock()
        reranker._model.rerank.return_value = mock.Mock(results=[item])

        docs = self._docs()
        self._assert_inputs_untouched(
            "cohere", lambda: reranker.rerank("q", docs, top_n=2), docs
        )

    def test_siliconflow_does_not_mutate_inputs(self):
        reranker = Reranker(provider="siliconflow", model_name="fake")
        reranker._model = {"api_base": "https://example.invalid/v1", "api_key": "k"}

        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "results": [{"index": 2, "relevance_score": 0.5}]
        }

        docs = self._docs()
        with mock.patch("retrieval.reranker.requests.post", return_value=response):
            self._assert_inputs_untouched(
                "siliconflow", lambda: reranker.rerank("q", docs, top_n=2), docs
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
