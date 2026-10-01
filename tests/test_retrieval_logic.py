"""
检索纯逻辑单元测试（无需网络 / API Key）

覆盖：
  - BM25 中英混合分词 `_tokenize`
  - RRF / 加权融合 `_rrf_fusion` / `_weighted_fusion`
  - 文档去重键 `_doc_id`
  - 融合不污染原始文档 metadata

运行方式：
  pytest tests/test_retrieval_logic.py
  或： python tests/test_retrieval_logic.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document

from retrieval.hybrid_retriever import BM25Retriever, HybridRetriever


def _doc(source, chunk_index, content="content", page=0):
    return Document(
        page_content=content,
        metadata={"source": source, "page": page, "chunk_index": chunk_index},
    )


# ============================================================
# 分词
# ============================================================

def test_tokenize_extracts_ascii_and_cjk():
    tokens = BM25Retriever()._tokenize("RAG 检索增强生成")
    # ASCII 词转小写
    assert "rag" in tokens
    # 中文 2-gram 覆盖
    assert "检索" in tokens
    # 中文 3-gram 覆盖
    assert "检索增" in tokens


def test_tokenize_normalizes_fullwidth_and_case():
    # 全角字母/数字经 NFKC 归一化并转小写
    tokens = BM25Retriever()._tokenize("ＡＢＣ123")
    assert "abc123" in tokens


def test_tokenize_empty_returns_empty():
    assert BM25Retriever()._tokenize("   ") == []


# ============================================================
# 融合：doc_id
# ============================================================

def test_doc_id_format():
    hr = HybridRetriever(bm25_retriever=None, vector_retriever=None)
    doc = _doc("a.md", 3, page=1)
    assert hr._doc_id(doc) == "a.md::1::3"


# ============================================================
# 融合：RRF
# ============================================================

def test_rrf_prefers_doc_ranked_high_in_both():
    hr = HybridRetriever(bm25_retriever=None, vector_retriever=None, rrf_k=60)
    shared = _doc("shared.md", 0)
    vector_results = [shared, _doc("v.md", 0)]
    bm25_results = [shared, _doc("b.md", 0)]

    fused = hr._rrf_fusion(bm25_results, vector_results, top_k=10)

    # 同时命中两路的文档应排第一
    assert hr._doc_id(fused[0]) == "shared.md::0::0"
    # 每个结果都带 rrf_score
    assert all("rrf_score" in d.metadata for d in fused)
    # 去重：3 个不同文档
    assert len(fused) == 3


def test_rrf_respects_top_k():
    hr = HybridRetriever(bm25_retriever=None, vector_retriever=None)
    vector_results = [_doc("v.md", i) for i in range(5)]
    fused = hr._rrf_fusion([], vector_results, top_k=2)
    assert len(fused) == 2


def test_rrf_does_not_mutate_inputs():
    hr = HybridRetriever(bm25_retriever=None, vector_retriever=None)
    src = _doc("v.md", 0)
    hr._rrf_fusion([], [src], top_k=5)
    # 原始文档 metadata 不应被写入融合分数
    assert "rrf_score" not in src.metadata


# ============================================================
# 融合：加权
# ============================================================

def test_weighted_fusion_uses_bm25_score():
    hr = HybridRetriever(bm25_retriever=None, vector_retriever=None, alpha=0.5)
    strong = _doc("strong.md", 0)
    strong.metadata["bm25_score"] = 10.0
    weak = _doc("weak.md", 0)
    weak.metadata["bm25_score"] = 1.0

    fused = hr._weighted_fusion([strong, weak], [], top_k=10)

    assert hr._doc_id(fused[0]) == "strong.md::0::0"
    assert all("hybrid_score" in d.metadata for d in fused)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
