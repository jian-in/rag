import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from chain.qa_chain import QAChain


def test_multi_query_retrieve_rrf_and_accurate_dedup():
    """测试多查询检索：具备倒数排名融合能力，且去重准确（不误伤前100字符相同的内容）。"""
    mock_retriever = MagicMock()
    
    # 构造文档：
    # doc_common 在 query1 和 query2 中都被召回（第1名和第2名）
    # doc_unique1 仅在 query1 中被召回（第2名）
    # doc_unique2 仅在 query2 中被召回（第1名）
    # doc_prefix_a 和 doc_prefix_b 具有相同的前100字符，但后续不同，不应被粗暴误去重
    prefix = "前缀说明" * 25  # 100 字符完全相同的前缀
    doc_prefix_a = Document(
        page_content=prefix + "=== 差异内容甲 ===",
        metadata={"source": "docA.txt", "chunk_index": 1}
    )
    doc_prefix_b = Document(
        page_content=prefix + "=== 差异内容乙 ===",
        metadata={"source": "docB.txt", "chunk_index": 2}
    )
    
    doc_common = Document(
        page_content="这是一篇核心共性文档，被多个查询同时重点召回。",
        metadata={"source": "common.txt", "chunk_index": 1}
    )
    doc_q1_only = Document(
        page_content="这是仅仅在查询1中出现的特定细节文档。",
        metadata={"source": "q1.txt", "chunk_index": 1}
    )
    
    # query1 返回: [doc_q1_only, doc_common, doc_prefix_a]
    # query2 返回: [doc_common, doc_prefix_b]
    def mock_retrieve(query, strategy="mmr", use_reranker=False):
        if query == "query1":
            return [doc_q1_only, doc_common, doc_prefix_a]
        elif query == "query2":
            return [doc_common, doc_prefix_b]
        return []
    
    mock_retriever.retrieve = MagicMock(side_effect=mock_retrieve)
    
    # 绕过 __init__ 里的网络初始化，直接构建 QAChain 实例
    chain = QAChain.__new__(QAChain)
    chain.retriever = mock_retriever
    
    results = chain._multi_query_retrieve(
        queries=["query1", "query2"],
        strategy="mmr",
        use_reranker=False,
    )
    
    # 1. doc_prefix_a 和 doc_prefix_b 都应该被保留，不能因为前100字相同就被吞掉一个
    contents = [d.page_content for d in results]
    assert doc_prefix_a.page_content in contents
    assert doc_prefix_b.page_content in contents
    
    # 2. doc_common 应该被去重且由于在两个查询中均被召回，其 RRF 综合得分更高，排在首位
    assert results[0].page_content == doc_common.page_content
