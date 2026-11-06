import pytest
from collections import deque
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from retrieval.query_rewriter import QueryRewriter
from retrieval.conversation_memory import ConversationMemory
from api import create_api


def test_decompose_strips_numbering_and_bullet_prefixes():
    """测试子问题分解：应当清洗掉模型输出的 1. / 2. / - / * 等前缀。"""
    qr = QueryRewriter.__new__(QueryRewriter)
    mock_resp = MagicMock()
    # 模拟大模型输出带有数字编号、点、破折号和项目符号的子查询
    mock_resp.content = (
        "1. 进程调度的基本概念是什么？\n"
        "2、常见操作系统调度算法有哪些？\n"
        "- 先来先服务算法的优缺点是什么？\n"
        "* 时间片轮转调度算法如何工作？"
    )
    qr.llm = MagicMock()
    qr.llm.invoke.return_value = mock_resp

    sub_queries = qr.decompose("介绍操作系统进程调度")
    # 断言所有前缀都被干净去除
    assert sub_queries == [
        "进程调度的基本概念是什么？",
        "常见操作系统调度算法有哪些？",
        "先来先服务算法的优缺点是什么？",
        "时间片轮转调度算法如何工作？",
    ]


def test_rewrite_fallback_to_original_query_on_error():
    """测试查询改写降级：当 LLM 出现网络波动或异常时，不应崩溃，而应降级返回原查询。"""
    qr = QueryRewriter.__new__(QueryRewriter)
    qr.llm = MagicMock()
    qr.llm.invoke.side_effect = TimeoutError("API Gateway Timeout")

    # 当改写抛出异常时，期望安全降级返回 [query]
    result = qr.rewrite("原始用户问题", strategy="expand")
    assert result == ["原始用户问题"]


def test_compress_old_turns_merges_with_existing_summary():
    """测试多轮对话摘要迭代压缩：二次压缩时应当携带旧摘要，不应直接覆盖丢弃。"""
    mem = ConversationMemory.__new__(ConversationMemory)
    mem.max_turns = 10
    mem.summary = "早期历史摘要：用户探讨了虚拟内存分页机制。"
    mem.history = deque([{"user": f"问{i}", "assistant": f"答{i}"} for i in range(8)], maxlen=10)

    mem.llm = MagicMock()
    mem.llm.invoke.return_value = MagicMock(content="综合摘要：用户先探讨了虚拟内存，后讨论了页面置换算法。")

    mem.compress_old_turns(keep_recent=4)

    # 验证传递给 LLM 的提示词中必须包含旧摘要，而不是直接抛弃
    invoked_prompt_text = str(mem.llm.invoke.call_args)
    assert "虚拟内存分页机制" in invoked_prompt_text, "旧摘要丢失，未传递给 LLM 进行级联摘要"
    assert mem.summary == "综合摘要：用户先探讨了虚拟内存，后讨论了页面置换算法。"


def test_api_ask_and_retrieve_empty_and_top_k_bounds():
    """测试 API 健壮性：空问题应当拦截，top_k 应当被钳制在合理范围内。"""
    class MockAppState:
        def is_ready(self):
            return True
        doc_count = 10
        memory = None
        hybrid_retriever = MagicMock()
        qa_chain = MagicMock()

    app_state = MockAppState()
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []
    app_state.hybrid_retriever = mock_retriever

    app = create_api(app_state, allowed_hosts="testserver")
    client = TestClient(app)

    # 1. /api/ask 发送空白问题
    resp_ask = client.post("/api/ask", json={"question": "   "})
    assert resp_ask.status_code == 200
    assert "问题不能为空" in resp_ask.json()["answer"]
    # 确认没有调用底层的 qa_chain
    assert app_state.qa_chain.answer.call_count == 0

    # 2. /api/retrieve 发送超大 top_k，应该被 clamp 到合理上限（如 50）
    resp_retrieve = client.post("/api/retrieve", json={"query": "测试", "top_k": 99999})
    assert resp_retrieve.status_code == 200
    # 校验传给底层的 search top_k 是否被限制了，而不是直接透传 99999
    called_top_k = mock_retriever.search.call_args[1].get("top_k") or mock_retriever.search.call_args[0][1]
    assert called_top_k <= 50
