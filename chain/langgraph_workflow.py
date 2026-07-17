"""
LangGraph 智能工作流模块

用 LangGraph 构建有向图，实现：
  1. 自动分析查询类型，选择最佳检索策略
  2. 答案质量检查，不满意则自动重试
  3. 自适应路由（简单查询走快速路径，复杂查询走深度路径）

面试亮点：
  - 理解 LangGraph 的状态图 (StateGraph) 概念
  - 掌握条件边 (Conditional Edge) 的使用
  - 知道如何设计"自我纠错"的 RAG 工作流
  - 展示对 Agent 架构的理解

LangGraph vs 普通 Chain 的区别：
  Chain: 固定的线性流程 A -> B -> C
  Graph: 有条件分支、循环、并行节点
         A -> (判断) -> B 或 C -> (质量检查不通过?) -> 回到 A
"""

from typing import TypedDict, List, Optional
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import config


# ============================================================
# 状态定义 (State)
# ============================================================
class RAGState(TypedDict):
    """
    LangGraph 的状态对象

    状态在图的节点之间传递，每个节点可以读取和修改状态。
    TypedDict 让类型检查器知道每个字段的类型。

    面试时可以解释：
      - 为什么用 TypedDict 而不是普通 dict（类型安全、IDE 提示）
      - 状态不可变性原则（每次修改产生新状态，不是原地修改）
    """
    original_query: str           # 用户原始问题
    rewritten_query: str          # 改写后的查询
    query_type: str               # 查询类型: "simple" / "complex" / "multi-step"
    retrieved_docs: List[dict]    # 检索到的文档（用 dict 方便序列化）
    answer: str                   # 生成的答案
    answer_quality: str           # 答案质量: "good" / "poor"
    retry_count: int              # 重试次数
    max_retries: int              # 最大重试次数
    strategy_used: str            # 使用的检索策略
    use_hybrid: bool              # 是否使用混合检索
    conversation_context: str     # 对话历史上下文


# ============================================================
# LangGraph 工作流
# ============================================================
class RAGWorkflow:
    """
    RAG 智能工作流

    用 LangGraph 构建的有向图，流程如下：

    [analyze_query] -> (query_type?)
                          |
              simple -> [simple_retrieve] -> [generate] -> [check_quality]
              complex -> [deep_retrieve] -> [generate] -> [check_quality]
                                                              |
                                              good -> [finalize] -> END
                                              poor -> (retry_count < max?)
                                                         |yes     |no
                                                   [rewrite_query] [finalize]
                                                         |
                                                    back to retrieve

    这个设计展示了"自我纠错"(Self-Corrective RAG) 的思想。
    """

    def __init__(self, retriever=None, hybrid_retriever=None,
                 query_rewriter=None, memory=None):
        """
        Args:
            retriever: SmartRetriever 实例（向量检索）
            hybrid_retriever: HybridRetriever 实例（混合检索，可选）
            query_rewriter: QueryRewriter 实例（查询改写，可选）
            memory: ConversationMemory 实例（对话记忆，可选）
        """
        self.retriever = retriever
        self.hybrid_retriever = hybrid_retriever
        self.query_rewriter = query_rewriter
        self.memory = memory

        llm_kwargs = {
            "model": config.LLM_MODEL_NAME,
            "temperature": 0,
            "openai_api_key": config.OPENAI_API_KEY,
        }
        if config.OPENAI_API_BASE:
            llm_kwargs["openai_api_base"] = config.OPENAI_API_BASE

        self.llm = ChatOpenAI(**llm_kwargs)

    # ============================================================
    # 节点函数 (Nodes)
    # ============================================================

    def analyze_query(self, state: RAGState) -> dict:
        """
        节点1：分析查询类型

        用 LLM 判断用户问题的复杂程度，决定后续走哪条路径：
          - simple: 事实性问题，直接检索即可
          - complex: 需要综合分析的问题，用 MMR + 更多文档
          - multi-step: 需要多步推理的问题，拆解子问题
        """
        prompt = ChatPromptTemplate.from_template(
            "分析以下查询的类型，只返回一个词：\n\n"
            "查询：{first_query}\n\n"
            "类型判断标准：\n"
            "- simple：事实性问题（如什么是RAG、Python的创始人是谁）\n"
            "- complex：需要综合分析的问题（如比较RAG和Fine-tuning的优缺点）\n"
            "- multi-step：需要多步推理的问题（如RAG系统的检索器如何影响最终答案的质量）\n\n"
            "只回答 simple、complex 或 multi-step："
        )

        messages = prompt.format_messages(first_query=state["original_query"])
        response = self.llm.invoke(messages)
        query_type = response.content.strip().lower()

        if query_type not in ("simple", "complex", "multi-step"):
            query_type = "simple"

        print(f"[LangGraph] Query type: {query_type}")
        return {
            "query_type": query_type,
            "rewritten_query": state["original_query"],
        }

    def simple_retrieve(self, state: RAGState) -> dict:
        """
        节点2a：简单检索路径

        使用基础向量检索，速度快，适合事实性问题。
        """
        if self.hybrid_retriever and state.get("use_hybrid", False):
            docs = self.hybrid_retriever.search(
                state["rewritten_query"], top_k=config.RETRIEVAL_TOP_K
            )
            strategy = "hybrid"
        elif self.retriever:
            docs = self.retriever.retrieve(state["rewritten_query"], strategy="similarity")
            strategy = "similarity"
        else:
            docs = []
            strategy = "none"

        print(f"[LangGraph] Simple retrieve: {len(docs)} docs via {strategy}")
        return {
            "retrieved_docs": [
                {"content": d.page_content, "metadata": d.metadata}
                for d in docs
            ],
            "strategy_used": strategy,
        }

    def deep_retrieve(self, state: RAGState) -> dict:
        """
        节点2b：深度检索路径

        使用 MMR + 更多文档，适合复杂问题。
        MMR 保证多样性，避免返回重复内容。
        """
        if self.hybrid_retriever and state.get("use_hybrid", False):
            docs = self.hybrid_retriever.search(
                state["rewritten_query"],
                top_k=config.RETRIEVAL_TOP_K + 2,
            )
            strategy = "hybrid+deep"
        elif self.retriever:
            docs = self.retriever.retrieve(
                state["rewritten_query"],
                strategy="mmr",
            )
            strategy = "mmr"
        else:
            docs = []
            strategy = "none"

        print(f"[LangGraph] Deep retrieve: {len(docs)} docs via {strategy}")
        return {
            "retrieved_docs": [
                {"content": d.page_content, "metadata": d.metadata}
                for d in docs
            ],
            "strategy_used": strategy,
        }

    def generate_answer(self, state: RAGState) -> dict:
        """
        节点3：生成答案

        将检索到的文档组装成 Prompt，调用 LLM 生成答案。
        """
        docs = state.get("retrieved_docs", [])
        if not docs:
            return {
                "answer": "根据现有知识库，我无法找到相关信息。建议换一种方式提问，或确认相关文档已上传。",
            }

        # 组装上下文
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc["metadata"].get("file_name", "未知")
            context_parts.append("[文档 " + str(i) + "] 来源: " + source + "\n" + doc["content"])
        context = "\n\n---\n\n".join(context_parts)

        # 对话历史
        conversation_ctx = state.get("conversation_context", "")
        history_hint = ""
        if conversation_ctx:
            history_hint = "\n\n对话历史参考：\n" + conversation_ctx

        query = state["rewritten_query"]
        template = (
            "请根据以下知识库文档来回答用户的问题。\n\n"
            "## 知识库文档\n\n"
            + context +
            history_hint + "\n\n"
            "## 用户问题\n\n"
            + query + "\n\n"
            "## 回答要求\n"
            "- 基于上述文档内容回答，不要添加文档中没有的信息\n"
            "- 在回答中用 [文件名] 格式标注信息来源\n"
            "- 如果文档中没有相关信息，请直接说明"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的知识库问答助手。基于提供的文档内容回答，不要编造。在回答中用 [文件名] 标注来源。"),
            ("human", template),
        ])

        messages = prompt.format_messages()
        response = self.llm.invoke(messages)

        print("[LangGraph] Answer generated (" + str(len(response.content)) + " chars)")
        return {"answer": response.content}

    def check_quality(self, state: RAGState) -> dict:
        """
        节点4：答案质量检查

        用 LLM 评估答案是否充分回答了用户的问题。
        这是 Self-Corrective RAG 的关键步骤。

        面试时可以讲解：
          - 为什么需要质量检查（LLM 可能幻觉、遗漏）
          - 如何设计评估标准（相关性、完整性、忠实度）
          - 重试策略的设计（避免死循环）
        """
        prompt = ChatPromptTemplate.from_template(
            "评估以下答案是否充分回答了用户的问题。\n\n"
            "用户问题：{query}\n\n"
            "生成的答案：{answer}\n\n"
            "评估标准：\n"
            "- 答案是否直接回应了问题\n"
            "- 答案是否包含足够的细节\n"
            "- 答案是否有明显的错误或不一致\n\n"
            "只回答 good 或 poor："
        )

        messages = prompt.format_messages(
            query=state["rewritten_query"],
            answer=state.get("answer", ""),
        )
        response = self.llm.invoke(messages)
        quality = response.content.strip().lower()

        if quality not in ("good", "poor"):
            quality = "good"

        print("[LangGraph] Quality check: " + quality)
        return {"answer_quality": quality}

    def rewrite_for_retry(self, state: RAGState) -> dict:
        """
        节点5：查询改写（重试前）

        当答案质量不佳时，尝试改写查询再重新检索。
        这是一种"换个角度问"的策略。
        """
        prompt = ChatPromptTemplate.from_template(
            "你是一个查询改写专家。原始查询没有得到满意的答案，请从不同角度改写查询。\n\n"
            "原始查询：{query}\n"
            "已尝试的答案（不满意）：{answer}\n\n"
            "请改写为一个更具体、更聚焦的查询，以便获得更好的检索结果。\n"
            "只输出改写后的查询："
        )

        messages = prompt.format_messages(
            query=state["original_query"],
            answer=state.get("answer", "")[:300],
        )
        response = self.llm.invoke(messages)
        rewritten = response.content.strip()

        orig = state["original_query"]
        print("[LangGraph] Rewrite: '" + orig + "' -> '" + rewritten + "'")
        return {
            "rewritten_query": rewritten,
            "retry_count": state.get("retry_count", 0) + 1,
        }

    def finalize(self, state: RAGState) -> dict:
        """节点6：最终输出"""
        retry_count = state.get("retry_count", 0)
        strategy = state.get("strategy_used", "")
        if retry_count > 0:
            msg = "[LangGraph] Finalized after " + str(retry_count) + " retries (strategy: " + strategy + ")"
            print(msg)
        else:
            print("[LangGraph] Finalized (strategy: " + strategy + ")")
        return {"answer": state.get("answer", "")}

    # ============================================================
    # 路由函数 (Conditional Edges)
    # ============================================================

    def route_by_query_type(self, state: RAGState) -> str:
        """条件边：根据查询类型路由到不同的检索节点"""
        query_type = state.get("query_type", "simple")
        if query_type in ("complex", "multi-step"):
            return "deep_retrieve"
        return "simple_retrieve"

    def route_after_quality_check(self, state: RAGState) -> str:
        """条件边：根据质量检查结果决定是否重试"""
        quality = state.get("answer_quality", "good")
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 2)

        if quality == "good":
            return "finalize"
        elif retry_count < max_retries:
            return "rewrite_for_retry"
        else:
            print("[LangGraph] Max retries (" + str(max_retries) + ") reached")
            return "finalize"

    # ============================================================
    # 构建并运行图
    # ============================================================

    def build_graph(self):
        """构建 LangGraph 状态图"""
        from langgraph.graph import StateGraph, START, END

        graph = StateGraph(RAGState)

        # 添加节点
        graph.add_node("analyze_query", self.analyze_query)
        graph.add_node("simple_retrieve", self.simple_retrieve)
        graph.add_node("deep_retrieve", self.deep_retrieve)
        graph.add_node("generate", self.generate_answer)
        graph.add_node("check_quality", self.check_quality)
        graph.add_node("rewrite_for_retry", self.rewrite_for_retry)
        graph.add_node("finalize", self.finalize)

        # 添加边
        graph.add_edge(START, "analyze_query")

        # 条件边：根据查询类型路由
        graph.add_conditional_edges(
            "analyze_query",
            self.route_by_query_type,
            {"simple_retrieve": "simple_retrieve", "deep_retrieve": "deep_retrieve"},
        )

        graph.add_edge("simple_retrieve", "generate")
        graph.add_edge("deep_retrieve", "generate")
        graph.add_edge("generate", "check_quality")

        # 条件边：根据质量检查决定是否重试
        graph.add_conditional_edges(
            "check_quality",
            self.route_after_quality_check,
            {"finalize": "finalize", "rewrite_for_retry": "rewrite_for_retry"},
        )

        # 重试时回到检索
        graph.add_conditional_edges(
            "rewrite_for_retry",
            self.route_by_query_type,
            {"simple_retrieve": "simple_retrieve", "deep_retrieve": "deep_retrieve"},
        )

        graph.add_edge("finalize", END)

        return graph.compile()

    def run(
        self,
        query: str,
        use_hybrid: bool = False,
        conversation_context: str = "",
        max_retries: int = 2,
    ) -> dict:
        """
        运行工作流

        Args:
            query: 用户查询
            use_hybrid: 是否使用混合检索
            conversation_context: 对话历史上下文
            max_retries: 最大重试次数

        Returns:
            dict: 包含 answer、docs、strategy_used 等信息
        """
        app = self.build_graph()

        initial_state: RAGState = {
            "original_query": query,
            "rewritten_query": query,
            "query_type": "simple",
            "retrieved_docs": [],
            "answer": "",
            "answer_quality": "",
            "retry_count": 0,
            "max_retries": max_retries,
            "strategy_used": "",
            "use_hybrid": use_hybrid,
            "conversation_context": conversation_context,
        }

        print("\n[LangGraph] Starting workflow for: '" + query + "'")
        final_state = app.invoke(initial_state)

        # 将 docs 转回 Document 对象方便上层使用
        docs = [
            Document(page_content=d["content"], metadata=d["metadata"])
            for d in final_state.get("retrieved_docs", [])
        ]

        return {
            "answer": final_state.get("answer", ""),
            "docs": docs,
            "query_type": final_state.get("query_type", ""),
            "strategy_used": final_state.get("strategy_used", ""),
            "retry_count": final_state.get("retry_count", 0),
        }
