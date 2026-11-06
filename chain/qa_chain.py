"""
QA 链模块

RAG 系统的核心组件，将检索、Prompt 组装和 LLM 调用串联起来。

面试亮点：
  - 理解 RAG Chain 的完整数据流
  - Prompt 的上下文注入策略
  - 错误处理和兜底逻辑
  - 可扩展性设计（方便切换 LLM、添加后处理等）
"""

import hashlib
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import config
from chain.prompts import SYSTEM_PROMPT, QA_PROMPT_TEMPLATE, NO_CONTEXT_PROMPT
from retrieval.retriever import SmartRetriever
from retrieval.query_rewriter import QueryRewriter


class QAChain:
    """
    RAG 问答链

    完整数据流：
      用户问题 → [查询改写] → 检索相关文档 → 组装 Prompt → LLM 生成答案 → 返回结果

    升级后的流程（面试亮点）：
      用户问题 → Query Rewriter → 多个改写查询 → 分别检索 → 合并去重 → 生成答案
    """

    def __init__(
        self,
        retriever: SmartRetriever,
        model_name: str = config.LLM_MODEL_NAME,
        temperature: float = config.LLM_TEMPERATURE,
    ):
        """
        Args:
            retriever: 智能检索器实例
            model_name: LLM 模型名称
            temperature: LLM 温度（RAG 场景建议低温，减少随机性）
        """
        self.retriever = retriever

        # 初始化 LLM
        llm_kwargs = {
            "model": model_name,
            "temperature": temperature,
            "openai_api_key": config.OPENAI_API_KEY,
        }
        if config.OPENAI_API_BASE:
            llm_kwargs["openai_api_base"] = config.OPENAI_API_BASE

        self.llm = ChatOpenAI(**llm_kwargs)

        # 初始化查询改写器
        self.rewriter = QueryRewriter(model_name=model_name)

        # 构建 Prompt 模板
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", QA_PROMPT_TEMPLATE),
        ])

        self.no_context_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", NO_CONTEXT_PROMPT),
        ])

        print(f"✅ QA 链已初始化 (模型: {model_name}, 含查询改写)")

    def answer(
        self,
        question: str,
        strategy: str = "mmr",
        use_reranker: bool = config.ENABLE_RERANKER,
        rewrite_strategy: Optional[str] = "expand",
    ) -> Tuple[str, List[Document]]:
        """
        回答用户问题（核心方法）

        升级后的完整流程：
          1. 查询改写（可选）：用 LLM 改写/扩展用户问题
          2. 多路检索：对每个改写查询分别检索
          3. 合并去重：合并所有检索结果
          4. 组装 Prompt 并调用 LLM
          5. 返回答案和引用的文档

        Args:
            question: 用户问题
            strategy: 检索策略（"similarity" 或 "mmr"）
            use_reranker: 是否使用重排序
            rewrite_strategy: 查询改写策略
                - None: 不改写，直接用原始问题检索
                - "expand": 扩展改写
                - "decompose": 子问题分解
                - "hyde": 假设性文档嵌入

        Returns:
            Tuple[str, List[Document]]: (答案文本, 引用的文档列表)
        """
        # 1. 查询改写 + 多路检索
        if rewrite_strategy:
            queries = self.rewriter.rewrite(question, strategy=rewrite_strategy)
            docs = self._multi_query_retrieve(
                queries, strategy=strategy, use_reranker=use_reranker
            )
        else:
            docs = self.retriever.retrieve(
                query=question,
                strategy=strategy,
                use_reranker=use_reranker,
            )

        # 2. 检查是否有相关上下文
        return self.answer_with_documents(question, docs)

    def answer_with_documents(
        self,
        question: str,
        docs: List[Document],
    ) -> Tuple[str, List[Document]]:
        """Use a precomputed document list to generate the final answer."""
        if not docs:
            answer = self._answer_no_context(question)
            return answer, []

        context = self.retriever.format_context(docs)
        messages = self.qa_prompt.format_messages(
            context=context,
            question=question,
        )

        response = self.llm.invoke(messages)
        answer = response.content

        return answer, docs

    def stream_answer_with_documents(
        self,
        question: str,
        docs: List[Document],
    ):
        """使用预先检索好的文档流式生成答案（用于混合检索模式的打字机效果）。

        Yields:
            Tuple[str, Optional[List[Document]]]:
              - 过程中: (当前累积答案, None)
              - 最后一次: (完整答案, 引用文档列表)
        """
        if not docs:
            messages = self.no_context_prompt.format_messages(question=question)
        else:
            context = self.retriever.format_context(docs)
            messages = self.qa_prompt.format_messages(
                context=context,
                question=question,
            )

        full_answer = ""
        for chunk in self.llm.stream(messages):
            full_answer += chunk.content
            yield full_answer, None

        yield full_answer, docs

    def _multi_query_retrieve(
        self,
        queries: List[str],
        strategy: str = "mmr",
        use_reranker: bool = config.ENABLE_RERANKER,
        rrf_k: int = 60,
    ) -> List[Document]:
        """
        多查询检索：对每个查询分别检索，并使用倒数排名融合（RRF）去重排序

        这是工业界常用的技巧——不同角度的查询能召回不同方面的文档，
        通过 RRF 融合能够使多个查询交叉召回的高相关文档排在前面。

        Args:
            queries: 查询列表
            strategy: 检索策略
            use_reranker: 是否使用重排序
            rrf_k: RRF 平滑系数（默认 60）

        Returns:
            List[Document]: 合并融合排序后的文档列表
        """
        def _doc_key(doc: Document) -> str:
            md = doc.metadata or {}
            source = str(md.get("source", ""))
            page = str(md.get("page", ""))
            chunk_index = str(md.get("chunk_index", ""))
            content_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()[:16]
            return f"{source}::{page}::{chunk_index}::{content_hash}"

        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for query in queries:
            docs = self.retriever.retrieve(
                query=query,
                strategy=strategy,
                use_reranker=use_reranker,
            )
            for rank, doc in enumerate(docs, start=1):
                key = _doc_key(doc)
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
                doc_map.setdefault(key, doc)

        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        results = [doc_map[k] for k in sorted_keys]

        print(f"📚 多路检索合并(RRF): {len(queries)} 个查询 → {len(results)} 个不重复文档")
        return results

    def _answer_no_context(self, question: str) -> str:
        """
        当检索不到相关文档时的兜底回答

        Args:
            question: 用户问题

        Returns:
            str: 友好的提示信息
        """
        messages = self.no_context_prompt.format_messages(question=question)
        response = self.llm.invoke(messages)
        return response.content

    def stream_answer(
        self,
        question: str,
        strategy: str = "mmr",
        use_reranker: bool = config.ENABLE_RERANKER,
    ):
        """
        流式回答（用于 Web UI 实时展示）

        与 answer() 类似，但逐 token 输出，提供更好的用户体验。

        Args:
            question: 用户问题
            strategy: 检索策略
            use_reranker: 是否使用重排序

        Yields:
            Tuple[str, Optional[List[Document]]]:
              - 第一个 yield: (当前累积的答案文本, None)
              - 最后一个 yield: (完整答案, 引用文档列表)
        """
        # 检索
        docs = self.retriever.retrieve(
            query=question,
            strategy=strategy,
            use_reranker=use_reranker,
        )

        if not docs:
            messages = self.no_context_prompt.format_messages(question=question)
        else:
            context = self.retriever.format_context(docs)
            messages = self.qa_prompt.format_messages(
                context=context,
                question=question,
            )

        # 流式输出
        full_answer = ""
        for chunk in self.llm.stream(messages):
            full_answer += chunk.content
            yield full_answer, None

        # 最后 yield 完整结果
        yield full_answer, docs
