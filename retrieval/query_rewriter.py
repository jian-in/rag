"""
查询改写模块

在检索之前，用 LLM 对用户问题进行智能改写。
面试亮点：
  - 理解 Query Rewriting 对 RAG 效果的重要性
  - 掌握多种改写策略（扩展、分解、HyDE）
  - 知道什么时候该改写、什么时候不需要

改写策略：
  1. 扩展改写：补充同义词、相关概念，让检索覆盖更广
  2. 子问题分解：把复杂问题拆成多个简单子问题，分别检索
  3. HyDE（Hypothetical Document Embeddings）：让 LLM 先生成一个"假设性答案"，
     用这个答案去检索，比直接用问题检索效果更好
"""

from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import config


# ============================================================
# 改写 Prompt 模板
# ============================================================

EXPAND_PROMPT = """你是一个查询改写专家。请将用户的原始问题改写为一个更适合在知识库中检索的版本。

改写要求：
1. 保留原始问题的核心意图
2. 补充可能相关的关键词和同义词
3. 如果问题太简短，适当扩展使其更具体
4. 直接输出改写后的查询，不要解释

原始问题：{query}

改写后的查询："""

DECOMPOSE_PROMPT = """你是一个查询分析专家。请将用户的复杂问题分解为多个简单的子问题。

要求：
1. 每个子问题应该独立可回答
2. 子问题数量控制在 2-4 个
3. 每行一个子问题，不要编号

如果问题很简单不需要分解，就直接输出原问题。

原始问题：{query}

子问题列表："""

HYDE_PROMPT = """请针对以下问题，写一段简短的、可能出现在知识库中的回答段落。
不需要完全准确，只需要涵盖可能相关的概念和关键词，用于辅助检索。

问题：{query}

假设性回答段落："""


class QueryRewriter:
    """
    查询改写器

    支持三种改写策略：
      - expand: 扩展改写，补充关键词
      - decompose: 子问题分解
      - hyde: 生成假设性文档用于检索

    面试时可以解释每种策略的适用场景和 trade-off。
    """

    def __init__(self, model_name: str = config.LLM_MODEL_NAME):
        """
        Args:
            model_name: 用于改写的 LLM 模型
        """
        llm_kwargs = {
            "model": model_name,
            "temperature": 0,
            "openai_api_key": config.OPENAI_API_KEY,
        }
        if config.OPENAI_API_BASE:
            llm_kwargs["openai_api_base"] = config.OPENAI_API_BASE

        self.llm = ChatOpenAI(**llm_kwargs)

    def expand(self, query: str) -> str:
        """
        扩展改写

        适用场景：用户问题太短或太模糊时
        例如："RAG" → "检索增强生成 RAG 技术原理 工作流程 向量检索"

        Args:
            query: 原始查询

        Returns:
            str: 改写后的查询
        """
        prompt = ChatPromptTemplate.from_template(EXPAND_PROMPT)
        messages = prompt.format_messages(query=query)
        response = self.llm.invoke(messages)
        return response.content.strip()

    def decompose(self, query: str) -> List[str]:
        """
        子问题分解

        适用场景：用户问了包含多个方面的复杂问题
        例如："RAG和Fine-tuning有什么区别，各自适合什么场景？"
          → "RAG和Fine-tuning的核心区别是什么？"
          → "RAG适合哪些应用场景？"
          → "Fine-tuning适合哪些应用场景？"

        Args:
            query: 原始查询

        Returns:
            List[str]: 子问题列表
        """
        prompt = ChatPromptTemplate.from_template(DECOMPOSE_PROMPT)
        messages = prompt.format_messages(query=query)
        response = self.llm.invoke(messages)

        # 按行分割，过滤空行
        sub_queries = [
            line.strip()
            for line in response.content.strip().split("\n")
            if line.strip()
        ]

        # 至少返回一个问题
        if not sub_queries:
            sub_queries = [query]

        return sub_queries

    def hyde(self, query: str) -> str:
        """
        HyDE (Hypothetical Document Embeddings)

        核心思想：与其用"问题"去检索，不如先让 LLM 生成一个"假想答案"，
        用这个假想答案的向量去检索，因为答案的语义和文档更接近。

        适用场景：用户问的是概念性问题，但知识库中是描述性文本
        例如：问题"什么是RAG？" → 假想答案"RAG是一种将检索和生成结合的技术..."
              用假想答案去检索，比用"什么是RAG？"检索效果更好

        Args:
            query: 原始查询

        Returns:
            str: 假设性文档
        """
        prompt = ChatPromptTemplate.from_template(HYDE_PROMPT)
        messages = prompt.format_messages(query=query)
        response = self.llm.invoke(messages)
        return response.content.strip()

    def rewrite(
        self, query: str, strategy: str = "expand"
    ) -> List[str]:
        """
        统一改写接口

        Args:
            query: 原始查询
            strategy: 改写策略 - "expand", "decompose", "hyde"

        Returns:
            List[str]: 改写后的查询列表
                      （expand/hyde 返回1个，decompose 可能返回多个）
        """
        if strategy == "expand":
            rewritten = self.expand(query)
            print(f"🔄 查询改写(expand): '{query}' → '{rewritten}'")
            return [rewritten]

        elif strategy == "decompose":
            sub_queries = self.decompose(query)
            print(f"🔄 查询改写(decompose): '{query}' → {sub_queries}")
            return sub_queries

        elif strategy == "hyde":
            hypothetical = self.hyde(query)
            print(f"🔄 查询改写(hyde): 生成了假设性文档")
            return [hypothetical]

        else:
            raise ValueError(f"不支持的改写策略: {strategy}")
