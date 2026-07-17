"""
多轮对话记忆模块

让 RAG 系统支持上下文记忆，实现多轮对话。

面试亮点：
  - 理解对话记忆在 RAG 中的作用
  - 掌握滑动窗口 + 摘要压缩的记忆策略
  - 知道如何用历史对话改写当前查询

多轮对话的核心挑战：
  用户说"刚才那个再详细说说"，系统需要知道"刚才那个"是什么。
  解决方案：用 LLM 结合对话历史，把模糊的追问改写成完整的查询。
"""

from typing import List, Dict, Optional
from collections import deque

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import config


class ConversationMemory:
    """
    对话记忆管理器

    使用滑动窗口策略：保留最近 N 轮对话的完整内容。
    当超出窗口时，将旧对话压缩为摘要。

    设计考量：
      - 完整保留所有历史会占用大量 token，影响 LLM 性能和成本
      - 只保留最近几轮可能丢失重要上下文
      - 滑动窗口 + 摘要压缩是平衡方案
    """

    def __init__(
        self,
        max_turns: int = 10,
        model_name: str = config.LLM_MODEL_NAME,
    ):
        """
        Args:
            max_turns: 最多保留的对话轮数
            model_name: 用于摘要压缩的 LLM 模型
        """
        self.max_turns = max_turns
        self.history: deque = deque(maxlen=max_turns)
        self.summary: str = ""

        llm_kwargs = {
            "model": model_name,
            "temperature": 0,
            "openai_api_key": config.OPENAI_API_KEY,
        }
        if config.OPENAI_API_BASE:
            llm_kwargs["openai_api_base"] = config.OPENAI_API_BASE

        self.llm = ChatOpenAI(**llm_kwargs)

    def add_turn(self, user_message: str, assistant_message: str):
        """
        添加一轮对话

        Args:
            user_message: 用户消息
            assistant_message: 助手回复
        """
        self.history.append({
            "user": user_message,
            "assistant": assistant_message,
        })

    def get_context(self) -> str:
        """
        获取对话上下文（用于注入 Prompt）

        如果有历史摘要，先放摘要，再放最近的对话记录。

        Returns:
            str: 格式化的对话上下文
        """
        parts = []

        if self.summary:
            parts.append(f"[早期对话摘要]\n{self.summary}")

        for i, turn in enumerate(self.history, 1):
            parts.append(f"第{i}轮 - 用户: {turn['user']}\n第{i}轮 - 助手: {turn['assistant']}")

        return "\n\n".join(parts)

    def rewrite_query_with_history(self, current_query: str) -> str:
        """
        结合对话历史改写当前查询

        这是多轮对话 RAG 的核心功能。
        用户的追问往往是模糊的（"再说说""为什么"），
        需要结合历史改写成完整的、适合检索的查询。

        例如：
          历史: 用户问了"什么是RAG"，助手回答了
          当前: "它的优缺点呢？"
          改写后: "RAG（检索增强生成）技术的优点和缺点是什么？"

        Args:
            current_query: 用户当前的查询

        Returns:
            str: 改写后的完整查询
        """
        if not self.history:
            return current_query

        prompt = ChatPromptTemplate.from_template(
            """你是一个查询改写专家。根据对话历史，将用户当前的问题改写为一个完整的、独立的查询。

对话历史：
{history}

用户当前问题：{query}

改写要求：
1. 如果当前问题引用了历史内容（如"刚才说的""它""这个"），请替换为具体指代
2. 如果当前问题已经是完整独立的，保持不变
3. 直接输出改写后的查询，不要解释

改写后的查询："""
        )

        history_text = self.get_context()
        messages = prompt.format_messages(history=history_text, query=current_query)
        response = self.llm.invoke(messages)

        rewritten = response.content.strip()
        if rewritten != current_query:
            print(f"Rewrite: '{current_query}' -> '{rewritten}'")

        return rewritten

    def compress_old_turns(self, keep_recent: int = 5):
        """
        压缩旧对话为摘要

        当对话轮数超过窗口时，将较早的对话压缩成一段摘要，
        释放 token 空间同时保留关键信息。

        Args:
            keep_recent: 保留最近多少轮的完整对话
        """
        if len(self.history) <= keep_recent:
            return

        old_turns = list(self.history)[:-keep_recent]
        recent_turns = list(self.history)[-keep_recent:]

        old_text = "\n".join(
            f"用户: {t['user']}\n助手: {t['assistant']}" for t in old_turns
        )

        prompt = ChatPromptTemplate.from_template(
            "请将以下对话内容压缩为一段简短的摘要（100字以内），保留关键信息：\n\n{conversation}\n\n摘要："
        )

        messages = prompt.format_messages(conversation=old_text)
        response = self.llm.invoke(messages)

        self.summary = response.content.strip()
        self.history = deque(recent_turns, maxlen=self.max_turns)
        print(f"Compressed {len(old_turns)} old turns into summary")

    def clear(self):
        """清空对话历史"""
        self.history.clear()
        self.summary = ""

    def get_turn_count(self) -> int:
        """获取当前对话轮数"""
        return len(self.history)
