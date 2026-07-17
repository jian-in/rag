"""
文本分块模块

将长文档切分为适合 Embedding 和检索的文本块。
面试亮点：
  - 理解 chunk_size 和 chunk_overlap 对检索效果的影响
  - 使用 RecursiveCharacterTextSplitter 实现语义感知的分块
  - 保留元数据的继承，确保切分后仍可溯源
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


class TextSplitter:
    """
    文本分块器

    使用 LangChain 的 RecursiveCharacterTextSplitter，按以下优先级切分：
    1. 先在段落边界切分（\n\n）
    2. 段落太大则在句子边界切分（\n）
    3. 句子还太大则在空格处切分
    4. 最后才按字符切分

    这种策略能最大程度保留语义完整性。
    """

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
    ):
        """
        Args:
            chunk_size: 每个文本块的最大字符数
            chunk_overlap: 相邻块之间的重叠字符数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
            is_separator_regex=False,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        对文档列表进行分块

        分块后的每个片段继承原文档的元数据，并额外添加：
          - chunk_index: 在原文档中的块序号
          - chunk_total: 原文档总共被切成了多少块

        Args:
            documents: 原始文档列表

        Returns:
            List[Document]: 分块后的文档列表
        """
        all_chunks = []

        for doc in documents:
            chunks = self._splitter.split_text(doc.page_content)

            for i, chunk_text in enumerate(chunks):
                # 继承原文档的元数据，并添加分块信息
                chunk_metadata = {
                    **doc.metadata,
                    "chunk_index": i + 1,
                    "chunk_total": len(chunks),
                    "chunk_size": len(chunk_text),
                }
                all_chunks.append(
                    Document(page_content=chunk_text, metadata=chunk_metadata)
                )

        print(
            f"✅ 分块完成: {len(documents)} 个文档 → {len(all_chunks)} 个文本块 "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return all_chunks

    def get_stats(self, documents: List[Document], chunks: List[Document]) -> dict:
        """
        获取分块统计信息（面试时展示数据分析能力）

        Args:
            documents: 原始文档
            chunks: 分块结果

        Returns:
            dict: 统计信息
        """
        chunk_sizes = [len(c.page_content) for c in chunks]
        return {
            "原始文档数": len(documents),
            "分块总数": len(chunks),
            "平均块大小": sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
            "最大块大小": max(chunk_sizes) if chunk_sizes else 0,
            "最小块大小": min(chunk_sizes) if chunk_sizes else 0,
            "设定 chunk_size": self.chunk_size,
            "设定 chunk_overlap": self.chunk_overlap,
        }
