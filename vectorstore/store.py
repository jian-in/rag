"""
向量存储管理模块

管理 ChromaDB 向量数据库的创建、持久化、加载和查询。
面试亮点：
  - 向量数据库的持久化与加载（知识不丢失）
  - 增量添加文档的能力（不需要每次重建索引）
  - 清晰的接口设计
"""

import os
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import config
from vectorstore.embeddings import get_embedding_model


class VectorStoreManager:
    """
    向量存储管理器

    封装 ChromaDB 的所有操作，提供高层接口：
      - create_from_documents(): 从文档创建知识库
      - load(): 加载已有的知识库
      - add_documents(): 增量添加文档
      - delete(): 清空知识库
      - get_stats(): 获取知识库统计信息
    """

    def __init__(
        self,
        persist_dir: str = config.CHROMA_PERSIST_DIR,
        collection_name: str = config.CHROMA_COLLECTION_NAME,
        embedding_model: Optional[Embeddings] = None,
    ):
        """
        Args:
            persist_dir: 向量数据库持久化目录
            collection_name: ChromaDB 集合名称
            embedding_model: Embedding 模型，默认使用配置中的模型
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model or get_embedding_model()
        self.vectorstore: Optional[Chroma] = None

    def create_from_documents(self, documents: List[Document]) -> Chroma:
        """
        从文档列表创建向量数据库

        这会对所有文档进行 Embedding 并存储到 ChromaDB。
        数据库会自动持久化到磁盘。

        Args:
            documents: 文档列表（通常是分块后的文本块）

        Returns:
            Chroma: 创建好的向量数据库实例
        """
        print(f"🔨 正在创建向量数据库，共 {len(documents)} 个文档...")

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_dir,
            collection_name=self.collection_name,
        )

        print(
            f"✅ 向量数据库创建完成！\n"
            f"   存储位置: {self.persist_dir}\n"
            f"   集合名称: {self.collection_name}\n"
            f"   文档数量: {len(documents)}"
        )
        return self.vectorstore

    def load(self) -> Chroma:
        """
        加载已有的向量数据库

        Returns:
            Chroma: 已加载的向量数据库实例

        Raises:
            FileNotFoundError: 如果数据库目录不存在
        """
        if not os.path.exists(self.persist_dir):
            raise FileNotFoundError(
                f"向量数据库目录不存在: {self.persist_dir}\n"
                "请先调用 create_from_documents() 创建知识库。"
            )

        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
        )

        count = self.vectorstore._collection.count()
        print(f"✅ 已加载向量数据库，包含 {count} 个向量")
        return self.vectorstore

    def add_documents(self, documents: List[Document]) -> None:
        """
        增量添加文档到已有知识库

        适用于知识库更新场景，不需要重建整个索引。

        Args:
            documents: 要添加的文档列表

        Raises:
            RuntimeError: 如果向量数据库尚未创建或加载
        """
        if self.vectorstore is None:
            raise RuntimeError("向量数据库未初始化，请先调用 create_from_documents() 或 load()")

        self.vectorstore.add_documents(documents)
        count = self.vectorstore._collection.count()
        print(f"✅ 增量添加 {len(documents)} 个文档，当前总量: {count}")

    def delete(self) -> None:
        """清空当前集合中的所有文档"""
        if self.vectorstore is None:
            raise RuntimeError("向量数据库未初始化")

        # 获取所有 ID 并删除
        collection = self.vectorstore._collection
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
        print(f"✅ 已清空向量数据库（删除了 {len(all_ids)} 个向量）")

    def get_vectorstore(self) -> Chroma:
        """
        获取向量数据库实例

        Returns:
            Chroma: 向量数据库实例

        Raises:
            RuntimeError: 如果数据库未初始化
        """
        if self.vectorstore is None:
            raise RuntimeError(
                "向量数据库未初始化，请先调用 create_from_documents() 或 load()"
            )
        return self.vectorstore

    def get_stats(self) -> dict:
        """
        获取向量数据库的统计信息

        Returns:
            dict: 包含文档数量、集合名称等信息
        """
        if self.vectorstore is None:
            return {"状态": "未初始化"}

        count = self.vectorstore._collection.count()
        return {
            "状态": "已就绪",
            "集合名称": self.collection_name,
            "向量总数": count,
            "存储路径": self.persist_dir,
        }
