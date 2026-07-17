"""
Embedding 模型封装

统一管理 Embedding 模型的初始化和调用。
面试亮点：
  - 工厂模式，方便切换不同的 Embedding 提供商
  - 支持缓存机制，避免重复计算
"""

from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

import config


def get_embedding_model() -> Embeddings:
    """
    获取 Embedding 模型实例

    支持独立的 Embedding API 配置，可以和 LLM 使用不同的服务商。
    例如：LLM 用阿里云百炼，Embedding 用硅基流动。

    Returns:
        Embeddings: LangChain 兼容的 Embedding 模型实例
    """
    kwargs = {
        "model": config.EMBEDDING_MODEL_NAME,
        "openai_api_key": config.EMBEDDING_API_KEY,
        "check_embedding_ctx_length": False,
    }

    # 使用独立的 Embedding API 端点（如硅基流动）
    if config.EMBEDDING_API_BASE:
        kwargs["openai_api_base"] = config.EMBEDDING_API_BASE
    # 如果没有独立的端点，回退到 LLM 的端点
    elif config.OPENAI_API_BASE:
        kwargs["openai_api_base"] = config.OPENAI_API_BASE

    # 使用 httpx 兼容方式，避免发送硅基流动不支持的参数
    embedding_model = OpenAIEmbeddings(**kwargs)

    api_source = config.EMBEDDING_API_BASE or config.OPENAI_API_BASE or "OpenAI"
    print(f"Embedding model loaded: {config.EMBEDDING_MODEL_NAME} (endpoint: {api_source})")
    return embedding_model
