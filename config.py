"""
RAG 知识库系统 - 统一配置管理

所有可调参数集中在此处管理，方便调优和实验对比。
面试中可以展示你对"配置驱动开发"的理解。
"""

import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# ============================================================
# LLM 配置
# ============================================================
# 支持 OpenAI 官方 API 或任何兼容接口（如本地部署的 vLLM、Ollama 等）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", None)  # 可选，用于自定义 API 端点
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))  # RAG 场景建议低温以保证准确性

# ============================================================
# Embedding 配置（支持独立于 LLM 的 API 端点）
# ============================================================
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "") or SILICONFLOW_API_KEY or OPENAI_API_KEY  # ????????? Embedding Key
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "https://api.siliconflow.cn/v1")  # ?????Embedding API ???
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))  # ???????????????

# ============================================================
# 多模态视觉模型配置
# ============================================================
VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "qwen-vl-max")

# ============================================================
# 文档分块配置
# ============================================================
# chunk_size: 每个文本块的最大字符数
#   - 太大 → 检索精度低（噪声多）
#   - 太小 → 上下文丢失（语义不完整）
#   - 推荐范围: 500-1500，需要根据实际文档调优
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))

# chunk_overlap: 相邻块之间的重叠字符数
#   - 防止关键信息被切断在两个块之间
#   - 推荐为 chunk_size 的 10%-20%
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ============================================================
# 检索配置
# ============================================================
# 基础检索返回的文档数量
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))

# MMR (Maximal Marginal Relevance) 的多样性参数
#   - 0.0 → 最大多样性（返回差异大的文档）
#   - 1.0 → 纯相似度检索（可能返回重复内容）
#   - 0.5-0.7 通常是较好的平衡点
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.6"))

# ???????
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "false").lower() == "true"
_DEFAULT_RERANKER_MODELS = {
    "cohere": "rerank-multilingual-v3.0",
    "cross-encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "siliconflow": "BAAI/bge-reranker-v2-m3",
}
_RERANKER_MODEL_ENV = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_PROVIDER = os.getenv("RERANKER_PROVIDER", "")
if not RERANKER_PROVIDER:
    if _RERANKER_MODEL_ENV in _DEFAULT_RERANKER_MODELS:
        RERANKER_PROVIDER = _RERANKER_MODEL_ENV
        RERANKER_MODEL_NAME = _DEFAULT_RERANKER_MODELS[_RERANKER_MODEL_ENV]
    else:
        RERANKER_PROVIDER = "siliconflow"
        RERANKER_MODEL_NAME = _RERANKER_MODEL_ENV
else:
    RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", _RERANKER_MODEL_ENV)
RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", "3"))  # ??????????
RERANKER_API_BASE = os.getenv("RERANKER_API_BASE", "https://api.siliconflow.cn/v1")
RERANKER_API_KEY = os.getenv("RERANKER_API_KEY", "") or SILICONFLOW_API_KEY
RERANKER_MAX_CHUNKS_PER_DOC = int(os.getenv("RERANKER_MAX_CHUNKS_PER_DOC", "1024"))
RERANKER_OVERLAP_TOKENS = int(os.getenv("RERANKER_OVERLAP_TOKENS", "60"))

# ============================================================
# ???????
# ============================================================
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "knowledge_base")

# ============================================================
# Cohere API（用于 Reranking）
# ============================================================
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

# ============================================================
# 评估配置
# ============================================================
EVAL_DATASET_PATH = os.getenv("EVAL_DATASET_PATH", "./data/eval_dataset.json")
