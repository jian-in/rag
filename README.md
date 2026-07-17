# RAG 知识库问答系统

基于 LangChain + ChromaDB 构建的检索增强生成（RAG）系统，支持多格式文档上传、智能检索、重排序优化和自动化评估。

## 系统架构

```
用户提问
   │
   ▼
┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│  文档加载     │────▶│   文本分块      │────▶│  向量化存储   │
│ PDF/TXT/MD   │     │ Recursive Split │     │  ChromaDB    │
│ /DOCX        │     │ + Overlap       │     │  Embedding   │
└──────────────┘     └────────────────┘     └──────┬───────┘
                                                   │
                                                   ▼
┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│  LLM 生成    │◀────│  Prompt 组装    │◀────│  智能检索     │
│  答案 + 引用  │     │  系统提示 + 上下 │     │  MMR + 重排序 │
└──────────────┘     │  文 + 用户问题   │     └──────────────┘
                     └────────────────┘
```

## 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 编排框架 | LangChain | RAG 链路编排 |
| 向量数据库 | ChromaDB | 轻量级向量存储 |
| LLM | OpenAI GPT-4o-mini | 答案生成（可替换） |
| Embedding | text-embedding-3-small | 文本向量化 |
| 重排序 | Cohere / Cross-Encoder | 检索精度优化 |
| Web UI | Gradio | 交互式界面 |
| 评估 | RAGAS | 系统效果量化评估 |

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env，填入你的 API Key
# OPENAI_API_KEY=sk-your-key-here
```

### 3. 启动应用

```bash
python main.py
```

浏览器打开 `http://localhost:7860` 即可使用。

### 4. 使用流程

1. 在「知识库构建」页面上传文档（支持 PDF/TXT/MD/DOCX）
2. 点击「构建知识库」，等待处理完成
3. 切换到「问答」页面，输入问题并获取答案
4. 右侧面板会展示检索到的相关文档片段

## 项目结构

```
rag-knowledge-base/
├── main.py                  # 应用入口（Gradio Web UI）
├── config.py                # 统一配置管理
├── ingestion/
│   ├── loader.py            # 多格式文档加载
│   └── splitter.py          # 文本分块策略
├── vectorstore/
│   ├── embeddings.py        # Embedding 模型封装
│   └── store.py             # ChromaDB 向量存储管理
├── retrieval/
│   ├── retriever.py         # 智能检索器（相似度 + MMR）
│   └── reranker.py          # 重排序模块
├── chain/
│   ├── qa_chain.py          # QA 问答链
│   └── prompts.py           # Prompt 模板
├── evaluation/
│   └── evaluator.py         # RAGAS 评估模块
├── data/
│   └── sample_docs/         # 示例文档目录
└── docs/
    └── architecture.md      # 详细架构文档
```

## 核心功能

### 文档处理

- **多格式支持**：PDF、TXT、Markdown、DOCX
- **智能分块**：RecursiveCharacterTextSplitter，按段落 → 句子 → 字符的优先级切分
- **元数据保留**：记录来源文件名、页码、块序号，支持答案溯源

### 检索策略

- **相似度检索**：基于向量余弦相似度的基础检索
- **MMR 检索**：Maximal Marginal Relevance，兼顾相关性和结果多样性
- **重排序**：Cohere Rerank API 或 Cross-Encoder 模型进行二次精排

### 评估体系

基于 RAGAS 框架的量化评估，核心指标包括：

- **Faithfulness（忠实度）**：答案是否完全基于检索上下文，衡量幻觉程度
- **Answer Relevancy（答案相关性）**：答案是否切题回答了问题
- **Context Precision（上下文精确度）**：检索文档的质量和相关性排序

```bash
# 运行评估
python evaluation/evaluator.py
```

## 配置说明

所有参数集中在 `config.py` 中管理，支持通过 `.env` 文件或环境变量覆盖。关键参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| CHUNK_SIZE | 1000 | 分块大小（字符数），太大检索精度低，太小上下文丢失 |
| CHUNK_OVERLAP | 200 | 分块重叠，防止关键信息被切断 |
| RETRIEVAL_TOP_K | 5 | 检索返回的文档数量 |
| MMR_LAMBDA | 0.6 | MMR 多样性参数（0=最大多样性，1=纯相似度） |
| ENABLE_RERANKER | false | 是否启用重排序 |

## 扩展方向

以下是可以进一步优化的方向，适合在面试中讨论：

1. **对话记忆**：集成 LangChain 的 ConversationBufferMemory，支持多轮对话上下文
2. **Hybrid Search**：结合 BM25 关键词检索 + 向量检索，提升召回率
3. **查询改写**：用 LLM 对用户查询进行改写/扩展，提高检索效果
4. **多知识库路由**：支持多个知识库，根据问题自动路由到最相关的库
5. **Agent 集成**：将 RAG 作为一个 Tool，集成到更大的 Agent 系统中

## License

MIT
