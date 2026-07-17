"""
RAG 知识库问答系统 - 主入口
提供 Gradio Web UI，支持：
  1. 文档上传与知识库构建
  2. 智能问答（流式输出）
  3. 检索结果可视化（展示相关文档片段）
  4. 知识库状态查看
运行方式：
    python main.py
面试亮点：
  - 完整的端到端演示系统
  - 流式输出提升用户体验
  - 检索结果透明可追溯
"""
import os
import sys
import tempfile
import shutil
from typing import List, Tuple
import gradio as gr
# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from ingestion.loader import DocumentLoader
from ingestion.splitter import TextSplitter
from vectorstore.store import VectorStoreManager
from retrieval.retriever import SmartRetriever
from retrieval.reranker import Reranker
from retrieval.query_rewriter import QueryRewriter
from retrieval.hybrid_retriever import BM25Retriever, HybridRetriever
from retrieval.conversation_memory import ConversationMemory
from chain.qa_chain import QAChain
from chain.langgraph_workflow import RAGWorkflow
from evaluation.evaluator import RAGEvaluator
from multimodal.vision import VisionAnalyzer
# ============================================================
# 全局状态管理
# ============================================================
class AppState:
    """应用全局状态"""

    def __init__(self):
        self.vectorstore_manager: VectorStoreManager = None
        self.qa_chain: QAChain = None
        self.vision_analyzer: VisionAnalyzer = None
        self.loader = DocumentLoader()
        self.splitter = TextSplitter()
        self.doc_count = 0
        self.status = "未初始化"
        # 新增：混合检索和对话记忆
        self.bm25_retriever: BM25Retriever = None
        self.hybrid_retriever: HybridRetriever = None
        self.memory: ConversationMemory = None
        self.workflow: RAGWorkflow = None
        self.use_hybrid: bool = False
        self.use_memory: bool = False
        self.use_langgraph: bool = False
        self._chunks = []  # 保留分块结果，供 BM25 索引构建
        self.evaluator: RAGEvaluator = None  # RAGAS 评估器

    def is_ready(self) -> bool:
        return self.qa_chain is not None

    def get_vision_analyzer(self) -> VisionAnalyzer:
        """延迟初始化视觉分析器"""
        if self.vision_analyzer is None:
            self.vision_analyzer = VisionAnalyzer(
                model_name=config.VISION_MODEL_NAME,
            )
        return self.vision_analyzer


app_state = AppState()


# ============================================================
# 核心功能函数
# ============================================================

def build_knowledge_base(files: List[str]) -> str:
    """
    从上传的文件构建知识库

    Args:
        files: Gradio 上传的文件路径列表

    Returns:
        str: 构建结果的状态信息
    """
    if not files:
        return "⚠️ 请先上传文件"

    try:
        # 1. 加载文档
        all_docs = []
        for file_path in files:
            docs = app_state.loader.load_file(file_path)
            all_docs.extend(docs)

        if not all_docs:
            return "⚠️ 上传的文件中没有提取到有效内容"

        # 2. 分块
        chunks = app_state.splitter.split_documents(all_docs)

        # 3. 构建向量数据库
        app_state.vectorstore_manager = VectorStoreManager()
        app_state.vectorstore_manager.create_from_documents(chunks)

        # 4. 初始化检索器和 QA 链
        vectorstore = app_state.vectorstore_manager.get_vectorstore()

        reranker = None
        if config.ENABLE_RERANKER:
            try:
                reranker = Reranker()
            except Exception as e:
                print(f"⚠️ 重排序器初始化失败，将使用基础检索: {e}")

        retriever = SmartRetriever(
            vectorstore=vectorstore,
            reranker=reranker,
        )

        app_state.qa_chain = QAChain(retriever=retriever)
        app_state.doc_count = len(chunks)
        app_state.status = "已就绪"

        # 5. 构建 BM25 索引（用于混合检索）
        app_state.bm25_retriever = BM25Retriever()
        app_state.bm25_retriever.build_index(chunks)

        # 6. 初始化混合检索器
        app_state.hybrid_retriever = HybridRetriever(
            bm25_retriever=app_state.bm25_retriever,
            vector_retriever=retriever,
        )

        # 7. 初始化对话记忆
        app_state.memory = ConversationMemory()

        # 8. 初始化 LangGraph 工作流
        query_rewriter = QueryRewriter()
        app_state.workflow = RAGWorkflow(
            retriever=retriever,
            hybrid_retriever=app_state.hybrid_retriever,
            query_rewriter=query_rewriter,
            memory=app_state.memory,
        )

        # 保留分块结果
        app_state._chunks = chunks

        # 统计信息
        stats = app_state.splitter.get_stats(all_docs, chunks)
        vs_stats = app_state.vectorstore_manager.get_stats()

        return (
            f"✅ 知识库构建成功！\n\n"
            f"📊 **构建统计**\n"
            f"- 原始文档数: {stats['原始文档数']}\n"
            f"- 分块总数: {stats['分块总数']}\n"
            f"- 平均块大小: {stats['平均块大小']:.0f} 字符\n"
            f"- 向量总数: {vs_stats['向量总数']}\n"
            f"- 存储路径: {vs_stats['存储路径']}\n\n"
            f"🎯 现在可以在「问答」标签页中提问了！"
        )

    except Exception as e:
        return f"❌ 构建失败: {str(e)}"


def clear_knowledge_base() -> str:
    """
    清空知识库

    删除 ChromaDB 持久化目录，重置所有应用状态。
    """
    try:
        # 删除向量数据库文件夹
        chroma_dir = config.CHROMA_PERSIST_DIR
        if os.path.exists(chroma_dir):
            shutil.rmtree(chroma_dir)
            print(f"已删除向量数据库: {chroma_dir}")

        # 重置应用状态
        app_state.vectorstore_manager = None
        app_state.qa_chain = None
        app_state.bm25_retriever = None
        app_state.hybrid_retriever = None
        app_state.memory = None
        app_state.workflow = None
        app_state.doc_count = 0
        app_state.status = "未初始化"
        app_state._chunks = []

        return (
            "🗑️ 知识库已清空！\n\n"
            "向量数据库和应用状态已全部重置。\n"
            "你可以重新上传文档并构建知识库。"
        )

    except Exception as e:
        return f"❌ 清空失败: {str(e)}"


def run_quick_eval(question: str, ground_truth: str) -> str:
    """
    快速评估单条问答

    在界面上输入问题和标准答案，系统自动生成答案并用 RAGAS 打分。

    Args:
        question: 评估问题
        ground_truth: 标准答案（人工编写的参考答案）

    Returns:
        str: 评估结果
    """
    if not app_state.is_ready():
        return "⚠️ 请先构建知识库"

    if not question.strip() or not ground_truth.strip():
        return "⚠️ 请填写问题和标准答案"

    try:
        # 1. 生成答案
        answer, docs = app_state.qa_chain.answer(question)
        contexts = [doc.page_content for doc in docs]

        # 2. 运行评估
        evaluator = RAGEvaluator()
        evaluator.add_sample(
            question=question,
            ground_truth=ground_truth,
            contexts=contexts,
            answer=answer,
        )

        results = evaluator.run_evaluation(
            qa_chain=app_state.qa_chain,
            metrics=["faithfulness", "answer_relevancy", "context_precision"],
        )

        # 3. 格式化结果
        output = f"📊 **评估结果**\n\n"
        output += f"**问题:** {question}\n\n"
        output += f"**标准答案:** {ground_truth[:200]}...\n\n"
        output += f"**系统回答:** {answer[:300]}...\n\n"
        output += "---\n\n"

        for metric_name, score in results.items():
            if isinstance(score, (int, float)):
                emoji = "🟢" if score >= 0.8 else "🟡" if score >= 0.6 else "🔴"
                label = {
                    "faithfulness": "忠实度（幻觉检测）",
                    "answer_relevancy": "答案相关性",
                    "context_precision": "检索精度",
                }.get(metric_name, metric_name)
                output += f"{emoji} **{label}:** {score:.2%}\n\n"

        output += "---\n"
        output += "📝 **指标说明:**\n"
        output += "- 忠实度: 答案是否基于检索文档（越高幻觉越少）\n"
        output += "- 答案相关性: 答案是否切题\n"
        output += "- 检索精度: 检索到的文档是否相关且排序正确\n"

        return output

    except ImportError:
        return "⚠️ 请先安装 ragas: `pip install ragas datasets`"
    except Exception as e:
        return f"❌ 评估失败: {str(e)}"


def answer_question(
    question: str,
    history: List[Tuple[str, str]],
    use_hybrid: bool = False,
    use_memory: bool = False,
    use_langgraph: bool = False,
) -> Tuple[str, str]:
    """
    回答用户问题（增强版）

    支持三种增强模式：
      1. 混合检索（BM25 + 向量）
      2. 对话记忆（多轮对话上下文）
      3. LangGraph 智能工作流（自适应检索 + 质量检查）

    Args:
        question: 用户问题
        history: 对话历史
        use_hybrid: 是否使用混合检索
        use_memory: 是否使用对话记忆
        use_langgraph: 是否使用 LangGraph 工作流

    Returns:
        Tuple[str, str]: (答案, 检索到的文档信息)
    """
    if not app_state.is_ready():
        return "⚠️ 请先在「知识库构建」标签页中上传文档并构建知识库。", ""

    if not question.strip():
        return "", ""

    try:
        # ---- 对话记忆：改写查询 ----
        actual_query = question
        if use_memory and app_state.memory:
            actual_query = app_state.memory.rewrite_query_with_history(question)

        # ---- 模式选择 ----
        if use_langgraph and app_state.workflow:
            # LangGraph 智能工作流模式
            conversation_ctx = ""
            if use_memory and app_state.memory:
                conversation_ctx = app_state.memory.get_context()

            result = app_state.workflow.run(
                query=actual_query,
                use_hybrid=use_hybrid,
                conversation_context=conversation_ctx,
            )
            answer = result["answer"]
            docs = result["docs"]

            # 在来源信息中显示额外的工作流信息
            source_info = _format_sources(docs)
            wf_info = (
                "\n\n---\n"
                "🔧 **工作流信息**\n"
                f"- 查询类型: {result['query_type']}\n"
                f"- 检索策略: {result['strategy_used']}\n"
                f"- 重试次数: {result['retry_count']}\n"
            )
            source_info += wf_info

        elif use_hybrid and app_state.hybrid_retriever:
            # 混合检索模式
            docs = app_state.hybrid_retriever.search(
                actual_query, top_k=config.RETRIEVAL_TOP_K
            )
            if docs:
                context = "\n\n---\n\n".join(
                    f"[文档 {i}] 来源: {d.metadata.get('file_name', '未知')}\n{d.page_content}"
                    for i, d in enumerate(docs, 1)
                )
            else:
                context = ""
            answer, _ = app_state.qa_chain.answer_with_documents(actual_query, docs)
            source_info = _format_sources(docs)
        else:
            # 标准模式（原有逻辑）
            answer, docs = app_state.qa_chain.answer(actual_query)
            source_info = _format_sources(docs)

        # ---- 对话记忆：记录本轮 ----
        if use_memory and app_state.memory:
            app_state.memory.add_turn(question, answer)
            # 每 10 轮压缩一次旧对话
            if app_state.memory.get_turn_count() >= 10:
                app_state.memory.compress_old_turns(keep_recent=5)

        # 如果查询被改写了，在答案中提示
        if actual_query != question:
            answer = f"🔄 *查询改写: \"{actual_query}\"*\n\n{answer}"

        return answer, source_info

    except Exception as e:
        return f"❌ 回答失败: {str(e)}", ""


def stream_answer_question(
    question: str,
    use_memory: bool = False,
):
    """
    流式回答（打字机效果）

    标准模式下逐 token 输出，提供实时打字效果。
    对于混合检索和 LangGraph 模式，仍使用非流式。

    Yields:
        Tuple[str, str]: (当前累积答案, 来源信息)
    """
    if not app_state.is_ready():
        yield "⚠️ 请先在「知识库构建」标签页中上传文档并构建知识库。", ""
        return

    if not question.strip():
        yield "", ""
        return

    try:
        # 对话记忆：改写查询
        actual_query = question
        if use_memory and app_state.memory:
            actual_query = app_state.memory.rewrite_query_with_history(question)

        # 使用 stream_answer 流式输出
        source_info = "⏳ 正在检索相关文档..."
        full_answer = ""

        for partial, docs in app_state.qa_chain.stream_answer(actual_query):
            full_answer = partial
            if docs is not None:
                source_info = _format_sources(docs)
            yield full_answer, source_info

        # 对话记忆：记录本轮
        if use_memory and app_state.memory:
            app_state.memory.add_turn(question, full_answer)
            if app_state.memory.get_turn_count() >= 10:
                app_state.memory.compress_old_turns(keep_recent=5)

        # 如果查询被改写了，在答案中提示
        if actual_query != question:
            final = f"🔄 *查询改写: \"{actual_query}\"*\n\n{full_answer}"
            yield final, source_info

    except Exception as e:
        yield f"❌ 回答失败: {str(e)}", ""


def answer_with_image(
    question: str,
    image_path: str,
    history: list,
) -> Tuple[str, str]:
    """
    带图片的问答

    两种模式：
      1. 知识库已就绪：图片描述 + 知识库检索 → 综合回答
      2. 知识库未就绪：纯视觉模型回答

    Args:
        question: 用户问题
        image_path: 图片文件路径
        history: 对话历史

    Returns:
        Tuple[str, str]: (答案, 来源信息)
    """
    try:
        vision = app_state.get_vision_analyzer()

        if app_state.is_ready():
            # 图文结合 RAG 模式
            enhanced_query, image_desc = vision.image_rag_query(
                image_path, question
            )
            answer, docs = app_state.qa_chain.answer(
                enhanced_query, rewrite_strategy=None
            )
            source_info = (
                f"🖼️ **图片理解**\n```\n{image_desc[:300]}...\n```\n\n"
                + _format_sources(docs)
            )
            final_answer = f"📷 **图片分析 + 知识库回答**\n\n{answer}"
        else:
            # 纯视觉模式
            answer = vision.analyze_image(image_path, question)
            source_info = f"🖼️ **图片理解结果**\n```\n{answer[:300]}\n```"
            final_answer = f"📷 **图片分析结果**（知识库未构建，仅基于视觉模型）\n\n{answer}"

        return final_answer, source_info

    except Exception as e:
        return f"❌ 图像分析失败: {str(e)}", ""


def _format_sources(docs) -> str:
    """格式化检索到的文档信息"""
    if not docs:
        return "未检索到相关文档"

    parts = ["📚 **检索到的相关文档片段**\n"]
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("file_name", "未知")
        page = doc.metadata.get("page", "")
        chunk_idx = doc.metadata.get("chunk_index", "")
        rerank_score = doc.metadata.get("rerank_score", "")
        bm25_score = doc.metadata.get("bm25_score", "")
        rrf_score = doc.metadata.get("rrf_score", "")

        meta_info = f"来源: {source}"
        if page:
            meta_info += f" | 第{page}页"
        if chunk_idx:
            meta_info += f" | 块#{chunk_idx}"
        if rerank_score:
            meta_info += f" | 重排序分: {rerank_score:.4f}"
        if bm25_score:
            meta_info += f" | BM25: {bm25_score:.2f}"
        if rrf_score:
            meta_info += f" | RRF: {rrf_score:.4f}"

        # 截取前 500 字符作为预览
        preview = doc.page_content[:500]
        if len(doc.page_content) > 500:
            preview += "..."

        parts.append(f"**[{i}] {meta_info}**\n```\n{preview}\n```\n")

    return "\n".join(parts)


def get_status() -> str:
    """获取系统状态"""
    if not app_state.is_ready():
        return "🔴 系统未就绪，请先构建知识库"

    vs_stats = {}
    if app_state.vectorstore_manager:
        vs_stats = app_state.vectorstore_manager.get_stats()

    # 对话记忆状态
    memory_info = "未初始化"
    if app_state.memory:
        turns = app_state.memory.get_turn_count()
        memory_info = f"已就绪（{turns} 轮对话）"

    # 混合检索状态
    hybrid_info = "未初始化"
    if app_state.hybrid_retriever:
        hybrid_info = "已就绪 (BM25 + 向量)"

    # LangGraph 状态
    langgraph_info = "未初始化"
    if app_state.workflow:
        langgraph_info = "已就绪"

    return (
        f"🟢 系统就绪\n\n"
        f"**知识库状态**\n"
        f"- 向量总数: {vs_stats.get('向量总数', 'N/A')}\n"
        f"- 集合名称: {vs_stats.get('集合名称', 'N/A')}\n"
        f"- 存储路径: {vs_stats.get('存储路径', 'N/A')}\n\n"
        f"**增强功能**\n"
        f"- 混合检索: {hybrid_info}\n"
        f"- 对话记忆: {memory_info}\n"
        f"- LangGraph 工作流: {langgraph_info}\n\n"
        f"**模型配置**\n"
        f"- LLM: {config.LLM_MODEL_NAME}\n"
        f"- Embedding: {config.EMBEDDING_MODEL_NAME}\n"
        f"- 分块大小: {config.CHUNK_SIZE}\n"
        f"- 分块重叠: {config.CHUNK_OVERLAP}\n"
        f"- 检索 Top-K: {config.RETRIEVAL_TOP_K}\n"
        f"- 重排序: {'开启' if config.ENABLE_RERANKER else '关闭'}\n"
    )


# ============================================================
# 自定义主题与样式
# ============================================================

CUSTOM_CSS = """
/* ===== 全局暗色样式 ===== */
.gradio-container {
    font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif !important;
    max-width: 1280px !important;
    margin: 0 auto !important;
    background: #0f1117 !important;
    min-height: 100vh;
    padding: 16px !important;
    color: #e2e8f0 !important;
}

/* ===== 顶部标题区域 ===== */
.header-section {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #a855f7 100%);
    border-radius: 20px;
    padding: 32px 40px;
    margin-bottom: 20px;
    color: white;
    box-shadow: 0 8px 32px rgba(79, 70, 229, 0.25);
    position: relative;
    overflow: hidden;
}
.header-section::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 300px;
    height: 300px;
    background: rgba(255,255,255,0.06);
    border-radius: 50%;
}
.header-section::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: 10%;
    width: 200px;
    height: 200px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.header-section h1 {
    margin: 0 0 8px 0;
    font-size: 1.8em;
    font-weight: 800;
    letter-spacing: -0.5px;
    position: relative;
    z-index: 1;
}
.header-section p {
    margin: 0;
    opacity: 0.85;
    font-size: 0.95em;
    font-weight: 400;
    position: relative;
    z-index: 1;
}

/* ===== 标签页 (胶囊风格 暗色) ===== */
.tab-nav {
    display: flex !important;
    gap: 4px !important;
    background: #1a1c2e !important;
    border-radius: 16px !important;
    padding: 6px !important;
    margin-bottom: 20px !important;
    border: 1px solid #2d2f45 !important;
}
.tab-nav button {
    border-radius: 12px !important;
    border: none !important;
    padding: 12px 24px !important;
    font-size: 0.95em !important;
    font-weight: 600 !important;
    color: #8b8fa3 !important;
    background: transparent !important;
    transition: all 0.3s ease !important;
}
.tab-nav button:hover {
    background: #252740 !important;
    color: #a78bfa !important;
}
.tab-nav button.selected {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    box-shadow: 0 4px 16px rgba(79, 70, 229, 0.3) !important;
}

/* ===== 聊天气泡 ===== */
.bot {
    background: #1e2035 !important;
    border: 1px solid #2d2f45 !important;
    border-radius: 16px 16px 16px 4px !important;
    color: #e2e8f0 !important;
    padding: 16px 20px !important;
}
.user {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border-radius: 16px 16px 4px 16px !important;
    box-shadow: 0 4px 16px rgba(79, 70, 229, 0.25) !important;
    padding: 16px 20px !important;
}

/* 聊天框容器 */
[class*="chatbot"] {
    border-radius: 20px !important;
    border: 1px solid #2d2f45 !important;
    background: #13141f !important;
}

/* ===== 输入框 ===== */
textarea, input[type="text"] {
    background: #1a1c2e !important;
    border: 2px solid #2d2f45 !important;
    border-radius: 14px !important;
    color: #e2e8f0 !important;
    transition: all 0.3s ease !important;
    padding: 12px 18px !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15) !important;
    outline: none !important;
    background: #1e2035 !important;
}

/* ===== 按钮 ===== */
.primary {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
    color: white !important;
    padding: 12px 28px !important;
    box-shadow: 0 4px 16px rgba(79, 70, 229, 0.3) !important;
    transition: all 0.3s ease !important;
}
.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(79, 70, 229, 0.4) !important;
}

/* ===== 面板和区块 ===== */
[class*="block"] {
    border-radius: 16px !important;
    border-color: #2d2f45 !important;
}

/* Markdown 区域 */
.markdown {
    border-radius: 16px !important;
    color: #cbd5e1 !important;
}

/* 代码块 */
code {
    background: #1a1c2e !important;
    color: #a78bfa !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', 'Consolas', monospace !important;
}
pre {
    background: #1a1c2e !important;
    border: 1px solid #2d2f45 !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', 'Consolas', monospace !important;
}

/* ===== Checkbox ===== */
input[type="checkbox"] {
    accent-color: #6366f1 !important;
}

/* ===== 文件上传区域 ===== */
[class*="upload"] {
    border: 2px dashed #3d3f5c !important;
    border-radius: 16px !important;
    background: #13141f !important;
    transition: all 0.3s ease !important;
}
[class*="upload"]:hover {
    border-color: #6366f1 !important;
    background: #1a1c2e !important;
}

/* ===== 表格 ===== */
table {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid #2d2f45 !important;
    background: #1a1c2e !important;
}
th {
    background: #252740 !important;
    font-weight: 600 !important;
    color: #a78bfa !important;
}
td {
    border-top: 1px solid #2d2f45 !important;
    color: #cbd5e1 !important;
}

/* ===== 下拉框 / 选择器 ===== */
select, [role="listbox"] {
    background: #1a1c2e !important;
    border: 1px solid #2d2f45 !important;
    color: #e2e8f0 !important;
    border-radius: 12px !important;
}

/* ===== 滚动条 ===== */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #13141f;
}
::-webkit-scrollbar-thumb {
    background: #3d3f5c;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #4f46e5;
}

/* ===== 动画 ===== */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
.header-section {
    animation: fadeIn 0.6s ease-out;
}
"""

CUSTOM_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "Segoe UI", "Microsoft YaHei", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "Consolas", "monospace"],
)


# ============================================================
# Gradio 界面构建
# ============================================================

def create_app() -> gr.Blocks:
    """创建 Gradio 应用"""

    with gr.Blocks(
        title="RAG 知识库问答系统",
        theme=CUSTOM_THEME,
        css=CUSTOM_CSS,
    ) as app:

        # ---- 顶部 Header ----
        gr.HTML("""
        <div class="header-section">
            <h1>🤖 RAG 知识库问答系统</h1>
            <p>基于 LangChain + LangGraph + ChromaDB · 混合检索 · 对话记忆 · 多模态图片理解</p>
        </div>
        """)

        with gr.Tabs():
            # ============================================================
            # Tab 1: 智能问答
            # ============================================================
            with gr.Tab("💬 智能问答", id="chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="对话",
                            height=520,
                        )
                        # 功能开关行
                        with gr.Row():
                            hybrid_toggle = gr.Checkbox(
                                label="混合检索 (BM25+向量)",
                                value=False,
                                scale=1,
                            )
                            memory_toggle = gr.Checkbox(
                                label="多轮对话记忆",
                                value=False,
                                scale=1,
                            )
                            langgraph_toggle = gr.Checkbox(
                                label="LangGraph 工作流",
                                value=False,
                                scale=1,
                            )
                        with gr.Row():
                            image_input = gr.Image(
                                label="📷",
                                type="filepath",
                                height=56,
                                show_label=False,
                                scale=1,
                            )
                            question_input = gr.Textbox(
                                label="输入你的问题",
                                placeholder="输入问题后按回车或点击发送...  支持附带图片一起提问",
                                show_label=False,
                                scale=4,
                                lines=1,
                                max_lines=3,
                            )
                            send_btn = gr.Button(
                                "发送",
                                variant="primary",
                                scale=1,
                            )
                        clear_btn = gr.Button(
                            "🗑️ 清空对话",
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("#### 📚 检索来源")
                        source_output = gr.Markdown(
                            value="*提问后将在此处显示检索到的相关文档片段及来源信息*",
                        )

                def respond(question, image, chat_history, hybrid, memory, langgraph):
                    chat_history = chat_history or []

                    if image:
                        user_content = [
                            {"type": "image", "path": image},
                            {"type": "text", "text": question or "请描述这张图片"},
                        ]
                        chat_history.append({"role": "user", "content": user_content})
                        chat_history.append({"role": "assistant", "content": ""})
                        yield "", None, chat_history, "⏳ 正在分析图片..."

                        answer, sources = answer_with_image(
                            question or "请描述这张图片", image, chat_history
                        )
                        chat_history[-1] = {"role": "assistant", "content": answer}
                        yield "", None, chat_history, sources

                    elif hybrid or langgraph:
                        # 混合检索或 LangGraph 模式：非流式
                        user_content = question
                        chat_history.append({"role": "user", "content": user_content})
                        chat_history.append({"role": "assistant", "content": ""})
                        yield "", None, chat_history, "⏳ 正在检索..."

                        answer, sources = answer_question(
                            question, chat_history,
                            use_hybrid=hybrid,
                            use_memory=memory,
                            use_langgraph=langgraph,
                        )
                        chat_history[-1] = {"role": "assistant", "content": answer}
                        yield "", None, chat_history, sources

                    else:
                        # 标准模式：流式输出（打字机效果）
                        user_content = question
                        chat_history.append({"role": "user", "content": user_content})
                        chat_history.append({"role": "assistant", "content": ""})

                        for partial_answer, sources in stream_answer_question(
                            question, use_memory=memory
                        ):
                            chat_history[-1] = {"role": "assistant", "content": partial_answer}
                            yield "", None, chat_history, sources

                send_btn.click(
                    fn=respond,
                    inputs=[question_input, image_input, chatbot,
                            hybrid_toggle, memory_toggle, langgraph_toggle],
                    outputs=[question_input, image_input, chatbot, source_output],
                )
                question_input.submit(
                    fn=respond,
                    inputs=[question_input, image_input, chatbot,
                            hybrid_toggle, memory_toggle, langgraph_toggle],
                    outputs=[question_input, image_input, chatbot, source_output],
                )

                def clear_all():
                    if app_state.memory:
                        app_state.memory.clear()
                    return [], "*提问后将在此处显示检索到的相关文档片段*"

                clear_btn.click(
                    fn=clear_all,
                    outputs=[chatbot, source_output],
                )

            # ============================================================
            # Tab 2: 知识库构建
            # ============================================================
            with gr.Tab("📁 知识库管理", id="build"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📄 上传文档")
                        gr.Markdown(
                            "支持的格式：**PDF** · **TXT** · **Markdown** · **DOCX**\n\n"
                            "上传后点击「构建知识库」按钮，系统将自动解析、分块、向量化。"
                        )
                        file_upload = gr.File(
                            label="拖拽文件到此处",
                            file_count="multiple",
                            file_types=[".pdf", ".txt", ".md", ".docx"],
                            height=180,
                        )
                        build_btn = gr.Button(
                            "🚀 构建知识库",
                            variant="primary",
                        )
                        clear_kb_btn = gr.Button(
                            "🗑️ 清空知识库",
                            variant="stop",
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("#### 📊 构建结果")
                        status_output = gr.Markdown(
                            value="*尚未构建知识库，请上传文档并点击构建按钮*",
                        )

                build_btn.click(
                    fn=build_knowledge_base,
                    inputs=[file_upload],
                    outputs=[status_output],
                )
                clear_kb_btn.click(
                    fn=clear_knowledge_base,
                    outputs=[status_output],
                )

            # ============================================================
            # Tab 3: 系统状态
            # ============================================================
            with gr.Tab("⚙️ 系统设置", id="settings"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 🟢 系统状态")
                        status_display = gr.Markdown()
                        refresh_btn = gr.Button("🔄 刷新状态")
                        refresh_btn.click(fn=get_status, outputs=[status_display])

                    with gr.Column(scale=1):
                        gr.Markdown(
                            "#### 📐 当前配置\n\n"
                            "| 参数 | 值 |\n"
                            "|------|----|\n"
                            f"| LLM 模型 | `{config.LLM_MODEL_NAME}` |\n"
                            f"| Embedding 模型 | `{config.EMBEDDING_MODEL_NAME}` |\n"
                            f"| 视觉模型 | `{config.VISION_MODEL_NAME}` |\n"
                            f"| 分块大小 | `{config.CHUNK_SIZE}` 字符 |\n"
                            f"| 分块重叠 | `{config.CHUNK_OVERLAP}` 字符 |\n"
                            f"| 检索 Top-K | `{config.RETRIEVAL_TOP_K}` |\n"
                            f"| MMR Lambda | `{config.MMR_LAMBDA}` |\n"
                            f"| 重排序 | `{'开启' if config.ENABLE_RERANKER else '关闭'}` |\n"
                            f"| 查询改写 | `expand`（默认） |\n"
                        )

                gr.Markdown(
                    "---\n"
                    "#### 🏗️ 系统架构\n\n"
                    "```\n"
                    "用户提问 → [对话记忆/查询改写] → [查询分析] → (simple/complex?)\n"
                    "                                      ↓                ↓\n"
                    "                              [BM25 + 向量检索]   [MMR 深度检索]\n"
                    "                                      ↓                ↓\n"
                    "                              [RRF 融合排序]  ←  ←  ←\n"
                    "                                      ↓\n"
                    "                              [质量检查] → (不满意?) → [改写重试]\n"
                    "                                      ↓\n"
                    "                              [Prompt组装] → LLM → 答案+引用\n"
                    "                                      ↕\n"
                    "                           ChromaDB + BM25 知识库\n"
                    "```"
                )

            # ============================================================
            # Tab 4: 系统评估
            # ============================================================
            with gr.Tab("📊 评估测试", id="eval"):
                gr.Markdown(
                    "#### RAGAS 自动评估\n\n"
                    "输入一个问题和标准答案，系统会自动生成回答并用 RAGAS 框架打分。\n"
                    "用于科学评估 RAG 系统的忠实度、相关性和检索精度。"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        eval_question = gr.Textbox(
                            label="评估问题",
                            placeholder="例如：什么是 RAG 技术？",
                            lines=2,
                        )
                        eval_truth = gr.Textbox(
                            label="标准答案（参考答案）",
                            placeholder="人工编写的参考答案，用于和系统回答对比",
                            lines=4,
                        )
                        eval_btn = gr.Button(
                            "🧪 运行评估",
                            variant="primary",
                        )
                    with gr.Column(scale=1):
                        eval_result = gr.Markdown(
                            value="*填写问题和标准答案后点击运行评估*",
                        )

                eval_btn.click(
                    fn=run_quick_eval,
                    inputs=[eval_question, eval_truth],
                    outputs=[eval_result],
                )

        # 页面加载时显示状态
        app.load(fn=get_status, outputs=[status_display])

    return app


# ============================================================
# 启动应用
# ============================================================
if __name__ == "__main__":
    from api import create_api

    # 创建 Gradio 应用
    app = create_app()

    # 创建 FastAPI 并挂载到同一个服务上
    fastapi_app = create_api(app_state)
    app = gr.mount_gradio_app(fastapi_app, app, path="/")

    # 启动（API 文档访问: http://localhost:7860/docs）
    import uvicorn
    print("\n🌐 Web UI:  http://localhost:7860")
    print("📡 API 文档: http://localhost:7860/docs")
    print("📡 API 测试: curl -X POST http://localhost:7860/api/ask -H 'Content-Type: application/json' -d '{\"question\": \"什么是RAG\"}'\n")

    uvicorn.run(app, host="0.0.0.0", port=7860)
