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
import traceback

# Windows GBK 控制台编码修复：强制 stdout/stderr 用 UTF-8
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
import shutil
from typing import List, Tuple
import gradio as gr
# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from ui_styles import CUSTOM_CSS, CUSTOM_THEME
from ingestion.loader import DocumentLoader
from ingestion.splitter import TextSplitter
from vectorstore.store import VectorStoreManager
from retrieval.retriever import SmartRetriever
from retrieval.reranker import Reranker
from retrieval.query_rewriter import QueryRewriter
from retrieval.hybrid_retriever import BM25Retriever, HybridRetriever
from langchain_core.documents import Document
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
        self.evaluator: RAGEvaluator = None  # LLM 裁判评估器（RAGAS 风格指标）

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

# 上传文件写入共享 Chroma 集合时使用的标记：可按 course_id / doc_id 删除，
# 与 SecondBrain 同步（source_kind=filesystem）和课策文档（origin=kece）区分开。
UPLOAD_COURSE_ID = "__upload__"
UPLOAD_ORIGIN = "upload"


def _initialize_runtime(manager: VectorStoreManager) -> int:
    """从向量库完整快照重建检索器、BM25、QA 链和工作流，返回文本块数。

    构建、加载、上传后都走这里，保证向量检索与 BM25 覆盖同一批文档
    （此前构建只用本次上传的块建 BM25，库里已有的内容只在向量侧可见）。
    """
    vectorstore = manager.get_vectorstore()
    result = manager.get_snapshot()
    documents = result.get("documents", []) or []
    metadatas = result.get("metadatas", []) or []
    chunks = [
        Document(page_content=content, metadata=(metadatas[i] if i < len(metadatas) else {}) or {})
        for i, content in enumerate(documents)
        if content
    ]

    reranker = None
    if config.ENABLE_RERANKER:
        try:
            reranker = Reranker()
        except Exception as e:
            print(f"⚠️ 重排序器初始化失败，将使用基础检索: {e}")

    retriever = SmartRetriever(vectorstore=vectorstore, reranker=reranker)
    app_state.vectorstore_manager = manager
    app_state.qa_chain = QAChain(retriever=retriever)
    app_state.bm25_retriever = BM25Retriever()
    app_state.bm25_retriever.build_index(chunks)
    app_state.hybrid_retriever = HybridRetriever(
        bm25_retriever=app_state.bm25_retriever,
        vector_retriever=retriever,
    )
    app_state.memory = ConversationMemory()
    app_state.workflow = RAGWorkflow(
        retriever=retriever,
        hybrid_retriever=app_state.hybrid_retriever,
        query_rewriter=QueryRewriter(),
        memory=app_state.memory,
    )
    app_state._chunks = chunks
    app_state.doc_count = len(chunks)
    app_state.status = "已就绪"
    return len(chunks)


def ingest_uploaded_files(uploads: List[Tuple[str, str]]) -> dict:
    """把上传的文件追加进知识库；同名文件替换旧版本。

    Args:
        uploads: [(落盘路径, 原始文件名)]。原始文件名用于引用展示和去重，
            避免引用里出现临时文件名。

    Returns:
        dict: {"files": int, "raw_docs": int, "added_chunks": int,
               "removed_chunks": int, "doc_count": int}
    """
    raw_docs: List[Document] = []
    names: List[str] = []
    for path, display_name in uploads:
        name = os.path.basename(display_name) or os.path.basename(path)
        loaded = app_state.loader.load_file(path)
        if not loaded:
            continue
        names.append(name)
        for doc in loaded:
            doc.metadata = {
                **doc.metadata,
                "source": f"upload::{name}",
                "file_name": name,
                "course_id": UPLOAD_COURSE_ID,
                "doc_id": name,
                "origin": UPLOAD_ORIGIN,
                "source_kind": UPLOAD_ORIGIN,
            }
            raw_docs.append(doc)

    if not raw_docs:
        return {
            "files": 0,
            "raw_docs": 0,
            "added_chunks": 0,
            "removed_chunks": 0,
            "doc_count": app_state.doc_count,
        }

    chunks = app_state.splitter.split_documents(raw_docs)
    manager = app_state.vectorstore_manager or VectorStoreManager()
    if manager.vectorstore is None:
        manager.open()

    removed = 0
    for name in dict.fromkeys(names):
        removed += manager.delete_by_metadata(
            {"$and": [{"course_id": UPLOAD_COURSE_ID}, {"doc_id": name}]}
        )
    manager.add_documents(chunks)
    doc_count = _initialize_runtime(manager)
    return {
        "files": len(set(names)),
        "raw_docs": len(raw_docs),
        "added_chunks": len(chunks),
        "removed_chunks": removed,
        "doc_count": doc_count,
    }


def build_knowledge_base(files: List[str]) -> str:
    """
    把上传的文件写入知识库（追加；同名文件替换旧版本）

    Args:
        files: Gradio 上传的文件路径列表

    Returns:
        str: 构建结果的状态信息
    """
    if not files:
        return "⚠️ 请先上传文件"

    try:
        result = ingest_uploaded_files(
            [(file_path, os.path.basename(file_path)) for file_path in files]
        )
        if not result["added_chunks"]:
            return "⚠️ 上传的文件中没有提取到有效内容"

        vs_stats = app_state.vectorstore_manager.get_stats()
        replaced = (
            f"- 替换同名旧版本: {result['removed_chunks']} 个块\n"
            if result["removed_chunks"]
            else ""
        )
        return (
            f"✅ 知识库更新成功！\n\n"
            f"📊 **构建统计**\n"
            f"- 上传文件数: {result['files']}\n"
            f"- 新增文本块: {result['added_chunks']}\n"
            f"{replaced}"
            f"- 知识库总块数: {result['doc_count']}\n"
            f"- 向量总数: {vs_stats['向量总数']}\n"
            f"- 存储路径: {vs_stats['存储路径']}\n\n"
            f"🎯 现在可以在「问答」标签页中提问了！"
        )

    except Exception as e:
        traceback.print_exc()
        return f"❌ 构建失败: {str(e)}"


def auto_load_knowledge_base():
    """启动时自动加载已有的 ChromaDB 知识库"""
    if not os.path.exists(config.CHROMA_PERSIST_DIR):
        print("[auto-load] 未检测到已有知识库，跳过")
        return False

    try:
        print("[auto-load] 正在加载已有知识库...")
        manager = VectorStoreManager()
        manager.load()
        if manager.count() == 0:
            print("[auto-load] 知识库为空")
            return False
        count = _initialize_runtime(manager)
        print(f"[auto-load] 知识库加载完成，共 {count} 个文本块")
        return True

    except Exception as e:
        print(f"[auto-load] 加载失败: {e}")
        return False


def clear_knowledge_base() -> str:
    """
    清空知识库

    删除集合中的全部向量并重置应用状态。不删除持久化目录：RAG MCP 服务
    等其他进程可能正打开着同一个 Chroma 库，Windows 上 rmtree 会删到一半
    失败、留下损坏的索引。
    """
    try:
        manager = app_state.vectorstore_manager
        if manager is None and os.path.exists(config.CHROMA_PERSIST_DIR):
            manager = VectorStoreManager()
            manager.load()
        if manager is not None:
            manager.delete()

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
            "向量数据和应用状态已全部重置。\n"
            "SecondBrain 笔记会在下次 sync_knowledge_base 时重新入库；"
            "也可以重新上传文档。"
        )

    except Exception as e:
        traceback.print_exc()
        return f"❌ 清空失败: {str(e)}"


def kb_add_documents(items: List[dict], course_id: str, origin: str = "kece") -> dict:
    """
    将外部资料（如课策上传的课程文档）增量写入知识库，并同步重建 BM25 索引。

    每个 chunk 会打上 course_id / doc_id / origin 标记，便于后续按课程或文档删除。

    Args:
        items: 文档列表，每项 {"doc_id": str, "name": str, "text": str}
        course_id: 课程标识（用于后续按课程删除）
        origin: 来源标记，默认 "kece"

    Returns:
        dict: {"added_chunks": int, "doc_count": int}

    Raises:
        RuntimeError: 知识库尚未初始化（需先构建或加载）
    """
    if app_state.vectorstore_manager is None or app_state.bm25_retriever is None:
        raise RuntimeError("知识库尚未初始化，请先构建或加载知识库")

    course_id = str(course_id)
    raw_docs: List[Document] = []
    for i, it in enumerate(items or []):
        text = (it.get("text") or "").strip()
        if not text:
            continue
        doc_id = str(it.get("doc_id") or it.get("id") or f"auto-{i}")
        name = str(it.get("name") or f"课程文档-{doc_id}")
        raw_docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": f"kece::{course_id}::{doc_id}",
                    "file_name": name,
                    "course_id": course_id,
                    "doc_id": doc_id,
                    "origin": str(origin),
                },
            )
        )

    if not raw_docs:
        return {"added_chunks": 0, "doc_count": app_state.doc_count}

    chunks = app_state.splitter.split_documents(raw_docs)
    app_state.vectorstore_manager.add_documents(chunks)

    # 同步 _chunks 并重建 BM25（否则新内容只在向量侧可见）
    app_state._chunks.extend(chunks)
    app_state.bm25_retriever.build_index(app_state._chunks)
    app_state.doc_count = len(app_state._chunks)

    print(f"✅ [kb-add] course_id={course_id} 写入 {len(chunks)} 个块，当前总量 {app_state.doc_count}")
    return {"added_chunks": len(chunks), "doc_count": app_state.doc_count}


def kb_delete_documents(course_id: str = None, doc_id: str = None) -> dict:
    """
    按课程/文档从知识库删除文档，并同步重建 BM25 索引。

    Args:
        course_id: 按课程删除（删除该课程的全部文档）
        doc_id: 按单个文档删除；与 course_id 同时提供时两者都匹配

    Returns:
        dict: {"deleted": int, "doc_count": int}
    """
    if app_state.vectorstore_manager is None:
        raise RuntimeError("知识库尚未初始化")
    if not course_id and not doc_id:
        raise ValueError("需要提供 course_id 或 doc_id 至少一个")

    conds = []
    if course_id:
        conds.append({"course_id": str(course_id)})
    if doc_id:
        conds.append({"doc_id": str(doc_id)})
    where = conds[0] if len(conds) == 1 else {"$and": conds}

    deleted = app_state.vectorstore_manager.delete_by_metadata(where)

    # 同步 _chunks
    def _match(md: dict) -> bool:
        if course_id and str(md.get("course_id")) != str(course_id):
            return False
        if doc_id and str(md.get("doc_id")) != str(doc_id):
            return False
        return True

    app_state._chunks = [c for c in app_state._chunks if not _match(c.metadata)]
    # 无条件重建：build_index 现在能处理空语料（清空索引），
    # 旧守卫 `and app_state._chunks` 会让「删到空库」时索引保持陈旧，
    # 已删除的文档仍会被 BM25 召回。
    if app_state.bm25_retriever is not None:
        app_state.bm25_retriever.build_index(app_state._chunks)
    app_state.doc_count = len(app_state._chunks)

    print(f"✅ [kb-delete] where={where} 删除 {deleted} 个，当前总量 {app_state.doc_count}")
    return {"deleted": deleted, "doc_count": app_state.doc_count}


def run_quick_eval(question: str, ground_truth: str) -> str:
    """
    快速评估单条问答

    在界面上输入问题和标准答案，系统自动生成答案，并用 LLM 裁判按 RAGAS 风格指标打分。

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
        if evaluator.errors:
            output += "⚠️ **部分指标未能判定:**\n"
            output += "".join(f"- {error}\n" for error in evaluator.errors)
            output += "\n"

        output += "---\n"
        output += "📝 **指标说明:**\n"
        output += "- 忠实度: 答案是否基于检索文档（越高幻觉越少）\n"
        output += "- 答案相关性: 答案是否切题\n"
        output += "- 检索精度: 检索到的文档是否相关且排序正确\n"

        return output

    except Exception as e:
        traceback.print_exc()
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
            # 窗口满时 add_turn 会自动把旧对话压缩进摘要
            app_state.memory.add_turn(question, answer)

        # 如果查询被改写了，在答案中提示
        if actual_query != question:
            answer = f"🔄 *查询改写: \"{actual_query}\"*\n\n{answer}"

        return answer, source_info

    except Exception as e:
        traceback.print_exc()
        return f"❌ 回答失败: {str(e)}", ""


def stream_answer_question(
    question: str,
    use_memory: bool = False,
    use_hybrid: bool = False,
):
    """
    流式回答（打字机效果）

    标准模式与混合检索模式均逐 token 输出，提供实时打字效果。
    LangGraph 多节点工作流仍使用非流式。

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

        source_info = "⏳ 正在检索相关文档..."
        full_answer = ""

        if use_hybrid and app_state.hybrid_retriever:
            # 混合检索：先检索再流式生成
            yield "", "⏳ 正在混合检索（BM25 + 向量召回 → RRF 融合）..."
            docs = app_state.hybrid_retriever.search(actual_query, top_k=config.RETRIEVAL_TOP_K)
            source_info = _format_sources(docs)
            yield "⏳ 检索完成，正在生成答案...", source_info
            stream_iter = app_state.qa_chain.stream_answer_with_documents(actual_query, docs)
        else:
            # 标准模式：检索 + 流式生成
            stream_iter = app_state.qa_chain.stream_answer(actual_query)

        for partial, docs in stream_iter:
            full_answer = partial
            if docs is not None:
                source_info = _format_sources(docs)
            yield full_answer, source_info

        # 对话记忆：记录本轮
        if use_memory and app_state.memory:
            app_state.memory.add_turn(question, full_answer)

        # 如果查询被改写了，在答案中提示
        if actual_query != question:
            final = f"🔄 *查询改写: \"{actual_query}\"*\n\n{full_answer}"
            yield final, source_info

    except Exception as e:
        traceback.print_exc()
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
        traceback.print_exc()
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
# CUSTOM_CSS 与 CUSTOM_THEME 已抽离到 ui_styles.py，此处直接从模块导入使用。


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

                    elif langgraph:
                        # LangGraph 工作流：多节点（分析→检索→生成→质检），非流式
                        user_content = question
                        chat_history.append({"role": "user", "content": user_content})
                        chat_history.append({"role": "assistant", "content": ""})
                        yield "", None, chat_history, "⏳ LangGraph 工作流运行中（分析查询 → 检索 → 生成 → 质量检查）..."

                        answer, sources = answer_question(
                            question, chat_history,
                            use_hybrid=hybrid,
                            use_memory=memory,
                            use_langgraph=True,
                        )
                        chat_history[-1] = {"role": "assistant", "content": answer}
                        yield "", None, chat_history, sources

                    elif hybrid:
                        # 混合检索模式：先检索再流式生成（打字机效果）
                        user_content = question
                        chat_history.append({"role": "user", "content": user_content})
                        chat_history.append({"role": "assistant", "content": ""})

                        for partial_answer, sources in stream_answer_question(
                            question, use_memory=memory, use_hybrid=True
                        ):
                            chat_history[-1] = {"role": "assistant", "content": partial_answer}
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
                    "#### 自动评估（RAGAS 风格指标）\n\n"
                    "输入一个问题和标准答案，系统会自动生成回答，并由 LLM 裁判按忠实度、相关性、检索精度打分。\n"
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
    fastapi_app = create_api(
        app_state,
        kb_add_documents,
        kb_delete_documents,
        build_documents_fn=ingest_uploaded_files,
        clear_fn=clear_knowledge_base,
    )
    app = gr.mount_gradio_app(fastapi_app, app, path="/")

    # 启动时自动加载已有知识库
    auto_load_knowledge_base()

    # 启动（默认只监听本机；开放局域网见 config.py 的 RAG_WEB_HOST 说明）
    import uvicorn
    import http_guard

    host, port = config.RAG_WEB_HOST, config.RAG_WEB_PORT
    display_host = f"[{host}]" if ":" in host else host
    if not http_guard.is_loopback_host(host):
        print(f"⚠️ RAG Web 服务监听 {host}，知识库内容可被局域网访问。")
        if not config.RAG_ALLOWED_HOSTS:
            print("⚠️ 未设置 RAG_ALLOWED_HOSTS：非本机 Host 的请求都会被 403 拒绝。")
        if not config.RAG_API_TOKEN:
            print("⚠️ 未设置 RAG_API_TOKEN：/api/* 没有鉴权，强烈建议设置。")
    print(f"\n🌐 Web UI:  http://{display_host}:{port}")
    print(f"📡 API 文档: http://{display_host}:{port}/docs")
    print(
        f"📡 API 测试: curl -X POST http://{display_host}:{port}/api/ask "
        "-H 'Content-Type: application/json' -d '{\"question\": \"什么是RAG\"}'\n"
    )

    uvicorn.run(app, host=host, port=port)
