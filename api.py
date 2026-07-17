"""
REST API 模块

提供 FastAPI 接口，让外部系统可以通过 HTTP 调用 RAG 知识库。

面试亮点：
  - 展示后端 API 设计能力
  - 理解 RESTful 规范
  - 知道如何用 curl/Postman 测试
  - 展示系统可集成性（不只有 Web UI）

启动方式：
  这个 API 会和 Gradio 一起启动，挂载在 /api 路径下。

测试方式：
  curl -X POST http://localhost:7860/api/ask \
    -H "Content-Type: application/json" \
    -d '{"question": "什么是RAG？"}'
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional

import config


# ============================================================
# 请求/响应模型
# ============================================================

class AskRequest(BaseModel):
    """问答请求"""
    question: str
    use_hybrid: bool = False
    use_memory: bool = False
    rewrite_strategy: Optional[str] = "expand"


class AskResponse(BaseModel):
    """问答响应"""
    answer: str
    sources: List[dict] = []
    rewritten_query: Optional[str] = None


class StatusResponse(BaseModel):
    """系统状态响应"""
    status: str
    doc_count: int
    model_name: str
    embedding_model: str


class BuildResponse(BaseModel):
    """构建结果响应"""
    success: bool
    message: str
    doc_count: int = 0


# ============================================================
# FastAPI 应用
# ============================================================

def create_api(app_state) -> FastAPI:
    """
    创建 API 路由

    Args:
        app_state: 应用全局状态（和 main.py 共享）

    Returns:
        FastAPI: API 应用实例
    """
    api = FastAPI(
        title="RAG 知识库 API",
        description="基于 LangChain + ChromaDB 的智能知识库问答 API",
        version="1.0.0",
    )

    @api.get("/api/status", response_model=StatusResponse)
    async def get_status():
        """
        获取系统状态

        示例：
            GET /api/status
        """
        return StatusResponse(
            status="ready" if app_state.is_ready() else "not_ready",
            doc_count=app_state.doc_count,
            model_name=config.LLM_MODEL_NAME,
            embedding_model=config.EMBEDDING_MODEL_NAME,
        )

    @api.post("/api/ask", response_model=AskResponse)
    async def ask_question(request: AskRequest):
        """
        向知识库提问

        示例：
            POST /api/ask
            {
                "question": "什么是RAG？",
                "use_hybrid": false,
                "use_memory": false
            }
        """
        if not app_state.is_ready():
            return AskResponse(
                answer="知识库未就绪，请先上传文档并构建知识库。",
                sources=[],
            )

        try:
            # 对话记忆改写
            actual_query = request.question
            if request.use_memory and app_state.memory:
                actual_query = app_state.memory.rewrite_query_with_history(
                    request.question
                )

            # 检索 + 生成
            if request.use_hybrid and app_state.hybrid_retriever:
                docs = app_state.hybrid_retriever.search(
                    actual_query,
                    top_k=config.RETRIEVAL_TOP_K,
                )
                answer, docs = app_state.qa_chain.answer_with_documents(
                    actual_query,
                    docs,
                )
            else:
                answer, docs = app_state.qa_chain.answer(
                    actual_query,
                    rewrite_strategy=request.rewrite_strategy,
                )

            sources = []
            for doc in docs:
                sources.append({
                    "file_name": doc.metadata.get("file_name", "未知"),
                    "page": doc.metadata.get("page", ""),
                    "content_preview": doc.page_content[:200],
                })

            # 记录对话
            if request.use_memory and app_state.memory:
                app_state.memory.add_turn(request.question, answer)

            return AskResponse(
                answer=answer,
                sources=sources,
                rewritten_query=actual_query if actual_query != request.question else None,
            )

        except Exception as e:
            return AskResponse(
                answer=f"回答失败: {str(e)}",
                sources=[],
            )

    @api.post("/api/build", response_model=BuildResponse)
    async def build_knowledge_base(files: List[UploadFile] = File(...)):
        """
        上传文件并构建知识库

        示例（curl）：
            curl -X POST http://localhost:7860/api/build \
                -F "files=@document.pdf"
        """
        import tempfile
        from ingestion.loader import DocumentLoader
        from ingestion.splitter import TextSplitter
        from vectorstore.store import VectorStoreManager
        from retrieval.retriever import SmartRetriever
        from retrieval.hybrid_retriever import BM25Retriever, HybridRetriever
        from retrieval.conversation_memory import ConversationMemory
        from retrieval.query_rewriter import QueryRewriter
        from chain.qa_chain import QAChain
        from chain.langgraph_workflow import RAGWorkflow

        try:
            loader = DocumentLoader()
            splitter = TextSplitter()

            all_docs = []
            for upload_file in files:
                suffix = os.path.splitext(upload_file.filename)[1]
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as tmp:
                    content = await upload_file.read()
                    tmp.write(content)
                    tmp_path = tmp.name

                docs = loader.load_file(tmp_path)
                all_docs.extend(docs)
                os.unlink(tmp_path)

            if not all_docs:
                return BuildResponse(
                    success=False,
                    message="上传的文件中没有提取到有效内容",
                )

            chunks = splitter.split_documents(all_docs)

            app_state.vectorstore_manager = VectorStoreManager()
            app_state.vectorstore_manager.create_from_documents(chunks)
            vectorstore = app_state.vectorstore_manager.get_vectorstore()

            retriever = SmartRetriever(vectorstore=vectorstore)
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

            app_state.doc_count = len(chunks)
            app_state.status = "已就绪"

            return BuildResponse(
                success=True,
                message=f"知识库构建成功，共 {len(chunks)} 个文本块",
                doc_count=len(chunks),
            )

        except Exception as e:
            return BuildResponse(
                success=False,
                message=f"构建失败: {str(e)}",
            )

    @api.post("/api/clear")
    async def clear_knowledge_base():
        """清空知识库"""
        import shutil

        try:
            chroma_dir = config.CHROMA_PERSIST_DIR
            if os.path.exists(chroma_dir):
                shutil.rmtree(chroma_dir)

            app_state.vectorstore_manager = None
            app_state.qa_chain = None
            app_state.bm25_retriever = None
            app_state.hybrid_retriever = None
            app_state.memory = None
            app_state.workflow = None
            app_state.doc_count = 0
            app_state.status = "未初始化"

            return {"success": True, "message": "知识库已清空"}
        except Exception as e:
            return {"success": False, "message": f"清空失败: {str(e)}"}

    return api
