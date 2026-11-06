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
import shutil
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, File, Request, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from typing import List, Optional

import config
import http_guard


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


class ClearRequest(BaseModel):
    """清空知识库请求：必须显式 confirm"""
    confirm: bool = False


class RetrieveRequest(BaseModel):
    """纯检索请求（不过 LLM，返回完整 chunk）"""
    query: str
    top_k: int = 5
    use_hybrid: bool = True


class RetrieveResult(BaseModel):
    """单个检索结果"""
    file_name: str
    content: str
    source: str = ""
    chunk_index: int = 0
    score: float = 0.0


class RetrieveResponse(BaseModel):
    """检索响应"""
    query: str
    results: List[RetrieveResult] = []
    total: int = 0


class AddDocumentItem(BaseModel):
    """待写入的单个文档"""
    doc_id: str
    name: str = ""
    text: str


class AddDocumentsRequest(BaseModel):
    """批量写入外部资料（如课策上传的课程文档）"""
    course_id: str
    documents: List[AddDocumentItem] = []
    origin: str = "kece"
    replace: bool = True  # True 时先删除该 course_id 旧内容再写入（镜像同步）


class AddDocumentsResponse(BaseModel):
    """写入响应"""
    success: bool
    added_chunks: int = 0
    deleted: int = 0
    doc_count: int = 0
    message: str = ""


class DeleteDocumentsRequest(BaseModel):
    """按课程/文档删除"""
    course_id: Optional[str] = None
    doc_id: Optional[str] = None


class DeleteDocumentsResponse(BaseModel):
    """删除响应"""
    success: bool
    deleted: int = 0
    doc_count: int = 0
    message: str = ""


# ============================================================
# FastAPI 应用
# ============================================================

BUILD_CLIENT_HEADER = "X-RAG-Client"


def create_api(
    app_state,
    add_documents_fn=None,
    delete_documents_fn=None,
    build_documents_fn=None,
    clear_fn=None,
    *,
    allowed_hosts: Optional[str] = None,
    api_token: Optional[str] = None,
) -> FastAPI:
    """
    创建 API 路由

    Args:
        app_state: 应用全局状态（和 main.py 共享）
        add_documents_fn: 写入文档的回调（由 main.py 传入 kb_add_documents，避免重复导入 main）
        delete_documents_fn: 删除文档的回调（由 main.py 传入 kb_delete_documents）
        build_documents_fn: 上传文件入库的回调，参数为 [(临时路径, 原始文件名)]
        clear_fn: 清空知识库的回调
        allowed_hosts: 额外信任的 Host（逗号分隔），默认读 config.RAG_ALLOWED_HOSTS
        api_token: /api/* 的 Bearer 令牌，默认读 config.RAG_API_TOKEN；空则不校验

    Returns:
        FastAPI: API 应用实例
    """
    api = FastAPI(
        title="RAG 知识库 API",
        description="基于 LangChain + ChromaDB 的智能知识库问答 API",
        version="1.0.0",
    )
    trusted = http_guard.trusted_hostnames(
        http_guard.parse_hostnames(
            config.RAG_ALLOWED_HOSTS if allowed_hosts is None else allowed_hosts
        )
    )
    token = config.RAG_API_TOKEN if api_token is None else api_token

    @api.middleware("http")
    async def guard_local_requests(request: Request, call_next):
        # Host 校验覆盖整个站点（含挂载的 Gradio UI），防 DNS Rebinding。
        rejection = http_guard.forbidden_host_response(request, trusted)
        if rejection is None and request.url.path.startswith("/api/"):
            rejection = http_guard.bearer_token_response(request, token)
            if rejection is None:
                if request.url.path == "/api/build":
                    rejection = http_guard.custom_header_required_response(
                        request, BUILD_CLIENT_HEADER
                    )
                else:
                    rejection = http_guard.json_required_response(request)
        return rejection or await call_next(request)

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
    def ask_question(request: AskRequest):
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
            traceback.print_exc()
            return AskResponse(
                answer=f"回答失败: {str(e)}",
                sources=[],
            )

    @api.post("/api/build", response_model=BuildResponse)
    async def build_knowledge_base(files: List[UploadFile] = File(...)):
        """
        上传文件并写入知识库（追加；同名文件会替换旧版本）

        multipart 上传无法要求 JSON，因此必须带自定义请求头 X-RAG-Client
        （任意非空值）：浏览器跨站发送自定义头会触发 CORS 预检而被拦下。

        示例（curl）：
            curl -X POST http://127.0.0.1:7860/api/build \
                -H "X-RAG-Client: curl" \
                -F "files=@document.pdf"
        """
        if build_documents_fn is None:
            return BuildResponse(success=False, message="服务未配置文档构建回调")

        temp_dir = tempfile.mkdtemp(prefix="rag-upload-")
        try:
            uploads = []
            for index, upload_file in enumerate(files):
                display_name = os.path.basename(upload_file.filename or "") or f"upload-{index}"
                suffix = os.path.splitext(display_name)[1]
                # 临时文件名只用于落盘，引用来源显示原始文件名
                tmp_path = os.path.join(temp_dir, f"{index}{suffix}")
                with open(tmp_path, "wb") as handle:
                    handle.write(await upload_file.read())
                uploads.append((tmp_path, display_name))

            result = await run_in_threadpool(build_documents_fn, uploads)
            if not result.get("added_chunks"):
                return BuildResponse(
                    success=False,
                    message="上传的文件中没有提取到有效内容",
                    doc_count=result.get("doc_count", 0),
                )
            return BuildResponse(
                success=True,
                message=f"已写入 {result['added_chunks']} 个文本块",
                doc_count=result.get("doc_count", 0),
            )
        except Exception as e:
            traceback.print_exc()
            return BuildResponse(success=False, message=f"构建失败: {str(e)}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @api.post("/api/clear")
    async def clear_knowledge_base(request: ClearRequest):
        """
        清空知识库（需显式确认）

        示例：
            POST /api/clear
            {"confirm": true}
        """
        if not request.confirm:
            return {"success": False, "message": '需要 {"confirm": true} 确认清空'}
        if clear_fn is None:
            return {"success": False, "message": "服务未配置清空回调"}
        try:
            message = await run_in_threadpool(clear_fn)
            return {"success": True, "message": message}
        except Exception as e:
            traceback.print_exc()
            return {"success": False, "message": f"清空失败: {str(e)}"}

    @api.post("/api/retrieve", response_model=RetrieveResponse)
    def retrieve_chunks(request: RetrieveRequest):
        """
        纯检索：返回完整 chunk 内容（不过 LLM）

        示例：
            POST /api/retrieve
            {"query": "进程调度算法", "top_k": 5, "use_hybrid": true}
        """
        if not app_state.is_ready():
            return RetrieveResponse(query=request.query, results=[], total=0)

        try:
            if request.use_hybrid and app_state.hybrid_retriever:
                docs = app_state.hybrid_retriever.search(
                    request.query, top_k=request.top_k
                )
            elif app_state.qa_chain and app_state.qa_chain.retriever:
                docs = app_state.qa_chain.retriever.retrieve(
                    request.query, top_k=request.top_k
                )
            else:
                return RetrieveResponse(query=request.query, results=[], total=0)

            results = []
            for doc in docs:
                # 混合检索写入的是 rrf_score/hybrid_score，红排写的是 score，依次回退
                md = doc.metadata
                score = md.get("score", md.get("rrf_score", md.get("hybrid_score", 0.0)))
                results.append(RetrieveResult(
                    file_name=doc.metadata.get("file_name", "未知"),
                    content=doc.page_content,
                    source=doc.metadata.get("source", ""),
                    chunk_index=doc.metadata.get("chunk_index", 0),
                    score=float(score or 0.0),
                ))

            return RetrieveResponse(
                query=request.query,
                results=results,
                total=len(results),
            )
        except Exception as e:
            traceback.print_exc()
            return RetrieveResponse(query=request.query, results=[], total=0)

    @api.post("/api/documents/add", response_model=AddDocumentsResponse)
    def add_documents(request: AddDocumentsRequest):
        """
        将外部资料（如课策上传的课程文档）写入知识库作为补充。

        replace=True（默认）时先删除该 course_id 的旧内容再写入，实现“课程文档 ↔ 知识库”镜像同步。

        示例：
            POST /api/documents/add
            {"course_id": "c1", "documents": [{"doc_id": "d1", "name": "第1章.md", "text": "..."}]}
        """
        if add_documents_fn is None or delete_documents_fn is None:
            return AddDocumentsResponse(success=False, message="服务未配置文档写入回调")
        if not app_state.is_ready():
            return AddDocumentsResponse(success=False, message="知识库尚未就绪，请先构建或加载")

        try:
            deleted = 0
            if request.replace:
                del_result = delete_documents_fn(course_id=request.course_id)
                deleted = del_result.get("deleted", 0)

            items = [it.model_dump() for it in request.documents]
            add_result = add_documents_fn(items, request.course_id, origin=request.origin)
            return AddDocumentsResponse(
                success=True,
                added_chunks=add_result.get("added_chunks", 0),
                deleted=deleted,
                doc_count=add_result.get("doc_count", 0),
                message="写入成功",
            )
        except Exception as e:
            traceback.print_exc()
            return AddDocumentsResponse(success=False, message=f"写入失败: {str(e)}")

    @api.post("/api/documents/delete", response_model=DeleteDocumentsResponse)
    def delete_documents(request: DeleteDocumentsRequest):
        """
        按课程（course_id）或单个文档（doc_id）从知识库删除。

        示例：
            POST /api/documents/delete
            {"course_id": "c1"}
        """
        if delete_documents_fn is None:
            return DeleteDocumentsResponse(success=False, message="服务未配置文档删除回调")
        if not app_state.is_ready():
            return DeleteDocumentsResponse(success=False, message="知识库尚未就绪")
        if not request.course_id and not request.doc_id:
            return DeleteDocumentsResponse(success=False, message="需要提供 course_id 或 doc_id")

        try:
            result = delete_documents_fn(course_id=request.course_id, doc_id=request.doc_id)
            return DeleteDocumentsResponse(
                success=True,
                deleted=result.get("deleted", 0),
                doc_count=result.get("doc_count", 0),
                message="删除成功",
            )
        except Exception as e:
            traceback.print_exc()
            return DeleteDocumentsResponse(success=False, message=f"删除失败: {str(e)}")

    return api
