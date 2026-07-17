"""Minimal stdio MCP server for the local RAG knowledge base."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from chain.qa_chain import QAChain
from ingestion.loader import DocumentLoader
from ingestion.splitter import TextSplitter
from retrieval.hybrid_retriever import BM25Retriever, HybridRetriever
from retrieval.reranker import Reranker
from retrieval.retriever import SmartRetriever
from vectorstore.store import VectorStoreManager


PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "rag-knowledge-base"
SERVER_VERSION = "1.0.0"

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


class MCPError(RuntimeError):
    """Raised when a tool call cannot be completed."""


def log(message: str) -> None:
    """Write diagnostics to stderr so stdout stays protocol-clean."""
    print(message, file=sys.stderr, flush=True)


@contextlib.contextmanager
def redirect_stdout_to_stderr():
    """Prevent library prints from corrupting the MCP stdio stream."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        yield
    output = buffer.getvalue()
    if output:
        print(output, file=sys.stderr, end="", flush=True)


class KnowledgeRuntime:
    """Lazy-loaded access layer for vector retrieval, BM25, and QA."""

    def __init__(self) -> None:
        self.loaded = False
        self.load_warning: Optional[str] = None
        self.vectorstore_manager: Optional[VectorStoreManager] = None
        self.vector_retriever: Optional[SmartRetriever] = None
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self.qa_chain: Optional[QAChain] = None
        self.doc_count = 0
        self.vector_count = 0

    def ensure_loaded(self) -> None:
        if self.loaded:
            return

        chroma_dir = (ROOT / config.CHROMA_PERSIST_DIR).resolve()
        data_dir = (ROOT / "data").resolve()

        if not chroma_dir.exists():
            raise MCPError(
                f"Vector index not found at {chroma_dir}. Build the knowledge base first."
            )

        if not data_dir.exists():
            raise MCPError(f"Data directory not found at {data_dir}.")

        with redirect_stdout_to_stderr():
            loader = DocumentLoader()
            splitter = TextSplitter()
            docs = loader.load_directory(str(data_dir))
            chunks = splitter.split_documents(docs)

            manager = VectorStoreManager(persist_dir=str(chroma_dir))
            vectorstore = manager.load()

            reranker = None
            warning = None
            if config.ENABLE_RERANKER:
                try:
                    reranker = Reranker()
                except Exception as exc:  # pragma: no cover - runtime fallback
                    warning = f"Reranker disabled: {exc}"

            vector_retriever = SmartRetriever(
                vectorstore=vectorstore,
                reranker=reranker,
            )
            bm25 = BM25Retriever()
            bm25.build_index(chunks)
            hybrid = HybridRetriever(
                bm25_retriever=bm25,
                vector_retriever=vector_retriever,
            )
            qa_chain = QAChain(retriever=vector_retriever)

        self.vectorstore_manager = manager
        self.vector_retriever = vector_retriever
        self.hybrid_retriever = hybrid
        self.qa_chain = qa_chain
        self.doc_count = len(chunks)
        self.vector_count = vectorstore._collection.count()
        self.load_warning = warning
        self.loaded = True

    def reload(self) -> Dict[str, Any]:
        self.loaded = False
        self.load_warning = None
        self.vectorstore_manager = None
        self.vector_retriever = None
        self.hybrid_retriever = None
        self.qa_chain = None
        self.doc_count = 0
        self.vector_count = 0
        self.ensure_loaded()
        return self.status()

    def status(self) -> Dict[str, Any]:
        chroma_dir = (ROOT / config.CHROMA_PERSIST_DIR).resolve()
        data_dir = (ROOT / "data").resolve()
        return {
            "ready": self.loaded,
            "root": str(ROOT),
            "data_dir": str(data_dir),
            "chroma_dir": str(chroma_dir),
            "doc_count": self.doc_count,
            "vector_count": self.vector_count,
            "llm_model": config.LLM_MODEL_NAME,
            "embedding_model": config.EMBEDDING_MODEL_NAME,
            "reranker_enabled": bool(config.ENABLE_RERANKER),
            "reranker_provider": config.RERANKER_PROVIDER,
            "reranker_model": config.RERANKER_MODEL_NAME,
            "warning": self.load_warning,
        }

    def _doc_key(self, doc) -> str:
        metadata = doc.metadata
        return "::".join(
            [
                str(metadata.get("source", "")),
                str(metadata.get("page", "")),
                str(metadata.get("chunk_index", "")),
            ]
        )

    def _serialize_doc(self, doc: Any) -> Dict[str, Any]:
        metadata = dict(doc.metadata)
        return {
            "file_name": metadata.get("file_name", "unknown"),
            "source": metadata.get("source", ""),
            "page": metadata.get("page", ""),
            "chunk_index": metadata.get("chunk_index", ""),
            "chunk_total": metadata.get("chunk_total", ""),
            "bm25_score": metadata.get("bm25_score"),
            "rrf_score": metadata.get("rrf_score"),
            "rerank_score": metadata.get("rerank_score"),
            "preview": doc.page_content[:400],
        }

    def _retrieve_docs(
        self,
        query: str,
        top_k: int,
        use_hybrid: bool,
        rewrite_strategy: Optional[str],
    ) -> List[Any]:
        self.ensure_loaded()
        assert self.vector_retriever is not None
        assert self.hybrid_retriever is not None
        assert self.qa_chain is not None

        if rewrite_strategy:
            with redirect_stdout_to_stderr():
                queries = self.qa_chain.rewriter.rewrite(query, strategy=rewrite_strategy)
            all_docs = []
            seen = set()
            for rewritten in queries:
                batch = self._retrieve_docs(
                    query=rewritten,
                    top_k=top_k,
                    use_hybrid=use_hybrid,
                    rewrite_strategy=None,
                )
                for doc in batch:
                    key = self._doc_key(doc)
                    if key not in seen:
                        seen.add(key)
                        all_docs.append(doc)
            return all_docs[:top_k]

        with redirect_stdout_to_stderr():
            if use_hybrid:
                docs = self.hybrid_retriever.search(
                    query,
                    top_k=top_k,
                    strategy="rrf",
                    use_reranker=config.ENABLE_RERANKER,
                )
            else:
                docs = self.vector_retriever.retrieve(
                    query=query,
                    strategy="mmr",
                    use_reranker=config.ENABLE_RERANKER,
                    top_k=top_k,
                    candidate_k=max(top_k * 4, top_k),
                    rerank_top_n=top_k,
                )
        return docs

    def search(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True,
        rewrite_strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        docs = self._retrieve_docs(
            query=query,
            top_k=top_k,
            use_hybrid=use_hybrid,
            rewrite_strategy=rewrite_strategy,
        )
        return {
            "query": query,
            "use_hybrid": use_hybrid,
            "rewrite_strategy": rewrite_strategy,
            "result_count": len(docs),
            "results": [self._serialize_doc(doc) for doc in docs],
        }

    def ask(
        self,
        question: str,
        top_k: int = 5,
        use_hybrid: bool = True,
        rewrite_strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.ensure_loaded()
        assert self.qa_chain is not None
        docs = self._retrieve_docs(
            query=question,
            top_k=top_k,
            use_hybrid=use_hybrid,
            rewrite_strategy=rewrite_strategy,
        )
        with redirect_stdout_to_stderr():
            answer, cited_docs = self.qa_chain.answer_with_documents(question, docs)
        return {
            "question": question,
            "answer": answer,
            "use_hybrid": use_hybrid,
            "rewrite_strategy": rewrite_strategy,
            "sources": [self._serialize_doc(doc) for doc in cited_docs],
        }


class MCPServer:
    """Small MCP server implementation over stdio."""

    def __init__(self) -> None:
        self.runtime = KnowledgeRuntime()
        self.initialized = False
        self._wire_mode = "framed"

    def _tool_definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "get_status",
                "description": "Load and return knowledge base status, model settings, and index readiness.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
            {
                "name": "reload_knowledge_base",
                "description": "Reload the persisted Chroma index and rebuild the BM25 cache from the local data directory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
            {
                "name": "search_knowledge",
                "description": "Search the local RAG knowledge base. Use this when the user asks for facts from the configured documents.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "User search query."},
                        "top_k": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 20,
                            "default": 5,
                        },
                        "use_hybrid": {
                            "type": "boolean",
                            "default": True,
                            "description": "Use hybrid BM25 + vector retrieval.",
                        },
                        "rewrite_strategy": {
                            "type": ["string", "null"],
                            "enum": [None, "expand", "decompose", "hyde"],
                            "default": None,
                            "description": "Optional query rewriting strategy before retrieval.",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "ask_knowledge",
                "description": "Answer a question using the local RAG knowledge base and cite supporting chunks. Use this before saying the knowledge base has no relevant content.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "Question to answer."},
                        "top_k": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 20,
                            "default": 5,
                        },
                        "use_hybrid": {
                            "type": "boolean",
                            "default": True,
                            "description": "Use hybrid BM25 + vector retrieval.",
                        },
                        "rewrite_strategy": {
                            "type": ["string", "null"],
                            "enum": [None, "expand", "decompose", "hyde"],
                            "default": None,
                            "description": "Optional query rewriting strategy before retrieval.",
                        },
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
            },
        ]

    def _read_message(self) -> Optional[Dict[str, Any]]:
        headers: Dict[str, str] = {}
        first_json_line: Optional[str] = None
        while True:
            line = sys.stdin.buffer.readline()
            if not line:
                return None
            if line in (b"\r\n", b"\n"):
                if headers:
                    break
                continue
            header_line = line.decode("utf-8", errors="replace").strip()
            if not header_line:
                if headers:
                    break
                continue
            if not headers and header_line.startswith("{"):
                first_json_line = header_line
                break
            if ":" not in header_line:
                if headers:
                    raise MCPError(f"Malformed MCP header line: {header_line}")
                continue
            key, value = header_line.split(":", 1)
            normalized_key = key.strip().lower()
            if normalized_key == "content-length":
                headers[normalized_key] = value.strip()
                continue
            if not headers and header_line.startswith("{"):
                first_json_line = header_line
                break
            headers[normalized_key] = value.strip()

        if first_json_line is not None:
            self._wire_mode = "json_line"
            return json.loads(first_json_line)

        self._wire_mode = "framed"
        if "content-length" not in headers:
            raise MCPError("Missing Content-Length header.")
        length = int(headers["content-length"])
        payload = sys.stdin.buffer.read(length)
        if len(payload) != length:
            return None
        return json.loads(payload.decode("utf-8"))

    def _send(self, message: Dict[str, Any]) -> None:
        payload = json.dumps(message, ensure_ascii=False).encode("utf-8")
        if self._wire_mode == "json_line":
            sys.stdout.buffer.write(payload + b"\n")
            sys.stdout.buffer.flush()
            return
        header = f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
        sys.stdout.buffer.write(header)
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()

    def _result(self, message_id: Any, result: Dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "id": message_id, "result": result})

    def _error(self, message_id: Any, code: int, message: str) -> None:
        self._send(
            {
                "jsonrpc": "2.0",
                "id": message_id,
                "error": {"code": code, "message": message},
            }
        )

    def _tool_response(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        return {
            "content": [{"type": "text", "text": text}],
            "structuredContent": payload,
            "isError": False,
        }

    def _dispatch_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if name == "get_status":
            if not self.runtime.loaded:
                try:
                    self.runtime.ensure_loaded()
                except Exception as exc:
                    self.runtime.load_warning = f"Auto-load failed: {exc}"
            return self._tool_response(self.runtime.status())
        if name == "reload_knowledge_base":
            return self._tool_response(self.runtime.reload())
        if name == "search_knowledge":
            query = arguments.get("query")
            if not query:
                raise MCPError("`query` is required.")
            top_k = int(arguments.get("top_k", 5))
            return self._tool_response(
                self.runtime.search(
                    query=query,
                    top_k=top_k,
                    use_hybrid=bool(arguments.get("use_hybrid", True)),
                    rewrite_strategy=arguments.get("rewrite_strategy"),
                )
            )
        if name == "ask_knowledge":
            question = arguments.get("question")
            if not question:
                raise MCPError("`question` is required.")
            top_k = int(arguments.get("top_k", 5))
            return self._tool_response(
                self.runtime.ask(
                    question=question,
                    top_k=top_k,
                    use_hybrid=bool(arguments.get("use_hybrid", True)),
                    rewrite_strategy=arguments.get("rewrite_strategy"),
                )
            )
        raise MCPError(f"Unknown tool: {name}")

    def handle(self, message: Dict[str, Any]) -> None:
        method = message.get("method")
        message_id = message.get("id")
        params = message.get("params", {})

        if method == "initialize":
            self.initialized = True
            self._result(
                message_id,
                {
                    "protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                },
            )
            return

        if method == "notifications/initialized":
            return

        if method == "ping":
            self._result(message_id, {})
            return

        if method == "tools/list":
            self._result(message_id, {"tools": self._tool_definitions()})
            return

        if method == "tools/call":
            try:
                tool_name = params.get("name", "")
                arguments = params.get("arguments", {}) or {}
                result = self._dispatch_tool(tool_name, arguments)
                self._result(message_id, result)
            except Exception as exc:  # pragma: no cover - protocol surface
                error_payload = {
                    "content": [{"type": "text", "text": str(exc)}],
                    "isError": True,
                }
                self._result(message_id, error_payload)
            return

        if method == "resources/list":
            self._result(message_id, {"resources": []})
            return

        if message_id is not None:
            self._error(message_id, -32601, f"Method not found: {method}")

    def serve(self) -> None:
        log(f"{SERVER_NAME} MCP server started from {ROOT}")
        while True:
            message = self._read_message()
            if message is None:
                break
            try:
                self.handle(message)
            except Exception:
                if "id" in message:
                    self._error(message.get("id"), -32603, traceback.format_exc())
                else:
                    log(traceback.format_exc())


def main() -> int:
    os.chdir(ROOT)
    server = MCPServer()
    server.serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
