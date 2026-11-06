from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .core import MEMORY_SCOPES, RETRIEVAL_MODES, MemoryHub


class MemoryMCPError(RuntimeError):
    pass


# 允许的 Host 头主机名（含任意端口）。防 DNS Rebinding：攻击者把自己的域名
# 解析到 127.0.0.1 后，浏览器发出的 Host 仍是攻击者域名。
TRUSTED_HOSTNAMES = frozenset({"127.0.0.1", "localhost", "::1"})
# 写操作必须是 JSON：text/plain、表单等 CORS 简单请求不触发预检，
# 任何网页都能跨站发出；强制 application/json 后浏览器会先预检，而本服务不放行 CORS。
_BODY_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _request_hostname(request: Request) -> str:
    # Host 形如 "127.0.0.1:8771" 或 "[::1]:8771"，补上 "//" 才能按 netloc 解析
    return (urlsplit("//" + request.headers.get("host", "")).hostname or "").lower()


def _is_json_request(request: Request) -> bool:
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    return media_type == "application/json"


class MemoryMCP:
    """Transport-independent MCP tool surface for the personal memory hub."""

    def __init__(self, hub: MemoryHub) -> None:
        self.hub = hub

    def tool_definitions(self) -> list[dict[str, Any]]:
        scope_schema = {
            "type": "array",
            "items": {"type": "string", "enum": list(MEMORY_SCOPES)},
            "uniqueItems": True,
            "description": "Optional isolated memory domains to query.",
        }
        retrieval_mode_schema = {
            "type": "string",
            "enum": list(RETRIEVAL_MODES),
            "default": "hybrid",
            "description": "Use local for lexical-only low-latency recall; hybrid may call configured embedding and reranking providers.",
        }
        return [
            {
                "name": "memory_status",
                "description": "Inspect the local personal memory indexes and provider readiness without exposing secrets.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
            {
                "name": "memory_sync",
                "description": "Incrementally synchronize selected personal memory domains from their local sources.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"scopes": scope_schema},
                    "additionalProperties": False,
                },
            },
            {
                "name": "memory_search",
                "description": "Search isolated personal memories and return ranked source citations. Use before asking the user to repeat known preferences or project context.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "minLength": 1},
                        "scopes": scope_schema,
                        "top_k": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 50,
                            "default": 8,
                        },
                        "retrieval_mode": retrieval_mode_schema,
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "memory_context",
                "description": "Build a compact, cited context bundle for an agent prompt from the relevant personal memories.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "minLength": 1},
                        "scopes": scope_schema,
                        "top_k": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 50,
                            "default": 8,
                        },
                        "char_budget": {
                            "type": "integer",
                            "minimum": 200,
                            "maximum": 30000,
                            "default": 6000,
                        },
                        "retrieval_mode": retrieval_mode_schema,
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "memory_propose",
                "description": "Submit a possible personal memory for local human review. This never writes directly to the memory sources.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "scope": {"type": "string", "enum": list(MEMORY_SCOPES)},
                        "content": {"type": "string", "minLength": 3, "maxLength": 4000},
                        "reason": {"type": "string", "maxLength": 1000},
                        "source": {"type": "string", "maxLength": 200},
                        "session_id": {"type": "string", "maxLength": 200},
                    },
                    "required": ["scope", "content"],
                    "additionalProperties": False,
                },
            },
        ]

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        arguments = arguments or {}
        scopes = arguments.get("scopes")
        if name == "memory_status":
            return self.hub.status()
        if name == "memory_sync":
            return self.hub.sync(scopes=scopes)
        if name == "memory_search":
            query = str(arguments.get("query", "")).strip()
            if not query:
                raise MemoryMCPError("query is required")
            retrieval_mode = str(arguments.get("retrieval_mode", "hybrid"))
            hits = self.hub.search(
                query,
                scopes=scopes,
                top_k=int(arguments.get("top_k", 8)),
                retrieval_mode=retrieval_mode,
            )
            return {
                "query": query,
                "retrieval_mode": retrieval_mode,
                "result_count": len(hits),
                "results": [hit.as_dict() for hit in hits],
            }
        if name == "memory_context":
            query = str(arguments.get("query", "")).strip()
            if not query:
                raise MemoryMCPError("query is required")
            return self.hub.context(
                query,
                scopes=scopes,
                top_k=int(arguments.get("top_k", 8)),
                char_budget=int(arguments.get("char_budget", 6000)),
                retrieval_mode=str(arguments.get("retrieval_mode", "hybrid")),
            )
        if name == "memory_propose":
            scope = str(arguments.get("scope", "")).strip()
            content = str(arguments.get("content", "")).strip()
            if not scope or not content:
                raise MemoryMCPError("scope and content are required")
            return self.hub.propose_memory(
                scope=scope,
                content=content,
                reason=str(arguments.get("reason", "")),
                source=str(arguments.get("source", "agent")),
                session_id=(
                    str(arguments["session_id"])
                    if arguments.get("session_id") is not None
                    else None
                ),
            )
        raise MemoryMCPError(f"unknown memory tool: {name}")


def create_app(
    hub: MemoryHub,
    *,
    trusted_hostnames: frozenset[str] = TRUSTED_HOSTNAMES,
) -> FastAPI:
    tools = MemoryMCP(hub)
    app = FastAPI(title="Feili Personal Memory Hub", version="0.3.0")
    static_root = Path(__file__).resolve().parent / "static"
    app.mount("/assets", StaticFiles(directory=static_root), name="memory-assets")

    @app.middleware("http")
    async def enforce_local_json_requests(request: Request, call_next):
        if _request_hostname(request) not in trusted_hostnames:
            return JSONResponse(
                status_code=403,
                content={"detail": "forbidden: untrusted Host header"},
            )
        if request.method in _BODY_METHODS and not _is_json_request(request):
            return JSONResponse(
                status_code=415,
                content={"detail": "Content-Type must be application/json"},
            )
        return await call_next(request)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "server": "feili-personal-memory", "version": "0.3.0"}

    @app.get("/.well-known/agent-memory.json")
    async def agent_manifest() -> dict[str, Any]:
        return {
            "name": "Feili Personal Memory Hub",
            "version": "0.3.0",
            "mcp_endpoint": "/mcp",
            "health_endpoint": "/health",
            "scopes": list(MEMORY_SCOPES),
            "retrieval_modes": list(RETRIEVAL_MODES),
            "default_retrieval_mode": "hybrid",
            "automatic_hook_mode": "local",
            "recommended_tool": "memory_context",
            "instruction": "Retrieve relevant memory before asking Afei to repeat known preferences or active project context.",
        }

    @app.get("/api/status")
    async def api_status() -> dict[str, Any]:
        return await run_in_threadpool(hub.status)

    @app.post("/api/sync")
    async def api_sync(request: Request) -> dict[str, Any]:
        payload = await request.json()
        return await run_in_threadpool(hub.sync, payload.get("scopes"))

    @app.post("/api/search")
    async def api_search(request: Request) -> dict[str, Any]:
        payload = await request.json()
        query = str(payload.get("query", "")).strip()
        if not query:
            return JSONResponse(status_code=422, content={"detail": "query is required"})
        retrieval_mode = str(payload.get("retrieval_mode", "hybrid"))
        if retrieval_mode not in RETRIEVAL_MODES:
            return JSONResponse(
                status_code=422,
                content={"detail": f"retrieval_mode must be one of: {', '.join(RETRIEVAL_MODES)}"},
            )
        hits = await run_in_threadpool(
            hub.search,
            query,
            payload.get("scopes"),
            int(payload.get("top_k", 8)),
            retrieval_mode,
        )
        return {
            "query": query,
            "retrieval_mode": retrieval_mode,
            "result_count": len(hits),
            "results": [hit.as_dict() for hit in hits],
        }

    @app.post("/api/context")
    async def api_context(request: Request) -> dict[str, Any]:
        payload = await request.json()
        query = str(payload.get("query", "")).strip()
        if not query:
            return JSONResponse(status_code=422, content={"detail": "query is required"})
        retrieval_mode = str(payload.get("retrieval_mode", "hybrid"))
        if retrieval_mode not in RETRIEVAL_MODES:
            return JSONResponse(
                status_code=422,
                content={"detail": f"retrieval_mode must be one of: {', '.join(RETRIEVAL_MODES)}"},
            )
        return await run_in_threadpool(
            hub.context,
            query,
            payload.get("scopes"),
            int(payload.get("top_k", 8)),
            int(payload.get("char_budget", 6000)),
            retrieval_mode,
        )

    @app.get("/api/candidates")
    async def api_candidates(status: str = "pending") -> JSONResponse:
        try:
            candidates = await run_in_threadpool(hub.review_memories, status=status)
            return JSONResponse(content={"status": status, "candidates": candidates})
        except ValueError as exc:
            return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.post("/api/candidates/propose")
    async def api_propose_candidate(request: Request) -> JSONResponse:
        payload = await request.json()
        try:
            candidate = await run_in_threadpool(
                hub.propose_memory,
                scope=str(payload.get("scope", "")),
                content=str(payload.get("content", "")),
                reason=str(payload.get("reason", "")),
                source=str(payload.get("source", "dashboard")),
                session_id=(
                    str(payload["session_id"])
                    if payload.get("session_id") is not None
                    else None
                ),
            )
            return JSONResponse(content={"candidate": candidate})
        except ValueError as exc:
            return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.patch("/api/candidates/{candidate_id}")
    async def api_update_candidate(candidate_id: str, request: Request) -> JSONResponse:
        payload = await request.json()
        try:
            candidate = await run_in_threadpool(
                hub.review_memory,
                candidate_id,
                "update",
                scope=payload.get("scope"),
                content=payload.get("content"),
                note=str(payload.get("note", "")),
            )
            return JSONResponse(content={"candidate": candidate})
        except KeyError as exc:
            return JSONResponse(status_code=404, content={"detail": str(exc)})
        except ValueError as exc:
            return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.post("/api/candidates/{candidate_id}/decision")
    async def api_decide_candidate(candidate_id: str, request: Request) -> JSONResponse:
        payload = await request.json()
        action = str(payload.get("action", "")).strip()
        if action not in {"approve", "reject", "forget"}:
            return JSONResponse(status_code=422, content={"detail": "invalid decision action"})
        try:
            candidate = await run_in_threadpool(
                hub.review_memory,
                candidate_id,
                action,
                scope=payload.get("scope"),
                content=payload.get("content"),
                note=str(payload.get("note", "")),
            )
            return JSONResponse(content={"candidate": candidate})
        except KeyError as exc:
            return JSONResponse(status_code=404, content={"detail": str(exc)})
        except (ValueError, RuntimeError) as exc:
            return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.post("/mcp")
    async def mcp(request: Request) -> JSONResponse:
        message = await request.json()
        message_id = message.get("id")
        method = message.get("method")
        params = message.get("params", {}) or {}

        if method == "notifications/initialized":
            return JSONResponse(status_code=202, content={})
        if method == "initialize":
            result = {
                "protocolVersion": params.get("protocolVersion", "2024-11-05"),
                "capabilities": {"tools": {}, "resources": {}},
                "serverInfo": {"name": "feili-personal-memory", "version": "0.3.0"},
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": tools.tool_definitions()}
        elif method == "tools/call":
            try:
                payload = await run_in_threadpool(
                    tools.call_tool,
                    str(params.get("name", "")),
                    params.get("arguments", {}) or {},
                )
                result = {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(payload, ensure_ascii=False, indent=2),
                        }
                    ],
                    "structuredContent": payload,
                    "isError": False,
                }
            except Exception as exc:
                result = {
                    "content": [{"type": "text", "text": str(exc)}],
                    "isError": True,
                }
        elif method == "resources/list":
            result = {
                "resources": [
                    {
                        "uri": "memory://entry",
                        "name": "Personal memory entry",
                        "description": "Discovery instructions and current index status.",
                        "mimeType": "application/json",
                    }
                ]
            }
        elif method == "resources/read" and params.get("uri") == "memory://entry":
            payload = {
                "instruction": "Call memory_context with the current task before asking Afei to repeat known information.",
                "scopes": list(MEMORY_SCOPES),
                "status": await run_in_threadpool(hub.status),
            }
            result = {
                "contents": [
                    {
                        "uri": "memory://entry",
                        "mimeType": "application/json",
                        "text": json.dumps(payload, ensure_ascii=False, indent=2),
                    }
                ]
            }
        else:
            return JSONResponse(
                content={
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }
            )
        return JSONResponse(content={"jsonrpc": "2.0", "id": message_id, "result": result})

    @app.get("/")
    async def dashboard() -> FileResponse:
        return FileResponse(static_root / "index.html", media_type="text/html")

    return app
