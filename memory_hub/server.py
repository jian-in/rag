from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .core import MEMORY_SCOPES, MemoryHub


class MemoryMCPError(RuntimeError):
    pass


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
                    },
                    "required": ["query"],
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
            hits = self.hub.search(
                query,
                scopes=scopes,
                top_k=int(arguments.get("top_k", 8)),
            )
            return {
                "query": query,
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
            )
        raise MemoryMCPError(f"unknown memory tool: {name}")


def create_app(hub: MemoryHub) -> FastAPI:
    tools = MemoryMCP(hub)
    app = FastAPI(title="Feili Personal Memory Hub", version="0.1.0")
    static_root = Path(__file__).resolve().parent / "static"
    app.mount("/assets", StaticFiles(directory=static_root), name="memory-assets")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "server": "feili-personal-memory", "version": "0.1.0"}

    @app.get("/.well-known/agent-memory.json")
    async def agent_manifest() -> dict[str, Any]:
        return {
            "name": "Feili Personal Memory Hub",
            "version": "0.1.0",
            "mcp_endpoint": "/mcp",
            "health_endpoint": "/health",
            "scopes": list(MEMORY_SCOPES),
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
        hits = await run_in_threadpool(
            hub.search,
            query,
            payload.get("scopes"),
            int(payload.get("top_k", 8)),
        )
        return {
            "query": query,
            "result_count": len(hits),
            "results": [hit.as_dict() for hit in hits],
        }

    @app.post("/api/context")
    async def api_context(request: Request) -> dict[str, Any]:
        payload = await request.json()
        query = str(payload.get("query", "")).strip()
        if not query:
            return JSONResponse(status_code=422, content={"detail": "query is required"})
        return await run_in_threadpool(
            hub.context,
            query,
            payload.get("scopes"),
            int(payload.get("top_k", 8)),
            int(payload.get("char_budget", 6000)),
        )

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
                "serverInfo": {"name": "feili-personal-memory", "version": "0.1.0"},
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
