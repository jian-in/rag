"""HTTP/SSE MCP server for the local RAG knowledge base."""

from __future__ import annotations

import asyncio
import json
import os
import traceback
import uuid
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
import uvicorn

from mcp_server import PROTOCOL_VERSION, SERVER_NAME, SERVER_VERSION, MCPServer, log

SSE_HEARTBEAT_SECONDS = 15

_LOCAL_PROXY_BYPASS = ("localhost", "127.0.0.1", "::1")
_BROKEN_PROXY_VALUES = {
    "http://127.0.0.1:9",
    "https://127.0.0.1:9",
    "http://localhost:9",
    "https://localhost:9",
}


def _sanitize_proxy_environment() -> None:
    """Keep localhost MCP traffic local and ignore known dead proxy placeholders."""
    existing_no_proxy = os.getenv("NO_PROXY") or os.getenv("no_proxy") or ""
    entries = [item.strip() for item in existing_no_proxy.split(",") if item.strip()]
    for item in _LOCAL_PROXY_BYPASS:
        if item not in entries:
            entries.append(item)
    os.environ["NO_PROXY"] = ",".join(entries)

    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        value = os.getenv(key, "").strip().rstrip("/").lower()
        if value in _BROKEN_PROXY_VALUES:
            os.environ.pop(key, None)


_sanitize_proxy_environment()


class HTTPMCPServer:
    """Expose the same MCP tools over HTTP and SSE transports."""

    def __init__(self) -> None:
        self.engine = MCPServer()
        self.sessions: Dict[str, asyncio.Queue[Optional[str]]] = {}

    def _result(self, message_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": message_id, "result": result}

    def _error(self, message_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": message_id,
            "error": {"code": code, "message": message},
        }

    def _format_sse(
        self,
        *,
        event: Optional[str] = None,
        data: Optional[str] = None,
        comment: Optional[str] = None,
        retry: Optional[int] = None,
    ) -> str:
        lines = []
        if comment is not None:
            lines.append(f": {comment}")
        if retry is not None:
            lines.append(f"retry: {retry}")
        if event is not None:
            lines.append(f"event: {event}")
        if data is not None:
            for line in str(data).splitlines() or [""]:
                lines.append(f"data: {line}")
        return "\n".join(lines) + "\n\n"

    def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        method = message.get("method")
        message_id = message.get("id")
        params = message.get("params", {}) or {}

        if method == "initialize":
            self.engine.initialized = True
            return self._result(
                message_id,
                {
                    "protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                },
            )

        if method == "notifications/initialized":
            return None

        if method == "ping":
            return self._result(message_id, {})

        if method == "tools/list":
            return self._result(message_id, {"tools": self.engine._tool_definitions()})

        if method == "tools/call":
            try:
                tool_name = params.get("name", "")
                arguments = params.get("arguments", {}) or {}
                result = self.engine._dispatch_tool(tool_name, arguments)
                return self._result(message_id, result)
            except Exception as exc:
                return self._result(
                    message_id,
                    {
                        "content": [{"type": "text", "text": str(exc)}],
                        "isError": True,
                    },
                )

        if method == "resources/list":
            return self._result(message_id, {"resources": []})

        if message_id is None:
            return None
        return self._error(message_id, -32601, f"Method not found: {method}")

    def open_session(self, session_id: str) -> asyncio.Queue[Optional[str]]:
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self.sessions[session_id] = queue
        log(f"SSE session opened: {session_id}")
        return queue

    async def sse_stream(
        self,
        session_id: str,
        base_url: str,
        queue: asyncio.Queue[Optional[str]],
        request: Request,
    ) -> AsyncIterator[bytes]:
        endpoint_url = f"{base_url.rstrip('/')}/messages?sessionId={session_id}"
        initial_payload = (
            self._format_sse(comment="connected")
            + self._format_sse(retry=1500)
            + self._format_sse(event="endpoint", data=endpoint_url)
        )
        yield initial_payload.encode("utf-8")

        try:
            while True:
                if await request.is_disconnected():
                    log(f"SSE client disconnected: {session_id}")
                    break
                try:
                    item = await asyncio.wait_for(
                        queue.get(),
                        timeout=SSE_HEARTBEAT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    yield self._format_sse(comment="keepalive").encode("utf-8")
                    continue

                if item is None:
                    break
                yield item.encode("utf-8")
        finally:
            self.sessions.pop(session_id, None)
            log(f"SSE session closed: {session_id}")

    async def enqueue_response(self, session_id: str, response: Dict[str, Any]) -> None:
        queue = self.sessions.get(session_id)
        if queue is None:
            raise HTTPException(status_code=404, detail=f"Unknown sessionId: {session_id}")
        payload = json.dumps(response, ensure_ascii=False)
        await queue.put(self._format_sse(event="message", data=payload))


server = HTTPMCPServer()
app = FastAPI(title="RAG MCP HTTP Server", version=SERVER_VERSION)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "server": SERVER_NAME,
        "version": SERVER_VERSION,
        "protocolVersion": PROTOCOL_VERSION,
    }


@app.post("/mcp")
async def mcp_post(request: Request):
    try:
        message = await request.json()
        response = server.handle_message(message)
        if response is None:
            return JSONResponse(status_code=202, content={})
        return JSONResponse(content=response)
    except Exception as exc:
        trace = traceback.format_exc()
        log(trace)
        return JSONResponse(
            status_code=500,
            content=server._error(None, -32603, str(exc)),
        )


@app.get("/sse")
async def sse(request: Request):
    session_id = uuid.uuid4().hex
    base_url = str(request.base_url).rstrip("/")
    queue = server.open_session(session_id)
    return StreamingResponse(
        server.sse_stream(session_id, base_url, queue, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/messages")
async def sse_messages(
    request: Request,
    session_id: str = Query(..., alias="sessionId"),
):
    try:
        message = await request.json()
        response = server.handle_message(message)
        if response is not None:
            await server.enqueue_response(session_id, response)
        return PlainTextResponse("accepted", status_code=202)
    except HTTPException:
        raise
    except Exception as exc:
        trace = traceback.format_exc()
        log(trace)
        error_response = server._error(
            message.get("id") if "message" in locals() else None,
            -32603,
            str(exc),
        )
        try:
            await server.enqueue_response(session_id, error_response)
        except HTTPException:
            log(f"Unable to deliver error response to closed session: {session_id}")
        return PlainTextResponse("accepted", status_code=202)


@app.post("/sse")
async def sse_messages_alias(
    request: Request,
    session_id: str = Query(..., alias="sessionId"),
):
    return await sse_messages(request, session_id)


def main() -> int:
    host = os.getenv("MCP_HTTP_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_HTTP_PORT", "8765"))
    log(f"{SERVER_NAME} HTTP MCP server listening on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
