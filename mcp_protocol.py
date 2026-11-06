"""Transport-independent MCP JSON-RPC dispatch shared by every MCP surface.

The stdio RAG server (mcp_server.py), its HTTP/SSE transport
(mcp_http_server.py) and the personal memory hub (memory_hub/server.py) used to
carry three hand-written copies of initialize / tools/list / tools/call
handling that had started to drift (error handling, resources support, invalid
requests). Each surface now only supplies its tools and, optionally, resources.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


DEFAULT_PROTOCOL_VERSION = "2024-11-05"

INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def result_message(message_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def error_message(message_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": message}}


def tool_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a JSON-serialisable tool payload as an MCP ``tools/call`` result."""
    return {
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)}],
        "structuredContent": payload,
        "isError": False,
    }


def tool_error(exc: BaseException) -> Dict[str, Any]:
    """Tool failures are reported in-band (``isError``) per the MCP spec."""
    return {"content": [{"type": "text", "text": str(exc)}], "isError": True}


@dataclass
class MCPSurface:
    """What one MCP server exposes; :meth:`dispatch` handles the protocol."""

    name: str
    version: str
    list_tools: Callable[[], List[Dict[str, Any]]]
    # Returns a finished ``tools/call`` result (use :func:`tool_result`).
    call_tool: Callable[[str, Dict[str, Any]], Dict[str, Any]]
    list_resources: Callable[[], List[Dict[str, Any]]] = field(default=lambda: [])
    # Returns ``resources/read`` contents for a known URI, or None if unknown.
    read_resource: Optional[Callable[[str], Optional[List[Dict[str, Any]]]]] = None
    on_initialize: Optional[Callable[[], None]] = None

    def capabilities(self) -> Dict[str, Any]:
        capabilities: Dict[str, Any] = {"tools": {}}
        if self.read_resource is not None:
            capabilities["resources"] = {}
        return capabilities

    def dispatch(self, message: Any) -> Optional[Dict[str, Any]]:
        """Handle one JSON-RPC message; ``None`` means "no response" (notification)."""
        if not isinstance(message, dict) or not isinstance(message.get("method"), str):
            message_id = message.get("id") if isinstance(message, dict) else None
            return error_message(message_id, INVALID_REQUEST, "Invalid Request")

        method = message["method"]
        message_id = message.get("id")
        is_notification = "id" not in message
        params = message.get("params") or {}
        if not isinstance(params, dict):
            return None if is_notification else error_message(
                message_id, INVALID_PARAMS, "params must be an object"
            )

        if method == "initialize":
            if self.on_initialize is not None:
                self.on_initialize()
            return result_message(
                message_id,
                {
                    "protocolVersion": params.get("protocolVersion", DEFAULT_PROTOCOL_VERSION),
                    "capabilities": self.capabilities(),
                    "serverInfo": {"name": self.name, "version": self.version},
                },
            )
        if method.startswith("notifications/"):
            return None
        if method == "ping":
            return result_message(message_id, {})
        if method == "tools/list":
            return result_message(message_id, {"tools": self.list_tools()})
        if method == "tools/call":
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, dict):
                return result_message(message_id, tool_error(ValueError("arguments must be an object")))
            try:
                result = self.call_tool(str(params.get("name", "")), arguments)
            except Exception as exc:
                result = tool_error(exc)
            return result_message(message_id, result)
        if method == "resources/list":
            return result_message(message_id, {"resources": self.list_resources()})
        if method == "resources/read" and self.read_resource is not None:
            uri = str(params.get("uri", ""))
            contents = self.read_resource(uri)
            if contents is None:
                return error_message(message_id, INVALID_PARAMS, f"Unknown resource: {uri}")
            return result_message(message_id, {"contents": contents})

        if is_notification:
            return None
        return error_message(message_id, METHOD_NOT_FOUND, f"Method not found: {method}")
