"""Shared JSON-RPC dispatch used by all three MCP surfaces."""

from __future__ import annotations

import unittest

import mcp_protocol
from mcp_protocol import MCPSurface


def _surface(**overrides) -> MCPSurface:
    calls = []

    def call_tool(name, arguments):
        calls.append((name, arguments))
        if name == "boom":
            raise RuntimeError("tool exploded")
        return mcp_protocol.tool_result({"echo": arguments})

    options = dict(
        name="test-server",
        version="9.9",
        list_tools=lambda: [{"name": "echo"}],
        call_tool=call_tool,
    )
    options.update(overrides)
    surface = MCPSurface(**options)
    surface.calls = calls
    return surface


class MCPProtocolTests(unittest.TestCase):
    def test_initialize_reports_server_and_capabilities(self) -> None:
        initialized = []
        surface = _surface(on_initialize=lambda: initialized.append(True))
        response = surface.dispatch(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}}
        )
        self.assertEqual(response["result"]["serverInfo"], {"name": "test-server", "version": "9.9"})
        self.assertEqual(response["result"]["protocolVersion"], "2025-03-26")
        self.assertEqual(response["result"]["capabilities"], {"tools": {}})
        self.assertEqual(initialized, [True])

    def test_notifications_get_no_response(self) -> None:
        surface = _surface()
        self.assertIsNone(surface.dispatch({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        self.assertIsNone(surface.dispatch({"jsonrpc": "2.0", "method": "notifications/cancelled"}))
        self.assertIsNone(surface.dispatch({"jsonrpc": "2.0", "method": "unknown/thing"}))

    def test_tool_calls_wrap_results_and_report_errors_in_band(self) -> None:
        surface = _surface()
        ok = surface.dispatch(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "echo", "arguments": {"a": 1}}}
        )
        self.assertEqual(ok["result"]["structuredContent"], {"echo": {"a": 1}})
        self.assertFalse(ok["result"]["isError"])

        failed = surface.dispatch(
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "boom"}}
        )
        self.assertTrue(failed["result"]["isError"])
        self.assertIn("tool exploded", failed["result"]["content"][0]["text"])

        bad_args = surface.dispatch(
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "echo", "arguments": [1]}}
        )
        self.assertTrue(bad_args["result"]["isError"])
        self.assertEqual(len(surface.calls), 2)

    def test_invalid_requests_and_unknown_methods(self) -> None:
        surface = _surface()
        for message in ([], "x", {"id": 5}, {"id": 6, "method": 7}):
            with self.subTest(message=message):
                response = surface.dispatch(message)
                self.assertEqual(response["error"]["code"], mcp_protocol.INVALID_REQUEST)
        missing = surface.dispatch({"jsonrpc": "2.0", "id": 8, "method": "nope"})
        self.assertEqual(missing["error"]["code"], mcp_protocol.METHOD_NOT_FOUND)
        bad_params = surface.dispatch({"jsonrpc": "2.0", "id": 9, "method": "ping", "params": [1]})
        self.assertEqual(bad_params["error"]["code"], mcp_protocol.INVALID_PARAMS)

    def test_resources_are_optional(self) -> None:
        plain = _surface()
        self.assertEqual(plain.dispatch({"id": 1, "method": "resources/list"})["result"], {"resources": []})
        self.assertEqual(
            plain.dispatch({"id": 2, "method": "resources/read", "params": {"uri": "x://y"}})["error"]["code"],
            mcp_protocol.METHOD_NOT_FOUND,
        )

        with_resources = _surface(
            list_resources=lambda: [{"uri": "x://y"}],
            read_resource=lambda uri: [{"uri": uri, "text": "hi"}] if uri == "x://y" else None,
        )
        self.assertIn("resources", with_resources.capabilities())
        read = with_resources.dispatch({"id": 3, "method": "resources/read", "params": {"uri": "x://y"}})
        self.assertEqual(read["result"]["contents"][0]["text"], "hi")
        unknown = with_resources.dispatch({"id": 4, "method": "resources/read", "params": {"uri": "x://z"}})
        self.assertEqual(unknown["error"]["code"], mcp_protocol.INVALID_PARAMS)

    def test_all_three_servers_share_the_dispatcher(self) -> None:
        import tempfile
        from pathlib import Path

        import mcp_http_server
        from fastapi.testclient import TestClient
        from memory_hub import MemoryHub, MemoryHubSettings
        from memory_hub.server import create_app
        from mcp_server import MCPServer

        stdio = MCPServer()
        self.assertEqual(
            stdio._surface.dispatch({"id": 1, "method": "tools/list"})["result"]["tools"],
            stdio._tool_definitions(),
        )
        self.assertEqual(
            mcp_http_server.server.handle_message({"id": 1, "method": "ping"}),
            {"jsonrpc": "2.0", "id": 1, "result": {}},
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hub = MemoryHub(
                MemoryHubSettings(
                    index_root=root / "i", vault_root=root / "v", memory_root=root / "m", rollout_root=root / "r"
                )
            )
            client = TestClient(create_app(hub), base_url="http://127.0.0.1")
            read = client.post(
                "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "resources/read", "params": {"uri": "memory://entry"}}
            ).json()
            self.assertIn("instruction", read["result"]["contents"][0]["text"])
            batch = client.post("/mcp", json=[{"jsonrpc": "2.0", "id": 1, "method": "ping"}]).json()
            self.assertEqual(batch["error"]["code"], mcp_protocol.INVALID_REQUEST)


if __name__ == "__main__":
    unittest.main()
