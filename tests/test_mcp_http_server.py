"""The HTTP MCP transport must stay responsive while a tool call runs."""

from __future__ import annotations

import asyncio
import threading
import unittest
from unittest import mock

import httpx

import mcp_http_server


TOOL_CALL = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {"name": "sync_knowledge_base", "arguments": {}},
}


def _client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=mcp_http_server.app)
    return httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8766")


class HTTPMCPServerTests(unittest.TestCase):
    def test_health_and_ping_answer_while_a_tool_call_blocks(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def slow_tool(name, arguments):
            started.set()
            release.wait(10)
            return {"content": [{"type": "text", "text": "done"}], "isError": False}

        async def scenario():
            async with _client() as client:
                call = asyncio.create_task(client.post("/mcp", json=TOOL_CALL))
                loop = asyncio.get_running_loop()
                self.assertTrue(await loop.run_in_executor(None, started.wait, 5))
                try:
                    health = await asyncio.wait_for(client.get("/health"), timeout=2)
                    ping = await asyncio.wait_for(
                        client.post("/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "ping"}),
                        timeout=2,
                    )
                finally:
                    release.set()
                return health, ping, await call

        with mock.patch.object(mcp_http_server.server.engine, "_dispatch_tool", slow_tool):
            health, ping, call = asyncio.run(scenario())

        self.assertEqual(health.status_code, 200)
        self.assertEqual(ping.json()["result"], {})
        self.assertEqual(call.json()["result"]["content"][0]["text"], "done")

    def test_tool_calls_run_one_at_a_time(self) -> None:
        active = 0
        peak = 0
        lock = threading.Lock()

        def tool(name, arguments):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            threading.Event().wait(0.05)
            with lock:
                active -= 1
            return {"content": [], "isError": False}

        async def scenario():
            async with _client() as client:
                await asyncio.gather(*(client.post("/mcp", json=TOOL_CALL) for _ in range(4)))

        with mock.patch.object(mcp_http_server.server.engine, "_dispatch_tool", tool):
            asyncio.run(scenario())
        self.assertEqual(peak, 1)

    def test_rejects_foreign_host_and_preflight_exempt_bodies(self) -> None:
        async def scenario():
            transport = httpx.ASGITransport(app=mcp_http_server.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://evil.example:8766") as evil:
                foreign = await evil.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
            async with _client() as client:
                plain = await client.post(
                    "/mcp",
                    content='{"jsonrpc": "2.0", "id": 1, "method": "ping"}',
                    headers={"Content-Type": "text/plain"},
                )
                ok = await client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
            return foreign, plain, ok

        foreign, plain, ok = asyncio.run(scenario())
        self.assertEqual(foreign.status_code, 403)
        self.assertEqual(plain.status_code, 415)
        self.assertEqual(ok.status_code, 200)


if __name__ == "__main__":
    unittest.main()
