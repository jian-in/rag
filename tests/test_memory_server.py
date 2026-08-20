from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from memory_hub import MemoryHub, MemoryHubSettings
from memory_hub.server import MemoryMCP, create_app


class MemoryServerTests(unittest.TestCase):
    def test_mcp_exposes_one_agent_entry_for_all_memory_operations(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            vault = root / "vault"
            memory = root / "memory"
            rollouts = root / "rollouts"
            vault.mkdir()
            memory.mkdir()
            rollouts.mkdir()
            (memory / "USER_PROFILE.md").write_text(
                "# 用户档案\n\n阿飞偏好清脆音乐。", encoding="utf-8"
            )
            hub = MemoryHub(
                MemoryHubSettings(
                    index_root=root / "indexes",
                    vault_root=vault,
                    memory_root=memory,
                    rollout_root=rollouts,
                    vault_include_dirs=(),
                )
            )
            hub.sync()
            mcp = MemoryMCP(hub)

            names = {tool["name"] for tool in mcp.tool_definitions()}
            result = mcp.call_tool(
                "memory_search",
                {"query": "清脆音乐", "scopes": ["profile"], "top_k": 3},
            )

            self.assertEqual(
                names,
                {
                    "memory_status",
                    "memory_sync",
                    "memory_search",
                    "memory_context",
                    "memory_propose",
                },
            )
            self.assertEqual(result["result_count"], 1)
            self.assertEqual(result["results"][0]["scope"], "profile")

    def test_http_app_serves_health_manifest_and_streamable_mcp(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            vault = root / "vault"
            memory = root / "memory"
            rollouts = root / "rollouts"
            vault.mkdir()
            memory.mkdir()
            rollouts.mkdir()
            (memory / "USER_PROFILE.md").write_text(
                "# 用户档案\n\n喜欢设计精巧的小东西。", encoding="utf-8"
            )
            hub = MemoryHub(
                MemoryHubSettings(
                    index_root=root / "indexes",
                    vault_root=vault,
                    memory_root=memory,
                    rollout_root=rollouts,
                    vault_include_dirs=(),
                )
            )
            hub.sync()
            client = TestClient(create_app(hub))

            health = client.get("/health")
            manifest = client.get("/.well-known/agent-memory.json")
            tools = client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            )
            search = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "memory_search",
                        "arguments": {"query": "精巧", "scopes": ["profile"]},
                    },
                },
            )

            self.assertTrue(health.json()["ok"])
            self.assertEqual(manifest.json()["mcp_endpoint"], "/mcp")
            self.assertEqual(len(tools.json()["result"]["tools"]), 5)
            self.assertEqual(
                search.json()["result"]["structuredContent"]["result_count"], 1
            )

            dashboard = client.get("/")
            stylesheet = client.get("/assets/app.css")
            self.assertIn("绯梨记忆观测台", dashboard.text)
            self.assertIn("候选记忆审核", dashboard.text)
            self.assertEqual(stylesheet.status_code, 200)

    def test_agent_can_only_propose_while_dashboard_api_controls_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            vault = root / "vault"
            memory = root / "memory"
            rollouts = root / "rollouts"
            vault.mkdir()
            memory.mkdir()
            rollouts.mkdir()
            (memory / "USER_PROFILE.md").write_text("# 用户档案\n", encoding="utf-8")
            hub = MemoryHub(
                MemoryHubSettings(
                    index_root=root / "indexes",
                    vault_root=vault,
                    memory_root=memory,
                    rollout_root=rollouts,
                    vault_include_dirs=(),
                )
            )
            mcp = MemoryMCP(hub)
            proposed = mcp.call_tool(
                "memory_propose",
                {
                    "scope": "profile",
                    "content": "阿飞偏好可审计的自动化流程。",
                    "reason": "用户表达稳定工作偏好。",
                    "source": "test-agent",
                },
            )
            self.assertEqual(proposed["status"], "pending")
            with self.assertRaisesRegex(Exception, "unknown memory tool"):
                mcp.call_tool("memory_approve", {"id": proposed["id"]})

            client = TestClient(create_app(hub))
            pending = client.get("/api/candidates", params={"status": "pending"})
            approved = client.post(
                f"/api/candidates/{proposed['id']}/decision",
                json={"action": "approve", "note": "人工确认。"},
            )

            self.assertEqual(pending.status_code, 200)
            self.assertEqual(len(pending.json()["candidates"]), 1)
            self.assertEqual(approved.status_code, 200)
            self.assertEqual(approved.json()["candidate"]["status"], "approved")
            self.assertIn(
                "可审计的自动化流程",
                (memory / "USER_PROFILE.md").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
