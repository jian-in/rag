from __future__ import annotations

import importlib.util
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "memory_client.py"
SPEC = importlib.util.spec_from_file_location("personal_memory_client", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send({"ok": True, "server": "feili-personal-memory", "version": "test"})
        elif self.path == "/api/status":
            self._send({"ready": True, "scopes": {}})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        if self.path == "/api/context":
            self._send(
                {
                    "query": payload["query"],
                    "context": "[profile] USER_PROFILE.md#视觉偏好\n偏爱深色界面。",
                    "hits": [{"scope": "profile", "locator": "USER_PROFILE.md#视觉偏好"}],
                    "char_budget": payload["char_budget"],
                }
            )
        elif self.path == "/api/search":
            self._send({"query": payload["query"], "result_count": 0, "results": []})
        elif self.path == "/api/sync":
            self._send({"scopes": payload.get("scopes")})
        elif self.path == "/api/candidates/propose":
            self._send({"candidate": {"id": "candidate-1", "status": "pending", **payload}})
        else:
            self.send_error(404)

    def _send(self, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class MemoryClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.endpoint = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_context_uses_explicit_scope_and_budget(self) -> None:
        client = MODULE.MemoryClient(self.endpoint)
        result = client.context("界面偏好", ["profile"], 3, 1200)

        self.assertEqual(result["hits"][0]["scope"], "profile")
        self.assertEqual(result["char_budget"], 1200)

    def test_non_loopback_endpoint_is_rejected(self) -> None:
        with self.assertRaises(MODULE.ClientError):
            MODULE.MemoryClient("https://example.com")

    def test_empty_search_result_adds_warning(self) -> None:
        result = MODULE.MemoryClient(self.endpoint).search("不存在", ["vault"], 3)
        self.assertTrue(result["_warnings"])

    def test_health_verifies_service_identity(self) -> None:
        result = MODULE.MemoryClient(self.endpoint).health()
        self.assertEqual(result["server"], "feili-personal-memory")

    def test_propose_only_creates_a_pending_review_candidate(self) -> None:
        result = MODULE.MemoryClient(self.endpoint).propose(
            "profile",
            "阿飞偏好可审计的自动化。",
            "稳定工作偏好。",
            "test-agent",
            "session-1",
        )
        self.assertEqual(result["status"], "pending")
        self.assertEqual(result["scope"], "profile")


if __name__ == "__main__":
    unittest.main()
