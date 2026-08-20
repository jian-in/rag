from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "codex_memory_hook.py"
SPEC = importlib.util.spec_from_file_location("codex_memory_hook", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeClient:
    def __init__(self) -> None:
        self.context_calls: list[tuple[str, list[str], int, int, str]] = []
        self.proposals: list[dict[str, object]] = []

    def context(
        self,
        query: str,
        scopes: list[str],
        top_k: int,
        char_budget: int,
        retrieval_mode: str = "hybrid",
    ) -> dict[str, object]:
        self.context_calls.append((query, scopes, top_k, char_budget, retrieval_mode))
        return {
            "context": "[episodes] CURRENT_CONTEXT.md#记忆中枢\n已启用三域记忆平台。",
            "hits": [{"scope": "episodes"}],
        }

    def propose(
        self,
        scope: str,
        content: str,
        reason: str,
        source: str,
        session_id: str | None,
    ) -> dict[str, object]:
        candidate = {
            "id": f"candidate-{len(self.proposals) + 1}",
            "status": "pending",
            "scope": scope,
            "content": content,
            "reason": reason,
            "source": source,
            "session_id": session_id,
        }
        self.proposals.append(candidate)
        return candidate


class CodexMemoryHookTests(unittest.TestCase):
    def test_complex_prompt_injects_cited_memory_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            client = FakeClient()
            result = MODULE.handle_event(
                {
                    "hook_event_name": "UserPromptSubmit",
                    "session_id": "session-1",
                    "prompt": "继续上次的 SecondBrain 记忆中枢工作，把审核流程接完整。",
                },
                client,
                Path(temp),
            )

            output = result["hookSpecificOutput"]
            self.assertEqual(output["hookEventName"], "UserPromptSubmit")
            self.assertIn("CURRENT_CONTEXT.md", output["additionalContext"])
            self.assertIn("episodes", client.context_calls[0][1])
            self.assertEqual(client.context_calls[0][4], "local")
            self.assertTrue(any(Path(temp).iterdir()))

    def test_trivial_prompt_is_skipped_without_retrieval_or_spooling(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            client = FakeClient()
            result = MODULE.handle_event(
                {"hook_event_name": "UserPromptSubmit", "prompt": "谢谢"},
                client,
                Path(temp),
            )

            self.assertEqual(result, {})
            self.assertEqual(client.context_calls, [])
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_spool_write_failure_does_not_suppress_memory_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            client = FakeClient()
            with patch.object(MODULE, "_save_spool", side_effect=OSError("locked")):
                result = MODULE.handle_event(
                    {
                        "hook_event_name": "UserPromptSubmit",
                        "session_id": "session-locked",
                        "prompt": "继续完善个人记忆平台并检查已有项目决定。",
                    },
                    client,
                    Path(temp),
                )

            self.assertIn("hookSpecificOutput", result)
            self.assertEqual(len(client.context_calls), 1)

    def test_explicit_remember_request_creates_only_a_pending_profile_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            client = FakeClient()
            MODULE.handle_event(
                {
                    "hook_event_name": "UserPromptSubmit",
                    "session_id": "session-2",
                    "prompt": "记住：我偏好深海蓝且结构清晰的界面。",
                },
                client,
                Path(temp),
            )

            self.assertEqual(len(client.proposals), 1)
            self.assertEqual(client.proposals[0]["scope"], "profile")
            self.assertEqual(client.proposals[0]["status"], "pending")

    def test_stop_event_proposes_completed_task_outcome_then_clears_spool(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            client = FakeClient()
            MODULE.handle_event(
                {
                    "hook_event_name": "UserPromptSubmit",
                    "session_id": "session-3",
                    "prompt": "继续完善个人记忆平台的自动审核与召回功能。",
                },
                client,
                root,
            )
            result = MODULE.handle_event(
                {
                    "hook_event_name": "Stop",
                    "session_id": "session-3",
                    "last_assistant_message": (
                        "已完成个人记忆平台的审核链路，并通过全部核心测试。"
                        "Agent 现在只能提交候选，批准和忘记由本机观测台控制。"
                    ),
                },
                client,
                root,
            )

            self.assertEqual(result, {})
            self.assertEqual(client.proposals[-1]["scope"], "episodes")
            self.assertIn("审核链路", str(client.proposals[-1]["content"]))
            self.assertEqual(list(root.iterdir()), [])

    def test_secret_like_prompt_is_never_retrieved_spooled_or_proposed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            client = FakeClient()
            result = MODULE.handle_event(
                {
                    "hook_event_name": "UserPromptSubmit",
                    "session_id": "session-4",
                    "prompt": "请记住 API_KEY=sk-example-secret-value-123456",
                },
                client,
                root,
            )

            self.assertEqual(result, {})
            self.assertEqual(client.context_calls, [])
            self.assertEqual(client.proposals, [])
            self.assertEqual(list(root.iterdir()), [])

    def test_main_forces_utf8_for_windows_native_hook_pipes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            client = FakeClient()
            client.context = lambda *args, **kwargs: {
                "context": "✅ 本机记忆已召回。",
                "hits": [],
            }
            incoming = io.BytesIO(
                json.dumps(
                    {
                        "hook_event_name": "UserPromptSubmit",
                        "session_id": "session-utf8",
                        "prompt": "请根据我的偏好继续处理这个复杂项目。",
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
            )
            outgoing = io.BytesIO()
            stdin = io.TextIOWrapper(incoming, encoding="gbk")
            stdout = io.TextIOWrapper(outgoing, encoding="gbk")
            stderr = io.TextIOWrapper(io.BytesIO(), encoding="gbk")

            with (
                patch.object(MODULE.sys, "stdin", stdin),
                patch.object(MODULE.sys, "stdout", stdout),
                patch.object(MODULE.sys, "stderr", stderr),
                patch.object(MODULE, "MemoryClient", return_value=client),
                patch.object(MODULE, "_default_spool_root", return_value=Path(temp)),
            ):
                self.assertEqual(MODULE.main(), 0)
                stdout.flush()
                result = json.loads(outgoing.getvalue().decode("utf-8"))

            self.assertIn("✅", result["hookSpecificOutput"]["additionalContext"])


if __name__ == "__main__":
    unittest.main()
