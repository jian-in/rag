from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from memory_hub import MemoryHub, MemoryHubSettings


class MemoryReviewTests(unittest.TestCase):
    def _hub(self, root: Path) -> MemoryHub:
        vault = root / "SecondBrain"
        memory = root / "memory"
        rollouts = root / "rollouts"
        vault.mkdir()
        memory.mkdir()
        rollouts.mkdir()
        (memory / "USER_PROFILE.md").write_text(
            "# 用户档案\n\n已有内容。\n", encoding="utf-8"
        )
        (memory / "FEILI_SOUL.md").write_text("# 绯梨\n\n可靠。\n", encoding="utf-8")
        (memory / "CURRENT_CONTEXT.md").write_text(
            "# 当前上下文\n\n已有项目。\n", encoding="utf-8"
        )
        (memory / "MEMORY_LOG.md").write_text("# 记忆日志\n", encoding="utf-8")
        return MemoryHub(
            MemoryHubSettings(
                index_root=root / "indexes",
                vault_root=vault,
                memory_root=memory,
                rollout_root=rollouts,
                vault_include_dirs=(),
            )
        )

    def test_agent_proposal_is_pending_and_deduplicated_without_touching_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hub = self._hub(root)
            profile_path = root / "memory" / "USER_PROFILE.md"
            before = profile_path.read_text(encoding="utf-8")

            first = hub.propose_memory(
                scope="profile",
                content="阿飞偏好蓝色且结构清晰的界面。",
                reason="用户明确表达了稳定偏好。",
                source="codex-stop-hook",
                session_id="session-1",
            )
            second = hub.propose_memory(
                scope="profile",
                content="阿飞偏好蓝色且结构清晰的界面。",
                reason="重复观察。",
                source="agent",
                session_id="session-1",
            )

            pending = hub.review_memories(status="pending")

            self.assertEqual(first["id"], second["id"])
            self.assertEqual(first["status"], "pending")
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0]["scope"], "profile")
            self.assertEqual(profile_path.read_text(encoding="utf-8"), before)

    def test_secret_like_content_is_rejected_before_it_reaches_the_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            hub = self._hub(Path(temp))

            with self.assertRaisesRegex(ValueError, "secret"):
                hub.propose_memory(
                    scope="profile",
                    content="SILICONFLOW_API_KEY=sk-this-must-never-be-stored",
                    source="codex-stop-hook",
                )

            self.assertEqual(hub.review_memories(status="pending"), [])

    def test_approval_can_correct_content_then_sync_it_into_the_selected_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hub = self._hub(root)
            candidate = hub.propose_memory(
                scope="profile",
                content="阿飞偏好绿色界面。",
                reason="待人工纠正颜色。",
                source="agent",
            )

            approved = hub.review_memory(
                str(candidate["id"]),
                "approve",
                scope="profile",
                content="阿飞偏好深海蓝且结构清晰的界面，内部代号记忆舟。",
                note="已按用户实际偏好纠正。",
            )

            profile = (root / "memory" / "USER_PROFILE.md").read_text(encoding="utf-8")
            self.assertEqual(approved["status"], "approved")
            self.assertIn(f"<!-- feili-memory:{candidate['id']}:start -->", profile)
            self.assertIn("深海蓝且结构清晰", profile)
            self.assertNotIn("偏好绿色", profile)
            hits = hub.search("记忆舟", scopes=("profile",), top_k=3)
            self.assertTrue(any("记忆舟" in hit.preview for hit in hits))

    def test_rejection_is_audited_without_changing_any_memory_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hub = self._hub(root)
            profile_path = root / "memory" / "USER_PROFILE.md"
            before = profile_path.read_text(encoding="utf-8")
            candidate = hub.propose_memory(
                scope="profile",
                content="这条推断并不可靠，不应该进入长期档案。",
                source="agent",
            )

            rejected = hub.review_memory(
                str(candidate["id"]), "reject", note="只是一次性推断。"
            )

            self.assertEqual(rejected["status"], "rejected")
            self.assertEqual(profile_path.read_text(encoding="utf-8"), before)
            event_types = [
                event["event_type"]
                for event in hub.memory_review_history(str(candidate["id"]))
            ]
            self.assertEqual(event_types, ["proposed", "rejected"])

    def test_forget_removes_only_the_managed_block_and_refreshes_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hub = self._hub(root)
            candidate = hub.propose_memory(
                scope="profile",
                content="用于忘记测试的唯一口令是珊瑚纸鸢。",
                source="agent",
            )
            hub.review_memory(str(candidate["id"]), "approve")
            self.assertTrue(hub.search("珊瑚纸鸢", scopes=("profile",), top_k=3))

            forgotten = hub.review_memory(
                str(candidate["id"]), "forget", note="用户要求忘记。"
            )

            profile = (root / "memory" / "USER_PROFILE.md").read_text(encoding="utf-8")
            self.assertEqual(forgotten["status"], "forgotten")
            self.assertIn("已有内容", profile)
            self.assertNotIn("珊瑚纸鸢", profile)
            self.assertFalse(hub.search("珊瑚纸鸢", scopes=("profile",), top_k=3))

    def test_vault_approval_uses_one_explicit_inbox_without_broadening_index_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hub = self._hub(root)
            unrelated = root / "SecondBrain" / "Private" / "unindexed.md"
            unrelated.parent.mkdir()
            unrelated.write_text("这份私人内容不应进入索引：灰色鲸落。", encoding="utf-8")
            candidate = hub.propose_memory(
                scope="vault",
                content="可沉淀知识的唯一短语是琥珀风铃。",
                source="agent",
            )

            approved = hub.review_memory(str(candidate["id"]), "approve")

            expected = root / "SecondBrain" / "00-Inbox" / "Agent Memory Inbox.md"
            self.assertEqual(Path(str(approved["target_path"])), expected)
            self.assertTrue(expected.is_file())
            self.assertTrue(hub.search("琥珀风铃", scopes=("vault",), top_k=3))
            self.assertFalse(hub.search("灰色鲸落", scopes=("vault",), top_k=3))

    def test_failed_approval_sync_rolls_back_source_and_can_be_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hub = self._hub(root)
            candidate = hub.propose_memory(
                scope="profile",
                content="故障恢复测试口令是星屿回声。",
                source="agent",
            )

            with patch.object(hub, "sync", side_effect=RuntimeError("sync failed")):
                with self.assertRaisesRegex(RuntimeError, "sync failed"):
                    hub.review_memory(str(candidate["id"]), "approve")

            profile_path = root / "memory" / "USER_PROFILE.md"
            self.assertNotIn("星屿回声", profile_path.read_text(encoding="utf-8"))
            self.assertEqual(
                hub.review_memories(status="pending")[0]["id"], candidate["id"]
            )
            retried = hub.review_memory(str(candidate["id"]), "approve")
            self.assertEqual(retried["status"], "approved")


if __name__ == "__main__":
    unittest.main()
