from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from memory_hub import MemoryHub, MemoryHubSettings


class MemoryHubTests(unittest.TestCase):
    def test_scope_restricted_search_keeps_memory_domains_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            vault = root / "SecondBrain"
            memory = root / "memory"
            rollouts = root / "rollouts"

            note = vault / "01-Projects" / "Astra.md"
            note.parent.mkdir(parents=True)
            note.write_text("# Astra\n\n星图导航项目使用蓝色界面。", encoding="utf-8")

            memory.mkdir(parents=True)
            (memory / "USER_PROFILE.md").write_text(
                "# 用户档案\n\n阿飞偏爱莓果甜品。", encoding="utf-8"
            )
            (memory / "FEILI_SOUL.md").write_text(
                "# 助手身份\n\n绯梨做事可靠。", encoding="utf-8"
            )
            (memory / "CURRENT_CONTEXT.md").write_text(
                "# 当前项目\n\n正在实现个人记忆中枢。", encoding="utf-8"
            )
            (memory / "MEMORY_LOG.md").write_text(
                "# 记录\n\n已确认三库隔离。", encoding="utf-8"
            )
            rollouts.mkdir(parents=True)

            hub = MemoryHub(
                MemoryHubSettings(
                    index_root=root / "indexes",
                    vault_root=vault,
                    memory_root=memory,
                    rollout_root=rollouts,
                    vault_include_dirs=("01-Projects",),
                )
            )
            hub.sync()

            profile_hits = hub.search("莓果甜品", scopes=("profile",), top_k=5)

            self.assertTrue(profile_hits)
            self.assertEqual({hit.scope for hit in profile_hits}, {"profile"})
            self.assertNotIn("Astra.md", {Path(hit.source_path).name for hit in profile_hits})

    def test_sync_is_idempotent_and_removes_deleted_note_from_only_its_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            vault = root / "SecondBrain"
            memory = root / "memory"
            rollouts = root / "rollouts"
            note = vault / "03-Resources" / "检索设计.md"
            note.parent.mkdir(parents=True)
            note.write_text("# 检索设计\n\n稳定分块标识。", encoding="utf-8")
            memory.mkdir(parents=True)
            (memory / "USER_PROFILE.md").write_text("# 档案\n\n偏好蓝色。", encoding="utf-8")
            (memory / "FEILI_SOUL.md").write_text("# 身份\n\n可靠。", encoding="utf-8")
            (memory / "CURRENT_CONTEXT.md").write_text("# 上下文\n\n实现中。", encoding="utf-8")
            (memory / "MEMORY_LOG.md").write_text("# 日志\n\n开始。", encoding="utf-8")
            rollouts.mkdir(parents=True)
            hub = MemoryHub(
                MemoryHubSettings(
                    index_root=root / "indexes",
                    vault_root=vault,
                    memory_root=memory,
                    rollout_root=rollouts,
                    vault_include_dirs=("03-Resources",),
                )
            )

            first = hub.sync()
            second = hub.sync()
            profile_count = second["status"]["scopes"]["profile"]["chunk_count"]

            self.assertEqual(first["scopes"]["vault"]["added"], 1)
            self.assertEqual(second["scopes"]["vault"]["unchanged"], 1)

            note.unlink()
            removed = hub.sync(scopes=("vault",))

            self.assertEqual(removed["scopes"]["vault"]["removed"], 1)
            self.assertEqual(hub.search("稳定分块", scopes=("vault",)), [])
            self.assertEqual(hub.status()["scopes"]["profile"]["chunk_count"], profile_count)

    def test_optional_embedding_provider_recalls_semantically_related_memory(self) -> None:
        class FakeEmbeddingProvider:
            model_name = "fake-embedding-v1"

            def embed(self, texts: list[str]) -> list[list[float]]:
                vectors = []
                for text in texts:
                    if "休息" in text or "睡眠" in text:
                        vectors.append([1.0, 0.0])
                    else:
                        vectors.append([0.0, 1.0])
                return vectors

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            vault = root / "SecondBrain"
            note = vault / "02-Areas" / "健康.md"
            note.parent.mkdir(parents=True)
            note.write_text("# 健康\n\n睡眠不足时应当早点结束工作。", encoding="utf-8")
            memory = root / "memory"
            memory.mkdir()
            rollouts = root / "rollouts"
            rollouts.mkdir()
            hub = MemoryHub(
                MemoryHubSettings(
                    index_root=root / "indexes",
                    vault_root=vault,
                    memory_root=memory,
                    rollout_root=rollouts,
                    vault_include_dirs=("02-Areas",),
                ),
                embedding_provider=FakeEmbeddingProvider(),
            )
            hub.sync(scopes=("vault",))

            hits = hub.search("现在该休息吗", scopes=("vault",), top_k=3)

            self.assertTrue(hits)
            self.assertEqual(hits[0].title, "健康")

    def test_optional_reranker_reorders_federated_candidates(self) -> None:
        class FakeRerankProvider:
            model_name = "fake-reranker-v1"

            def rerank(
                self, query: str, documents: list[str], top_n: int
            ) -> list[tuple[int, float]]:
                preferred = next(index for index, text in enumerate(documents) if "第二候选" in text)
                rest = [index for index in range(len(documents)) if index != preferred]
                return [(preferred, 0.99)] + [
                    (index, 0.5 - position * 0.01) for position, index in enumerate(rest)
                ][: max(0, top_n - 1)]

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            vault = root / "SecondBrain" / "03-Resources"
            vault.mkdir(parents=True)
            (vault / "A.md").write_text("# A\n\n索引设计，第一候选。", encoding="utf-8")
            (vault / "B.md").write_text("# B\n\n索引设计，第二候选。", encoding="utf-8")
            memory = root / "memory"
            memory.mkdir()
            rollouts = root / "rollouts"
            rollouts.mkdir()
            hub = MemoryHub(
                MemoryHubSettings(
                    index_root=root / "indexes",
                    vault_root=root / "SecondBrain",
                    memory_root=memory,
                    rollout_root=rollouts,
                    vault_include_dirs=("03-Resources",),
                ),
                rerank_provider=FakeRerankProvider(),
            )
            hub.sync(scopes=("vault",))

            hits = hub.search("索引设计", scopes=("vault",), top_k=2)

            self.assertEqual(hits[0].title, "B")
            self.assertGreater(hits[0].score, hits[1].score)

    def test_context_respects_budget_and_keeps_obsidian_locator(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            vault = root / "Afei Vault" / "01-Projects"
            vault.mkdir(parents=True)
            (vault / "记忆中枢.md").write_text(
                "# 检索入口\n\n" + "Agent 可以按需召回这段项目背景。" * 30,
                encoding="utf-8",
            )
            memory = root / "memory"
            memory.mkdir()
            rollouts = root / "rollouts"
            rollouts.mkdir()
            hub = MemoryHub(
                MemoryHubSettings(
                    index_root=root / "indexes",
                    vault_root=root / "Afei Vault",
                    memory_root=memory,
                    rollout_root=rollouts,
                    vault_include_dirs=("01-Projects",),
                    chunk_size=180,
                    chunk_overlap=20,
                )
            )
            hub.sync(scopes=("vault",))

            bundle = hub.context(
                "Agent 项目背景", scopes=("vault",), top_k=5, char_budget=160
            )

            self.assertLessEqual(len(bundle["context"]), 160)
            self.assertTrue(bundle["hits"])
            self.assertIn("#检索入口", bundle["hits"][0]["locator"])
            self.assertTrue(
                bundle["hits"][0]["obsidian_uri"].startswith(
                    "obsidian://open?vault=Afei%20Vault&file="
                )
            )

    def test_chinese_search_prefers_meaningful_phrase_over_common_character_noise(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            memory = root / "memory"
            memory.mkdir()
            (memory / "USER_PROFILE.md").write_text(
                "# 视觉偏好\n\n偏爱深色界面布局与蓝色细节。", encoding="utf-8"
            )
            (memory / "CURRENT_CONTEXT.md").write_text(
                "# 噪声\n\n" + "我 的 什么 样 喜欢 " * 40,
                encoding="utf-8",
            )
            vault = root / "vault"
            vault.mkdir()
            rollouts = root / "rollouts"
            rollouts.mkdir()
            hub = MemoryHub(
                MemoryHubSettings(
                    index_root=root / "indexes",
                    vault_root=vault,
                    memory_root=memory,
                    rollout_root=rollouts,
                    vault_include_dirs=(),
                )
            )
            hub.sync(scopes=("profile", "episodes"))

            hits = hub.search("我喜欢什么样的界面", scopes=("profile", "episodes"), top_k=3)

            self.assertTrue(hits)
            self.assertEqual(hits[0].scope, "profile")
            self.assertEqual(hits[0].heading, "视觉偏好")

    def test_sync_rebuilds_derived_index_when_pipeline_version_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            memory = root / "memory"
            memory.mkdir()
            (memory / "USER_PROFILE.md").write_text(
                "# 视觉偏好\n\n偏爱深色界面布局。", encoding="utf-8"
            )
            vault = root / "vault"
            vault.mkdir()
            rollouts = root / "rollouts"
            rollouts.mkdir()
            index_root = root / "indexes"
            hub = MemoryHub(
                MemoryHubSettings(
                    index_root=index_root,
                    vault_root=vault,
                    memory_root=memory,
                    rollout_root=rollouts,
                    vault_include_dirs=(),
                )
            )
            first = hub.sync(scopes=("profile",))
            self.assertEqual(first["scopes"]["profile"]["added"], 1)

            connection = sqlite3.connect(index_root / "profile.sqlite3")
            try:
                connection.execute(
                    "UPDATE metadata SET value = 'legacy-pipeline' WHERE key = 'pipeline_version'"
                )
                connection.commit()
            finally:
                connection.close()

            rebuilt = hub.sync(scopes=("profile",))

            self.assertEqual(rebuilt["scopes"]["profile"]["added"], 1)
            self.assertEqual(rebuilt["scopes"]["profile"]["unchanged"], 0)
            self.assertTrue(hub.search("深色界面", scopes=("profile",)))

    def test_project_heading_outranks_generic_progress_words(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            memory = root / "memory"
            memory.mkdir()
            (memory / "CURRENT_CONTEXT.md").write_text(
                "# SecondBrain RAG MCP\n\n增量同步已经完成。", encoding="utf-8"
            )
            (memory / "MEMORY_LOG.md").write_text(
                "# 日志\n\n" + "当前 进展 下一步 继续 " * 80, encoding="utf-8"
            )
            vault_note = root / "vault" / "03-Resources" / "进程.md"
            vault_note.parent.mkdir(parents=True)
            vault_note.write_text(
                "# 当前进程\n\nCPU 切换当前进程，时序电路具有记忆功能。",
                encoding="utf-8",
            )
            rollouts = root / "rollouts"
            rollouts.mkdir()
            hub = MemoryHub(
                MemoryHubSettings(
                    index_root=root / "indexes",
                    vault_root=root / "vault",
                    memory_root=memory,
                    rollout_root=rollouts,
                    vault_include_dirs=("03-Resources",),
                )
            )
            hub.sync(scopes=("episodes", "vault"))

            hits = hub.search(
                "SecondBrain RAG 记忆中枢 当前进展 下一步",
                scopes=("episodes", "vault"),
                top_k=3,
            )

            self.assertTrue(hits)
            self.assertEqual(hits[0].heading, "SecondBrain RAG MCP")
            self.assertEqual({hit.scope for hit in hits}, {"episodes"})


if __name__ == "__main__":
    unittest.main()
