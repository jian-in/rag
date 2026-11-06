"""Evaluator without ragas, LangGraph label parsing, memory window, env toggles, secondbrain_rag."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from langchain_core.documents import Document

from chain.langgraph_workflow import parse_label
from evaluation import evaluator as evaluator_module
from evaluation.evaluator import RAGEvaluator
from memory_hub.providers import providers_from_environment
from retrieval.conversation_memory import ConversationMemory


class ScriptedJudge:
    """Fake judge LLM: picks a canned JSON reply by the metric's prompt wording."""

    def __init__(self, replies: dict, fail: tuple = ()):
        self.replies = replies
        self.fail = fail
        self.prompts: list[str] = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        for marker, reply in self.replies.items():
            if marker in prompt:
                if marker in self.fail:
                    return SimpleNamespace(content="抱歉，我无法判断。")
                return SimpleNamespace(content=reply)
        raise AssertionError(f"unexpected prompt: {prompt[:80]}")


REPLIES = {
    "事实核查员": json.dumps(
        {"statements": [{"statement": "a", "supported": True}, {"statement": "b", "supported": True},
                        {"statement": "c", "supported": False}]}
    ),
    "是否直接、完整地回应": '```json\n{"score": 8, "reason": "基本切题"}\n```',
    "对得出【标准答案】是否有用": json.dumps(
        {"verdicts": [{"index": 1, "useful": True}, {"index": 2, "useful": "false"}, {"index": 3, "useful": True}]}
    ),
    "逐句判断能否在【上下文】中找到依据": json.dumps(
        {"sentences": [{"sentence": "x", "attributed": True}, {"sentence": "y", "attributed": False}]}
    ),
}


class EvaluatorTests(unittest.TestCase):
    def _evaluator(self, **judge_kwargs) -> tuple[RAGEvaluator, ScriptedJudge]:
        judge = ScriptedJudge(REPLIES, **judge_kwargs)
        evaluator = RAGEvaluator(llm=judge)
        evaluator.add_sample("问题", "标准答案", contexts=["c1", "c2", "c3"], answer="答案")
        return evaluator, judge

    def test_metrics_are_computed_from_judge_verdicts(self) -> None:
        evaluator, _ = self._evaluator()
        results = evaluator.run_evaluation(metrics=list(evaluator_module.METRICS))

        self.assertAlmostEqual(results["faithfulness"], 2 / 3)
        self.assertAlmostEqual(results["answer_relevancy"], 0.8)
        # Average precision for verdicts [1, 0, 1]: (1/1 + 2/3) / 2
        self.assertAlmostEqual(results["context_precision"], (1 + 2 / 3) / 2)
        self.assertAlmostEqual(results["context_recall"], 0.5)
        self.assertEqual(evaluator.errors, [])
        # main.run_quick_eval iterates results.items() expecting plain floats.
        self.assertTrue(all(isinstance(v, float) for v in results.values()))

    def test_unparseable_verdict_is_reported_not_fatal(self) -> None:
        evaluator, _ = self._evaluator(fail=("事实核查员",))
        results = evaluator.run_evaluation(metrics=["faithfulness", "answer_relevancy"])

        self.assertNotIn("faithfulness", results)
        self.assertAlmostEqual(results["answer_relevancy"], 0.8)
        self.assertEqual(len(evaluator.errors), 1)
        self.assertIn("faithfulness", evaluator.errors[0])
        self.assertIsNone(evaluator.sample_scores[0]["faithfulness"])

    def test_empty_contexts_are_filled_from_the_qa_chain(self) -> None:
        judge = ScriptedJudge(REPLIES)
        evaluator = RAGEvaluator(llm=judge)
        evaluator.add_sample("问题", "标准答案", contexts=[], answer="")
        qa_chain = mock.Mock()
        qa_chain.answer.return_value = ("生成的答案", [Document(page_content="检索到的片段")])

        evaluator.run_evaluation(qa_chain=qa_chain, metrics=["faithfulness"])

        qa_chain.answer.assert_called_once_with("问题")
        self.assertEqual(evaluator.eval_samples[0]["contexts"], ["检索到的片段"])
        self.assertEqual(evaluator.eval_samples[0]["answer"], "生成的答案")
        self.assertIn("检索到的片段", judge.prompts[0])

    def test_unknown_metric_is_rejected(self) -> None:
        evaluator, _ = self._evaluator()
        with self.assertRaises(ValueError):
            evaluator.run_evaluation(metrics=["bleu"])

    def test_default_judge_uses_the_project_llm_endpoint(self) -> None:
        with mock.patch.object(evaluator_module.config, "OPENAI_API_BASE", "https://gateway.example/v1"), \
                mock.patch("langchain_openai.ChatOpenAI") as chat:
            RAGEvaluator().llm
        self.assertEqual(chat.call_args.kwargs["openai_api_base"], "https://gateway.example/v1")

    def test_ragas_is_no_longer_imported(self) -> None:
        source = Path(evaluator_module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from ragas", source)
        self.assertNotIn("from datasets", source)


class ParseLabelTests(unittest.TestCase):
    QUALITY = (("poor", ("poor", "not good", "bad", "不好", "不够", "不充分", "差")),
               ("good", ("good", "好", "充分")))
    QUERY_TYPE = (("multi-step", ("multi-step", "multi step", "multistep", "多步")),
                  ("complex", ("complex", "复杂")),
                  ("simple", ("simple", "简单")))

    def test_quality_labels_tolerate_punctuation_and_prose(self) -> None:
        cases = {
            "poor": "poor", "Poor.": "poor", "poor。": "poor", "答案质量：poor": "poor",
            "Not good enough": "poor", "答案不够充分": "poor",
            "good": "good", "Good!": "good", "答案充分": "good",
            "": "good", "无法判断": "good", "goodness": "good",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_label(text, self.QUALITY, default="good"), expected)

    def test_query_type_prefers_the_most_specific_label(self) -> None:
        cases = {
            "complex": "complex", "Complex。": "complex", "multi-step": "multi-step",
            "Multi-step reasoning, not just complex": "multi-step", "simple": "simple",
            "simplest": "simple", "这是一个复杂问题": "complex", "???": "simple",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_label(text, self.QUERY_TYPE, default="simple"), expected)


class ConversationWindowTests(unittest.TestCase):
    def _memory(self, max_turns: int, llm) -> ConversationMemory:
        memory = ConversationMemory.__new__(ConversationMemory)
        memory.max_turns = max_turns
        from collections import deque

        memory.history = deque(maxlen=max_turns)
        memory.summary = ""
        memory.llm = llm
        return memory

    def test_full_window_is_summarised_instead_of_silently_evicted(self) -> None:
        llm = mock.Mock()
        llm.invoke.return_value = SimpleNamespace(content="早期讨论了第0轮到第1轮")
        memory = self._memory(4, llm)
        for index in range(5):
            memory.add_turn(f"问题{index}", f"回答{index}")

        llm.invoke.assert_called_once()
        self.assertIn("问题0", llm.invoke.call_args.args[0][0].content)
        self.assertEqual(memory.summary, "早期讨论了第0轮到第1轮")
        self.assertEqual([turn["user"] for turn in memory.history], ["问题2", "问题3", "问题4"])

    def test_llm_failure_falls_back_to_truncated_summary(self) -> None:
        llm = mock.Mock()
        llm.invoke.side_effect = TimeoutError("rate limited")
        memory = self._memory(4, llm)
        for index in range(5):
            memory.add_turn(f"问题{index}", f"回答{index}")

        self.assertIn("问题0", memory.summary)
        self.assertIn("问题1", memory.summary)
        self.assertEqual(len(memory.history), 3)
        self.assertIn("问题0", memory.get_context())


class EnvToggleTests(unittest.TestCase):
    def test_blank_toggle_means_unset(self) -> None:
        embedding, reranker = providers_from_environment(
            {"SILICONFLOW_API_KEY": "shared", "MEMORY_VECTOR_ENABLED": "", "MEMORY_RERANK_ENABLED": "  "}
        )
        self.assertIsNone(embedding)
        self.assertIsNone(reranker)

        embedding, _ = providers_from_environment(
            {"SILICONFLOW_API_KEY": "shared", "MEMORY_VECTOR_ENABLED": "true"}
        )
        self.assertIsNotNone(embedding)

        embedding, _ = providers_from_environment(
            {"MEMORY_EMBEDDING_API_KEY": "dedicated", "MEMORY_VECTOR_ENABLED": ""}
        )
        self.assertIsNotNone(embedding)  # a dedicated key still opts in by default


class SecondBrainScriptTests(unittest.TestCase):
    def test_scan_keys_are_relative_to_the_vault_root(self) -> None:
        import secondbrain_rag

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for folder in ("02-Areas", "03-Resources"):
                (root / folder).mkdir()
                (root / folder / "same.md").write_text(f"# {folder}", encoding="utf-8")
            (root / "02-Areas" / ".hidden").mkdir()
            (root / "02-Areas" / ".hidden" / "skip.md").write_text("x", encoding="utf-8")

            manifest = secondbrain_rag.scan_files(
                [str(root / "02-Areas"), str(root / "03-Resources"), str(root / "missing")], str(root)
            )

        self.assertEqual(
            sorted(manifest),
            sorted([os.path.join("02-Areas", "same.md"), os.path.join("03-Resources", "same.md")]),
        )

    def test_paths_come_from_config_not_the_old_machine_location(self) -> None:
        import secondbrain_rag

        self.assertNotIn("Backup", secondbrain_rag.CHROMA_DIR)
        self.assertTrue(os.path.isabs(secondbrain_rag.CHROMA_DIR))


if __name__ == "__main__":
    unittest.main()
