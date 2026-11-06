"""Follow-ups to the chunking / rewriting / encoding fixes (06a3645, 02bc823)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ingestion.incremental_sync import _pipeline_signature
from ingestion.loader import DocumentLoader
from ingestion.splitter import TextSplitter
from retrieval.conversation_memory import ConversationMemory
from retrieval.query_rewriter import QueryRewriter, strip_list_marker


class PipelineSignatureTests(unittest.TestCase):
    def test_signature_changes_when_chunking_rules_change(self) -> None:
        baseline = TextSplitter(chunk_size=100, chunk_overlap=10)
        changed = TextSplitter(chunk_size=100, chunk_overlap=10)
        changed.separators = ["\n\n", "\n", " ", ""]
        kept = TextSplitter(chunk_size=100, chunk_overlap=10)
        kept.keep_separator = "start"

        self.assertEqual(_pipeline_signature(baseline), _pipeline_signature(TextSplitter(100, 10)))
        self.assertNotEqual(_pipeline_signature(baseline), _pipeline_signature(changed))
        self.assertNotEqual(_pipeline_signature(baseline), _pipeline_signature(kept))
        self.assertIn("separators=", _pipeline_signature(baseline))


class ListMarkerTests(unittest.TestCase):
    def test_strips_real_list_markers(self) -> None:
        cases = {
            "1. 什么是RAG": "什么是RAG",
            "1.什么是RAG": "什么是RAG",
            "2、优点": "优点",
            "3) 缺点": "缺点",
            "4）场景": "场景",
            "(1) 第一问": "第一问",
            "（2）第二问": "第二问",
            "一、第一问": "第一问",
            "十二. 第十二问": "第十二问",
            "- 场景": "场景",
            "* 时间片轮转": "时间片轮转",
            "• 要点": "要点",
            "- 1. 嵌套编号": "嵌套编号",
            "  10. 带缩进  ": "带缩进",
        }
        for line, expected in cases.items():
            with self.subTest(line=line):
                self.assertEqual(strip_list_marker(line), expected)

    def test_leaves_question_text_alone(self) -> None:
        for line in (
            "1.5倍杠杆的风险是什么",
            "3 个场景分别是什么",
            "2024年考研政策是什么",
            "-1 的平方根是什么",
            "**重点** 是什么",
            "第一问是什么",
        ):
            with self.subTest(line=line):
                self.assertEqual(strip_list_marker(line), line)


class RewriteDegradationTests(unittest.TestCase):
    def _rewriter(self) -> QueryRewriter:
        rewriter = QueryRewriter.__new__(QueryRewriter)
        rewriter.llm = mock.Mock()
        return rewriter

    def test_unknown_strategy_is_a_caller_error_not_a_silent_fallback(self) -> None:
        rewriter = self._rewriter()
        with self.assertRaises(ValueError):
            rewriter.rewrite("问题", strategy="bogus")
        rewriter.llm.invoke.assert_not_called()

    def test_llm_failure_still_falls_back(self) -> None:
        rewriter = self._rewriter()
        rewriter.llm.invoke.side_effect = TimeoutError("gateway timeout")
        for strategy in ("expand", "decompose", "hyde"):
            with self.subTest(strategy=strategy):
                self.assertEqual(rewriter.rewrite("问题", strategy=strategy), ["问题"])

    def test_history_rewrite_falls_back_to_original_question(self) -> None:
        memory = ConversationMemory.__new__(ConversationMemory)
        memory.history = [{"user": "什么是RAG", "assistant": "检索增强生成"}]
        memory.summary = ""
        memory.get_context = lambda: "用户: 什么是RAG\n助手: 检索增强生成"
        memory.llm = mock.Mock()

        memory.llm.invoke.side_effect = TimeoutError("rate limited")
        self.assertEqual(memory.rewrite_query_with_history("它的缺点呢"), "它的缺点呢")

        memory.llm.invoke.side_effect = None
        memory.llm.invoke.return_value = mock.Mock(content="   ")
        self.assertEqual(memory.rewrite_query_with_history("它的缺点呢"), "它的缺点呢")

        memory.llm.invoke.return_value = mock.Mock(content="RAG 的缺点是什么")
        self.assertEqual(memory.rewrite_query_with_history("它的缺点呢"), "RAG 的缺点是什么")


class EncodingTests(unittest.TestCase):
    def _load(self, data: bytes, suffix: str = ".md") -> str:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / f"note{suffix}"
            path.write_bytes(data)
            docs = DocumentLoader().load_file(str(path))
        return docs[0].page_content

    def test_utf16_with_bom_is_decoded(self) -> None:
        text = "# 标题\n\n记事本另存为 Unicode 的笔记。"
        for encoding in ("utf-16-le", "utf-16-be"):
            bom = b"\xff\xfe" if encoding.endswith("le") else b"\xfe\xff"
            with self.subTest(encoding=encoding):
                content = self._load(bom + text.encode(encoding), ".txt")
                self.assertEqual(content, text)
                self.assertNotIn("\x00", content)

    def test_existing_encodings_and_newlines_still_work(self) -> None:
        text = "第一段。\n\n第二段。"
        self.assertEqual(self._load(text.encode("utf-8")), text)
        self.assertEqual(self._load(b"\xef\xbb\xbf" + text.encode("utf-8")), text)
        self.assertEqual(self._load(text.encode("gbk")), text)
        # CRLF files must split on paragraph boundaries like LF files.
        self.assertEqual(self._load(text.replace("\n", "\r\n").encode("utf-8")), text)


if __name__ == "__main__":
    unittest.main()
