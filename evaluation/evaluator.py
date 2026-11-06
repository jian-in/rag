"""
RAG 评估模块

用项目自身配置的 LLM 做裁判（LLM-as-judge），计算四个 RAGAS 风格的指标。

为什么不再依赖 ragas：
  - ragas 0.4.x 导入 ``langchain_community.chat_models.vertexai``，该模块已在
    langchain-community 0.4 中移除；与本项目的 langchain 1.x 无可共存版本，
    导致评估功能完全不可用（界面还误报「请先安装 ragas」）。
  - ragas 默认读 ``OPENAI_BASE_URL``，而本项目用 ``OPENAI_API_BASE`` 指向
    兼容网关，即便能导入也会把请求发到 api.openai.com。

指标说明（取值 0~1，越高越好）：
  - faithfulness（忠实度）：答案拆成事实陈述，能被检索上下文支持的比例（衡量幻觉）
  - answer_relevancy（答案相关性）：答案对问题的切题程度（LLM 0~10 打分归一化；
    ragas 用「反向生成问题 + 向量相似度」，两者数值不可直接比较）
  - context_precision（上下文精确度）：按排名计算的平均精度——有用的上下文是否排在前面
  - context_recall（上下文召回率）：标准答案中能在上下文里找到依据的句子比例
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

import config


METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
DEFAULT_METRICS = ("faithfulness", "answer_relevancy", "context_precision")
MAX_CONTEXTS = 10
CONTEXT_CHAR_LIMIT = 1500


class EvaluationError(RuntimeError):
    """裁判 LLM 的输出无法解析为预期结构。"""


def _default_llm():
    from langchain_openai import ChatOpenAI

    kwargs = {
        "model": config.LLM_MODEL_NAME,
        "temperature": 0,
        "openai_api_key": config.OPENAI_API_KEY,
    }
    if config.OPENAI_API_BASE:
        kwargs["openai_api_base"] = config.OPENAI_API_BASE
    return ChatOpenAI(**kwargs)


def _extract_json(text: str) -> Dict[str, Any]:
    """从模型回答中取出第一个 JSON 对象（容忍 ```json 代码块与前后说明文字）。"""
    raw = (text or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise EvaluationError(f"裁判输出中没有 JSON 对象: {raw[:200]}")
    try:
        payload = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"裁判输出不是合法 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvaluationError("裁判输出的 JSON 不是对象")
    return payload


def _format_contexts(contexts: Sequence[str]) -> str:
    parts = []
    for index, context in enumerate(list(contexts)[:MAX_CONTEXTS], 1):
        text = str(context)
        if len(text) > CONTEXT_CHAR_LIMIT:
            text = text[:CONTEXT_CHAR_LIMIT] + "…"
        parts.append(f"[上下文 {index}]\n{text}")
    return "\n\n".join(parts) if parts else "（无上下文）"


def _bool_list(items: Any, key: str) -> List[bool]:
    if not isinstance(items, list):
        raise EvaluationError(f"裁判输出缺少列表字段 {key}")
    verdicts = []
    for item in items:
        value = item.get(key) if isinstance(item, dict) else item
        if isinstance(value, str):
            value = value.strip().lower() in {"true", "yes", "1", "是"}
        verdicts.append(bool(value))
    return verdicts


class RAGEvaluator:
    """
    RAG 系统评估器

    支持：
      1. 从预设评估集运行评估
      2. 手动添加评估样本
      3. 生成评估报告

    ``llm`` 可注入任意带 ``invoke(prompt) -> 有 .content 的对象`` 的模型，
    便于测试或换用其他裁判模型；默认使用 config 中的 LLM。
    """

    def __init__(self, llm: Any = None):
        self.eval_samples: List[Dict] = []
        self._llm = llm
        self.sample_scores: List[Dict[str, Optional[float]]] = []
        self.errors: List[str] = []

    @property
    def llm(self):
        if self._llm is None:
            self._llm = _default_llm()
        return self._llm

    def add_sample(
        self,
        question: str,
        ground_truth: str,
        contexts: Optional[List[str]] = None,
        answer: Optional[str] = None,
    ):
        """
        添加一条评估样本

        Args:
            question: 评估问题
            ground_truth: 标准答案（人工标注）
            contexts: 检索到的上下文；None 或空列表都会在评估时自动从系统获取
            answer: 系统生成的答案；None 或空字符串都会在评估时自动生成
        """
        self.eval_samples.append({
            "question": question,
            "ground_truth": ground_truth,
            "contexts": contexts,
            "answer": answer,
        })

    def load_eval_dataset(self, file_path: str):
        """
        从 JSON 文件加载评估数据集

        JSON 格式示例：
        [
            {
                "question": "Transformer 模型的核心机制是什么？",
                "ground_truth": "Transformer 的核心是自注意力机制...",
                "contexts": []
            }
        ]
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"评估数据集不存在: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            self.add_sample(
                question=item["question"],
                ground_truth=item["ground_truth"],
                contexts=item.get("contexts"),
                answer=item.get("answer"),
            )

        print(f"✅ 已加载 {len(data)} 条评估样本")

    def run_evaluation(
        self,
        qa_chain=None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        运行评估

        缺少 answer 或 contexts 的样本会先调用 qa_chain 补全。单个样本的某个
        指标判定失败时记为 None 并写入 ``self.errors``，不影响其余样本与指标。

        Returns:
            Dict[str, float]: 各指标在可计算样本上的平均分；逐样本分数见
            ``self.sample_scores``。
        """
        if not self.eval_samples:
            raise ValueError("没有评估样本，请先添加样本或加载评估数据集")

        selected = list(metrics) if metrics else list(DEFAULT_METRICS)
        unknown = [m for m in selected if m not in METRICS]
        if unknown:
            raise ValueError(f"不支持的评估指标: {', '.join(unknown)}；可选: {', '.join(METRICS)}")

        self._fill_missing(qa_chain)

        scorers: Dict[str, Callable[[Dict], Optional[float]]] = {
            "faithfulness": self._faithfulness,
            "answer_relevancy": self._answer_relevancy,
            "context_precision": self._context_precision,
            "context_recall": self._context_recall,
        }
        self.sample_scores = []
        self.errors = []
        for index, sample in enumerate(self.eval_samples, 1):
            scores: Dict[str, Optional[float]] = {}
            for metric in selected:
                try:
                    scores[metric] = scorers[metric](sample)
                except Exception as exc:
                    scores[metric] = None
                    self.errors.append(f"样本 {index} / {metric}: {exc}")
            self.sample_scores.append(scores)

        results: Dict[str, float] = {}
        for metric in selected:
            values = [s[metric] for s in self.sample_scores if s.get(metric) is not None]
            if values:
                results[metric] = sum(values) / len(values)
        if self.errors:
            print(f"⚠️ {len(self.errors)} 个指标判定失败，已跳过：")
            for error in self.errors[:5]:
                print(f"   - {error}")
        return results

    def _fill_missing(self, qa_chain=None):
        """补全评估样本中缺失的 answer 和 contexts（空列表/空字符串同样视为缺失）"""
        for sample in self.eval_samples:
            if not sample["answer"] or not sample["contexts"]:
                if qa_chain is None:
                    raise ValueError(
                        "评估样本中缺少 answer 或 contexts，"
                        "需要提供 qa_chain 来自动生成"
                    )
                answer, docs = qa_chain.answer(sample["question"])
                if not sample["answer"]:
                    sample["answer"] = answer
                if not sample["contexts"]:
                    sample["contexts"] = [doc.page_content for doc in docs]

    # ------------------------------------------------------------
    # 指标
    # ------------------------------------------------------------

    def _judge(self, prompt: str) -> Dict[str, Any]:
        response = self.llm.invoke(prompt)
        return _extract_json(getattr(response, "content", response))

    def _faithfulness(self, sample: Dict) -> Optional[float]:
        payload = self._judge(
            "你是严格的事实核查员。把【答案】拆成若干条独立的事实陈述，逐条判断能否由【上下文】"
            "直接推出（不能使用上下文之外的常识）。\n\n"
            f"【问题】\n{sample['question']}\n\n"
            f"【上下文】\n{_format_contexts(sample['contexts'])}\n\n"
            f"【答案】\n{sample['answer']}\n\n"
            '只输出 JSON：{"statements": [{"statement": "陈述", "supported": true}]}'
        )
        verdicts = _bool_list(payload.get("statements"), "supported")
        if not verdicts:
            return None  # 答案不含可核查的陈述（例如「无法回答」）
        return sum(verdicts) / len(verdicts)

    def _answer_relevancy(self, sample: Dict) -> Optional[float]:
        payload = self._judge(
            "评估【答案】是否直接、完整地回应了【问题】（不评判事实对错）。"
            "按 0~10 打分：10=完全切题且无冗余，5=部分回答或明显跑题，0=完全无关或拒答。\n\n"
            f"【问题】\n{sample['question']}\n\n"
            f"【答案】\n{sample['answer']}\n\n"
            '只输出 JSON：{"score": 0到10的数字, "reason": "一句话理由"}'
        )
        try:
            score = float(payload["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationError("裁判输出缺少数值字段 score") from exc
        return min(max(score, 0.0), 10.0) / 10.0

    def _context_precision(self, sample: Dict) -> Optional[float]:
        contexts = list(sample["contexts"])[:MAX_CONTEXTS]
        if not contexts:
            return 0.0
        payload = self._judge(
            "依次判断每段【上下文】对得出【标准答案】是否有用。按上下文编号顺序输出与上下文"
            f"数量相同（{len(contexts)} 个）的判断。\n\n"
            f"【问题】\n{sample['question']}\n\n"
            f"【标准答案】\n{sample['ground_truth']}\n\n"
            f"【上下文】\n{_format_contexts(contexts)}\n\n"
            '只输出 JSON：{"verdicts": [{"index": 1, "useful": true}]}'
        )
        verdicts = _bool_list(payload.get("verdicts"), "useful")
        if len(verdicts) != len(contexts):
            raise EvaluationError(
                f"判断数量 {len(verdicts)} 与上下文数量 {len(contexts)} 不一致"
            )
        # 平均精度：每个有用位置 k 的 precision@k 的平均值
        hits = 0
        precision_sum = 0.0
        for rank, useful in enumerate(verdicts, 1):
            if useful:
                hits += 1
                precision_sum += hits / rank
        return precision_sum / hits if hits else 0.0

    def _context_recall(self, sample: Dict) -> Optional[float]:
        payload = self._judge(
            "把【标准答案】拆成若干句独立陈述，逐句判断能否在【上下文】中找到依据。\n\n"
            f"【标准答案】\n{sample['ground_truth']}\n\n"
            f"【上下文】\n{_format_contexts(sample['contexts'])}\n\n"
            '只输出 JSON：{"sentences": [{"sentence": "句子", "attributed": true}]}'
        )
        verdicts = _bool_list(payload.get("sentences"), "attributed")
        if not verdicts:
            return None
        return sum(verdicts) / len(verdicts)

    # ------------------------------------------------------------
    # 报告
    # ------------------------------------------------------------

    def save_results(self, results: Dict, output_path: str = "evaluation_report.json"):
        """保存评估结果（整体分数 + 逐样本分数）到 JSON 文件"""
        result_dict = {
            "overall_scores": {
                name: round(score, 4)
                for name, score in results.items()
                if isinstance(score, (int, float))
            },
            "sample_results": [
                {"question": sample["question"], **scores}
                for sample, scores in zip(self.eval_samples, self.sample_scores)
            ],
            "errors": self.errors,
        }

        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            output_path,
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)

        print(f"✅ 评估报告已保存到: {output_path}")

    def print_report(self, results: Dict):
        """打印可读的评估报告"""
        print("\n" + "=" * 50)
        print("📊 RAG 系统评估报告")
        print("=" * 50)

        for metric_name, score in results.items():
            if isinstance(score, (int, float)):
                emoji = "🟢" if score >= 0.8 else "🟡" if score >= 0.6 else "🔴"
                print(f"  {emoji} {metric_name}: {score:.4f}")

        print("\n" + "=" * 50)
        print("📝 指标说明:")
        print("  - Faithfulness: 答案陈述被上下文支持的比例（越高幻觉越少）")
        print("  - Answer Relevancy: 答案是否切题")
        print("  - Context Precision: 有用的上下文是否排在前面")
        print("  - Context Recall: 标准答案的依据是否都被检索到")
        print("=" * 50 + "\n")


# ============================================================
# 示例：如何生成评估数据集模板
# ============================================================

def create_sample_eval_dataset(output_path: str = "data/eval_dataset.json"):
    """
    创建一个示例评估数据集模板

    ``contexts`` 留空列表表示「评估时自动检索」。实际使用时，请根据自己的
    知识库内容人工编写问题和标准答案。
    """
    samples = [
        {
            "question": "什么是 RAG（检索增强生成）？",
            "ground_truth": "RAG 是一种将信息检索与大语言模型生成相结合的技术。它先从外部知识库中检索相关文档，然后将这些文档作为上下文提供给 LLM，使其能够基于事实生成准确的回答。",
            "contexts": [],
        },
        {
            "question": "向量数据库在 RAG 系统中的作用是什么？",
            "ground_truth": "向量数据库用于存储和检索文档的向量表示（Embedding）。它将文本转换为高维向量，通过计算向量之间的相似度来找到与查询最相关的文档片段。",
            "contexts": [],
        },
        {
            "question": "MMR 检索和基础相似度检索有什么区别？",
            "ground_truth": "MMR（最大边际相关性）检索在保证相关性的同时，会尽量返回多样化的结果，避免返回内容高度重复的文档。而基础相似度检索只考虑查询与文档的相似度，可能返回多个内容相似的文档。",
            "contexts": [],
        },
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"✅ 示例评估数据集已创建: {output_path}")
    print("📝 请根据你的知识库内容修改问题和标准答案")


if __name__ == "__main__":
    # 运行示例：创建评估数据集模板
    create_sample_eval_dataset()
