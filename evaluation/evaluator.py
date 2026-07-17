"""
RAG 评估模块

基于 RAGAS 框架对 RAG 系统进行量化评估。
面试亮点：
  - 知道如何科学地评估 RAG 系统（不只是"感觉效果好不好"）
  - 掌握核心评估指标的含义和优化方向
  - 能自动生成评估报告

评估指标说明：
  - Faithfulness（忠实度）：答案是否完全基于检索到的上下文，有没有幻觉
  - Answer Relevancy（答案相关性）：答案是否切题回答了用户的问题
  - Context Precision（上下文精确度）：检索到的文档中，相关内容的排名是否靠前
  - Context Recall（上下文召回率）：是否检索到了所有需要的信息
"""

import json
import os
from typing import Dict, List, Optional

from datasets import Dataset

import config


class RAGEvaluator:
    """
    RAG 系统评估器

    使用 RAGAS 框架进行自动化评估，支持：
      1. 从预设评估集运行评估
      2. 手动添加评估样本
      3. 生成评估报告
    """

    def __init__(self):
        """初始化评估器"""
        self.eval_samples: List[Dict] = []

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
            contexts: 检索到的上下文（如果不提供，会自动从系统获取）
            answer: 系统生成的答案（如果不提供，会自动调用系统生成）
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
                "contexts": ["相关文档片段1", "相关文档片段2"]
            },
            ...
        ]

        Args:
            file_path: 评估数据集文件路径
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
    ) -> Dict:
        """
        运行评估

        如果评估样本中没有 answer，会自动调用 qa_chain 生成。
        如果评估样本中没有 contexts，会自动从 qa_chain 的检索器获取。

        Args:
            qa_chain: QA 链实例（用于自动生成答案和上下文）
            metrics: 要计算的指标列表，默认计算所有核心指标

        Returns:
            Dict: 评估结果
        """
        if not self.eval_samples:
            raise ValueError("没有评估样本，请先添加样本或加载评估数据集")

        if metrics is None:
            metrics = ["faithfulness", "answer_relevancy", "context_precision"]

        # 补全缺失的 answer 和 contexts
        self._fill_missing(qa_chain)

        # 构建 RAGAS Dataset
        eval_data = {
            "question": [s["question"] for s in self.eval_samples],
            "answer": [s["answer"] for s in self.eval_samples],
            "contexts": [s["contexts"] for s in self.eval_samples],
            "ground_truth": [s["ground_truth"] for s in self.eval_samples],
        }
        dataset = Dataset.from_dict(eval_data)

        # 选择评估指标
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )

        metric_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }

        selected_metrics = [metric_map[m] for m in metrics if m in metric_map]

        # 运行评估
        results = evaluate(
            dataset=dataset,
            metrics=selected_metrics,
        )

        return results

    def _fill_missing(self, qa_chain=None):
        """补全评估样本中缺失的 answer 和 contexts"""
        for sample in self.eval_samples:
            if sample["answer"] is None or sample["contexts"] is None:
                if qa_chain is None:
                    raise ValueError(
                        "评估样本中缺少 answer 或 contexts，"
                        "需要提供 qa_chain 来自动生成"
                    )

                answer, docs = qa_chain.answer(sample["question"])

                if sample["answer"] is None:
                    sample["answer"] = answer

                if sample["contexts"] is None:
                    sample["contexts"] = [doc.page_content for doc in docs]

    def save_results(self, results: Dict, output_path: str = "evaluation_report.json"):
        """
        保存评估结果到 JSON 文件

        Args:
            results: 评估结果字典
            output_path: 输出文件路径
        """
        # RAGAS 返回的是 EvaluationResult 对象
        result_dict = {
            "overall_scores": {},
            "sample_results": [],
        }

        # 提取整体分数
        for metric_name, score in results.items():
            if isinstance(score, (int, float)):
                result_dict["overall_scores"][metric_name] = round(score, 4)

        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            output_path,
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)

        print(f"✅ 评估报告已保存到: {output_path}")

    def print_report(self, results: Dict):
        """
        打印可读的评估报告

        Args:
            results: 评估结果
        """
        print("\n" + "=" * 50)
        print("📊 RAG 系统评估报告")
        print("=" * 50)

        for metric_name, score in results.items():
            if isinstance(score, (int, float)):
                emoji = "🟢" if score >= 0.8 else "🟡" if score >= 0.6 else "🔴"
                print(f"  {emoji} {metric_name}: {score:.4f}")

        print("\n" + "=" * 50)
        print("📝 指标说明:")
        print("  - Faithfulness: 答案是否基于检索上下文（越高幻觉越少）")
        print("  - Answer Relevancy: 答案是否切题")
        print("  - Context Precision: 检索文档的相关性和排序质量")
        print("  - Context Recall: 是否检索到了所有必要信息")
        print("=" * 50 + "\n")


# ============================================================
# 示例：如何生成评估数据集模板
# ============================================================
def create_sample_eval_dataset(output_path: str = "data/eval_dataset.json"):
    """
    创建一个示例评估数据集模板

    实际使用时，你需要根据自己的知识库内容，
    人工编写问题和标准答案。

    Args:
        output_path: 输出文件路径
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
