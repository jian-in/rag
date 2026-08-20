from __future__ import annotations

import unittest

from memory_hub.providers import (
    SiliconFlowEmbeddingProvider,
    SiliconFlowRerankProvider,
    providers_from_environment,
)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.status_code = 200
        self.text = ""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class MemoryProviderTests(unittest.TestCase):
    def test_embedding_adapter_uses_openai_compatible_siliconflow_contract(self) -> None:
        requests_seen: list[dict] = []

        def fake_post(url: str, **kwargs):
            requests_seen.append({"url": url, **kwargs})
            return _FakeResponse(
                {"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]}
            )

        provider = SiliconFlowEmbeddingProvider(
            api_key="local-test-key",
            api_base="https://example.invalid/v1",
            model_name="BAAI/bge-m3",
            post=fake_post,
        )

        vectors = provider.embed(["一条记忆"])

        self.assertEqual(vectors, [[0.1, 0.2, 0.3]])
        self.assertEqual(requests_seen[0]["url"], "https://example.invalid/v1/embeddings")
        self.assertEqual(requests_seen[0]["json"]["model"], "BAAI/bge-m3")
        self.assertEqual(requests_seen[0]["json"]["input"], ["一条记忆"])
        self.assertEqual(
            requests_seen[0]["headers"]["Authorization"], "Bearer local-test-key"
        )

    def test_rerank_adapter_returns_ranked_indexes_without_documents(self) -> None:
        requests_seen: list[dict] = []

        def fake_post(url: str, **kwargs):
            requests_seen.append({"url": url, **kwargs})
            return _FakeResponse(
                {
                    "results": [
                        {"index": 1, "relevance_score": 0.91},
                        {"index": 0, "relevance_score": 0.42},
                    ]
                }
            )

        provider = SiliconFlowRerankProvider(
            api_key="local-test-key",
            api_base="https://example.invalid/v1",
            model_name="BAAI/bge-reranker-v2-m3",
            post=fake_post,
        )

        ranked = provider.rerank("问题", ["第一条", "第二条"], top_n=2)

        self.assertEqual(ranked, [(1, 0.91), (0, 0.42)])
        self.assertEqual(requests_seen[0]["url"], "https://example.invalid/v1/rerank")
        self.assertFalse(requests_seen[0]["json"]["return_documents"])

    def test_provider_factory_requires_explicit_opt_in_for_shared_key(self) -> None:
        embedding, reranker = providers_from_environment({})
        self.assertIsNone(embedding)
        self.assertIsNone(reranker)

        embedding, reranker = providers_from_environment(
            {"SILICONFLOW_API_KEY": "shared-key"}
        )
        self.assertIsNone(embedding)
        self.assertIsNone(reranker)

        embedding, reranker = providers_from_environment(
            {
                "SILICONFLOW_API_KEY": "shared-key",
                "MEMORY_VECTOR_ENABLED": "true",
                "MEMORY_RERANK_ENABLED": "true",
                "MEMORY_EMBEDDING_MODEL": "embedding-test",
                "MEMORY_RERANK_MODEL": "rerank-test",
            }
        )

        self.assertEqual(embedding.model_name, "embedding-test")
        self.assertEqual(reranker.model_name, "rerank-test")


if __name__ == "__main__":
    unittest.main()
