from __future__ import annotations

import os
from collections.abc import Callable
from collections.abc import Mapping
from typing import Any

import requests


PostCallable = Callable[..., Any]


class SiliconFlowEmbeddingProvider:
    """OpenAI-compatible embedding adapter for SiliconFlow."""

    def __init__(
        self,
        *,
        api_key: str,
        api_base: str = "https://api.siliconflow.cn/v1",
        model_name: str = "BAAI/bge-m3",
        timeout: float = 60.0,
        post: PostCallable = requests.post,
    ) -> None:
        if not api_key.strip():
            raise ValueError("SiliconFlow embedding API key is required")
        self._api_key = api_key.strip()
        self._api_base = api_base.rstrip("/")
        self.model_name = model_name
        self._timeout = timeout
        self._post = post

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._post(
            f"{self._api_base}/embeddings",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model_name, "input": texts},
            timeout=self._timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"SiliconFlow embedding request failed: {response.status_code} {response.text}"
            ) from exc
        data = response.json().get("data", [])
        ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
        vectors = [item.get("embedding") for item in ordered]
        if len(vectors) != len(texts) or any(not isinstance(vector, list) for vector in vectors):
            raise RuntimeError("SiliconFlow embedding response has an unexpected shape")
        return [[float(value) for value in vector] for vector in vectors]


class SiliconFlowRerankProvider:
    """SiliconFlow rerank adapter returning source indexes and relevance scores."""

    def __init__(
        self,
        *,
        api_key: str,
        api_base: str = "https://api.siliconflow.cn/v1",
        model_name: str = "BAAI/bge-reranker-v2-m3",
        timeout: float = 60.0,
        post: PostCallable = requests.post,
    ) -> None:
        if not api_key.strip():
            raise ValueError("SiliconFlow rerank API key is required")
        self._api_key = api_key.strip()
        self._api_base = api_base.rstrip("/")
        self.model_name = model_name
        self._timeout = timeout
        self._post = post

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[tuple[int, float]]:
        if not documents:
            return []
        response = self._post(
            f"{self._api_base}/rerank",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "query": query,
                "documents": documents,
                "top_n": min(top_n, len(documents)),
                "return_documents": False,
            },
            timeout=self._timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"SiliconFlow rerank request failed: {response.status_code} {response.text}"
            ) from exc
        results = []
        for item in response.json().get("results", []):
            index = int(item.get("index", -1))
            if 0 <= index < len(documents):
                results.append((index, float(item.get("relevance_score", 0.0))))
        return results


def providers_from_environment(
    environment: Mapping[str, str] | None = None,
) -> tuple[SiliconFlowEmbeddingProvider | None, SiliconFlowRerankProvider | None]:
    env = environment if environment is not None else os.environ
    shared_key = env.get("SILICONFLOW_API_KEY", "").strip()
    dedicated_embedding_key = env.get("MEMORY_EMBEDDING_API_KEY", "").strip()
    dedicated_rerank_key = env.get("MEMORY_RERANK_API_KEY", "").strip()
    embedding_key = dedicated_embedding_key or shared_key
    rerank_key = dedicated_rerank_key or shared_key
    api_base = env.get("MEMORY_SILICONFLOW_API_BASE", "https://api.siliconflow.cn/v1")

    embedding = None
    if embedding_key and _enabled(
        env.get("MEMORY_VECTOR_ENABLED"), default=bool(dedicated_embedding_key)
    ):
        embedding = SiliconFlowEmbeddingProvider(
            api_key=embedding_key,
            api_base=api_base,
            model_name=env.get("MEMORY_EMBEDDING_MODEL", "BAAI/bge-m3"),
        )

    reranker = None
    if rerank_key and _enabled(
        env.get("MEMORY_RERANK_ENABLED"), default=bool(dedicated_rerank_key)
    ):
        reranker = SiliconFlowRerankProvider(
            api_key=rerank_key,
            api_base=api_base,
            model_name=env.get("MEMORY_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
        )
    return embedding, reranker


def _enabled(value: str | None, *, default: bool) -> bool:
    # 留空（如 .env 里写了 "MEMORY_VECTOR_ENABLED=" 却没填值）等同未设置：
    # 此前空串不在关闭列表里，会被当成开启，把记忆文本发往外部服务。
    if value is None or not value.strip():
        return default
    return value.strip().casefold() not in {"0", "false", "no", "off"}
