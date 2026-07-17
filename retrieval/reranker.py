"""Reranker backends for retrieval quality improvements."""

from typing import List

import requests
from langchain_core.documents import Document

import config


class Reranker:
    """Support Cohere, local cross-encoder, and SiliconFlow rerank backends."""

    def __init__(
        self,
        provider: str = config.RERANKER_PROVIDER,
        model_name: str = config.RERANKER_MODEL_NAME,
    ):
        self.provider = provider
        self.model_name = model_name
        self._model = None

    def _init_model(self):
        if self._model is not None:
            return

        if self.provider == "cohere":
            self._init_cohere()
        elif self.provider == "cross-encoder":
            self._init_cross_encoder()
        elif self.provider == "siliconflow":
            self._init_siliconflow()
        else:
            raise ValueError(f"Unsupported reranker provider: {self.provider}")

    def _init_cohere(self):
        if not config.COHERE_API_KEY:
            raise ValueError(
                "Cohere rerank requires COHERE_API_KEY. "
                "Set it in .env or switch to cross-encoder / siliconflow."
            )

        import cohere

        self._model = cohere.Client(api_key=config.COHERE_API_KEY)
        print(f"Loaded Cohere reranker: {self.model_name}")

    def _init_cross_encoder(self):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self.model_name)
        print(f"Loaded cross-encoder reranker: {self.model_name}")

    def _init_siliconflow(self):
        if not config.RERANKER_API_KEY:
            raise ValueError(
                "SiliconFlow rerank requires SILICONFLOW_API_KEY or RERANKER_API_KEY."
            )

        self._model = {
            "api_base": config.RERANKER_API_BASE.rstrip("/"),
            "api_key": config.RERANKER_API_KEY,
        }
        print(f"Loaded SiliconFlow reranker: {self.model_name}")

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int = config.RERANKER_TOP_N,
    ) -> List[Document]:
        if not documents:
            return []

        self._init_model()

        if self.provider == "cohere":
            return self._rerank_cohere(query, documents, top_n)
        if self.provider == "cross-encoder":
            return self._rerank_cross_encoder(query, documents, top_n)
        return self._rerank_siliconflow(query, documents, top_n)

    def _rerank_cohere(
        self, query: str, documents: List[Document], top_n: int
    ) -> List[Document]:
        doc_texts = [doc.page_content for doc in documents]
        results = self._model.rerank(
            query=query,
            documents=doc_texts,
            top_n=top_n,
            model=self.model_name,
        )

        reranked = []
        for item in results.results:
            doc = documents[item.index]
            doc.metadata["rerank_score"] = item.relevance_score
            reranked.append(doc)
        return reranked

    def _rerank_cross_encoder(
        self, query: str, documents: List[Document], top_n: int
    ) -> List[Document]:
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.predict(pairs)
        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda item: item[0], reverse=True)

        reranked = []
        for score, doc in scored_docs[:top_n]:
            doc.metadata["rerank_score"] = float(score)
            reranked.append(doc)
        return reranked

    def _rerank_siliconflow(
        self, query: str, documents: List[Document], top_n: int
    ) -> List[Document]:
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": [doc.page_content for doc in documents],
            "top_n": top_n,
            "return_documents": False,
        }

        if self.model_name in {
            "BAAI/bge-reranker-v2-m3",
            "Pro/BAAI/bge-reranker-v2-m3",
            "netease-youdao/bce-reranker-base_v1",
        }:
            payload["max_chunks_per_doc"] = config.RERANKER_MAX_CHUNKS_PER_DOC
            payload["overlap_tokens"] = config.RERANKER_OVERLAP_TOKENS

        response = requests.post(
            f"{self._model['api_base']}/rerank",
            headers={
                "Authorization": f"Bearer {self._model['api_key']}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"SiliconFlow rerank request failed: {response.status_code} {response.text}"
            ) from exc

        data = response.json()
        reranked = []
        for item in data.get("results", []):
            doc = documents[item["index"]]
            doc.metadata["rerank_score"] = float(item.get("relevance_score", 0.0))
            reranked.append(doc)
        return reranked
