"""Retriever utilities for vector and reranked retrieval."""

from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

import config
from retrieval.reranker import Reranker


class SmartRetriever:
    """Wrap vector retrieval strategies and optional reranking."""

    def __init__(
        self,
        vectorstore: Chroma,
        top_k: int = config.RETRIEVAL_TOP_K,
        mmr_lambda: float = config.MMR_LAMBDA,
        reranker: Optional[Reranker] = None,
    ):
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.mmr_lambda = mmr_lambda
        self.reranker = reranker

    def similarity_search(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """Return vector similarity matches."""
        k = top_k or self.top_k
        return self.vectorstore.similarity_search(query=query, k=k)

    def mmr_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        fetch_k: Optional[int] = None,
    ) -> List[Document]:
        """Return MMR matches with a larger candidate pool when needed."""
        k = top_k or self.top_k
        candidate_pool = max(fetch_k or (k * 4), k)
        return self.vectorstore.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=candidate_pool,
            lambda_mult=self.mmr_lambda,
        )

    def retrieve(
        self,
        query: str,
        strategy: str = "mmr",
        use_reranker: bool = config.ENABLE_RERANKER,
        top_k: Optional[int] = None,
        candidate_k: Optional[int] = None,
        rerank_top_n: Optional[int] = None,
    ) -> List[Document]:
        """Run recall first, then optionally rerank a wider candidate set."""
        final_top_k = top_k or self.top_k
        candidate_limit = max(candidate_k or final_top_k, final_top_k)

        if strategy == "mmr":
            docs = self.mmr_search(
                query,
                top_k=candidate_limit,
                fetch_k=max(candidate_limit * 4, final_top_k),
            )
        else:
            docs = self.similarity_search(query, top_k=candidate_limit)

        if use_reranker and self.reranker is not None:
            rerank_limit = min(rerank_top_n or final_top_k, len(docs))
            return self.reranker.rerank(query, docs, top_n=rerank_limit)

        return docs[:final_top_k]

    def format_context(self, docs: List[Document]) -> str:
        """Format retrieved documents into a prompt-friendly context string."""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("file_name", "unknown")
            page = doc.metadata.get("page", "")
            page_info = f" (page {page})" if page else ""
            context_parts.append(f"[Doc {i}] Source: {source}{page_info}\n{doc.page_content}")
        return "\n\n---\n\n".join(context_parts)
