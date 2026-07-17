"""Hybrid retrieval with BM25 plus vector recall and optional reranking."""

import re
import unicodedata
from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document

import config


class BM25Retriever:
    """Keyword retriever tuned for mixed Chinese and English content."""

    def __init__(self):
        self._index = None
        self._documents: List[Document] = []

    def build_index(self, documents: List[Document]):
        from rank_bm25 import BM25Okapi

        self._documents = documents
        tokenized_docs = [self._tokenize(self._document_text(doc)) for doc in documents]
        self._index = BM25Okapi(tokenized_docs)
        print(f"BM25 index built with {len(documents)} chunks")

    def search(self, query: str, top_k: int = config.RETRIEVAL_TOP_K) -> List[Document]:
        if self._index is None:
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        scores = self._index.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self._documents[idx]
                doc.metadata["bm25_score"] = float(scores[idx])
                results.append(doc)
        return results

    def _document_text(self, doc: Document) -> str:
        file_name = doc.metadata.get("file_name", "")
        title = Path(file_name).stem if file_name else ""
        return "\n".join(part for part in [title, file_name, doc.page_content] if part)

    def _tokenize(self, text: str) -> List[str]:
        import jieba

        normalized = unicodedata.normalize("NFKC", text).lower()

        word_tokens = [token.strip() for token in jieba.lcut(normalized) if token.strip()]
        word_tokens = [token for token in word_tokens if re.search(r"[\w\u4e00-\u9fff]", token)]

        ascii_tokens = re.findall(r"[a-z0-9][a-z0-9._/-]*", normalized)

        cjk_ngrams = []
        for span in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
            for n in (2, 3):
                if len(span) < n:
                    continue
                cjk_ngrams.extend(span[i : i + n] for i in range(len(span) - n + 1))

        return [token for token in (word_tokens + ascii_tokens + cjk_ngrams) if token]


class HybridRetriever:
    """Fuse BM25 and vector recall before applying an optional reranker."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_retriever,
        alpha: float = 0.5,
        rrf_k: int = 60,
    ):
        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        self.alpha = alpha
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = config.RETRIEVAL_TOP_K,
        strategy: str = "rrf",
        use_reranker: bool = config.ENABLE_RERANKER,
    ) -> List[Document]:
        fetch_k = max(top_k * 4, top_k)
        bm25_results = self.bm25.search(query, top_k=fetch_k)
        vector_results = self.vector.retrieve(
            query,
            strategy="mmr",
            use_reranker=False,
            top_k=fetch_k,
            candidate_k=fetch_k,
        )

        fusion_k = max(top_k * 3, top_k)
        if strategy == "rrf":
            fused_results = self._rrf_fusion(bm25_results, vector_results, fusion_k)
        else:
            fused_results = self._weighted_fusion(bm25_results, vector_results, fusion_k)

        if use_reranker and self.vector.reranker is not None:
            rerank_limit = min(top_k, len(fused_results))
            return self.vector.reranker.rerank(query, fused_results, top_n=rerank_limit)

        return fused_results[:top_k]

    def _rrf_fusion(
        self,
        bm25_results: List[Document],
        vector_results: List[Document],
        top_k: int,
    ) -> List[Document]:
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(vector_results):
            doc_id = self._doc_id(doc)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            doc_map[doc_id] = doc

        for rank, doc in enumerate(bm25_results):
            doc_id = self._doc_id(doc)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            doc_map.setdefault(doc_id, doc)

        sorted_ids = sorted(rrf_scores.keys(), key=lambda item: rrf_scores[item], reverse=True)
        results = []
        for doc_id in sorted_ids[:top_k]:
            doc = doc_map[doc_id]
            doc.metadata["rrf_score"] = rrf_scores[doc_id]
            results.append(doc)

        print(f"Hybrid(RRF): vector {len(vector_results)} + bm25 {len(bm25_results)} -> fused {len(results)}")
        return results

    def _weighted_fusion(
        self,
        bm25_results: List[Document],
        vector_results: List[Document],
        top_k: int,
    ) -> List[Document]:
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(vector_results, start=1):
            doc_id = self._doc_id(doc)
            score = self.alpha * (1.0 / rank)
            scores[doc_id] = scores.get(doc_id, 0.0) + score
            doc_map[doc_id] = doc

        if bm25_results:
            max_bm25 = max(doc.metadata.get("bm25_score", 0.0) for doc in bm25_results)
            if max_bm25 > 0:
                for doc in bm25_results:
                    doc_id = self._doc_id(doc)
                    normalized = doc.metadata.get("bm25_score", 0.0) / max_bm25
                    score = (1 - self.alpha) * normalized
                    scores[doc_id] = scores.get(doc_id, 0.0) + score
                    doc_map.setdefault(doc_id, doc)

        sorted_ids = sorted(scores.keys(), key=lambda item: scores[item], reverse=True)
        results = []
        for doc_id in sorted_ids[:top_k]:
            doc = doc_map[doc_id]
            doc.metadata["hybrid_score"] = scores[doc_id]
            results.append(doc)
        return results

    def _doc_id(self, doc: Document) -> str:
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        chunk_index = doc.metadata.get("chunk_index", "")
        return f"{source}::{page}::{chunk_index}"
