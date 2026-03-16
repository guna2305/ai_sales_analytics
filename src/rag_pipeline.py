from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from sentence_transformers import SentenceTransformer

try:
    import faiss  # type: ignore
except Exception:
    faiss = None


@dataclass
class Doc:
    text: str
    meta: Dict[str, Any]


class RAGIndex:
    def __init__(self, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if faiss is None:
            raise ImportError("FAISS not installed. Install faiss-cpu.")
        self.embedder = SentenceTransformer(embed_model)
        self.index = None
        self.docs: List[Doc] = []

    def _embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype("float32")

    def build(self, docs: List[Doc]) -> None:
        self.docs = docs
        vectors = self._embed([d.text for d in docs])
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)

    def search_with_scores(self, query: str, k: int = 5):
        if self.index is None or not self.docs:
            return []
        qv = self._embed([query])
        scores, idxs = self.index.search(qv, k)
        results = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            results.append((float(score), self.docs[int(i)]))
        return results


def build_docs_from_outputs(kpis_text: str, forecast_insights_text: str, forecast_table_text: str) -> List[Doc]:
    docs: List[Doc] = []
    docs.append(Doc(text=kpis_text, meta={"type": "kpis"}))
    docs.append(Doc(text=forecast_insights_text, meta={"type": "forecast_insights"}))
    docs.append(Doc(text=forecast_table_text, meta={"type": "forecast_table"}))
    return docs