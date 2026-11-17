
from sentence_transformers import CrossEncoder

from app.config import RERANK_TOP_K


class DocumentReranker:

    def __init__(self):

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(self, query, documents):

        if not documents:
            return []

        pairs = [
            (query, doc.page_content)
            for doc in documents
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: float(x[1]),
            reverse=True
        )

        return [
            doc
            for doc, score in ranked[:RERANK_TOP_K]
        ]
