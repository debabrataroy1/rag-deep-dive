
from rank_bm25 import BM25Okapi

from langchain_core.documents import Document

from langchain_chroma import Chroma

from app.config import (
    CHROMA_PATH,
    CHROMA_COLLECTION,
    TOP_K
)

from app.embedding import EmbeddingModel


class RAGRetriever:

    def __init__(self):

        # Initialize embedding model
        self.embeddings = EmbeddingModel()

        # Initialize ChromaDB
        self.vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION,
            persist_directory=CHROMA_PATH,
            embedding_function=self.embeddings
        )

        # Load indexed documents
        stored = self.vectorstore.get(
            include=["documents", "metadatas"]
        )
        self.documents = []

        for content, metadata in zip(
            stored["documents"],
            stored["metadatas"]
        ):

            self.documents.append(
                Document(
                    page_content=content,
                    metadata=metadata or {}
                )
            )

        if not self.documents:
            raise ValueError(
                "No indexed documents. Run ingestion first."
            )

        # Initialize BM25
        tokenized_corpus = [
            doc.page_content.lower().split()
            for doc in self.documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_corpus
        )

    # BM25 SEARCH
    def bm25_search(self, query):

        tokens = query.lower().split()

        scores = self.bm25.get_scores(tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:TOP_K]

        return [
            self.documents[i]
            for i in ranked_indices
            if scores[i] > 0
        ]

    # VECTOR SEARCH
    def vector_search(self, query):

        return self.vectorstore.similarity_search(
            query,
            k=TOP_K
        )

    # HYBRID SEARCH
    def hybrid_search(self, query):

        bm25_results = self.bm25_search(query)

        vector_results = self.vector_search(query)

        return [
            bm25_results,
            vector_results
        ]
