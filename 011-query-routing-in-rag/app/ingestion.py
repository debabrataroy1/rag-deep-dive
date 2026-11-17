
from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma

from app.config import (
    DOCUMENT_PATH,
    CHROMA_PATH,
    CHROMA_COLLECTION
)

from app.embedding import EmbeddingModel


def ingest_documents():

    print("Loading documents...")

    loader = PyPDFDirectoryLoader(DOCUMENT_PATH)

    documents = loader.load()

    if not documents:
        raise ValueError("No PDF documents found.")

    print("Documents loaded:", len(documents))

    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print("Total chunks:", len(chunks))

    if not chunks:
        raise ValueError("No text chunks generated.")

    # Initialize embedding model
    embeddings = EmbeddingModel()

    print("Generating embeddings and storing in ChromaDB...")

    # Initialize ChromaDB
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    # Store documents and embeddings
    vectorstore.add_documents(
        documents=chunks
    )

    print("Documents successfully stored in ChromaDB.")

    return vectorstore


if __name__ == "__main__":

    ingest_documents()
