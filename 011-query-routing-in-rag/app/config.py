
import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "llama3.2:latest"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2"
)

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

CHROMA_PATH = os.path.join(
    BASE_DIR,
    "chroma_db"
)

CHROMA_COLLECTION = os.getenv(
    "CHROMA_COLLECTION",
    "enterprise_knowledge"
)

DOCUMENT_PATH = os.getenv(
    "DOCUMENT_PATH",
    "./data/documents"
)

TOP_K = int(os.getenv("TOP_K", 5))

RERANK_TOP_K = int(
    os.getenv("RERANK_TOP_K", 3)
)
