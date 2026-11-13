# Day XX - Basic RAG Using ChromaDB and Ollama

## 1. Concept
Learned how to build a basic RAG (Retrieval-Augmented Generation) application using ChromaDB and Ollama with default embedding system.

## 2. Why do we need it?
To provide LLMs with external knowledge so they can answer questions using our own documents instead of relying only on their training data.

## 3. Architecture
1. Add company policy documents to ChromaDB.
2. ChromaDB generates embeddings and stores them.
3. User asks a question.
4. ChromaDB retrieves relevant documents.
5. Retrieved documents are passed to Ollama.
6. Llama 3.2 generates the answer.

## 4. Implementation
Implemented a basic RAG pipeline using:
- Python
- ChromaDB
- Ollama
- Llama 3.2

## 5. How to run

Install dependencies:

```bash
pip install chromadb ollama
```

Download the model:

```bash
ollama pull llama3.2:latest
```

Run the application:

```bash
python main.py
```

## 6. Test cases

1. How many days can I work from home?
2. How many PTO days do employees get?
3. Do I need manager approval for remote work?
4. When should I submit expense reports?
5. What is the company's leave policy?
6. Can I work remotely every day?
7. What is the maternity leave policy?
8. Summarize all company policies.

## 7. Results
- Successfully stored documents in ChromaDB.
- Retrieved relevant documents using semantic search.
- Generated answers using Ollama.
- Executed the complete RAG pipeline locally.

## 8. Comparison
Compared with a normal LLM request, RAG retrieves relevant information from external documents before generating an answer.

## 9. Learnings
- Basic understanding of RAG.
- How ChromaDB stores documents and embeddings.
- How semantic search works.
- How to retrieve relevant context.
- How to integrate Ollama with Python.
