# RAG with Fixed-Size Chunking

## 1. Concept

Fixed-size chunking is a text-splitting technique used in Retrieval-Augmented Generation (RAG).

It divides a large document into smaller chunks based on a predefined character size.

In this exercise:
- Chunk size: `500` characters
- Overlap: `50` characters
- Embedding model: `all-MiniLM-L6-v2`
- Vector database: ChromaDB
- LLM: Llama 3.2 using Ollama

## 2. Why Do We Need It?

LLMs have context-window limitations. Sending an entire PDF document to an LLM can be inefficient.

Fixed-size chunking helps us:
- Split large documents into smaller pieces.
- Generate embeddings for individual chunks.
- Retrieve only relevant information.
- Reduce unnecessary context sent to the LLM.
- Improve retrieval efficiency.

## 3. Architecture

```text
          PDF Document
                |
                v
          Extract Text
                |
                v
       Fixed-Size Chunking
         (500 characters)
                |
                v
         Generate Embeddings
       (all-MiniLM-L6-v2)
                |
                v
           ChromaDB
                |
                |
         User Question
                |
                v
       Generate Question Embedding
                |
                v
        Similarity Search
                |
                v
        Retrieve Relevant Chunks
                |
                v
          Build Context
                |
                v
          Create Prompt
                |
                v
          Llama 3.2
                |
                v
          Final Answer
```

## 4. Implementation

### Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyPDF | Extract text from PDF |
| Sentence Transformers | Generate embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| ChromaDB | Store and retrieve vectors |
| Ollama | Run LLM locally |
| Llama 3.2 | Generate answers |

### Chunking Configuration

```python
def text_chunking(text, chunk_size=500, overlap=50):
```

- `chunk_size`: Maximum characters in each chunk.
- `overlap`: Number of characters shared between consecutive chunks.
- `step`: `chunk_size - overlap`, which is 450 characters.

### Example

Original text:

```text
ABCDEFGHIJKLMNOPQRSTUVWXYZ...
```

Chunk 1:

```text
Characters 1-500
```

Chunk 2:

```text
Characters 451-950
```

Chunk 3:

```text
Characters 901-1400
```

The overlapping text helps preserve context between consecutive chunks.

## 5. How to Run

### Install Dependencies

```bash
pip install chromadb ollama pypdf sentence-transformers
```

### Start Ollama

```bash
ollama pull llama3.2
```

### Run the Application

```bash
python main.py
```

Make sure `java_notes.pdf` is available in the project directory.

## 6. Test Cases

| Test Case | Expected Result |
|---|---|
| Load PDF | PDF should load successfully |
| Extract text | Text should be extracted from PDF pages |
| Chunk text | Text should be split into 500-character chunks |
| Check overlap | Consecutive chunks should share 50 characters |
| Generate embeddings | Each chunk should have a 384-dimensional vector |
| Store embeddings | Embeddings should be saved in ChromaDB |
| Retrieve data | Relevant chunks should be returned |
| Generate answer | LLM should answer using retrieved context |

## 7. Results

- Successfully extracted text from the PDF.
- Split extracted text into fixed-size chunks.
- Generated embeddings using Sentence Transformers.
- Stored embeddings in ChromaDB.
- Retrieved relevant chunks using vector similarity search.
- Generated answers using the local Llama 3.2 model.
