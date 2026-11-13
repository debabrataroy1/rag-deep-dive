
# Text Chunking Techniques in RAG

Text chunking is the process of dividing large documents into smaller pieces of text called chunks. It is commonly used in NLP, LLMs, and Retrieval-Augmented Generation (RAG).

This document covers seven popular text chunking techniques, their Python implementations, advantages, limitations, and use cases.

---

## 1. Fixed-Size Chunking

**Definition:** Splits text into chunks containing a fixed number of characters, regardless of word or sentence boundaries.

### Python Implementation

```python
def fixed_size_chunking(text, chunk_size=100):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    return chunks


text = "Artificial Intelligence is transforming the world. " * 10

chunks = fixed_size_chunking(text, chunk_size=100)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")
```

### Advantages
- Simple and easy to implement.
- Fast processing.
- Predictable chunk sizes.
- Useful for large text documents.

### Limitations
- May split words in the middle.
- May break sentences and paragraphs.
- Does not preserve semantic meaning.

### Use Cases
- Basic text preprocessing.
- Large document processing.
- Simple NLP applications.

---

## 2. Fixed-Token Chunking

**Definition:** Splits text based on a fixed number of tokens rather than characters.

A token can be a complete word, part of a word, or punctuation, depending on the tokenizer.

### Installation

```bash
pip install tiktoken
```

### Python Implementation

```python
import tiktoken

def fixed_token_chunking(text, chunk_size=100):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)

    chunks = []

    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        chunks.append(tokenizer.decode(chunk_tokens))

    return chunks


text = "Artificial Intelligence is transforming the world. " * 20

chunks = fixed_token_chunking(text, chunk_size=100)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")
```

### Advantages
- Controls the number of tokens in each chunk.
- Useful for LLM context-window limits.
- More suitable than character-based chunking when token limits matter.

### Limitations
- Requires a tokenizer.
- Can split sentences and words.
- Different models may use different tokenizers.

### Use Cases
- LLM applications.
- RAG pipelines.
- Token-limit management.
- Embedding generation.

---

## 3. Overlapping Chunking

**Definition:** Creates chunks with some repeated text between consecutive chunks.

This helps preserve context when information spans across chunk boundaries.

### Python Implementation

```python
def overlapping_chunking(text, chunk_size=100, overlap=20):
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk_size")

    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]

        if chunk:
            chunks.append(chunk)

    return chunks


text = "Artificial Intelligence is transforming the world. " * 10

chunks = overlapping_chunking(
    text,
    chunk_size=100,
    overlap=20
)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")
```

### Advantages
- Preserves context between consecutive chunks.
- Reduces information loss at chunk boundaries.
- Useful for RAG and semantic search.

### Limitations
- Creates duplicate content.
- Increases storage requirements.
- Increases embedding costs.
- May retrieve redundant information.

### Use Cases
- Retrieval-Augmented Generation.
- Semantic search.
- Question-answering systems.

---

## 4. Non-Overlapping Chunking

**Definition:** Splits text into consecutive chunks without repeating any content.

### Python Implementation

```python
def non_overlapping_chunking(text, chunk_size=100):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    return chunks


text = "Artificial Intelligence is transforming the world. " * 10

chunks = non_overlapping_chunking(text, chunk_size=100)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")
```

### Advantages
- No duplicate content.
- Lower storage requirements.
- Faster processing.
- Easy to implement.

### Limitations
- Context can be lost at chunk boundaries.
- Related information may be separated.
- May reduce retrieval accuracy in RAG applications.

### Use Cases
- Basic document processing.
- Large-scale text preprocessing.
- Applications where duplicate content should be avoided.

---

## 5. Sentence-Based Chunking

**Definition:** Splits text using sentence boundaries instead of fixed character or token counts.

### Installation

```bash
pip install nltk
```

### Python Implementation

```python
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")

from nltk.tokenize import sent_tokenize

def sentence_based_chunking(text):
    return sent_tokenize(text)


text = """
Artificial Intelligence is transforming the world.
Machine learning is a subset of AI.
Deep learning uses neural networks.
"""

chunks = sentence_based_chunking(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")
```

### Advantages
- Preserves sentence boundaries.
- Produces readable chunks.
- Avoids splitting sentences in the middle.
- Useful for NLP and text analysis.

### Limitations
- Sentence lengths can vary significantly.
- Some sentences may exceed LLM token limits.
- A single sentence may not contain enough context.
- Sentence detection can be difficult with abbreviations.

### Use Cases
- FAQ systems.
- NLP applications.
- Text classification.
- Sentence-level semantic search.

---

## 6. Paragraph-Based Chunking

**Definition:** Splits text using paragraph boundaries, usually separated by blank lines.

### Python Implementation

```python
def paragraph_based_chunking(text):
    paragraphs = text.split("\n\n")

    return [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]


text = """
Artificial Intelligence is transforming the world.

Machine learning is a subset of Artificial Intelligence.

Deep learning uses neural networks to learn complex patterns.
"""

chunks = paragraph_based_chunking(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")
```

### Advantages
- Preserves document structure.
- Maintains related information together.
- Produces meaningful chunks.
- Useful for books, articles, reports, and documentation.

### Limitations
- Paragraph sizes can vary significantly.
- Large paragraphs may exceed LLM context limits.
- Some documents have inconsistent paragraph formatting.
- Small paragraphs may provide insufficient context.

### Use Cases
- Books and articles.
- PDF and DOCX processing.
- Documentation retrieval.
- Research papers.
- RAG applications.

---

## 7. Word-Based Chunking

**Definition:** Splits text based on a fixed number of words.

### Python Implementation

```python
def word_based_chunking(text, chunk_size=20):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


text = "Artificial Intelligence is transforming the world. " * 10

chunks = word_based_chunking(text, chunk_size=20)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")
```

### Advantages
- Easy to implement.
- Preserves complete words.
- More predictable than sentence-based chunking.
- Useful for basic NLP preprocessing.

### Limitations
- Does not preserve sentence boundaries.
- Word counts do not correspond directly to token counts.
- May split meaningful phrases and sentences.
- Different languages have different word-boundary rules.

### Use Cases
- Basic NLP preprocessing.
- Text analysis.
- Simple document processing.
- Word-frequency analysis.

---

## 8. Comparison of All Chunking Techniques

| Technique | Splitting Basis | Context Preservation | Complexity |
|---|---|---|---|
| Fixed-size | Characters | Low | Very Low |
| Fixed-token | Tokens | Low | Low |
| Overlapping | Characters or tokens | Better | Low |
| Non-overlapping | Characters or tokens | Low | Very Low |
| Sentence-based | Sentences | Moderate | Low |
| Paragraph-based | Paragraphs | Good | Very Low |
| Word-based | Words | Moderate | Very Low |

---

## 9. Which Chunking Technique Should You Use for RAG?

| Use Case | Recommended Technique |
|---|---|
| Simple document processing | Fixed-size |
| LLM token limit management | Fixed-token |
| General-purpose RAG | Overlapping |
| Avoiding duplicate embeddings | Non-overlapping |
| FAQ documents | Sentence-based |
| Books and articles | Paragraph-based |
| Basic NLP preprocessing | Word-based |

---

## 10. Hybrid Chunking Strategy for RAG

In practical RAG applications, multiple chunking techniques can be combined to improve retrieval quality.

### Recommended Workflow

1. Paragraph-based splitting: Identify meaningful sections.
2. Sentence-based splitting: Maintain sentence boundaries.
3. Token-based splitting: Enforce the desired token limit.
4. Overlapping chunking: Maintain contextual continuity.
5. Embedding generation: Convert chunks into vector embeddings.
6. Vector database: Store embeddings for semantic retrieval.

### Workflow Diagram

```text
             Input Document
           (PDF, DOCX, TXT, HTML)
                    |
                    v
          Paragraph-Based Splitting
                    |
                    v
           Sentence-Based Splitting
                    |
                    v
            Token-Based Splitting
                    |
                    v
              Overlapping Chunks
                    |
                    v
             Embedding Generation
                    |
                    v
              Vector Database
                    |
                    v
             Semantic Retrieval
                    |
                    v
                   LLM
                    |
                    v
               Final Answer
```

---

## 11. Recommended Starting Configuration for RAG

| Parameter | Suggested Value |
|---|---|
| Chunk size | 500 tokens |
| Chunk overlap | 50–100 tokens |
| Splitting preference | Paragraph → Sentence → Token limit |
| Embedding model | Consistent model for indexing and querying |
| Storage | Vector database |

**Note:** These are starting values, not universal optimal settings. Evaluate chunk sizes and overlap using actual documents and retrieval-quality tests.

---

## 12. Final Recommendation

Each chunking technique has its own advantages and limitations.

- Fixed-size chunking: Simple text processing.
- Fixed-token chunking: LLM context-window management.
- Overlapping chunking: Contextual continuity.
- Non-overlapping chunking: Avoiding duplicate content.
- Sentence-based chunking: Sentence-level meaning.
- Paragraph-based chunking: Preserving document structure.
- Word-based chunking: Basic NLP preprocessing.

