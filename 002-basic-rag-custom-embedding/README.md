# Semantic Search Using Text Embeddings

## 1. Concept

### What is an Embedding?

An embedding is a numerical representation of text (words, sentences, or documents) in the form of a vector (a list of numbers).

Machine learning models cannot directly understand the meaning of human language. They work with numerical data.

Embeddings convert text into numerical vectors so that machines can compare the semantic meaning of different texts.

For example:

```python
text1 = "How many days can I work from home?"
text2 = "What is the remote working policy?"
text3 = "How many vacation days do employees get?"
```

Although `text1` and `text2` use different words, they have similar meanings.

An embedding model converts each sentence into a numerical vector:

```python
[
    0.123,
    -0.456,
    0.789,
    0.234,
    ...
]
```

Sentences with similar meanings generally have vectors that are close to each other in the embedding space.

### How does an Embedding Model work?

1. Input text is passed to an embedding model.
2. The model tokenizes the text into tokens.
3. Tokens are converted into numerical representations.
4. The neural network processes these representations and captures contextual meaning.
5. The model generates a fixed-dimensional vector representing the input text.

In this implementation, we use:

`all-MiniLM-L6-v2`

This Sentence Transformer model generates a **384-dimensional vector** for each sentence.

### What is Vector Dimension?

A dimension represents one numerical component of an embedding vector.

For example:

```python
embedding = [0.12, 0.45, -0.32, 0.67]
```

This vector has 4 dimensions.

Our model produces 384 dimensions:

```python
embedding = embedding_model.encode("Remote work policy")

print(len(embedding))

# Output: 384
```

Each dimension contributes to the learned representation of the text. Individual dimensions generally do not have directly interpretable meanings.

### How do we compare Embeddings?

To identify similar text, we calculate the similarity between vectors.

## 2. Why do we need it?

Traditional keyword search looks for matching words.

For example:

Question:

```text
How many days can I work from home?
```

Document:

```text
Remote work is allowed up to 3 days per week with manager approval.
```

A keyword-based search might struggle because the question and document use different words.

Embedding-based semantic search understands similarities in meaning rather than relying only on matching keywords.

### Problems solved by Embeddings

1. Understand semantic similarity between sentences.
2. Retrieve documents even when exact keywords are different.
3. Improve document retrieval in RAG applications.
4. Support natural-language questions.
5. Enable semantic search across large document collections.

---

## 3. Architecture

### Embedding-Based RAG Workflow

```text
             DOCUMENTS
                 |
                 v
          Embedding Model
                 |
                 v
        Document Embeddings
                 |
                 v
             ChromaDB
           (Vector Storage)
                 |
                 |
USER QUESTION   |
      |          |
      v          |
 Embedding Model |
      |          |
      v          |
Query Embedding  |
      |          |
      v          v
      Semantic Search
                 |
                 v
       Relevant Documents
                 |
                 v
          Prompt Creation
                 |
                 v
            Ollama LLM
                 |
                 v
          Generated Answer
```

### Workflow Explanation

**Step 1: Initialize Embedding Model**

Load the Sentence Transformer model.

**Step 2: Generate Document Embeddings**

Convert each company policy into a 384-dimensional vector.

**Step 3: Store Embeddings**

Store document text, IDs, and embeddings in ChromaDB.

**Step 4: Generate Query Embedding**

Convert the user's question into an embedding using the same model.

**Step 5: Semantic Search**

ChromaDB compares the query embedding against stored document embeddings and retrieves relevant documents.

**Step 6: Generate Answer**

Pass the retrieved documents and original question to the Llama 3.2 model.

---

## 4. Implementation

### Complete Python Implementation

```python
import chromadb
import ollama

from sentence_transformers import SentenceTransformer

# Initialize ChromaDB
chromadb_client = chromadb.Client()

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create collection
collection = chromadb_client.create_collection(
    "company_policies"
)

# Company policy documents
documents = [
    "Employees get 25 days of paid time off (PTO) per year.",
    "Remote work is allowed up to 3 days per week with manager approval.",
    "Expense reports must be submitted within 14 days of travel."
]

# Generate embeddings for documents
documents_embedding = embedding_model.encode(
    documents
).tolist()

# Unique document IDs
ids = [
    "policy_01",
    "policy_02",
    "policy_03"
]

# Store documents and embeddings
collection.upsert(
    ids=ids,
    embeddings=documents_embedding,
    documents=documents
)

# User question
user_question = "How many days can I work from home?"

# Generate query embedding
question_embedding = embedding_model.encode(
    user_question
).tolist()

# Search relevant documents
result = collection.query(
    query_embeddings=[question_embedding],
    n_results=2
)

# Retrieve matched documents
retrieved_documents = result["documents"][0]

# Create prompt
prompt = f"""
Answer the question using only the context below.

Context:
{retrieved_documents}

Question:
{user_question}
"""

# Generate answer using Ollama
response = ollama.generate(
    model="llama3.2:latest",
    prompt=prompt
)

print(response["response"])
```

### Important Embedding Operations

**Generate an embedding:**

```python
embedding = embedding_model.encode(
    "Remote work policy"
)

print(embedding)
```

**Check embedding dimensions:**

```python
print(len(embedding))

# Output: 384
```

**Generate multiple embeddings:**

```python
texts = [
    "Remote work policy",
    "Employee vacation policy",
    "Expense reimbursement policy"
]

embeddings = embedding_model.encode(texts)

print(embeddings.shape)

# Output: (3, 384)
```

**Convert embeddings into a Python list:**

```python
embeddings = embedding_model.encode(texts).tolist()
```

---

## 5. How to run

### Prerequisites

- Python 3.10+
- Ollama installed
- ChromaDB
- Sentence Transformers
- Llama 3.2 model

### Step 1: Install dependencies

```bash
pip install chromadb sentence-transformers ollama
```

### Step 2: Download Ollama model

```bash
ollama pull llama3.2:latest
```

### Step 3: Start Ollama

```bash
ollama serve
```

### Step 4: Execute Python script

```bash
python main.py
```

### Expected Output

```text
Remote work is allowed up to 3 days per week,
with manager approval.
```

---

## 6. Test cases

Test different questions to understand how semantic search works.

| No. | Test Query | Expected Relevant Policy |
|---|---|---|
| 1 | How many days can I work from home? | Remote work |
| 2 | What is the company's work-from-home policy? | Remote work |
| 3 | Can I work remotely three days a week? | Remote work |
| 4 | How much annual vacation do employees receive? | PTO |
| 5 | How many paid holidays are available? | PTO |
| 6 | When should I submit my travel expenses? | Expense reports |
| 7 | What is the deadline for expense reimbursement? | Expense reports |
| 8 | Can I work from another location? | Remote work |
| 9 | How many days of leave do I get annually? | PTO |
| 10 | What happens if I submit expenses late? | Expense reports |

Note: These are semantic retrieval test queries. Some questions introduce information not explicitly covered by the policies. The retrieval result should not be treated as proof that the policy supports that information.

---

## 7. Results

### Expected Results

For the query:

```text
How many days can I work from home?
```

The relevant retrieved document should be:

```text
Remote work is allowed up to 3 days per week
with manager approval.
```

The LLM should generate an answer similar to:

```text
You can work remotely up to 3 days per week
with manager approval.
```

### What happens internally?

1. The question becomes a 384-dimensional vector.
2. ChromaDB compares it against the stored document vectors.
3. The most semantically relevant document is retrieved.
4. The retrieved document becomes context for the LLM.
5. The LLM generates a natural-language answer.

---

## 8. Comparison

| Feature | Keyword Search | Embedding Search |
|---|---|---|
| Search method | Word matching | Semantic similarity |
| Understands meaning | Limited | Yes |
| Different wording | May fail | Often works |
| Synonyms | Limited | Better support |
| Numerical representation | Not required | Required |
| Semantic retrieval | No | Yes |
| RAG integration | Possible | Commonly used |

### Comparison with Previous Implementations

**Previous: Basic RAG**

- Introduced document ingestion.
- Generated embeddings.
- Stored vectors.
- Retrieved relevant context.
- Generated answers using an LLM.

**Current: Understanding Embeddings**

- Focuses on how text becomes numerical vectors.
- Understands embedding dimensions.
- Explores vector similarity.
- Understands cosine similarity, dot product, and Euclidean distance.
- Explores how semantic retrieval works internally.

---

## 9. Learnings

1. An embedding is a numerical representation of text.
2. Embedding models convert sentences into vectors.
3. `all-MiniLM-L6-v2` generates 384-dimensional embeddings.
4. Similar meanings generally produce nearby vectors in embedding space.
5. Different sentences can have similar embeddings even when they use different words.
6. Cosine similarity measures the directional similarity between vectors.
7. Dot product measures vector alignment and magnitude.
8. Euclidean distance measures the distance between vectors.
9. ChromaDB stores and retrieves document embeddings.
10. Query and document embeddings must use compatible embedding models and configurations.
11. Embeddings are the foundation of semantic search and modern RAG architectures.

### Key Takeaway

**Embeddings bridge the gap between human language and machine-readable numerical representations, allowing RAG systems to retrieve information based on meaning rather than exact keyword matches.**