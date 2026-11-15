

# 1. Similarity Metrics in RAG

Three important mathematical techniques:

1. Cosine Similarity
2. Euclidean Distance
3. Dot Product

## 1.1 Cosine Similarity

Measures the angle between two vectors.

### Formula

Cosine Similarity = (A · B) / (|A| × |B|)

Where:
- A = Query embedding
- B = Document embedding
- A · B = Dot product
- |A| = Magnitude of vector A
- |B| = Magnitude of vector B

### Characteristics

- Range: -1 to +1
- Higher score generally means greater similarity.
- Independent of vector magnitude.
- Commonly used in semantic search.

## 1.2 Euclidean Distance

Measures the straight-line distance between two vectors.

### Formula

Distance = sqrt(sum((A[i] - B[i])²))

### Characteristics

- Range: 0 to infinity.
- Smaller distance means closer vectors.
- Sensitive to vector magnitude.
- Used in nearest-neighbor search.

## 1.3 Dot Product

Multiplies corresponding vector elements and adds the results.

### Formula

A · B = sum(A[i] × B[i])

### Characteristics

- Range: negative infinity to positive infinity.
- Considers vector magnitude and alignment.
- Higher score generally indicates stronger alignment.
- Used in vector retrieval.

---

# 2. Numerical Example

Consider:

Query A = [1, 2]

Document 1 B = [2, 4]

Document 2 C = [4, 2]

### 4.1 Cosine Similarity

For Document 1:

Dot product:

(1 × 2) + (2 × 4) = 10

Magnitude of A:

sqrt(1² + 2²) = sqrt(5)

Magnitude of B:

sqrt(2² + 4²) = sqrt(20)

Cosine:

10 / (sqrt(5) × sqrt(20)) = 1.0

For Document 2:

Dot product:

(1 × 4) + (2 × 2) = 8

Cosine:

8 / (sqrt(5) × sqrt(20)) = 0.8

Document 1 wins.

### 2.2 Euclidean Distance

For Document 1:

sqrt((1-2)² + (2-4)²)

= sqrt(1 + 4)

= 2.2361

For Document 2:

sqrt((1-4)² + (2-2)²)

= sqrt(9)

= 3.0

Document 1 wins because its distance is smaller.

### 2.3 Dot Product

For Document 1:

(1 × 2) + (2 × 4) = 10

For Document 2:

(1 × 4) + (2 × 2) = 8

Document 1 wins because its dot product is higher.

### Comparison

| Metric | Document 1 | Document 2 |
|---|---:|---:|
| Cosine | 1.0 | 0.8 |
| Euclidean | 2.2361 | 3.0 |
| Dot Product | 10 | 8 |

---

# 3. Why the Metrics Can Give Different Results

Consider:

Query A = [1, 2]

Document D = [10, 20]

Results:

- Cosine Similarity = 1.0
- Euclidean Distance = 20.1246
- Dot Product = 50

Explanation:

- Cosine similarity is 1 because both vectors have the same direction.
- Euclidean distance is high because their magnitudes differ.
- Dot product is high because the vectors have large magnitudes.

This demonstrates that the metrics measure different properties.

---

# 4. Comparison Table

| Feature | Cosine | Euclidean | Dot Product |
|---|---|---|---|
| Measures | Angle | Distance | Alignment and magnitude |
| Best direction | Higher | Lower | Higher |
| Range | -1 to 1 | 0 to infinity | -infinity to infinity |
| Magnitude sensitive | No | Yes | Yes |
| Used in RAG | Yes | Yes | Yes |

## Important Relationship

For normalized vectors:

Euclidean Distance² = 2 - 2 × Cosine Similarity

Therefore:

- Higher cosine similarity means smaller Euclidean distance.
- Cosine similarity and dot product produce the same ranking for normalized vectors.
- All three produce equivalent rankings for normalized vectors when Euclidean distance is ranked smallest first.


# Real-World Scenarios: When to Use Which Metric?

Choosing between Cosine Similarity, Euclidean Distance, and Dot Product depends on your embedding model, data characteristics, and retrieval requirements.

## Scenario 1: Semantic Search in a Chatbot

**Requirement:**

A user asks:

"What is artificial intelligence?"

Your knowledge base contains:
- Introduction to AI
- Machine Learning Basics
- History of Computers
- Python Programming

**Best choice: Cosine Similarity**

**Why?**
- We want to compare the meaning of the query and documents.
- We care about the direction of embeddings rather than their magnitude.
- It is commonly used for semantic search.

**Use case:**
- Customer support chatbots
- Document Q&A
- Knowledge-base search
- General-purpose RAG

---

## Scenario 2: Image Similarity Search

**Requirement:**

A user uploads an image of a dog.

The system searches an image database to find visually similar images using image embeddings.

**Best choice: Cosine Similarity**

**Why?**
- Image embeddings represent visual features.
- We want to find images with similar embedding directions.
- Magnitude differences may not be relevant to the task.

**Use case:**
- Image retrieval
- Visual search
- Product image matching
- Face embedding similarity systems

Note: The appropriate metric depends on how the image embedding model was trained.

---

## Scenario 3: Recommendation System

**Requirement:**

A recommendation system represents users and products as vectors.

It recommends products based on user preferences and learned vector representations.

**Best choice: Dot Product**

**Why?**
- Dot product considers both vector direction and magnitude.
- Vector magnitude can encode useful information, depending on the model.
- Many recommendation models are trained using dot-product scoring.

**Use case:**
- Product recommendations
- Movie recommendations
- Personalized content recommendations
- User-item matching

---

## Scenario 4: Geographical Location Search

**Requirement:**

A user searches for the nearest hospital.

The system has coordinates:

User = (2, 3)

Hospital A = (3, 4)

Hospital B = (8, 9)

**Best choice: Euclidean Distance**

**Why?**
- We want the closest point.
- Smaller distance means a closer location.
- It directly measures geometric distance in coordinate space.

**Use case:**
- Nearest-neighbor search
- Location-based applications
- Clustering
- Spatial data analysis

Note: For real geographical coordinates (latitude and longitude), use Haversine distance or a suitable geospatial distance method rather than ordinary Euclidean distance.

---

## Scenario 5: Large-Scale Vector Search

**Requirement:**

A RAG application contains millions of document embeddings.

The application needs fast retrieval using a vector database.

**Best choice: Dot Product or Cosine Similarity**

**Why?**
- Both are commonly supported by vector databases.
- Dot product is efficient for normalized vectors.
- Cosine similarity is useful when vector magnitude should not affect similarity.

**Use case:**
- Large-scale RAG
- Enterprise search
- Semantic document retrieval
- AI knowledge bases

Important: Choose the metric compatible with your embedding model and indexing configuration.

---

## Scenario 6: Comparing Normalized Embeddings

**Requirement:**

Your embedding model generates normalized vectors.

You need to retrieve the most similar documents.

**Best choice: Cosine Similarity or Dot Product**

**Why?**

For normalized vectors:

Cosine Similarity = Dot Product

And:

Euclidean Distance² = 2 - 2 × Cosine Similarity

All three metrics produce equivalent rankings if Euclidean distance is ranked in ascending order.

---

#  Scenario Comparison Table

| Scenario | Recommended Metric | Reason |
|---|---|---|
| RAG chatbot | Cosine | Semantic similarity |
| Document search | Cosine | Meaning-based retrieval |
| Image search | Cosine | Embedding direction |
| Product recommendation | Dot Product | Alignment and magnitude |
| Movie recommendation | Dot Product | Learned user-item scoring |
| Nearest hospital | Euclidean* | Geometric distance |
| Clustering | Euclidean* | Point-to-point distance |
| Normalized embeddings | Cosine / Dot Product | Equivalent ranking |
| Large-scale RAG | Cosine / Dot Product | Efficient vector retrieval |

*For geographical or other specialized data, use an appropriate distance metric.

---

# How to Choose the Right Metric?

Follow this decision guide:

1. Do you want to compare semantic meaning?
   - Choose Cosine Similarity.

2. Do you want to find the closest points in numerical space?
   - Choose Euclidean Distance.

3. Does your model intentionally encode useful information in vector magnitude?
   - Consider Dot Product.

4. Are your vectors normalized?
   - Cosine and Dot Product give the same ranking.

5. Does your embedding model or vector database specify a recommended metric?
   - Follow the model's recommended configuration.

---

#  Practical RAG Recommendation

For a typical document-based RAG application:

| Component | Recommendation |
|---|---|
| Embedding generation | Embedding model |
| Vector storage | FAISS, Chroma, Pinecone, or Qdrant |
| Initial retrieval | Cosine Similarity or compatible metric |
| Candidate selection | Top-K retrieval |
| Improved relevance | Cross-encoder reranking |
| Answer generation | LLM |

### Example RAG Architecture

User Query
    |
    v
Embedding Model
    |
    v
Query Vector
    |
    v
Vector Database
    |
    v
Cosine Similarity Search
    |
    v
Top-K Documents
    |
    v
Reranking
    |
    v
LLM
    |
    v
Final Answer

---


**Question: When would you use Cosine Similarity, Euclidean Distance, and Dot Product?**

**Answer:**

I would use Cosine Similarity when I want to compare semantic meaning between query and document embeddings, particularly in RAG applications.

I would use Euclidean Distance when I need to calculate the actual distance between numerical vectors, such as nearest-neighbor or spatial searches.

I would use Dot Product when both vector direction and magnitude are meaningful, such as in recommendation systems.

For a typical RAG application, I would start with Cosine Similarity, provided it is compatible with the embedding model. If embeddings are normalized, Dot Product is also a suitable option because both produce the same ranking.

