
# RAG Search Techniques – Complete Practical Guide

## Table of Contents

1. Introduction to RAG Search
2. Traditional Search Techniques
3. Statistical Keyword Search
4. Sparse Vector Search
5. Dense Vector Search
6. Similarity and Distance-Based Search
7. Hybrid Search
8. Nearest Neighbor Search
9. Approximate Nearest Neighbor Search
10. Graph-Based Search
11. Specialized Search Techniques
12. Advanced Search Techniques
13. Search Optimization Techniques
14. Real-World Scenario Comparison
15. How to Choose the Right Search Technique
16. Search Technique Comparison

---

# 1. Introduction to RAG Search

RAG Search is the process of finding relevant information from a knowledge base based on a user's query.

The knowledge base may contain:

- PDFs
- Word documents
- Websites
- Databases
- Technical documentation
- Research papers
- Company policies
- Product catalogs

### Basic RAG Search Architecture

```text
User Query
    |
    v
Query Processing
    |
    v
Search Engine
    |
    v
Search Knowledge Base
    |
    v
Relevant Results
    |
    v
LLM
    |
    v
Final Answer
```

### Main Categories of RAG Search

1. Keyword Search
2. Statistical Search
3. Sparse Vector Search
4. Dense Vector Search
5. Similarity-Based Search
6. Hybrid Search
7. Nearest Neighbor Search
8. Graph-Based Search
9. Multimodal Search
10. Advanced Search Techniques

---

# 2. Traditional Search Techniques

## 2.1 Exact Match Search

### What is it?

Exact Match Search finds documents containing an exact word, phrase, identifier, or value.

### Example

User query:

`GPT-4o`

Knowledge base:

- Document A: Introduction to GPT-4o
- Document B: Introduction to GPT-4
- Document C: Introduction to Generative AI

Document A is the exact match.

### Purpose

Find information where exact matching is important.

### Best Real-World Scenarios

- Product ID searches
- Employee ID searches
- Invoice numbers
- Error codes
- API endpoint searches
- Version-specific documentation

### Advantages

- Accurate for exact identifiers.
- Simple to implement.
- Fast.

### Limitations

- Cannot understand semantic meaning.
- May miss synonyms and alternative wording.

---

## 2.2 Keyword Search

### What is it?

Keyword Search finds documents based on the occurrence of words in a user's query.

### Example

User query:

`Python inheritance`

Knowledge base:

- Document A: Python inheritance concepts.
- Document B: Java programming.
- Document C: Machine learning basics.

Document A receives a strong match because it contains both keywords.

### Purpose

Find documents containing specific words or phrases.

### Best Real-World Scenarios

- Technical documentation
- Legal document search
- Product catalog search
- Company policy search
- Error message search

### Advantages

- Simple and efficient.
- Works well for exact terminology.
- Does not require an embedding model.

### Limitations

- Does not understand context.
- May miss semantically similar documents.

---

## 2.3 Boolean Search

### What is it?

Boolean Search uses logical operators to combine search terms.

Common operators:

- AND
- OR
- NOT

### Example

Query:

`Python AND Machine Learning`

Finds documents containing both terms.

Query:

`Python OR Java`

Finds documents containing either term.

Query:

`Python NOT Java`

Finds documents containing Python but excluding Java.

### Purpose

Create precise search conditions.

### Best Real-World Scenarios

- Legal research
- Academic research
- Enterprise document search
- Compliance searches

### Advantages

- Precise filtering.
- Supports complex search conditions.
- Easy to understand.

### Limitations

- Requires carefully constructed queries.
- Does not understand semantic relationships.

---

## 2.4 Full-Text Search

### What is it?

Full-Text Search searches the complete textual content of documents rather than only searching titles or metadata.

It commonly uses an inverted index.

### Example

Suppose a PDF contains:

"Retrieval-Augmented Generation combines external knowledge with large language models."

User query:

`external knowledge`

Full-text search locates the document because those terms occur in its content.

### Purpose

Search large collections of textual documents efficiently.

### Best Real-World Scenarios

- Enterprise document management
- PDF search engines
- Website search
- Product documentation
- Knowledge portals

### Popular Technologies

- Elasticsearch
- Apache Solr
- PostgreSQL Full-Text Search
- OpenSearch

### Advantages

- Efficient textual search.
- Supports phrase matching and filters.
- Suitable for large text collections.

### Limitations

- Usually relies on lexical matching.
- May miss documents using different terminology.

---

# 3. Statistical Keyword Search

## 3.1 TF-IDF Search

### What is it?

TF-IDF stands for Term Frequency-Inverse Document Frequency.

It measures the importance of a word in a document relative to a document collection.

It has two components.

### Term Frequency (TF)

Measures how frequently a term appears in a document.

### Inverse Document Frequency (IDF)

Measures how rare or informative a term is across the document collection.

### Formula

$$
TF\text{-}IDF = TF \times IDF
$$

A common IDF formula:

$$
IDF(t) = \log\left(\frac{N}{DF(t)}\right)
$$

Where:

- N = Total number of documents.
- DF(t) = Number of documents containing term t.

### Example

Suppose we have 100 documents.

The word "Python" occurs in 10 documents.

$$
IDF(Python) = \log(100/10)
$$

$$
IDF(Python) = \log(10)
$$

A term appearing in fewer documents generally receives a higher IDF score.

### Purpose

Identify documents containing important query terms.

### Best Real-World Scenarios

- Small document collections
- Research paper search
- News article search
- Basic search engines
- Text classification

### Advantages

- Simple mathematical approach.
- Easy to implement.
- Does not require machine learning embeddings.

### Limitations

- Does not understand word meaning.
- Does not naturally understand synonyms.
- May perform poorly for complex semantic queries.

---

## 3.2 BM25 Search

### What is it?

BM25 stands for Best Matching 25.

It is a probabilistic ranking algorithm used in information retrieval.

It ranks documents based on:

- Term frequency
- Inverse document frequency
- Document length
- Query term matching

### Example

User query:

`Python inheritance`

Documents:

- Document A: Python inheritance and object-oriented programming.
- Document B: Python introduction.
- Document C: Java inheritance.

BM25 scores documents based on how well their terms match the query.

### Purpose

Rank documents based on keyword relevance.

### Best Real-World Scenarios

- Enterprise search
- Technical documentation
- E-commerce product search
- Legal document search
- Error code search
- RAG keyword retrieval

### Advantages

- Strong keyword relevance.
- Considers document length.
- Efficient with inverted indexes.
- Does not require embeddings.

### Limitations

- Does not fully understand semantic meaning.
- Synonyms may be missed without additional processing.

### Popular Implementations

- Elasticsearch
- OpenSearch
- Apache Lucene
- Whoosh
- rank-bm25

---

# 4. Sparse Vector Search

## 4.1 What is Sparse Vector Search?

Sparse Vector Search represents documents using vectors where most values are zero.

Each dimension represents a term or feature.

### Example

Vocabulary:

`[Python, Java, AI, RAG, Database]`

Document:

`Python and AI`

Sparse vector:

`[1, 0, 1, 0, 0]`

The vector contains mostly zero values.

### Purpose

Represent textual information using sparse numerical vectors and perform efficient lexical or learned sparse matching.

### Best Real-World Scenarios

- Technical documentation
- Exact terminology search
- Large document collections
- Enterprise knowledge bases

### Popular Approaches

- TF-IDF
- BM25
- SPLADE

---

## 4.2 SPLADE Search

### What is it?

SPLADE is a learned sparse retrieval technique.

It uses language models to generate sparse representations that can include semantically related terms.

### Example

Query:

`Car repair`

A learned sparse representation may activate related terms such as:

- Automobile
- Vehicle
- Maintenance
- Repair

### Purpose

Combine lexical matching with learned semantic expansion.

### Best Real-World Scenarios

- Technical document search
- Enterprise search
- Domain-specific information retrieval

### Advantages

- Sparse representation.
- Can capture related terminology.
- Supports inverted-index search.

### Limitations

- Requires a trained model.
- More complex than traditional BM25.

---

# 5. Dense Vector Search

## 5.1 Semantic Search

### What is it?

Semantic Search finds documents based on meaning rather than requiring exact keyword matches.

It uses embedding models to convert text into dense numerical vectors.

### Example

User query:

"How can I make my computer faster?"

Document:

"Methods for improving system performance and reducing processing time."

The wording is different, but the meanings are related.

Semantic search can retrieve the document.

### How It Works

1. Convert documents into embeddings.
2. Store embeddings in a vector database.
3. Convert the user query into an embedding.
4. Compare query and document vectors.
5. Return the most relevant results.

### Purpose

Find conceptually related documents.

### Best Real-World Scenarios

- AI chatbots
- Semantic document search
- Customer support
- Knowledge-base Q&A
- Research assistants

### Popular Embedding Models

- OpenAI Embeddings
- Sentence Transformers
- BGE Embeddings
- E5 Embeddings

### Popular Vector Databases

- FAISS
- Chroma
- Pinecone
- Qdrant
- Weaviate
- Milvus

### Advantages

- Understands semantic similarity.
- Can retrieve documents with different wording.
- Useful for natural-language queries.

### Limitations

- May miss exact identifiers.
- Quality depends on embedding models.
- Requires embedding generation and vector indexing.

---

## 5.2 Dense Vector Search

Dense Vector Search represents each document using a dense numerical vector.

### Example

Document:

"Machine Learning is a branch of Artificial Intelligence."

Embedding:

`[0.21, 0.48, 0.12, 0.87, 0.34]`

These values represent learned semantic features.

The query is also converted into a vector and compared with document vectors.

### Purpose

Retrieve semantically similar documents using vector representations.

### Best Real-World Scenarios

- Semantic search
- RAG applications
- Recommendation systems
- Similar document discovery
- Natural-language search

---


# 6. Similarity and Distance-Based Search

These techniques compare query and document embeddings to identify similar information.

## 6.1 Cosine Similarity

### What is it?

Cosine Similarity measures the angle between two vectors.

It determines how similar two vectors are based on their direction, rather than their magnitude.

### Formula

$$
Cosine(A,B) = \frac{A \cdot B}{|A| \times |B|}
$$

Where:

- A and B = Two vectors.
- A · B = Dot product.
- |A| and |B| = Vector magnitudes.

### Example

Query vector:

```python
A = [1, 2]
```

Document vector:

```python
B = [2, 4]
```

Dot product:

```text
(1 × 2) + (2 × 4) = 10
```

Magnitudes:

```text
|A| = sqrt(5)
|B| = sqrt(20)
```

Cosine Similarity:

```text
10 / (sqrt(5) × sqrt(20)) = 1.0
```

The vectors have identical directions, so their cosine similarity is 1.

### Purpose

Measure directional similarity between embeddings.

### Best Real-World Scenarios

- Semantic document search
- Chatbot knowledge bases
- Similarity search
- Image embedding search
- Recommendation systems

### Advantages

- Focuses on vector direction.
- Works well with normalized embeddings.
- Commonly supported by vector databases.

### Limitations

- Does not consider magnitude differences.
- Results depend on embedding quality.
- A high similarity score does not guarantee factual relevance.

---

## 6.2 Euclidean Distance

### What is it?

Euclidean Distance measures the straight-line distance between two vectors.

### Formula

$$
D(A,B) = \sqrt{\sum_{i=1}^{n}(A_i-B_i)^2}
$$

### Example

Vector A:

```python
A = [1, 2]
```

Vector B:

```python
B = [2, 4]
```

Calculation:

```text
sqrt((1-2)^2 + (2-4)^2)

= sqrt(1 + 4)

= sqrt(5)

= 2.236
```

### Purpose

Measure numerical distance between vectors.

### Best Real-World Scenarios

- Nearest-neighbor search
- Clustering
- Spatial search
- Numerical feature comparison
- Geometric similarity applications

### Advantages

- Easy to understand.
- Represents actual geometric distance.
- Useful for numerical data.

### Limitations

- Sensitive to vector scale.
- Can be affected by differences in magnitude.
- Requires suitable normalization when dimensions have different scales.

---

## 6.3 Dot Product Search

### What is it?

Dot Product multiplies corresponding vector elements and adds the results.

### Formula

$$
A \cdot B = \sum_{i=1}^{n} A_i B_i
$$

### Example

Vector A:

```python
A = [1, 2]
```

Vector B:

```python
B = [2, 4]
```

Calculation:

```text
(1 × 2) + (2 × 4)

= 2 + 8

= 10
```

### Purpose

Measure vector alignment while considering magnitude.

### Best Real-World Scenarios

- Vector databases
- Recommendation systems
- Embedding similarity search
- Machine learning ranking models
- Maximum inner product search

### Advantages

- Computationally efficient.
- Useful for learned embedding models.
- Works well when vector magnitude carries useful information.

### Limitations

- Scores are affected by vector magnitude.
- Scores are not necessarily normalized.
- Different embedding models may produce different score distributions.

---

## 6.4 Cosine Similarity vs Euclidean Distance vs Dot Product

| Metric | Measures | Best Score | Magnitude Sensitive |
|---|---|---|---|
| Cosine Similarity | Vector direction | Higher | No |
| Euclidean Distance | Vector distance | Lower | Yes |
| Dot Product | Alignment and magnitude | Higher | Yes |

### When Should You Choose Which?

**Cosine Similarity**

Choose when vector direction matters more than magnitude.

Example:
- Semantic search
- Document similarity
- Text embeddings

**Euclidean Distance**

Choose when actual geometric distance matters.

Example:
- Spatial data
- Clustering
- Numerical feature comparison

**Dot Product**

Choose when vector magnitude contains useful information.

Example:
- Recommendation systems
- Learned ranking models
- Maximum inner product search

### Important Point

For normalized vectors:

$$
Cosine(A,B) = A \cdot B
$$

Also:

$$
||A-B||^2 = 2(1-Cosine(A,B))
$$

Therefore, cosine similarity, dot product and Euclidean distance produce equivalent rankings for unit-normalized vectors, with Euclidean distance ranked in the opposite direction.

---

# 7. Hybrid Search

## 7.1 What is Hybrid Search?

Hybrid Search combines two or more search methods.

The most common combination in RAG is:

1. Keyword Search (BM25)
2. Semantic Search (Dense Vector Search)

Keyword Search finds exact terms.

Semantic Search finds conceptually related information.

Hybrid Search combines both results to improve search coverage.

### Example

User query:

```text
GPT-4 architecture
```

Knowledge base:

Document A:
"GPT-4 technical architecture and model design."

Document B:
"Transformer-based neural network architecture."

Document C:
"Introduction to GPT-4."

BM25 Search:

- Finds documents containing GPT-4 and architecture.

Semantic Search:

- Finds documents discussing transformer models and neural network design.

Hybrid Search:

- Combines the results to identify documents relevant to both exact terminology and meaning.

### Purpose

Improve search relevance by combining lexical matching with semantic understanding.

### Best Real-World Scenarios

- Enterprise RAG
- Technical documentation
- Research papers
- Product catalogs
- Large knowledge bases
- Customer support applications

### Advantages

- Exact keyword matching.
- Semantic understanding.
- Can improve recall.
- Handles different query types.

### Limitations

- Requires score or rank combination.
- More complex than a single search method.
- Needs evaluation and tuning.

---

## 7.2 Reciprocal Rank Fusion (RRF)

### What is it?

RRF is a ranking algorithm that combines results from multiple search systems.

For example:

- BM25 results
- Dense vector search results
- Other ranked search results

It combines rankings rather than requiring the original scores to be directly comparable.

### Formula

$$
RRF(d) = \sum_{r \in R} \frac{1}{k + rank_r(d)}
$$

Where:

- d = Document.
- R = Set of ranking lists.
- k = Ranking constant.
- rank(d) = Document position in a result list.

### Example

BM25 ranking:

```text
1. Document A
2. Document B
3. Document C
```

Vector ranking:

```text
1. Document B
2. Document C
3. Document A
```

RRF calculates a combined ranking score.

Assume k = 60.

Document A:

```text
1/(60+1) + 1/(60+3)
```

Document B:

```text
1/(60+2) + 1/(60+1)
```

Document C:

```text
1/(60+3) + 1/(60+2)
```

Document B receives the highest combined score.

### Purpose

Combine multiple search rankings without requiring their original scores to share the same scale.

### Best Real-World Scenarios

- Hybrid Search
- Enterprise search
- Multiple search engines
- Combining lexical and semantic results
- Search systems using multiple retrieval models

### Advantages

- Does not require score normalization.
- Easy to combine different ranked lists.
- Often improves hybrid search performance.

### Limitations

- Does not directly use the original relevance scores.
- Ranking quality depends on the input result lists.
- Requires tuning the ranking constant.

---

## 7.3 Weighted Hybrid Search

### What is it?

Weighted Hybrid Search combines keyword and vector scores using configurable weights.

### Formula

$$
HybridScore = \alpha \times VectorScore
+ \beta \times KeywordScore
$$

Where:

- VectorScore = Semantic similarity score.
- KeywordScore = Keyword relevance score.
- α = Vector weight.
- β = Keyword weight.

Usually, the scores need normalization or calibration before combining.

### Example

Suppose:

```text
Vector Score = 0.90
Keyword Score = 0.80

Vector Weight = 0.60
Keyword Weight = 0.40
```

Calculation:

```text
Hybrid Score = (0.60 × 0.90) + (0.40 × 0.80)

Hybrid Score = 0.54 + 0.32

Hybrid Score = 0.86
```

### Purpose

Control the relative importance of semantic and lexical relevance.

### Best Real-World Scenarios

- Technical documentation
- Enterprise knowledge search
- Product search
- Domain-specific RAG applications

### When to use it?

Use a higher keyword weight when:

- Exact product codes matter.
- Error messages must match.
- Technical identifiers are important.

Use a higher vector weight when:

- Natural-language questions are common.
- Users describe concepts differently.
- Semantic understanding is important.

---

# 8. Nearest Neighbor Search

## 8.1 K-Nearest Neighbors (KNN)

### What is it?

KNN finds the K closest vectors to a query vector.

It uses a distance or similarity metric to identify the nearest vectors.

### Example

User query embedding:

```python
query = [1, 2]
```

Document embeddings:

```python
A = [2, 4]
B = [4, 2]
C = [1, 3]
```

If K = 2, the search engine returns the two closest documents according to the selected metric.

### Purpose

Retrieve the K nearest vectors from a dataset.

### Best Real-World Scenarios

- Small vector databases
- Document similarity
- Recommendation systems
- Classification
- Nearest-neighbor search

### Advantages

- Simple to understand.
- Exact results are possible when all vectors are compared.
- Useful for smaller datasets.

### Limitations

- Can become expensive for very large datasets.
- Requires distance calculations against many vectors in exact search.

---

## 8.2 Exact Nearest Neighbor Search

### What is it?

Exact Nearest Neighbor Search calculates the distance or similarity between the query and every vector in the dataset.

### Example

Suppose there are 10,000 document vectors.

The query is compared against all 10,000 vectors.

The top 5 results are returned.

### Purpose

Obtain exact nearest-neighbor results.

### Best Real-World Scenarios

- Small and medium datasets.
- Benchmarking approximate algorithms.
- High-accuracy search requirements.
- Search systems where exhaustive comparison is affordable.

### Advantages

- Finds the true nearest neighbors under the selected metric.
- No approximation error from the index.
- Useful as a benchmark for ANN systems.

### Limitations

- Computational cost increases with dataset size.
- Can be expensive for high-dimensional vectors.
- May not meet strict latency requirements at large scale.

---

## 8.3 KNN vs Exact Nearest Neighbor Search

| Feature | KNN | Exact Nearest Neighbor |
|---|---|---|
| Main purpose | Find K closest items | Find exact nearest items |
| Result count | K | Usually K |
| Search method | Can be exact or approximate | Exhaustive or otherwise exact |
| Use case | Similarity search | Ground-truth nearest neighbors |

Important: KNN describes the task of finding K neighbors. Exact and approximate methods describe how that task is performed.

---


# 9. Approximate Nearest Neighbor Search (ANN)

## 9.1 What is ANN Search?

ANN stands for Approximate Nearest Neighbor.

ANN algorithms find vectors that are approximately close to a query vector without comparing the query against every vector in the database.

They trade a controlled amount of search accuracy for faster search and lower computational cost.

### Example

Suppose a vector database contains 1 million document embeddings.

Exact Search:

- Compares the query against all 1 million vectors.
- Finds the exact nearest neighbors.

ANN Search:

- Uses an optimized index.
- Searches a smaller candidate set.
- Returns approximately the nearest neighbors.

### Purpose

To improve vector search performance on large datasets.

### Best Real-World Scenarios

- Large RAG applications
- Real-time semantic search
- Product recommendation
- Image similarity search
- Large-scale vector databases

### Advantages

- Faster than exhaustive search in many workloads.
- Suitable for large vector collections.
- Reduces search computations.

### Limitations

- May miss some true nearest neighbors.
- Requires index configuration.
- Search quality and latency need evaluation.

---

## 9.2 HNSW Search

### What is it?

HNSW stands for Hierarchical Navigable Small World.

It is a graph-based Approximate Nearest Neighbor algorithm.

HNSW organizes vectors into multiple graph layers.

The search starts at an upper layer and navigates toward increasingly relevant vectors in lower layers.

### How It Works

1. Create a graph containing vector nodes.
2. Connect nearby vectors.
3. Organize connections into multiple layers.
4. Start searching from an upper layer.
5. Navigate toward closer vectors.
6. Search the bottom layer for final candidates.
7. Return the nearest results.

### Example

Suppose a database contains 1 million document embeddings.

Instead of comparing the query against every vector, HNSW navigates through a graph to identify nearby vectors.

### Simplified Architecture

```text
          Layer 2
             A
            / \
           B   C
          / \   \
         D   E   F

          Layer 1
       A -- B -- C
       |  / | \  |
       D -- E -- F
        \   |   /
          G -- H

          Layer 0
       More detailed
       vector connections
```

### Purpose

Provide fast approximate nearest-neighbor search.

### Best Real-World Scenarios

- Large RAG applications
- Real-time semantic search
- Product recommendation
- High-dimensional vector search
- Chatbot knowledge bases

### Advantages

- Fast search.
- High recall when properly configured.
- Supports large vector collections.
- Effective for many real-time search workloads.

### Limitations

- Uses additional memory.
- Index construction requires resources.
- Search parameters need tuning.
- Index updates can have operational costs.

### Important Parameters

- M: Maximum number of connections per node.
- efConstruction: Controls index construction quality and effort.
- efSearch: Controls search-time exploration.

### When to Use HNSW?

Use HNSW when:

- Low search latency is important.
- You have sufficient memory.
- Your dataset is large.
- You need good approximate recall.

---

## 9.3 IVF Search

### What is it?

IVF stands for Inverted File Index.

It divides vectors into clusters using a clustering algorithm, commonly K-means.

During search, it identifies relevant clusters and searches within selected clusters instead of examining the entire dataset.

### How It Works

1. Divide vectors into clusters.
2. Calculate cluster centroids.
3. Assign each vector to a cluster.
4. Convert the query into a vector.
5. Find the nearest cluster centroids.
6. Search vectors inside selected clusters.
7. Return the nearest candidates.

### Example

Suppose a database contains 1 million vectors.

IVF organizes them into 1,000 clusters.

The query searches selected clusters rather than all 1 million vectors.

### Simplified Architecture

```text
             Query
               |
               v
        Find Nearest Clusters
               |
       +-------+-------+
       |       |       |
       v       v       v
    Cluster  Cluster  Cluster
       1        5       12
       |        |        |
       v        v        v
    Search   Search   Search
    Vectors  Vectors  Vectors
       |        |        |
       +--------+--------+
                |
                v
          Top Results
```

### Purpose

Reduce the number of vectors examined during search.

### Best Real-World Scenarios

- Large vector databases
- Memory-conscious search systems
- High-volume semantic search
- Large document collections

### Advantages

- Reduces search computations.
- Supports large datasets.
- Offers configurable speed and recall trade-offs.

### Limitations

- Requires clustering.
- Search quality depends on index configuration.
- Poor cluster selection can miss relevant vectors.
- Training and maintenance may be required.

### Important Parameters

- nlist: Number of clusters.
- nprobe: Number of clusters searched during a query.

### When to Use IVF?

Use IVF when:

- You have a very large vector dataset.
- You want to control search effort.
- Memory and search performance need balancing.
- You can tolerate approximate results.

---

## 9.4 Product Quantization (PQ)

### What is it?

Product Quantization compresses vectors by dividing them into smaller subvectors and quantizing those subvectors.

Instead of storing every original floating-point value, it stores compact codes.

### How It Works

1. Divide the original vector into smaller subvectors.
2. Train a codebook for each subvector group.
3. Find the closest codebook entry for each group.
4. Store the corresponding compact code.
5. Use compressed representations during approximate distance calculations.

### Example

A 128-dimensional vector is divided into smaller groups.

Each group is represented using a compact code.

This reduces memory requirements.

### Simplified Architecture

```text
Original Vector
      |
      v
128 Dimensions
      |
      v
Divide into Subvectors
      |
      +--------+--------+
      |        |        |
      v        v        v
    Group 1  Group 2  Group 3
      |        |        |
      v        v        v
   Codebook Codebook Codebook
      |        |        |
      +--------+--------+
               |
               v
       Compressed Vector
```

### Purpose

Reduce vector storage and accelerate approximate distance computations.

### Best Real-World Scenarios

- Very large vector databases
- Memory-constrained systems
- Billion-scale similarity search
- Large image embedding collections

### Advantages

- Significant memory savings.
- Useful with IVF indexes.
- Efficient approximate distance calculations.

### Limitations

- Quantization introduces approximation error.
- Requires training codebooks.
- May reduce recall or ranking quality.

### When to Use PQ?

Use PQ when:

- Memory usage is a major concern.
- You have millions or billions of vectors.
- Some approximation is acceptable.
- You need compressed vector storage.

---

## 9.5 IVF + PQ Search

### What is it?

IVF + PQ combines two indexing techniques:

- IVF divides vectors into clusters.
- PQ compresses vectors.

This combination reduces both the number of vectors searched and the memory required to store them.

### Example

Suppose you have 100 million document embeddings.

IVF:

- Organizes vectors into clusters.
- Searches only selected clusters.

PQ:

- Compresses vectors within the index.
- Reduces storage requirements.

Together, they enable efficient large-scale approximate vector search.

### Purpose

Optimize vector search for large datasets with memory and latency constraints.

### Best Real-World Scenarios

- Large-scale enterprise search
- Massive document repositories
- Image retrieval systems
- Large vector databases

### Advantages

- Lower memory usage.
- Reduced search computations.
- Supports very large collections.

### Limitations

- More complex configuration.
- Compression may reduce accuracy.
- Requires careful tuning.

---


## 9.6 Locality-Sensitive Hashing (LSH)

### What is it?

LSH stands for Locality-Sensitive Hashing.

It hashes similar vectors into the same or nearby buckets with higher probability.

Instead of searching the entire dataset, the system searches relevant hash buckets.

### How It Works

1. Generate hash functions.
2. Convert document vectors into hash values.
3. Store vectors in hash buckets.
4. Convert the query vector into hash values.
5. Identify matching buckets.
6. Search candidates within those buckets.
7. Return the nearest results.

### Example

A query vector is converted into hash values.

The search system examines vectors in matching buckets instead of scanning the entire database.

### Simplified Architecture

```text
          Query Vector
               |
               v
          Hash Function
               |
               v
          Hash Bucket
               |
       +-------+-------+
       |       |       |
       v       v       v
    Vector A Vector B Vector C
       |       |       |
       +-------+-------+
               |
               v
         Similar Results
```

### Purpose

Find approximate nearest neighbors efficiently.

### Best Real-World Scenarios

- Large-scale similarity search
- Duplicate detection
- Image similarity
- High-dimensional data
- Approximate matching applications

### Advantages

- Reduces search space.
- Supports approximate similarity matching.
- Can be useful for large high-dimensional datasets.

### Limitations

- Requires hash configuration.
- Can have lower recall depending on parameters.
- May require multiple hash tables.
- Other ANN indexes may perform better for particular workloads.

---

## 9.7 Maximum Inner Product Search (MIPS)

### What is it?

MIPS stands for Maximum Inner Product Search.

It searches for vectors with the highest inner product with a query vector.

### Formula

$$
Score(A,B) = A \cdot B
$$

### Example

Query vector:

```python
A = [1, 2]
```

Document vectors:

```python
B = [2, 4]
C = [4, 2]
```

Scores:

```text
A · B = 10

A · C = 8
```

Document B receives the higher score.

### Purpose

Find vectors with maximum dot-product scores.

### Best Real-World Scenarios

- Recommendation systems
- Embedding retrieval
- Learned ranking models
- Large-scale vector search
- Personalized content recommendations

### Advantages

- Directly optimizes inner-product ranking.
- Useful when vector magnitudes carry information.
- Can be accelerated with specialized indexes.

### Limitations

- Magnitude affects rankings.
- Requires compatible indexing methods.
- Scores are not automatically normalized.

---

## 9.8 Comparison of ANN Techniques

| Technique | Main Idea | Main Advantage | Best Fit |
|---|---|---|---|
| HNSW | Graph navigation | Fast search and good recall | Real-time RAG |
| IVF | Vector clustering | Searches selected clusters | Large datasets |
| PQ | Vector compression | Lower memory usage | Memory-constrained search |
| IVF + PQ | Clustering + compression | Memory and search efficiency | Very large datasets |
| LSH | Similarity hashing | Candidate reduction | Approximate similarity search |
| MIPS | Maximum inner product | Highest dot-product ranking | Recommendation systems |

---

# 10. Graph-Based Search

## 10.1 Graph-Based Search

### What is it?

Graph-Based Search finds information by navigating relationships between entities.

A graph contains:

- Nodes: Entities.
- Edges: Relationships.
- Properties: Entity attributes.

### Example

Consider the following relationships:

```text
OpenAI
   |
   | develops
   v
GPT
   |
   | is a
   v
Large Language Model
   |
   | uses
   v
Transformer
   |
   | uses
   v
Attention Mechanism
```

User question:

"Which attention mechanism is used by the model developed by OpenAI?"

The search system follows connected graph relationships.

### Purpose

Retrieve connected information and answer relationship-based questions.

### Best Real-World Scenarios

- Enterprise knowledge graphs
- Financial relationship analysis
- Supply chain systems
- Fraud detection
- Multi-hop information search
- Organizational knowledge discovery

### Popular Technologies

- Neo4j
- Amazon Neptune
- NetworkX
- Microsoft GraphRAG

### Advantages

- Supports relationship-based search.
- Useful for multi-hop questions.
- Makes connections between entities explicit.
- Can combine structured relationships with text retrieval.

### Limitations

- Requires graph construction.
- Entity extraction and relationship extraction can be complex.
- Graph maintenance can be expensive.
- Incomplete relationships can affect search results.

---

## 10.2 Knowledge Graph Search

### What is it?

Knowledge Graph Search searches structured relationships between entities.

It represents knowledge as connected entities and relationships.

### Example

```text
Company A
    |
    | owns
    v
Company B
    |
    | manufactures
    v
Product C
```

Question:

"Which product is manufactured by a company owned by Company A?"

The system follows graph relationships to find Product C.

### Purpose

Answer questions that depend on structured entity relationships.

### Best Real-World Scenarios

- Corporate research
- Supply chain analysis
- Healthcare knowledge bases
- Fraud detection
- Enterprise relationship discovery

### Advantages

- Represents explicit relationships.
- Supports multi-hop queries.
- Useful for connected information.

### Limitations

- Requires structured knowledge.
- Entity resolution can be difficult.
- Graph updates require maintenance.

---

## 10.3 Graph Traversal Search

### What is it?

Graph Traversal Search explores nodes and edges according to a traversal strategy.

Common traversal algorithms include:

- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- Dijkstra's Algorithm
- A* Search

### Example

Suppose an enterprise graph contains:

```text
Employee A
    |
    v
Department B
    |
    v
Manager C
    |
    v
Project D
```

A query asks:

"Which project is associated with the department of Employee A?"

The system traverses the graph to find Project D.

### Purpose

Explore relationships and find connected information.

### Best Real-World Scenarios

- Organizational relationship search
- Social networks
- Fraud investigation
- Supply chain analysis
- Dependency analysis

### Advantages

- Supports multi-step relationship discovery.
- Can identify connected entities.
- Useful for structured graph data.

### Limitations

- Large graphs can be expensive to traverse.
- Incorrect relationships can produce misleading results.
- Traversal strategy affects performance.

---

## 10.4 GraphRAG

### What is it?

GraphRAG combines knowledge graphs with Retrieval-Augmented Generation.

It uses extracted entities and relationships to organize information and support more context-aware search.

A typical GraphRAG architecture may include:

1. Document ingestion.
2. Entity extraction.
3. Relationship extraction.
4. Graph construction.
5. Community detection or graph organization.
6. Graph-based query processing.
7. Context retrieval.
8. LLM answer generation.

### Example

An enterprise has thousands of documents about:

- Employees
- Departments
- Projects
- Customers
- Products
- Suppliers

User question:

"Which suppliers are connected to projects managed by the engineering department?"

A GraphRAG system can use relationships between departments, projects, and suppliers to find connected information.

### Purpose

Improve retrieval for questions requiring relationships across multiple documents or entities.

### Best Real-World Scenarios

- Enterprise knowledge management
- Financial research
- Supply chain intelligence
- Complex organizational questions
- Research assistants
- Multi-hop question answering

### Advantages

- Captures relationships between entities.
- Supports complex multi-hop questions.
- Can summarize information across connected entities.
- Combines structured and unstructured information.

### Limitations

- Graph construction can be expensive.
- Entity and relationship extraction may contain errors.
- Requires graph maintenance.
- Can increase system complexity and latency.

### When to Use GraphRAG?

Use GraphRAG when:

- Questions require connecting multiple entities.
- Relationships are important to the answer.
- Information is distributed across many documents.
- Traditional vector search does not provide enough relationship context.

---


# 11. Specialized Search Techniques

## 11.1 Fuzzy Search

### What is it?

Fuzzy Search finds documents even when the user's query contains spelling mistakes, typos, or slightly different words.

It uses approximate string matching algorithms such as Levenshtein Distance.

### Example

User query:

`Retrival Augmented Generation`

Actual document:

`Retrieval Augmented Generation`

Fuzzy Search can identify the intended document despite the spelling mistake.

### Purpose

Handle spelling errors and approximate text matching.

### Best Real-World Scenarios

- Search engines
- Customer name searches
- Product catalog search
- Employee directory search
- Technical documentation
- Error message search

### Popular Algorithms

- Levenshtein Distance
- Damerau-Levenshtein Distance
- Jaro-Winkler Similarity
- N-gram Matching

### Advantages

- Handles spelling mistakes.
- Improves user search experience.
- Useful for approximate text matching.

### Limitations

- May return irrelevant results.
- Can be computationally expensive for large datasets.
- Similar-looking words may have different meanings.

---

## 11.2 Faceted Search

### What is it?

Faceted Search allows users to narrow search results using predefined categories or attributes.

Users can filter results by multiple dimensions.

### Example

User query:

`Machine Learning courses`

Available filters:

- Category: AI, Data Science, Programming
- Level: Beginner, Intermediate, Advanced
- Duration: 1 month, 3 months, 6 months
- Language: English, Hindi

The user selects:

- Category: AI
- Level: Beginner
- Language: English

The search system returns matching documents.

### Purpose

Allow users to refine large search result collections.

### Best Real-World Scenarios

- E-commerce websites
- Product catalogs
- Enterprise document portals
- Research paper databases
- Learning management systems

### Advantages

- Improves search navigation.
- Helps users narrow results.
- Supports multiple filters.

### Limitations

- Requires structured metadata.
- Too many filters can confuse users.
- Poor metadata quality affects results.

---

## 11.3 Metadata Filtering

### What is it?

Metadata Filtering restricts search results using document attributes before or during similarity search.

Metadata may include:

- Document type
- Department
- Author
- Creation date
- Category
- Access permissions
- Document version

### Example

User query:

`Employee leave policy`

Metadata filter:

```python
{
    "department": "HR",
    "document_type": "policy",
    "year": 2026
}
```

The search engine retrieves relevant documents that match the specified metadata conditions.

### Purpose

Restrict search to relevant document subsets.

### Best Real-World Scenarios

- Enterprise RAG
- HR knowledge assistants
- Financial document search
- Multi-tenant applications
- Permission-aware document search

### Advantages

- Reduces irrelevant results.
- Improves search precision.
- Supports access control.
- Helps narrow large datasets.

### Limitations

- Requires accurate metadata.
- Incorrect filters may exclude relevant documents.
- Filter execution can affect index performance.

### Important Note

Metadata filtering is not the same as semantic search.

Semantic search identifies meaning-based similarity, while metadata filtering restricts which documents are eligible for retrieval.

---

## 11.4 Multimodal Search

### What is it?

Multimodal Search retrieves information using different types of data.

It can search across:

- Text
- Images
- Audio
- Video
- Documents containing mixed media

### Example

User query:

`Find the diagram showing the RAG architecture.`

Knowledge base contains:

- Text documents
- Architecture diagrams
- Screenshots
- PDF files

A multimodal search system can retrieve the relevant diagram.

### Purpose

Search information across multiple data modalities.

### Best Real-World Scenarios

- Medical image search
- Product image search
- Video knowledge bases
- Technical documentation
- Multimedia archives
- Document understanding systems

### Advantages

- Supports different content formats.
- Enables image and text-based search.
- Useful for rich document collections.

### Limitations

- Requires modality-specific processing.
- More complex indexing.
- Multimodal embeddings may require additional computational resources.

---

## 11.5 Image-Based Search

### What is it?

Image-Based Search retrieves images based on visual similarity or image content.

It can use image embeddings and computer vision models.

### Example

User uploads a picture of a laptop.

The system searches a product database and returns visually similar laptops.

### Purpose

Find visually similar images.

### Best Real-World Scenarios

- E-commerce product discovery
- Medical image retrieval
- Architecture diagram search
- Manufacturing defect detection
- Image asset management

### Popular Technologies

- CLIP
- OpenCLIP
- FAISS
- Qdrant
- Weaviate

### Advantages

- Searches based on visual features.
- Can retrieve images without exact text descriptions.
- Supports visual similarity applications.

### Limitations

- Similar appearance does not guarantee identical meaning.
- Image quality affects results.
- Requires suitable visual embeddings.

---

## 11.6 Cross-Modal Search

### What is it?

Cross-Modal Search allows a query in one modality to retrieve information in another modality.

For example:

- Text query retrieves images.
- Image query retrieves text descriptions.
- Audio query retrieves related video segments.

### Example

User query:

`A red sports car on a mountain road`

The system retrieves matching images from an image collection.

The user does not need to upload an image.

### Purpose

Connect different modalities through a shared representation space.

### Best Real-World Scenarios

- Text-to-image search
- Image-to-text retrieval
- Video search
- Multimedia knowledge bases
- Product discovery

### Advantages

- Enables cross-format search.
- Supports natural-language image discovery.
- Useful for multimodal RAG.

### Limitations

- Depends on cross-modal embedding quality.
- May struggle with fine-grained visual details.
- Requires compatible multimodal models.

---

## 11.7 Temporal Search

### What is it?

Temporal Search retrieves information based on time-related conditions.

It considers dates, timestamps, periods, or chronological relationships.

### Example

User query:

`Show all financial reports published between January and March 2026.`

The search system filters documents using publication dates.

### Purpose

Retrieve time-sensitive or date-specific information.

### Best Real-World Scenarios

- Financial reports
- News archives
- Legal records
- Historical document search
- Event management
- Version history search

### Advantages

- Supports time-based filtering.
- Helps retrieve historical information.
- Useful for chronological analysis.

### Limitations

- Requires reliable timestamps.
- Time zones may introduce complexity.
- Publication dates and event dates may differ.

---

## 11.8 Geospatial Search

### What is it?

Geospatial Search retrieves information based on geographical location.

It can use:

- Coordinates
- Addresses
- Geographic regions
- Distance from a location
- Spatial relationships

### Example

User query:

`Find hospitals within 5 km of my location.`

The system searches using geographic coordinates and distance constraints.

### Purpose

Find information based on geographical relationships.

### Best Real-World Scenarios

- Nearby service discovery
- Logistics
- Location-based recommendations
- Real estate search
- Delivery applications

### Advantages

- Supports location-aware search.
- Enables radius and region filtering.
- Useful for spatial applications.

### Limitations

- Requires accurate geographic data.
- Distance calculations must use appropriate geographic models.
- Location privacy must be considered.

---

# 12. Query Transformation and Query Understanding Techniques

These techniques modify or analyze the user's query before performing the actual search.

Their goal is to improve the quality of the search input.

## 12.1 Query Expansion

### What is it?

Query Expansion adds related terms, synonyms, or alternative expressions to the original query.

This helps retrieve documents that use different terminology.

### Example

Original query:

`Car maintenance`

Expanded query:

```text
Car maintenance
Vehicle servicing
Automobile repair
Engine maintenance
Vehicle inspection
```

The search system uses the expanded terms to retrieve more relevant documents.

### Purpose

Improve search recall by including related terminology.

### Best Real-World Scenarios

- Medical literature search
- Enterprise document search
- Technical documentation
- Legal research
- Customer support search

### Advantages

- Finds documents using alternative terminology.
- Improves search coverage.
- Useful when users do not know the exact terminology.

### Limitations

- May introduce irrelevant terms.
- Can increase search result noise.
- Requires good synonym or expansion strategies.

---

## 12.2 Query Rewriting

### What is it?

Query Rewriting converts the user's original query into a clearer or more search-friendly query.

It can correct grammar, resolve ambiguity, and improve terminology.

### Example

Original query:

`How make python faster?`

Rewritten query:

`How can Python application performance be optimized?`

The rewritten query is sent to the search engine.

### Purpose

Improve the clarity and searchability of user queries.

### Best Real-World Scenarios

- Conversational RAG
- Customer support assistants
- Technical knowledge bases
- Enterprise chatbots
- Search applications with natural-language queries

### Advantages

- Improves query clarity.
- Handles informal language.
- Can resolve conversational references.

### Limitations

- Rewriting can change the user's intent.
- Requires validation.
- Incorrect rewriting may reduce retrieval quality.

---

## 12.3 Query Normalization

### What is it?

Query Normalization converts different forms of a query into a consistent representation.

It may include:

- Lowercasing
- Removing unnecessary punctuation
- Correcting whitespace
- Standardizing abbreviations
- Normalizing special characters

### Example

Original query:

`  R.A.G   Search!!! `

Normalized query:

`rag search`

### Purpose

Improve consistency between queries and indexed documents.

### Best Real-World Scenarios

- Enterprise search
- Product search
- Database search
- Technical documentation
- Large-scale search systems

### Advantages

- Reduces formatting inconsistencies.
- Improves lexical matching.
- Simple to implement.

### Limitations

- Over-normalization can remove meaningful information.
- Case and punctuation may matter for some identifiers.
- Domain-specific normalization may be necessary.

---

## 12.4 Query Decomposition

### What is it?

Query Decomposition breaks a complex user question into smaller, independent sub-queries.

Each sub-query is searched separately.

The retrieved results are then combined to answer the original question.

### Example

Original question:

`Compare Python and Java in terms of performance, scalability, and enterprise applications.`

Decomposed queries:

1. Python performance characteristics.
2. Java performance characteristics.
3. Python scalability in enterprise applications.
4. Java scalability in enterprise applications.
5. Comparison of Python and Java enterprise use cases.

The system searches each sub-query and combines the results.

### Purpose

Handle complex questions that require information from multiple topics.

### Best Real-World Scenarios

- Research assistants
- Enterprise knowledge systems
- Financial analysis
- Technical comparison systems
- Multi-document question answering

### Advantages

- Breaks complex questions into manageable parts.
- Improves topic coverage.
- Supports multi-document reasoning.
- Can improve retrieval for multi-part questions.

### Limitations

- Increases search operations.
- Sub-queries may overlap.
- Incorrect decomposition may omit important parts.
- Requires result aggregation.

---


# 13. Advanced Query Transformation and Intelligent Search Techniques

## 13.1 Hypothetical Document Embeddings (HyDE)

### What is it?

HyDE stands for Hypothetical Document Embeddings.

It is a query transformation technique that uses an LLM to generate a hypothetical answer or document based on the user's query.

The hypothetical document is converted into an embedding and used to search the actual knowledge base.

The retrieved documents come from the real knowledge base, not from the hypothetical document.

### How It Works

1. User submits a query.
2. LLM generates a hypothetical document that could answer the query.
3. Convert the hypothetical document into an embedding.
4. Search the vector database using that embedding.
5. Retrieve relevant real documents.
6. Provide retrieved documents to the LLM.
7. Generate the final answer.

### Example

User query:

`How can organizations prevent unauthorized access to enterprise systems?`

LLM generates hypothetical content:

`Organizations can prevent unauthorized access through multi-factor authentication, role-based access control, identity management, encryption, and continuous monitoring.`

The hypothetical document is embedded and searched against the actual knowledge base.

### Purpose

Improve semantic retrieval when the original query is short, vague, or expressed differently from the documents.

### Best Real-World Scenarios

- Enterprise knowledge search
- Technical documentation
- Research assistants
- Security policy search
- Domain-specific RAG applications

### Advantages

- Expands a short query into richer semantic context.
- Can improve retrieval for complex natural-language questions.
- Helps bridge the vocabulary gap between queries and documents.

### Limitations

- Hypothetical content may contain incorrect assumptions.
- Additional LLM calls increase latency and cost.
- Generated content may introduce semantic drift.

### When to Use HyDE?

Use HyDE when:

- Queries are short or underspecified.
- Users ask questions rather than provide keywords.
- The knowledge base uses different terminology.
- Ordinary vector search produces weak results.

---

## 13.2 Step-Back Prompting

### What is it?

Step-Back Prompting asks the LLM to move from a specific question to a more general question before searching.

The goal is to identify the underlying concepts required to answer the original question.

The system retrieves information using the broader question and may also search using the original question.

### How It Works

1. User submits a specific question.
2. LLM identifies the broader concept behind it.
3. LLM generates a step-back question.
4. Search the knowledge base using the broader question.
5. Optionally search using the original question.
6. Combine retrieved information.
7. Generate the final answer.

### Example

Original question:

`Why does my Python API become slow when 10,000 users access it simultaneously?`

Step-back question:

`What factors affect the scalability and performance of web APIs under high concurrent traffic?`

Searches can retrieve information about:

- Concurrency
- Database bottlenecks
- Thread management
- Connection pooling
- Load balancing
- Caching

### Purpose

Retrieve broader foundational knowledge that may not be obvious from the original query.

### Best Real-World Scenarios

- Technical troubleshooting
- Complex engineering questions
- Research assistants
- Enterprise knowledge systems
- Educational question answering

### Advantages

- Finds foundational information.
- Helps with complex questions.
- Provides broader context.
- Can improve multi-step reasoning.

### Limitations

- May retrieve overly general information.
- Requires additional LLM processing.
- The broader question must remain relevant to the original intent.

### When to Use Step-Back Prompting?

Use it when:

- The question depends on foundational concepts.
- The user asks why something happens.
- The original query is too specific.
- Relevant background knowledge is needed.

---

## 13.3 Multi-Query Search

### What is it?

Multi-Query Search generates multiple alternative queries from one original user query.

Each query is searched independently.

The resulting documents are combined and deduplicated.

### How It Works

1. Receive the original query.
2. Generate multiple alternative queries.
3. Execute searches for all generated queries.
4. Combine results.
5. Remove duplicate documents.
6. Rank the combined results.
7. Return the most relevant documents.

### Example

Original query:

`How to improve database performance?`

Generated queries:

1. Database query optimization techniques.
2. Database indexing strategies.
3. SQL performance tuning.
4. Database caching techniques.
5. Database scalability best practices.

The system searches all five queries.

### Purpose

Improve search recall by exploring different interpretations and aspects of the original question.

### Best Real-World Scenarios

- Research assistants
- Technical documentation
- Enterprise knowledge search
- Complex troubleshooting
- Broad information discovery

### Advantages

- Finds documents using different terminology.
- Covers multiple aspects of a question.
- Improves search recall.
- Reduces dependence on one query formulation.

### Limitations

- Requires multiple searches.
- Can increase latency and cost.
- May produce duplicate or irrelevant results.

---

## 13.4 Self-Query Search

### What is it?

Self-Query Search uses an LLM to convert a natural-language query into a structured search query containing:

- Semantic search text
- Metadata filters
- Comparison operators
- Other structured conditions

### How It Works

1. User submits a natural-language question.
2. LLM identifies search intent.
3. LLM extracts metadata conditions.
4. Generate a structured query.
5. Execute vector search with filters.
6. Return matching documents.

### Example

User query:

`Find HR policy documents published after 2024.`

LLM generates:

```json
{
  "search_query": "HR policy",
  "filter": {
    "department": "HR",
    "year": {
      "$gt": 2024
    }
  }
}
```

The system performs semantic search with the generated metadata filter.

### Purpose

Convert natural-language queries into structured search operations.

### Best Real-World Scenarios

- Enterprise document search
- Product catalogs
- Financial document search
- Multi-tenant knowledge bases
- HR and policy assistants

### Advantages

- Understands natural-language filters.
- Combines semantic search and metadata filtering.
- Reduces manual filter selection.

### Limitations

- Depends on metadata quality.
- Incorrect filter generation can exclude relevant documents.
- Requires schema awareness and query validation.

---

## 13.5 Iterative Search

### What is it?

Iterative Search performs multiple search operations in sequence.

The system uses the results from one search to improve or formulate the next search.

### How It Works

1. Receive the original query.
2. Perform an initial search.
3. Evaluate the retrieved results.
4. Identify missing information.
5. Generate a refined query.
6. Perform another search.
7. Combine the retrieved results.
8. Generate the final answer.

### Example

Original query:

`How does Kubernetes handle application scaling?`

First search:

`Kubernetes horizontal pod autoscaling`

Retrieved information identifies additional concepts.

Second search:

`Kubernetes cluster autoscaling and resource management`

Third search:

`Kubernetes scaling limitations and best practices`

The system combines the information from all searches.

### Purpose

Improve information coverage through multiple search iterations.

### Best Real-World Scenarios

- Complex technical research
- Multi-document analysis
- Troubleshooting assistants
- Enterprise knowledge discovery
- Research automation

### Advantages

- Improves coverage.
- Can identify missing information.
- Supports progressive query refinement.
- Useful for complex questions.

### Limitations

- More search operations.
- Increased latency.
- Can repeat searches unnecessarily.
- Requires stopping conditions.

---

## 13.6 Agentic Search

### What is it?

Agentic Search uses an LLM-powered agent to decide how, when, and where to search.

Instead of following one fixed search strategy, the agent can select different search tools depending on the question.

Possible tools include:

- Vector Search
- BM25 Search
- Web Search
- Database Search
- Graph Search
- SQL Queries
- API Search

### How It Works

1. User submits a query.
2. Agent analyzes the request.
3. Agent selects an appropriate search tool.
4. Search is executed.
5. Agent evaluates the results.
6. Agent decides whether additional searches are necessary.
7. Agent combines the evidence.
8. Agent generates the final response.

### Example

User question:

`Compare the latest company policy with the previous version and explain the major changes.`

The agent may:

1. Search the latest policy.
2. Search the previous policy.
3. Compare both documents.
4. Identify changed sections.
5. Retrieve additional references if necessary.
6. Generate a comparison.

### Purpose

Enable intelligent, dynamic search planning and tool selection.

### Best Real-World Scenarios

- Enterprise research assistants
- Financial analysis
- Complex customer support
- Multi-source knowledge systems
- Technical troubleshooting agents

### Advantages

- Dynamically selects search strategies.
- Supports multiple data sources.
- Can perform iterative research.
- Handles complex multi-step questions.

### Limitations

- More complex architecture.
- Increased latency and operational cost.
- Agent decisions may be incorrect.
- Requires tool permissions, validation, and execution limits.

---

## 13.7 Contextual Search

### What is it?

Contextual Search uses previous conversation messages or surrounding information to understand the current query.

It is especially useful in conversational RAG applications.

### Example

Conversation:

User:

`What is vector search?`

Assistant explains vector search.

User:

`What are its limitations?`

A contextual search system understands that "its" refers to vector search.

It transforms the query into:

`What are the limitations of vector search?`

### Purpose

Use conversational context to resolve references and improve search relevance.

### Best Real-World Scenarios

- AI chatbots
- Customer support assistants
- Conversational knowledge bases
- Technical assistants
- Multi-turn RAG systems

### Advantages

- Understands follow-up questions.
- Resolves pronouns and references.
- Improves conversational search experience.

### Limitations

- Incorrect conversation history can mislead search.
- Long conversations require context management.
- Irrelevant previous messages may affect retrieval.

---

## 13.8 Conversational Search

### What is it?

Conversational Search allows users to search for information through a sequence of natural-language interactions.

The search engine maintains context between questions.

### Example

User:

`Find information about cloud computing.`

User:

`Explain its security challenges.`

User:

`How can organizations prevent those challenges?`

The system understands the relationship between all three questions.

### Purpose

Support natural, multi-turn information discovery.

### Best Real-World Scenarios

- Customer service chatbots
- Enterprise assistants
- Educational assistants
- Research applications
- Technical support systems

### Advantages

- Natural user experience.
- Supports follow-up questions.
- Maintains conversational context.
- Reduces repeated query information.

### Limitations

- Requires conversation state management.
- Context may become too large.
- Incorrect reference resolution can affect retrieval.

---

## 13.9 Query Routing

### What is it?

Query Routing determines which search engine, index, database, or retrieval strategy should handle a particular query.

Different queries may require different search techniques.

### How It Works

1. Receive the user query.
2. Classify the query type.
3. Identify the appropriate search destination.
4. Execute the selected search.
5. Return the results.

### Example

A RAG application has multiple knowledge sources:

- HR documents
- Technical documentation
- Financial reports
- Product catalogs

User queries:

`What is the employee leave policy?`

Route to HR documents.

`How does our authentication API work?`

Route to technical documentation.

`What was the revenue in 2025?`

Route to financial reports.

### Purpose

Send queries to the most appropriate search source.

### Best Real-World Scenarios

- Enterprise RAG
- Multi-department knowledge systems
- Multi-index vector databases
- Multi-domain assistants
- Multi-source research applications

### Advantages

- Avoids searching irrelevant collections.
- Improves search efficiency.
- Supports multiple knowledge domains.
- Enables specialized search strategies.

### Limitations

- Incorrect routing can miss relevant documents.
- Requires routing rules or classification models.
- Multiple sources may need combined search.

---

## 13.10 Ensemble Search

### What is it?

Ensemble Search combines results from multiple search methods or ranking models.

It can combine:

- BM25
- Dense Vector Search
- Sparse Vector Search
- Graph Search
- Different embedding models
- Different ranking algorithms

### Example

A search system uses:

1. BM25 to find exact terminology.
2. Dense Search to find semantic matches.
3. SPLADE to find learned sparse matches.
4. RRF to combine their rankings.

The combined results provide broader search coverage.

### Purpose

Combine the strengths of different search approaches.

### Best Real-World Scenarios

- Enterprise search
- Large RAG applications
- Technical documentation
- Research assistants
- High-quality search systems

### Advantages

- Combines multiple relevance signals.
- Can improve search recall and ranking quality.
- Reduces dependence on one retrieval model.

### Limitations

- More complex system architecture.
- Higher computational cost.
- Requires result deduplication and ranking.
- Needs evaluation and tuning.

---

## 13.11 Diversified Search

### What is it?

Diversified Search attempts to return results covering different aspects of a query instead of returning many nearly identical documents.

One common technique is Maximal Marginal Relevance (MMR).

### What is Maximal Marginal Relevance?

MMR balances:

- Relevance to the original query.
- Diversity among selected documents.

### Formula

$$
MMR = \arg\max_{d \in D \setminus S}
\left[
\lambda Sim(d,Q)
-
(1-\lambda)\max_{s \in S}Sim(d,s)
\right]
$$

Where:

- D = Candidate documents.
- S = Already selected documents.
- Q = User query.
- Sim(d,Q) = Relevance to the query.
- Sim(d,s) = Similarity to already selected documents.
- λ = Relevance-diversity trade-off.

### Example

User query:

`Explain cloud computing security.`

Search results:

- Document A: Cloud encryption.
- Document B: Cloud encryption techniques.
- Document C: Identity and access management.
- Document D: Cloud network security.

A standard similarity search may return A and B because they are very similar.

Diversified Search may select A, C, and D to provide broader coverage.

### Purpose

Reduce redundant results and increase topic coverage.

### Best Real-World Scenarios

- Research assistants
- Enterprise RAG
- Literature reviews
- Product discovery
- Multi-document question answering

### Advantages

- Reduces duplicate information.
- Improves result diversity.
- Covers multiple aspects of a topic.

### Limitations

- Excessive diversity can reduce relevance.
- Requires balancing relevance and diversity.
- Depends on the quality of candidate results.

---

# 14. Additional Advanced Search Techniques

## 14.1 Sparse Vector Search

### What is it?

Sparse Vector Search represents text using sparse numerical vectors, where most dimensions contain zero values.

Unlike dense embeddings, sparse vectors often preserve interpretable keyword-related signals.

A popular approach is SPLADE.

### Example

User query:

`How to troubleshoot database connection errors?`

Sparse Search may identify terms such as:

- database
- connection
- error
- troubleshooting

It can also learn related terms and expansions.

### Purpose

Improve lexical and semantic matching using sparse representations.

### Best Real-World Scenarios

- Technical documentation
- Enterprise search
- Error message search
- Domain-specific search
- Hybrid RAG systems

### Advantages

- Good keyword matching.
- Can capture learned term relationships.
- Can be combined with dense vector search.

### Limitations

- Requires sparse representations or learned models.
- May require additional indexing.
- Can increase system complexity.

---

## 14.2 Learned Sparse Search (SPLADE)

### What is it?

SPLADE stands for Sparse Lexical and Expansion Model.

It uses a neural language model to produce sparse lexical representations of queries and documents.

Unlike traditional keyword search, SPLADE can expand terms based on learned semantic relationships.

### Example

User query:

`How to fix slow SQL queries?`

SPLADE may associate the query with terms such as:

- SQL
- query optimization
- database indexing
- execution plans
- database performance

This helps retrieve documents that use different terminology.

### Purpose

Combine lexical interpretability with learned semantic term expansion.

### Best Real-World Scenarios

- Technical search engines
- Enterprise document search
- Research paper search
- Hybrid RAG applications

### Advantages

- Supports learned term expansion.
- Retains sparse representations.
- Can complement dense vector search.

### Limitations

- Requires model inference.
- More complex than traditional BM25.
- Requires model-specific indexing and evaluation.

---

## 14.3 Late Interaction Search (ColBERT)

### What is it?

ColBERT stands for Contextualized Late Interaction over BERT.

It represents queries and documents using multiple token-level embeddings rather than a single embedding for the entire document.

It compares query token embeddings with document token embeddings to calculate relevance.

### How It Works

1. Convert the query into token embeddings.
2. Convert documents into token embeddings.
3. Compare query tokens with document tokens.
4. Calculate token-level relevance.
5. Aggregate the matching scores.
6. Rank documents.

### Example

User query:

`Python memory optimization techniques`

Document:

`Python applications can reduce memory consumption through generators, efficient data structures, and memory profiling.`

ColBERT can identify token-level relationships between query concepts and document content.

### Purpose

Provide more fine-grained relevance matching than single-vector document representations.

### Best Real-World Scenarios

- Technical documentation
- Research paper search
- Enterprise knowledge search
- Complex question answering
- High-quality semantic search

### Advantages

- Preserves token-level information.
- Can improve fine-grained matching.
- Captures detailed query-document interactions.

### Limitations

- Requires more storage than single-vector representations.
- More expensive than basic dense retrieval.
- Requires specialized indexing and ranking infrastructure.

---

## 14.4 Parent-Child Search

### What is it?

Parent-Child Search indexes smaller document chunks but retrieves their larger parent sections when necessary.

The child chunk is used to identify relevant information.

The parent document or section provides additional context.

### Example

A 100-page technical document is divided into:

Parent:
`Chapter 5 - Database Optimization`

Children:
- Indexing strategies
- Query execution plans
- Connection pooling
- Database caching

User query:

`How does connection pooling improve performance?`

The system:

1. Finds the relevant child chunk.
2. Identifies its parent chapter.
3. Retrieves the parent context.
4. Passes relevant context to the LLM.

### Purpose

Balance precise retrieval with sufficient contextual information.

### Best Real-World Scenarios

- Large PDF documents
- Technical manuals
- Legal documents
- Research papers
- Enterprise knowledge bases

### Advantages

- More precise initial matching.
- Provides broader contextual information.
- Reduces fragmented answers.

### Limitations

- Requires parent-child relationships.
- May retrieve too much context.
- Requires careful chunk organization.

---

## 14.5 Hierarchical Search

### What is it?

Hierarchical Search organizes information into multiple levels.

Search begins at a broader level and progressively moves toward more specific information.

### Example

An enterprise knowledge base contains:

```text
Enterprise Knowledge Base
          |
          +-- HR
          |    |
          |    +-- Leave Policy
          |    +-- Payroll
          |
          +-- Technology
          |    |
          |    +-- Cloud
          |    +-- Security
          |
          +-- Finance
               |
               +-- Revenue
               +-- Expenses
```

User query:

`How does cloud authentication work?`

The system identifies:

1. Technology.
2. Cloud.
3. Authentication documentation.
4. Relevant document sections.

### Purpose

Organize and search large knowledge collections efficiently.

### Best Real-World Scenarios

- Enterprise knowledge systems
- Large document repositories
- Technical documentation portals
- Organizational knowledge bases
- Research archives

### Advantages

- Supports structured search navigation.
- Can reduce unnecessary search.
- Provides context at multiple levels.

### Limitations

- Requires hierarchical organization.
- Poor categorization affects search.
- Information may belong to multiple categories.

---

## 14.6 Community-Based Search

### What is it?

Community-Based Search organizes graph entities into communities or clusters.

Instead of searching only individual entities, the system can retrieve information about broader groups of related entities.

This is an important technique used in some GraphRAG architectures.

### Example

A company knowledge graph contains relationships between:

- Employees
- Projects
- Departments
- Products
- Customers

Community detection may identify a group of entities related to a particular business domain.

User query:

`Summarize the major challenges faced by the company's cloud migration projects.`

The system can retrieve information from the relevant graph community.

### Purpose

Retrieve high-level information from groups of related entities.

### Best Real-World Scenarios

- Enterprise GraphRAG
- Financial research
- Organizational intelligence
- Large knowledge graphs
- Multi-document summarization

### Advantages

- Supports high-level information discovery.
- Helps summarize connected information.
- Useful for large graph collections.

### Limitations

- Depends on graph quality.
- Community boundaries may not reflect user intent.
- Graph processing can be expensive.

---

# 15. Comparison of Advanced RAG Search Techniques

| Technique | Main Purpose | Best Real-World Fit |
|---|---|---|
| Sparse Vector Search | Keyword-oriented vector matching | Technical search |
| SPLADE | Learned lexical expansion | Enterprise search |
| ColBERT | Fine-grained token matching | Research search |
| Parent-Child Search | Retrieve precise chunks with broader context | Large PDFs |
| Hierarchical Search | Search organized information levels | Enterprise knowledge |
| Community-Based Search | Retrieve related entity groups | GraphRAG |

---

# 16. Which RAG Search Technique Should You Choose?

Different search techniques solve different problems.

There is no single best technique for every RAG application.

The right choice depends on:

- Query complexity
- Knowledge base size
- Data format
- Search accuracy requirements
- Latency requirements
- Memory constraints
- Relationships between documents
- Business requirements

## 16.1 Real-World Scenario-Based Selection

| Real-World Scenario | Recommended Technique | Why? |
|---|---|---|
| User asks a simple factual question | Dense Vector Search | Finds semantically relevant content |
| User searches using exact product codes | BM25 | Matches exact keywords and identifiers |
| User uses different terminology | Query Expansion | Adds related terms |
| User asks a vague question | HyDE | Generates richer hypothetical search content |
| User asks a complex technical question | Query Decomposition | Breaks it into smaller searches |
| User asks a question requiring background knowledge | Step-Back Prompting | Retrieves foundational information |
| User asks follow-up questions | Contextual Search | Understands conversation history |
| User asks questions involving multiple entities | Graph-Based Search | Follows entity relationships |
| User asks about relationships across many documents | GraphRAG | Combines graph relationships and document knowledge |
| User searches millions of vectors | HNSW or IVF | Accelerates nearest-neighbor search |
| Vector storage is expensive | IVF + PQ | Reduces memory requirements |
| User makes spelling mistakes | Fuzzy Search | Handles approximate text matching |
| User searches images using text | Cross-Modal Search | Connects text and image representations |
| User searches by department or date | Metadata Filtering | Restricts the search space |
| User wants diverse results | MMR | Reduces redundant documents |
| User needs fine-grained document matching | ColBERT | Uses token-level interactions |
| User searches large PDF documents | Parent-Child Search | Combines precise chunks with broader context |
| User asks questions across multiple databases | Agentic Search | Selects and coordinates search tools |
| User wants multiple interpretations explored | Multi-Query Search | Generates alternative queries |
| User needs exact and semantic matching | Hybrid Search | Combines BM25 and dense search |

---

## 16.2 Recommended Techniques by Application

### Enterprise Knowledge Assistant

Recommended:

1. Hybrid Search
2. Metadata Filtering
3. Query Rewriting
4. Contextual Search
5. RRF
6. Reranking (if used as a ranking stage)

Why?

Enterprise knowledge systems contain policies, technical documentation, HR documents, and reports.

Keyword and semantic search together provide better coverage.

### Technical Documentation Assistant

Recommended:

1. Hybrid Search
2. Query Expansion
3. HyDE
4. Query Decomposition
5. ColBERT
6. Parent-Child Search

Why?

Technical documentation contains exact terminology, abbreviations, error messages, and detailed explanations.

### Research Assistant

Recommended:

1. Multi-Query Search
2. Query Decomposition
3. Step-Back Prompting
4. Ensemble Search
5. MMR
6. Graph-Based Search

Why?

Research questions frequently require information from multiple sources and different perspectives.

### E-Commerce Search

Recommended:

1. Hybrid Search
2. Metadata Filtering
3. Faceted Search
4. Cross-Modal Search
5. Query Expansion

Why?

Users may search using product names, descriptions, specifications, categories, or images.

### Financial Research Assistant

Recommended:

1. Hybrid Search
2. Metadata Filtering
3. Temporal Search
4. Graph-Based Search
5. Query Decomposition
6. Agentic Search

Why?

Financial research often requires comparing companies, reporting periods, relationships, and multiple documents.

### Customer Support Assistant

Recommended:

1. Hybrid Search
2. Query Rewriting
3. Contextual Search
4. Query Expansion
5. Parent-Child Search

Why?

Customers often use informal language and ask follow-up questions.

---

# 17. Recommended RAG Search Pipeline

A practical RAG application may combine multiple search techniques into a single pipeline.

### Example Architecture

```text
                 User Query
                      |
                      v
              Query Understanding
                      |
                      v
                Query Routing
                      |
                      v
             Query Transformation
                      |
             +--------+--------+
             |                 |
             v                 v
        Query Rewrite       HyDE /
        / Expansion         Multi-Query
             |                 |
             +--------+--------+
                      |
                      v
                 Hybrid Search
                /           \
               v             v
          BM25 Search    Dense Search
               |             |
               +------+------+
                      |
                      v
                     RRF
                      |
                      v
               Candidate Results
                      |
                      v
                  Reranking
                      |
                      v
                MMR / Filtering
                      |
                      v
                Context Assembly
                      |
                      v
                  LLM Answer
                      |
                      v
                Final Response
```

### Explanation of Each Stage

| Stage | Technique | Purpose |
|---|---|---|
| Query Understanding | Query Classification | Understand user intent |
| Query Routing | Query Routing | Select the correct data source |
| Query Transformation | Rewrite, HyDE, Expansion | Improve query quality |
| Retrieval | BM25, Dense Search | Find relevant documents |
| Result Combination | RRF | Combine search rankings |
| Ranking | Reranking | Improve document relevance |
| Result Selection | MMR | Reduce redundant results |
| Context Assembly | Parent-Child Search | Build sufficient context |
| Generation | LLM | Generate the final answer |

Note: This is a reference architecture, not a mandatory pipeline. A simple RAG system may need only a subset of these techniques.

---

# 18. Final Recommendations

## Start with a Simple RAG System

For most applications, begin with:

1. Dense Vector Search.
2. BM25 Keyword Search.
3. Hybrid Search.
4. Metadata Filtering.
5. RRF.
6. Reranking.

Measure retrieval quality before introducing additional complexity.

## Add Advanced Techniques When Needed

| Problem | Technique to Consider |
|---|---|
| Poor retrieval for vague queries | HyDE |
| Complex multi-part questions | Query Decomposition |
| Missing terminology | Query Expansion |
| Follow-up questions fail | Contextual Search |
| Relationship-based questions fail | GraphRAG |
| Search results are repetitive | MMR |
| Search latency is high | ANN indexing |
| Memory consumption is high | Product Quantization |
| Exact terminology is missed | BM25 or SPLADE |
| Fine-grained relevance is poor | ColBERT |
| Too many irrelevant documents | Metadata Filtering |
| Multiple data sources need searching | Agentic Search |

## Important Evaluation Metrics

When selecting search techniques, evaluate:

- Recall@K: How many relevant documents appear in the top K results?
- Precision@K: How many retrieved documents are relevant?
- MRR: How high is the first relevant result ranked?
- NDCG@K: How well are relevant documents ranked?
- Hit Rate@K: Whether at least one relevant document appears in the top K.
- Latency: How long does search take?
- Cost: How much computation and infrastructure does search require?

A technique should be selected based on measured improvements, not simply because it is advanced.

---

# Conclusion

RAG Search is not limited to vector similarity search.

Modern RAG systems can combine:

- Keyword Search
- Semantic Search
- Hybrid Search
- Graph-Based Search
- Query Transformation
- ANN Algorithms
- Multimodal Search
- Contextual Search
- Agentic Search
- Advanced Ranking and Diversification

The most effective approach is to select techniques based on the actual search problem.

**Start simple, evaluate retrieval quality, identify limitations, and introduce advanced search techniques only when they provide measurable improvements.**

---


