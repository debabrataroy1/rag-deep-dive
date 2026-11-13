
# Complete List of Text Chunking Strategies in RAG

Chunking is the process of splitting large documents into smaller pieces before generating embeddings and storing them in a vector database.

There are several chunking strategies. They differ in how they split documents, preserve context, and optimize retrieval quality.

Below is a structured learning roadmap, from beginner to advanced and production-level chunking.

---

## 1. Basic Chunking Strategies

These are the starting points for understanding chunking.

| # | Strategy | Description |
|---|---|---|
| 1 | Fixed-size chunking | Splits text into chunks of a fixed character count |
| 2 | Fixed-token chunking | Splits text based on token count |
| 3 | Overlapping chunking | Adds overlapping text between consecutive chunks |
| 4 | Non-overlapping chunking | Splits text without overlap |
| 5 | Sentence-based chunking | Splits text into individual sentences |
| 6 | Paragraph-based chunking | Splits text using paragraph boundaries |
| 7 | Word-based chunking | Splits text based on a fixed number of words |

---

## 2. Structure-Based Chunking Strategies

These strategies preserve the natural structure of documents.

| # | Strategy | Description |
|---|---|---|
| 8 | Recursive character chunking | Recursively splits text using separators such as paragraphs, newlines and sentences |
| 9 | Markdown chunking | Splits documents based on Markdown headings and sections |
| 10 | HTML chunking | Splits content using HTML elements and tags |
| 11 | JSON chunking | Splits structured JSON data into meaningful units |
| 12 | XML chunking | Splits XML documents based on elements |
| 13 | Document layout chunking | Uses document layout, sections and formatting |
| 14 | Page-based chunking | Splits PDF documents page by page |
| 15 | Section-based chunking | Splits content based on logical sections |
| 16 | Heading-based chunking | Uses headings and subheadings as boundaries |

---

## 3. Semantic Chunking Strategies

These strategies focus on the meaning of the text rather than fixed boundaries.

| # | Strategy | Description |
|---|---|---|
| 17 | Semantic chunking | Splits text when semantic meaning changes |
| 18 | Embedding-based chunking | Uses embeddings to identify boundaries between related and unrelated text |
| 19 | Similarity-based chunking | Groups sentences based on semantic similarity |
| 20 | Topic-based chunking | Creates chunks around individual topics |
| 21 | Clustering-based chunking | Groups semantically related sentences using clustering algorithms |
| 22 | Context-aware chunking | Uses surrounding context to determine chunk boundaries |
| 23 | Dynamic chunking | Adjusts chunk size based on content |
| 24 | Adaptive chunking | Changes chunking strategy based on document type or content |

---

## 4. Advanced Chunking Strategies

| # | Strategy | Description |
|---|---|---|
| 25 | Agentic chunking | Uses an LLM or agent to decide how documents should be split |
| 26 | LLM-based chunking | Uses an LLM to identify meaningful chunk boundaries |
| 27 | Proposition-based chunking | Splits content into individual factual statements |
| 28 | Late chunking | Generates token-level embeddings for a larger context before pooling representations into chunk embeddings |
| 29 | Sliding window chunking | Moves a window across text to create chunks with controlled overlap |
| 30 | Hierarchical chunking | Creates multiple levels of chunks, such as document, section and paragraph |
| 31 | Parent-child chunking | Stores small child chunks for retrieval and larger parent chunks for context |
| 32 | Sentence-window chunking | Retrieves a sentence but provides surrounding sentences as additional context |
| 33 | Small-to-big chunking | Retrieves smaller chunks and expands them into larger contextual sections |
| 34 | Contextual chunking | Adds document or section context to individual chunks before embedding |
| 35 | Recursive semantic chunking | Combines recursive splitting with semantic boundary detection |

---

## 5. Specialized Chunking Strategies

Useful for specific document types and production applications.

| # | Strategy | Description |
|---|---|---|
| 36 | Table-based chunking | Splits tables while preserving headers and row relationships |
| 37 | Code-based chunking | Splits source code using functions, classes and methods |
| 38 | Function-based chunking | Creates one chunk per function |
| 39 | AST-based chunking | Uses Abstract Syntax Trees to understand code structure |
| 40 | Metadata-aware chunking | Uses metadata to organize and enrich chunks |
| 41 | Multimodal chunking | Splits documents containing text, images, audio and video |
| 42 | Image-aware chunking | Preserves relationships between images and associated text |
| 43 | Audio chunking | Splits audio based on timestamps, sentences or speaker changes |
| 44 | Video chunking | Splits videos using scenes, timestamps or frame intervals |
| 45 | Spatial/layout-aware chunking | Preserves relationships between content based on document layout |

---

## 6. Comparison of Important Strategies

These are particularly worth experimenting with when improving retrieval quality.

| Strategy | Complexity | Best Use Case |
|---|---|---|
| Fixed-size | Low | Basic text documents |
| Overlapping | Low | Maintaining neighboring context |
| Recursive | Low–Medium | General-purpose documents |
| Sentence-based | Low | Short textual content |
| Semantic | Medium–High | Topic-changing documents |
| Proposition-based | High | Fact-based knowledge retrieval |
| Parent-child | Medium–High | Large knowledge bases |
| Late chunking | High | Context-dependent text |
| Agentic | High | Complex document structures |
| AST-based | Medium–High | Code repositories |

---

## 7. Recommended Learning Order

Don't implement all 45 strategies immediately. Follow this progression:

### 1. Beginner

- Fixed-size chunking
- Fixed-token chunking
- Overlapping chunking
- Sentence-based chunking
- Paragraph-based chunking

### 2. Intermediate

- Recursive character chunking
- Markdown chunking
- Semantic chunking
- Embedding-based chunking
- Metadata-aware chunking

### 3. Advanced

- Hierarchical chunking
- Parent-child chunking
- Sentence-window chunking
- Proposition-based chunking
- Contextual chunking

### 4. Expert / Production

- Late chunking
- Agentic chunking
- AST-based chunking
- Multimodal chunking
- Adaptive chunking

---

## 8. Final Recommendation

Start with basic chunking techniques and gradually move toward advanced strategies.

For practical RAG development, prioritize:

1. Recursive character chunking for general-purpose documents.
2. Semantic chunking for meaning-aware document segmentation.
3. Parent-child chunking for better retrieval and contextual expansion.
4. Metadata-aware chunking for structured document retrieval.
5. Late chunking for context-dependent embedding generation.
6. AST-based chunking for code repositories.

**Important:** No single chunking strategy is best for every application. The ideal approach depends on document structure, embedding models, retrieval requirements, and the type of questions users ask.
