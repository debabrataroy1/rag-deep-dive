
# Production RAG Search Architecture

## How to Manage Multiple Search Techniques in a Single RAG Application

# 1. Introduction

In real-world RAG applications, we don't enable every search technique for every query.

Different search techniques solve different problems.

For example:

- BM25 is useful for exact keyword matching.
- Vector Search is useful for semantic similarity.
- Hybrid Search combines keyword and semantic search.
- Graph Search is useful for relationship-based questions.
- HyDE improves retrieval for vague or short queries.
- Query Decomposition handles complex questions.
- Query Expansion improves search coverage.
- RRF combines results from multiple search engines.
- Reranking improves the ordering of retrieved documents.

The main architectural challenge is:

**How can one RAG application intelligently manage all these techniques?**

The solution is to build an intelligent search orchestration layer.

This layer analyzes the user's query, selects the appropriate search techniques, executes searches, combines results, and passes relevant context to the LLM.

---

# 2. How a Single RAG Application Manages Multiple Search Techniques

Consider an Enterprise RAG application containing:

- HR documents
- Technical documentation
- Financial reports
- Product documentation
- Customer support knowledge
- Company policies

Different users ask different questions.

| User Query | Appropriate Search Strategy |
|---|---|
| What is the leave policy? | Hybrid Search |
| Find document containing error code E1024 | BM25 |
| How can I improve API performance? | Hybrid + Query Expansion |
| Explain the relationship between projects and departments | Graph Search |
| Compare financial reports from 2024 and 2025 | Metadata + Hybrid Search |
| What about its disadvantages? | Contextual Search |
| Find documents about cloud security using different terminology | Multi-Query Search |
| Explain a complex topic using multiple documents | Query Decomposition |

One fixed search strategy cannot handle every query equally well.

A production RAG application therefore needs multiple search capabilities and a mechanism to select them dynamically.

---

# 3. Intelligent RAG Search Router

The solution is to introduce a Query Router before the search layer.

## 3.1 Architecture

```text
                    USER QUERY
                         |
                         v
                QUERY UNDERSTANDING
                         |
                         v
                 QUERY CLASSIFIER
                         |
                         v
                   QUERY ROUTER
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
     SIMPLE QUERY   COMPLEX QUERY   ENTITY QUERY
          |              |              |
          v              v              v
     Hybrid Search  Query Decomposition Graph Search
          |              |              |
          |         Multi-Query Search   |
          |              |              |
          +--------------+--------------+
                         |
                         v
                  RESULT AGGREGATION
                         |
                         v
                 DEDUPLICATION
                         |
                         v
                     RERANKING
                         |
                         v
                 CONTEXT SELECTION
                         |
                         v
                  LLM GENERATION
                         |
                         v
                   FINAL ANSWER
```

## 3.2 How It Works

1. User submits a query.
2. Query Understanding analyzes the query.
3. Query Classifier identifies the query type.
4. Query Router selects appropriate search strategies.
5. Selected search engines execute their searches.
6. Results are combined.
7. Duplicate documents are removed.
8. Results are reranked.
9. Relevant context is selected.
10. LLM generates the final answer.

Important:

Not every query passes through every search technique.

The router selects only the techniques that are appropriate for the query.

---

# 4. Query Routing Approaches

There are three common approaches to query routing.

## 4.1 Rule-Based Routing

### What is it?

Rule-Based Routing uses predefined conditions to select search strategies.

Rules can be based on:

- Keywords
- Query patterns
- Exact identifiers
- Query length
- Metadata conditions
- Business requirements

### Example

```python
def route_query(query):

    query = query.lower()

    if any(word in query for word in [
        "error code", "exact id", "document number"
    ]):
        return "bm25"

    elif any(word in query for word in [
        "relationship", "connected", "related to"
    ]):
        return "graph"

    elif any(word in query for word in [
        "compare", "difference", "versus"
    ]):
        return "multi_query"

    elif any(word in query for word in [
        "explain", "how", "why"
    ]):
        return "hybrid"

    return "hybrid"
```

### Advantages

- Easy to implement.
- Low latency.
- Predictable behavior.
- No additional LLM cost.
- Easy to debug.

### Disadvantages

- Rules become difficult to maintain.
- Cannot understand every variation in natural language.
- Complex queries may require multiple strategies.
- Requires manual rule updates.

### Best Fit

- Small RAG applications.
- Predictable business workflows.
- Applications with limited query types.

---

## 4.2 LLM-Based Query Routing

### What is it?

LLM-Based Routing uses an LLM to analyze the user's query and decide which search strategy is appropriate.

The LLM classifies the query and generates a structured search plan.

### Example 1: Complex Comparison Query

User query:

"Compare the security architecture of our old and new applications."

LLM-generated routing decision:

```json
{
  "query_type": "comparison",
  "complexity": "high",
  "strategies": [
    "query_decomposition",
    "hybrid_search",
    "reranking"
  ],
  "metadata_filters": {},
  "requires_multiple_documents": true
}
```

### Example 2: Exact Lookup Query

User query:

"Find error code E1024."

LLM-generated routing decision:

```json
{
  "query_type": "exact_lookup",
  "complexity": "low",
  "strategies": [
    "bm25"
  ],
  "metadata_filters": {},
  "requires_multiple_documents": false
}
```

### Advantages

- Understands natural language.
- Handles different query types.
- Supports multiple search strategies.
- More flexible than fixed rules.
- Can identify complex user intent.

### Disadvantages

- Additional LLM latency.
- Additional token cost.
- Classification errors are possible.
- Requires structured output validation.

### Best Fit

- Enterprise RAG.
- Conversational AI applications.
- Complex knowledge assistants.
- Multi-domain search applications.

---

## 4.3 Hybrid Routing

### What is it?

Hybrid Routing combines Rule-Based Routing and LLM-Based Routing.

It uses simple rules for predictable queries and an LLM for complex queries.

### How It Works

1. Check whether the query contains an exact identifier.
2. Check metadata constraints.
3. Identify whether it is a follow-up question.
4. Use rules for straightforward queries.
5. Use an LLM for complex query classification.
6. Select the search strategies.
7. Execute searches.

### Advantages

- Balances latency and flexibility.
- Reduces unnecessary LLM calls.
- Supports complex queries.
- Easier to maintain than a completely LLM-driven router.

### Best Fit

Enterprise RAG applications and production conversational search systems.

**Recommended Approach: Hybrid Routing.**


# 5. Combining Multiple Search Techniques in One Query

Sometimes one query requires multiple search techniques simultaneously.

Consider:

"Compare the security policies of the HR and IT departments published after 2024."

This query needs multiple techniques.

| Technique | Why? |
|---|---|
| Query Decomposition | Separate HR and IT policy searches |
| Metadata Filtering | Filter documents published after 2024 |
| Hybrid Search | Find relevant documents |
| RRF | Combine search results |
| Reranking | Rank the most relevant documents |

## 5.1 Execution Flow

```text
                 USER QUERY
                      |
                      v
              QUERY DECOMPOSITION
                      |
             +--------+--------+
             |                 |
             v                 v
        HR Policy Query   IT Policy Query
             |                 |
             v                 v
        Metadata Filter  Metadata Filter
        Year > 2024       Year > 2024
             |                 |
             v                 v
        Hybrid Search    Hybrid Search
             |                 |
             +--------+--------+
                      |
                      v
                     RRF
                      |
                      v
                  RERANKING
                      |
                      v
                CONTEXT ASSEMBLY
                      |
                      v
                  LLM ANSWER
```

This demonstrates how multiple search techniques can work together in one RAG application.

---

# 6. Practical Production Architecture

A real-world enterprise RAG application can be organized into the following components.

## 6.1 Component Responsibilities

### Component 1: Query Understanding

Responsibilities:

- Identify user intent.
- Extract important entities.
- Detect exact identifiers.
- Identify metadata conditions.
- Understand conversation context.
- Detect complex questions.

### Component 2: Query Router

Responsibilities:

- Select search strategies.
- Select appropriate knowledge sources.
- Decide whether multiple searches are necessary.
- Determine whether query transformation is needed.

### Component 3: Query Processing

Possible techniques:

- Query Rewriting
- Query Expansion
- HyDE
- Query Decomposition
- Multi-Query Search
- Step-Back Prompting
- Contextual Search

Responsibility:

Improve the original query before search execution.

### Component 4: Search Execution

Possible search engines:

- BM25
- Dense Vector Search
- Sparse Vector Search
- Graph Search
- Metadata Filtering
- Multimodal Search

Responsibility:

Find relevant documents from the knowledge base.

### Component 5: Result Processing

Possible techniques:

- Reciprocal Rank Fusion
- Deduplication
- Reranking
- Maximal Marginal Relevance

Responsibility:

Combine, rank, and select relevant search results.

### Component 6: Context Builder

Responsibilities:

- Select relevant document chunks.
- Retrieve parent document context if needed.
- Remove redundant information.
- Respect the LLM context window.
- Preserve document references and citations.

### Component 7: LLM Generation

Responsibilities:

- Generate answers using retrieved context.
- Avoid unsupported claims.
- Include source references.
- Handle insufficient evidence appropriately.

## 6.2 Architecture Diagram

```text
                  USER QUERY
                       |
                       v
              QUERY UNDERSTANDING
                       |
                       v
                  QUERY ROUTER
                       |
                       v
             +---------+---------+
             |                   |
             v                   v
       QUERY PROCESSING     SEARCH SELECTION
             |                   |
       +-----+------+       +----+-----+
       |     |      |       |    |     |
       v     v      v       v    v     v
     HyDE  Rewrite  Multi   BM25 Vector Graph
                   Query          Search Search
       |     |      |       |    |     |
       +-----+------+       +----+-----+
             |                   |
             +---------+---------+
                       |
                       v
                SEARCH EXECUTION
                       |
                       v
                RESULT AGGREGATION
                       |
                       v
                     RRF
                       |
                       v
                  RERANKING
                       |
                       v
                  MMR / FILTER
                       |
                       v
                 CONTEXT BUILDER
                       |
                       v
                 LLM GENERATION
                       |
                       v
                  FINAL ANSWER
```

This is a reference architecture, not a mandatory pipeline.

A simple RAG application may need only a subset of these components.

---

# 7. Practical Python Implementation

The following example demonstrates how one RAG application can select and combine multiple search strategies.

It is a simplified orchestration example. Search engines, databases, rerankers, and LLM functions are represented by integration placeholders.

## 7.1 Define a Search Plan

First, define a search plan.

```python
from dataclasses import dataclass


@dataclass
class SearchPlan:

    strategies: list

    filters: dict

    reason: str
```

## 7.2 Query Classification

```python
def classify_query(query):

    q = query.lower()

    # Exact identifier search
    if "error code" in q or "document id" in q:

        return SearchPlan(
            strategies=["bm25"],
            filters={},
            reason="Exact identifier lookup"
        )

    # Relationship-based search
    if any(x in q for x in [
        "relationship",
        "connected to",
        "related to",
        "linked with"
    ]):

        return SearchPlan(
            strategies=["graph", "hybrid"],
            filters={},
            reason="Relationship-based query"
        )

    # Comparison queries
    if any(x in q for x in [
        "compare",
        "comparison",
        "difference"
    ]):

        return SearchPlan(
            strategies=[
                "query_decomposition",
                "hybrid",
                "reranking"
            ],
            filters={},
            reason="Multi-part comparison"
        )

    # General semantic questions
    return SearchPlan(
        strategies=[
            "hybrid",
            "reranking"
        ],
        filters={},
        reason="General semantic question"
    )
```

## 7.3 Search Execution

```python
def execute_search(query, plan):

    results = []

    for strategy in plan.strategies:

        if strategy == "bm25":

            results.append(
                bm25_search(query)
            )

        elif strategy == "hybrid":

            results.append(
                hybrid_search(query)
            )

        elif strategy == "graph":

            results.append(
                graph_search(query)
            )

        elif strategy == "query_decomposition":

            sub_queries = decompose_query(query)

            for sub_query in sub_queries:

                results.append(
                    hybrid_search(sub_query)
                )

        elif strategy == "reranking":

            # Reranking is applied after
            # candidate aggregation.

            continue

    combined = reciprocal_rank_fusion(results)

    ranked = rerank(combined, query)

    return ranked
```

## 7.4 Complete RAG Pipeline

```python
def rag_pipeline(query):

    # Step 1: Classify the query
    plan = classify_query(query)

    print(
        "Selected strategies:",
        plan.strategies
    )

    print(
        "Reason:",
        plan.reason
    )

    # Step 2: Execute selected searches
    documents = execute_search(
        query,
        plan
    )

    # Step 3: Build context
    context = build_context(documents)

    # Step 4: Generate answer
    answer = generate_answer(
        query=query,
        context=context
    )

    return answer
```

## 7.5 Example Execution

```python
query = """
Compare Python and Java in terms of
performance and scalability.
"""

answer = rag_pipeline(query)

print(answer)
```

Expected routing:

```text
Selected strategies:

- Query Decomposition
- Hybrid Search
- Reranking

Reason:

Multi-part comparison
```

The actual answer depends on the connected knowledge base and search implementations.

### Production Considerations

A production implementation should additionally include:

- LLM-based structured routing where appropriate.
- Query and metadata validation.
- Access-control filtering.
- Parallel execution of independent searches.
- Search timeouts and fallbacks.
- Result deduplication.
- Source and citation tracking.
- Logging and monitoring.
- Retrieval evaluation.

Important: Reranking is a ranking stage, not an independent search engine. It should operate on aggregated candidate documents.

---

# 8. Recommended Implementation Strategy

Do not implement all search techniques at once.

Build your RAG application incrementally.

| Stage | Implementation |
|---|---|
| Stage 1 | Dense Vector Search + BM25 |
| Stage 2 | Hybrid Search + RRF |
| Stage 3 | Metadata Filtering + Reranking |
| Stage 4 | Query Rewriting + Contextual Search |
| Stage 5 | Query Router + Query Decomposition |
| Stage 6 | HyDE + Multi-Query Search |
| Stage 7 | GraphRAG for relationship-based questions |
| Stage 8 | Agentic Search for complex multi-source workflows |

At every stage, evaluate whether the new technique improves retrieval quality.

Do not introduce a technique simply because it is advanced.


# 9. Important Architectural Distinction

There are three major categories that should be kept separate.

| Category | Examples | Responsibility |
|---|---|---|
| Query Processing | HyDE, Query Expansion, Decomposition, Rewriting | Improve the query |
| Retrieval | BM25, Vector Search, Graph Search, ANN | Find candidate information |
| Ranking and Selection | RRF, Reranking, MMR | Select and order relevant results |

The LLM generation layer comes after these stages.

## 9.1 Query Processing

Query Processing modifies or expands the user's query before searching.

### Example

Original query:

```text
How to improve API performance?
```

Expanded queries:

- API optimization
- Database performance
- Caching strategies
- Connection pooling
- API scalability

Possible techniques:

- Query Rewriting
- Query Expansion
- HyDE
- Query Decomposition
- Multi-Query Search
- Step-Back Prompting
- Contextual Search

Purpose:

Improve the quality of the search query and increase the chances of finding relevant documents.

## 9.2 Retrieval

Retrieval searches the knowledge base and finds candidate documents.

Possible techniques:

- BM25 Search
- Dense Vector Search
- Sparse Vector Search
- Hybrid Search
- Graph Search
- Multimodal Search
- ANN Search

### Example

User query:

```text
How to improve API performance?
```

The retrieval system searches the knowledge base and returns documents related to:

- API optimization
- Database indexing
- Caching
- Connection pooling

Purpose:

Find relevant information from the knowledge base.

## 9.3 Ranking and Selection

Ranking and Selection processes retrieved candidates.

Possible techniques:

- Reciprocal Rank Fusion (RRF)
- Cross-Encoder Reranking
- Maximal Marginal Relevance (MMR)
- Score-based Filtering
- Deduplication

### Example

Suppose the retrieval system returns 100 documents.

The ranking pipeline:

1. Combines results from different search engines.
2. Removes duplicate documents.
3. Reranks the candidates.
4. Selects the most relevant documents.
5. Reduces redundant information.

Purpose:

Improve the quality and relevance of the final retrieved context.

## 9.4 Generation

The LLM uses the selected context to produce the final answer.

Responsibilities:

- Generate answers using retrieved context.
- Avoid unsupported claims.
- Include source references.
- Handle insufficient evidence appropriately.

---

# 10. Final Architectural Recommendation

A practical production RAG application should start with the following architecture.

## 10.1 Basic Production Architecture

```text
                 USER QUERY
                      |
                      v
              QUERY UNDERSTANDING
                      |
                      v
                 QUERY ROUTER
                      |
                      v
             HYBRID SEARCH PIPELINE
                /           \
               v             v
          BM25 Search    Vector Search
               |             |
               +------+------+
                      |
                      v
                     RRF
                      |
                      v
                  RERANKING
                      |
                      v
                CONTEXT BUILDER
                      |
                      v
                 LLM GENERATION
                      |
                      v
                 FINAL ANSWER
```

This is a good starting point for many enterprise RAG applications.

## 10.2 Extend the Architecture Based on Requirements

Additional techniques should be introduced only when the application needs them.

| Requirement | Technique to Consider |
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

## 10.3 Recommended Production Components

| Component | Recommended Approach |
|---|---|
| Query Understanding | LLM or lightweight classifier |
| Query Routing | Hybrid Routing |
| Keyword Search | BM25 |
| Semantic Search | Dense Vector Search |
| Combined Search | Hybrid Search |
| Result Combination | RRF |
| Document Ranking | Cross-Encoder Reranking |
| Context Selection | MMR and Context Builder |
| Relationship Search | Graph Search when required |
| Answer Generation | LLM |

These are recommendations, not mandatory components for every application.

---

# 11. Evaluation and Monitoring

A production RAG system should continuously evaluate search quality.

Adding more search techniques does not automatically improve the application.

Each technique should be evaluated against a representative dataset of real user questions.

## 11.1 Important Retrieval Metrics

| Metric | Purpose |
|---|---|
| Recall@K | Measures how many relevant documents are retrieved in the top K |
| Precision@K | Measures how many retrieved documents are relevant |
| MRR | Measures the position of the first relevant result |
| NDCG@K | Measures ranking quality |
| Hit Rate@K | Checks whether at least one relevant document appears in the top K |
| Latency | Measures search response time |
| Cost | Measures infrastructure and model expenses |

## 11.2 Compare Search Strategies

For example, test the following configurations:

| Configuration | Search Strategy |
|---|---|
| A | BM25 |
| B | Dense Vector Search |
| C | Hybrid Search |
| D | Hybrid Search + RRF |
| E | Hybrid Search + RRF + Reranking |
| F | Query Decomposition + Hybrid Search + Reranking |

Compare their:

- Recall@K
- Precision@K
- MRR
- NDCG@K
- Latency
- Cost

This helps identify whether additional techniques provide measurable improvements.

## 11.3 Monitoring in Production

Track the following information for each query:

- Original user query.
- Classified query type.
- Selected search strategy.
- Search engines executed.
- Retrieved document IDs.
- Retrieval scores.
- Reranking scores.
- Number of documents passed to the LLM.
- Total response latency.
- Token consumption.
- User feedback.
- Answer quality.

This information helps identify routing mistakes and retrieval problems.

Important:

Apply access-control and tenant filters before documents are exposed to the LLM. A search router must never bypass document permissions.

---

# 12. Real-World Example: Enterprise RAG Application

Consider a company that wants to build one RAG application for all departments.

Its knowledge base contains:

```text
Company Knowledge Base
          |
          +-- HR
          |    |
          |    +-- Leave Policies
          |    +-- Payroll Documents
          |    +-- Employee Handbook
          |
          +-- IT
          |    |
          |    +-- API Documentation
          |    +-- Security Policies
          |    +-- Troubleshooting Guides
          |
          +-- Finance
          |    |
          |    +-- Financial Reports
          |    +-- Revenue Statements
          |
          +-- Sales
               |
               +-- Product Documentation
               +-- Customer Information
```

Different employees submit different questions.

## Scenario 1: HR Policy Question

User:

```text
What is the company's annual leave policy?
```

Selected techniques:

- Hybrid Search
- Metadata Filtering
- Reranking

Reason:

The query requires finding relevant HR policy documents.

## Scenario 2: Technical Error Lookup

User:

```text
Find the troubleshooting instructions for error E1024.
```

Selected techniques:

- BM25
- Metadata Filtering

Reason:

The exact error identifier is important.

## Scenario 3: Complex Comparison

User:

```text
Compare the security policies of the IT department
from 2024 and 2025.
```

Selected techniques:

- Query Decomposition
- Metadata Filtering
- Hybrid Search
- RRF
- Reranking

Reason:

The system needs to retrieve documents from two different periods and compare their content.

## Scenario 4: Relationship-Based Question

User:

```text
Which projects are connected to the cloud migration initiative?
```

Selected techniques:

- Query Understanding
- Graph Search
- Hybrid Search

Reason:

The question requires exploring relationships between projects and entities.

## Scenario 5: Conversational Follow-Up

User:

```text
What are the benefits of our leave policy?
```

Follow-up:

```text
What about its limitations?
```

Selected techniques:

- Contextual Search
- Query Rewriting
- Hybrid Search

Reason:

The system needs to understand that "its" refers to the leave policy discussed previously.

## Scenario 6: Research Question

User:

```text
Explain the major challenges of migrating
enterprise applications to cloud infrastructure.
```

Selected techniques:

- Query Decomposition
- Multi-Query Search
- Hybrid Search
- RRF
- Reranking
- MMR

Reason:

The question covers multiple dimensions and may require evidence from several documents.

---

# 13. Final Best Practices

## 13.1 Start with a Simple Architecture

Begin with:

1. Dense Vector Search.
2. BM25.
3. Hybrid Search.
4. Metadata Filtering.
5. RRF.
6. Reranking.

This provides a strong baseline for many RAG applications.

## 13.2 Introduce Techniques Based on Problems

Do not select techniques based only on popularity.

Identify the actual problem first.

For example:

- Poor semantic matching: Evaluate embedding models or HyDE.
- Missing exact terms: Improve BM25 or hybrid weighting.
- Complex questions: Introduce query decomposition.
- Repeated documents: Introduce MMR.
- Relationship questions: Introduce Graph Search.
- Poor follow-up handling: Introduce contextual query rewriting.

## 13.3 Keep Search Strategies Modular

Implement each search strategy as an independent component.

For example:

```text
rag_application/
    |
    +-- query_understanding/
    |       |
    |       +-- classifier.py
    |       +-- entity_extraction.py
    |
    +-- query_router/
    |       |
    |       +-- router.py
    |       +-- search_plan.py
    |
    +-- query_processing/
    |       |
    |       +-- rewriting.py
    |       +-- hyde.py
    |       +-- decomposition.py
    |
    +-- search/
    |       |
    |       +-- bm25.py
    |       +-- vector_search.py
    |       +-- hybrid_search.py
    |       +-- graph_search.py
    |
    +-- ranking/
    |       |
    |       +-- rrf.py
    |       +-- reranking.py
    |       +-- mmr.py
    |
    +-- context/
    |       |
    |       +-- builder.py
    |
    +-- generation/
    |       |
    |       +-- answer.py
    |
    +-- evaluation/
            |
            +-- metrics.py
            +-- monitoring.py
```

This modular design makes it easier to add, remove, test, and optimize search techniques.

---

# 14. Conclusion

A production RAG application does not need 30 separate RAG systems for 30 different search techniques.

It needs:

1. One Query Understanding layer.
2. One intelligent Query Router.
3. Multiple search capabilities.
4. A Search Orchestration layer.
5. A Result Aggregation and Ranking layer.
6. A Context Builder.
7. One LLM Generation layer.

The router decides which techniques should run for each query.

Multiple techniques can also work together when a query requires them.

**The key principle is:**

Build one RAG application with multiple search capabilities and intelligent strategy selection.

Start simple, evaluate retrieval quality, identify limitations, and introduce advanced search techniques only when they provide measurable improvements.

---

