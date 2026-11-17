
from typing import Literal

from pydantic import BaseModel, Field

from app.config import LLM_MODEL

from app.llm_wrapper import OllamaStructured as ollama

class SearchPlan(BaseModel):

    query_type: Literal[
        "exact_lookup",
        "semantic",
        "comparison",
        "complex",
        "relationship",
        "general"
    ]

    strategies: list[
        Literal[
            "bm25",
            "vector",
            "hybrid",
            "query_rewrite",
            "multi_query",
            "decomposition",
            "hyde"
        ]
    ] = Field(
        description="Search strategies to execute"
    )

    reason: str


class QueryRouter:

    def __init__(self):

        self.llm = ollama(
            model=LLM_MODEL,
            temperature=0
        ).with_structured_output(SearchPlan)

    def route(self, query):

        prompt = f"""
        You are an intelligent RAG query router.

        Analyze the user query and select
        appropriate search strategies.

        Available strategies:

        1. bm25:
           Exact keyword, error code,
           document ID and terminology lookup.

        2. vector:
           Semantic similarity search.

        3. hybrid:
           Combine keyword and vector search.

        4. query_rewrite:
           Rewrite unclear or conversational queries.

        5. multi_query:
           Generate multiple alternative queries.

        6. decomposition:
           Break complex questions into smaller questions.

        7. hyde:
           Generate hypothetical documents
           to improve semantic retrieval.

        Routing guidelines:

        - Exact identifiers: bm25.
        - Simple semantic questions: vector or hybrid.
        - Complex questions: decomposition and hybrid.
        - Comparison questions: decomposition and hybrid.
        - Vague queries: query_rewrite or hyde.
        - Broad research questions: multi_query and hybrid.
        - Relationship questions: hybrid for this
          starter project. Graph retrieval is not
          implemented yet.

        Select only the strategies that are useful.

        User query:
        {query}
        """

        plan = self.llm.invoke(prompt)

        return plan
