
from app.llm_wrapper import OllamaStructured as ollama

from app.config import LLM_MODEL


class QueryProcessor:

    def __init__(self):

        self.llm = ollama(
            model=LLM_MODEL,
            temperature=0
        )

    def rewrite_query(self, query):

        prompt = f"""
        Rewrite the following query to improve
        document retrieval.

        Preserve the original intent.

        Return only the rewritten query.

        Query:
        {query}
        """

        response = self.llm.invoke(prompt)

        return response.strip()

    def generate_multi_queries(self, query):

        prompt = f"""
        Generate 3 alternative search queries
        for the following question.

        Preserve the original intent.

        Return one query per line.
        Do not add numbering.

        Question:
        {query}
        """

        response = self.llm.invoke(prompt)

        queries = [
            line.strip("- ").strip()
            for line in response.splitlines()
            if line.strip()
        ]

        return queries[:3]

    def decompose_query(self, query):

        prompt = f"""
        Break this complex question into
        smaller independent search questions.

        Return one question per line.
        Do not add numbering.

        Question:
        {query}
        """

        response = self.llm.invoke(prompt)

        questions = [
            line.strip("- ").strip()
            for line in response.splitlines()
            if line.strip()
        ]

        return questions[:4]

    def generate_hyde_document(self, query):

        prompt = f"""
        Write a hypothetical informative passage
        that would answer the following question.

        The passage is used for semantic retrieval.
        It does not need to be factually verified.

        Question:
        {query}
        """

        response = self.llm.invoke(prompt)

        return response.strip()
