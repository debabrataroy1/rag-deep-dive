
from typing import Type, TypeVar, Optional

from ollama import Client
from pydantic import BaseModel

from langchain_core.runnables import Runnable

T = TypeVar("T", bound=BaseModel)


class OllamaStructured(Runnable):

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0,
        schema=None
    ):

        self.client = Client(host=base_url)
        self.model = model
        self.temperature = temperature
        self._schema = schema

    def with_structured_output(
        self,
        schema: Type[T]
    ):

        return OllamaStructured(
            model=self.model,
            base_url="http://localhost:11434",
            temperature=self.temperature,
            schema=schema
        )

    def invoke(self, input, config=None, **kwargs):

        if isinstance(input, str):
            prompt = input

        elif hasattr(input, "to_string"):
            prompt = input.to_string()

        elif isinstance(input, dict):
            prompt = str(input)

        else:
            prompt = str(input)

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            format=(
                self._schema.model_json_schema()
                if self._schema else ""
            ),
            options={
                "temperature": self.temperature
            }
        )

        content = response.message.content

        if self._schema:
            return self._schema.model_validate_json(content)

        return content
