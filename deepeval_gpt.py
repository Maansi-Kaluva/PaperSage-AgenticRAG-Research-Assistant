from deepeval.models import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI


class GPTModel(DeepEvalBaseLLM):

    def __init__(self):
        self.model = ChatOpenAI(
            model="gpt-5.4-mini",
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        response = await self.model.ainvoke(prompt)
        return response.content

    def get_model_name(self):
        return "gpt-5.4-mini"