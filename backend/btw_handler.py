import os
from typing import Generator

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

from backend.models import BtwRouteDecision
from backend.guardrails import MAX_OUTPUT_TOKENS

load_dotenv()

llm = ChatOpenAI(
    model="gpt-5-mini",
    max_tokens=MAX_OUTPUT_TOKENS,  # GENERATION GUARDRAIL: cap output tokens
    )

def handle_btw(query: str) -> Generator[str, None, None]:   # return type says this function will yield strings.
    """Off-topic side channel — never touches the vector store or checkpointer."""
    route_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Decide if answering this question requires a real-time web search (recent events, "
         "current prices, breaking news) or if your general knowledge is sufficient."),
        ("human", "{query}"),
    ])
    decision = (route_prompt | llm.with_structured_output(BtwRouteDecision)).invoke({"query": query})

    if decision.needs_web_search:
        try:
            client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
            results = client.search(query, max_results=3)
            context = "\n\n".join(r["content"] for r in results["results"])
            sources = "\n".join(f"- {r['url']}" for r in results["results"])

            answer_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "Answer the question using the web search results below. Be concise.\n\n"
                 f"Results:\n{context}\n\nSources:\n{sources}"),
                ("human", "{query}"),
            ])
        except Exception:
            # FALLBACK STRATEGY: web search service unavailable.
            # Tell the user clearly and degrade to general knowledge
            # instead of letting the whole request fail.
            yield "Live web search is temporarily unavailable, so I'll answer from general knowledge instead:\n\n"
            answer_prompt = ChatPromptTemplate.from_messages([
                ("system", "Answer the question concisely from your general knowledge."),
                ("human", "{query}"),
            ])

    else:
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question concisely from your general knowledge."),
            ("human", "{query}"),
        ])

    generated_anything = False

    for chunk in (answer_prompt | llm).stream({"query": query}):
        if chunk.content:
            generated_anything = True   
            yield chunk.content

    if not generated_anything:
        yield "Sorry, I couldn't generate a response."      # Every piece of generated text is immediately sent to the caller. Ex:
    # Instead of waiting for: "Python is a programming language..."
    # all at once, the LLM streams:
    # "Python"
    # "is a"
    # "programming"
    # "language..."
    # one chunk at a time