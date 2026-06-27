from pydantic import BaseModel
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate

class PlannerDecision(BaseModel):
    action: Literal[
        "retrieve",
        "discover_papers",
        "verify_claim",
        "direct_answer",
    ]
PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a routing agent for a research paper assistant.
            Choose exactly ONE action:
            retrieve
            - Use when the query can plausibly be answered from uploaded papers.
            - If one or more papers are loaded, retrieval is the default action.
            - When in doubt between retrieve and direct_answer, ALWAYS choose retrieve.
            - If the user references a concept, system, method, model, dataset, experiment, result, claim, figure, table, section, author, citation, acronym, term, or topic that may exist in the uploaded papers, choose retrieve.
            - This includes but is not limited to:
            * summaries
            * explanations
            * architecture
            * methods
            * results
            * findings
            * comparisons
            * differences
            * advantages
            * disadvantages
            * conclusions
            * recommendations
            * implementation details
            * performance metrics
            * benchmarks
            * evaluations
            * experiments
            * datasets
            * ablations
            * future work
            * limitations
            * related work
            * formulas
            * proofs
            * definitions
            * terminology
            * citations
            * tables
            * figures
            * security
            * deployment
            * design decisions
            * technical details
            - If papers are loaded and the user asks:
            * "What is..."
            * "What does X mean..."
            * "Who are..."
            * "Explain..."
            * "Compare..."
            * "How does..."
            * "Why..."
            * "Summarize..."
            * "Tell me about..."
            * "What are the differences..."
            * "What does the paper say..."
            * "What are the findings..."
            choose retrieve.

            discover_papers
            - Use when the user explicitly wants new papers found, searched, discovered, or recommended.

            verify_claim
            - Use when the user asks whether a specific fact/claim is true, supported, contradicted, or outdated according to recent literature.

            direct_answer
            - Use ONLY when the query is clearly conversational or unrelated to research content.
            - Examples:
            * greetings
            * small talk
            * jokes
            * casual conversation
            * general knowledge unrelated to uploaded papers
            * arithmetic and simple utility questions
            - Never use direct_answer if papers are loaded and the query could reasonably relate to the content of the uploaded papers.
            Return action only.
            """
        ),
        ("human", "{query}")
    ]
)

def get_planner_chain(llm):
    return (
        PLANNER_PROMPT | llm.with_structured_output(PlannerDecision)
    )