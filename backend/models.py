from typing import Literal
from pydantic import BaseModel

class RelevancyDecision(BaseModel):
    is_relevant: bool
    reason: str  # reason for taking the decision

class SupersedingPaper(BaseModel):
    title: str
    url: str
    summary: str

class ClaimVerificationResult(BaseModel):
    is_superseded: bool
    verdict_summary: str
    superseding_papers: list[SupersedingPaper]

class BtwRouteDecision(BaseModel):
    needs_web_search: bool   # checks if the user query needs web_search or can it answer directly

class PlannerDecision(BaseModel):
    action: Literal[
        "retrieve",
        "discover_papers",
        "verify_claim",
        "direct_answer",
    ]

# --- GUARDRAIL MODELS ---

class InputGuardrailResult(BaseModel):
    """
    Pre-graph input guardrail check. Catches empty/too-short queries,
    queries that are too long (token-cost risk), and clearly unsafe
    or out-of-scope requests before any retrieval/LLM cost is incurred.
    """
    is_valid: bool
    reason: str