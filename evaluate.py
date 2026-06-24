import json
import sys
from pathlib import Path  # to define paths
from uuid import uuid4   # to create session id

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from deepeval import evaluate    # API which controls our evaluation tasks
from deepeval.evaluate import AsyncConfig  # utility 
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,   # metrics
)
from deepeval.synthesizer import Synthesizer # component which curates our dataset from the LLM
from deepeval.synthesizer.config import ContextConstructionConfig   # sets configuration of synthesizer
from deepeval.test_case import LLMTestCase  # to build testcases

from backend.paper_loader import load_document
from backend.rag_graph import build_graph
from backend.vector_store import add_paper_if_new

from deepeval_gpt import GPTModel

load_dotenv()

PDF_PATH = "documents/Openclaw_Research_Report.pdf"
GOLDENS_FILE        = Path("goldens.json")  # curated QA pairs (goldens) will be stored in this file
MAX_CONTEXTS        = 5  # after filteration we'll have 5 groups as our context at the end of curation step
GOLDENS_PER_CONTEXT = 2  # 2 goldens (2 qa pairs) from each group - total 10 pairs overall
METRIC_THRESHOLD    = 0.7 # minimum score 

def generate_goldens() -> list[dict]:
    synthesizer = Synthesizer(
    model = GPTModel())  # Synthesizer object
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths = [PDF_PATH],  # doc path is in the form of a list - can pass one doc or multiple docs
        include_expected_output = True,
        max_goldens_per_context = GOLDENS_PER_CONTEXT,
        context_construction_config = ContextConstructionConfig(
            max_contexts_per_document = MAX_CONTEXTS,
        ),
    )

    pairs = [     # saving the above goldens into a json file
        {"input": g.input, 
         "expected_output": g.expected_output}
        for g in goldens
        if g.input and g.expected_output  # Keeps only valid samples. Removes any sample where question or answer is missing 
    ]

    GOLDENS_FILE.write_text(
        json.dumps(pairs, indent=2, ensure_ascii=False), # ensure_ascii = False - preserves non-English characters as-is
        encoding="utf-8")  # saves so that all Unicode characters (₹, తెలుగు, 中文, etc.) are stored correctly
    return pairs

def load_goldens() -> list[dict]:  # utility function to load json file
    return json.loads(GOLDENS_FILE.read_text(encoding="utf-8"))

def run_rag_query(graph, query: str, session_id: str) -> tuple[str, list[str]]:   # this fcn runs rag query
    config = {
        "configurable": {"thread_id": str(session_id)}
    }

    final_state = graph.invoke(             # invokes rag graph
        {
            "messages": [HumanMessage(content=query)],
            "session_id": session_id,
            "query": query,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "rewrite_count": 0,
            "paper_filter": None, 
            "route": None,
            "claim_verdict": None,
            "claim_source": None,
            "superseding_papers": [],
            "answer": None,
            "is_relevant": None,
            "avg_rerank_score": None,
        },
        config=config,
    )

    answer = final_state.get("answer") or ""    # answer - generational LLM output of generation node of graph
    retrieval_context = [doc.page_content for doc in (final_state.get("retrieved_docs") or [])] # retrieved docs - stored in retrieval_context 
    return answer, retrieval_context   # now we'll have input query, grounded response (from the above fcn) and generated response, retrieved context from this fcn - we can now build our EVALUATION DATASET

def mean_reciprocal_rank(expected_answer: str, retrieved_context: list[str]) -> float:
    if not retrieved_context:
        return 0.0
    
    expected_terms = {
        word.lower() for word in expected_answer.split() if len(word) > 3
    }
    
    for rank, chunk in enumerate(retrieved_context, start=1):
        chunk_terms = {word.lower() for word in chunk.split()}
        if expected_terms.intersection(chunk_terms):
            return round(1 / rank, 4)
    
    return 0.0

def main() -> None:
    pairs = load_goldens()

    docs = load_document(PDF_PATH)
    graph = build_graph(db_path = "eval_checkpoints.db")   # adding the loaded document in vector_store
 
    judge_model = GPTModel()

    metrics = [
        ContextualPrecisionMetric(threshold = METRIC_THRESHOLD, model = judge_model),
        ContextualRecallMetric(threshold = METRIC_THRESHOLD, model = judge_model),   
        ContextualRelevancyMetric(threshold = METRIC_THRESHOLD, model = judge_model),
        AnswerRelevancyMetric(threshold = METRIC_THRESHOLD, model = judge_model),
        FaithfulnessMetric(threshold = METRIC_THRESHOLD, model = judge_model),      # LLM-as-a-judge - gpt-5-mini 
    ]

    test_cases = []
    mrr_scores = []

    for pair in pairs:   # loop over each {"input": ..., "expected_output": ...} from goldens.json.
        session_id = f"evaluation_session_{uuid4()}"  # every golden has a unique session_id - fresh start of RAG pipeline for every QA pair 
        add_paper_if_new(docs, session_id)   # indexes the PDF into Qdrant under that session ID so the retriever has something to search against.

        query = pair["input"] + " as per the report in knowledge base"    # appends a hint to the query so the planner routes to "retrieve" node instead of "direct_answer"
        answer, retrieval_context = run_rag_query(graph, query, session_id)   # Runs the full RAG pipeline and returns the generated answer and the retrieved chunks.

        mrr_score = mean_reciprocal_rank(pair["expected_output"], retrieval_context)
        mrr_scores.append(mrr_score)  # reuse the same list, just rename conceptually

        test_cases.append(      # Builds a DeepEval evaluation dataset with all four components needed to run the 5 metrics.
            LLMTestCase(  # our evaluation dataset
                input = pair["input"],  # query of each QA pair
                actual_output = answer, # LLM generated response
                expected_output = pair["expected_output"],
                retrieval_context = retrieval_context, # output of retriever
            )
        )

    results = evaluate(   # results returns deepeval.evaluate.EvaluationResult object containing ".test_results", ".success"
        test_cases,
        metrics,
        async_config = AsyncConfig(max_concurrent=2, throttle_value=6),  # evaluates all test cases against all 5 metrics and max 3 can run in parallel, with a 5 second throttle between batches to avoid rate limits
    )

    summary = []
    for idx, test_result in enumerate(results.test_results):    # storing results in a json file
        summary.append({
            "input": test_result.input,
            "actual_output": test_result.actual_output,
            "success": test_result.success,
            "mrr": mrr_scores[idx],
            "metrics": [
                {
                    "name": m.name,
                    "score": m.score,
                    "passed": m.success,
                    "reason": m.reason,
                }
                for m in test_result.metrics_data
            ],
        })

    results_path = Path("eval_results.json")   # json file name
    results_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {results_path}.")    # Dumps the entire summary to :eval_results.json" file

if __name__ == "__main__":
    main()
