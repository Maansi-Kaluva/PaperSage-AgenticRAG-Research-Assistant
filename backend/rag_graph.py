import os
import sqlite3
import warnings
from typing import Annotated
import concurrent.futures

warnings.filterwarnings("ignore", message="The default value of `allowed_objects`")

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition
from langgraph.types import Command
from pydantic import BaseModel, Field
from tavily import TavilyClient

from backend.models import ClaimVerificationResult, RelevancyDecision
from backend.hybrid_retriever import hybrid_search
from backend.planner_agent import get_planner_chain
from backend.paper_discovery import discover_papers
from backend.guardrails import MAX_OUTPUT_TOKENS, cap_retrieved_docs, validate_generation_output, build_context_with_citations, format_citations_block



load_dotenv()

llm = ChatOpenAI(
    model="gpt-5-mini",
    max_tokens=MAX_OUTPUT_TOKENS,  # GENERATION GUARDRAIL: cap output tokens
    )

# RAG STATE
class RAGState(MessagesState):
    session_id: str
    query: str
    paper_filter: str | None
    route: str | None
    retrieved_docs: list[Document]
    retrieval_attempts: int
    claim_verdict: str | None   # used for verifying a claim/fact - if it is outdated or still the same
    claim_source: str | None
    superseding_papers: list[dict] | None
    answer: str | None
    is_relevant: bool | None
    rewrite_count: int
    avg_rerank_score: float | None
    planner_action: str | None
    discovered_papers: list[dict] | None

# NODE 2: TOOL SCHEMAS FOR TOOL_RETRIEVAL NODE
# telling the agent "HOW" to do the retrieval

class RetrieverInput(BaseModel):   # retrieval from database
    query: str = Field(description="Semantic query to search research paper chunks")
    k: int = Field(default=4, ge=1, le=10, description="Number of chunks to retrieve")  # retrieve 4 chunks by default but LLM can retrieve any number of chunks between 1 and 10


class WebSearchInput(BaseModel):   # retrieval using web search
    optimized_query: str = Field(description="Query rewritten and optimized for web search")  # agent rewrites the original query
    max_results: int = Field(default=3, ge=1, le=10, description="Number of web results to return")

# TOOLS
# if model/agent decides to retrieve from the vectordb or perform vectorstore_search, it must have query and topk value.
# that particular schema is available in RetrieverInput class which will be inherited by the "retrieve_from_vectorstore" tool/fcn

@tool(args_schema = RetrieverInput)
def retrieve_from_vectorstore(
    query: str,   # provided by the LLM when calling the tool.
    k: int,  # provided by the LLM when calling the tool.
    session_id: Annotated[str, InjectedState("session_id")],   # automatically injected from state["session_id"]. 
    # InjectedState - used when we want to add any field of the state into a tool
    # str → actual type of session_id
    # InjectedState("session_id") → extra instruction/metadata for LangGraph saying "automatically inject state['session_id'] here"
    # needed because retrieval is session-specific (e.g., user-specific docs, chat-specific storage, logging, checkpoints).

    paper_filter: Annotated[str | None, InjectedState("paper_filter")],
    current_docs: Annotated[list, InjectedState("retrieved_docs")],   # syntax - Annotated[type, metadata]
    tool_call_id: Annotated[str, InjectedToolCallId], # automatically injected by LangGraph for the current tool call. this gives an id for every toolcall made by the agent. When the tool returns a ToolMessage, you attach the same tool_call_id so LangGraph/LLM knows which tool call that result belongs to.
) -> list: 
    """Search the uploaded research paper vector store for relevant passages."""
    docs = hybrid_search(
        query,
        session_id,
        paper_title = paper_filter,
    )
    if not docs:
        return [ToolMessage(content="No relevant documents found in the vector store.", tool_call_id=tool_call_id)]

    # RETRIEVAL GUARDRAIL: hard cap on number of chunks passed downstream,
    # regardless of what the retriever returns.
    docs = cap_retrieved_docs(docs)

    summary = f"Retrieved {len(docs)} chunk(s) from the vector store."
    return [
        ToolMessage(content=summary, tool_call_id=tool_call_id),
        Command(update={"retrieved_docs": (current_docs or []) + docs}),  # Command - helps to update the state's fields from inside of any node.
    ]

@tool(args_schema=WebSearchInput)
def web_search(
    optimized_query: str,
    max_results: int,
    current_docs: Annotated[list, InjectedState("retrieved_docs")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> list:
    """Search the web for current or supplementary information using Tavily."""
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    results = client.search(optimized_query, max_results=max_results)
    if not results.get("results"):
        return [ToolMessage(content="No web results found.", tool_call_id=tool_call_id)]
    web_docs = [
        Document(
            page_content=r["content"],
            metadata={"url": r["url"], "title": r.get("title", "Web Result")},
        )
        for r in results["results"]
    ]
    summary = f"Found {len(web_docs)} web result(s) for: {optimized_query}"
    return [
        ToolMessage(content=summary, tool_call_id=tool_call_id),   # response of the tool sent back to the agent
        Command(update={"retrieved_docs": (current_docs or []) + web_docs}),
    ]

# RETRIVAL AGENT SINGLETONS
RETRIEVAL_TOOLS = [retrieve_from_vectorstore, web_search]
retrieval_llm = llm.bind_tools(RETRIEVAL_TOOLS)
base_tool_node = ToolNode(RETRIEVAL_TOOLS)

RETRIEVE_SYSTEM = (
    "You are a research assistant gathering context to answer a user's question about research papers.\n\n"
    "You have two tools available and full control over how you use them:\n\n"
    "1. retrieve_from_vectorstore — searches the uploaded paper collection.\n"
    "   You decide:\n"
    "   - query: the semantic search query (phrase it to best match relevant paper chunks)\n"
    "   - k: how many chunks to retrieve (1-10; use more for broad questions, fewer for specific ones)\n\n"
    "2. web_search — searches the live web via Tavily.\n"
    "   You decide:\n"
    "   - optimized_query: rewrite the user's question as a concise, keyword-rich web search query\n"
    "   - max_results: how many results to fetch (1-10)\n\n"
    "Choose the right source based on the question:\n"
    "- Questions about the uploaded papers → use retrieve_from_vectorstore\n"
    "- Questions about current events, recent developments, or supplementary information → use web_search\n"
    "- Call only one tool per turn.\n\n"
    "Do NOT produce a final answer. Only call tools to collect context."
    "IMPORTANT: Do NOT ask clarifying questions. Do NOT produce any text response. "
    "Only call tools. If you have enough context, stop calling tools and return nothing.\n"
)

# RELEVANCY_CHECK

RELEVANCY_CHECK_SYSTEM = (   # checking the relevancy for each individual chunk increases latency and cost. So relevancy check will be done on the whole context rather than each chunk individually
    "You are evaluating whether retrieved document chunks are relevant enough "
    "to answer a user's question about research papers.\n\n"
    "Return is_relevant=true if the chunks contain information that meaningfully "
    "addresses the question — even partially. "
    "Return is_relevant=false only if the chunks are clearly off-topic or contain "
    "no useful information.\n\nBe lenient: if there is any substantive overlap, return true."
)

relevancy_llm = llm.with_structured_output(RelevancyDecision)

QUERY_REWRITE_SYSTEM = (
    "You are a query rewriting assistant for a research paper retrieval system. "
    "The previous query failed to retrieve relevant document chunks. "
    "Rewrite the query using more specific or alternative terminology, "
    "domain-specific keywords, or a narrower sub-question.\n\n"
    "Return ONLY the rewritten query as plain text. No explanation, no preamble."
)

# AGENT NODE
MAX_RETRIEVAL_ATTEMPTS = 3
def agent_node(state: RAGState) -> dict:   # Node function that receives the current graph state and returns state updates in the form of a dictionary
    current_attempts = state.get("retrieval_attempts", 0)
    
    lm = llm if current_attempts >= MAX_RETRIEVAL_ATTEMPTS else retrieval_llm
    # Once at the cap, use plain LLM so the agent cannot emit more tool calls.
    # This prevents orphaned tool_call IDs from entering the persisted message history.
    # retrieval llm --> tool call --> tool result
    # llm --> no tools are bounded --> tool call

    messages = state["messages"]
    clean_messages = []
    tool_call_ids_with_responses = {
        msg.tool_call_id
        for msg in messages
        if hasattr(msg, "tool_call_id")  # ToolMessages
    }
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            # Only keep if all tool_calls have responses
            if all(tc["id"] in tool_call_ids_with_responses for tc in msg.tool_calls):
                clean_messages.append(msg)
            # else: drop the orphaned AIMessage with tool_calls
        else:
            clean_messages.append(msg)

    recent_messages = clean_messages[-6:]  # also apply the history cap
    messages_to_send = [{"role": "system", "content": RETRIEVE_SYSTEM}] + recent_messages

    response = lm.invoke(messages_to_send)

    updates: dict = {"messages": [response]}
    if getattr(response, "tool_calls", None):
        updates["retrieval_attempts"] = current_attempts + 1
    return updates

def relevancy_check_node(state: RAGState) -> dict:   # check if the retrieved_docs are relevant to answer the given query
    query = state["query"]
    docs = state.get("retrieved_docs") or []
    doc_snippets = "\n\n".join(doc.page_content[:300] for doc in docs[:3])
    if not doc_snippets:
        return {"is_relevant": False}
    prompt = (
        f"Question: {query}\n\nRetrieved chunks:\n{doc_snippets}\n\n"
        "Are these chunks relevant to answering the question?"
    )
    decision: RelevancyDecision = relevancy_llm.invoke([
        {
            "role": "system",
            "content": RELEVANCY_CHECK_SYSTEM
        },
        {
            "role": "user",
            "content": prompt
        }
    ])

    return {"is_relevant": decision.is_relevant}

def query_rewrite_node(state: RAGState) -> dict:
    original_query = state["query"]
    rewrite_count = state.get("rewrite_count", 0)
    response = llm.invoke([
        {"role": "system", "content": QUERY_REWRITE_SYSTEM},
        {"role": "user", "content": f"Original query: {original_query}\n\nWrite an improved search query."},
    ])
    rewritten = response.content.strip()
    return {
        "messages": [HumanMessage(content=rewritten)],
        "query": rewritten,
        "retrieved_docs": [],
        "retrieval_attempts": 0,
        "rewrite_count": rewrite_count + 1,
        "is_relevant": None,
    }

CLAIM_ANALYSIS_PROMPT = (
    "You are a research fact-checker. Given a claim from a research paper and "
    "a set of recent web and arXiv search results, determine:\n"
    "1. Has this claim been superseded, significantly challenged, or updated by more recent work?\n"
    "2. Identify up to 3 papers from the provided results that supersede or update the claim.\n\n"
    "Rules:\n"
    "- Use ONLY titles and URLs that appear verbatim in the provided search results.\n"
    "- Prefer arXiv paper links (arxiv.org) over general web links when available.\n"
    "- For each superseding paper, write one sentence explaining how it supersedes the claim.\n"
    "- If the claim still holds, set is_superseded=false and return an empty superseding_papers list.\n"
    "- verdict_summary should be 1-2 sentences suitable for display to the user."
)

verification_llm = ChatOpenAI(
    model="gpt-5-mini",
    max_tokens=8000,
).with_structured_output(ClaimVerificationResult)

def verify_claim_node(state: RAGState) -> dict:
    claim = state["query"]

    try:
        tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

        # General web search for recent work superseding the claim
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:

            general_future = pool.submit(
                tavily_client.search,
                f"recent research superseding {claim[:200]}",
                max_results=5,
            )

            arxiv_future = pool.submit(
                tavily_client.search,
                f"site:arxiv.org {claim[:200]}",
                max_results=5,
            )

            general_results = (
                general_future.result()
                .get("results", [])
            )

            arxiv_results = (
                arxiv_future.result()
                .get("results", [])
            )

    except Exception:
        # Web search service down — fail gracefully with a clear,
        # non-technical message instead of crashing the graph run.
        return {
            "claim_verdict": (
                "Unable to verify this claim right now because the web "
                "search service is temporarily unavailable. Please try again "
                "shortly."
            ),
            "claim_source": None,
            "superseding_papers": [],
        }

    # Build context block
    lines = ["=== General Web Search Results ==="]   # empty list
    for r in general_results:
        lines.append(
            f"Title: {r.get('title', '')}\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r.get('content', '')[:300]}\n"
        )

    lines.append("=== arXiv Paper Search Results ===")
    for r in arxiv_results:
        lines.append(
            f"Title: {r.get('title', '')}\n"
            f"URL: {r['url']}\n"   # URL must exist if not "KeyError" will hpn
            f"Snippet: {r.get('content', '')[:300]}\n"   # Get content. If missing, use empty string. Take first 300 characters.
        )

    context = "\n".join(lines)[:3000]

    prompt = (
        f"{CLAIM_ANALYSIS_PROMPT}\n\n"
        f"Claim to verify:\n{claim}\n\n"
        f"Search Results:\n{context}"
    )
    
    result: ClaimVerificationResult = verification_llm.invoke([
        {"role": "user", "content": prompt}
    ])

    papers_dicts = []   # list of dictionaries containing metadata of papers (refer chatgpt)

    for p in result.superseding_papers[:3]:
        papers_dicts.append(p.model_dump())   # model_dump - converts a Pydantic model object into a normal Python dictionary 
    
    return {
        "claim_verdict": result.verdict_summary,
        "claim_source": papers_dicts[0]["url"] if papers_dicts else None,
        "superseding_papers": papers_dicts,   # stores the entire papers_docts list into the graph state.
    }   # returns state's updates

def generate_answer_node(state: RAGState) -> dict:
    if state.get("planner_action") == "discover_papers":
        papers = state.get("discovered_papers") or []
        answer = "I found these relevant papers:\n\n"

        for i, paper in enumerate(papers, start=1):
            answer += (
                 f"{i}. {paper['title']}\n"
                f"Published: {paper['published']}\n"
                f"{paper['summary'][:250]}...\n"
                f"{paper['pdf_url']}\n\n"
            )

        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)]
        }
    
    route = state.get("route")
    query = state["query"]

    simple_queries = {
            "hi",
            "hello",
            "hey",
            "good morning",
            "good evening",
            "good afternoon",
            "how are you",
        }

    if query.strip().lower() in simple_queries:
        return {
            "answer": "Hello! How can I help you with your papers today?",
            "messages": [AIMessage(content="Hello! How can I help you with your papers today?")],
        }

    if route == "retrieve":
        if state.get("is_relevant") is False and state.get("rewrite_count", 0) >= 2:
            answer = (
                "I wasn't able to find relevant information in the uploaded papers "
                "to answer your question. You may want to rephrase your question "
                "or upload additional papers."
            )

        else:
            docs = state.get("retrieved_docs") or []

            if not docs:
                response = llm.invoke([{"role": "user", "content": f"Answer concisely from your general knowledge: {query}"}])
                answer = validate_generation_output(response.content)

            else:
                # RETRIEVAL GUARDRAIL: cap total context size before sending
                # to the LLM — bounds input token cost and avoids
                # context-window overflow / "lost in the middle" issues.
                context, citations = build_context_with_citations(docs)
                prompt = f""" You are a research assistant. Answer ONLY using the supplied context. Rules:
                1. Do not invent facts.
                2. If information is missing, explicitly say so.
                3. Prefer evidence from the context over assumptions.
                4. For conflicting information, mention the conflict.
                5. Keep the answer concise but complete — aim for under 250 words.
                6. Mention limitations when appropriate.
                7. Each context chunk is tagged with a bracketed number, e.g. [1]. Cite it inline when used.
                Context: {context}
                Question: {query}
                Answer: """

                try:
                    response = llm.invoke([{"role": "user", "content": prompt}])
                    answer = validate_generation_output(response.content)
                    answer += format_citations_block(citations)
                except Exception as e:
                    answer = f"GENERATION ERROR: {str(e)}"

    elif route == "verify_claim":
        verdict = state.get("claim_verdict", "")
        papers = state.get("superseding_papers") or []
        claim_text = state["query"]

        if papers:
            papers_block = "\n\n".join(
                f"{i + 1}. **{p['title']}**\n"
                f"   {p['summary']}\n"
                f"   Link: {p['url']}"
                for i, p in enumerate(papers)
            )

            answer = (
                f"**Claim Verification Result**\n\n"
                f"> {claim_text}\n\n"
                f"**Verdict:** {verdict}\n\n"
                f"**Superseding Papers:**\n\n"
                f"{papers_block}\n\n"
                f"*You can load any of these papers into your knowledge base "
                f"to continue your research with the latest findings.*"
            )

        else:
            answer = (
                f"**Claim Verification Result**\n\n"
                f"> {claim_text}\n\n"
                f"**Verdict:** {verdict}\n\n"
                f"*No papers directly superseding this claim were found "
                f"in recent literature.*"
            )
        
    else:
        prompt = f"""
        Answer from your parametric knowledge.
        Rules:
        1. Be accurate.
        2. If unsure, state uncertainty.
        3. Avoid speculation.
        4. Keep the answer concise.
        Question: {query}"""

        try:
            response = llm.invoke([{"role": "user", "content": prompt}])
            answer = validate_generation_output(response.content)
        except Exception as e:
            answer = f"GENERATION ERROR: {str(e)}"

    return {
        "answer": answer,   # stores final answer in state["answer"]
        "messages": [AIMessage(content=answer)]  # adds the answer to the chat history so future nodes/conversations can see it.
        }

# GRAPH - HOW IS THE ROUTING DONE AND ALL THE CONDITIONAL EDGES
def agent_routing(state: RAGState) -> str:
    tc = tools_condition(state)  # tools_condition - prebuilt LangGraph routing function that checks whether the last AI message contains a tool call.
    if tc == "tools":
        return "retrieval"
    if state.get("retrieval_attempts", 0) >= MAX_RETRIEVAL_ATTEMPTS:
        return "generate_answer"
    return "relevancy_check"

def after_relevancy_routing(state: RAGState) -> str:
    if state.get("is_relevant", False):  # give me state["is_relevant"] and if that key doesn't exist, use False instead.
        return "generate_answer"
    if state.get("is_relevant") is False and state.get("rewrite_count", 0) < 2:  # Docs not relevant, rewrites still available
        return "query_rewrite"
    return "generate_answer"  # Docs not relevant, rewrite limit reached
    # If retrieval worked → answer; if retrieval failed and rewrites remain → rewrite query; otherwise stop and give the best possible response.

def planner_route(state):
    return state["planner_action"]

# planner node
planner_chain = get_planner_chain(llm)

def planner_node(state):
    query = state["query"]
    decision = planner_chain.invoke({"query": query})

    return {
        "planner_action": decision.action,
        "route": decision.action
    }

def paper_discovery_node(state):

    query = state["query"]

    papers = discover_papers(query, max_results=5)

    return {
        "discovered_papers": papers
    }

def build_graph(db_path: str = "checkpoints.db"):   # "checkpoints.db" - SQLite db file which stores the graph
    conn = sqlite3.connect(db_path, check_same_thread = False) # open a connection to a SQLite database file.
    checkpointer = SqliteSaver(conn) # Create a LangGraph checkpointer that stores graph state in that SQLite database.

    graph = StateGraph(RAGState)
    graph.add_node("planner", planner_node)
    graph.add_node("paper_discovery", paper_discovery_node)
    graph.add_node("agent_node", agent_node)
    graph.add_node("retrieval", base_tool_node)
    graph.add_node("relevancy_check", relevancy_check_node)
    graph.add_node("query_rewrite", query_rewrite_node)
    graph.add_node("verify_claim", verify_claim_node)
    graph.add_node("generate_answer", generate_answer_node)

    graph.set_entry_point("planner")
    graph.add_conditional_edges(
        "planner",
        planner_route,
        {
            "retrieve": "agent_node",
            "verify_claim": "verify_claim",
            "direct_answer": "generate_answer",
            "discover_papers": "paper_discovery",
        },
    )

    graph.add_conditional_edges(
        "agent_node",
        agent_routing,
        {
            "retrieval": "retrieval",
            "relevancy_check": "relevancy_check",
            "generate_answer": "generate_answer",
        },
    )

    graph.add_edge("retrieval", "agent_node")

    graph.add_conditional_edges(
        "relevancy_check",
        after_relevancy_routing,
        {"query_rewrite": "query_rewrite", "generate_answer": "generate_answer"},
    )
    graph.add_edge("query_rewrite", "agent_node")

    graph.add_edge("paper_discovery", "generate_answer")

    graph.add_edge("verify_claim", "generate_answer")

    graph.add_edge("generate_answer", END)

    return graph.compile(checkpointer=checkpointer)