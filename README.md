# 📚 PaperSage — Agentic RAG Research Paper Assistant

> Drop a paper. Ask anything. Let the agent do the rest.

PaperSage is a research-focused, agentic RAG system built for researchers and engineers who need deep, reliable insights from research papers; not surface-level summaries.

---

## THE PROBLEM

Generic chatbots fail at research paper Q&A. They hallucinate answers, provide no source tracing, can't recover when retrieval fails, and have no awareness of whether a claim has already been superseded by newer literature. Most RAG tools use naive top-k vector search, which means one missed chunk silently produces a wrong answer.

**PaperSage addresses this end-to-end:** answers are grounded and cited, retrieval is hybrid with automatic query rewriting on failure, claims are verified against live arXiv literature, and off-topic questions are isolated in a separate side channel that never contaminates the knowledge base.

---

## KEY FEATURES

- **Multi-format document ingestion**: load papers from local PDF, TXT, Markdown, DOCX, web URLs, or directly by arXiv ID or title; SHA-256 deduplication silently skips re-uploads.
- **Agentic planner**: a GPT-5-mini planner routes every query to one of four actions (`retrieve`, `discover_papers`, `verify_claim`, `direct_answer`) before any retrieval cost is incurred.
- **Hybrid retrieval with cross-encoder reranking**: dense vector search (Qdrant) + BM25 re-scoring + `BAAI/bge-reranker-large` cross-encoder, warmed up at startup to eliminate first-query latency.
- **Relevancy check + automatic query rewriting**: after retrieval, a judge LLM checks chunk relevance; on failure the query is rewritten and retried up to 2 times before falling back gracefully.
- **Inline citations**: every grounded answer includes bracketed `[N]` inline citations mapped to the exact source paper and page number, with a formatted sources block appended.
- **Claim verification**: checks whether a specific finding is still current by searching arXiv and the live web, returning a verdict and links to superseding papers that can be loaded directly into the session.
- **Paper discovery**: queries the arXiv API for a topic, fetches a relevance pool, and re-sorts by recency so the latest work in fast-moving fields always surfaces first.
- **`/btw` side channel**: off-topic questions prefixed with `/btw` bypass the RAG graph entirely, auto-routing to web search or general knowledge without polluting session history.
- **Multi-session management**: each session gets an isolated Qdrant collection and LangGraph SQLite checkpoint thread; sessions are auto-named by Gemini 2.5 Flash Lite from the first message and persist across restarts.
- **Three-layer guardrails**: input validation (empty / length), retrieval cap (5 chunks, 8,000-char context), and generation cap (1,024 output tokens) applied at every stage with zero extra LLM calls.
- **DeepEval evaluation harness**: `evaluate.py` runs the full pipeline against synthesized golden QA pairs, scoring Contextual Precision, Recall, Relevancy, Answer Relevancy, Faithfulness, and a custom MRR implementation.
- **Session analytics dashboard**: sidebar displays per-session query count, routing breakdown, and (when LangSmith is configured) average latency and total token usage.

---

## HOW IT WORKS

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    INPUT GUARDRAILS                     │
│         (length check, empty check, safety)             │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │    PLANNER AGENT      │  ← Routes to one of 4 actions
              │  (GPT-5-mini + tools) │
              └───┬───┬───┬───┬───────┘
                  │   │   │   │
         ┌────────┘   │   │   └────────────────┐
         ▼            │   ▼                    ▼
  ┌─────────────┐     │  ┌──────────────┐  ┌──────────────────┐
  │ AGENT NODE  │     │  │ PAPER        │  │  VERIFY CLAIM    │
  │ (Retrieval  │     │  │ DISCOVERY    │  │  (arXiv search   │
  │  Agent)     │     │  │ (arXiv API)  │  │  + web search)   │
  └──────┬──────┘     │  └──────────────┘  └──────────────────┘
         │            │ (direct_answer)
    ┌────┴────┐        └──────────────┐
    │         │                       ▼
    ▼         ▼               ┌──────────────┐
┌────────┐ ┌──────────────┐   │   GENERATE   │
│RETRIEV-│ │  RELEVANCY   │   │   ANSWER     │ ← Final output
│AL NODE │ │  CHECK NODE  │   │  (citations  │   with inline
│(Hybrid │ │ (GPT-5-mini) │   │   + sources) │   citations
│ Search)│ └──────┬───────┘   └──────────────┘
└────────┘        │
                  │ Irrelevant?
                  ▼
          ┌───────────────┐
          │ QUERY REWRITE │ → back to AGENT NODE (max 2 retries)
          └───────────────┘
```

### HYBRID RETRIEVAL PIPELINE (INSIDE AGENT NODE)

```
Query
  │
  ├─── Vector Search (Qdrant, text-embedding-3-small, top-15)
  │
  ├─── BM25 Re-rank (rank-bm25, top-10 from vector results)
  │
  ├─── Deduplication (by content fingerprint)
  │
  └─── Cross-Encoder Rerank (BAAI/bge-reranker-large, top-5)
              │
              └─── Context truncation (guardrail: 8,000 chars max)
                          │
                          └─── LLM generation with inline [N] citations
```

## PROJECT STRUCTURE

```
PAPERSAGE/
│
├── app.py                      # Streamlit entrypoint - UI, session management, chat loop
├── login_wall.py               # Password-gated login screen
│
├── backend/
│   ├── __init__.py
│   ├── rag_graph.py            # LangGraph graph definition - all nodes, edges, state
│   ├── planner_agent.py        # Planner agent - routing prompt + structured output chain
│   ├── hybrid_retriever.py     # Vector + BM25 + cross-encoder reranking pipeline
│   ├── vector_store.py         # Qdrant client, collection management, add/search/dedup
│   ├── reranker.py             # BAAI/bge-reranker-large cross-encoder (warmed up at init)
│   ├── paper_loader.py         # Loaders for PDF, TXT, Markdown, DOCX, URL, arXiv
│   ├── paper_discovery.py      # arXiv API search + recency re-sort
│   ├── btw_handler.py          # Off-topic side channel with Tavily web search fallback
│   ├── guardrails.py           # Input / retrieval / generation guardrails (single source of truth)
│   ├── models.py               # Pydantic models for all structured LLM outputs
│   └── dashboard.py            # Session analytics - local stats + LangSmith integration
│
├── evaluate.py                 # Full RAG evaluation harness (DeepEval + MRR)
├── deepeval_gpt.py             # DeepEval-compatible GPT-5-mini judge wrapper
├── goldens.json                # Synthesized golden QA pairs for evaluation
├── eval_results.json           # Last evaluation run results
│
├── .streamlit/
│   ├── config.toml             # Streamlit theme and server config
│   └── secrets.toml            # App password and API keys (not committed)
│
├── documents/                  # Evaluation PDFs (not committed)
├── embedding_cache/            # Disk-backed embedding cache (auto-created)
├── checkpoints.db              # LangGraph SQLite checkpoint store (app sessions)
├── eval_checkpoints.db         # LangGraph SQLite checkpoint store (eval runs)
├── sessions.json               # Session metadata index
│
├── Dockerfile                  # Docker image definition
├── pyproject.toml              # Project metadata and dependency spec (uv)
├── requirements.txt            # Pinned dependency lockfile
├── rag_graph.png               # Visual graph diagram
└── visualize_graph.py          # Script to render and export the LangGraph diagram
```

---

## TECH STACK

| Layer | Technology |
|---|---|
| LLM | GPT-5-mini (OpenAI) |
| Session naming | Gemini 2.5 Flash Lite (Google) |
| Orchestration | LangGraph (StateGraph with SQLite checkpointing) |
| Vector store | Qdrant Cloud |
| Embeddings | `text-embedding-3-small` (OpenAI), cached via `CacheBackedEmbeddings` |
| Sparse retrieval | BM25 (`rank-bm25`) |
| Reranker | `BAAI/bge-reranker-large` (cross-encoder, `sentence-transformers`) |
| Web search | Tavily |
| arXiv | `arxiv` Python client |
| PDF parsing | PyMuPDF |
| UI | Streamlit |
| Evaluation | DeepEval |
| Observability | LangSmith |
| Containerization | Docker |

---

## SETUP AND INSTALLATION

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- [Qdrant Cloud](https://qdrant.tech) account (free tier works)
- OpenAI API key
- Tavily API key
- Google API key (Gemini, for session auto-naming)
- LangSmith API key (optional for tracing)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/papersage.git
cd papersage
```

### 2. Create a virtual environment

Using `uv` (recommended):

```bash
uv venv --python 3.12
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux
uv pip install -r requirements.txt
```

Or with standard `venv`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
GOOGLE_API_KEY=...
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=...

# Optional — enables LangSmith analytics in the dashboard
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=papersage
```

Create `.streamlit/secrets.toml` for the login wall:

```toml
APP_PASSWORD = "your-secure-password"
```

### 4. Run the app

```bash
streamlit run app.py
```

Open the local URL Streamlit prints in your terminal, load a paper from the sidebar, and start asking questions.

---

## DATABASE SCHEMA

Two SQLite databases are created automatically on first launch.

**`checkpoints.db`** — LangGraph conversation state (one thread per session):
- Stores the full `RAGState` snapshot at every graph step, enabling turn-by-turn history and resumable sessions.

**`eval_checkpoints.db`** — isolated checkpoint store used exclusively by the evaluation harness so eval runs never touch live session state.

**`sessions.json`** — lightweight session index (not a database):
- `id` · `name` · `created_at` · `is_named`
- Powers the session switcher in the sidebar and persists auto-generated session names across restarts.

---

## FUTURE ENHANCEMENTS

- [ ] Adaptive query routing that learns from past retrieval failures within a session
- [ ] Exercise auto-detection of paper type (survey vs. empirical vs. theoretical) to tailor answer style
- [ ] Multi-paper cross-referencing to surface contradictions and consensus across a loaded corpus
- [ ] Personalized session memory that carries user-defined reading goals across sessions
- [ ] Export grounded answers with full citations as a formatted PDF or Markdown report