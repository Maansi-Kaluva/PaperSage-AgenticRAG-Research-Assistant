from login_wall import require_login
require_login()
import json
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.btw_handler import handle_btw
from backend.paper_loader import load_arxiv, load_document, load_webpage
from backend.rag_graph import build_graph
from backend.vector_store import add_paper_if_new, list_papers
from backend.guardrails import validate_input_query
from backend.dashboard import get_local_session_stats, get_langsmith_session_stats

st.set_page_config(
    page_title="Papeer — Research Paper Assistant",
    page_icon="📄",
    layout="wide",
)

st.markdown("""
<style>
/* Typography */
html, body, [class*="css"] {
    font-family: "Inter", -apple-system, "Segoe UI", sans-serif;
    color: #ffffff !important;
}

p, li, span, div, label, .stMarkdown, .stText {
    color: #ffffff !important;
}

.block-container {
    padding-top: 1rem;
    max-width: 100%;
}

h1, h2, h3 {
    font-weight: 600;
    letter-spacing: -0.01em;
}

/* Buttons */
.stButton > button {
    border-radius: 6px;
    border: 1px solid #1a3a3a;
    background-color: #0d1f1f;
    color: #b2dfdb;
    font-weight: 500;
    transition: all 0.15s ease;
    box-shadow: none;
}
.stButton > button:hover {
    border-color: #00897b;
    color: #e0f2f1;
    background-color: #0d1f1f;
}
.stButton > button[kind="primary"] {
    background-color: #00695c;
    border: 1px solid #00897b;
    color: #e0f2f1;
}
.stButton > button[kind="primary"]:hover {
    background-color: #00796b;
    border-color: #4db6ac;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0a1a1a;
    border-right: 1px solid #1a3a3a;
    width: 380px !important;
    min-width: 380px !important;
}

/* Chat bubbles */
[data-testid="stChatMessage"] {
    border-radius: 8px;
    border: 1px solid #1a3a3a;
}

/* User bubble */
[data-testid="stChatMessage"]:has(
[data-testid="stChatMessageAvatarUser"]
) {
    background-color: #0d2020;
    border-left: 3px solid #00897b;
}

/* Assistant bubble */
[data-testid="stChatMessage"]:has(
[data-testid="stChatMessageAvatarAssistant"]
) {
    background-color: #0a1515;
    border-left: 3px solid #00695c;
}

/* Progress bar */
.stProgress > div > div > div > div {
    background-color: #00897b;
}
.stProgress > div > div {
    background-color: #1a3a3a;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #0d2020;
    border: 1px solid #1a3a3a;
    border-radius: 6px;
    padding: 0.6rem 0.8rem;
}
[data-testid="stMetricLabel"] {
    color: #4db6ac;
}
[data-testid="stMetricValue"] {
    color: #b2dfdb;
}

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid #1a3a3a;
    border-radius: 6px;
    background-color: #0a1515;
}

/* Dividers */
hr {
    border-color: #1a3a3a;
}

/* Captions */
.stCaption, [data-testid="stCaptionContainer"] {
    color: #4db6ac;
}

/* Text inputs / text areas / file uploader */
.stTextInput > div > div, .stTextArea > div > div, [data-testid="stFileUploaderDropzone"] {
    border-radius: 6px;
    border: 1px solid #1a3a3a;
    background-color: #0d2020;
    color: #b2dfdb;
}

/* Selectbox */
.stSelectbox > div > div {
    border: 1px solid #1a3a3a;
    background-color: #0d2020;
    border-radius: 6px;
    color: #b2dfdb;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-track {
    background: #0a1515;
}
::-webkit-scrollbar-thumb {
    background: #00695c;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #00897b;
}

div[data-testid="stStatusWidget"] {
    visibility: visible;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_graph():
    return build_graph()  # prevents multiple runs when changes are made to the UI when user interacts with the UI. Always live in the interface

SESSIONS_FILE = Path("sessions.json")

_rename_llm = ChatGoogleGenerativeAI(    # helps in renaming of sessions
    model="gemini-2.5-flash-lite"
    )

def load_sessions() -> dict:
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    
def save_sessions(sessions_meta: dict) -> None:   # saves sessins in the format mentioned in the json file
    SESSIONS_FILE.write_text(json.dumps(sessions_meta, indent=2), encoding="utf-8")

def _serialize_state(values: dict) -> dict:
    out = {}
    for k, v in values.items():
        if k == "messages":
            out[k] = [
                {
                    "type": type(m).__name__,
                    "content": (
                        m.content[:300]
                        if isinstance(m.content, str)
                        else repr(m.content)[:300]
                    ),
                }
                for m in (v or [])
            ]
        elif k == "retrieved_docs":
            out[k] = [
                {"content": d.page_content[:300], "metadata": d.metadata}
                for d in (v or [])
            ]
        else:
            out[k] = v
    return out

def generate_session_name(first_message: str) -> str:   # generates a session name using LLM - name decided basd in the chat's first query
    try:
        response = _rename_llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Generate a concise 3-5 word title for a research chat session "
                        "based on the user's first message. Return only the title, "
                        "no punctuation at the end, no quotes."
                    ),
                },
                {"role": "user", "content": first_message[:500]},
            ]
        )
        return response.content.strip()
    except Exception:
        return "New Session"   # if the title isnt getting generated, name the session "New Session"
    
def maybe_rename_session(session_id: str, first_message: str) -> None:   # checks if the session already has a name or not
    if st.session_state.sessions_meta.get(session_id, {}).get("is_named"):
        return
    name = generate_session_name(first_message)
    st.session_state.sessions_meta[session_id]["name"] = name
    st.session_state.sessions_meta[session_id]["is_named"] = True
    save_sessions(st.session_state.sessions_meta)
# we might run a converstion more than once which can lead to renaming of the session again and again. To prevent that we write the above piece of code.

def create_session() -> str:   # creates a session suign a unique session id for every session
    sid = str(uuid.uuid4())
    st.session_state.sessions_meta[sid] = {   # store streamlit session's metadata
        "id": sid,
        "name": "New Session",  # initially, after the chat started, it'll be renames accordingly
        "created_at": datetime.now().isoformat(),
        "is_named": False,
    }
    save_sessions(st.session_state.sessions_meta)   # saved to json
    st.session_state.chats[sid] = []   # saves the streamlit's messages state - new session so initiaally the list is empty
    st.session_state.turns[sid] = 0  # 1 turn = 1 ip + 1 op. Stores the number of turns per conversation/session/chat
    return sid

def load_session_chats(session_id: str) -> list[dict]:
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = graph.get_state(config)
        state_values = state.values
        if not state or not state.values:
            return []
        chats = []
        turn = 0
        messages = state_values.get("messages", [])
        
        i = 0
        while i < len(messages):
            msg = messages[i]
            type_name = type(msg).__name__
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            
            if type_name == "HumanMessage":
                chats.append({"role": "user", "content": content})
                # find the LAST non-empty AIMessage that follows this HumanMessage
                # before the next HumanMessage
                j = i + 1
                last_ai_content = None
                while j < len(messages):
                    next_type = type(messages[j]).__name__
                    if next_type == "HumanMessage":
                        break
                    if next_type == "AIMessage":
                        c = messages[j].content if isinstance(messages[j].content, str) else str(messages[j].content)
                        if c.strip():
                            last_ai_content = c
                    j += 1
                if last_ai_content:
                    turn += 1
                    chats.append({
                        "role": "assistant",
                        "content": last_ai_content,
                        "turn": turn,
                        "graph_state": _serialize_state(state_values),
                        "route": state_values.get("route"),
                    })
                i = j  # jump to next HumanMessage
            else:
                i += 1
        return chats
    except Exception:
        return []
    
def switch_session(session_id: str) -> None:    # if we switch sessions in between, st is going to load the opened session's session_id and chats
    st.session_state.active_session_id = session_id
    if session_id not in st.session_state.chats:  # loading session chats
        st.session_state.chats[session_id] = load_session_chats(session_id)
    if session_id not in st.session_state.turns:
        turn_count = sum(1 for m in st.session_state.chats[session_id] if m["role"] == "assistant")
        st.session_state.turns[session_id] = turn_count  # saving and displaying turn counts

graph = get_graph()

# BOOTSTRAP
# managing st's session states
if "sessions_meta" not in st.session_state:
    st.session_state.sessions_meta = load_sessions()   # storing metadata 
if "chats" not in st.session_state:   # storing session's chats
    st.session_state.chats = {}
if "turns" not in st.session_state:  # storing turns per chat
    st.session_state.turns = {}
if "active_session_id" not in st.session_state:  # store active session id
    if st.session_state.sessions_meta:
        latest = max(
            st.session_state.sessions_meta.values(),
            key=lambda s: s["created_at"],
        )
        switch_session(latest["id"])
    else:
        sid = create_session()
        st.session_state.active_session_id = sid

active_sid = st.session_state.active_session_id   # loading the active session state from streamlit

# SIDEBAR
with st.sidebar:
    if st.button("+ New Chat", use_container_width=True):
        new_sid = create_session()  # click the new session button - a sessionid for that is created
        st.session_state.active_session_id = new_sid  # new sid added to the sessions json file - session is rerun again
        active_sid = new_sid
        st.rerun()
    st.divider()
    st.markdown("## Sessions")

    sorted_sessions = sorted(  # all the sessions listed are sorted - latest ones are shown on top
        st.session_state.sessions_meta.values(),
        key=lambda s: s["created_at"],
        reverse=True,
    )
    for session in sorted_sessions:
        sid = session["id"]
        is_active = sid == st.session_state.active_session_id
        btn_type = "primary" if is_active else "secondary"
        if st.button(
            session["name"],
            key=f"sess_{sid}",
            use_container_width=True,
            type=btn_type,
        ):
            if not is_active:
                switch_session(sid)
                st.rerun()

    st.divider()
    st.markdown("## Documents") 

    # FILE UPLOAD
    st.markdown("**Upload Files**")
    uploaded_files = st.file_uploader(
            "PDF, TXT, or Markdown",
            type=["pdf", "txt", "md", "markdown", "docx"],
            accept_multiple_files=True,
            key=f"uploader_{active_sid}",
            label_visibility="collapsed",
        )
    if st.button("Add Files", use_container_width=True, key="btn_add_files"):
            if uploaded_files:
                processed_key = f"processed_files_{active_sid}"
                if processed_key not in st.session_state:
                    st.session_state[processed_key] = set()
                with st.spinner("Processing files…"):
                    for f in uploaded_files:
                        file_id = f"{f.name}_{f.size}"
                        if file_id in st.session_state[processed_key]:
                            st.info(f"Already loaded: {f.name}")
                            continue
                        suffix = Path(f.name).suffix
                        tmp_path = None
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                                tmp.write(f.read())
                                tmp_path = tmp.name
                            docs = load_document(tmp_path)
                            for doc in docs:
                                doc.metadata["title"] = Path(f.name).stem
                            was_added = add_paper_if_new(
                                docs,
                                active_sid,
                            )
                            if was_added:
                                st.success(f"Added: {f.name}")
                            else:
                                st.info(
                                    f"Skipped duplicate (already indexed): {file_id}"
                                )

                            st.session_state[processed_key].add(file_id)
                        except Exception as e:
                            st.error(f"Failed: {f.name} — {e}")
                        finally:
                            if tmp_path:
                                Path(tmp_path).unlink(missing_ok=True)
                st.rerun()
            else:
                st.warning("No files selected.")

    st.markdown("**Web Pages**")
    url_input = st.text_area(
        "URLs (one per line)",
        key=f"url_area_{active_sid}",
        height=80,
        label_visibility="collapsed",
        placeholder="https://example.com/paper",
    )
    if st.button("Load URLs", use_container_width=True, key="btn_load_urls"):
        urls = [u.strip() for u in url_input.splitlines() if u.strip()]
        if urls:
            with st.spinner("Loading web pages…"):
                for url in urls:
                    try:
                        docs = load_webpage(url)
                        was_added = add_paper_if_new(
                            docs,
                            active_sid,
                        )
                        if was_added:
                            st.success(f"Loaded: {url[:60]}")
                        else:
                            st.info(f"URL already indexed or produced no new content: {url[:60]}")
                    except Exception as e:
                        st.error(f"Failed: {url[:60]} — {e}")
                st.rerun()
        else:
            st.warning("Enter at least one URL.")

    # ARXIV LOADER
    st.markdown("**ArXiv Papers**")
    arxiv_title = st.text_input(
        "Paper title or ArXiv ID",
        key=f"arxiv_input_{active_sid}",
        label_visibility="collapsed",
        placeholder="1706.03762  or  Attention Is All You Need",
    )
    if st.button("Load ArXiv Paper", use_container_width=True, key="btn_load_arxiv"):
        if arxiv_title.strip():
            with st.spinner("Loading from ArXiv…"):
                try:
                    docs = load_arxiv(arxiv_title.strip())
                    was_added = add_paper_if_new(
                        docs,
                        active_sid,
                    )
                    loaded_title = docs[0].metadata.get("title") if docs else arxiv_title.strip()
                    if was_added:
                        st.success(f"Loaded: {loaded_title}")
                    else:
                        st.info(
                            f"Skipped duplicate: {loaded_title}"
                        )
                except Exception as e:
                    st.error(f"Failed: {e}")
            st.rerun()
        else:
            st.warning("Enter a paper title or ArXiv ID.")

    # LOADED DOCUMENTS LIST
    st.divider()

    st.markdown("### Loaded Documents")
    try:
        doc_titles = sorted(set(list_papers(active_sid)))
    except Exception:
        doc_titles = None
    if doc_titles is None:
        st.caption("Could not load document list — try refreshing.")
    elif doc_titles:
        for title in doc_titles:
            st.markdown(f"- {title}")
    else:
        st.caption("No documents loaded yet.")
    
    st.divider()

    filter_options = ["All papers"] + (doc_titles or [])

    selected_filter = st.selectbox(
        "Restrict retrieval to a single paper",
        filter_options,
        key=f"paper_filter_{active_sid}",
    )

    paper_filter = (
        None
        if selected_filter == "All papers"
        else selected_filter
    )

    # SESSION ANALYTICS DASHBOARD
    st.divider()
    st.markdown("### Session Analytics")

    chat_history = st.session_state.chats.get(active_sid, [])
    local_stats = get_local_session_stats(chat_history)

    if local_stats["total_queries"] == 0:
        st.caption("Analytics will appear after your first query.")
    else:
        st.markdown(f"**Queries answered:** {local_stats['total_queries']}")

        # Route breakdown — shows users what the system actually did
        route_counts = local_stats["route_counts"]
        if route_counts:
            st.markdown("**Query routing breakdown**")
            route_labels = {
                "retrieve": "Paper retrieval",
                "discover_papers": "Paper discovery",
                "verify_claim": "Claim verification",
                "direct_answer": "Direct answer",
                "unknown": "Unknown",
            }
            for route, count in route_counts.items():
                label = route_labels.get(route, route)
                st.caption(f"{label}: {count}")

        # LangSmith stats — latency and token usage
        ls_stats = get_langsmith_session_stats(active_sid)
        if ls_stats:
            st.markdown("**Pipeline performance** *(via LangSmith)*")
            if ls_stats["avg_latency_sec"] is not None:
                st.metric("Avg response time", f"{ls_stats['avg_latency_sec']}s")
            if ls_stats["total_tokens"] is not None:
                st.metric("Total tokens used", f"{ls_stats['total_tokens']:,}")

# PAGE HEADER

st.markdown("<div style='height:20vh'></div>", unsafe_allow_html=True)

st.markdown("""
<div style="padding: 2rem 0 1.5rem 0;">
    <div style="display: flex; align-items: baseline; gap: 0.6rem; margin-bottom: 0.5rem;">
        <span style="font-size: 2.8rem;">📚</span>
        <span style="font-size: 2.6rem; font-weight: 750; color: #e0f2f1; letter-spacing: -0.04em; line-height: 1;">PaperSage</span>
    </div>
    <p style="font-size: 1rem; color: #4db6ac; font-weight: 400; margin: 0.4rem 0 1.6rem 0; letter-spacing: 0.015em;">
        Drop a paper. Ask anything. Let the agent do the rest.
    </p>
    <div style="width: 48px; height: 3px; background: linear-gradient(90deg, #00897b, transparent); border-radius: 2px; margin-bottom: 1.4rem;"></div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; max-width: 600px;">
        <div style="background: #0d2020; border: 1px solid #1a3a3a; border-radius: 10px; padding: 0.75rem 1rem;">
            <div style="font-size: 1.1rem; margin-bottom: 0.2rem;">🔎</div>
            <div style="font-size: 0.82rem; color: #e0f2f1; font-weight: 600; margin-bottom: 0.1rem;">Deep Paper Q&A</div>
            <div style="font-size: 0.75rem; color: #4db6ac; line-height: 1.4;">Ask anything about your uploaded research — get grounded, cited answers.</div>
        </div>
        <div style="background: #0d2020; border: 1px solid #1a3a3a; border-radius: 10px; padding: 0.75rem 1rem;">
            <div style="font-size: 1.1rem; margin-bottom: 0.2rem;">✅</div>
            <div style="font-size: 0.82rem; color: #e0f2f1; font-weight: 600; margin-bottom: 0.1rem;">Claim Verification</div>
            <div style="font-size: 0.75rem; color: #4db6ac; line-height: 1.4;">Check if a finding still holds — verified against recent web and arXiv sources.</div>
        </div>
        <div style="background: #0d2020; border: 1px solid #1a3a3a; border-radius: 10px; padding: 0.75rem 1rem;">
            <div style="font-size: 1.1rem; margin-bottom: 0.2rem;">🌐</div>
            <div style="font-size: 0.82rem; color: #e0f2f1; font-weight: 600; margin-bottom: 0.1rem;">Live Web Search</div>
            <div style="font-size: 0.75rem; color: #4db6ac; line-height: 1.4;">Pulls in real-time context from the web when your papers don't have the answer.</div>
        </div>
        <div style="background: #0d2020; border: 1px solid #1a3a3a; border-radius: 10px; padding: 0.75rem 1rem;">
            <div style="font-size: 1.1rem; margin-bottom: 0.2rem;">🗂️</div>
            <div style="font-size: 0.82rem; color: #e0f2f1; font-weight: 600; margin-bottom: 0.1rem;">Paper Discovery</div>
            <div style="font-size: 0.75rem; color: #4db6ac; line-height: 1.4;">Finds and surfaces the most relevant recent papers on any research topic.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()


# CHAT DISPLAY.
for msg in st.session_state.chats.get(active_sid, []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("graph_state"):
                with st.expander(f"Graph state · turn {msg['turn']}", expanded=False):
                    st.json(msg["graph_state"])

# CHAT INPUT
if prompt := st.chat_input("Ask about your papers, verify a claim, or search the web…"):
    is_btw = prompt.strip().lower().startswith("/btw")

    if is_btw:
        query = prompt.strip()[4:].strip()

        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption("Side channel — not saved to session history.")

        with st.chat_message("assistant"):
            if not query:
                st.markdown("Please add a question after `/btw`, e.g. `/btw What is attention?`")
            else:
                # INPUT GUARDRAIL: validate before calling the LLM
                is_valid, message = validate_input_query(query)
                if not is_valid:
                    st.markdown(message)
                else:
                    placeholder = st.empty()
                    response_text = ""
                    for chunk in handle_btw(query):
                        response_text += chunk
                        placeholder.markdown(response_text + "▌")
                    placeholder.markdown(response_text)
            st.caption("Side channel — not saved to session history.")
    else:
        # INPUT GUARDRAIL: validate before entering the graph at all —
        # avoids any retrieval/LLM cost on empty or oversized queries.
        is_valid, message = validate_input_query(prompt)
        if not is_valid:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                st.markdown(message)
        else:
            if active_sid not in st.session_state.chats:
                st.session_state.chats[active_sid] = []
            if active_sid not in st.session_state.turns:
                st.session_state.turns[active_sid] = 0

            is_first_message = len(st.session_state.chats[active_sid]) == 0

            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.chats[active_sid].append({"role": "user", "content": prompt})
            st.session_state.turns[active_sid] += 1
            current_turn = st.session_state.turns[active_sid]

            input_state = {
                "messages": [HumanMessage(content=prompt)],
                "session_id": active_sid,
                "query": prompt,
                "route": None,
                "retrieved_docs": [],
                "retrieval_attempts": 0,
                "claim_verdict": None,
                "claim_source": None,
                "superseding_papers": [],
                "answer": None,
                "is_relevant": None,
                "rewrite_count": 0,
                "avg_rerank_score": None,
                "paper_filter": paper_filter,
            }
            config = {"configurable": {"thread_id": active_sid}}

            with st.chat_message("assistant"):
                placeholder = st.empty()
                response_text = ""

                final_values = graph.invoke(
                    input_state,
                    config=config,
                )
                response_text = final_values.get("answer") or "No response generated."
                if is_first_message:
                    maybe_rename_session(active_sid, prompt)

                placeholder.markdown(response_text)

                state_snapshot = _serialize_state(final_values)


                with st.expander(f"Graph state · turn {current_turn}", expanded=False):
                    st.json(state_snapshot)

            st.session_state.chats[active_sid].append(
                {
                    "role": "assistant",
                    "content": response_text,
                    "graph_state": state_snapshot,
                    "turn": current_turn,
                }
            )

            st.rerun()