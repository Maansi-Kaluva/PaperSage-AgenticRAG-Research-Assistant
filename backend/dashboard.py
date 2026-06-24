"""
Lightweight in-app dashboard data provider.

Combines:
  - Local session stats (confidence scores, route types, turn count) —
    derived from data already stored in Streamlit's session state /
    LangGraph checkpointer, no extra cost.
  - LangSmith run stats (latency, token usage) — fetched via the
    LangSmith SDK, scoped to this session's thread_id.

Designed to degrade gracefully: if LangSmith isn't configured (no
LANGCHAIN_API_KEY) or the API call fails, the dashboard still renders
using local stats only.
"""

import os
import statistics


def get_local_session_stats(chat_history: list[dict]) -> dict:
    """
    Derive stats purely from the in-memory chat history for this session.
    `chat_history` is st.session_state.chats[active_sid] — a list of
    {"role": ..., "content": ..., "confidence_score": ..., "graph_state": ...}
    """
    assistant_msgs = [
        m for m in chat_history
        if m["role"] == "assistant" and m.get("content", "").strip()
    ]

    route_counts: dict[str, int] = {}
    for m in assistant_msgs:
        route = (m.get("graph_state") or {}).get("route") or "unknown"
        route_counts[route] = route_counts.get(route, 0) + 1

    return {
        "total_queries": len([m for m in chat_history if m["role"] == "user"]),
        "route_counts": route_counts,
    }


def get_langsmith_session_stats(thread_id: str, limit: int = 50) -> dict | None:
    """
    Fetch recent run stats from LangSmith for this session's thread_id.
    Returns None if LangSmith isn't configured or the call fails — the
    caller should fall back to local-only stats in that case.
    """
    if not os.environ.get("LANGCHAIN_API_KEY"):
        return None

    try:
        from langsmith import Client

        client = Client()
        project_name = os.environ.get("LANGCHAIN_PROJECT", "papersage")

        runs = list(
            client.list_runs(
                project_name=project_name,
                filter=f'has(metadata, \'{{"thread_id": "{thread_id}"}}\')',
                limit=limit,
            )
        )

        if not runs:
            return None

        latencies = []
        total_tokens = 0
        for run in runs:
            if run.start_time and run.end_time:
                latencies.append((run.end_time - run.start_time).total_seconds())
            usage = (run.outputs or {}).get("usage_metadata") or {}
            total_tokens += usage.get("total_tokens", 0)

        return {
            "run_count": len(runs),
            "avg_latency_sec": round(statistics.mean(latencies), 2) if latencies else None,
            "total_tokens": total_tokens or None,
        }
    except Exception:
        return None