"""Capability hints for upstream web models, not client-side tool execution."""
from __future__ import annotations

NATIVE_SEARCH_HINT = (
    "The web backend's native web search is enabled, independently of the listed client function tools. "
    "You have standing permission to use it autonomously whenever it can help complete the task, "
    "without waiting for an explicit search request or asking for per-search approval. "
    "Search proactively to verify facts, resolve uncertainty, find references or documentation, "
    "compare options, or gather relevant information; you are not limited to time-sensitive questions. "
    "Choose and refine queries, and repeat searches as needed. "
    "Respect explicit no-search instructions and privacy constraints; permissions for client-side tools remain unchanged. "
    "Do not invent client search tool calls, search results, or sources, "
    "and only claim to have searched when the backend actually performed a search."
)


def native_search_prompt(prompt: str, enabled: bool) -> str:
    return f"{NATIVE_SEARCH_HINT}\n\n{prompt}" if enabled else prompt
