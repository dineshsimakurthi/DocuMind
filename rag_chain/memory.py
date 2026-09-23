"""Session-scoped chat history — the store `RunnableWithMessageHistory` reads
from and writes to automatically around every chain call.

A plain in-process dict keyed by `session_id`. Fine for a single-process
Streamlit app: each browser session gets its own `session_id` (generated
once and stashed in `st.session_state`), so different tabs/users don't share
history — but it's in-memory only, so a server restart clears everything.
For persisted history across restarts, swap `InMemoryChatMessageHistory` for
a persisted implementation (e.g. `SQLChatMessageHistory`) — `get_session_history`
is the only function anything else in this project calls.

Note: as of this writing, `RunnableWithMessageHistory` itself is flagged as
pending-deprecated in LangChain, in favor of LangGraph's built-in
checkpointing/persistence. It still works correctly (confirmed by an
end-to-end test showing history growing turn over turn) — this is a
forward-looking heads-up, not a current bug, but worth checking LangChain's
migration guidance before this project sits unmaintained for a long time.
"""
from __future__ import annotations

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

_store: dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[session_id]


def clear_session(session_id: str) -> None:
    _store.pop(session_id, None)
