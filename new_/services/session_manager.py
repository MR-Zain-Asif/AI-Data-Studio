"""DataStudio AI — Session manager bridge connecting agent to existing session store."""

# The session store is set by main.py at startup.
# Agent router calls these helpers to read/write DataFrames from the shared session.

_session_store: dict = {}


def set_agent_session_store(store: dict):
    """Called by main.py to inject the shared session store."""
    global _session_store
    _session_store = store


def get_session_df(session_id: str):
    """Return the current DataFrame for a session, or None."""
    session = _session_store.get(session_id)
    if session is None:
        return None
    return session.get("current_df")


def update_session_df(session_id: str, df):
    """Replace the current DataFrame in the session."""
    session = _session_store.get(session_id)
    if session is None:
        return
    # Save to history before replacing
    if "history" not in session:
        session["history"] = []
    old_df = session.get("current_df")
    if old_df is not None:
        session["history"].append(old_df.copy())
    session["current_df"] = df
    # Also record in operations
    if "operations" not in session:
        session["operations"] = []
    import pandas as pd
    session["operations"].append({
        "timestamp": pd.Timestamp.now().isoformat(),
        "operation": "DataStudio AI Agent Pipeline",
        "rows_affected": len(df)
    })
