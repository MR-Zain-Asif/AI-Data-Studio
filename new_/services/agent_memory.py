"""DataStudio AI — Agent Memory for session state and conversation history."""

from datetime import datetime
from typing import Dict, Any, List


class AgentMemory:
    """In-memory session store for agent conversations, operations, and context."""

    def __init__(self):
        self.sessions: Dict[str, dict] = {}

    def get_memory(self, session_id: str) -> dict:
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "conversation_history": [],
                "executed_operations": [],
                "dataset_info": None,
                "last_model": None,
                "user_preferences": {},
                "session_stats": {
                    "total_commands": 0,
                    "operations_executed": 0,
                    "models_trained": 0
                }
            }
        return self.sessions[session_id]

    def add_conversation(self, session_id: str, role: str, content: str):
        memory = self.get_memory(session_id)
        memory["conversation_history"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        # Keep last 30 messages
        if len(memory["conversation_history"]) > 30:
            memory["conversation_history"] = memory["conversation_history"][-30:]
        if role == "user":
            memory["session_stats"]["total_commands"] += 1

    def add_operation(self, session_id: str, operation: str, rows_affected: int, success: bool):
        memory = self.get_memory(session_id)
        memory["executed_operations"].append({
            "operation": operation,
            "rows_affected": rows_affected,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })
        memory["session_stats"]["operations_executed"] += 1
        if operation == "auto_train":
            memory["session_stats"]["models_trained"] += 1

    def update_dataset_info(self, session_id: str, df_info: dict):
        self.get_memory(session_id)["dataset_info"] = df_info

    def get_context_summary(self, session_id: str) -> str:
        memory = self.get_memory(session_id)
        ops = memory["executed_operations"][-5:]
        ds = memory["dataset_info"]
        summary = ""
        if ds:
            summary += f"Dataset: {ds.get('rows', 0)} rows, {ds.get('cols', 0)} cols. "
        if ops:
            op_names = [o["operation"] for o in ops if o.get("success")]
            if op_names:
                summary += f"Already done: {', '.join(op_names)}. "
        return summary or "No prior context."

    def clear_memory(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

    def learn_preference(self, session_id: str, pref_type: str, value: Any):
        memory = self.get_memory(session_id)
        memory["user_preferences"][pref_type] = value


agent_memory = AgentMemory()
