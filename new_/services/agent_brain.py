"""DataStudio AI — Agent Brain powered by Groq LLM.

The brain receives dataset context + user command/question, answers general chat questions,
produces structured execution plans for data actions, and narrates every result in plain English.
"""

import json
from services.groq_client import call_groq, call_groq_json

# ── MASTER AGENT SYSTEM INSTRUCTIONS ─────────────────────────────────────────
_SYSTEM_INSTRUCTIONS = """
You are DataStudio AI — an intelligent, highly capable autonomous Data Scientist and Analytics Assistant.

YOUR IDENTITY & ROLE:
- You are the native AI engine of DataStudio, with deep data science, analytics, and data cleaning expertise.
- You can answer ANY normal question (general knowledge, coding, pandas, Python, algorithms, math, data science concepts, data modeling, etc.).
- When a dataset is uploaded, you are FULLY CONSCIOUS and AWARE of its structure (row count, column count, column names, data types, missing value stats, duplicate count, statistical summary, and sample data).

CORE OPERATIONAL RULES:
1. If the user asks a GENERAL QUESTION or conversation (e.g. "What is linear regression?", "Hi!", "How do I deal with missing values in Python?", "Who are you?"):
   - Respond directly in helpful, clear, professional GFM Markdown format (with headings, bold text, bullet points, and code blocks if relevant).
2. If the user asks a QUESTION ABOUT THE UPLOADED DATASET (e.g. "What data is uploaded?", "How many rows are missing?", "Which column has the highest mean?"):
   - Use the uploaded dataset context provided to answer precisely with actual numbers, column names, and statistics.
3. If the user COMMANDS A DATA ACTION (e.g. "clean dataset", "drop duplicates", "fill missing values", "train ML model", "fix outliers"):
   - Provide a step-by-step technical execution plan using available tools.

AVAILABLE DATA STUDIO TOOLS:
- ANALYSIS: analyze_quality, profile_columns, find_correlations, detect_outliers, scan_pii, describe_dataset
- CLEANING: remove_duplicates, fill_missing, fix_outliers, standardize_text, convert_types, remove_empty_columns
- ENGINEERING: extract_date_features, encode_categorical, scale_features, bin_numeric
- ML: auto_train, tune_best_model, explain_model, cluster_data
- INSIGHTS: generate_insights, create_report, export_data

ALWAYS maintain a polite, intelligent, expert tone.
"""

_PLANNER_SYSTEM = _SYSTEM_INSTRUCTIONS + """
CLASSIFY & PLAN INSTRUCTION:
Determine if the user's prompt requires executing dataset cleaning/ML tools or if it's a general question/answer.

If it requires dataset cleaning/ML tools, return JSON:
{
  "is_tool_action": true,
  "understanding": "Specific 1-2 sentence description of what you see and will execute",
  "dataset_type": "sales/hr/healthcare/finance/ecommerce/education/general",
  "detected_target": "column_name or null",
  "key_issues": ["issue1", "issue2"],
  "phases": [
    {
      "phase_name": "Analysis / Cleaning / FE / ML",
      "phase_description": "What this phase accomplishes",
      "steps": [
        {
          "step_id": "s1",
          "tool": "tool_name",
          "description": "Specific description with column names and numbers",
          "params": {},
          "priority": "critical/high/medium",
          "estimated_seconds": 2,
          "why": "Why this step matters"
        }
      ]
    }
  ],
  "total_steps": 4,
  "estimated_minutes": 1,
  "needs_clarification": false,
  "clarification": null
}

If it is a general question, explanation request, or conversational prompt, return JSON:
{
  "is_tool_action": false,
  "understanding": "Direct answer to user prompt",
  "direct_answer": "Complete markdown answer to user prompt using dataset context if available."
}
"""


class AgentBrain:
    """Groq-powered brain that plans, answers questions, and narrates data science operations."""

    def chat_or_answer(self, prompt: str, df_context: str, memory_context: str) -> dict:
        """Answer general questions or dataset queries directly via Groq API."""
        system_prompt = _SYSTEM_INSTRUCTIONS + f"""
CURRENT UPLOADED DATASET CONTEXT:
{df_context if df_context else "No dataset uploaded yet."}

PREVIOUS CONVERSATION CONTEXT:
{memory_context if memory_context else "None"}
"""
        user_msg = f"User Query: {prompt}"
        answer = call_groq(system_prompt, user_msg, fast=False, max_tokens=1500)
        return {"message": answer, "df_context_used": bool(df_context)}

    def create_plan(self, command: str, df_context: str, memory_context: str) -> dict:
        """Decompose user command into an execution plan or general answer."""
        user_msg = f"""
User command/question: {command}

Dataset context:
{df_context if df_context else "No dataset uploaded yet."}

Previous context:
{memory_context if memory_context else "None"}

Respond with JSON plan or direct answer as specified in system prompt.
"""
        result = call_groq_json(_PLANNER_SYSTEM, user_msg, max_tokens=2500)
        if not result or ("phases" not in result and "direct_answer" not in result):
            return self._fallback_plan(command, df_context)
        return result

    def _fallback_plan(self, command: str, df_context: str) -> dict:
        """Fallback plan when Groq plan parsing falls back."""
        cmd = command.lower()
        
        # Check if it looks like general conversation
        if any(w in cmd for w in ["hi", "hello", "hey", "what is", "explain", "who are you", "how to"]):
            answer = self.chat_or_answer(command, df_context, "")
            return {
                "is_tool_action": False,
                "understanding": answer["message"],
                "direct_answer": answer["message"]
            }

        steps_cleaning = [
            {"step_id": "s1", "tool": "analyze_quality", "description": "Analyze data quality and detect issues", "params": {}, "priority": "critical", "estimated_seconds": 1, "why": "Establish data quality baseline"},
            {"step_id": "s2", "tool": "remove_duplicates", "description": "Remove duplicate rows", "params": {}, "priority": "critical", "estimated_seconds": 1, "why": "Eliminate redundant records"},
            {"step_id": "s3", "tool": "fill_missing", "description": "Fill missing values automatically", "params": {"method": "auto"}, "priority": "critical", "estimated_seconds": 1, "why": "Prevent missing value corruption"},
            {"step_id": "s4", "tool": "fix_outliers", "description": "Cap statistical outliers with IQR method", "params": {"method": "cap"}, "priority": "high", "estimated_seconds": 1, "why": "Normalize extreme values"},
        ]

        phases = [
            {"phase_name": "Cleaning & Quality Repair", "phase_description": "Fix all data quality issues automatically", "steps": steps_cleaning}
        ]

        return {
            "is_tool_action": True,
            "understanding": f"Executing cleaning pipeline for: {command}",
            "dataset_type": "general",
            "phases": phases,
            "total_steps": len(steps_cleaning),
            "estimated_minutes": 1,
            "confidence": 0.9,
            "needs_clarification": False
        }

    def explain_dataset(self, df_stats: dict) -> str:
        """Plain-English dataset explanation for first-time view."""
        system = _SYSTEM_INSTRUCTIONS + """
Analyze these statistics and write a 3-sentence welcome:
Sentence 1: What type of dataset this is and its exact row/column counts.
Sentence 2: Key data quality issues (missing values, duplicates, outliers).
Sentence 3: Recommended first action.
Be specific — mention actual column names and numbers."""
        user = f"Dataset statistics: {json.dumps(df_stats, default=str)}"
        res = call_groq(system, user, fast=False)
        if not res or "unavailable" in res.lower():
            shape = df_stats.get("shape", [0, 0])
            res = f"Loaded dataset with {shape[0]} rows and {shape[1]} columns. I spottd missing values and potential duplicates. Click 'Run Auto-Repair' or ask me to clean it."
        return res

    def explain_cleaning_step(self, operation: str, before: dict, after: dict, column: str = None) -> str:
        """2-sentence explanation of a cleaning step."""
        system = """Explain this data cleaning step in 2 clean sentences.
Sentence 1: What was found and modified.
Sentence 2: Why this improves data quality."""
        user = f"Operation: {operation}\nColumn: {column or 'all columns'}\nBefore: {before}\nAfter: {after}"
        res = call_groq(system, user, fast=True, max_tokens=150)
        if not res or "unavailable" in res.lower():
            res = f"Successfully executed {operation}. Data quality and consistency updated."
        return res

    def answer_question(self, question: str, df_context: str, computed_answer) -> str:
        """Answer a data question using Groq with computed data as context."""
        system = _SYSTEM_INSTRUCTIONS + """
Explain the computed answer to the user in a friendly ChatGPT style. Mention exact numbers and columns."""
        user = f"Question: {question}\nComputed answer: {json.dumps(computed_answer, default=str)[:800]}\nDataset context: {df_context}"
        return call_groq(system, user, fast=False, max_tokens=600)


agent_brain = AgentBrain()
