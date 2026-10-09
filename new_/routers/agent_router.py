"""DataStudio AI — Agent router (WebSocket + REST endpoints).

Endpoints:
  WS   /api/agent/ws/{session_id}       — main agent WebSocket
  POST /api/assistant/{session_id}      — HTTP chat & command endpoint
  POST /api/agent/chat/{session_id}     — HTTP chat & command endpoint (alias)
  GET  /api/agent/suggestions/{sid}     — dynamic command & question suggestions
  GET  /api/agent/memory/{sid}          — session memory
  GET  /api/agent/instructions          — agent system prompt & instructions
  DEL  /api/agent/memory/{sid}          — clear memory
  GET  /api/agent/history/{sid}         — conversation history
"""

import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from pydantic import BaseModel
from typing import Optional

from services.agent_brain import agent_brain
from services.agent_executor import agent_executor
from services.agent_memory import agent_memory
from services.session_manager import get_session_df, update_session_df
from routers.quality import calculate_quality_metrics

router = APIRouter(prefix="/api", tags=["Agent"])

# Active WebSocket connections keyed by session_id
_active_connections: dict = {}


class ChatRequest(BaseModel):
    command: Optional[str] = None
    message: Optional[str] = None
    question: Optional[str] = None


def _build_dataset_info(session_id: str):
    """Build full dataset metadata context for Groq AI prompt."""
    df = get_session_df(session_id)
    if df is None:
        return None, ""
    
    df_info = {
        "rows": len(df),
        "cols": len(df.columns),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_total": int(df.isnull().sum().sum()),
        "missing_per_col": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_cols": list(df.select_dtypes(include='number').columns),
        "text_cols": list(df.select_dtypes(include='object').columns),
        "sample": df.head(3).fillna("").astype(str).to_dict(orient='records')
    }
    df_context = json.dumps(df_info, default=str)
    agent_memory.update_dataset_info(session_id, df_info)
    return df, df_context


@router.websocket("/agent/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    _active_connections[session_id] = websocket

    await websocket.send_json({
        "type": "connected",
        "message": "DataStudio ChatGPT AI Agent ready.",
        "session_id": session_id,
        "capabilities": [
            "Dataset Consciousness & Full Context Awareness",
            "General Conversation & Data Science Q&A via Groq",
            "Automatic Data Cleaning & Repair",
            "Feature Engineering & Outlier Capping",
            "AutoML Model Training & Benchmarking",
            "Business Insights Generation"
        ]
    })

    async def broadcast(message: dict):
        try:
            await websocket.send_json(message)
        except Exception:
            pass

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await broadcast({"type": "error", "message": "Invalid JSON received"})
                continue

            msg_type = data.get("type", "")

            if msg_type == "ping":
                await broadcast({"type": "pong"})
                continue

            elif msg_type in ["command", "chat", "question"]:
                user_msg = (data.get("command") or data.get("message") or data.get("question") or "").strip()
                if not user_msg:
                    continue

                agent_memory.add_conversation(session_id, "user", user_msg)
                df, df_context = _build_dataset_info(session_id)

                await broadcast({
                    "type": "agent_thinking",
                    "message": "Processing with Groq AI..."
                })

                memory_context = agent_memory.get_context_summary(session_id)
                plan = agent_brain.create_plan(user_msg, df_context, memory_context)

                # Check if it's direct answer vs tool execution plan
                if not plan.get("is_tool_action", True) or "direct_answer" in plan:
                    direct_ans = plan.get("direct_answer") or plan.get("understanding") or "I processed your request."
                    agent_memory.add_conversation(session_id, "assistant", direct_ans)
                    await broadcast({
                        "type": "agent_response",
                        "message": direct_ans,
                        "code": "",
                        "next_steps": []
                    })
                    continue

                await broadcast({
                    "type": "plan_ready",
                    "plan": plan,
                    "understanding": plan.get("understanding", ""),
                    "summary": f"Executing cleaning pipeline for: {user_msg}"
                })

                if df is not None:
                    exec_result = await agent_executor.execute_plan(plan, session_id, df, broadcast)
                    final_df = exec_result.get("final_df")
                    if final_df is not None:
                        update_session_df(session_id, final_df)

                    final_explanation = agent_brain.explain_cleaning_step(
                        user_msg,
                        {"rows": len(df), "columns": len(df.columns)},
                        {"rows": len(final_df) if final_df is not None else len(df),
                         "columns": len(final_df.columns) if final_df is not None else len(df.columns)},
                    )
                    agent_memory.add_conversation(session_id, "assistant", final_explanation)

                    await broadcast({
                        "type": "agent_response",
                        "message": final_explanation,
                        "code": exec_result.get("code", ""),
                        "next_steps": exec_result.get("next_steps", [])
                    })
                else:
                    ans = agent_brain.chat_or_answer(user_msg, "", memory_context)
                    agent_memory.add_conversation(session_id, "assistant", ans["message"])
                    await broadcast({
                        "type": "agent_response",
                        "message": ans["message"],
                        "code": "",
                        "next_steps": []
                    })

    except WebSocketDisconnect:
        _active_connections.pop(session_id, None)
    except Exception as e:
        _active_connections.pop(session_id, None)


# ── HTTP REST ENDPOINTS FOR ASSISTANT ────────────────────────────────────────

@router.post("/assistant/{session_id}")
@router.post("/agent/chat/{session_id}")
async def handle_assistant_chat(session_id: str, req: ChatRequest):
    """HTTP REST Endpoint for ChatGPT Assistant conversation and data actions."""
    user_msg = (req.command or req.message or req.question or "").strip()
    if not user_msg:
        return {"success": False, "result_message": "Message is required."}

    agent_memory.add_conversation(session_id, "user", user_msg)
    df, df_context = _build_dataset_info(session_id)
    memory_context = agent_memory.get_context_summary(session_id)

    # 1. Ask brain to classify and plan/answer
    plan = agent_brain.create_plan(user_msg, df_context, memory_context)

    # If it's a general question or direct answer
    if not plan.get("is_tool_action", True) or "direct_answer" in plan:
        answer_text = plan.get("direct_answer")
        if not answer_text:
            ans = agent_brain.chat_or_answer(user_msg, df_context, memory_context)
            answer_text = ans["message"]

        agent_memory.add_conversation(session_id, "assistant", answer_text)
        return {
            "success": True,
            "result_message": answer_text,
            "is_action": False,
            "code": ""
        }

    # If it's a data action command and dataset is available
    if df is not None:
        async def dummy_broadcast(msg): pass
        exec_result = await agent_executor.execute_plan(plan, session_id, df, dummy_broadcast)

        final_df = exec_result.get("final_df")
        if final_df is not None:
            update_session_df(session_id, final_df)

        explanation = agent_brain.explain_cleaning_step(
            user_msg,
            {"rows": len(df), "columns": len(df.columns)},
            {"rows": len(final_df) if final_df is not None else len(df),
             "columns": len(final_df.columns) if final_df is not None else len(df.columns)}
        )
        agent_memory.add_conversation(session_id, "assistant", explanation)

        return {
            "success": True,
            "result_message": explanation,
            "is_action": True,
            "code": exec_result.get("code", ""),
            "df_updated": True,
            "next_steps": exec_result.get("next_steps", [])
        }
    else:
        ans = agent_brain.chat_or_answer(user_msg, "", memory_context)
        agent_memory.add_conversation(session_id, "assistant", ans["message"])
        return {
            "success": True,
            "result_message": ans["message"],
            "is_action": False,
            "code": ""
        }


@router.get("/agent/instructions")
async def get_agent_instructions():
    """Returns agent system instructions & capabilities."""
    return {
        "title": "DataStudio AI Assistant Guide",
        "engine": "DataStudio Intelligence Engine",
        "instructions": [
            "1. Answer any general question, programming topic, or data science concept in clear Markdown format.",
            "2. Continuously monitor uploaded dataset schema, row/column counts, missing records, duplicates, and stats.",
            "3. Answer specific queries about the dataset using exact numbers and column names.",
            "4. Decompose cleaning and repair instructions into automated, multi-phase tool execution pipelines.",
            "5. Generate reproducible Python code for all data transformations and ML model fits."
        ],
        "capabilities": [
            "Conversational Q&A & Advice",
            "Dataset Metadata Consciousness",
            "Automatic Deduplication & Missing Value Imputation",
            "Statistical Outlier Detection & Capping",
            "AutoML Model Benchmarking (RandomForest, XGBoost, LightGBM, CatBoost)",
            "Business Insights Extraction"
        ]
    }


@router.get("/agent/suggestions/{session_id}")
async def get_suggestions(session_id: str):
    df = get_session_df(session_id)
    if df is not None:
        cols = list(df.columns)
        num_cols = list(df.select_dtypes(include='number').columns)
        missing_total = df.isnull().sum().sum()
        dupes = df.duplicated().sum()

        suggestions = [
            f"Summarize uploaded dataset ({len(df)} rows, {len(cols)} columns)",
            f"Fix missing values ({missing_total} total missing detected)",
            f"Remove duplicate records ({dupes} duplicates found)",
            "Train AutoML models and show performance leaderboard",
            "What is data cleaning and why is it important?",
            "Give me 5 business insights from this dataset",
            "Cap outliers in numerical columns using IQR",
            "Write a Python pandas script to filter this dataset"
        ]
        if num_cols:
            suggestions[1] = f"Analyze column '{num_cols[0]}' distribution & outliers"
        return {"suggestions": suggestions[:8]}

    return {
        "suggestions": [
            "What can you do as a Data Science AI assistant?",
            "How do I clean missing values in Pandas?",
            "Explain Random Forest vs XGBoost",
            "How should I structure my CSV dataset for ML?",
            "What are common data quality problems?",
            "How do I detect outliers statistically?"
        ]
    }


@router.get("/agent/memory/{session_id}")
async def get_memory(session_id: str):
    memory = agent_memory.get_memory(session_id)
    return {
        "conversation_count": len(memory["conversation_history"]),
        "operations": memory["executed_operations"][-10:],
        "dataset_info": memory["dataset_info"],
        "stats": memory["session_stats"]
    }


@router.delete("/agent/memory/{session_id}")
async def clear_memory(session_id: str):
    agent_memory.clear_memory(session_id)
    return {"status": "Memory cleared"}


@router.get("/agent/history/{session_id}")
async def get_history(session_id: str):
    memory = agent_memory.get_memory(session_id)
    return {"history": memory["conversation_history"][-20:]}
