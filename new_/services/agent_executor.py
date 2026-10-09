"""DataStudio AI — Agent Executor: runs plans step by step with live WebSocket broadcast."""

import asyncio
import pandas as pd
from services.agent_brain import agent_brain
from services.agent_tools_complete import tools
from services.agent_memory import agent_memory
from services.session_manager import update_session_df


class AgentExecutor:
    """Executes an agent plan phase by phase, broadcasting progress via WebSocket."""

    async def execute_plan(self, plan: dict, session_id: str, df: pd.DataFrame, broadcast) -> dict:
        current_df = df.copy() if df is not None else None
        all_code = ["import pandas as pd", "import numpy as np", ""]
        all_results = []

        # Baseline quality calculation
        current_quality_score = 50.0
        if current_df is not None:
            try:
                q_base = tools.analyze_quality(current_df)
                current_quality_score = float(q_base.get("quality_score", 50.0))
            except Exception:
                current_quality_score = 50.0

        phases = plan.get("phases", [])
        all_steps = []
        for phase in phases:
            for step in phase.get("steps", []):
                step = dict(step)
                step["phase_name"] = phase.get("phase_name", "")
                all_steps.append(step)

        total_steps = len(all_steps)

        await broadcast({
            "type": "execution_start",
            "total_steps": total_steps,
            "phases": [p.get("phase_name") for p in phases]
        })

        for i, step in enumerate(all_steps):
            progress = int((i / max(total_steps, 1)) * 100)

            await broadcast({
                "type": "step_start",
                "step_id": step.get("step_id", f"s{i+1}"),
                "step_number": i + 1,
                "total_steps": total_steps,
                "tool": step.get("tool", ""),
                "description": step.get("description", ""),
                "phase": step.get("phase_name", ""),
                "progress_pct": progress
            })

            prev_score = current_quality_score
            result = await self._execute_step(step, session_id, current_df)

            if result.get("df") is not None:
                current_df = result["df"]
                update_session_df(session_id, current_df)

                # Recalculate quality score immediately after dataframe update
                try:
                    q_new = tools.analyze_quality(current_df)
                    current_quality_score = float(q_new.get("quality_score", current_quality_score))
                    improvement = round(current_quality_score - prev_score, 1)

                    await broadcast({
                        "type": "dashboard_update",
                        "quality_score": current_quality_score,
                        "rows": int(len(current_df)),
                        "cols": int(len(current_df.columns)),
                        "missing_total": int(current_df.isnull().sum().sum()),
                        "duplicate_rows": int(current_df.duplicated().sum()),
                        "issues_count": int(q_new.get("total_issues", 0)),
                        "improvement": float(improvement)
                    })
                except Exception:
                    pass

            if result.get("code"):
                all_code.append(f"# Step {i+1}: {step.get('description', '')}")
                all_code.append(result["code"])
                all_code.append("")

            agent_memory.add_operation(
                session_id,
                step.get("tool", ""),
                result.get("rows_affected", 0),
                result.get("success", True)
            )

            all_results.append(result)

            await broadcast({
                "type": "step_complete",
                "step_id": step.get("step_id", f"s{i+1}"),
                "step_number": i + 1,
                "total_steps": total_steps,
                "tool": step.get("tool", ""),
                "message": result.get("message", "Done"),
                "rows_affected": result.get("rows_affected", 0),
                "data": result.get("data"),
                "code": result.get("code", ""),
                "progress_pct": int(((i + 1) / max(total_steps, 1)) * 100),
                "success": result.get("success", True)
            })

            await asyncio.sleep(0.15)

        final_code = "\n".join(all_code)

        # Suggest next steps
        next_steps = []
        try:
            if current_df is not None:
                completed = [s.get("tool", "") for s in all_steps]
                df_stats = {"shape": list(current_df.shape), "columns": list(current_df.columns)}
                dataset_type = plan.get("dataset_type", "general")
                next_steps = agent_brain.suggest_next_steps(completed, df_stats, dataset_type)
        except Exception:
            next_steps = []

        await broadcast({
            "type": "execution_complete",
            "total_steps": total_steps,
            "code": final_code,
            "next_steps": next_steps,
            "final_shape": list(current_df.shape) if current_df is not None else None
        })

        return {
            "results": all_results,
            "final_df": current_df,
            "code": final_code,
            "next_steps": next_steps
        }

    async def _execute_step(self, step: dict, session_id: str, df) -> dict:
        tool = step.get("tool", "")
        params = step.get("params", {}) or {}

        if df is None and tool not in ("generate_insights", "export_data"):
            return {"success": False, "message": "No dataset loaded", "rows_affected": 0, "code": ""}

        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._run_tool_sync, tool, params, df, session_id
            )
        except Exception as e:
            recovery = await self._recover(tool, params, str(e), df)
            return recovery

    def _run_tool_sync(self, tool: str, params: dict, df, session_id: str) -> dict:
        """Synchronous tool dispatch — runs in thread pool."""

        if tool == "analyze_quality":
            result = tools.analyze_quality(df)
            return {
                "success": True, "data": result,
                "message": f"Found {result['total_issues']} issues. Quality score: {result['quality_score']}/100",
                "rows_affected": 0, "code": "quality_report = tools.analyze_quality(df)"
            }

        elif tool == "profile_columns":
            result = tools.profile_columns(df)
            return {
                "success": True, "data": result,
                "message": f"Profiled {len(result)} columns in detail",
                "rows_affected": 0, "code": "column_profiles = tools.profile_columns(df)"
            }

        elif tool == "find_correlations":
            result = tools.find_correlations(df)
            high = len(result.get("high_correlations", []))
            return {
                "success": True, "data": result,
                "message": f"Found {high} significant correlations",
                "rows_affected": 0, "code": "corr_matrix = df.corr()"
            }

        elif tool == "detect_outliers":
            cols = params.get("columns")
            result = tools.detect_outliers(df, cols)
            total = sum(v["count"] for v in result.values())
            return {
                "success": True, "data": result,
                "message": f"Detected {total} outliers across {len(result)} columns",
                "rows_affected": total, "code": "# IQR outlier detection complete"
            }

        elif tool == "scan_pii":
            result = tools.scan_pii(df)
            return {
                "success": True, "data": {"pii": result},
                "message": f"Found {len(result)} PII instances in dataset",
                "rows_affected": 0, "code": "# PII scan complete"
            }

        elif tool == "remove_duplicates":
            new_df, msg, affected, code = tools.remove_duplicates(df)
            return {"success": True, "df": new_df, "message": msg, "rows_affected": affected, "code": code}

        elif tool == "fill_missing":
            col = params.get("column")
            method = params.get("method", "auto")
            new_df, msg, affected, code = tools.fill_missing(df, col, method)
            return {"success": True, "df": new_df, "message": msg, "rows_affected": affected, "code": code}

        elif tool == "fix_outliers":
            col = params.get("column")
            method = params.get("method", "cap")
            new_df, msg, affected, code = tools.fix_outliers(df, col, method)
            return {"success": True, "df": new_df, "message": msg, "rows_affected": affected, "code": code}

        elif tool == "standardize_text":
            col = params.get("column")
            new_df, msg, affected, code = tools.standardize_text(df, col)
            return {"success": True, "df": new_df, "message": msg, "rows_affected": affected, "code": code}

        elif tool == "convert_types":
            new_df, msg, affected, code = tools.convert_types(df)
            return {"success": True, "df": new_df, "message": msg, "rows_affected": affected, "code": code}

        elif tool == "remove_empty_columns":
            new_df, msg, affected, code = tools.remove_empty_columns(df)
            return {"success": True, "df": new_df, "message": msg, "rows_affected": affected, "code": code}

        elif tool == "remove_constant_columns":
            new_df, msg, affected, code = tools.remove_constant_columns(df)
            return {"success": True, "df": new_df, "message": msg, "rows_affected": affected, "code": code}

        elif tool == "extract_date_features":
            col = params.get("column")
            if not col:
                date_cols = [c for c in df.columns if any(kw in c.lower() for kw in ['date', 'time', 'created', 'updated', 'timestamp'])]
                col = date_cols[0] if date_cols else None
            if col:
                new_df, msg, affected, code = tools.extract_date_features(df, col)
                return {"success": True, "df": new_df, "message": msg, "rows_affected": affected, "code": code}
            return {"success": False, "message": "No date column found for feature extraction", "rows_affected": 0, "code": ""}

        elif tool == "encode_categorical":
            col = params.get("column")
            method = params.get("method", "auto")
            new_df, msg, affected, code = tools.encode_categorical(df, col, method)
            return {"success": True, "df": new_df, "message": msg, "rows_affected": affected, "code": code}

        elif tool == "scale_features":
            method = params.get("method", "robust")
            cols = params.get("columns")
            new_df, msg, affected, code = tools.scale_features(df, method, cols)
            return {"success": True, "df": new_df, "message": msg, "rows_affected": affected, "code": code}

        elif tool == "auto_train":
            target = params.get("target_col") or params.get("target_column")
            task_type = params.get("task_type")

            if not target:
                from services.ml_service import detect_target_column
                try:
                    det = detect_target_column(df)
                    target = det.get("suggested_target")
                    if not task_type:
                        task_type = det.get("task_type")
                except Exception:
                    target = df.columns[-1] if len(df.columns) > 0 else None

            if not target:
                return {"success": False, "message": "Cannot infer target column. Please specify.", "rows_affected": 0, "code": ""}

            result = tools.auto_train(df, target, task_type)
            if "error" in result:
                return {"success": False, "message": result["error"], "rows_affected": 0, "code": ""}

            leaderboard = result.get("leaderboard", [])
            best = result.get("best_model", "Unknown")
            best_metrics = leaderboard[0]["metrics"] if leaderboard else {}
            primary = "f1_weighted" if result.get("task_type") == "classification" else "r2"
            best_score = best_metrics.get(primary, 0)
            return {
                "success": True, "data": result,
                "message": f"Trained {len(leaderboard)} models. Best: {best} ({round(best_score * 100, 1)}% {primary})",
                "rows_affected": 0, "code": result.get("code", "")
            }

        elif tool == "generate_insights":
            dataset_type = params.get("dataset_type", "general")
            analysis = params.get("analysis", {})
            insights = tools.generate_insights(df, analysis, dataset_type)
            return {
                "success": True, "data": {"insights": insights},
                "message": f"Generated {len(insights)} business insights",
                "rows_affected": 0, "code": "# Business insights generated by DataStudio AI"
            }

        elif tool == "export_data":
            fmt = params.get("format", "csv")
            _, msg, _, code = tools.export_data(df, fmt)
            return {"success": True, "message": msg, "rows_affected": 0, "code": code}

        else:
            return {"success": False, "message": f"Unknown tool: '{tool}'", "rows_affected": 0, "code": ""}

    async def _recover(self, tool: str, params: dict, error: str, df) -> dict:
        """Fallback recovery when a tool fails."""
        recovery_map = {
            "fill_missing": ("fill_missing", {**params, "method": "median"}, "KNN failed; using median imputation"),
            "encode_categorical": ("encode_categorical", {**params, "method": "label"}, "One-hot failed; using label encoding"),
            "scale_features": ("scale_features", {**params, "method": "minmax"}, "Robust scaler failed; using MinMax"),
        }

        if tool in recovery_map:
            fallback_tool, fallback_params, recovery_msg = recovery_map[tool]
            try:
                step = {"tool": fallback_tool, "params": fallback_params, "description": recovery_msg}
                result = await self._execute_step(step, None, df)
                result["recovery_message"] = recovery_msg
                return result
            except Exception:
                pass

        return {
            "success": False,
            "message": f"Step failed: {error[:120]}",
            "recovery_message": "Skipped and continuing",
            "rows_affected": 0,
            "code": f"# Skipped: {tool} — {error[:80]}"
        }


agent_executor = AgentExecutor()
