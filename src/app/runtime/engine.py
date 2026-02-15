from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from app.core.config import settings
from app.core.utils.helpers import (
    clamp_str,
    moderate_text,
    safe_json_loads,
    call_chat_completion,
)
from app.core.schemas import ExecutionPlan
from app.core.registry import get_agent_factory, validate_agent_input

logger = logging.getLogger(__name__)


# -----------------------------
# Trace
# -----------------------------


@dataclass
class ExecutionTrace:
    trace_id: str
    started_at: float = field(default_factory=time.time)
    plan: dict[str, Any] | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    status: str = "running"
    error: str | None = None
    ended_at: float | None = None

    def add_step(
        self,
        step: int,
        agent: str,
        step_input: dict[str, Any],
        output: Any,
        duration_s: float,
    ) -> None:
        self.steps.append(
            {
                "step": step,
                "agent": agent,
                "input": _safe_log_payload(step_input),
                "output": _safe_log_payload(output),
                "duration_s": duration_s,
            }
        )

    def finalize(self, status: str, error: str | None = None) -> None:
        self.status = status
        self.error = error
        self.ended_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "error": self.error,
            "plan": self.plan,
            "steps": self.steps,
            "total_duration_s": (self.ended_at or time.time()) - self.started_at,
        }


def _safe_log_payload(x: Any) -> Any:
    # Avoid logging huge prompts/documents; keep structure + size.
    try:
        s = json.dumps(x, ensure_ascii=False)
        if len(s) > 4000:
            return {"_type": type(x).__name__, "_note": "truncated", "_len": len(s)}
        return x
    except Exception:
        return {"_type": type(x).__name__}


def _resolve_placeholders(obj: Any, state: dict[str, Any]) -> Any:
    if isinstance(obj, str):
        for k, v in state.items():
            placeholder = f"$${k}$$"
            # Direct substitution: $$STEP_1_OUTPUT$$ -> entire value
            if obj == placeholder:
                return v
            # Handle nested field access: $$STEP_1_OUTPUT.fieldname$$
            if obj.startswith(placeholder + "."):
                field_path = obj[len(placeholder + ".") :]
                logger.debug(
                    f"Resolving placeholder {obj}: k={k}, v_type={type(v)}, field_path={field_path}"
                )
                if isinstance(v, dict) and field_path in v:
                    result = v[field_path]
                    logger.debug(f"Resolved {obj} to {type(result)}")
                    return result
                else:
                    logger.warning(
                        f"Placeholder {obj} could not be resolved: {k}={type(v)}, has_field={isinstance(v, dict) and field_path in v if isinstance(v, dict) else 'N/A'}"
                    )
            # Partial replacement in strings (fallback)
            if placeholder in obj:
                obj = obj.replace(placeholder, str(v))
        return obj
    if isinstance(obj, list):
        return [_resolve_placeholders(i, state) for i in obj]
    if isinstance(obj, dict):
        return {k: _resolve_placeholders(v, state) for k, v in obj.items()}
    return obj


def _validate_plan_shape(plan: ExecutionPlan) -> None:
    if not plan.plan:
        raise ValueError("Plan must include at least one step")
    if len(plan.plan) > settings.max_steps:
        raise ValueError(f"Plan too long: {len(plan.plan)} > {settings.max_steps}")
    expected = 1
    for s in plan.plan:
        if s.step != expected:
            raise ValueError(
                f"Plan steps must be sequential starting at 1 (expected {expected}, got {s.step})"
            )
        expected += 1


# -----------------------------
# Planning
# -----------------------------


def plan_steps(client: OpenAI, goal: str) -> ExecutionPlan:
    goal = clamp_str(goal, settings.max_input_chars)

    system = (
        "You are a planning system that outputs ONLY valid JSON.\n"
        "You must produce a plan for a document Q&A pipeline using these agents:\n"
        "- Librarian: input {intent_query} -> output dict with {purpose, tone, format, constraints}\n"
        "- Researcher: input {topic_query, top_k, doc_id?} -> output dict with {answer, evidence, claims}\n"
        "- Summarizer: input {text_to_summarize, max_words} -> output dict with {summary}\n"
        "- Writer: input {blueprint_json, facts, style_notes} -> output dict with {draft}\n"
        "- Verifier: input {draft, reference, verification_objective?} -> output dict with {verified_draft, is_valid, issues, suggestions}\n"
        "  reference should be the Researcher output dict (containing answer, evidence, claims) - the agent will format it internally.\n\n"
        "Rules:\n"
        f"- Max steps: {settings.max_steps}\n"
        '- Output JSON schema: {"plan": [{"step": 1, "agent": "...", "input": {...}}, ...]}\n'
        "- Use placeholders to reference prior outputs: $$STEP_1_OUTPUT$$ refers to the entire output dict.\n"
        "- To extract a field from prior output, use: $$STEP_1_OUTPUT.fieldname$$ or just $$STEP_1_OUTPUT$$ if passing the whole dict.\n"
        "- IMPORTANT: topic_query for Researcher MUST be a STRING (the user question or search topic), NOT a dict.\n"
        "- For Writer: blueprint_json=$$STEP_1_OUTPUT$$ passes the entire Librarian blueprint dict.\n"
        "- Keep inputs minimal and properly typed.\n"
        "- Typical flow: Librarian -> Researcher -> Writer -> Verifier.\n"
        "- Use Summarizer only if you have a long draft or long references.\n"
    )

    user = (
        f"Goal:\n{goal}\n\n"
        "Create a plan that:\n"
        "1) Gets a blueprint (Librarian)\n"
        "2) Retrieves evidence (Researcher)\n"
        "3) Writes final output (Writer)\n"
        "4) Verifies the draft vs evidence (Verifier)\n"
        "If the verifier needs changes, it should produce a suggested revision.\n"
    )

    raw = call_chat_completion(
        client=client,
        model=settings.planning_model,
        system=system,
        user=user,
        max_tokens=min(settings.max_tokens_per_call, 900),
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    try:
        obj = safe_json_loads(raw)
        plan = ExecutionPlan(**obj)
        _validate_plan_shape(plan)
        return plan
    except Exception as e:
        # one repair pass
        repair_system = (
            "You repair JSON to match the schema strictly.\n"
            'Return ONLY valid JSON for the schema: {"plan": [ {"step": int, "agent": str, "input": object} ] }.\n'
        )
        repair_user = (
            f"Broken JSON:\n{raw}\n\nError: {str(e)}\n\nReturn repaired JSON only."
        )
        repaired = call_chat_completion(
            client=client,
            model=settings.planning_model,
            system=repair_system,
            user=repair_user,
            max_tokens=min(settings.max_tokens_per_call, 900),
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        obj2 = safe_json_loads(repaired)
        plan2 = ExecutionPlan(**obj2)
        _validate_plan_shape(plan2)
        return plan2


# -----------------------------
# Execution
# -----------------------------


def run_engine(
    client: OpenAI,
    pinecone_index: Any,
    goal: str,
    namespace_context: str,
    namespace_knowledge: str,
    *,
    doc_id: str | None = None,
) -> dict[str, Any]:
    """
    Runs the planning + execution loop.
    If doc_id is provided, Researcher will filter retrieval to that document.
    """
    trace = ExecutionTrace(trace_id=str(uuid.uuid4()))

    if settings.enable_input_moderation:
        mod_in = moderate_text(client, goal)
        if mod_in.get("flagged"):
            trace.plan = {"blocked": "input_moderation", "moderation": mod_in}
            trace.finalize("blocked", "Input flagged by moderation")
            return {
                "trace_id": trace.trace_id,
                "output": "Request blocked by safety policy.",
                "blocked": True,
                "moderation": mod_in,
                "trace": trace.to_dict(),
            }

    try:
        plan = plan_steps(client, goal)
        trace.plan = plan.model_dump()

        state: dict[str, Any] = {}
        state["DOC_ID"] = doc_id or ""

        # create factory bound to this client + pinecone index
        factory = get_agent_factory(client, pinecone_index)

        for step in plan.plan:
            t0 = time.time()
            logger.debug(f"Step {step.step}: state keys = {list(state.keys())}")
            resolved_input = _resolve_placeholders(step.input, state)

            # Inject doc_id for Researcher if user provided one and planner didn't.
            if step.agent == "Researcher" and doc_id and "doc_id" not in resolved_input:
                resolved_input["doc_id"] = doc_id

            # Fix for planner bug: if topic_query is a dict (from Librarian output), extract purpose
            if step.agent == "Researcher" and isinstance(
                resolved_input.get("topic_query"), dict
            ):
                librarian_output = resolved_input["topic_query"]
                if "purpose" in librarian_output:
                    resolved_input["topic_query"] = librarian_output["purpose"]
                else:
                    # Fallback: use the first string value or convert to string
                    resolved_input["topic_query"] = str(librarian_output)

            # Hard bounds: input chars per field
            for k, v in list(resolved_input.items()):
                if isinstance(v, str) and len(v) > settings.max_input_chars:
                    resolved_input[k] = clamp_str(v, settings.max_input_chars)

            # Strict per-agent validation
            validated = validate_agent_input(step.agent, resolved_input)

            agent_obj = factory.create_agent(step.agent)

            if step.agent == "Librarian":
                out = agent_obj.execute(intent_query=validated["intent_query"])

            elif step.agent == "Researcher":
                out = agent_obj.execute(
                    topic_query=validated["topic_query"],
                    namespace_knowledge=namespace_knowledge,
                    top_k=validated.get("top_k", 5),
                    doc_id=validated.get("doc_id"),
                )

            elif step.agent == "Summarizer":
                out = agent_obj.execute(
                    text_to_summarize=validated["text_to_summarize"],
                    max_words=validated.get("max_words", 250),
                )

            elif step.agent == "Writer":
                out = agent_obj.execute(
                    blueprint_json=validated["blueprint_json"],
                    facts=validated["facts"],
                    style_notes=validated.get("style_notes"),
                )

            elif step.agent == "Verifier":
                # Special handling: Verifier expects reference as string, but plan might pass dict
                reference_input = validated["reference"]
                if isinstance(reference_input, dict):
                    # If it's the Researcher output dict, format the evidence as string
                    if "evidence" in reference_input and isinstance(
                        reference_input["evidence"], list
                    ):
                        evidence_list = reference_input["evidence"]
                        reference_str = "\n\n".join(
                            [
                                f"Evidence {i+1}: {json.dumps(ev, indent=2)}"
                                for i, ev in enumerate(
                                    evidence_list[:5]
                                )  # Limit to 5 pieces
                            ]
                        )
                    else:
                        reference_str = json.dumps(reference_input, indent=2)
                else:
                    reference_str = str(reference_input)

                out = agent_obj.execute(
                    draft=validated["draft"],
                    reference=reference_str,
                    verification_objective=validated.get("verification_objective"),
                )

            else:
                raise RuntimeError(f"Unsupported agent: {step.agent}")

            dt = time.time() - t0
            state[f"STEP_{step.step}_OUTPUT"] = out
            trace.add_step(step.step, step.agent, validated, out, dt)

        final_obj = state.get(f"STEP_{plan.plan[-1].step}_OUTPUT", {})
        final_text = ""

        # If last step is verifier, choose suggested revision when needed.
        if isinstance(final_obj, dict) and final_obj.get("verdict") == "needs_revision":
            final_text = final_obj.get("suggested_revision") or ""
        elif isinstance(final_obj, dict):
            final_text = (
                final_obj.get("final")
                or final_obj.get("summary")
                or json.dumps(final_obj, ensure_ascii=False)
            )
        else:
            final_text = str(final_obj)

        # Output moderation
        mod_out = moderate_text(client, final_text)
        if mod_out.get("flagged"):
            trace.finalize("blocked", "Output flagged by moderation")
            return {
                "trace_id": trace.trace_id,
                "output": "Output blocked by safety policy.",
                "blocked": True,
                "moderation": mod_out,
                "trace": trace.to_dict(),
            }

        trace.finalize("ok")
        return {
            "trace_id": trace.trace_id,
            "output": final_text,
            "blocked": False,
            "moderation": None,
            "trace": trace.to_dict(),
        }

    except Exception as e:
        logger.exception("Engine failure trace_id=%s", trace.trace_id)
        trace.finalize("error", str(e))
        return {
            "trace_id": trace.trace_id,
            "output": "Internal error while processing request.",
            "blocked": False,
            "moderation": None,
            "trace": trace.to_dict(),
        }
