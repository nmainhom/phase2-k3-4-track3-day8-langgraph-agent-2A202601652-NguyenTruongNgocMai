"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, make_event


class Classification(BaseModel):
    """Structured intent returned by the classifier model."""

    route: Literal["simple", "tool", "missing_info", "risky", "error"]
    rationale: str = Field(description="Short explanation of the selected route")


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── TODO(student): implement ALL nodes below ────────────────────────


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM.

    *** MUST use a real LLM call — keyword-only heuristics will lose points. ***

    Use .with_structured_output() or equivalent to get reliable enum classification.
    The LLM should classify into one of: simple, tool, missing_info, risky, error.

    Hints:
    - See llm.py for the get_llm() helper
    - Use Pydantic model or TypedDict with .with_structured_output()
    - Set risk_level to "high" for risky routes, "low" otherwise
    - Priority guide: risky > tool > missing_info > error > simple

    Return: {"route": str, "risk_level": str, "events": [make_event(...)]}
    """
    prompt = """Classify this support ticket into exactly one route.
Priority when intents overlap: risky > tool > missing_info > error > simple.
- risky: side-effecting requests such as refunds, deletion, cancellation, or sending email
- tool: factual lookups such as order status, tracking, or account search
- missing_info: vague request with no actionable subject or details
- error: timeout, crash, service failure, or unavailable system
- simple: a general support question answerable without a tool

Ticket: {query}""".format(query=state.get("query", ""))
    result = get_llm().with_structured_output(Classification).invoke(prompt)
    route = result.route
    return {
        "route": route,
        "risk_level": "high" if route == "risky" else "low",
        "events": [
            make_event(
                "classify", "completed", f"classified as {route}", rationale=result.rationale
            )
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulate transient failures for error-route scenarios to test retry loops.

    Requirements:
    - Read current attempt count from state
    - If route is "error" and attempt < 2: return error result (string containing "ERROR")
    - Otherwise: return a mock success result string
    - Append result to tool_results list

    Return: {"tool_results": [result_string], "events": [make_event(...)]}
    """
    attempt = int(state.get("attempt", 0))
    route = state.get("route", "")
    if route == "error" and attempt < 2:
        result = f"ERROR: transient support service failure (attempt {attempt + 1})"
        event_type = "failed"
    else:
        result = f"SUCCESS: mock support-tool result for: {state.get('query', '')}"
        event_type = "completed"
    return {
        "tool_results": [result],
        "events": [make_event("tool", event_type, result, attempt=attempt)],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Check whether the latest tool result is satisfactory or needs retry.

    SHOULD use LLM-as-judge for bonus points. Heuristic (e.g., check for "ERROR" substring)
    is acceptable for base score.

    Requirements:
    - Read the latest entry from tool_results
    - Set evaluation_result to "needs_retry" or "success"
    - This field drives route_after_evaluate conditional edge

    Note: You may need to add 'evaluation_result' to AgentState if not present.

    Return: {"evaluation_result": str, "events": [make_event(...)]}
    """
    latest = (state.get("tool_results") or [""])[-1]
    evaluation = "needs_retry" if "ERROR" in latest.upper() else "success"
    return {
        "evaluation_result": evaluation,
        "events": [make_event("evaluate", "completed", evaluation, tool_result=latest)],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM.

    *** MUST use a real LLM call — hardcoded strings will lose points. ***

    The LLM should generate a helpful response grounded in available context:
    - tool_results (if any)
    - approval decision (if risky route)
    - original query

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    context = "\n".join(state.get("tool_results") or []) or "No tool call was required."
    approval = state.get("approval")
    prompt = f"""You are a careful customer-support agent. Answer the customer's ticket concisely.
Use only the supplied context; do not claim that an action occurred unless the context says it did.

Customer ticket: {state.get("query", "")}
Tool context: {context}
Approval record: {approval or "No approval required"}
"""
    response = get_llm().invoke(prompt)
    content = response.content
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    answer = str(content).strip()
    return {
        "final_answer": answer,
        "events": [make_event("answer", "completed", "LLM answer generated")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generate a specific clarification question based on the vague/incomplete query.

    Note: You may need to add 'pending_question' to AgentState if not present.

    Return: {"pending_question": str, "final_answer": str, "events": [make_event(...)]}
    """
    question = "Could you share the affected account, order, or service and describe what outcome you need?"
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "requested missing details")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Describe the proposed action and why it requires approval.

    Note: You may need to add 'proposed_action' to AgentState if not present.

    Return: {"proposed_action": str, "events": [make_event(...)]}
    """
    action = f"Proposed high-impact action based on ticket: {state.get('query', '')}"
    return {"proposed_action": action, "events": [make_event("risky_action", "pending", action)]}


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default behavior: mock approval (approved=True) so tests and CI run offline.
    Extension: if env LANGGRAPH_INTERRUPT=true, use langgraph.types.interrupt() for real HITL.

    Return: {"approval": {"approved": bool, "reviewer": str, "comment": str}, "events": [make_event(...)]}
    """
    if os.getenv("LANGGRAPH_INTERRUPT", "false").lower() == "true":
        from langgraph.types import interrupt

        decision = interrupt(
            {"proposed_action": state.get("proposed_action", ""), "request": "Approve action?"}
        )
        approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
        reviewer = (
            decision.get("reviewer", "human-reviewer")
            if isinstance(decision, dict)
            else "human-reviewer"
        )
    else:
        approved, reviewer = True, "mock-reviewer"
    approval = {"approved": approved, "reviewer": reviewer, "comment": "approval recorded"}
    return {
        "approval": approval,
        "events": [make_event("approval", "completed", "approved" if approved else "rejected")],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Increment the attempt counter and log the transient failure.

    Requirements:
    - Read current attempt from state, increment by 1
    - Add an error message to errors list
    - Return updated attempt count

    Return: {"attempt": int, "errors": [str], "events": [make_event(...)]}
    """
    attempt = int(state.get("attempt", 0)) + 1
    message = f"Retry {attempt} of {state.get('max_attempts', 3)} scheduled"
    return {
        "attempt": attempt,
        "errors": [message],
        "events": [make_event("retry", "completed", message)],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded.

    This is the third layer: retry → fallback → dead letter.
    Log the failure and set a final_answer explaining that the request could not be completed.

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    answer = "We could not complete this request after several attempts. It has been escalated to support."
    return {
        "final_answer": answer,
        "events": [make_event("dead_letter", "completed", "retry limit reached")],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END.

    Return: {"events": [make_event("finalize", "completed", "workflow finished")]}
    """
    return {"events": [make_event("finalize", "completed", "workflow finished")]}
