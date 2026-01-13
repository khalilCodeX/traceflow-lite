from state import TraceFlowState, TaskSpec, Plan, Draft, EvalReport

__all__ = [
    "intake_node",
    "planner_node",
    "retriever_node",
    "executor_node",
    "evaluator_node",
    "router_node",
]
import time
from dotenv import load_dotenv
from tf_types import EvalDecision, Mode, Strictness
from providers import get_provider, calculate_cost, count_tokens
from .node_prompts import mode_prompts
import re

load_dotenv()

# --- Nodes ---


def intake_node(state: TraceFlowState) -> dict:
    start = time.perf_counter()
    task_spec = state.task_spec
    config = state.config
    user_input = task_spec.user_input.strip()

    error = None
    if user_input == "":
        error = "User input is empty."

    constraints = task_spec.constraints
    if config.mode == Mode.GROUNDED_QA:
        constraints["require_citations"] = "true"
    elif config.mode == Mode.TRIAGE_PLAN:
        constraints["output_format"] = "structured_plan"
    elif config.mode == Mode.CHANGE_SAFETY:
        constraints["assess_risk"] = "true"

    validated_spec = TaskSpec(user_input=user_input, constraints=task_spec.constraints)

    steps = list(state.executed_steps)
    steps.append(
        {
            "node_name": "intake",
            "input_data": {"user_input": user_input[:200], "mode": config.mode.value},
            "output_data": validated_spec.model_dump(),
            "error": error,
            "latency_ms": (time.perf_counter() - start) * 1000,
            "tokens": 0,
            "cost_usd": 0.0,
        }
    )

    return {"task_spec": validated_spec, "error": error, "executed_steps": steps}


def planner_node(state: TraceFlowState) -> dict:
    start = time.perf_counter()
    task_spec = state.task_spec
    config = state.config

    system_prompt = mode_prompts.get(config.mode, mode_prompts[Mode.GROUNDED_QA])
    user_prompt = f"Create a plan for the following task: {task_spec.user_input}"
    provider = get_provider(state.config)
    response = provider.chat_complete(
        model=config.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=config.max_tokens,
    )

    import json

    try:
        plan_json = json.loads(response.content)
        plan = Plan(
            steps=plan_json.get("steps", ["analyze", "respond"]),
            needs_context=plan_json.get("needs_context", False),
            focus_areas=plan_json.get("focus_areas", None),
        )
    except json.JSONDecodeError:
        plan = Plan(
            steps=["understand query", "generate response"], needs_context=False, focus_areas=None
        )

    tokens = response.input_tokens + response.output_tokens
    cost = calculate_cost(
        model=response.model,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )

    steps = list(state.executed_steps)
    steps.append(
        {
            "node_name": "planner",
            "input_data": {"user_input": task_spec.user_input[:200], "mode": config.mode.value},
            "output_data": plan.model_dump(),
            "error": None,
            "latency_ms": (time.perf_counter() - start) * 1000,
            "tokens": tokens,
            "cost_usd": cost,
            "cache_hit": getattr(provider, "last_cache_hit", False),
        }
    )

    return {"plan": plan, "executed_steps": steps}


def retriever_node(state: TraceFlowState) -> dict:
    """Query vector store (stub)."""
    start = time.perf_counter()
    context = []
    if state.config.retriever_fn:
        context = state.config.retriever_fn(state.task_spec.user_input)

    steps = list(state.executed_steps)
    steps.append(
        {
            "node_name": "retriever",
            "input_data": {"query": state.task_spec.user_input[:200], "top_k": state.config.top_k},
            "output_data": {"chunk_count": len(context)},
            "error": None,
            "latency_ms": (time.perf_counter() - start) * 1000,
            "tokens": 0,
            "cost_usd": 0.0,
        }
    )

    return {"context": context, "executed_steps": steps}


def executor_node(state: TraceFlowState) -> dict:
    start = time.perf_counter()
    user_input = state.task_spec.user_input if state.task_spec else ""
    provider = get_provider(state.config)
    input_token_count = count_tokens(user_input, model=state.config.model)

    context_str = ""
    if state.context:
        context_str = "\n".join([f"- {chunk.content}" for chunk in state.context])
        context_str = f"\n\nContext:\n{context_str}"
    system_prompt = (
        f"You are a helpful assistant. Answer the user's question concisely.{context_str}"
    )

    response = provider.chat_complete(
        model=state.config.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=state.config.temperature or 0.2,
        max_tokens=state.config.max_tokens,
    )

    total_latency_ms = (time.perf_counter() - start) * 1000
    total_tokens = input_token_count + response.output_tokens
    cost_usd = calculate_cost(
        model=response.model, input_tokens=input_token_count, output_tokens=response.output_tokens
    )

    citation_matches = re.findall(r"\[(\d+)\]|\[Source:\s*([^\]]+)\]", response.content)
    citations = [m[0] or m[1] for m in citation_matches if m[0] or m[1]]

    draft = Draft(
        content=response.content,
        citations=citations,
        model=response.model,
        token_count=total_tokens,
        latency_ms=total_latency_ms,
        cost_usd=cost_usd,
    )

    steps = list(state.executed_steps)
    steps.append(
        {
            "node_name": "executor",
            "input_data": {"user_input": state.task_spec.user_input[:200]},
            "output_data": draft.model_dump(),
            "error": None,
            "latency_ms": total_latency_ms,
            "tokens": total_tokens,
            "cost_usd": cost_usd,
            "cache_hit": getattr(provider, "last_cache_hit", False),
        }
    )

    return {"draft": draft, "executed_steps": steps, "final_answer": response.content}


def _get_revision_instructions(
    mode: Mode, citation_ok: bool, schema_ok: bool, risks: list[str]
) -> str:
    if mode == Mode.GROUNDED_QA and not citation_ok:
        return "Include citations [1], [2], etc. referencing the provided context sources."
    elif mode == Mode.TRIAGE_PLAN and not schema_ok:
        return "Structure your response as a numbered list (1. 2. 3.) or bullet points."
    elif mode == Mode.CHANGE_SAFETY and risks:
        return f"Add explicit safety warnings for: {', '.join(risks)}"
    return "Revise to meet mode requirements."


def evaluator_node(state: TraceFlowState) -> dict:
    start = time.perf_counter()
    draft = state.draft
    config = state.config
    context = state.context

    cost_exceeded = config.max_cost_usd and draft and draft.cost_usd > config.max_cost_usd
    latency_exceeded = config.max_latency_ms and draft and draft.latency_ms > config.max_latency_ms
    revision_limit_reached = state.revisions >= config.max_revisions

    # Mode-specific checks
    citation_coverage_ok = True
    schema_valid = True
    risk_flags = []

    if draft:
        if config.mode == Mode.GROUNDED_QA:
            # Check: citations exist OR content references context sources
            has_citations = len(draft.citations) > 0
            mentions_source = (
                any(chunk.source.lower() in draft.content.lower() for chunk in context)
                if context
                else False
            )
            citation_coverage_ok = has_citations or mentions_source or len(context) == 0

        elif config.mode == Mode.TRIAGE_PLAN:
            # Check: output has numbered list, bullets, or JSON structure
            has_numbered = bool(re.search(r"^\s*\d+\.", draft.content, re.MULTILINE))
            has_bullets = bool(re.search(r"^\s*[-•*]", draft.content, re.MULTILINE))
            has_json = draft.content.strip().startswith("{")
            schema_valid = has_numbered or has_bullets or has_json

        elif config.mode == Mode.CHANGE_SAFETY:
            # Check: flag risky keywords
            risk_keywords = [
                "delete",
                "drop",
                "truncate",
                "remove all",
                "rm -rf",
                "destroy",
                "production",
                "prod",
                "force",
                "--force",
                "sudo",
            ]
            content_lower = draft.content.lower()
            risk_flags = [kw for kw in risk_keywords if kw in content_lower]

    # Determine if mode check failed
    mode_check_failed = False
    mode_failure_reason = ""

    if config.mode == Mode.GROUNDED_QA and not citation_coverage_ok:
        mode_check_failed = True
        mode_failure_reason = "Missing citations for grounded answer"
    elif config.mode == Mode.TRIAGE_PLAN and not schema_valid:
        mode_check_failed = True
        mode_failure_reason = "Output is not structured (expected list or steps)"
    elif config.mode == Mode.CHANGE_SAFETY and risk_flags:
        # Only fail in STRICT mode; otherwise just warn
        if config.strictness == Strictness.STRICT:
            mode_check_failed = True
            mode_failure_reason = f"Risky operations detected: {risk_flags}"

    # Build EvalReport
    if cost_exceeded or latency_exceeded:
        if revision_limit_reached:
            report = EvalReport(
                decision=EvalDecision.FALLBACK,
                reasons=[
                    f"Cost: {draft.cost_usd:.4f}, Latency: {draft.latency_ms:.2f}ms exceeded limits"
                ],
                scores={"cost": 0.0, "latency": 0.0},
                citation_coverage_ok=citation_coverage_ok,
                schema_valid=schema_valid,
            )
        else:
            reason = "cost" if cost_exceeded else "latency"
            report = EvalReport(
                decision=EvalDecision.REVISE,
                reasons=[f"Exceeded {reason} limit"],
                revision_instructions="Optimize response to reduce " + reason,
                scores={reason: 0.0},
                citation_coverage_ok=citation_coverage_ok,
                schema_valid=schema_valid,
            )
    elif mode_check_failed:
        if revision_limit_reached:
            report = EvalReport(
                decision=EvalDecision.FALLBACK,
                reasons=[mode_failure_reason, "Revision limit reached"],
                scores={"mode_compliance": 0.0},
                citation_coverage_ok=citation_coverage_ok,
                schema_valid=schema_valid,
            )
        else:
            report = EvalReport(
                decision=EvalDecision.REVISE,
                reasons=[mode_failure_reason],
                revision_instructions=_get_revision_instructions(
                    config.mode, citation_coverage_ok, schema_valid, risk_flags
                ),
                scores={"mode_compliance": 0.5},
                citation_coverage_ok=citation_coverage_ok,
                schema_valid=schema_valid,
            )
    else:
        report = EvalReport(
            decision=EvalDecision.PASS,
            reasons=["Passed all checks"],
            scores={"cost": 1.0, "latency": 1.0, "mode_compliance": 1.0},
            citation_coverage_ok=citation_coverage_ok,
            schema_valid=schema_valid,
        )

    # Step tracking
    latency_ms = (time.perf_counter() - start) * 1000
    steps = list(state.executed_steps)
    steps.append(
        {
            "node_name": "evaluator",
            "input_data": {
                "draft_cost": draft.cost_usd if draft else None,
                "draft_latency": draft.latency_ms if draft else None,
                "mode": config.mode.value,
                "risk_flags": risk_flags,
            },
            "output_data": report.model_dump(),
            "error": None,
            "latency_ms": latency_ms,
            "tokens": 0,
            "cost_usd": 0.0,
        }
    )

    new_revisions = state.revisions
    if report.decision == EvalDecision.REVISE:
        new_revisions = state.revisions + 1

    return {"eval_report": report, "executed_steps": steps, "revisions": new_revisions}


def router_node(state: TraceFlowState) -> dict:
    """Handle fallback."""
    start = time.perf_counter()
    fallback_msg = "I'm sorry, I couldn't generate a reliable response."

    steps = list(state.executed_steps)
    steps.append(
        {
            "node_name": "router",
            "input_data": {
                "eval_decision": state.eval_report.decision.value if state.eval_report else None
            },
            "output_data": {"fallback_msg": fallback_msg},
            "error": None,
            "latency_ms": (time.perf_counter() - start) * 1000,
            "tokens": 0,
            "cost_usd": 0.0,
        }
    )

    return {
        "final_answer": fallback_msg,
        "executed_steps": steps,
    }
