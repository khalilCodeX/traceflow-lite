from state import TraceFlowState, TaskSpec,Plan, Draft, EvalReport
__all__ = ["intake_node", "planner_node", "retriever_node", "executor_node", "evaluator_node", "router_node"]
import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from tf_types import EvalDecision
from providers import get_provider, calculate_cost, count_tokens

load_dotenv()

# --- Nodes ---

def intake_node(state: TraceFlowState) -> dict:
    usr_input = state.task_spec.user_input if state.task_spec else ""

    return {
        "task_spec": TaskSpec(
            user_input=usr_input,
            constraints=None
        )
    }

def planner_node(state: TraceFlowState) -> dict:
    user_input = state.task_spec.user_input if state.task_spec else ""
    needs_context = any(word in user_input.lower() for word in ["?", "how", "what", "explain", "why"])
        
    return {
        "plan": Plan(
            steps=["understand query", "generate response"],
            needs_context=needs_context,
            focus_areas=None
        )
    }

def retriever_node(state: TraceFlowState) -> dict:
    """Query vector store (stub)."""
    context = []
    if state.config.retriever_fn:
        context = state.config.retriever_fn(state.task_spec.user_input)
        
    return {"context": context}

def executor_node(state: TraceFlowState) -> dict:
    user_input = state.task_spec.user_input if state.task_spec else ""
    provider = get_provider(state.config)
    input_token_count = count_tokens(user_input, model=state.config.model)

    context_str = ""
    if state.context:
        context_str = "\n".join([f"- {chunk.content}" for chunk in state.context])
        context_str = f"\n\nContext:\n{context_str}"
    system_prompt = f"You are a helpful assistant. Answer the user's question concisely.{context_str}"

    start_time = time.perf_counter()
    response = provider.chat_complete(
        model=state.config.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=state.config.temperature or 0.2,
        max_tokens=state.config.max_tokens,
    )

    total_latency_ms = (time.perf_counter() - start_time) * 1000
    cost_usd = calculate_cost(
        model=response.model,
        input_tokens=input_token_count,
        output_tokens=response.output_tokens
    )
    
    return {
        "draft": Draft(
            content=response.content,
            citations=[],
            model=response.model,
            token_count=input_token_count + response.output_tokens,
            latency_ms=total_latency_ms,
            cost_usd=cost_usd
        ),
        "final_answer": response.content
    }

def evaluator_node(state: TraceFlowState) -> dict:
    draft = state.draft
    config = state.config

    cost_exceeded = config.max_cost_usd and draft and draft.cost_usd > config.max_cost_usd
    latency_exceeded = config.max_latency_ms and draft and draft.latency_ms > config.max_latency_ms
    revision_limit_reached = state.revisions >= config.max_revisions

    if cost_exceeded or latency_exceeded:
        if revision_limit_reached:
            return {
                "eval_report": EvalReport(
                    decision=EvalDecision.FALLBACK,
                    reasons=[f"Cost: {draft.cost_usd:.4f}, Latency: {draft.latency_ms:.2f}ms, Revisions: {state.revisions}"],
                    scores={"cost": 0.0, "latency": 0.0, "revisions": 0.0}
                )
            }
        else:
            reason = "cost" if cost_exceeded else "latency"
            return {
                "eval_report": EvalReport(
                    decision=EvalDecision.REVISE,
                    reasons=[f"Exceeded limit on {reason}."],
                    revision_instructions="Please optimize the response to meet the specified constraints.",
                    scores={reason: 0.0}
                )
            }

    return {
        "eval_report": EvalReport(
            decision=EvalDecision.PASS,
            reasons=["Passed cost and latency checks."],
            scores={"stub": 1.0}
        )
    }

def router_node(state: TraceFlowState) -> dict:
    """Handle fallback."""
    return {
        "final_answer": "I'm sorry, I couldn't generate a reliable response."
    }