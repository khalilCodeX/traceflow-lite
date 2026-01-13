from langgraph.graph import StateGraph, START, END
from state import TraceFlowState
from tf_types import EvalDecision
from graph_flow.nodes.nodes import (
    intake_node,
    planner_node,
    retriever_node,
    executor_node,
    evaluator_node,
    router_node,
)


def router_after_planner(state: TraceFlowState):
    """Route to retriever after planner completes."""
    if state.plan and state.plan.needs_context:
        return "retriever"
    return "executor"


def route_after_evaluator(state: TraceFlowState):
    """Route after evaluator completes."""
    if state.eval_report and state.eval_report.decision == EvalDecision.PASS:
        return "end"
    if (
        state.eval_report
        and state.eval_report.decision == EvalDecision.REVISE
        and state.revisions < state.config.max_revisions
    ):
        return "executor"
    return "router"


def build_traceflow_graph():
    graph = StateGraph(TraceFlowState)

    # Add nodes
    graph.add_node("intake", intake_node)
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("executor", executor_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("router", router_node)

    # Define edges
    graph.add_edge(START, "intake")
    graph.add_edge("intake", "planner")
    graph.add_conditional_edges(
        "planner", router_after_planner, {"retriever": "retriever", "executor": "executor"}
    )
    graph.add_edge("retriever", "executor")
    graph.add_edge("executor", "evaluator")
    graph.add_conditional_edges(
        "evaluator", route_after_evaluator, {"end": END, "executor": "executor", "router": "router"}
    )
    graph.add_edge("router", END)

    return graph.compile()
