# ADR-0002: Use LangGraph for Agent Orchestration

## Status

Accepted

## Context

TraceFlow Lite implements a multi-step agent workflow with distinct phases: intake, planning, execution, evaluation, and revision. The workflow requires:

- Conditional branching based on evaluation results
- State management across nodes
- Clear separation between planning and execution phases
- Support for iterative refinement (revision loops)
- Observable, debuggable execution flow

Options considered:
1. **Custom state machine** - Full control but significant implementation effort
2. **LangChain LCEL** - Good for linear chains, limited branching support
3. **LangGraph** - Graph-based orchestration with conditional edges
4. **Temporal/Prefect** - Enterprise workflow engines, heavy dependencies
5. **Simple function composition** - Minimal overhead but poor observability

## Decision

Use LangGraph for agent workflow orchestration.

Graph structure:
```
intake_node → planner_node → executor_node → eval_node
                   ↑                              ↓
                   └──── revision_node ←──────────┘
                              ↓
                         (conditional: retry or end)
```

Key design choices:
- **TypedDict state** - Strongly typed state passed between nodes
- **Conditional edges** - Route based on evaluation pass/fail
- **Node isolation** - Each node has single responsibility
- **Explicit wiring** - Graph structure defined declaratively

## Consequences

### Positive

- **Visual clarity** - Graph structure maps directly to agent workflow
- **Conditional routing** - Native support for eval-based branching
- **State typing** - TypedDict provides IDE support and validation
- **Debugging** - Clear node boundaries for step-by-step inspection
- **LangChain ecosystem** - Compatible with LangChain tools and callbacks
- **Extensibility** - Easy to add new nodes or modify routing logic

### Negative

- **Dependency** - Adds `langgraph` as a required dependency
- **Learning curve** - Graph concepts may be unfamiliar to some developers
- **Overhead** - More abstraction than simple function calls
- **Version coupling** - Must track LangGraph API changes

### Neutral

- Graph compilation creates a static execution plan
- State is passed by value between nodes (immutable pattern)
- Requires explicit `END` node for termination

## References

- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [LangGraph Conditional Edges](https://langchain-ai.github.io/langgraph/concepts/low_level/#conditional-edges)
- [TraceFlow Workflow Diagram](../Architecture/traceflow_workflow.md)
