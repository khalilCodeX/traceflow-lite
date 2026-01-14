# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for TraceFlow Lite.

ADRs are documents that capture important architectural decisions made during development, including the context, the decision itself, and its consequences.

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-0001](0001-use-sqlite-with-wal-mode.md) | Use SQLite with WAL Mode for Persistence | Accepted |
| [ADR-0002](0002-langgraph-for-agent-orchestration.md) | Use LangGraph for Agent Orchestration | Accepted |
| [ADR-0003](0003-provider-abstraction-pattern.md) | Provider Abstraction Pattern for LLM Integration | Accepted |
| [ADR-0004](0004-llm-response-caching.md) | LLM Response Caching Strategy | Accepted |
| [ADR-0005](0005-evaluation-gate-pattern.md) | Evaluation Gate Pattern for Quality Control | Accepted |
| [ADR-0006](0006-streamlit-for-ui.md) | Use Streamlit for Operations UI | Accepted |

## ADR Template

See [template.md](template.md) for creating new ADRs.

## References

- [Michael Nygard's ADR Article](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
- [ADR GitHub Organization](https://adr.github.io/)
