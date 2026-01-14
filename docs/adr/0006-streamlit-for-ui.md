# ADR-0006: Use Streamlit for Operations UI

## Status

Accepted

## Context

TraceFlow Lite needs a user interface for:

- Viewing trace history and step details
- Inspecting LLM inputs/outputs for debugging
- Monitoring costs and token usage
- Analyzing evaluation results
- Toggling runtime settings (e.g., caching)

The UI is primarily for developers and operators, not end users. Requirements:

- Rapid development (observability tool, not the core product)
- Python-native (same language as the agent)
- Real-time data display
- Interactive filtering and selection
- Minimal frontend expertise required

Options considered:
1. **Flask/FastAPI + React** - Full control but high development cost
2. **Jupyter notebooks** - Interactive but poor UX for dashboards
3. **Streamlit** - Rapid Python-native dashboards
4. **Gradio** - ML-focused, less suitable for data dashboards
5. **Panel/Dash** - More complex than Streamlit

## Decision

Use Streamlit for the operations UI.

Architecture:
```
ui/
├── app.py          # Main Streamlit application
└── README.md       # UI-specific documentation
```

Key features implemented:
- **Trace list** - Filterable table of all traces
- **Step viewer** - Detailed view of individual steps with syntax highlighting
- **Cost analytics** - Token usage and cost breakdown
- **Eval dashboard** - Evaluation scores and feedback
- **Cache toggle** - Runtime enable/disable of LLM caching
- **Dark theme** - Developer-friendly appearance

State management:
```python
# Session state for UI persistence
if "selected_trace" not in st.session_state:
    st.session_state.selected_trace = None
```

## Consequences

### Positive

- **Rapid development** - UI built in hours, not days
- **Python-native** - No JavaScript/TypeScript required
- **Reactive** - Automatic re-rendering on data changes
- **Built-in components** - Tables, charts, code blocks included
- **Easy deployment** - `streamlit run ui/app.py`
- **Session state** - Maintains UI state across interactions

### Negative

- **Limited customization** - Constrained to Streamlit's component model
- **Performance** - Full page re-render on each interaction
- **Threading model** - Requires careful handling of shared resources
- **Not production-grade** - Suitable for internal tools, not customer-facing
- **Scaling limits** - Single-process, not designed for high concurrency

### Neutral

- Streamlit Cloud available for hosted deployments
- Custom components possible but add complexity
- Mobile support is limited

## References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Session State](https://docs.streamlit.io/library/api-reference/session-state)
- [Streamlit Theming](https://docs.streamlit.io/library/advanced-features/theming)
