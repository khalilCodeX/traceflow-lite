"""
TraceFlow Lite - Streamlit UI
A modern interface for running and inspecting agent traces.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from client import TraceFlowClient
from tf_types import Mode, RunConfig, Strictness
from datetime import datetime
import os

# Optional RAG imports
try:
    from utils.retriever_utils import chroma_retriever
    from utils.vector_types import chroma_params

    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="TraceFlow Lite", layout="wide", initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown(
    """
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Cards */
    .trace-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .trace-card:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(99, 102, 241, 0.5);
    }
    
    /* Status badges */
    .status-done {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-failed {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-running {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* Mode badges */
    .mode-badge {
        background: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 500;
    }
    
    /* Metrics */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #a5b4fc;
    }
    .metric-label {
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.5);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Steps timeline */
    .step-item {
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #6366f1;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-radius: 0 8px 8px 0;
    }
    .step-item.executor {
        border-left-color: #10b981;
    }
    .step-item.evaluator {
        border-left-color: #f59e0b;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: rgba(255, 255, 255, 0.6);
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
    
    /* Input fields */
    .stTextArea textarea, .stTextInput input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        color: white !important;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background: rgba(0, 0, 0, 0.3) !important;
        border-radius: 8px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Initialize client
@st.cache_resource
def get_client():
    return TraceFlowClient()


@st.cache_resource
def get_retriever(collection_name: str, db_path: str):
    """Get or create a cached retriever instance."""
    if not RAG_AVAILABLE:
        return None
    params = chroma_params(collection=collection_name, directory=db_path)
    return chroma_retriever(local=True, params=params)


def format_timestamp(dt: datetime) -> str:
    """Format datetime for display."""
    if dt is None:
        return "—"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def get_status_badge(status: str) -> str:
    """Return HTML for status badge."""
    status_lower = status.lower() if isinstance(status, str) else status.value.lower()
    return f'<span class="status-{status_lower}">{status_lower.upper()}</span>'


def get_mode_badge(mode: str) -> str:
    """Return HTML for mode badge."""
    mode_str = mode if isinstance(mode, str) else mode.value
    return f'<span class="mode-badge">{mode_str.upper()}</span>'


def render_sidebar():
    """Render sidebar with navigation and new run form."""
    with st.sidebar:
        st.markdown('<p class="main-header">TraceFlow</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="sub-header">Agent Observability Platform</p>', unsafe_allow_html=True
        )

        st.divider()

        # Navigation
        page = st.radio(
            "Navigation",
            ["New Run", "Trace History", "Analytics"],
            label_visibility="collapsed",
        )

        st.divider()

        # Quick stats
        client = get_client()
        traces = client.list_traces(limit=100)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Traces", len(traces))
        with col2:
            done_count = sum(1 for t in traces if t.status.value == "done")
            st.metric("Success Rate", f"{(done_count / len(traces) * 100) if traces else 0:.0f}%")

        return page


def render_new_run_page():
    """Render the new run page."""
    st.markdown("## New Run")
    st.markdown("Execute a new agent workflow with custom configuration.")

    col1, col2 = st.columns([2, 1])

    with col1:
        user_input = st.text_area(
            "User Input", placeholder="Enter your question or task...", height=150, key="user_input"
        )

    with col2:
        st.markdown("### Configuration")

        mode = st.selectbox(
            "Mode",
            [Mode.GROUNDED_QA, Mode.TRIAGE_PLAN, Mode.CHANGE_SAFETY],
            format_func=lambda x: {
                Mode.GROUNDED_QA: "Grounded QA",
                Mode.TRIAGE_PLAN: "Triage Plan",
                Mode.CHANGE_SAFETY: "Change Safety",
            }.get(x, x.value),
        )

        provider = st.selectbox(
            "Provider",
            ["openai", "anthropic"],
            format_func=lambda x: "OpenAI" if x == "openai" else "Anthropic",
        )

        model_options = {
            "openai": ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
            "anthropic": [
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
                "claude-3-haiku-20240307",
            ],
        }

        model = st.selectbox("Model", model_options[provider])

        strictness = st.selectbox(
            "Strictness",
            [Strictness.LENIENT, Strictness.BALANCED, Strictness.STRICT],
            index=1,
            format_func=lambda x: x.value.capitalize(),
        )

        with st.expander("Advanced Settings"):
            max_tokens = st.slider("Max Tokens", 100, 4096, 1024)
            max_cost = st.number_input("Max Cost ($)", 0.01, 10.0, 1.50, step=0.1)
            max_latency = st.number_input("Max Latency (ms)", 1000, 60000, 30000, step=1000)
            max_revisions = st.slider("Max Revisions", 0, 5, 3)
            enable_cache = st.checkbox(
                "Enable LLM Cache",
                value=True,
                help="Cache responses to save cost on repeated queries",
            )

    # RAG Configuration Section
    st.divider()
    st.markdown("### RAG Configuration (Optional)")

    enable_rag = st.checkbox(
        "Enable RAG",
        value=False,
        help="Use Retrieval-Augmented Generation with your own documents",
    )

    retriever_fn = None
    top_k = 5  # Default value
    if enable_rag:
        if not RAG_AVAILABLE:
            st.error(
                "RAG dependencies not available. Install chromadb and sentence-transformers."
            )
        elif not os.getenv("OPENAI_API_KEY"):
            st.warning("OPENAI_API_KEY not set. Required for embeddings.")
        else:
            # Document input
            doc_input_method = st.radio(
                "Document Input Method",
                ["Paste Text", "Upload Files"],
                horizontal=True,
            )

            documents = []
            if doc_input_method == "Paste Text":
                doc_text = st.text_area(
                    "Paste your documents (one per line or separated by blank lines)",
                    height=150,
                    placeholder="Document 1 content here...\n\nDocument 2 content here...",
                )
                if doc_text.strip():
                    # Split by double newlines or treat as single doc
                    documents = [d.strip() for d in doc_text.split("\n\n") if d.strip()]
            else:
                uploaded_files = st.file_uploader(
                    "Upload text files",
                    type=["txt", "md"],
                    accept_multiple_files=True,
                )
                for file in uploaded_files:
                    content = file.read().decode("utf-8")
                    documents.append(content)

            # Vector store settings
            col1, col2 = st.columns(2)
            with col1:
                collection_name = st.text_input(
                    "Collection Name",
                    value="traceflow_docs",
                    help="Name for the vector store collection",
                )
            with col2:
                top_k = st.slider("Top K Results", 1, 10, 5, help="Number of chunks to retrieve")

            # Initialize/update vector store
            if documents:
                st.info(f"{len(documents)} document(s) ready")

                if st.button("Create/Update Vector Store"):
                    try:
                        db_path = "./chroma_db"
                        retriever = get_retriever(collection_name, db_path)
                        
                        # Create progress bar
                        progress_bar = st.progress(0, text="Creating embeddings...")
                        
                        def update_progress(current, total):
                            progress = current / total
                            progress_bar.progress(
                                progress,
                                text=f"Processing batch {current}/{total}..."
                            )
                        
                        retriever.create_vector_store(
                            documents,
                            progress_callback=update_progress
                        )
                        
                        progress_bar.progress(1.0, text="Complete!")
                        st.session_state["rag_ready"] = True
                        st.session_state["rag_collection"] = collection_name
                        st.success(
                            f"Vector store created with {len(documents)} documents!"
                        )
                    except Exception as e:
                        st.error(f"Failed to create vector store: {e}")

            # Check if RAG is ready
            if st.session_state.get("rag_ready") and st.session_state.get(
                "rag_collection"
            ) == collection_name:
                st.success("RAG is ready! Retriever will be used in queries.")
                retriever = get_retriever(collection_name, "./chroma_db")
                retriever_fn = retriever.retrieve_similar_docs
            elif st.session_state.get("rag_collection"):
                # Collection exists from previous session
                try:
                    retriever = get_retriever(collection_name, "./chroma_db")
                    # Test if collection has data
                    if retriever.collection.count() > 0:
                        st.success(
                            f"Using existing collection '{collection_name}' "
                            f"({retriever.collection.count()} chunks)"
                        )
                        retriever_fn = retriever.retrieve_similar_docs
                        st.session_state["rag_ready"] = True
                        st.session_state["rag_collection"] = collection_name
                except Exception:
                    pass

    if st.button("Execute Run", use_container_width=True, type="primary"):
        if not user_input.strip():
            st.error("Please enter a user input.")
            return

        # Validate RAG setup if enabled
        if enable_rag and retriever_fn is None:
            st.error(
                "RAG is enabled but vector store is not ready. "
                "Please create the vector store first."
            )
            return

        with st.spinner("Running workflow..."):
            client = get_client()
            config = RunConfig(
                mode=mode,
                provider=provider,
                model=model,
                strictness=strictness,
                max_tokens=max_tokens,
                max_cost_usd=max_cost,
                max_latency_ms=max_latency,
                max_revisions=max_revisions,
                enable_cache=enable_cache,
                retriever_fn=retriever_fn,
                top_k=top_k if enable_rag else 5,
            )

            result = client.run(user_input, config)

        # Display result
        st.divider()

        if result.status.value == "done":
            st.success("Run completed successfully!")
        else:
            st.error(f"Run failed: {result.err}")

        # Result card
        st.markdown("### Result")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">Status</div>
                <div class="metric-value">{result.status.value.upper()}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with col2:
            eval_decision = result.eval_decision.decision.value if result.eval_decision else "—"
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">Eval Decision</div>
                <div class="metric-value">{eval_decision.upper()}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">Mode</div>
                <div class="metric-value" style="font-size: 0.9rem;">{mode.value}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">Trace ID</div>
                <div class="metric-value" style="font-size: 0.7rem;">{result.trace_id[:12]}...</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("### Answer")
        st.markdown(
            f"""
        <div class="trace-card">
            {result.answer if result.answer else "<em>No answer generated</em>"}
        </div>
        """,
            unsafe_allow_html=True,
        )

        if result.eval_decision and result.eval_decision.reasons:
            st.markdown("### Eval Reasons")
            for reason in result.eval_decision.reasons:
                st.info(reason)

        # Store for viewing details
        st.session_state["last_trace_id"] = result.trace_id


def render_trace_history_page():
    """Render the trace history page."""
    st.markdown("## Trace History")
    st.markdown("View and inspect previous agent runs.")

    client = get_client()

    # Detail view - show at TOP if a trace is selected
    if "selected_trace_id" in st.session_state:
        render_trace_detail(st.session_state["selected_trace_id"])
        st.divider()
        st.markdown("### All Traces")

    traces = client.list_traces(limit=50)

    if not traces:
        st.info("No traces found. Run your first workflow!")
        return

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox("Filter by Status", ["All", "done", "failed", "running"])
    with col2:
        mode_filter = st.selectbox("Filter by Mode", ["All"] + [m.value for m in Mode])
    with col3:
        search = st.text_input("Search", placeholder="Search user input...")

    # Filter traces
    filtered_traces = traces
    if status_filter != "All":
        filtered_traces = [t for t in filtered_traces if t.status.value == status_filter]
    if mode_filter != "All":
        filtered_traces = [t for t in filtered_traces if t.mode.value == mode_filter]
    if search:
        filtered_traces = [t for t in filtered_traces if search.lower() in t.user_input.lower()]

    st.markdown(f"**Showing {len(filtered_traces)} traces**")

    # Trace list
    for trace in filtered_traces:
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.markdown(
                    f"""
                <div class="trace-card">
                    <div style="display: flex; gap: 8px; margin-bottom: 8px;">
                        {get_status_badge(trace.status)}
                        {get_mode_badge(trace.mode)}
                        <span style="color: rgba(255,255,255,0.4); font-size: 0.75rem;">
                            {trace.model} • {format_timestamp(trace.created_at)}
                        </span>
                    </div>
                    <div style="color: rgba(255,255,255,0.8); margin-bottom: 8px;">
                        {trace.user_input[:100]}{"..." if len(trace.user_input) > 100 else ""}
                    </div>
                    <div style="color: rgba(255,255,255,0.4); font-size: 0.75rem;">
                        ID: {trace.trace_id[:16]}...
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                if st.button("Details", key=f"view_{trace.trace_id}"):
                    st.session_state["selected_trace_id"] = trace.trace_id
                    st.rerun()

            with col3:
                if st.button("Replay", key=f"replay_{trace.trace_id}"):
                    with st.spinner("Replaying..."):
                        result = client.replay(trace.trace_id)
                        st.session_state["last_trace_id"] = result.trace_id
                        st.success(f"Replayed! New trace: {result.trace_id[:12]}...")


def render_trace_detail(trace_id: str):
    """Render detailed view of a trace."""
    st.divider()
    st.markdown("## Trace Detail")

    client = get_client()
    trace = client.get_trace(trace_id)

    if not trace:
        st.error("Trace not found.")
        return

    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            f"""
        <div style="display: flex; gap: 12px; align-items: center; margin-bottom: 1rem;">
            {get_status_badge(trace.status)}
            {get_mode_badge(trace.mode)}
            <span style="color: rgba(255,255,255,0.6);">{trace.model}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        if st.button("Close"):
            del st.session_state["selected_trace_id"]
            st.rerun()

    # Metadata
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Trace ID</div>
            <div style="font-size: 0.65rem; color: #a5b4fc; word-break: break-all;">{trace.trace_id}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Provider</div>
            <div class="metric-value">{trace.provider}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Created</div>
            <div style="font-size: 0.8rem; color: #a5b4fc;">{format_timestamp(trace.created_at)}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Finished</div>
            <div style="font-size: 0.8rem; color: #a5b4fc;">{format_timestamp(trace.finished_at)}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Input/Output
    st.markdown("### Input")
    st.code(trace.user_input, language=None)

    st.markdown("### Output")
    if trace.final_answer:
        st.markdown(trace.final_answer)
    else:
        st.warning("No output generated.")

    if trace.error:
        st.markdown("### Error")
        st.error(trace.error)

    # Steps timeline
    st.markdown("### Execution Steps")
    steps = client.dbStore.get_steps(trace_id)

    if steps:
        # Step metrics
        total_tokens = sum(s.tokens for s in steps)
        total_cost = sum(s.cost_usd for s in steps)
        total_latency = sum(s.latency_ms for s in steps)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Steps", len(steps))
        with col2:
            st.metric("Total Tokens", f"{total_tokens:,}")
        with col3:
            st.metric("Total Cost", f"${total_cost:.4f}")
        with col4:
            st.metric("Total Latency", f"{total_latency:.0f}ms")

        # Steps list
        for step in steps:
            cache_badge = " [Cached]" if step.cache_hit else ""

            with st.expander(
                f"**{step.step_seq + 1}. {step.node_name.upper()}**{cache_badge} — {step.latency_ms:.0f}ms"
            ):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tokens", step.tokens)
                with col2:
                    st.metric("Cost", f"${step.cost_usd:.6f}")
                with col3:
                    st.metric("Latency", f"{step.latency_ms:.2f}ms")

                if step.input_data:
                    st.markdown("**Input:**")
                    st.json(step.input_data)

                if step.output_data:
                    st.markdown("**Output:**")
                    st.json(step.output_data)

                if step.error:
                    st.error(f"Error: {step.error}")
    else:
        st.info("No execution steps recorded.")


def render_analytics_page():
    """Render analytics dashboard."""
    st.markdown("## Analytics")
    st.markdown("Insights into your agent workflows.")

    client = get_client()
    traces = client.list_traces(limit=100)

    if not traces:
        st.info("No data yet. Run some workflows first!")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    total = len(traces)
    done = sum(1 for t in traces if t.status.value == "done")
    failed = sum(1 for t in traces if t.status.value == "failed")

    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value">{total}</div>
            <div class="metric-label">Total Runs</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #10b981;">{done}</div>
            <div class="metric-label">Successful</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #ef4444;">{failed}</div>
            <div class="metric-label">Failed</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        success_rate = (done / total * 100) if total > 0 else 0
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value">{success_rate:.1f}%</div>
            <div class="metric-label">Success Rate</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.divider()

    # Mode breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Runs by Mode")
        mode_counts = {}
        for t in traces:
            mode = t.mode.value
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

        for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            st.markdown(
                f"""
            <div style="margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span>{mode}</span>
                    <span>{count} ({pct:.0f}%)</span>
                </div>
                <div style="background: rgba(255,255,255,0.1); border-radius: 4px; height: 8px;">
                    <div style="background: linear-gradient(90deg, #6366f1, #a855f7); width: {pct}%; height: 100%; border-radius: 4px;"></div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("### Runs by Provider")
        provider_counts = {}
        for t in traces:
            provider = t.provider
            provider_counts[provider] = provider_counts.get(provider, 0) + 1

        for provider, count in sorted(provider_counts.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            st.markdown(
                f"""
            <div style="margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span>{provider}</span>
                    <span>{count} ({pct:.0f}%)</span>
                </div>
                <div style="background: rgba(255,255,255,0.1); border-radius: 4px; height: 8px;">
                    <div style="background: linear-gradient(90deg, #10b981, #059669); width: {pct}%; height: 100%; border-radius: 4px;"></div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Recent activity
    st.divider()
    st.markdown("### Recent Activity")

    for trace in traces[:10]:
        st.markdown(
            f"""
        <div style="display: flex; gap: 12px; align-items: center; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
            {get_status_badge(trace.status)}
            <span style="flex: 1; color: rgba(255,255,255,0.8);">{trace.user_input[:60]}{"..." if len(trace.user_input) > 60 else ""}</span>
            <span style="color: rgba(255,255,255,0.4); font-size: 0.75rem;">{format_timestamp(trace.created_at)}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )


def main():
    """Main application entry point."""
    page = render_sidebar()

    if page == "New Run":
        render_new_run_page()
    elif page == "Trace History":
        render_trace_history_page()
    elif page == "Analytics":
        render_analytics_page()


if __name__ == "__main__":
    main()
