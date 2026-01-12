# TraceFlow Lite

**A control plane that makes agentic systems shippable: tracing + eval gates + fallback + budgets + replay.**
**A production-grade SDK for building reliable, observable AI agent workflows.**

TraceFlow Lite wraps your LLM calls with enterprise-ready reliability mechanisms: eval gates, cost/latency constraints, provider abstraction, trace persistence, and replay capabilities — all orchestrated through LangGraph.

---

## Why TraceFlow Lite?

Building reliable AI agents is hard. You need to handle:
- **Cost blowouts** — A single runaway query can drain your budget
- **Latency spikes** — Users abandon slow responses
- **Quality inconsistency** — LLMs hallucinate and go off-topic
- **Debugging nightmares** — "What prompt caused that output?"
- **Provider lock-in** — Switching from OpenAI to Anthropic shouldn't require rewrites

TraceFlow Lite solves these by providing a **control plane** that sits between your application and LLM providers.

---

## Features

| Feature | Description |
|---------|-------------|
| 🔄 **LangGraph Orchestration** | Multi-step agent workflow with conditional routing and revision loops |
| 🛡️ **Eval Gates** | Automatic cost, latency, and quality checks before responses are finalized |
| 💰 **Cost Tracking** | Per-request token counting via tiktoken with USD cost calculation |
| 🔁 **Retry & Revision** | Tenacity-powered retries + intelligent revision loop for quality |
| 📊 **Trace Persistence** | SQLite storage (WAL mode) for debugging, analytics, and replay |
| 🔌 **Pluggable Retriever** | Bring your own RAG with flexible callback interface |
| 🏭 **Provider Abstraction** | Easily swap or add LLM providers without code changes |

---

## Architecture

![TraceFlow Lite Architecture](./Architecture/traceflow-workflow.png)

### Workflow Overview

1. **Client** receives a user query and configuration
2. **Intake Node** extracts and validates the input
3. **Planner Node** decides if context retrieval is needed
4. **Retriever Node** (optional) fetches relevant documents via your RAG callback
5. **Executor Node** calls the LLM provider with retry logic
6. **Evaluator Node** checks cost/latency constraints and quality
7. **Router Node** directs traffic based on eval decision:
   - `PASS` → Return final answer
   - `REVISE` → Loop back to executor with refinement instructions
   - `FALLBACK` → Return graceful fallback message

All interactions are traced to SQLite for debugging and replay.

---

## Installation

```bash
git clone https://github.com/your-repo/traceflow-lite.git
cd traceflow-lite
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Dependencies

```bash
pip install langgraph openai tiktoken tenacity pydantic python-dotenv opentelemetry-sdk
# Optional for RAG:
pip install chromadb langchain-text-splitters
```

---

## Quick Start

```python
from client import TraceFlowClient
from tf_types import RunConfig, Mode

client = TraceFlowClient()

# Basic usage with defaults
result = client.run("What is machine learning?")
print(result.answer)

# With custom configuration
config = RunConfig(
    mode=Mode.GROUNDED_QA,
    model="gpt-3.5-turbo",
    max_tokens=500,
    max_cost_usd=0.10,      # Budget limit
    max_latency_ms=10000,   # Latency limit
    max_revisions=2         # Max retry attempts
)
result = client.run("Explain neural networks", config)

print(f"Answer: {result.answer}")
print(f"Status: {result.status}")
print(f"Trace ID: {result.trace_id}")
```

---

## Using a Custom Retriever (RAG)

TraceFlow Lite doesn't lock you into a specific vector database. Provide your own retriever function:

```python
from tf_types import RunConfig, RetrievedChunk

def my_retriever(query: str) -> list[RetrievedChunk]:
    # Your Chroma/Pinecone/Weaviate/custom implementation
    return [
        RetrievedChunk(
            chunk_id="doc_1",
            content="Relevant context here...",
            source="knowledge_base",
            relevance_score=0.95
        )
    ]

config = RunConfig(retriever_fn=my_retriever)
result = client.run("Question needing context", config)
```

### Using the Built-in Chroma Helper

```python
from utils.retriever_utils import chroma_retriever
from utils.vector_types import chroma_params

# Setup retriever with your documents
documents = [
    "AI is the simulation of human intelligence by machines.",
    "Machine learning is a subset of AI that learns from data.",
]

params = chroma_params(
    documents=documents,
    collection="my_knowledge_base",
    directory="./chroma_db"
)

retriever = chroma_retriever(local=True, params=params)
retriever.create_vector_store(documents)

# Use in your runs
config = RunConfig(retriever_fn=retriever.retrieve_similar_docs)
result = client.run("What is AI?", config)
```

---

## Trace Persistence & Replay

Every run is automatically persisted. Debug issues or experiment with different configs:

```python
# List recent traces
traces = client.list_traces(limit=10)
for trace in traces:
    print(f"{trace.trace_id}: {trace.user_input[:50]}...")

# Get a specific trace
trace = client.get_trace("abc123...")
print(trace.final_answer)

# Replay with different configuration
result = client.replay(
    trace_id="abc123...",
    overrides=RunConfig(model="gpt-4o", max_tokens=1000)
)
```

---

## Configuration Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `Mode` | `GROUNDED_QA` | Workflow mode (GROUNDED_QA, TRIAGE_PLAN, CHANGE_SAFETY) |
| `model` | `str` | `gpt-3.5-turbo` | LLM model identifier |
| `provider` | `str` | `openai` | Provider name |
| `max_tokens` | `int` | `1024` | Max output tokens |
| `temperature` | `float` | `0.2` | Sampling temperature |
| `max_cost_usd` | `float` | `1.50` | Budget limit per run (USD) |
| `max_latency_ms` | `int` | `30000` | Latency limit (milliseconds) |
| `max_revisions` | `int` | `3` | Max revision attempts before fallback |
| `strictness` | `Strictness` | `BALANCED` | Eval gate strictness (LENIENT, BALANCED, STRICT) |
| `retriever_fn` | `Callable` | `None` | Custom retriever callback for RAG |
| `enable_cache` | `bool` | `True` | Enable response caching (roadmap) |

---

## Project Structure

```
traceflow-lite/
├── client.py                 # Public SDK entrypoint
├── tf_types.py               # Types, enums, dataclasses
├── state.py                  # Pydantic state models
│
├── graph_flow/
│   ├── graph.py              # LangGraph workflow definition
│   └── nodes/
│       └── nodes.py          # Node implementations
│
├── providers/
│   ├── base.py               # BaseProvider ABC
│   ├── openai_provider.py    # OpenAI implementation
│   ├── router.py             # Provider factory
│   ├── cost.py               # Token counting & pricing
│   └── retry.py              # Tenacity retry decorator
│
├── persistence/
│   ├── sqlite.py             # DB connection & schema
│   └── trace_store.py        # CRUD operations
│
├── utils/
│   ├── retriever_utils.py    # Chroma helper
│   └── vector_types.py       # Retriever config types
│
└── tests/
    └── test_client.py        # Integration tests
```

---

## Environment Variables

Create a `.env` file in your project root:

```env
OPENAI_API_KEY=sk-...

# Optional for Chroma Cloud:
CHROMA_API_KEY=...
CHROMA_TENANT=...
CHROMA_DATABASE=...
```

---

## Running Tests

```bash
# Run all tests
pytest tests/test_client.py -v

# Run specific test
pytest tests/test_client.py::test_basic_run_without_retriever -v
```

---

## Roadmap

- [ ] Multi-provider fallback (Anthropic, Gemini)
- [ ] Response caching implementation
- [ ] Streamlit ops dashboard
- [ ] CLI tool (`traceflow run "query"`)
- [ ] Advanced evaluators (relevance scoring, citation validation)
- [ ] Async execution support
- [ ] OpenTelemetry export integration

---

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit PRs.

---

## License

Apache 2.0
