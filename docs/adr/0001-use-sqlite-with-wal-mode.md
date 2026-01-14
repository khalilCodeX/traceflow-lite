# ADR-0001: Use SQLite with WAL Mode for Persistence

## Status

Accepted

## Context

TraceFlow Lite needs a persistence layer to store traces, steps, evaluations, and cached LLM responses. The system is designed as a lightweight, single-node observability tool that developers can run locally or in small-scale deployments.

Requirements:
- Zero external dependencies (no separate database server)
- ACID compliance for data integrity
- Support for concurrent reads during writes (Streamlit UI + agent execution)
- Simple deployment and backup (single file)
- Good performance for read-heavy workloads (trace replay, analytics)

Options considered:
1. **PostgreSQL/MySQL** - Full-featured but requires external server
2. **SQLite (default journal mode)** - Simple but blocks readers during writes
3. **SQLite with WAL mode** - Simple with concurrent read/write support
4. **File-based JSON/JSONL** - No schema, poor query performance
5. **Redis** - In-memory, requires external server

## Decision

Use SQLite with Write-Ahead Logging (WAL) mode enabled.

Configuration:
```python
connection = sqlite3.connect(
    db_path,
    check_same_thread=False  # Required for Streamlit's threading model
)
connection.execute("PRAGMA journal_mode=WAL")
connection.execute("PRAGMA busy_timeout=5000")
```

Schema design:
- `traces` - Parent records for agent runs
- `steps` - Individual LLM calls within a trace
- `evals` - Evaluation results linked to steps
- `llm_cache` - Cached LLM responses keyed by hash

## Consequences

### Positive

- **Zero infrastructure** - No database server to install, configure, or maintain
- **Portable** - Single `.db` file can be copied, backed up, or shared
- **Concurrent access** - WAL mode allows reads during writes (critical for live UI)
- **ACID compliance** - Full transaction support with rollback capability
- **Fast reads** - Excellent performance for the read-heavy trace viewer
- **Python stdlib** - No additional dependencies beyond `sqlite3`

### Negative

- **Single-node only** - Cannot scale horizontally (acceptable for target use case)
- **Write throughput** - Lower than dedicated databases (sufficient for observability)
- **No built-in replication** - Manual backup required for durability
- **Threading complexity** - Requires `check_same_thread=False` for Streamlit

### Neutral

- WAL mode creates additional `-wal` and `-shm` files alongside the database
- Database file grows monotonically; periodic `VACUUM` may be needed

## References

- [SQLite WAL Mode Documentation](https://www.sqlite.org/wal.html)
- [SQLite Threading Modes](https://www.sqlite.org/threadsafe.html)
- [Streamlit Threading Model](https://docs.streamlit.io/library/api-reference/performance/st.cache_data)
