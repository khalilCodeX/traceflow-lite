%% TraceFlow Lite — Architecture (Control Plane + SDK)
%% Save as: docs/architecture.mmd  (or .md with ```mermaid blocks)
%% Mermaid supports multiple diagrams per file; keep each diagram separated.

```mermaid
flowchart LR
  %% =========================
  %% 1) BOX DIAGRAM ARCHITECTURE
  %% =========================

  subgraph APP["User App / Service (Developer)"]
    U["Developer Code<br/>calls TraceFlow SDK"]
    BYO["Optional: Existing Agent<br/>(callable / workflow)"]
  end

  subgraph TF["TraceFlow Lite SDK + Engine (Control Plane)"]
    SDK["TraceFlow SDK<br/>(TraceFlowClient)"]
    ENG["TraceFlow Engine<br/>(LangGraph Orchestrator)"]

    subgraph NODES["LangGraph Nodes"]
      N1["1) Intake / Normalizer"]
      N2["2) Planner"]
      N3["3) Retriever (RAG)"]
      N4["4) Executor"]
      N5["5) Eval Gate (Critic)"]
      N6["6) Router / Recovery"]
    end

    PROV["Provider Abstraction<br/>(multi-LLM, retry, fallback)"]
    CACHE["Cache Layer<br/>(SQLite cache table)"]
    STORE["Trace Persistence<br/>(SQLite traces/steps/evals)"]
    KB["Vector Store<br/>(Chroma local)"]
  end

  subgraph UI["Ops UI (Streamlit)"]
    CHAT["Chat"]
    TV["Trace Viewer"]
    ED["Eval Dashboard"]
    RP["Replay"]
    SET["Settings"]
  end

  %% Developer usage
  U -->|"run(task)"| SDK --> ENG
  BYO -. optional adapter .-> SDK

  %% Engine orchestration
  ENG --> N1 --> N2
  N2 -->|needs_context| N3 --> KB
  N2 -->|no_context| N4
  N3 --> N4

  %% Execution + gating + recovery
  N4 --> PROV --> CACHE
  N4 --> N5
  N5 -->|PASS| ENDNODE((END))
  N5 -->|REVISE| N4
  N5 -->|"FALLBACK/ERROR"| N6 --> PROV
  N6 -->|safe-mode| ENDNODE

  %% Persistence + UI reads
  ENG --> STORE
  UI --> STORE
  UI --> KB
```

  %% =========================
  %% 2) SEQUENCE DIAGRAM: "SDK RUN" FLOW
  %% =========================

```mermaid
sequenceDiagram
  autonumber
  actor Dev as Developer
  participant SDK as TraceFlowClient (SDK)
  participant Eng as TraceFlow Engine (LangGraph)
  participant TS as TraceStore (SQLite)
  participant VS as VectorStore (Chroma)
  participant LLM as Provider Router (LLM Abstraction)
  participant EG as Eval Gate
  participant UI as Streamlit UI

  Dev->>SDK: run(user_input, mode, settings[, agent_adapter])
  SDK->>TS: create_trace(trace_id, mode, inputs, settings)

  SDK->>Eng: invoke(graph, initial_state)
  Eng->>Eng: Intake/Normalize
  Eng->>Eng: Planner

  alt needs_context = true
    Eng->>VS: retrieve(query, top_k)
    VS-->>Eng: chunks + metadata (C1..Ck)
  else needs_context = false
    Eng->>Eng: skip retrieval
  end

  Eng->>LLM: generate(draft) via provider abstraction
  LLM-->>Eng: draft + tokens/cost/latency

  Eng->>EG: eval_gate(draft, mode, strictness)
  EG-->>Eng: decision PASS/REVISE/FALLBACK + reasons

  alt decision = REVISE and loops_remaining
    Eng->>LLM: regenerate(with revision instructions)
    LLM-->>Eng: revised draft
    Eng->>EG: eval_gate(revised)
    EG-->>Eng: PASS/REVISE/FALLBACK
  end

  alt decision = FALLBACK or exception
    Eng->>LLM: retry/backoff or switch provider/model
    LLM-->>Eng: fallback draft OR error
  end

  Eng->>TS: append_step traces/steps/evals (timeline)
  Eng->>TS: finalize_trace(status, totals, final_answer)

  SDK-->>Dev: result(final_answer, trace_id, eval_report, telemetry)

  Dev->>UI: open Trace Viewer / Replay
  UI->>TS: load(trace_id)
  TS-->>UI: steps + evals + telemetry
  UI-->>Dev: timeline + replay controls
