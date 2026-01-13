from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable, Optional
from datetime import datetime, timezone


class Mode(StrEnum):
    """
    Selects the TraceFlow workflow + output contract + eval gates.
    """

    GROUNDED_QA = "grounded_qa"
    TRIAGE_PLAN = "triage_plan"
    CHANGE_SAFETY = "change_safety"


class RunStatus(StrEnum):
    """
    Overall run status (not the eval decision).
    """

    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Strictness(StrEnum):
    """
    Controls how hard the eval gate is and how verbose/structured outputs must be.
    """

    LENIENT = "lenient"
    BALANCED = "balanced"
    STRICT = "strict"


class EvalDecision(StrEnum):
    """
    Possible eval decisions for a TraceFlow run.
    """

    PASS = "pass"
    REVISE = "revise"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class RunConfig:
    """
    Configuration for a TraceFlow run.
    """

    mode: Mode = Mode.GROUNDED_QA
    strictness: Strictness = Strictness.BALANCED

    # Reliability
    model: str = "gpt-3.5-turbo"
    provider: str = "openai"
    max_revisions: int = 3
    max_latency_ms: int | None = 30000
    temperature: float | None = 0.2
    top_k: int | None = 5
    enable_cache: bool = True
    max_tokens: int = 1024

    # budget
    max_cost_usd: float | None = 1.50

    # RAG
    retriever_fn: Callable[[str], list[RetrievedChunk]] | None = None


@dataclass
class RetrievedChunk:
    chunk_id: str
    content: str
    source: str
    relevance_score: float
    metadata: str | None = None


@dataclass(frozen=True)
class EvalSummary:
    """
    Summary of eval results for a TraceFlow run.
    """

    decision: EvalDecision
    reasons: list[str]
    scores: dict[str, float]


@dataclass(frozen=True)
class TelemetrySummary:
    """
    Summary of telemetry data for a TraceFlow run.
    """

    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    total_latency_ms: int = 0
    model: str = ""
    provider: str = ""
    total_cost_usd: float = 0.0

    step_count: int = 0
    revision_count: int = 0


@dataclass(frozen=True)
class RunResult:
    """
    Result of a TraceFlow run.
    """

    trace_id: str
    status: RunStatus
    mode: Mode
    answer: str
    eval_decision: Optional[EvalSummary] = None
    telemetry: TelemetrySummary = TelemetrySummary()
    err: Optional[str] = None


# Persistence records (for SQLite)
@dataclass
class TraceRecord:
    trace_id: str
    user_input: str
    config: RunConfig
    mode: Mode
    status: RunStatus = RunStatus.RUNNING
    model: str = "gpt-3.5-turbo"
    provider: str = "openai"
    final_answer: str | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None


@dataclass
class StepRecord:
    trace_id: str
    step_seq: int
    node_name: str
    input_data: dict | None = None
    output_data: dict | None = None
    tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cache_hit: bool = False


@dataclass
class EvalRecord:
    trace_id: str
    step_seq: int
    decision: EvalDecision
    reasons: list[str] | None = None
    scores: dict[str, float] | None = None
    revision_instructions: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
