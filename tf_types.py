from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

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
    model: str | None = None
    provider: str | None = "openai"
    max_revisions: int = 3
    max_latency_ms: int | None = 30000
    temperature: float | None = 0.2
    top_k: int | None = 5
    enable_cache: bool = True

    # budget
    max_cost_usd: float | None = 1.50

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