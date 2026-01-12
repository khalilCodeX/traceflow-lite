from pydantic import BaseModel
from tf_types import RunConfig, EvalDecision, RetrievedChunk    

class TaskSpec(BaseModel):
    user_input: str
    constraints: dict[str, str] | None = None

class Plan(BaseModel):
    steps: list[str]
    needs_context: bool = False
    focus_areas: list[str] | None = None

class Draft(BaseModel):
    content: str
    citations: list[str]
    model: str
    token_count: int
    latency_ms: float
    cost_usd: float = 0.0
    
class EvalReport(BaseModel):
    decision: EvalDecision
    reasons: list[str]
    scores: dict[str, float]
    revision_instructions: str | None = None
    citation_coverage_ok: bool = True
    schema_valid: bool = True

class TraceFlowState(BaseModel):
    trace_id: str
    config: RunConfig
    task_spec: TaskSpec | None = None
    plan: Plan | None = None
    draft: Draft | None = None
    eval_report: EvalReport | None = None
    error: str | None = None
    final_answer: str | None = None
    context: list[RetrievedChunk] = []
    revisions: int = 0