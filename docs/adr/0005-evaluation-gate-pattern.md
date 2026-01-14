# ADR-0005: Evaluation Gate Pattern for Quality Control

## Status

Accepted

## Context

LLM outputs are non-deterministic and can produce incorrect, unsafe, or low-quality results. TraceFlow Lite needs a mechanism to:

- Automatically assess output quality before proceeding
- Block unsafe or incorrect outputs from reaching users
- Provide structured feedback for iterative improvement
- Track evaluation metrics for observability

The agent workflow includes execution steps that produce code or text that must meet quality thresholds before being accepted.

Options considered:
1. **No evaluation** - Fast but no quality control
2. **Human-in-the-loop** - High quality but blocks automation
3. **Rule-based validation** - Fast but limited to known patterns
4. **LLM-as-judge** - Flexible but adds latency and cost
5. **Hybrid (rules + LLM)** - Balance of speed and flexibility

## Decision

Implement an evaluation gate pattern using LLM-as-judge with structured output.

Evaluation dimensions:
```python
class EvalResult(TypedDict):
    correctness: float      # 0.0-1.0: Does it solve the problem?
    code_quality: float     # 0.0-1.0: Is it well-structured?
    safety: float           # 0.0-1.0: Is it safe to execute?
    change_safety: float    # 0.0-1.0: Are changes minimal and reversible?
    passed: bool            # Overall pass/fail
    feedback: str           # Specific improvement suggestions
```

Gate logic:
```python
def should_pass(eval_result: EvalResult) -> bool:
    return (
        eval_result["correctness"] >= 0.7 and
        eval_result["safety"] >= 0.8 and
        eval_result["passed"]
    )
```

Workflow integration:
```
executor_node → eval_node → [pass] → END
                    ↓
                 [fail]
                    ↓
              revision_node → planner_node (retry loop)
```

## Consequences

### Positive

- **Quality assurance** - Catches errors before they reach users
- **Structured feedback** - Specific suggestions enable targeted revision
- **Multi-dimensional** - Separate scores for correctness, safety, quality
- **Auditable** - All evaluations stored in `evals` table
- **Iterative improvement** - Failed evals trigger revision loop
- **Configurable thresholds** - Adjust strictness per use case

### Negative

- **Added latency** - Extra LLM call for evaluation
- **Added cost** - Evaluation consumes tokens
- **False negatives** - May reject valid outputs
- **False positives** - May accept flawed outputs
- **Revision loops** - Can get stuck if evaluation is too strict

### Neutral

- Maximum revision attempts configurable (default: 3)
- Evaluation prompts tuned for specific output types
- Feedback quality depends on evaluation model capability

## References

- [LLM-as-Judge Paper](https://arxiv.org/abs/2306.05685)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)
- [OpenAI Evals Framework](https://github.com/openai/evals)
