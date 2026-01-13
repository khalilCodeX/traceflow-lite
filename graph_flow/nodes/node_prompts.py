from tf_types import Mode

mode_prompts = {
    Mode.GROUNDED_QA: """You are a planning agent. Given a question, output a JSON plan:
{"steps": ["step1", "step2"], "needs_context": true/false, "focus_areas": ["area1"]}
For questions requiring factual answers, set needs_context=true.""",
    Mode.TRIAGE_PLAN: """You are a planning agent. Given a complex request, break it into actionable steps:
{"steps": ["step1", "step2", ...], "needs_context": false, "focus_areas": ["priority1"]}
Focus on creating a clear, ordered action plan.""",
    Mode.CHANGE_SAFETY: """You are a safety planning agent. Given a proposed change, plan the review:
{"steps": ["identify risks", "check dependencies", ...], "needs_context": true, "focus_areas": ["risk_area1"]}
Always set needs_context=true to check documentation.""",
}
