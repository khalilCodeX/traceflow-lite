import pytest
from client import TraceFlowClient
from tf_types import RunConfig, Mode


@pytest.fixture
def client():
    """Create a test client."""
    return TraceFlowClient()


def test_step_persistence(client):
    """Test that executed steps are saved to database."""
    config = RunConfig(mode=Mode.GROUNDED_QA)
    result = client.run("Test question", config)

    # Get steps for this trace
    steps = client.dbStore.get_steps(result.trace_id)

    assert len(steps) >= 4  # intake, planner, retriever, executor, evaluator

    # Verify step structure
    node_names = [s.node_name for s in steps]
    assert "intake" in node_names
    assert "planner" in node_names
    assert "executor" in node_names
    assert "evaluator" in node_names

    # Verify step sequence
    for i, step in enumerate(steps):
        assert step.step_seq == i


def test_step_has_latency(client):
    """Test that steps record latency."""
    config = RunConfig(mode=Mode.GROUNDED_QA)
    result = client.run("Test question", config)

    steps = client.dbStore.get_steps(result.trace_id)

    # Executor should have non-zero latency
    executor_step = next(s for s in steps if s.node_name == "executor")
    assert executor_step.latency_ms > 0


def test_step_has_tokens_and_cost(client):
    """Test that executor step records tokens and cost."""
    config = RunConfig(mode=Mode.GROUNDED_QA)
    result = client.run("Test question", config)

    steps = client.dbStore.get_steps(result.trace_id)

    executor_step = next(s for s in steps if s.node_name == "executor")
    assert executor_step.tokens > 0
    assert executor_step.cost_usd > 0


def test_grounded_qa_adds_citation_constraint(client):
    """Test that GROUNDED_QA mode adds require_citations constraint."""
    config = RunConfig(mode=Mode.GROUNDED_QA)
    result = client.run("What is AI?", config)

    # Verify trace completed
    assert result.status == "done"

    # Check intake step output for constraint
    steps = client.dbStore.get_steps(result.trace_id)
    intake_step = next(s for s in steps if s.node_name == "intake")
    assert intake_step.output_data.get("constraints", {}).get("require_citations") == "true"


def test_triage_plan_mode(client):
    """Test TRIAGE_PLAN mode adds structured output constraint."""
    config = RunConfig(mode=Mode.TRIAGE_PLAN)
    result = client.run("How do I deploy to production?", config)

    assert result.status == "done"

    steps = client.dbStore.get_steps(result.trace_id)
    intake_step = next(s for s in steps if s.node_name == "intake")
    assert intake_step.output_data.get("constraints", {}).get("output_format") == "structured_plan"


def test_change_safety_mode(client):
    """Test CHANGE_SAFETY mode adds risk assessment constraint."""
    config = RunConfig(mode=Mode.CHANGE_SAFETY)
    result = client.run("DROP TABLE users;", config)

    assert result.status == "done"

    steps = client.dbStore.get_steps(result.trace_id)
    intake_step = next(s for s in steps if s.node_name == "intake")
    assert intake_step.output_data.get("constraints", {}).get("assess_risk") == "true"


def test_planner_creates_plan(client):
    """Test that planner node creates a valid plan."""
    config = RunConfig(mode=Mode.GROUNDED_QA)
    result = client.run("Explain quantum computing", config)

    steps = client.dbStore.get_steps(result.trace_id)
    planner_step = next(s for s in steps if s.node_name == "planner")

    # Verify plan structure in output
    assert "steps" in planner_step.output_data
    assert isinstance(planner_step.output_data["steps"], list)
    assert planner_step.tokens > 0  # LLM was called
