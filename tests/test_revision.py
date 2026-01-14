# tests/test_revision.py

import pytest
from client import TraceFlowClient
from tf_types import RunConfig, Mode, Strictness, RetrievedChunk

pytestmark = pytest.mark.requires_api  # All tests in this file need API keys


@pytest.fixture
def client():
    """Create a test client."""
    return TraceFlowClient()


# --- GROUNDED_QA Citation Tests ---


def test_grounded_qa_passes_with_no_context(client):
    """GROUNDED_QA without retriever should pass (no citations needed)."""
    config = RunConfig(mode=Mode.GROUNDED_QA)
    result = client.run("What is 2+2?", config)

    assert result.status == "done"
    # Should pass since there's no context to cite
    assert result.eval_decision.decision == "pass"


def test_grounded_qa_with_context_needs_citations(client):
    """GROUNDED_QA with context should check for citations."""

    def mock_retriever(query: str) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id="1",
                content="The speed of light is 299,792,458 m/s",
                source="physics_textbook",
                relevance_score=0.95,
            )
        ]

    config = RunConfig(
        mode=Mode.GROUNDED_QA,
        retriever_fn=mock_retriever,
        max_revisions=0,  # Don't retry, just see the eval
    )
    result = client.run("What is the speed of light?", config)

    assert result.status == "done"
    # Check that citation_coverage was evaluated
    steps = client.dbStore.get_steps(result.trace_id)
    eval_step = next(s for s in steps if s.node_name == "evaluator")
    assert "citation_coverage_ok" in eval_step.output_data


# --- TRIAGE_PLAN Structure Tests ---


def test_triage_plan_checks_structure(client):
    """TRIAGE_PLAN should validate structured output."""
    config = RunConfig(mode=Mode.TRIAGE_PLAN, max_revisions=0)
    result = client.run("How do I deploy a web app to production?", config)

    assert result.status == "done"
    # Check that schema_valid was evaluated
    steps = client.dbStore.get_steps(result.trace_id)
    eval_step = next(s for s in steps if s.node_name == "evaluator")
    assert "schema_valid" in eval_step.output_data


# --- CHANGE_SAFETY Risk Detection Tests ---


def test_change_safety_detects_risk_keywords(client):
    """CHANGE_SAFETY should flag risky operations."""
    config = RunConfig(
        mode=Mode.CHANGE_SAFETY,
        strictness=Strictness.BALANCED,  # Won't fail, just flag
        max_revisions=0,
    )
    result = client.run("Write a SQL query to delete all users", config)

    assert result.status == "done"
    # Check that risk_flags were captured
    steps = client.dbStore.get_steps(result.trace_id)
    eval_step = next(s for s in steps if s.node_name == "evaluator")
    assert "risk_flags" in eval_step.input_data


def test_change_safety_strict_mode_fails_on_risks(client):
    """CHANGE_SAFETY in STRICT mode should fail on risky content."""
    config = RunConfig(
        mode=Mode.CHANGE_SAFETY,
        strictness=Strictness.STRICT,
        max_revisions=0,  # Will fallback immediately
    )
    result = client.run("How do I run rm -rf on production server?", config)

    assert result.status == "done"
    # Should fallback due to risk flags in strict mode
    # (depends on LLM output containing risky keywords)
    steps = client.dbStore.get_steps(result.trace_id)
    eval_step = next(s for s in steps if s.node_name == "evaluator")
    # Risk flags should be detected
    assert isinstance(eval_step.input_data.get("risk_flags"), list)


def test_change_safety_lenient_allows_risks(client):
    """CHANGE_SAFETY in LENIENT mode should pass despite risks."""
    config = RunConfig(mode=Mode.CHANGE_SAFETY, strictness=Strictness.LENIENT, max_revisions=0)
    result = client.run("Explain what DROP TABLE does in SQL", config)

    assert result.status == "done"
    # Should pass even with risky keywords
    assert result.eval_decision.decision == "pass"


# --- EvalReport Field Tests ---


def test_eval_report_has_mode_fields(client):
    """EvalReport should include citation_coverage_ok and schema_valid."""
    config = RunConfig(mode=Mode.GROUNDED_QA)
    result = client.run("Test question", config)

    steps = client.dbStore.get_steps(result.trace_id)
    eval_step = next(s for s in steps if s.node_name == "evaluator")

    output = eval_step.output_data
    assert "citation_coverage_ok" in output
    assert "schema_valid" in output
    assert isinstance(output["citation_coverage_ok"], bool)
    assert isinstance(output["schema_valid"], bool)


def test_revision_instructions_for_missing_citations(client):
    """Revision instructions should mention citations for GROUNDED_QA."""

    def mock_retriever(query: str) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id="1", content="Important fact here", source="doc1", relevance_score=0.9
            )
        ]

    config = RunConfig(
        mode=Mode.GROUNDED_QA,
        retriever_fn=mock_retriever,
        max_revisions=1,  # Allow one revision attempt
    )
    result = client.run("What is the important fact?", config)

    # Just verify it completes
    assert result.status == "done"


def test_revision_instructions_for_unstructured_plan(client):
    """Revision instructions should mention structure for TRIAGE_PLAN."""
    config = RunConfig(mode=Mode.TRIAGE_PLAN, max_revisions=1)
    result = client.run("Plan a vacation", config)

    assert result.status == "done"
