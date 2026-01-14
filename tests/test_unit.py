"""
Mocked unit tests for TraceFlow client, nodes, and routing.
These tests use mocking and don't require API keys - safe for CI.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from tf_types import (
    RunConfig,
    Mode,
    Strictness,
    EvalDecision,
    RetrievedChunk,
    StepRecord,
    TraceRecord,
)
from state import TraceFlowState, TaskSpec, Plan, Draft, EvalReport
from providers.base import ProviderResponse


# --- Mock Fixtures ---


@pytest.fixture
def mock_provider_response():
    """Standard mock LLM response."""
    return ProviderResponse(
        content='{"steps": ["analyze", "respond"], "needs_context": false}',
        input_tokens=50,
        output_tokens=20,
        model="gpt-3.5-turbo",
    )


@pytest.fixture
def mock_executor_response():
    """Mock executor LLM response."""
    return ProviderResponse(
        content="The answer is 4. [1] This is based on basic arithmetic.",
        input_tokens=100,
        output_tokens=30,
        model="gpt-3.5-turbo",
    )


@pytest.fixture
def mock_db_store():
    """Mock database store."""
    store = MagicMock()
    store.create_trace.return_value = None
    store.update_trace.return_value = None
    store.insert_step.return_value = None
    store.get_trace.return_value = None
    store.list_traces.return_value = []
    store.get_steps.return_value = []
    return store


# --- Intake Node Tests ---


class TestIntakeNode:
    """Tests for intake node functionality."""

    def test_intake_validates_empty_input(self):
        """Intake should flag empty user input."""
        from graph_flow.nodes.nodes import intake_node

        state = TraceFlowState(
            trace_id="test-123",
            config=RunConfig(mode=Mode.GROUNDED_QA),
            task_spec=TaskSpec(user_input=""),
        )

        result = intake_node(state)

        assert result["error"] == "User input is empty."

    def test_intake_adds_grounded_qa_constraint(self):
        """GROUNDED_QA mode should add require_citations constraint."""
        from graph_flow.nodes.nodes import intake_node

        state = TraceFlowState(
            trace_id="test-123",
            config=RunConfig(mode=Mode.GROUNDED_QA),
            task_spec=TaskSpec(user_input="What is AI?"),
        )

        result = intake_node(state)

        assert result["task_spec"].constraints.get("require_citations") == "true"

    def test_intake_adds_triage_plan_constraint(self):
        """TRIAGE_PLAN mode should add output_format constraint."""
        from graph_flow.nodes.nodes import intake_node

        state = TraceFlowState(
            trace_id="test-123",
            config=RunConfig(mode=Mode.TRIAGE_PLAN),
            task_spec=TaskSpec(user_input="How to deploy?"),
        )

        result = intake_node(state)

        assert result["task_spec"].constraints.get("output_format") == "structured_plan"

    def test_intake_adds_change_safety_constraint(self):
        """CHANGE_SAFETY mode should add assess_risk constraint."""
        from graph_flow.nodes.nodes import intake_node

        state = TraceFlowState(
            trace_id="test-123",
            config=RunConfig(mode=Mode.CHANGE_SAFETY),
            task_spec=TaskSpec(user_input="DELETE FROM users"),
        )

        result = intake_node(state)

        assert result["task_spec"].constraints.get("assess_risk") == "true"

    def test_intake_records_step(self):
        """Intake should record execution step."""
        from graph_flow.nodes.nodes import intake_node

        state = TraceFlowState(
            trace_id="test-123",
            config=RunConfig(mode=Mode.GROUNDED_QA),
            task_spec=TaskSpec(user_input="Test question"),
        )

        result = intake_node(state)

        assert len(result["executed_steps"]) == 1
        assert result["executed_steps"][0]["node_name"] == "intake"
        assert result["executed_steps"][0]["latency_ms"] >= 0


# --- Evaluator Node Tests ---


class TestEvaluatorNode:
    """Tests for evaluator node functionality."""

    def test_evaluator_passes_valid_draft(self):
        """Evaluator should PASS a valid draft."""
        from graph_flow.nodes.nodes import evaluator_node

        draft = Draft(
            content="The answer is 4.",
            citations=[],
            model="gpt-3.5-turbo",
            token_count=50,
            latency_ms=500,
            cost_usd=0.001,
        )

        state = TraceFlowState(
            trace_id="test-123",
            config=RunConfig(mode=Mode.GROUNDED_QA, max_cost_usd=1.0, max_latency_ms=30000),
            task_spec=TaskSpec(user_input="What is 2+2?"),
            draft=draft,
            context=[],  # No context = no citations needed
        )

        result = evaluator_node(state)

        assert result["eval_report"].decision == EvalDecision.PASS

    def test_evaluator_revises_on_cost_exceeded(self):
        """Evaluator should REVISE when cost exceeds limit."""
        from graph_flow.nodes.nodes import evaluator_node

        draft = Draft(
            content="Expensive answer",
            citations=[],
            model="gpt-4",
            token_count=1000,
            latency_ms=500,
            cost_usd=5.0,  # Exceeds limit
        )

        state = TraceFlowState(
            trace_id="test-123",
            config=RunConfig(
                mode=Mode.GROUNDED_QA,
                max_cost_usd=1.0,  # Limit is $1
                max_revisions=3,
            ),
            task_spec=TaskSpec(user_input="Expensive question"),
            draft=draft,
            context=[],
            revisions=0,
        )

        result = evaluator_node(state)

        assert result["eval_report"].decision in [EvalDecision.REVISE, EvalDecision.FALLBACK]

    def test_evaluator_fallbacks_on_revision_limit(self):
        """Evaluator should FALLBACK when revision limit reached."""
        from graph_flow.nodes.nodes import evaluator_node

        draft = Draft(
            content="Bad answer",
            citations=[],
            model="gpt-3.5-turbo",
            token_count=50,
            latency_ms=50000,  # Exceeds latency
            cost_usd=0.001,
        )

        state = TraceFlowState(
            trace_id="test-123",
            config=RunConfig(mode=Mode.GROUNDED_QA, max_latency_ms=1000, max_revisions=2),
            task_spec=TaskSpec(user_input="Test"),
            draft=draft,
            context=[],
            revisions=2,  # Already at limit
        )

        result = evaluator_node(state)

        assert result["eval_report"].decision == EvalDecision.FALLBACK

    def test_evaluator_checks_citations_for_grounded_qa(self):
        """GROUNDED_QA should check citation coverage when context exists."""
        from graph_flow.nodes.nodes import evaluator_node

        draft = Draft(
            content="The answer without any citations.",
            citations=[],  # No citations!
            model="gpt-3.5-turbo",
            token_count=50,
            latency_ms=500,
            cost_usd=0.001,
        )

        context = [
            RetrievedChunk(
                chunk_id="1",
                content="Important source info",
                source="document.pdf",
                relevance_score=0.9,
            )
        ]

        state = TraceFlowState(
            trace_id="test-123",
            config=RunConfig(mode=Mode.GROUNDED_QA, max_revisions=3),
            task_spec=TaskSpec(user_input="What does the document say?"),
            draft=draft,
            context=context,
            revisions=0,
        )

        result = evaluator_node(state)

        # Should fail citation check and request revision
        assert result["eval_report"].citation_coverage_ok is False

    def test_evaluator_passes_with_citations(self):
        """GROUNDED_QA should pass when citations are present."""
        from graph_flow.nodes.nodes import evaluator_node

        draft = Draft(
            content="According to [1], the answer is correct.",
            citations=["1"],  # Has citation
            model="gpt-3.5-turbo",
            token_count=50,
            latency_ms=500,
            cost_usd=0.001,
        )

        context = [
            RetrievedChunk(
                chunk_id="1", content="Source info", source="doc.pdf", relevance_score=0.9
            )
        ]

        state = TraceFlowState(
            trace_id="test-123",
            config=RunConfig(mode=Mode.GROUNDED_QA, max_cost_usd=1.0, max_latency_ms=30000),
            task_spec=TaskSpec(user_input="Question?"),
            draft=draft,
            context=context,
            revisions=0,
        )

        result = evaluator_node(state)

        assert result["eval_report"].citation_coverage_ok is True

    def test_evaluator_checks_structure_for_triage_plan(self):
        """TRIAGE_PLAN should check for structured output."""
        from graph_flow.nodes.nodes import evaluator_node

        # Unstructured response
        draft = Draft(
            content="Just do it however you want.",
            citations=[],
            model="gpt-3.5-turbo",
            token_count=50,
            latency_ms=500,
            cost_usd=0.001,
        )

        state = TraceFlowState(
            trace_id="test-123",
            config=RunConfig(mode=Mode.TRIAGE_PLAN, max_revisions=3),
            task_spec=TaskSpec(user_input="How to deploy?"),
            draft=draft,
            context=[],
            revisions=0,
        )

        result = evaluator_node(state)

        assert result["eval_report"].schema_valid is False

    def test_evaluator_passes_structured_plan(self):
        """TRIAGE_PLAN should pass with numbered list."""
        from graph_flow.nodes.nodes import evaluator_node

        draft = Draft(
            content="1. First step\n2. Second step\n3. Third step",
            citations=[],
            model="gpt-3.5-turbo",
            token_count=50,
            latency_ms=500,
            cost_usd=0.001,
        )

        state = TraceFlowState(
            trace_id="test-123",
            config=RunConfig(mode=Mode.TRIAGE_PLAN, max_cost_usd=1.0, max_latency_ms=30000),
            task_spec=TaskSpec(user_input="How to deploy?"),
            draft=draft,
            context=[],
            revisions=0,
        )

        result = evaluator_node(state)

        assert result["eval_report"].schema_valid is True

    def test_evaluator_detects_risk_keywords(self):
        """CHANGE_SAFETY should detect risky keywords."""
        from graph_flow.nodes.nodes import evaluator_node

        draft = Draft(
            content="Run rm -rf / to delete everything and DROP TABLE users",
            citations=[],
            model="gpt-3.5-turbo",
            token_count=50,
            latency_ms=500,
            cost_usd=0.001,
        )

        state = TraceFlowState(
            trace_id="test-123",
            config=RunConfig(
                mode=Mode.CHANGE_SAFETY, strictness=Strictness.BALANCED, max_revisions=3
            ),
            task_spec=TaskSpec(user_input="How to clean disk?"),
            draft=draft,
            context=[],
            revisions=0,
        )

        result = evaluator_node(state)

        # Should have detected risk flags
        step = result["executed_steps"][-1]
        assert (
            "rm -rf" in step["input_data"]["risk_flags"]
            or "drop" in step["input_data"]["risk_flags"]
        )

    def test_evaluator_strict_mode_fails_on_risks(self):
        """CHANGE_SAFETY + STRICT should fail on risky content."""
        from graph_flow.nodes.nodes import evaluator_node

        draft = Draft(
            content="DELETE FROM production WHERE 1=1",
            citations=[],
            model="gpt-3.5-turbo",
            token_count=50,
            latency_ms=500,
            cost_usd=0.001,
        )

        state = TraceFlowState(
            trace_id="test-123",
            config=RunConfig(
                mode=Mode.CHANGE_SAFETY,
                strictness=Strictness.STRICT,
                max_revisions=0,  # Immediate fallback
            ),
            task_spec=TaskSpec(user_input="Delete query"),
            draft=draft,
            context=[],
            revisions=0,
        )

        result = evaluator_node(state)

        # Should fallback due to risk + strict mode + no revisions left
        assert result["eval_report"].decision == EvalDecision.FALLBACK


# --- Routing Tests ---


class TestRouting:
    """Tests for graph routing decisions."""

    def test_route_after_evaluator_returns_end_on_pass(self):
        """route_after_evaluator should route to end on PASS."""
        from graph_flow.graph import route_after_evaluator

        state = TraceFlowState(
            trace_id="test-123",
            config=RunConfig(),
            task_spec=TaskSpec(user_input="Test"),
            eval_report=EvalReport(decision=EvalDecision.PASS, reasons=["All good"], scores={}),
        )

        result = route_after_evaluator(state)

        assert result == "end"

    def test_route_after_evaluator_returns_executor_on_revise(self):
        """route_after_evaluator should route to executor on REVISE."""
        from graph_flow.graph import route_after_evaluator

        state = TraceFlowState(
            trace_id="test-123",
            config=RunConfig(max_revisions=3),
            task_spec=TaskSpec(user_input="Test"),
            revisions=0,
            eval_report=EvalReport(
                decision=EvalDecision.REVISE, reasons=["Need improvements"], scores={}
            ),
        )

        result = route_after_evaluator(state)

        assert result == "executor"

    def test_route_after_evaluator_returns_router_on_fallback(self):
        """route_after_evaluator should route to router on FALLBACK."""
        from graph_flow.graph import route_after_evaluator

        state = TraceFlowState(
            trace_id="test-123",
            config=RunConfig(max_revisions=0),
            task_spec=TaskSpec(user_input="Test"),
            revisions=0,
            eval_report=EvalReport(
                decision=EvalDecision.FALLBACK, reasons=["Can't complete"], scores={}
            ),
        )

        result = route_after_evaluator(state)

        assert result == "router"

    def test_route_after_evaluator_fallback_when_max_revisions(self):
        """route_after_evaluator should fallback when max revisions reached."""
        from graph_flow.graph import route_after_evaluator

        state = TraceFlowState(
            trace_id="test-123",
            config=RunConfig(max_revisions=2),
            task_spec=TaskSpec(user_input="Test"),
            revisions=2,  # At max
            eval_report=EvalReport(
                decision=EvalDecision.REVISE, reasons=["Still needs work"], scores={}
            ),
        )

        result = route_after_evaluator(state)

        # Should go to router since we can't revise anymore
        assert result == "router"

    def test_router_after_planner_with_context(self):
        """router_after_planner should go to retriever when context needed."""
        from graph_flow.graph import router_after_planner

        state = TraceFlowState(
            trace_id="test-123",
            config=RunConfig(),
            task_spec=TaskSpec(user_input="Test"),
            plan=Plan(steps=["search", "answer"], needs_context=True),
        )

        result = router_after_planner(state)

        assert result == "retriever"

    def test_router_after_planner_without_context(self):
        """router_after_planner should go to executor when no context needed."""
        from graph_flow.graph import router_after_planner

        state = TraceFlowState(
            trace_id="test-123",
            config=RunConfig(),
            task_spec=TaskSpec(user_input="Test"),
            plan=Plan(steps=["answer"], needs_context=False),
        )

        result = router_after_planner(state)

        assert result == "executor"


# --- Router Node Tests ---


class TestRouterNode:
    """Tests for router (fallback) node."""

    def test_router_returns_fallback_message(self):
        """Router should return a graceful fallback message."""
        from graph_flow.nodes.nodes import router_node

        state = TraceFlowState(
            trace_id="test-123",
            config=RunConfig(mode=Mode.GROUNDED_QA),
            task_spec=TaskSpec(user_input="Test"),
            eval_report=EvalReport(decision=EvalDecision.FALLBACK, reasons=["Failed"], scores={}),
        )

        result = router_node(state)

        assert (
            "sorry" in result["final_answer"].lower()
            or "couldn't" in result["final_answer"].lower()
        )


# --- Provider Router Tests ---


class TestProviderRouter:
    """Tests for provider routing and selection."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_get_provider_returns_openai(self):
        """get_provider should return OpenAI provider for openai config."""
        from providers.router import get_provider
        from providers.openai_provider import OpenAIProvider

        config = RunConfig(provider="openai", enable_cache=False)
        provider = get_provider(config)

        assert isinstance(provider, OpenAIProvider)

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("providers.anthropic_provider.Anthropic")
    def test_get_provider_returns_anthropic(self, mock_anthropic):
        """get_provider should return Anthropic provider for anthropic config."""
        from providers.router import get_provider
        from providers.anthropic_provider import AnthropicProvider

        config = RunConfig(provider="anthropic", enable_cache=False)
        provider = get_provider(config)

        assert isinstance(provider, AnthropicProvider)
        mock_anthropic.assert_called_once()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_get_provider_wraps_with_cache(self):
        """get_provider should wrap with CachedProvider when cache enabled."""
        from providers.router import get_provider
        from providers.cache_provider import CachedProvider

        config = RunConfig(provider="openai", enable_cache=True)

        with patch("providers.cache_provider.LLMCache"):
            provider = get_provider(config)

        assert isinstance(provider, CachedProvider)

    def test_get_provider_raises_on_invalid(self):
        """get_provider should raise ValueError for unknown provider."""
        from providers.router import get_provider

        config = RunConfig(provider="invalid_provider", enable_cache=False)

        with pytest.raises(ValueError, match="Unsupported provider"):
            get_provider(config)


# --- Cost Calculation Tests ---


class TestCostCalculation:
    """Tests for token counting and cost calculation."""

    def test_calculate_cost_openai(self):
        """Cost calculation for OpenAI models."""
        from providers.cost import calculate_cost

        cost = calculate_cost(model="gpt-3.5-turbo", input_tokens=1000, output_tokens=500)

        # gpt-3.5-turbo: $0.50/1M input, $1.50/1M output
        expected = (1000 / 1_000_000) * 0.50 + (500 / 1_000_000) * 1.50
        assert abs(cost - expected) < 0.000001

    def test_calculate_cost_anthropic(self):
        """Cost calculation for Anthropic models."""
        from providers.cost import calculate_cost

        cost = calculate_cost(
            model="claude-3-5-sonnet-20241022", input_tokens=1000, output_tokens=500
        )

        # claude-3-5-sonnet: $3.00/1M input, $15.00/1M output
        expected = (1000 / 1_000_000) * 3.00 + (500 / 1_000_000) * 15.00
        assert abs(cost - expected) < 0.000001

    def test_calculate_cost_unknown_model_returns_zero(self):
        """Unknown model should return $0 with warning."""
        from providers.cost import calculate_cost

        cost = calculate_cost(model="unknown-model-xyz", input_tokens=1000, output_tokens=500)

        assert cost == 0.0

    def test_count_tokens(self):
        """Token counting should return positive count."""
        from providers.cost import count_tokens

        count = count_tokens("Hello, world! This is a test.")

        assert count > 0
        assert isinstance(count, int)


# --- TraceStore Tests ---


class TestTraceStore:
    """Tests for trace persistence."""

    @pytest.fixture
    def mock_connection(self):
        """Create mock SQLite connection."""
        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = None
        conn.execute.return_value.fetchall.return_value = []
        return conn

    def test_create_trace_inserts_record(self, mock_connection):
        """create_trace should INSERT into traces table."""
        with patch("persistence.trace_store.Sqlite") as mock_sqlite:
            mock_sqlite.get_connection.return_value = mock_connection
            mock_sqlite.init_db.return_value = None

            from persistence.trace_store import TraceStore

            store = TraceStore()
            store.conn = mock_connection

            trace = TraceRecord(
                trace_id="test-123",
                user_input="Test question",
                config=RunConfig(),
                mode=Mode.GROUNDED_QA,
                model="gpt-3.5-turbo",
                provider="openai",
            )

            store.create_trace(trace)

            # Verify INSERT was called
            insert_calls = [
                c for c in mock_connection.execute.call_args_list if "INSERT INTO traces" in str(c)
            ]
            assert len(insert_calls) >= 1

    def test_insert_step_includes_cache_hit(self, mock_connection):
        """insert_step should include cache_hit field."""
        with patch("persistence.trace_store.Sqlite") as mock_sqlite:
            mock_sqlite.get_connection.return_value = mock_connection
            mock_sqlite.init_db.return_value = None

            from persistence.trace_store import TraceStore

            store = TraceStore()
            store.conn = mock_connection

            step = StepRecord(
                trace_id="test-123",
                step_seq=0,
                node_name="executor",
                tokens=100,
                cost_usd=0.001,
                latency_ms=500,
                cache_hit=True,
            )

            store.insert_step(step)

            # Verify cache_hit was included
            insert_calls = [
                c
                for c in mock_connection.execute.call_args_list
                if "INSERT INTO trace_steps" in str(c)
            ]
            assert len(insert_calls) >= 1


# --- Retriever Node Tests ---


class TestRetrieverNode:
    """Tests for retriever node."""

    def test_retriever_with_no_fn_returns_empty(self):
        """Retriever without fn should return empty context."""
        from graph_flow.nodes.nodes import retriever_node

        state = TraceFlowState(
            trace_id="test-123",
            config=RunConfig(retriever_fn=None),
            task_spec=TaskSpec(user_input="Test question"),
        )

        result = retriever_node(state)

        assert result["context"] == []

    def test_retriever_calls_fn_and_returns_chunks(self):
        """Retriever should call retriever_fn and return chunks."""
        from graph_flow.nodes.nodes import retriever_node

        mock_chunks = [
            RetrievedChunk(
                chunk_id="1", content="Relevant content", source="doc.pdf", relevance_score=0.95
            )
        ]
        mock_fn = Mock(return_value=mock_chunks)

        state = TraceFlowState(
            trace_id="test-123",
            config=RunConfig(retriever_fn=mock_fn),
            task_spec=TaskSpec(user_input="Test question"),
        )

        result = retriever_node(state)

        mock_fn.assert_called_once_with("Test question")
        assert result["context"] == mock_chunks

    def test_retriever_records_step(self):
        """Retriever should record execution step."""
        from graph_flow.nodes.nodes import retriever_node

        state = TraceFlowState(
            trace_id="test-123",
            config=RunConfig(retriever_fn=None),
            task_spec=TaskSpec(user_input="Test"),
        )

        result = retriever_node(state)

        assert len(result["executed_steps"]) == 1
        assert result["executed_steps"][0]["node_name"] == "retriever"
