from datetime import datetime, timezone
from persistence.trace_store import TraceStore
from tf_types import Mode, RunConfig, RunResult, RunStatus, StepRecord, TraceRecord, EvalSummary
from state import TraceFlowState, TaskSpec
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from graph_flow.graph import build_traceflow_graph

class TraceFlowClient:
    def __init__(self):
        # TODO: Load config, init stores, build graph
        self.dbStore = TraceStore(db_path="traceflow.db")
        self.id_generator = RandomIdGenerator()
        self.graph = build_traceflow_graph()

    def run(self, user_input: str, config: RunConfig | None = None) -> RunResult:
        # TODO: Wire up real graph
        trace_id = self._generate_trace_id()
        config = config or RunConfig()

        self.dbStore.create_trace(TraceRecord(
            trace_id=trace_id,
            user_input=user_input,
            config=config,
            mode=config.mode,
            model=config.model,
            provider=config.provider
        ))

        try:
            initial_state = TraceFlowState(
                trace_id=trace_id,
                config=config,
                task_spec=TaskSpec(user_input=user_input)
            )

            final_state_dict = self.graph.invoke(initial_state)
            final_state = TraceFlowState(**final_state_dict)
            answer = final_state.final_answer or (final_state.draft.content if final_state.draft else "")

            for step_seq, step_data in enumerate(final_state.executed_steps):
                self.dbStore.insert_step(StepRecord(
                    trace_id=trace_id,
                    step_seq=step_seq,
                    node_name=step_data.get("node_name", ""),
                    input_data=step_data.get("input_data"),
                    output_data=step_data.get("output_data"),
                    tokens=step_data.get("tokens", 0),
                    cost_usd=step_data.get("cost_usd", 0.0),
                    latency_ms=step_data.get("latency_ms", 0.0),
                    error=step_data.get("error"),
                    cache_hit=step_data.get("cache_hit", False)
                ))

            self.dbStore.update_trace(
                trace_id=trace_id,
                status=RunStatus.DONE,
                final_answer=answer,
                error=None
            )

            return RunResult(
                trace_id=trace_id,
                status=RunStatus.DONE,
                mode=config.mode,
                answer=answer,
                eval_decision=EvalSummary(
                decision=final_state.eval_report.decision,
                reasons=final_state.eval_report.reasons,
                scores=final_state.eval_report.scores
                ) if final_state.eval_report else None
            )
        
        except Exception as e:
            self.dbStore.update_trace(trace_id, RunStatus.FAILED, error=str(e))
            return RunResult(
                trace_id=trace_id,
                status=RunStatus.FAILED,
                mode=config.mode,
                answer="",
                eval_decision=None,
                err=str(e),
            )

    def replay(self, trace_id: str, overrides: RunConfig | None = None) -> RunResult:
        """Replay a trace with optional config overrides."""
        original_trace_record = self.dbStore.get_trace(trace_id)
        if not original_trace_record:
            raise ValueError(f"Trace with ID {trace_id} not found.")
        
        config = overrides if overrides else original_trace_record.config
        return self.run(original_trace_record.user_input, config)

    def list_traces(self, limit: int = 50) ->list[TraceRecord]:
        return self.dbStore.list_traces(limit=limit)
    
    def get_trace(self, trace_id: str) -> TraceRecord | None:
        return self.dbStore.get_trace(trace_id)
    
    def _generate_trace_id(self) -> str:
        return str(format(self.id_generator.generate_trace_id(), '032x'))
    

def __main__(self):
    # Simple CLI for testing
    client = TraceFlowClient()
    user_input = "Explain the theory of relativity."
    result = client.run(user_input)
    print(f"Run Result: {result}")

if __name__ == "__main__":
    __main__(None)

    

