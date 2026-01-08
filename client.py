from datetime import datetime, timezone
from persistence.trace_store import TraceStore
from tf_types import Mode, RunConfig, RunResult, RunStatus, TraceRecord
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator


class TraceFlowClient:
    def __init__(self):
        # TODO: Load config, init stores, build graph
        self.dbStore = TraceStore(db_path="traceflow.db")
        self.id_generator = RandomIdGenerator()

    def run(self, user_input: str, config: RunConfig | None = None) -> RunResult:
        # TODO: Wire up real graph
        trace_id = self._generate_trace_id()
        config = config or RunConfig()

        self.dbStore.create_trace(TraceRecord(
            trace_id=trace_id,
            user_input=user_input,
            config=config,
            mode=config.mode
        ))

         # TODO: Run the graph here
        answer = "This is a placeholder answer."

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
            answer=answer
        )

    def replay(self, trace_id: str, overrides: RunConfig | None = None) -> RunResult:
        # TODO: Load trace, re-run with overrides
        raise NotImplementedError("replay not implemented yet")

    def list_traces(self, limit: int = 50) ->list[TraceRecord]:
        return self.dbStore.list_traces(limit=limit)
    
    def get_trace(self, trace_id: str) -> TraceRecord | None:
        return self.dbStore.get_trace(trace_id)
    
    def _generate_trace_id(self) -> str:
        return str(format(self.id_generator.generate_trace_id(), '032x'))

    

