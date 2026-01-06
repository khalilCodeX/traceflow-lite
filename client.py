from tf_types import Mode, RunConfig, RunResult, RunStatus
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator


class TraceFlowClient:
    def __init__(self):
        # TODO: Load config, init stores, build graph
        self.id_generator = RandomIdGenerator()

    def run(self, user_input: str, config: RunConfig | None = None) -> RunResult:
        # TODO: Wire up real graph
        trace_id = self._generate_trace_id()
        config = config or RunConfig() 
        return RunResult(
            trace_id=trace_id,
            status=RunStatus.DONE,
            mode=config.mode,
            answer="This is a placeholder answer."
        )

    def replay(self, trace_id: str, overrides: RunConfig | None = None) -> RunResult:
        # TODO: Load trace, re-run with overrides
        raise NotImplementedError("replay not implemented yet")

    def list_traces(self, limit: int = 50) ->list[dict]:
        # TODO: Query SQLite
        return []
    
    def get_trace(self, trace_id: str) -> dict:
        # TODO: Load from SQLite
        return {}
    
    def _generate_trace_id(self) -> str:
        return str(format(self.id_generator.generate_trace_id(), '032x'))

    

