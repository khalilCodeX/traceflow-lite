import json
from dataclasses import fields
from datetime import datetime, timezone
from .sqlite import Sqlite
from tf_types import EvalDecision, EvalRecord, Mode, RunStatus, StepRecord, TraceRecord, RunConfig

class TraceStore:

    def __init__(self, db_path: str = "traceflow.db"):
        self.conn = Sqlite.get_connection(db_path)
        Sqlite.init_db(self.conn)

    def create_trace(self, trace_data: TraceRecord) -> None:
        """
                trace_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                mode TEXT NOT NULL,
                user_input TEXT NOT NULL,
                config_json TEXT NOT NULL,
                model TEXT NOT NULL,
                provider TEXT NOT NULL, 
                final_answer TEXT,
                error TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                finished_at TIMESTAMP,
                updated_db_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        """
        # Exclude retriever_fn from JSON (functions can't be serialized)
        config_dict = {}
        for f in fields(trace_data.config):
            if f.name != 'retriever_fn':
                value = getattr(trace_data.config, f.name)
                # Convert enums to their string value
                if hasattr(value, 'value'):
                    value = value.value
                config_dict[f.name] = value
        
        self.conn.execute("""
        INSERT INTO traces (trace_id, status, mode, user_input, config_json, model, provider, final_answer, error, created_at, finished_at, updated_db_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            trace_data.trace_id,
            trace_data.status.value,
            trace_data.mode.value,
            trace_data.user_input,
            json.dumps(config_dict),
            trace_data.model,
            trace_data.provider,
            trace_data.final_answer,
            trace_data.error,
            trace_data.created_at.isoformat(),
            trace_data.finished_at.isoformat() if trace_data.finished_at else None,
            datetime.now(timezone.utc).isoformat()
        ))
        self.conn.commit()

    def update_trace(self, trace_id: str, status: RunStatus, final_answer: str | None = None, error: str | None = None) -> None:
        
        now = datetime.now(timezone.utc).isoformat()
        finished = now if status in (RunStatus.DONE, RunStatus.FAILED) else None

        self.conn.execute("""
        UPDATE traces SET status = ?, final_answer = ?, error = ?, finished_at = ?, updated_db_at = ?
        WHERE trace_id = ?;
        """, (
            status.value,
            final_answer,
            error,
            finished,
            now,
            trace_id,
        ))

        self.conn.commit()

    def _row_to_trace_record(self, row) -> TraceRecord:
        return TraceRecord(
            trace_id=row["trace_id"],
            user_input=row["user_input"],
            config=RunConfig(**json.loads(row["config_json"])),
            mode=Mode(row["mode"]),
            status=RunStatus(row["status"]),
            model=row["model"],
            provider=row["provider"],
            final_answer=row["final_answer"],
            error=row["error"],
            created_at=datetime.fromisoformat(row["created_at"]),
            finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None
        )  

    def get_trace(self, trace_id: str) -> TraceRecord | None:

        cursor = self.conn.execute("""
        SELECT * FROM traces WHERE trace_id = ?;
        """, (
            trace_id,
        ))
        
        row = cursor.fetchone()
        return self._row_to_trace_record(row) if row else None


    def list_traces(self, limit: int = 10) -> list[TraceRecord]:
        # Returns most recent traces, ordered by created_at DESC
        cursor = self.conn.execute("""
        SELECT * FROM traces ORDER BY created_at DESC LIMIT ?;
        """, (limit,))  
        
        return [self._row_to_trace_record(row) for row in cursor.fetchall()]
        
    def insert_step(self, step_record: StepRecord) -> None:
        """
                step_id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                step_seq INTEGER NOT NULL,
                node_name TEXT NOT NULL,
                input_json TEXT,
                output_json TEXT,
                tokens INTEGER DEFAULT 0,
                latency_ms INTEGER DEFAULT 0,
                cost_usd REAL DEFAULT 0.0,
                error TEXT,
                cache_hit INTEGER DEFAULT 0,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_db_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        """
        self.conn.execute("""
        INSERT INTO trace_steps (trace_id, step_seq, node_name, input_json, output_json, tokens, latency_ms, cost_usd, error, cache_hit, created_at, updated_db_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            step_record.trace_id,
            step_record.step_seq,
            step_record.node_name,
            json.dumps(step_record.input_data) if step_record.input_data else None,
            json.dumps(step_record.output_data) if step_record.output_data else None,
            step_record.tokens,
            step_record.latency_ms,
            step_record.cost_usd,
            step_record.error,
            step_record.cache_hit,
            datetime.now(timezone.utc).isoformat(),
            datetime.now(timezone.utc).isoformat()
        ))
        self.conn.commit()

    def _row_to_step_record(self, row) -> StepRecord:
        """
        trace_id: str
        step_seq: int
        node_name: str
        input_data: dict | None = None
        output_data: dict | None = None
        tokens: int = 0
        cost_usd: float = 0.0
        latency_ms: float = 0.0
        error: str | None = None,
        cache_hit: bool = False
        created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
       """
        return StepRecord(
            trace_id=row["trace_id"],
            step_seq=row["step_seq"],
            node_name=row["node_name"],
            input_data=json.loads(row["input_json"]) if row["input_json"] else None,
            output_data=json.loads(row["output_json"]) if row["output_json"] else None,
            tokens=row["tokens"],
            latency_ms=row["latency_ms"],
            cost_usd=row["cost_usd"],
            error=row["error"],
            cache_hit=bool(row["cache_hit"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def get_steps(self, trace_id: str) -> list[StepRecord]:
        
        cursor = self.conn.execute("""
        SELECT * FROM trace_steps WHERE trace_id = ? ORDER BY step_seq ASC;
        """, (trace_id,))
        
        return [self._row_to_step_record(row) for row in cursor.fetchall()]

    def insert_eval_report(self, eval_record: EvalRecord) -> None:
        """
                eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                step_seq INTEGER NOT NULL,
                decision TEXT NOT NULL,
                reasons_json TEXT,
                scores_json TEXT,
                revision_instructions TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_db_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        """
        self.conn.execute("""
        INSERT INTO trace_evals (trace_id, step_seq, decision, reasons_json, scores_json, revision_instructions, created_at, updated_db_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            eval_record.trace_id,
            eval_record.step_seq,
            eval_record.decision.value,
            json.dumps(eval_record.reasons) if eval_record.reasons else None,
            json.dumps(eval_record.scores) if eval_record.scores else None,
            eval_record.revision_instructions,
            datetime.now(timezone.utc).isoformat(),
            datetime.now(timezone.utc).isoformat()
        ))
        self.conn.commit()

    def _row_to_eval_record(self, row) -> EvalRecord:
        """
        trace_id: str
        step_seq: int
        decision: EvalDecision
        reasons: list[str] | None = None
        scores: dict[str, float] | None = None
        revision_instructions: str | None = None
        created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
       """
        return EvalRecord(
            trace_id=row["trace_id"],
            step_seq=row["step_seq"],
            decision=EvalDecision(row["decision"]),
            reasons=json.loads(row["reasons_json"]) if row["reasons_json"] else None,
            scores=json.loads(row["scores_json"]) if row["scores_json"] else None,
            revision_instructions=row["revision_instructions"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def get_evals(self, trace_id: str) -> list[EvalRecord]:
        
        cursor = self.conn.execute("""
        SELECT * FROM trace_evals WHERE trace_id = ? ORDER BY step_seq ASC;
        """, (trace_id,))
        
        return [self._row_to_eval_record(row) for row in cursor.fetchall()]