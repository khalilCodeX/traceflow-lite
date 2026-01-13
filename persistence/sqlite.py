import sqlite3
import os
class Sqlite:

    @staticmethod
    def get_connection(db_path: str = "traceflow.db") -> sqlite3.Connection:
        """
        Creates a connection with optimized PRAGMAs for performance and concurrency.
        Enables WAL mode and sets safe synchronous levels.
        """
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        # check_same_thread=False allows connection to be used across threads
        # This is safe when combined with WAL mode which handles concurrent access
        conn = sqlite3.connect(db_path, check_same_thread=False)

        # Use a dictionary-like row factory for better usability
        conn.row_factory = sqlite3.Row
    
        # Execute optimized PRAGMAs
        # journal_mode=WAL: Enables Write-Ahead Logging for better concurrency
        # synchronous=NORMAL: Safe when using WAL; reduces disk sync overhead
        # mmap_size: Uses memory-mapped I/O for faster reading
        # foreign_keys=ON: Ensures data integrity (not enabled by default)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA mmap_size = 30000000000;")  # Up to 30GB if available
        conn.execute("PRAGMA foreign_keys = ON;")
        
        return conn
    
    @staticmethod
    def init_db(conn: sqlite3.Connection) -> None:
        """
          Create tables if not exist
          Called once on startup
        """
        with conn:
            # TODO: Need to add model, provider, cost breakdown, etc.
            #-- traces: one row per run()
            conn.execute("""
            CREATE TABLE IF NOT EXISTS traces (
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
            );
            """)
            
            #-- steps: one row per node execution
            conn.execute("""
            CREATE TABLE IF NOT EXISTS trace_steps (
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
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_db_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (trace_id) REFERENCES traces(trace_id) ON DELETE CASCADE
                );
                """)
            
            #-- evals: one row per eval gate decision
            conn.execute("""
            CREATE TABLE IF NOT EXISTS trace_evals (
                eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                step_seq INTEGER NOT NULL,
                decision TEXT NOT NULL,
                reasons_json TEXT,
                scores_json TEXT,
                revision_instructions TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_db_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (trace_id) REFERENCES traces(trace_id) ON DELETE CASCADE
                );
                """)