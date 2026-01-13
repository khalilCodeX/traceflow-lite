import hashlib
import json
from persistence.sqlite import Sqlite
from .base import ProviderResponse

class LLMCache:
    def __init__(self, db_path: str = "traceflow.db"):
        self.conn = Sqlite.get_connection(db_path)
        Sqlite.init_db(self.conn)
    
    def compute_key(self, model: str, messages: list[dict], **kwargs) -> str:
        data = {"model": model, "messages": messages, **kwargs}
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    def get(self, cache_key: str) -> ProviderResponse | None:
        cursor = self.conn.execute(
            "SELECT response_content, input_tokens, output_tokens, model FROM llm_cache WHERE cache_key = ?",
            (cache_key,)
        )
        row = cursor.fetchone()
        if row:
            self.conn.execute("UPDATE llm_cache SET hit_count = hit_count + 1 WHERE cache_key = ?", (cache_key,))
            self.conn.commit()
            return ProviderResponse(content=row[0], input_tokens=row[1], output_tokens=row[2], model=row[3])
        return None
    
    def set(self, cache_key: str, model: str, response: ProviderResponse) -> None:
        self.conn.execute("""
            INSERT OR REPLACE INTO llm_cache (cache_key, model, response_content, input_tokens, output_tokens)
            VALUES (?, ?, ?, ?, ?)
        """, (cache_key, model, response.content, response.input_tokens, response.output_tokens))
        self.conn.commit()