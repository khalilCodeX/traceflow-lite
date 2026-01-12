from pydantic import BaseModel

class chroma_params(BaseModel):
    documents: list[str]
    collection: str = "traceflow-kb"
    directory: str = "./chroma_db"