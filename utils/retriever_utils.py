import os
import chromadb
from openai import OpenAI
from .vector_types import chroma_params
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tf_types import RetrievedChunk
from dotenv import load_dotenv

load_dotenv()


class chroma_retriever:
    """Chroma retriever implementation (stub)."""

    def __init__(self, local: bool = True, params: chroma_params | None = None):
        if local:
            if params is None:
                raise ValueError("Chroma parameters must be provided for local instance.")
            self.chroma_params = params
            self.chroma_client = chromadb.PersistentClient(
                path=params.directory,
            )
            self.collection = self.chroma_client.get_or_create_collection(name=params.collection)
        else:
            self.chroma_client = chromadb.CloudClient(
                api_key=os.getenv("CHROMA_API_KEY"),
                tenant=os.getenv("CHROMA_TENANT"),
                database=os.getenv("CHROMA_DATABASE"),
            )
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def chunk_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        all_splits = text_splitter.split_documents(documents)
        print(f"Split into {len(all_splits)} chunks.")

        return all_splits

    def create_document_batches(self, splits: list[str], batch_size: int):
        for i in range(0, len(splits), batch_size):
            yield splits[i : i + batch_size]

    def create_vector_store(
        self,
        splits: list[str],
        batch_size: int = 1000,
        progress_callback: callable = None,
    ):
        """Create vector store from document splits.

        Args:
            splits: List of document chunks to embed
            batch_size: Number of documents per batch
            progress_callback: Optional callback(current, total) for progress updates
        """
        batches = list(self.create_document_batches(splits, batch_size))
        total_batches = len(batches)

        for i, batch in enumerate(tqdm(batches, desc="embedding batches", unit="batch")):
            embeddings = self.client.embeddings.create(model="text-embedding-3-small", input=batch)

            embedding_vectors = [e.embedding for e in embeddings.data]

            self.collection.add(
                ids=[str(i * batch_size + j) for j in range(len(batch))],
                embeddings=embedding_vectors,
                documents=batch,
                metadatas=[{"source": f"doc_{i * batch_size + j}"} for j in range(len(batch))],
            )

            # Call progress callback if provided
            if progress_callback:
                progress_callback(i + 1, total_batches)

    def retrieve_similar_docs(self, query: str, n_results: int = 5) -> list[RetrievedChunk]:
        query_response = self.client.embeddings.create(
            model="text-embedding-3-small", input=[query]
        )
        query_embedding = query_response.data[0].embedding
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        retrieved_chunks = []
        for doc, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=metadata.get("source", ""),
                    content=doc,
                    metadata=metadata,
                    relevance_score=(1.0 - distance / 2),
                    source=metadata.get("source", ""),
                )
            )

        return retrieved_chunks
