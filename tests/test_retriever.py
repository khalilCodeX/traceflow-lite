"""
Tests for RAG retriever utilities.
All tests are mocked to run without API keys or external services.
"""

import pytest
from unittest.mock import Mock, patch
from utils.vector_types import chroma_params
from tf_types import RetrievedChunk


class TestChromaParams:
    """Tests for chroma_params dataclass."""

    def test_chroma_params_defaults(self):
        """chroma_params should have sensible defaults."""
        params = chroma_params()

        assert params.documents is None
        assert params.collection == "traceflow-kb"
        assert params.directory == "./chroma_db"

    def test_chroma_params_custom_values(self):
        """chroma_params should accept custom values."""
        params = chroma_params(
            documents=["doc1", "doc2"], collection="my-collection", directory="/custom/path"
        )

        assert params.documents == ["doc1", "doc2"]
        assert params.collection == "my-collection"
        assert params.directory == "/custom/path"

    def test_chroma_params_empty_documents(self):
        """chroma_params should accept empty document list."""
        params = chroma_params(documents=[])
        assert params.documents == []


class TestChromaRetriever:
    """Tests for chroma_retriever class."""

    @patch("utils.retriever_utils.chromadb.PersistentClient")
    @patch("utils.retriever_utils.OpenAI")
    def test_init_local_mode(self, mock_openai, mock_chroma):
        """Retriever should initialize in local mode with params."""
        from utils.retriever_utils import chroma_retriever

        mock_collection = Mock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        params = chroma_params(collection="test-coll", directory="./test_db")
        retriever = chroma_retriever(local=True, params=params)

        mock_chroma.assert_called_once_with(path="./test_db")
        mock_chroma.return_value.get_or_create_collection.assert_called_once_with(name="test-coll")
        assert retriever.collection == mock_collection

    @patch("utils.retriever_utils.chromadb.PersistentClient")
    @patch("utils.retriever_utils.OpenAI")
    def test_init_local_mode_requires_params(self, mock_openai, mock_chroma):
        """Retriever should raise error if local mode without params."""
        from utils.retriever_utils import chroma_retriever

        with pytest.raises(ValueError, match="Chroma parameters must be provided"):
            chroma_retriever(local=True, params=None)

    @patch("utils.retriever_utils.chromadb.PersistentClient")
    @patch("utils.retriever_utils.OpenAI")
    def test_create_document_batches(self, mock_openai, mock_chroma):
        """create_document_batches should yield correct batch sizes."""
        from utils.retriever_utils import chroma_retriever

        mock_chroma.return_value.get_or_create_collection.return_value = Mock()

        params = chroma_params(collection="test", directory="./test")
        retriever = chroma_retriever(local=True, params=params)

        documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        batches = list(retriever.create_document_batches(documents, batch_size=2))

        assert len(batches) == 3
        assert batches[0] == ["doc1", "doc2"]
        assert batches[1] == ["doc3", "doc4"]
        assert batches[2] == ["doc5"]

    @patch("utils.retriever_utils.chromadb.PersistentClient")
    @patch("utils.retriever_utils.OpenAI")
    def test_create_document_batches_exact_multiple(self, mock_openai, mock_chroma):
        """create_document_batches should handle exact multiples."""
        from utils.retriever_utils import chroma_retriever

        mock_chroma.return_value.get_or_create_collection.return_value = Mock()

        params = chroma_params(collection="test", directory="./test")
        retriever = chroma_retriever(local=True, params=params)

        documents = ["doc1", "doc2", "doc3", "doc4"]
        batches = list(retriever.create_document_batches(documents, batch_size=2))

        assert len(batches) == 2
        assert batches[0] == ["doc1", "doc2"]
        assert batches[1] == ["doc3", "doc4"]

    @patch("utils.retriever_utils.chromadb.PersistentClient")
    @patch("utils.retriever_utils.OpenAI")
    def test_create_document_batches_empty(self, mock_openai, mock_chroma):
        """create_document_batches should handle empty list."""
        from utils.retriever_utils import chroma_retriever

        mock_chroma.return_value.get_or_create_collection.return_value = Mock()

        params = chroma_params(collection="test", directory="./test")
        retriever = chroma_retriever(local=True, params=params)

        batches = list(retriever.create_document_batches([], batch_size=2))
        assert batches == []

    @patch("utils.retriever_utils.chromadb.PersistentClient")
    @patch("utils.retriever_utils.OpenAI")
    def test_create_vector_store(self, mock_openai, mock_chroma):
        """create_vector_store should embed and store documents."""
        from utils.retriever_utils import chroma_retriever

        # Mock collection
        mock_collection = Mock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        # Mock OpenAI embeddings
        mock_embedding = Mock()
        mock_embedding.embedding = [0.1, 0.2, 0.3]
        mock_openai.return_value.embeddings.create.return_value = Mock(
            data=[mock_embedding, mock_embedding]
        )

        params = chroma_params(collection="test", directory="./test")
        retriever = chroma_retriever(local=True, params=params)

        documents = ["doc1", "doc2"]
        retriever.create_vector_store(documents, batch_size=2)

        # Verify embeddings were created
        mock_openai.return_value.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input=["doc1", "doc2"]
        )

        # Verify documents were added to collection
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args
        assert call_args.kwargs["ids"] == ["0", "1"]
        assert call_args.kwargs["documents"] == ["doc1", "doc2"]

    @patch("utils.retriever_utils.chromadb.PersistentClient")
    @patch("utils.retriever_utils.OpenAI")
    def test_create_vector_store_multiple_batches(self, mock_openai, mock_chroma):
        """create_vector_store should handle multiple batches with correct IDs."""
        from utils.retriever_utils import chroma_retriever

        mock_collection = Mock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        # Mock embeddings for each batch
        mock_embedding = Mock()
        mock_embedding.embedding = [0.1, 0.2, 0.3]
        mock_openai.return_value.embeddings.create.return_value = Mock(
            data=[mock_embedding, mock_embedding]
        )

        params = chroma_params(collection="test", directory="./test")
        retriever = chroma_retriever(local=True, params=params)

        documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        retriever.create_vector_store(documents, batch_size=2)

        # Should have 3 batches
        assert mock_collection.add.call_count == 3

        # Verify IDs are sequential across batches
        calls = mock_collection.add.call_args_list
        assert calls[0].kwargs["ids"] == ["0", "1"]
        assert calls[1].kwargs["ids"] == ["2", "3"]
        assert calls[2].kwargs["ids"] == ["4"]

    @patch("utils.retriever_utils.chromadb.PersistentClient")
    @patch("utils.retriever_utils.OpenAI")
    def test_create_vector_store_progress_callback(self, mock_openai, mock_chroma):
        """create_vector_store should call progress callback."""
        from utils.retriever_utils import chroma_retriever

        mock_collection = Mock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        mock_embedding = Mock()
        mock_embedding.embedding = [0.1, 0.2, 0.3]
        mock_openai.return_value.embeddings.create.return_value = Mock(
            data=[mock_embedding, mock_embedding]
        )

        params = chroma_params(collection="test", directory="./test")
        retriever = chroma_retriever(local=True, params=params)

        # Track progress calls
        progress_calls = []

        def track_progress(current, total):
            progress_calls.append((current, total))

        documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        retriever.create_vector_store(documents, batch_size=2, progress_callback=track_progress)

        # Should have 3 progress updates
        assert progress_calls == [(1, 3), (2, 3), (3, 3)]

    @patch("utils.retriever_utils.chromadb.PersistentClient")
    @patch("utils.retriever_utils.OpenAI")
    def test_retrieve_similar_docs(self, mock_openai, mock_chroma):
        """retrieve_similar_docs should query and return chunks."""
        from utils.retriever_utils import chroma_retriever

        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["Document 1 content", "Document 2 content"]],
            "metadatas": [[{"source": "doc_0"}, {"source": "doc_1"}]],
            "distances": [[0.1, 0.3]],
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        # Mock query embedding
        mock_embedding = Mock()
        mock_embedding.embedding = [0.1, 0.2, 0.3]
        mock_openai.return_value.embeddings.create.return_value = Mock(data=[mock_embedding])

        params = chroma_params(collection="test", directory="./test")
        retriever = chroma_retriever(local=True, params=params)

        results = retriever.retrieve_similar_docs("test query", n_results=2)

        # Verify query was made
        mock_collection.query.assert_called_once()

        # Verify results
        assert len(results) == 2
        assert isinstance(results[0], RetrievedChunk)
        assert results[0].content == "Document 1 content"
        assert results[0].source == "doc_0"
        assert results[0].relevance_score == pytest.approx(0.95, rel=0.01)  # 1 - 0.1/2
        assert results[1].relevance_score == pytest.approx(0.85, rel=0.01)  # 1 - 0.3/2

    @patch("utils.retriever_utils.chromadb.PersistentClient")
    @patch("utils.retriever_utils.OpenAI")
    def test_retrieve_similar_docs_empty_results(self, mock_openai, mock_chroma):
        """retrieve_similar_docs should handle empty results."""
        from utils.retriever_utils import chroma_retriever

        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        mock_embedding = Mock()
        mock_embedding.embedding = [0.1, 0.2, 0.3]
        mock_openai.return_value.embeddings.create.return_value = Mock(data=[mock_embedding])

        params = chroma_params(collection="test", directory="./test")
        retriever = chroma_retriever(local=True, params=params)

        results = retriever.retrieve_similar_docs("test query")

        assert results == []

    @patch("utils.retriever_utils.chromadb.PersistentClient")
    @patch("utils.retriever_utils.OpenAI")
    def test_chunk_documents(self, mock_openai, mock_chroma):
        """chunk_documents should split documents using text splitter."""
        from utils.retriever_utils import chroma_retriever
        from langchain_core.documents import Document

        mock_chroma.return_value.get_or_create_collection.return_value = Mock()

        params = chroma_params(collection="test", directory="./test")
        retriever = chroma_retriever(local=True, params=params)

        # Create a document that will be split
        long_text = "This is a test. " * 200  # ~3200 chars
        documents = [Document(page_content=long_text)]

        chunks = retriever.chunk_documents(documents)

        # Should be split into multiple chunks
        assert len(chunks) > 1


class TestRetrievedChunk:
    """Tests for RetrievedChunk dataclass."""

    def test_retrieved_chunk_creation(self):
        """RetrievedChunk should be created with all fields."""
        chunk = RetrievedChunk(
            chunk_id="chunk_1", content="Test content", source="test_doc.txt", relevance_score=0.95
        )

        assert chunk.chunk_id == "chunk_1"
        assert chunk.content == "Test content"
        assert chunk.source == "test_doc.txt"
        assert chunk.relevance_score == 0.95


class TestVectorStoreIntegration:
    """Integration-style tests for vector store operations."""

    @patch("utils.retriever_utils.chromadb.PersistentClient")
    @patch("utils.retriever_utils.OpenAI")
    def test_full_index_and_retrieve_flow(self, mock_openai, mock_chroma):
        """Test full flow: create vector store then retrieve."""
        from utils.retriever_utils import chroma_retriever

        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["AI is artificial intelligence"]],
            "metadatas": [[{"source": "doc_0"}]],
            "distances": [[0.2]],
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        # Mock embeddings
        mock_embedding = Mock()
        mock_embedding.embedding = [0.1] * 1536
        mock_openai.return_value.embeddings.create.return_value = Mock(data=[mock_embedding])

        params = chroma_params(collection="test", directory="./test")
        retriever = chroma_retriever(local=True, params=params)

        # Index documents
        retriever.create_vector_store(["AI is artificial intelligence"])

        # Retrieve
        results = retriever.retrieve_similar_docs("What is AI?")

        assert len(results) == 1
        assert "artificial intelligence" in results[0].content

    @patch("utils.retriever_utils.chromadb.PersistentClient")
    @patch("utils.retriever_utils.OpenAI")
    def test_batch_size_affects_api_calls(self, mock_openai, mock_chroma):
        """Smaller batch size should result in more API calls."""
        from utils.retriever_utils import chroma_retriever

        mock_collection = Mock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        mock_embedding = Mock()
        mock_embedding.embedding = [0.1, 0.2, 0.3]
        mock_openai.return_value.embeddings.create.return_value = Mock(data=[mock_embedding])

        params = chroma_params(collection="test", directory="./test")
        retriever = chroma_retriever(local=True, params=params)

        documents = ["doc1", "doc2", "doc3", "doc4"]

        # With batch_size=1, should make 4 API calls
        retriever.create_vector_store(documents, batch_size=1)
        assert mock_openai.return_value.embeddings.create.call_count == 4

        # Reset
        mock_openai.return_value.embeddings.create.reset_mock()
        mock_collection.reset_mock()

        # With batch_size=4, should make 1 API call
        mock_openai.return_value.embeddings.create.return_value = Mock(data=[mock_embedding] * 4)
        retriever.create_vector_store(documents, batch_size=4)
        assert mock_openai.return_value.embeddings.create.call_count == 1
