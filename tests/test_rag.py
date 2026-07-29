import io
import pytest
from uuid import uuid4
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException

from src.main import app
from src.rag.schemas import (
    KnowledgeDocumentResponse,
    RAGSearchResult,
    RAGGenerateResponse,
    RebuildIndexResponse,
)
from src.rag.service import RAGService
from src.rag.dependencies import get_rag_service

client = TestClient(app)


@pytest.fixture
def mock_rag_service():
    service = AsyncMock(spec=RAGService)
    service.repository = AsyncMock()
    app.dependency_overrides[get_rag_service] = lambda: service
    yield service
    app.dependency_overrides.clear()



def test_rag_upload_document_success(mock_rag_service):
    doc_id = uuid4()
    mock_rag_service.upload_and_index_document.return_value = KnowledgeDocumentResponse(
        id=doc_id,
        title="ICAR Chilli Pest Advisory",
        source="ICAR Publications",
        category="Pest Control",
        language="te",
        state="Telangana",
        crop="Chilli",
        pdf_path="/path/file.pdf",
        chunk_count=3,
        created_at="2026-07-29T10:00:00",
        updated_at="2026-07-29T10:00:00",
    )

    file_content = b"Sample ICAR advisory content for chilli thrips control."
    files = {"file": ("advisory.pdf", io.BytesIO(file_content), "application/pdf")}
    data = {
        "title": "ICAR Chilli Pest Advisory",
        "source": "ICAR Publications",
        "category": "Pest Control",
        "language": "te",
        "state": "Telangana",
        "crop": "Chilli",
    }

    response = client.post("/rag/upload", data=data, files=files)
    assert response.status_code == 201
    res_data = response.json()
    assert res_data["title"] == "ICAR Chilli Pest Advisory"
    assert res_data["source"] == "ICAR Publications"
    assert res_data["chunk_count"] == 3


def test_rag_upload_empty_file(mock_rag_service):
    files = {"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")}
    data = {
        "title": "Empty Doc",
        "source": "KVK",
        "category": "Advisory",
    }

    response = client.post("/rag/upload", data=data, files=files)
    assert response.status_code == 400


def test_rag_upload_missing_file_422(mock_rag_service):
    """Missing file parameter returns HTTP 422 Unprocessable Entity."""
    data = {
        "title": "No File Doc",
        "category": "general",
    }
    response = client.post("/rag/upload", data=data)
    assert response.status_code == 422


def test_rag_upload_unsupported_file_type_400(mock_rag_service):
    """Unsupported binary file returns HTTP 400 Bad Request."""
    files = {"file": ("script.exe", io.BytesIO(b"\x7fELF\x01\x01\x01"), "application/octet-stream")}
    data = {
        "category": "general",
    }
    response = client.post("/rag/upload", data=data, files=files)
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]



def test_rag_list_documents(mock_rag_service):
    doc_id = uuid4()
    mock_doc = KnowledgeDocumentResponse(
        id=doc_id,
        title="KVK Fertilizer Guide",
        source="KVK Advisories",
        category="Fertilizer",
        language="te",
        state="Telangana",
        crop="Paddy",
        pdf_path=None,
        chunk_count=2,
        created_at="2026-07-29T10:00:00",
        updated_at="2026-07-29T10:00:00",
    )
    mock_rag_service.repository.list_documents.return_value = [mock_doc]

    response = client.get("/rag/documents")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["title"] == "KVK Fertilizer Guide"


def test_rag_search_knowledge(mock_rag_service):
    mock_results = [
        RAGSearchResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            document_title="ICAR Pest Guide",
            source="ICAR Publications",
            category="Pest Control",
            language="te",
            state="Telangana",
            crop="Chilli",
            chunk_text="Use Imidacloprid 17.8 SL for chilli thrips.",
            similarity_score=0.92,
        )
    ]
    mock_rag_service.search_knowledge.return_value = mock_results
    mock_rag_service.hybrid_search_knowledge.return_value = mock_results


    response = client.get("/rag/search?query=chilli%20thrips&top_k=3")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["similarity_score"] == 0.92
    assert "Imidacloprid" in data[0]["chunk_text"]


def test_rag_rebuild_index(mock_rag_service):
    mock_rag_service.rebuild_index.return_value = RebuildIndexResponse(
        success=True,
        message="Vector index rebuilt successfully across all documents.",
        documents_processed=5,
        total_chunks=25,
    )

    response = client.post("/rag/rebuild")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["documents_processed"] == 5
    assert data["total_chunks"] == 25


def test_rag_delete_document_success(mock_rag_service):
    mock_rag_service.repository.delete_document.return_value = True

    doc_id = uuid4()
    response = client.delete(f"/rag/document/{doc_id}")
    assert response.status_code == 200
    assert response.json()["success"] is True


def test_rag_delete_document_not_found(mock_rag_service):
    mock_rag_service.repository.delete_document.return_value = False

    doc_id = uuid4()
    response = client.delete(f"/rag/document/{doc_id}")
    assert response.status_code == 404


def test_rag_query_grounded_answer(mock_rag_service):
    mock_rag_service.generate_rag_response.return_value = RAGGenerateResponse(
        answer="Spray Imidacloprid 17.8 SL at 0.5 ml/L of water for chilli thrips control.",
        reasoning="Based on ICAR Pest Advisory 2026 for Telangana Chilli farmers.",
        source_documents=[
            RAGSearchResult(
                chunk_id=uuid4(),
                document_id=uuid4(),
                document_title="ICAR Pest Guide",
                source="ICAR Publications",
                category="Pest Control",
                language="te",
                state="Telangana",
                crop="Chilli",
                chunk_text="Spray Imidacloprid 17.8 SL at 0.5 ml/L.",
                similarity_score=0.95,
            )
        ],
        confidence_score=0.95,
    )

    farmer_id = str(uuid4())
    response = client.post("/rag/query", json={
        "farmer_id": farmer_id,
        "message": "What pesticide to use for chilli thrips?"
    })

    assert response.status_code == 200
    data = response.json()
    assert "Imidacloprid" in data["answer"]
    assert data["confidence_score"] == 0.95
    assert len(data["source_documents"]) == 1


def test_rag_service_unit_chunking_and_similarity():
    """Unit test for Service text chunking and cosine similarity functions."""
    from src.rag.service import RAGService
    service = RAGService(repository=None)

    # 1. Chunking test
    sample_text = "Paragraph 1: ICAR recommendation for cotton.\n\nParagraph 2: KVK fertilizer dosage guidelines.\n\nParagraph 3: Weather advisory notice."
    chunks = service.chunk_text(sample_text, max_chunk_size=100, overlap=20)
    assert len(chunks) >= 2

    # 2. Embedding & Cosine Similarity test
    v1 = service.generate_embedding("pesticide dosage for chilli thrips")
    v2 = service.generate_embedding("pesticide dosage for chilli thrips")
    v3 = service.generate_embedding("rainfall weather warning forecast")

    sim_same = service.cosine_similarity(v1, v2)
    sim_diff = service.cosine_similarity(v1, v3)

    assert sim_same == pytest.approx(1.0, rel=1e-3)
    assert sim_same > sim_diff


def test_hybrid_search_keyword_and_metadata_re_ranking():
    """Unit test for TF-IDF keyword scoring and metadata match re-ranking."""
    from src.rag.service import RAGService
    service = RAGService(repository=None)

    # Keyword scoring
    kw_high = service.calculate_keyword_score("chilli thrips imidacloprid", "Apply Imidacloprid for controlling chilli thrips outbreak.")
    kw_low = service.calculate_keyword_score("chilli thrips imidacloprid", "General rainfall forecast for paddy farmers.")

    assert kw_high > kw_low
    assert kw_high > 0.30

    # Metadata scoring
    m_match = service.calculate_metadata_score(
        doc_crop="Chilli", doc_state="Telangana", doc_category="Pest Control", doc_language="te", doc_source="ICAR Publications",
        farmer_crop="Chilli", farmer_state="Telangana", filter_category="Pest Control", filter_language="te", filter_source="ICAR Publications"
    )
    m_mismatch = service.calculate_metadata_score(
        doc_crop="Wheat", doc_state="Punjab", doc_category="Soil Health", doc_language="en", doc_source="Private",
        farmer_crop="Chilli", farmer_state="Telangana", filter_category="Pest Control", filter_language="te", filter_source="ICAR Publications"
    )

    assert m_match > m_mismatch
    assert m_match == pytest.approx(1.0, abs=0.01)


def test_page_extraction_and_chunking():
    """Unit test for page extraction and page number retention in chunking."""
    from src.rag.service import RAGService
    service = RAGService(repository=None)

    pages = [
        (1, "Page 1 content: ICAR Chilli Advisory."),
        (2, "Page 2 content: Recommended dosage for imidacloprid."),
    ]
    chunk_records = service.chunk_pages(pages, max_chunk_size=100)

    assert len(chunk_records) >= 2
    assert chunk_records[0]["page_number"] == 1
    assert chunk_records[-1]["page_number"] == 2

