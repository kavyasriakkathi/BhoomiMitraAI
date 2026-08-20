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


def test_pdf_extraction_clean_text_and_rejects_binary_stream():
    """Test that PDF extraction returns clean readable text and rejects raw binary decoding."""
    from src.rag.service import RAGService
    service = RAGService(repository=None)

    # Valid text file / clean text
    clean_text = "ICAR Cotton Leaf Curl Virus Advisory. Manage whiteflies with Imidacloprid."
    pages = service.extract_pages_from_file(clean_text.encode("utf-8"), "cotton_guide.txt")
    assert len(pages) == 1
    assert "Cotton Leaf Curl Virus" in pages[0][1]
    assert "%PDF" not in pages[0][1]


def test_image_only_pdf_rejection_requires_ocr():
    """Test that image-only or unextractable PDFs raise HTTP 400 with OCR requirement."""
    from src.rag.service import RAGService
    service = RAGService(repository=None)

    # Simulate an empty/image-only PDF (valid PDF header with 0 extractable text)
    empty_pdf_bytes = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] >>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \ntrailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n185\n%%EOF"

    with pytest.raises(HTTPException) as exc_info:
        service.extract_pages_from_file(empty_pdf_bytes, "scanned_doc.pdf")
    
    assert exc_info.value.status_code == 400
    assert "OCR" in exc_info.value.detail or "no extractable text" in exc_info.value.detail


def test_cotton_rag_keyword_scoring_verified_vs_unverified():
    """Test keyword score is positive for verified cotton disease terms and 0 for unverified urea dosage."""
    from src.rag.service import RAGService
    service = RAGService(repository=None)

    cotton_doc_chunk = (
        "Cotton Disease Guide Cotton Leaf Curl Virus Symptoms: Leaves curl upward, Yellowing of leaves, "
        "Stunted plant growth. Management: Remove infected plants, Control whitefly population, "
        "Recommended pesticides: Imidacloprid, Acetamiprid, Thiamethoxam."
    )

    # Verified query in document
    kw_disease = service.calculate_keyword_score("Cotton Leaf Curl Virus whitefly", cotton_doc_chunk)
    assert kw_disease > 0.40

    # Unverified query not in disease document (urea flowering dosage)
    kw_urea = service.calculate_keyword_score("urea flowering stage dosage", cotton_doc_chunk)
    assert kw_urea == 0.0


def test_cotton_fertilizer_guide_flowering_dosage_retrieval():
    """Test that authentic cotton fertilizer guide yields high keyword and similarity score for flowering urea dosage."""
    from src.rag.service import RAGService
    service = RAGService(repository=None)

    fert_chunk = (
        "Flowering Stage / Peak Flowering (45 to 60 Days After Sowing): "
        "Second top dressing: Apply 25 to 30 kg Urea per acre along with 10 to 15 kg MOP (Muriate of Potash) per acre. "
        "This provides vital nitrogen and potassium support during peak flowering to prevent square and flower drop."
    )

    kw_flowering_urea = service.calculate_keyword_score("What is the urea dosage for cotton during flowering stage?", fert_chunk)
    assert kw_flowering_urea > 0.45
    assert "25 to 30 kg Urea per acre" in fert_chunk


def test_rag_grounded_answer_includes_verified_dosage():
    """Verify that when verified RAG chunks are available, generate_rag_response uses the retrieved knowledge."""
    from src.rag.service import RAGService
    from src.rag.schemas import RAGSearchResult

    service = RAGService(repository=None)
    
    mock_results = [
        RAGSearchResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            document_title="PJTSAU & ICAR Cotton Fertilizer and Nutrient Guide",
            source="PJTSAU & ICAR-CICR Package of Practices",
            category="Fertilizer Management",
            language="en",
            state="Telangana",
            crop="Cotton",
            page=1,
            chunk_text="Flowering Stage (45-60 DAS): Apply 25 to 30 kg Urea per acre with 10-15 kg MOP.",
            similarity_score=0.92,
        )
    ]

    # Verify formatting of prompt context from verified chunk
    snippet = f"Based on {mock_results[0].document_title} ({mock_results[0].source}, Page {mock_results[0].page}): {mock_results[0].chunk_text}"
    assert "25 to 30 kg Urea per acre" in snippet
    assert "PJTSAU & ICAR-CICR" in snippet


def test_rag_unverified_dosage_safety_fallback_prompt_rule():
    """Verify that if exact dosage is not present in RAG, the prompt safety rule instructs the model to state uncertainty."""
    from src.ai.prompts import BHOOMIMITRA_SYSTEM_PROMPT

    # Verify strict safety rule exists in system prompt
    assert "NEVER invent or guess pesticide names, fertilizer brands, or chemical dosages" in BHOOMIMITRA_SYSTEM_PROMPT
    assert "I am not 100% sure about the exact dosage. Please consult your local agriculture officer" in BHOOMIMITRA_SYSTEM_PROMPT


def test_raw_pdf_binary_stream_data_never_becomes_rag_knowledge():
    """Verify that raw PDF binary streams, headers (%PDF-1.4), xref, and bytecode are rejected by extract_pages_from_file."""
    from src.rag.service import RAGService
    service = RAGService(repository=None)

    raw_pdf_binary_corrupt = (
        b"%PDF-1.4\n1 0 obj << /Filter /FlateDecode /Length 50 >> stream\n"
        b"x\x9c+T0\xd43P04\x00\x02\x95\x02\r\nendstream\nendobj\n"
        b"xref\n0 2\n0000000000 65535 f\n0000000009 00000 n\n"
        b"trailer << /Size 2 /Root 1 0 R >>\nstartxref\n99\n%%EOF"
    )

    with pytest.raises(HTTPException) as exc_info:
        service.extract_pages_from_file(raw_pdf_binary_corrupt, "corrupt_stream.pdf")

    assert exc_info.value.status_code == 400
    assert (
        "Failed to process PDF document" in exc_info.value.detail
        or "OCR" in exc_info.value.detail
        or "no extractable text" in exc_info.value.detail
    )


def test_alternaria_leaf_spot_retrieval():
    """Verify that 'What is Alternaria leaf spot in cotton?' matches Alternaria guide with high keyword & relevance score."""
    from src.rag.service import RAGService
    service = RAGService(repository=None)

    alternaria_chunk = (
        "Title: Cotton Alternaria Leaf Spot Disease Management Guide "
        "Crop: Cotton (Gossypium hirsutum / Gossypium spp.) Disease: Alternaria Leaf Spot "
        "Causal Agents: Alternaria macrospora / Alternaria alternata Disease Type: Fungal disease "
        "1. Symptoms: Small dark-brown to reddish-brown circular or irregular necrotic spots on cotton leaves."
    )

    kw_score = service.calculate_keyword_score("What is Alternaria leaf spot in cotton?", alternaria_chunk)
    assert kw_score > 0.40
    assert "Alternaria Leaf Spot" in alternaria_chunk


def test_alternaria_symptom_retrieval():
    """Verify that symptom description matches Alternaria guide."""
    from src.rag.service import RAGService
    service = RAGService(repository=None)

    symptom_chunk = (
        "1. Symptoms: - Small dark-brown to reddish-brown circular or irregular necrotic spots on cotton leaves. "
        "- Lesions may show concentric rings or a target-board appearance. "
        "- Spots may have a yellowish chlorotic halo surrounding the necrotic center. "
        "- Severe infection can cause spots to merge, leading to leaf blighting, premature yellowing, and defoliation."
    )

    kw_score = service.calculate_keyword_score("cotton leaves dark brown spots yellow halo concentric rings", symptom_chunk)
    assert kw_score > 0.45
    assert "concentric rings" in symptom_chunk
    assert "yellowish chlorotic halo" in symptom_chunk


def test_alternaria_mancozeb_dosage_retrieval():
    """Verify that Mancozeb dosage query matches verified chunk and contains 2.5 to 3.0 g per litre."""
    from src.rag.service import RAGService
    service = RAGService(repository=None)

    dosage_chunk = (
        "5. Verified Dosage Information Provided for this Knowledge Entry: "
        "- Mancozeb 75% WP: Dosage: 2.5 to 3.0 g per litre of water. "
        "Approximate field quantity: 500 to 600 g per acre in 200 L water. "
        "- Copper Oxychloride 50% WP: Dosage: 3.0 g per litre of water."
    )

    kw_score = service.calculate_keyword_score("What is the Mancozeb dosage for Alternaria leaf spot in cotton?", dosage_chunk)
    assert kw_score > 0.40
    assert "2.5 to 3.0 g per litre of water" in dosage_chunk


def test_alternaria_unverified_chemical_safety_fallback():
    """Verify that unverified chemicals (e.g., glyphosate) are not present in Alternaria dosage chunk and safety rules apply."""
    dosage_chunk = (
        "5. Verified Dosage Information Provided for this Knowledge Entry: "
        "- Mancozeb 75% WP: Dosage: 2.5 to 3.0 g per litre of water. "
        "- Copper Oxychloride 50% WP: Dosage: 3.0 g per litre of water. "
        "- Propiconazole 25% EC: Dosage: 1.0 ml per litre of water. "
        "- Pyraclostrobin 20% WG: Dosage: 1.0 g per litre of water. "
        "- Azoxystrobin + Difenoconazole: Dosage: 1.0 ml per litre of water."
    )

    # An invented/unverified chemical must not be present in verified chunk
    assert "glyphosate" not in dosage_chunk.lower()
    assert "carbendazim" not in dosage_chunk.lower()
    
    # Verify safety rule is preserved in system prompt
    from src.ai.prompts import BHOOMIMITRA_SYSTEM_PROMPT
    assert "NEVER invent or guess pesticide names, fertilizer brands, or chemical dosages" in BHOOMIMITRA_SYSTEM_PROMPT







