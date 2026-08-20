from uuid import UUID
from typing import List, Optional
from fastapi import APIRouter, Depends, File, UploadFile, Form, HTTPException, Query, status
from src.rag.schemas import (
    KnowledgeDocumentResponse,
    RAGSearchResult,
    RAGGenerateRequest,
    RAGGenerateResponse,
    RebuildIndexResponse,
)
from src.rag.service import RAGService
from src.rag.dependencies import get_rag_service

router = APIRouter()


@router.post(
    "/upload",
    response_model=KnowledgeDocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload & Index Agriculture Knowledge PDF Document",
    tags=["RAG Knowledge Engine"],
)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    source: str = Form("ICAR Publications"),
    category: str = Form("general"),
    language: str = Form("en"),
    state: Optional[str] = Form(None),
    crop: Optional[str] = Form(None),
    service: RAGService = Depends(get_rag_service),
):
    """
    Ingests an agricultural PDF or text file through the Document Pipeline.
    Accepts multipart/form-data file uploads with metadata fields.
    Renders 'Choose File' button in FastAPI Swagger UI.
    """
    filename = file.filename or "document.pdf"
    content_type = file.content_type or ""

    is_pdf_ext = filename.lower().endswith(".pdf")
    is_txt_ext = filename.lower().endswith(".txt")
    is_pdf_type = "pdf" in content_type.lower()
    is_txt_type = "text" in content_type.lower() or "plain" in content_type.lower()

    if not (is_pdf_ext or is_txt_ext or is_pdf_type or is_txt_type):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Please upload a PDF or text file."
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    if not (file_bytes.startswith(b"%PDF") or is_pdf_ext or is_txt_ext):
        try:
            file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported binary file format. Please upload a valid PDF or text document."
            )

    document_title = title or filename or "Untitled Agricultural Document"
    return await service.upload_and_index_document(
        file_bytes=file_bytes,
        filename=filename,
        title=document_title,
        source=source,
        category=category,
        language=language,
        state=state,
        crop=crop,
    )


@router.post(
    "/rebuild",
    response_model=RebuildIndexResponse,
    summary="Rebuild RAG Vector Index Across All Documents",
    tags=["RAG Knowledge Engine"],
)
async def rebuild_index(
    service: RAGService = Depends(get_rag_service),
):
    """
    Re-extracts text, re-chunks, and re-embeds all existing knowledge documents in the database.
    """
    return await service.rebuild_index()


@router.get(
    "/search",
    response_model=List[RAGSearchResult],
    summary="True Hybrid Search (Vector + Keyword + Metadata Re-Ranking) Over Knowledge Engine",
    tags=["RAG Knowledge Engine"],
)
async def search_knowledge(
    query: str = Query(..., min_length=2, description="Search query or farmer question"),
    top_k: int = Query(5, ge=1, le=20, description="Top K results to return (default 5)"),
    category: Optional[str] = Query(None, description="Filter category"),
    language: Optional[str] = Query(None, description="Filter language"),
    state: Optional[str] = Query(None, description="Filter state"),
    crop: Optional[str] = Query(None, description="Filter crop"),
    source: Optional[str] = Query(None, description="Filter source"),
    service: RAGService = Depends(get_rag_service),
):
    """
    Performs True Hybrid Search (Vector Cosine Similarity + Keyword TF-IDF + Metadata Re-Ranking)
    and returns top K highest-ranked text chunks with scores and page metadata.
    """
    return await service.hybrid_search_knowledge(
        query=query,
        top_k=top_k,
        category=category,
        language=language,
        state=state,
        crop=crop,
        source=source,
    )


@router.get(
    "/documents",
    response_model=List[KnowledgeDocumentResponse],
    summary="List Knowledge Base Documents",
    tags=["RAG Knowledge Engine"],
)
async def list_documents(
    category: Optional[str] = Query(None),
    source: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    service: RAGService = Depends(get_rag_service),
):
    """
    Lists uploaded knowledge documents with filtering and pagination.
    """
    docs = await service.repository.list_documents(
        category=category,
        source=source,
        skip=skip,
        limit=limit,
    )
    return [KnowledgeDocumentResponse.model_validate(d) for d in docs]


@router.delete(
    "/document/{id}",
    summary="Delete Knowledge Document by ID",
    tags=["RAG Knowledge Engine"],
)
async def delete_document(
    id: UUID,
    service: RAGService = Depends(get_rag_service),
):
    """
    Deletes document, associated chunks, vector embedding metadata, and stored PDF file.
    """
    deleted = await service.repository.delete_document(id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {id} not found."
        )
    return {"success": True, "message": "Document and associated vector chunks deleted successfully."}


@router.post(
    "/query",
    response_model=RAGGenerateResponse,
    summary="Generate Grounded RAG AI Answer for Farmer",
    tags=["RAG Knowledge Engine"],
)
async def query_rag_answer(
    request: RAGGenerateRequest,
    service: RAGService = Depends(get_rag_service),
):
    """
    Retrieves trusted agricultural knowledge, combines with Farmer Memory, and generates a grounded response.
    """
    return await service.generate_rag_response(
        farmer_id=request.farmer_id,
        message=request.message,
        conversation_id=request.conversation_id,
    )
