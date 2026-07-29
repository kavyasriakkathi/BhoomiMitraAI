import base64
import re
from uuid import UUID
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
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


def _parse_multipart_form(body_bytes: bytes, content_type: str):
    """
    Native zero-dependency multipart/form-data boundary parser.
    Avoids requiring third-party python-multipart library.
    """
    fields = {}
    file_bytes = b""
    filename = "document.pdf"

    if "boundary=" not in content_type:
        return fields, file_bytes, filename

    boundary = content_type.split("boundary=")[-1].strip().strip('"').encode()
    parts = body_bytes.split(b"--" + boundary)

    for part in parts:
        if not part or part.startswith(b"--"):
            continue
        if b"\r\n\r\n" in part:
            header_bytes, content_bytes = part.split(b"\r\n\r\n", 1)
            if content_bytes.endswith(b"\r\n"):
                content_bytes = content_bytes[:-2]

            header_str = header_bytes.decode("utf-8", errors="ignore")
            name_match = re.search(r'name="([^"]+)"', header_str)
            filename_match = re.search(r'filename="([^"]+)"', header_str)

            if name_match:
                field_name = name_match.group(1)
                if filename_match:
                    filename = filename_match.group(1)
                    file_bytes = content_bytes
                else:
                    fields[field_name] = content_bytes.decode("utf-8", errors="ignore")

    return fields, file_bytes, filename


@router.post(
    "/upload",
    response_model=KnowledgeDocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload & Index Agriculture Knowledge PDF Document",
    tags=["RAG Knowledge Engine"],
)
async def upload_document(
    request: Request,
    service: RAGService = Depends(get_rag_service),
):
    """
    Ingests an agricultural PDF or text file through the Document Pipeline.
    Supports both multipart/form-data file uploads and application/json requests.
    """
    content_type = request.headers.get("content-type", "")

    title = None
    source = "ICAR Publications"
    category = "Pest Control"
    language = "te"
    state = None
    crop = None
    file_bytes = b""
    filename = "document.pdf"

    if "application/json" in content_type:
        try:
            json_body = await request.json()
            title = json_body.get("title")
            source = json_body.get("source", "ICAR Publications")
            category = json_body.get("category", "Pest Control")
            language = json_body.get("language", "te")
            state = json_body.get("state")
            crop = json_body.get("crop")
            filename = json_body.get("filename", "document.pdf")
            
            if "file_base64" in json_body:
                file_bytes = base64.b64decode(json_body["file_base64"])
            elif "file_content" in json_body:
                file_bytes = json_body["file_content"].encode("utf-8")
        except Exception as json_err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON request body: {json_err}"
            )
    else:
        body_bytes = await request.body()
        fields, file_bytes, filename = _parse_multipart_form(body_bytes, content_type)

        title = fields.get("title")
        source = fields.get("source", source)
        category = fields.get("category", category)
        language = fields.get("language", language)
        state = fields.get("state")
        crop = fields.get("crop")

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty or missing."
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
