from uuid import UUID
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


class KnowledgeDocumentCreate(BaseModel):
    """Schema for creating or uploading a new knowledge document."""
    title: str = Field(..., description="Title of the agricultural document")
    source: str = Field(..., description="Source (e.g., ICAR, KVK, Govt, University, Fertilizer Manual)")
    category: str = Field(..., description="Category (e.g., Pest Control, Fertilizer, Advisory, Crop Calendar)")
    language: str = Field(default="te", description="Language of document (te, hi, en)")
    state: Optional[str] = Field(default=None, description="Target state if applicable (e.g., Telangana)")
    crop: Optional[str] = Field(default=None, description="Target crop if applicable (e.g., Chilli)")


class KnowledgeDocumentResponse(BaseModel):
    """Schema for returned Knowledge Document details."""
    id: UUID
    title: str
    source: str
    category: str
    language: str
    state: Optional[str] = None
    crop: Optional[str] = None
    pdf_path: Optional[str] = None
    chunk_count: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class KnowledgeChunkResponse(BaseModel):
    """Schema for individual document chunks."""
    id: UUID
    document_id: UUID
    chunk_index: int
    chunk_text: str
    embedding_id: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class RAGSearchQuery(BaseModel):
    """Request schema for semantic vector search."""
    query: str = Field(..., min_length=2, description="Search query or farmer question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    category: Optional[str] = Field(default=None, description="Filter by document category")
    language: Optional[str] = Field(default=None, description="Filter by language")
    state: Optional[str] = Field(default=None, description="Filter by state")
    crop: Optional[str] = Field(default=None, description="Filter by crop")


class RAGSearchResult(BaseModel):
    """SearchResult schema containing chunk text, source, document title, and similarity score."""
    chunk_id: UUID
    document_id: UUID
    document_title: str
    source: str
    category: str
    language: str
    state: Optional[str] = None
    crop: Optional[str] = None
    chunk_text: str
    similarity_score: float

    model_config = ConfigDict(from_attributes=True)


class RAGGenerateRequest(BaseModel):
    """Request to generate a grounded RAG AI answer."""
    farmer_id: UUID
    message: str
    conversation_id: Optional[UUID] = None


class RAGGenerateResponse(BaseModel):
    """Standard grounded AI response schema containing Answer, Reasoning, Sources, and Confidence score."""
    answer: str
    reasoning: str
    source_documents: List[RAGSearchResult]
    confidence_score: float


class RebuildIndexResponse(BaseModel):
    """Response schema after rebuilding vector index."""
    success: bool
    message: str
    documents_processed: int
    total_chunks: int
