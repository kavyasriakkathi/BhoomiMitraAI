import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Integer, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from src.core.database import Base

class KnowledgeDocument(Base):
    """
    Agricultural Knowledge Source Document Model.
    Supports ICAR, KVK, Govt PDFs, University Research, Fertilizer Manuals,
    Pesticide Labels, Crop Calendars, and Weather Advisories.
    """
    __tablename__ = "knowledge_documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False, index=True)
    source = Column(String(100), nullable=False, index=True)
    category = Column(String(100), nullable=False, index=True)
    language = Column(String(10), default="te", index=True)
    state = Column(String(100), nullable=True, index=True)
    crop = Column(String(100), nullable=True, index=True)
    pdf_path = Column(String(500), nullable=True)
    chunk_count = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    chunks = relationship("KnowledgeChunk", back_populates="document", cascade="all, delete-orphan")


class KnowledgeChunk(Base):
    """
    Segmented text chunk extracted from Knowledge Document for RAG indexing.
    """
    __tablename__ = "knowledge_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("knowledge_documents.id"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding_id = Column(String(100), nullable=True, index=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("KnowledgeDocument", back_populates="chunks")
    embedding_metadata = relationship("EmbeddingMetadata", back_populates="chunk", uselist=False, cascade="all, delete-orphan")


class EmbeddingMetadata(Base):
    """
    Stores vector embedding floats and metadata for semantic RAG search.
    """
    __tablename__ = "embedding_metadata"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("knowledge_chunks.id"), nullable=False, unique=True, index=True)
    embedding_id = Column(String(100), nullable=False, index=True)
    vector = Column(JSON, nullable=False) # List of float embedding values
    dimension = Column(Integer, default=768)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    chunk = relationship("KnowledgeChunk", back_populates="embedding_metadata")
