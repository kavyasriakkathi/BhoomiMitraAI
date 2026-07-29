import uuid
from typing import List, Optional, Tuple
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from src.rag.models import KnowledgeDocument, KnowledgeChunk, EmbeddingMetadata
from src.core.logging import logger


class RAGRepository:
    """Async repository for RAG documents, chunks, and vector embedding metadata."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_document(
        self,
        title: str,
        source: str,
        category: str,
        language: str = "te",
        state: Optional[str] = None,
        crop: Optional[str] = None,
        pdf_path: Optional[str] = None,
    ) -> KnowledgeDocument:
        doc = KnowledgeDocument(
            id=uuid.uuid4(),
            title=title,
            source=source,
            category=category,
            language=language,
            state=state,
            crop=crop,
            pdf_path=pdf_path,
            chunk_count=0
        )
        self.session.add(doc)
        await self.session.commit()
        await self.session.refresh(doc)
        return doc

    async def get_document(self, doc_id: uuid.UUID) -> Optional[KnowledgeDocument]:
        result = await self.session.execute(
            select(KnowledgeDocument)
            .options(selectinload(KnowledgeDocument.chunks))
            .where(KnowledgeDocument.id == doc_id)
        )
        return result.scalar_one_or_none()

    async def list_documents(
        self,
        category: Optional[str] = None,
        source: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> List[KnowledgeDocument]:
        query = select(KnowledgeDocument).order_by(KnowledgeDocument.created_at.desc())
        if category:
            query = query.where(KnowledgeDocument.category == category)
        if source:
            query = query.where(KnowledgeDocument.source == source)
        query = query.offset(skip).limit(limit)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def delete_document(self, doc_id: uuid.UUID) -> bool:
        doc = await self.get_document(doc_id)
        if not doc:
            return False
        await self.session.delete(doc)
        await self.session.commit()
        return True

    async def create_chunks_and_embeddings(
        self,
        doc_id: uuid.UUID,
        chunks_with_vectors: List[dict]
    ) -> List[KnowledgeChunk]:
        """
        chunks_with_vectors is a list of dicts:
        [
          {
            "chunk_index": 0,
            "chunk_text": "...",
            "vector": [0.1, -0.2, ...],
            "dimension": 768
          }, ...
        ]
        """
        created_chunks = []
        for item in chunks_with_vectors:
            chunk_id = uuid.uuid4()
            embedding_id = f"emb_{chunk_id.hex[:12]}"

            chunk = KnowledgeChunk(
                id=chunk_id,
                document_id=doc_id,
                chunk_index=item["chunk_index"],
                page_number=item.get("page_number", 1),
                chunk_text=item["chunk_text"],
                embedding_id=embedding_id,
            )

            self.session.add(chunk)

            emb_meta = EmbeddingMetadata(
                id=uuid.uuid4(),
                chunk_id=chunk_id,
                embedding_id=embedding_id,
                vector=item["vector"],
                dimension=item.get("dimension", len(item["vector"])),
            )
            self.session.add(emb_meta)
            created_chunks.append(chunk)

        # Update document chunk count
        doc = await self.get_document(doc_id)
        if doc:
            doc.chunk_count = len(chunks_with_vectors)
            self.session.add(doc)

        await self.session.commit()
        return created_chunks

    async def delete_chunks_by_document(self, doc_id: uuid.UUID):
        result = await self.session.execute(
            select(KnowledgeChunk.id).where(KnowledgeChunk.document_id == doc_id)
        )
        chunk_ids = list(result.scalars().all())
        if chunk_ids:
            await self.session.execute(
                delete(EmbeddingMetadata).where(EmbeddingMetadata.chunk_id.in_(chunk_ids))
            )
            await self.session.execute(
                delete(KnowledgeChunk).where(KnowledgeChunk.document_id == doc_id)
            )
            await self.session.commit()

    async def get_all_chunks_with_embeddings(
        self,
        category: Optional[str] = None,
        language: Optional[str] = None,
        state: Optional[str] = None,
        crop: Optional[str] = None,
    ) -> List[Tuple[KnowledgeChunk, EmbeddingMetadata, KnowledgeDocument]]:
        """
        Retrieves all chunks, their embedding vectors, and parent document metadata,
        applying filters if specified.
        """
        query = (
            select(KnowledgeChunk, EmbeddingMetadata, KnowledgeDocument)
            .join(EmbeddingMetadata, KnowledgeChunk.id == EmbeddingMetadata.chunk_id)
            .join(KnowledgeDocument, KnowledgeChunk.document_id == KnowledgeDocument.id)
        )

        if category:
            query = query.where(KnowledgeDocument.category == category)
        if language:
            query = query.where(KnowledgeDocument.language == language)
        if state:
            query = query.where(
                (KnowledgeDocument.state == state) | (KnowledgeDocument.state == "All India") | (KnowledgeDocument.state.is_(None))
            )
        if crop:
            query = query.where(
                (KnowledgeDocument.crop == crop) | (KnowledgeDocument.crop == "All Crops") | (KnowledgeDocument.crop.is_(None))
            )

        result = await self.session.execute(query)
        return list(result.all())
