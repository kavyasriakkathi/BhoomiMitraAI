import os
import io
import math
import hashlib
import uuid
from typing import List, Optional, Tuple, Dict, Any
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PyPDF2 = None
    PYPDF2_AVAILABLE = False

from fastapi import HTTPException, status
from src.core.logging import logger
from src.rag.repository import RAGRepository
from src.rag.schemas import (
    KnowledgeDocumentResponse,
    RAGSearchResult,
    RAGGenerateResponse,
    RebuildIndexResponse,
)
from src.rag.prompts import build_rag_context_prompt
from src.config import get_settings


class RAGService:
    """
    RAG Knowledge Engine Service.
    Handles Document Upload, Text Extraction, Chunking, Embedding Generation,
    Vector Cosine Similarity Search, Index Rebuilding, and Grounded AI Response Generation.
    """

    def __init__(self, repository: RAGRepository):
        self.repository = repository
        self.upload_dir = os.path.join(os.getcwd(), "static", "rag_docs")
        os.makedirs(self.upload_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Text Extraction
    # ------------------------------------------------------------------
    def extract_text_from_file(self, file_bytes: bytes, filename: str) -> str:
        """Extract text from uploaded PDF or text file."""
        text = ""
        is_pdf = filename.lower().endswith(".pdf") or file_bytes.startswith(b"%PDF")

        if is_pdf and PYPDF2_AVAILABLE:

            try:
                reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                extracted_pages = []
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        extracted_pages.append(page_text)
                text = "\n\n".join(extracted_pages)
                logger.info(f"PyPDF2 extracted {len(text)} chars from {filename}")
            except Exception as pdf_err:
                logger.warning(f"PyPDF2 extraction failed for {filename}: {pdf_err}. Using text decoding fallback.")

        if not text.strip():
            # Fallback to UTF-8 / latin-1 decoding
            try:
                text = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text = file_bytes.decode("latin-1", errors="ignore")

        cleaned_text = " ".join(text.split())
        if not cleaned_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unable to extract readable text from the provided file."
            )
        return cleaned_text

    # ------------------------------------------------------------------
    # 2. Text Chunking
    # ------------------------------------------------------------------
    def chunk_text(self, text: str, max_chunk_size: int = 600, overlap: int = 100) -> List[str]:
        """Recursive/character-based chunking with sliding window overlap."""
        if len(text) <= max_chunk_size:
            return [text]

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + max_chunk_size
            if end >= text_len:
                chunk = text[start:]
                chunks.append(chunk)
                break

            # Try to break at paragraph or sentence end boundary
            boundary = text.rfind("\n", start, end)
            if boundary == -1 or boundary <= start:
                boundary = text.rfind(". ", start, end)
            if boundary == -1 or boundary <= start:
                boundary = text.rfind(" ", start, end)
            if boundary == -1 or boundary <= start:
                boundary = end

            chunk = text[start:boundary].strip()
            if chunk:
                chunks.append(chunk)
            
            start = max(boundary + 1, start + max_chunk_size - overlap)

        return chunks

    # ------------------------------------------------------------------
    # 3. Embedding Generation (Gemini API with Local Vector Fallback)
    # ------------------------------------------------------------------
    def generate_embedding(self, text: str, dimension: int = 768) -> List[float]:
        """
        Generates embedding vector. Uses Google Gemini Embeddings API if configured,
        or a deterministic normalized local feature vector fallback.
        """
        settings = get_settings()
        if settings.google_gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.google_gemini_api_key)
                res = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document"
                )
                if res and "embedding" in res:
                    vec = res["embedding"]
                    # Normalize vector
                    norm = math.sqrt(sum(x * x for x in vec))
                    if norm > 0:
                        vec = [x / norm for x in vec]
                    return vec
            except Exception as err:
                logger.warning(f"Gemini Embeddings API call failed, using deterministic local embedding: {err}")

        # Local Deterministic Cosine-compatible Feature Hashing Embedding Vector
        vec = [0.0] * dimension
        words = text.lower().split()
        if not words:
            return vec

        for word in words:
            # Generate hash value for word
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % dimension
            sign = 1.0 if (h & 1) else -1.0
            vec[idx] += sign

        # L2 Normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    # ------------------------------------------------------------------
    # 4. Vector Cosine Similarity Search
    # ------------------------------------------------------------------
    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """Compute Cosine Similarity between two normalized float vectors."""
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    async def search_knowledge(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        language: Optional[str] = None,
        state: Optional[str] = None,
        crop: Optional[str] = None,
    ) -> List[RAGSearchResult]:
        """Performs semantic vector search across indexed knowledge chunks."""
        query_vec = self.generate_embedding(query)
        chunks_data = await self.repository.get_all_chunks_with_embeddings(
            category=category, language=language, state=state, crop=crop
        )

        results = []
        for chunk, emb_meta, doc in chunks_data:
            chunk_vec = emb_meta.vector
            score = self.cosine_similarity(query_vec, chunk_vec)
            
            # Boost score if query keywords directly match text
            query_words = set(query.lower().split())
            chunk_words = set(chunk.chunk_text.lower().split())
            overlap = len(query_words.intersection(chunk_words))
            if query_words:
                keyword_boost = (overlap / len(query_words)) * 0.2
                score = min(1.0, score + keyword_boost)

            results.append(
                RAGSearchResult(
                    chunk_id=chunk.id,
                    document_id=doc.id,
                    document_title=doc.title,
                    source=doc.source,
                    category=doc.category,
                    language=doc.language,
                    state=doc.state,
                    crop=doc.crop,
                    chunk_text=chunk.chunk_text,
                    similarity_score=round(score, 4),
                )
            )

        # Sort by similarity score descending
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # 5. Document Ingestion Pipeline
    # ------------------------------------------------------------------
    async def upload_and_index_document(
        self,
        file_bytes: bytes,
        filename: str,
        title: str,
        source: str,
        category: str,
        language: str = "te",
        state: Optional[str] = None,
        crop: Optional[str] = None,
    ) -> KnowledgeDocumentResponse:
        """PDF Upload -> Text Extraction -> Chunking -> Embedding Generation -> Vector & DB Storage."""
        # 1. Save file to disk
        safe_filename = f"{uuid.uuid4().hex[:8]}_{filename}"
        file_path = os.path.join(self.upload_dir, safe_filename)
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        # 2. Extract text
        extracted_text = self.extract_text_from_file(file_bytes, filename)

        # 3. Create Document Record
        doc = await self.repository.create_document(
            title=title or filename,
            source=source,
            category=category,
            language=language,
            state=state,
            crop=crop,
            pdf_path=file_path,
        )

        # 4. Chunk text
        text_chunks = self.chunk_text(extracted_text)

        # 5. Generate embeddings for each chunk
        chunks_with_vectors = []
        for idx, chunk_str in enumerate(text_chunks):
            vec = self.generate_embedding(chunk_str)
            chunks_with_vectors.append({
                "chunk_index": idx,
                "chunk_text": chunk_str,
                "vector": vec,
                "dimension": len(vec),
            })

        # 6. Save chunks & vector metadata
        await self.repository.create_chunks_and_embeddings(doc.id, chunks_with_vectors)

        # Fetch updated document
        updated_doc = await self.repository.get_document(doc.id)
        return KnowledgeDocumentResponse.model_validate(updated_doc)

    # ------------------------------------------------------------------
    # 6. Rebuild Index
    # ------------------------------------------------------------------
    async def rebuild_index(self) -> RebuildIndexResponse:
        """Re-extracts text, re-chunks, and re-embeds all existing documents."""
        docs = await self.repository.list_documents(limit=1000)
        processed_count = 0
        total_chunks_created = 0

        for doc in docs:
            # Delete old chunks
            await self.repository.delete_chunks_by_document(doc.id)

            extracted_text = ""
            if doc.pdf_path and os.path.exists(doc.pdf_path):
                with open(doc.pdf_path, "rb") as f:
                    file_bytes = f.read()
                extracted_text = self.extract_text_from_file(file_bytes, os.path.basename(doc.pdf_path))
            
            if not extracted_text:
                extracted_text = f"Document Title: {doc.title}. Source: {doc.source}. Category: {doc.category}. State: {doc.state}. Crop: {doc.crop}."

            text_chunks = self.chunk_text(extracted_text)
            chunks_with_vectors = []
            for idx, chunk_str in enumerate(text_chunks):
                vec = self.generate_embedding(chunk_str)
                chunks_with_vectors.append({
                    "chunk_index": idx,
                    "chunk_text": chunk_str,
                    "vector": vec,
                    "dimension": len(vec),
                })

            created = await self.repository.create_chunks_and_embeddings(doc.id, chunks_with_vectors)
            processed_count += 1
            total_chunks_created += len(created)

        logger.info(f"Rebuilt index for {processed_count} documents with {total_chunks_created} total chunks.")
        return RebuildIndexResponse(
            success=True,
            message="Vector index rebuilt successfully across all documents.",
            documents_processed=processed_count,
            total_chunks=total_chunks_created,
        )

    # ------------------------------------------------------------------
    # 7. Grounded RAG AI Response Generation
    # ------------------------------------------------------------------
    async def generate_rag_response(
        self,
        farmer_id: uuid.UUID,
        message: str,
        conversation_id: Optional[uuid.UUID] = None,
    ) -> RAGGenerateResponse:
        """
        1. Fetch Farmer Memory & Profile
        2. Perform Semantic Vector Search for top trusted agricultural chunks
        3. Build Grounded System Prompt
        4. Query Gemini API
        5. Return Answer, Reasoning, Source Documents, and Confidence score
        """
        # 1. Fetch profile & memory
        from src.ai.repository import AIRepository
        from src.memory.service import FarmerMemoryService
        from src.memory.repository import FarmerMemoryRepository

        ai_repo = AIRepository(self.repository.session)
        profile = await ai_repo.get_farmer_profile(farmer_id)
        
        state = profile.state if profile else None
        crop = profile.current_crop if profile else None

        mem_repo = FarmerMemoryRepository(self.repository.session)
        mem_service = FarmerMemoryService(mem_repo)
        memory_context = await mem_service.format_memory_for_system_prompt(farmer_id)

        profile_context = f"Crop: {crop or 'General'}, State: {state or 'All India'}, District: {profile.district if profile else 'Not set'}"

        # 2. Retrieve Top Knowledge Chunks
        search_results = await self.search_knowledge(
            query=message,
            top_k=4,
            state=state,
            crop=crop,
        )

        # Format retrieved knowledge chunks into prompt text
        knowledge_snippets = []
        for idx, res in enumerate(search_results, 1):
            snippet = f"[Source {idx}: {res.document_title} ({res.source} - {res.category})]\n{res.chunk_text}"
            knowledge_snippets.append(snippet)

        retrieved_knowledge_text = "\n\n".join(knowledge_snippets)

        # 3. Build Grounded System Prompt
        full_system_prompt = build_rag_context_prompt(
            farmer_profile_context=profile_context,
            farmer_memory_context=memory_context,
            retrieved_knowledge_context=retrieved_knowledge_text,
        )

        # 4. Call Gemini
        from src.ai.gemini_client import generate_response
        
        try:
            ai_text = await generate_response(
                system_prompt=full_system_prompt,
                conversation_history=[],
                user_message=message,
            )
        except Exception as e:
            logger.warning(f"Gemini API error in RAG response generation: {e}")
            ai_text = None

        if not ai_text:
            # Fallback response grounded in top search result
            if search_results:
                top_res = search_results[0]
                ai_text = f"Based on {top_res.document_title} ({top_res.source}): {top_res.chunk_text}"
            else:
                ai_text = "Please follow standard agricultural management practices for your crop and consult local Krishi Vigyan Kendra (KVK)."

        # Determine confidence score based on top search match
        top_score = search_results[0].similarity_score if search_results else 0.5
        confidence = min(0.99, max(0.60, round(top_score * 0.9 + 0.1, 2)))

        # Format reasoning and source references
        sources = [f"{r.document_title} ({r.source})" for r in search_results] if search_results else ["General Indian Agronomic Guide"]
        reasoning = f"Answer was synthesized by grounding user question '{message}' with top {len(search_results)} verified agricultural document sources and farmer profile ({crop}, {state})."

        return RAGGenerateResponse(
            answer=ai_text,
            reasoning=reasoning,
            source_documents=search_results,
            confidence_score=confidence,
        )
