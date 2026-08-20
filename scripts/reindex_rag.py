import asyncio
import os
import sys

# Ensure UTF-8 stdout encoding on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select
from src.core.database import AsyncSessionLocal
from src.rag.models import KnowledgeDocument, KnowledgeChunk
from src.rag.repository import RAGRepository
from src.rag.service import RAGService


async def main():
    async with AsyncSessionLocal() as session:
        repo = RAGRepository(session)
        service = RAGService(repo)

        # Update metadata for cotton disease guide if generic
        docs = (await session.execute(select(KnowledgeDocument))).scalars().all()
        for doc in docs:
            if doc.pdf_path and "cotton_disease_guide" in doc.pdf_path:
                doc.title = "ICAR Cotton Leaf Curl Virus Advisory"
                doc.crop = "Cotton"
                doc.category = "Pest & Disease Control"
                doc.source = "ICAR Publications"
                doc.language = "en"
                doc.state = "All India"
                session.add(doc)
        await session.commit()

        # Check and register cotton fertilizer guide if not present
        fertilizer_doc_path = os.path.join(os.getcwd(), "static", "rag_docs", "cotton_fertilizer_management_guide.txt")
        fert_doc = (await session.execute(
            select(KnowledgeDocument).where(KnowledgeDocument.pdf_path == fertilizer_doc_path)
        )).scalar_one_or_none()

        if not fert_doc:
            print("[REINDEX] Registering new document: PJTSAU & ICAR Cotton Fertilizer Guide...")
            fert_doc = await repo.create_document(
                title="PJTSAU & ICAR Cotton Fertilizer and Nutrient Guide",
                source="PJTSAU & ICAR-CICR Package of Practices",
                category="Fertilizer Management",
                language="en",
                state="Telangana",
                crop="Cotton",
                pdf_path=fertilizer_doc_path,
            )

        # Check and register cotton Alternaria leaf spot guide if not present
        alternaria_doc_path = os.path.join(os.getcwd(), "static", "rag_docs", "cotton_alternaria_leaf_spot_guide.txt")
        alt_doc = (await session.execute(
            select(KnowledgeDocument).where(KnowledgeDocument.pdf_path == alternaria_doc_path)
        )).scalar_one_or_none()

        if not alt_doc:
            print("[REINDEX] Registering new document: Cotton Alternaria Leaf Spot Disease Management Guide...")
            alt_doc = await repo.create_document(
                title="Cotton Alternaria Leaf Spot Disease Management Guide",
                source="PJTSAU & ICAR-CICR Disease Advisory",
                category="disease",
                language="en",
                state="Telangana",
                crop="cotton",
                pdf_path=alternaria_doc_path,
            )

        print("[REINDEX] Rebuilding RAG index across all documents...")

        resp = await service.rebuild_index()
        print(f"[REINDEX] Result: {resp.message}")
        print(f"[REINDEX] Processed {resp.documents_processed} documents, created {resp.total_chunks} clean chunks.")

        # Print all chunks in DB
        chunks = (await session.execute(select(KnowledgeChunk))).scalars().all()
        print(f"\n[REINDEX] Total Knowledge Chunks in DB after reindex: {len(chunks)}")
        for c in chunks:
            print(f"  Chunk ID: {c.id} | Doc ID: {c.document_id} | Page: {c.page_number} | Index: {c.chunk_index}")
            print(f"  Text:\n{c.chunk_text}\n")



if __name__ == "__main__":
    asyncio.run(main())
