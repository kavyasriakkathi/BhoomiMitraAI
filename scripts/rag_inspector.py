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
from src.rag.models import KnowledgeDocument, KnowledgeChunk, EmbeddingMetadata
from src.rag.repository import RAGRepository
from src.rag.service import RAGService


async def main():
    async with AsyncSessionLocal() as session:
        # 1. Print all documents in the database
        docs = (await session.execute(select(KnowledgeDocument))).scalars().all()
        print(f"Total Knowledge Documents in DB: {len(docs)}")
        for d in docs:
            print(f"  Doc ID: {d.id} | Title: '{d.title}' | Source: '{d.source}' | Category: '{d.category}' | State: '{d.state}' | Crop: '{d.crop}' | PDF: '{d.pdf_path}' | Chunks: {d.chunk_count}")

        # 2. Print all chunks in DB
        chunks = (await session.execute(select(KnowledgeChunk))).scalars().all()
        print(f"\nTotal Knowledge Chunks in DB: {len(chunks)}")
        for c in chunks:
            print(f"  Chunk ID: {c.id} (Doc: {c.document_id}, Page: {c.page_number}, Index: {c.chunk_index}):")
            print(f"    Text: {c.chunk_text[:200]}...")

        # 3. Test queries
        repo = RAGRepository(session)
        service = RAGService(repo)

        queries = [
            "What is Alternaria leaf spot in cotton?",
            "cotton leaf dark brown spots yellow halo",
            "What is the Mancozeb dosage for Alternaria leaf spot?",
            "ఈ వ్యాధికి ఎంత మందు వేయాలి?",
            "What is the urea dosage for cotton during flowering stage?",
        ]




        for q in queries:
            print(f"\n==================================================")
            print(f"QUERY: '{q}'")
            print(f"==================================================")
            results = await service.search_knowledge(query=q, top_k=5)
            if not results:
                print("  No results found.")
            for i, res in enumerate(results, 1):
                print(f"  [{i}] Doc: '{res.document_title}' | Similarity: {res.similarity_score:.4f} (vec: {res.vector_score:.4f}, kw: {res.keyword_score:.4f}, meta: {res.metadata_score:.4f})")
                print(f"      Text: {res.chunk_text}")


if __name__ == "__main__":
    asyncio.run(main())
