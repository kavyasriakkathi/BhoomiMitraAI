"""
Focused tests proving non-blocking async embedding execution for RAG.

Verifies:
1. Async embedding offloads the synchronous blocking call to a worker thread via asyncio.to_thread.
2. The async event loop remains fully responsive and unblocked during embedding generation.
3. The embedding vector returned by generate_embedding_async matches generate_embedding exactly.
4. RAG search still functions and returns valid grounded results.
5. Concurrency is handled safely without event loop stalls.
"""

import asyncio
import time
import pytest
from unittest.mock import MagicMock

from src.rag.service import RAGService, clear_static_baseline_cache


@pytest.fixture(autouse=True)
def clean_rag_cache():
    clear_static_baseline_cache()
    yield
    clear_static_baseline_cache()


@pytest.mark.asyncio
async def test_async_embedding_does_not_block_event_loop():
    """
    Proves that a slow/blocking synchronous embedding call does NOT block
    other coroutines running on the asyncio event loop.
    """
    rag_service = RAGService(repository=None)

    # Simulate a blocking synchronous embedding API call taking 150ms
    def slow_sync_embedding(text: str, dimension: int = 768):
        time.sleep(0.15)
        return [0.05] * dimension

    rag_service.generate_embedding = slow_sync_embedding

    # Run a concurrent lightweight coroutine ticking every 20ms on the event loop
    heartbeat_ticks = 0
    stop_heartbeat = False

    async def heartbeat():
        nonlocal heartbeat_ticks
        while not stop_heartbeat:
            await asyncio.sleep(0.02)
            heartbeat_ticks += 1

    heartbeat_task = asyncio.create_task(heartbeat())

    # Execute the async embedding operation
    result = await rag_service.generate_embedding_async("fertilizer for paddy")

    stop_heartbeat = True
    await heartbeat_task

    # If the call had blocked the event loop synchronously, heartbeat_ticks would be 0
    # Because it was offloaded to asyncio.to_thread, the event loop ticked multiple times
    assert heartbeat_ticks >= 3, f"Event loop was blocked! Ticks observed: {heartbeat_ticks}"
    assert len(result) == 768
    assert result[0] == 0.05


@pytest.mark.asyncio
async def test_embedding_result_is_unchanged():
    """
    Verifies that generate_embedding_async returns the exact same vector
    values as the synchronous generate_embedding.
    """
    rag_service = RAGService(repository=None)
    sample_text = "What fertilizer dosage should I use for paddy in Telangana?"

    sync_vector = rag_service.generate_embedding(sample_text)
    async_vector = await rag_service.generate_embedding_async(sample_text)

    assert sync_vector == async_vector
    assert len(async_vector) == 768
    # Also verify alias
    alias_vector = await rag_service.a_generate_embedding(sample_text)
    assert alias_vector == sync_vector


@pytest.mark.asyncio
async def test_hybrid_search_runs_non_blocking_query_embedding():
    """
    Verifies that hybrid_search_knowledge runs the query embedding asynchronously
    without blocking concurrent tasks on the event loop.
    """
    rag_service = RAGService(repository=None)

    original_generate = rag_service.generate_embedding

    def delayed_query_generate(text: str, dimension: int = 768):
        time.sleep(0.12)
        return original_generate(text, dimension)

    rag_service.generate_embedding = delayed_query_generate

    heartbeat_ticks = 0
    stop_heartbeat = False

    async def heartbeat():
        nonlocal heartbeat_ticks
        while not stop_heartbeat:
            await asyncio.sleep(0.02)
            heartbeat_ticks += 1

    heartbeat_task = asyncio.create_task(heartbeat())

    results = await rag_service.hybrid_search_knowledge(
        query="What fertilizer should I use for paddy?",
        top_k=3,
        crop="Paddy",
    )

    stop_heartbeat = True
    await heartbeat_task

    assert heartbeat_ticks >= 2, f"Event loop was blocked during hybrid_search! Ticks: {heartbeat_ticks}"
    assert len(results) > 0
    assert results[0].chunk_text is not None
    assert results[0].similarity_score > 0.0
