"""
Tests for RAG Static Baseline Embedding Cache.

Verifies:
1. First RAG request generates the 5 static baseline embeddings.
2. Second RAG request reuses all 5 cached embeddings.
3. Embedding API is NOT called again for cached chunks.
4. User-query embedding is still generated for every new query.
5. Thread-safe concurrency behavior.
6. Graceful fallback when embedding generation fails.
7. Existing RAG ranking results match exactly.
"""

import pytest
from unittest.mock import MagicMock, patch, call
from uuid import uuid4
import threading
import math

from src.rag.service import (
    RAGService,
    DEFAULT_VERIFIED_AGRI_KNOWLEDGE,
    _STATIC_BASELINE_EMBEDDINGS,
    get_cached_baseline_embedding,
    clear_static_baseline_cache,
)


@pytest.fixture(autouse=True)
def reset_baseline_cache():
    """Ensure cache is clean before and after each test."""
    clear_static_baseline_cache()
    yield
    clear_static_baseline_cache()


@pytest.mark.asyncio
async def test_first_rag_request_populates_cache_and_second_reuses_it():
    """Verify first request generates static embeddings and second request reuses all 5 without calling generate_embedding for them."""
    rag_service = RAGService(repository=None)
    
    # Track which texts generate_embedding was called for
    call_log = []
    original_generate = rag_service.generate_embedding

    def mock_generate(text, dimension=768):
        call_log.append(text)
        return original_generate(text, dimension)

    rag_service.generate_embedding = mock_generate

    # 1. First search request
    query1 = "cotton alternaria leaf spot treatment"
    results1 = await rag_service.hybrid_search_knowledge(query1, top_k=3, crop="Cotton")
    
    num_chunks = len(DEFAULT_VERIFIED_AGRI_KNOWLEDGE)
    # Should have called generate_embedding for query1 (1) + baseline chunks = 1 + num_chunks calls
    assert len(call_log) == 1 + num_chunks
    assert call_log[0] == query1
    # Check that all chunks are in the cache
    assert len(_STATIC_BASELINE_EMBEDDINGS) == num_chunks
    for idx in range(num_chunks):
        assert idx in _STATIC_BASELINE_EMBEDDINGS
        assert len(_STATIC_BASELINE_EMBEDDINGS[idx]) == 768

    call_log.clear()

    # 2. Second search request with different query
    query2 = "paddy blast disease pesticide"
    results2 = await rag_service.hybrid_search_knowledge(query2, top_k=3, crop="Paddy")

    # Only query2 should be embedded; baseline chunks MUST come from cache (0 new calls for chunks)
    assert len(call_log) == 1
    assert call_log[0] == query2
    assert len(_STATIC_BASELINE_EMBEDDINGS) == num_chunks


@pytest.mark.asyncio
async def test_user_query_embedding_never_cached_as_static():
    """Verify that different user queries always trigger generate_embedding."""
    rag_service = RAGService(repository=None)
    embedded_queries = []

    def mock_generate(text, dimension=768):
        embedded_queries.append(text)
        return [0.1] * dimension

    rag_service.generate_embedding = mock_generate

    # Run query 1
    await rag_service.hybrid_search_knowledge("Query one", top_k=2)
    # Run query 2
    await rag_service.hybrid_search_knowledge("Query two", top_k=2)

    # Both distinct queries must have been embedded
    assert "Query one" in embedded_queries
    assert "Query two" in embedded_queries


def test_cache_fallback_when_embedding_call_fails():
    """Verify get_cached_baseline_embedding safely falls back to local deterministic vector if generation fails."""
    rag_service = RAGService(repository=None)

    def failing_generate(text, dimension=768):
        raise ConnectionError("Google Embedding API unreachable")

    rag_service.generate_embedding = failing_generate

    # Should not raise exception; falls back to deterministic local vector
    vec = get_cached_baseline_embedding(rag_service, 0, DEFAULT_VERIFIED_AGRI_KNOWLEDGE[0]["chunk_text"])
    
    assert isinstance(vec, list)
    assert len(vec) == 768
    # Norm of fallback vector is 1.0 (normalized)
    norm = math.sqrt(sum(x * x for x in vec))
    assert pytest.approx(norm, rel=1e-3) == 1.0
    # Cached successfully
    assert 0 in _STATIC_BASELINE_EMBEDDINGS


def test_concurrent_access_does_not_duplicate_work():
    """Verify thread-safety when multiple threads request the same chunk simultaneously."""
    rag_service = RAGService(repository=None)
    call_count = 0
    lock = threading.Lock()

    original_generate = rag_service._deterministic_local_embedding

    def tracked_generate(text, dimension=768):
        nonlocal call_count
        with lock:
            call_count += 1
        return original_generate(text, dimension)

    rag_service.generate_embedding = tracked_generate

    threads = []
    results = [None] * 10

    def worker(i):
        results[i] = get_cached_baseline_embedding(rag_service, 1, DEFAULT_VERIFIED_AGRI_KNOWLEDGE[1]["chunk_text"])

    for i in range(10):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # generate_embedding must have been called exactly once for chunk index 1
    assert call_count == 1
    # All 10 threads received the exact same vector
    for r in results:
        assert r == results[0]


@pytest.mark.asyncio
async def test_cached_and_uncached_results_are_identical():
    """Verify that search results with cache match uncached search results bit-for-bit."""
    rag_service = RAGService(repository=None)

    # Uncached search (first run)
    clear_static_baseline_cache()
    res1 = await rag_service.hybrid_search_knowledge("cotton alternaria", top_k=3, crop="Cotton")

    # Cached search (second run with exact same query)
    res2 = await rag_service.hybrid_search_knowledge("cotton alternaria", top_k=3, crop="Cotton")

    assert len(res1) == len(res2)
    for r1, r2 in zip(res1, res2):
        assert r1.chunk_id == r2.chunk_id
        assert r1.document_title == r2.document_title
        assert r1.similarity_score == r2.similarity_score
        assert r1.vector_score == r2.vector_score
        assert r1.keyword_score == r2.keyword_score
        assert r1.chunk_text == r2.chunk_text
