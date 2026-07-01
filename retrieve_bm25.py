"""
retrieve_bm25.py

Production BM25 Retriever (Native Qdrant Sparse Search)

Architecture
------------
Query
    ↓
FastEmbed Sparse Text Embedding (Qdrant/bm25)
    ↓
Qdrant Named Vector Search (using="bm25")

Features
--------
✓ Stateless (No Memory Bloat)
✓ Real-time (No Index Rebuilds)
✓ Qdrant Cloud Compatible
✓ Thread-Safe Singleton
"""

import time
import threading
from typing import Any, Dict

from qdrant_client import models
from fastembed import SparseTextEmbedding

# Reuse the client singleton from the dense retriever
from retrieve import get_qdrant_client

# ==================================================
# SINGLETON MODEL
# ==================================================

_sparse_model = None
_model_lock = threading.Lock()

def get_sparse_model():
    global _sparse_model
    if _sparse_model is None:
        with _model_lock:
            if _sparse_model is None:  # Double-checked locking
                print("Loading sparse BM25 model...")
                _sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
                print("Sparse model loaded.")
    return _sparse_model


# ==================================================
# RETRIEVE
# ==================================================

def bm25_retrieve(
    query: str,
    collection_name: str,
    k: int = 10,
) -> Dict[str, Any]:

    start = time.perf_counter()

    client = get_qdrant_client()
    sparse_model = get_sparse_model()

    # ------------------------------
    # Embed Query (Sparse)
    # ------------------------------
    embed_start = time.perf_counter()
    
    # fastembed's query_embed returns an iterable of SparseEmbedding objects
    query_sparse_embedding = list(sparse_model.query_embed(query))[0]
    embedding_ms = round((time.perf_counter() - embed_start) * 1000, 2)

    # ------------------------------
    # Qdrant Search
    # ------------------------------
    search_start = time.perf_counter()

    # Convert to Qdrant's expected SparseVector format
    query_vector = models.SparseVector(
        indices=query_sparse_embedding.indices.tolist(),
        values=query_sparse_embedding.values.tolist(),
    )

    try:
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using="bm25",       # Explicitly target the named sparse vector
            limit=k,
            with_payload=True,  # Fetch full payload including text
        )
    except Exception as e:
        raise RuntimeError(f"Qdrant sparse retrieval failed: {e}")

    results = []
    for point in response.points:
        payload = point.payload or {}
        results.append({
            "score": float(point.score),
            "payload": payload,
            # Explicitly expose keys to match the fusion pipeline expectations
            "chunk_id": payload.get("chunk_id"),
            "parent_id": payload.get("parent_id"),
            "paper_name": payload.get("paper_name"),
        })

    vector_search_ms = round((time.perf_counter() - search_start) * 1000, 2)
    total_ms = round((time.perf_counter() - start) * 1000, 2)

    return {
        "results": results,
        "metrics": {
            "embedding_ms": embedding_ms,
            "vector_search_ms": vector_search_ms,
            "bm25_ms": total_ms,
            "collection": collection_name,
            "cache_hit": False,  # Kept for frontend metrics compatibility
        },
    }


# ==================================================
# LEGACY CACHE MANAGEMENT (DEPRECATED)
# ==================================================
# These functions are kept as safe no-ops so that app.py and upload.py 
# do not throw import errors or crash before they are updated.

def get_bm25_index(collection_name: str):
    """Legacy warmup hook. Now just warms the FastEmbed model."""
    get_sparse_model()
    return {}, False

def clear_bm25_cache():
    pass

def invalidate_collection_cache(collection_name: str):
    """Legacy invalidation hook. Native Qdrant handles state automatically."""
    pass


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":
    response = bm25_retrieve(
        query="What is LoRA?",
        collection_name="research_papers",
        k=5,
    )

    print("\nMetrics:")
    print(response["metrics"])

    for rank, result in enumerate(response["results"], start=1):
        print(f"\nRank {rank}")
        print(f"Score: {result['score']:.4f}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Paper: {result['paper_name']}")