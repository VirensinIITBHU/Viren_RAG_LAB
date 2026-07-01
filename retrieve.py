"""
retrieve.py

Production Dense Retriever (Native Qdrant Hybrid Ready)

Features
--------
✓ Thread-Safe Singleton Model & Client
✓ Native Qdrant Dense Vector Target (using="")
✓ Full Child-Chunk Payload Retrieval
✓ Production Safe Error Handling
✓ Detailed Metrics
"""

import os
import time
import threading
from typing import Any, Dict

import torch
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from FlagEmbedding import FlagAutoModel

load_dotenv()

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

_model = None
_model_lock = threading.Lock()

_client = None
_client_lock = threading.Lock()

# ==================================================
# DEVICE DETECTION
# ==================================================

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# ==================================================
# EMBEDDING MODEL (THREAD-SAFE)
# ==================================================

def get_embedding_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:  # Double-checked locking
                print(f"Loading dense embedding model: {EMBED_MODEL} on {DEVICE}")
                _model = FlagAutoModel.from_finetuned(
                    EMBED_MODEL,
                    query_instruction_for_retrieval=
                    "Represent this sentence for searching relevant passages:",
                    use_fp16=False,  # Matched to embed.py for stability
                )
                print("Embedding model loaded.")
    return _model


# ==================================================
# QDRANT CLIENT (THREAD-SAFE)
# ==================================================

def get_qdrant_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                qdrant_url = os.getenv("QDRANT_URL")
                qdrant_api_key = os.getenv("QDRANT_API_KEY")
                
                if not qdrant_url or not qdrant_api_key:
                    print("WARNING: QDRANT_URL or QDRANT_API_KEY not set in environment.")

                _client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key,
                    timeout=60,
                )
                print("Connected to Qdrant.")
    return _client


# ==================================================
# RETRIEVE
# ==================================================

def retrieve(
    query: str,
    collection_name: str,
    k: int = 10,
) -> Dict[str, Any]:

    client = get_qdrant_client()
    model = get_embedding_model()

    # ------------------------------
    # Embed Query
    # ------------------------------
    embed_start = time.perf_counter()
    query_embedding = model.encode([query])[0]
    embedding_ms = round((time.perf_counter() - embed_start) * 1000, 2)

    # ------------------------------
    # Search Qdrant
    # ------------------------------
    search_start = time.perf_counter()

    try:
        response = client.query_points(
            collection_name=collection_name,
            query=query_embedding.tolist(),
            using="",  # Explicitly target the unnamed dense vector
            limit=k,
            with_payload=True,  # Fetch full payload including text for reranking
        )
    except Exception as e:
        raise RuntimeError(f"Qdrant dense retrieval failed: {e}")

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

    return {
        "results": results,
        "metrics": {
            "embedding_ms": embedding_ms,
            "vector_search_ms": vector_search_ms,
            "total_ms": round(embedding_ms + vector_search_ms, 2),
        },
    }


# ==================================================
# WARMUP
# ==================================================

def warmup():
    print("Running dense retriever warmup...")
    get_qdrant_client()
    get_embedding_model()
    print("Warmup complete.")


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":
    warmup()
    
    response = retrieve(
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