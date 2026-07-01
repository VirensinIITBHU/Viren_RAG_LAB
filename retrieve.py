"""
retrieve.py

Production Dense Retriever (Late Mapping)

Features
--------
✓ Singleton Embedding Model
✓ Singleton Qdrant Client
✓ Qdrant Cloud Ready
✓ FastAPI Ready
✓ Production Safe
✓ Detailed Metrics
"""

import os
import time
from typing import Any, Dict

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from FlagEmbedding import FlagAutoModel

load_dotenv()

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

_model = None
_client = None


# ==================================================
# EMBEDDING MODEL
# ==================================================

def get_embedding_model():
    global _model
    if _model is None:
        print(f"Loading dense embedding model: {EMBED_MODEL}")
        _model = FlagAutoModel.from_finetuned(
            EMBED_MODEL,
            query_instruction_for_retrieval=
            "Represent this sentence for searching relevant passages:",
            use_fp16=False, # Matched to embed.py for stability
        )
        print("Embedding model loaded.")
    return _model


# ==================================================
# QDRANT CLIENT
# ==================================================

def get_qdrant_client():
    global _client
    if _client is None:
        _client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=30,
        )
        print("Connected to Qdrant.")
    return _client


# ==================================================
# RETRIEVE (LATE MAPPING)
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
            limit=k,
            # We ONLY ask for the parent_id and metadata, not the text
            with_payload=["parent_id", "paper_name", "chunk_id"],
        )
    except Exception as e:
        raise RuntimeError(f"Qdrant retrieval failed: {e}")

    results = []
    for point in response.points:
        results.append({
            "score": float(point.score),
            "parent_id": point.payload.get("parent_id"),
            "paper_name": point.payload.get("paper_name"),
            "child_chunk_id": point.payload.get("chunk_id")
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
    print("Running retriever warmup...")
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
        print(f"Parent ID: {result['parent_id']}")
        print(f"Paper: {result['paper_name']}")