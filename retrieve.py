"""
retrieve.py

Production Dense Retriever

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

        print(
            f"Loading embedding model: {EMBED_MODEL}"
        )

        _model = FlagAutoModel.from_finetuned(
            EMBED_MODEL,
            query_instruction_for_retrieval=
            "Represent this sentence for searching relevant passages:",
            use_fp16=True,
        )

        print(
            "Embedding model loaded."
        )

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

        print(
            "Connected to Qdrant."
        )

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

    query_embedding = model.encode(
        [query]
    )[0]

    embedding_ms = round(
        (
            time.perf_counter()
            - embed_start
        )
        * 1000,
        2,
    )

    # ------------------------------
    # Search
    # ------------------------------

    search_start = time.perf_counter()

    try:

        response = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=k,
            with_payload=True,
        )

    except Exception as e:

        raise RuntimeError(
            f"Qdrant retrieval failed: {e}"
        )

    vector_search_ms = round(
        (
            time.perf_counter()
            - search_start
        )
        * 1000,
        2,
    )

    return {

        "results":
        response.points,

        "metrics": {

            "embedding_ms":
            embedding_ms,

            "vector_search_ms":
            vector_search_ms,

            "total_ms":
            round(
                embedding_ms
                + vector_search_ms,
                2,
            ),
        },
    }


# ==================================================
# WARMUP
# ==================================================

def warmup():

    print(
        "Running retriever warmup..."
    )

    get_qdrant_client()
    get_embedding_model()

    print(
        "Warmup complete."
    )


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    warmup()

    response = retrieve(
        query="What is LoRA?",
        collection_name=
        "research_papers",
        k=5,
    )

    print(
        "\nMetrics:"
    )

    print(
        response["metrics"]
    )

    for rank, point in enumerate(
        response["results"],
        start=1,
    ):

        payload = (
            point.payload
            or {}
        )

        print(
            f"\nRank {rank}"
        )

        print(
            f"Score: {point.score:.4f}"
        )

        print(
            f"Paper: "
            f"{payload.get('paper_name')}"
        )

        print(
            f"Chunk: "
            f"{payload.get('chunk_id')}"
        )