"""
retrieve_reranked_hybrid.py

Production Hybrid Retriever

Pipeline
--------
Query
    ↓
Dense Retrieval
    +
BM25 Retrieval
    ↓
RRF Fusion
    ↓
CrossEncoder Reranking
    ↓
Top-K Results

Features
--------
✓ Configurable Dense K
✓ Configurable BM25 K
✓ Configurable Rerank Candidates
✓ Configurable Final K
✓ RRF Fusion
✓ CrossEncoder Reranking
✓ Frontend Ready
✓ FastAPI Ready
"""

from collections import defaultdict
import time

from retrieve import retrieve
from retrieve_bm25 import bm25_retrieve
from reranker import rerank
from concurrent.futures import ThreadPoolExecutor


RRF_K = 60


def reciprocal_rank_fusion(
    bm25_results,
    dense_results,
    rrf_k: int = RRF_K,
):
    """
    RRF Fusion

    Score = Σ 1 / (rrf_k + rank)
    """

    fused_scores = defaultdict(float)

    # BM25 Results

    for rank, result in enumerate(
        bm25_results,
        start=1,
    ):
        payload = result.get(
            "payload",
            {},
        )

        chunk_id = payload.get(
            "chunk_id"
        )

        if chunk_id is None:
            continue

        fused_scores[
            chunk_id
        ] += 1 / (
            rrf_k + rank
        )

    # Dense Results

    for rank, point in enumerate(
        dense_results,
        start=1,
    ):
        payload = (
            point.payload
            or {}
        )

        chunk_id = payload.get(
            "chunk_id"
        )

        if chunk_id is None:
            continue

        fused_scores[
            chunk_id
        ] += 1 / (
            rrf_k + rank
        )

    return sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )


def build_chunk_map(
    bm25_results,
    dense_results,
):
    """
    Creates chunk_id -> payload lookup
    """

    chunk_map = {}

    # BM25

    for result in bm25_results:

        payload = result.get(
            "payload",
            {},
        )

        chunk_id = payload.get(
            "chunk_id"
        )

        if chunk_id is not None:

            chunk_map[
                chunk_id
            ] = payload

    # Dense

    for point in dense_results:

        payload = (
            point.payload
            or {}
        )

        chunk_id = payload.get(
            "chunk_id"
        )

        if chunk_id is not None:

            chunk_map[
                chunk_id
            ] = payload

    return chunk_map


def retrieve_hybrid(
    query: str,
    collection_name: str,
    dense_k: int = 20,
    bm25_k: int = 20,
    rerank_candidates: int = 20,
    final_k: int = 5,
):
    """
    Hybrid Retrieval

    Dense
    +
    BM25
    ↓
    RRF
    ↓
    CrossEncoder
    """

    total_start = (
        time.perf_counter()
    )

    # ==========================================
    # BM25
    # ==========================================

    # ==========================================
    # BM25 + DENSE (PARALLEL)
    # ==========================================

    parallel_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=2) as executor:

        bm25_future = executor.submit(
            bm25_retrieve,
            query=query,
            collection_name=collection_name,
            k=bm25_k,
        )

        dense_future = executor.submit(
            retrieve,
            query=query,
            collection_name=collection_name,
            k=dense_k,
        )

        bm25_response = bm25_future.result()
        dense_response = dense_future.result()

    parallel_ms = round(
        (time.perf_counter() - parallel_start) * 1000,
        2,
    )
    print(f"BM25 + Dense retrieval completed in {parallel_ms} ms.") ##DEBUGGG

    bm25_results = bm25_response["results"]
    dense_results = dense_response["results"]

    # Keep the original timings from each retriever

    bm25_ms = bm25_response["metrics"]["bm25_ms"]

    dense_ms = dense_response["metrics"]["total_ms"]

    # ==========================================
    # RRF
    # ==========================================

    fusion_start = (
        time.perf_counter()
    )

    fused_results = (
        reciprocal_rank_fusion(
            bm25_results,
            dense_results,
        )
    )

    fusion_ms = round(
        (
            time.perf_counter()
            - fusion_start
        )
        * 1000,
        2,
    )

    # ==========================================
    # BUILD LOOKUP
    # ==========================================

    chunk_map = (
        build_chunk_map(
            bm25_results,
            dense_results,
        )
    )

    # ==========================================
    # RERANK INPUTS
    # ==========================================

    rerank_inputs = []

    for chunk_id, _ in fused_results[
        :rerank_candidates
    ]:

        payload = chunk_map.get(
            chunk_id
        )

        if payload is None:
            continue

        rerank_inputs.append(
            (
                chunk_id,
                payload,
            )
        )

    # ==========================================
    # RERANK
    # ==========================================

    rerank_response = (
        rerank(
            query=query,
            candidates=rerank_inputs,
        )
    )

    reranked = (
        rerank_response["results"]
    )

    rerank_metrics = (
        rerank_response["metrics"]
    )

    # ==========================================
    # FINAL TOP K
    # ==========================================

    final_results = (
        reranked[:final_k]
    )

    total_ms = round(
        (
            time.perf_counter()
            - total_start
        )
        * 1000,
        2,
    )

    return {

        "results":
        final_results,

        "metrics": {

            "dense_k":
            dense_k,

            "bm25_k":
            bm25_k,

            "rerank_candidates":
            rerank_candidates,

            "final_k":
            final_k,

            "bm25_ms":
            bm25_ms,

            "dense_ms":
            dense_ms,

            "fusion_ms":
            fusion_ms,

            "rerank_ms":
            rerank_metrics[
                "rerank_ms"
            ],

            "candidate_count":
            rerank_metrics[
                "candidate_count"
            ],

            "device":
            rerank_metrics[
                "device"
            ],

            "model":
            rerank_metrics[
                "model"
            ],

            "total_ms":
            total_ms,
            "parallel_ms": parallel_ms,
        },
    }



def format_sources(
    results,
):
    """
    Converts reranked output
    into frontend-friendly format
    """

    sources = []

    for (
        chunk_id,
        payload,
        score,
    ) in results:

        sources.append({

            "chunk_id":
            chunk_id,

            "paper_name":
            payload.get(
                "paper_name"
            ),

            "page":
            payload.get(
                "page"
            ),

            "source":
            payload.get(
                "source"
            ),

            "score":
            round(
                score,
                4,
            ),
        })

    return sources


if __name__ == "__main__":

    response = retrieve_hybrid(

        query=
        "What is LoRA?",

        collection_name=
        "research_papers",

        dense_k=20,

        bm25_k=20,

        rerank_candidates=20,

        final_k=5,
    )

    print(
        response["metrics"]
    )

    print()

    for source in format_sources(
        response["results"]
    ):
        print(source)