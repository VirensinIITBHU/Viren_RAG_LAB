"""
retrieve_reranked_hybrid.py

Production Hybrid Retriever with Parent Hydration

Pipeline
--------
Query
    ↓
Parallel Native Qdrant Retrieval (Dense "" + Sparse "bm25")
    ↓
Reciprocal Rank Fusion (RRF) on chunk_id
    ↓
CrossEncoder Reranking of top N child chunks
    ↓
SQLite Hydration (Expand winning child chunks to Parent Text)
    ↓
Top-K Context Results
"""

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from retrieve import retrieve
from retrieve_bm25 import bm25_retrieve
from reranker import rerank
from database import SessionLocal, get_parent_chunks_by_ids

# ==================================================
# CONFIG
# ==================================================

RRF_K = 60

# ==================================================
# RRF FUSION
# ==================================================

def reciprocal_rank_fusion(bm25_results, dense_results, rrf_k: int = RRF_K):
    """
    Fuses two ranked lists using Reciprocal Rank Fusion.
    Returns normalized scores [0, 1] for display as confidence.
    """
    fused_scores = defaultdict(float)

    # Score BM25 Results
    for rank, result in enumerate(bm25_results, start=1):
        chunk_id = result.get("chunk_id")
        if chunk_id is not None:
            fused_scores[chunk_id] += 1.0 / (rrf_k + rank)

    # Score Dense Results
    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result.get("chunk_id")
        if chunk_id is not None:
            fused_scores[chunk_id] += 1.0 / (rrf_k + rank)

    if not fused_scores:
        return []

    # Sort to find the max score
    sorted_scores = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Normalize by the highest score so the top result is always 1.0 (100%)
    max_score = sorted_scores[0][1]
    
    normalized_results = [
        (chunk_id, score / max_score) 
        for chunk_id, score in sorted_scores
    ]

    return normalized_results


def build_chunk_map(bm25_results, dense_results):
    """Creates a deduplicated chunk_id -> payload lookup dictionary."""
    chunk_map = {}

    for result in bm25_results + dense_results:
        chunk_id = result.get("chunk_id")
        payload = result.get("payload")
        if chunk_id is not None and payload:
            chunk_map[chunk_id] = payload

    return chunk_map


# ==================================================
# MAIN PIPELINE
# ==================================================

def retrieve_hybrid(
    query: str,
    collection_name: str,
    dense_k: int = 20,
    bm25_k: int = 20,
    rerank_candidates: int = 20,
    final_k: int = 5,
):
    total_start = time.perf_counter()

    # ==========================================
    # 1. PARALLEL RETRIEVAL
    # ==========================================
    parallel_start = time.perf_counter()
    
    bm25_response = {"results": [], "metrics": {"bm25_ms": 0}}
    dense_response = {"results": [], "metrics": {"total_ms": 0}}

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_type = {
            executor.submit(bm25_retrieve, query, collection_name, bm25_k): "bm25",
            executor.submit(retrieve, query, collection_name, dense_k): "dense",
        }

        for future in as_completed(future_to_type):
            req_type = future_to_type[future]
            try:
                if req_type == "bm25":
                    bm25_response = future.result()
                else:
                    dense_response = future.result()
            except Exception as exc:
                print(f"WARNING: {req_type} retrieval generated an exception: {exc}")

    parallel_ms = round((time.perf_counter() - parallel_start) * 1000, 2)

    bm25_results = bm25_response.get("results", [])
    dense_results = dense_response.get("results", [])

    # ==========================================
    # 2. RRF FUSION
    # ==========================================
    fusion_start = time.perf_counter()
    
    fused_results = reciprocal_rank_fusion(bm25_results, dense_results)
    chunk_map = build_chunk_map(bm25_results, dense_results)
    
    fusion_ms = round((time.perf_counter() - fusion_start) * 1000, 2)

    # ==========================================
    # 3. CROSS-ENCODER RERANKING
    # ==========================================
    rerank_inputs = []
    
    for chunk_id, _ in fused_results[:rerank_candidates]:
        payload = chunk_map.get(chunk_id)
        if payload:
            rerank_inputs.append((chunk_id, payload))

    rerank_response = rerank(query=query, candidates=rerank_inputs)
    reranked = rerank_response["results"]
    rerank_metrics = rerank_response["metrics"]

    # ==========================================
    # 4. PARENT-CHILD HYDRATION
    # ==========================================
    hydration_start = time.perf_counter()
    final_results = reranked[:final_k]
    
    # Extract unique parent IDs from the winning child chunks
    parent_ids_to_fetch = [
        payload.get("parent_id") 
        for _, payload, _ in final_results 
        if payload.get("parent_id") is not None
    ]

    parent_map = {}
    if parent_ids_to_fetch:
        db = SessionLocal()
        try:
            parent_map = get_parent_chunks_by_ids(db, collection_name, parent_ids_to_fetch)
        except Exception as e:
            print(f"WARNING: Database hydration failed: {e}")
        finally:
            db.close()

    hydration_ms = round((time.perf_counter() - hydration_start) * 1000, 2)

    # Hydrate the final payloads with parent text
    hydrated_results = []
    for chunk_id, payload, score in final_results:
        parent_id = payload.get("parent_id")
        
        # If we successfully found the parent context, override the child text
        if parent_id is not None and parent_id in parent_map:
            payload["text"] = parent_map[parent_id]["text"]
            # Mark it as expanded for observability/debugging
            payload["context_expanded"] = True
        else:
            payload["context_expanded"] = False
            
        hydrated_results.append((chunk_id, payload, score))

    total_ms = round((time.perf_counter() - total_start) * 1000, 2)

    return {
        "results": hydrated_results,
        "metrics": {
            "dense_k": dense_k,
            "bm25_k": bm25_k,
            "rerank_candidates": rerank_candidates,
            "final_k": final_k,
            "bm25_ms": bm25_response["metrics"].get("bm25_ms", 0),
            "dense_ms": dense_response["metrics"].get("total_ms", 0),
            "parallel_ms": parallel_ms,
            "fusion_ms": fusion_ms,
            "rerank_ms": rerank_metrics["rerank_ms"],
            "hydration_ms": hydration_ms,
            "total_ms": total_ms,
            "candidate_count": rerank_metrics["candidate_count"],
        },
    }

# ==================================================
# FRONTEND FORMATTING
# ==================================================

def format_sources(results):
    """Converts hydrated reranked output into frontend-friendly format."""
    sources = []

    for chunk_id, payload, score in results:
        sources.append({
            "chunk_id": chunk_id,
            "parent_id": payload.get("parent_id"),
            "paper_name": payload.get("paper_name"),
            "page": payload.get("page"),
            "source": payload.get("source"),
            "content": payload.get("text", ""),  # Send the text to the UI!
            "expanded": payload.get("context_expanded", False),
            "score": round(score, 4),
        })

    return sources

if __name__ == "__main__":
    response = retrieve_hybrid(
        query="What is LoRA?",
        collection_name="research_papers",
        dense_k=20,
        bm25_k=20,
        rerank_candidates=20,
        final_k=5,
    )

    print("\nMetrics:")
    print(response["metrics"])
    print()

    print("Hydrated Sources:")
    for source in format_sources(response["results"]):
        print(f"Parent {source['parent_id']} | Expanded: {source['expanded']} | Score: {source['score']} | {source['paper_name']}")