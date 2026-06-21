"""
retrieve_bm25.py

Production BM25 Retriever

Architecture

Qdrant
 ↓
Build BM25 Once
 ↓
Cache Per Collection
 ↓
Retrieve

Features
--------
✓ Qdrant Cloud Compatible
✓ Collection Aware
✓ Cached BM25
✓ No Pickles
✓ FastAPI Ready
✓ Latency Profiling
"""

import re
import time

from rank_bm25 import BM25Okapi

from retrieve import (
    get_qdrant_client,
)

# ==================================================
# CACHE
# ==================================================

_bm25_cache = {}


# ==================================================
# TOKENIZER
# ==================================================

def tokenize(
    text: str,
):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    return text.split()


# ==================================================
# LOAD COLLECTION FROM QDRANT
# ==================================================

def load_collection_chunks(
    collection_name: str,
):

    client = (
        get_qdrant_client()
    )

    all_points = []

    next_offset = None

    while True:

        points, next_offset = (
            client.scroll(
                collection_name=
                collection_name,

                limit=1000,

                offset=
                next_offset,

                with_payload=True,

                with_vectors=False,
            )
        )

        all_points.extend(
            points
        )

        if next_offset is None:
            break

    return all_points


# ==================================================
# BUILD INDEX
# ==================================================

def get_bm25_index(
    collection_name: str,
):

    global _bm25_cache

    if (
        collection_name
        in _bm25_cache
    ):

        return _bm25_cache[
            collection_name
        ], True

    print(
        f"Building BM25 index "
        f"for {collection_name}"
    )

    points = (
        load_collection_chunks(
            collection_name
        )
    )

    corpus = []

    payloads = []

    for point in points:

        payload = (
            point.payload
            or {}
        )

        text = payload.get(
            "text"
        )

        if not text:
            continue

        corpus.append(
            tokenize(text)
        )

        payloads.append(
            payload
        )

    bm25 = BM25Okapi(
        corpus
    )

    index_data = {

        "bm25":
        bm25,

        "payloads":
        payloads,
    }

    _bm25_cache[
        collection_name
    ] = index_data

    return index_data, False


# ==================================================
# RETRIEVE
# ==================================================

def bm25_retrieve(
    query: str,
    collection_name: str,
    k: int = 10,
):

    start = (
        time.perf_counter()
    )

    index_data ,cache_hit= (
        get_bm25_index(
            collection_name
        )
    )

    bm25 = (
        index_data["bm25"]
    )

    payloads = (
        index_data["payloads"]
    )

    tokenized_query = (
        tokenize(query)
    )

    scores = (
        bm25.get_scores(
            tokenized_query
        )
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    results = []

    for idx in ranked_indices[:k]:

        results.append({

            "score":
            float(
                scores[idx]
            ),

            "payload":
            payloads[idx],
        })

    total_ms = round(
        (
            time.perf_counter()
            - start
        ) * 1000,
        2,
    )

    return {

        "results":
        results,

        "metrics": {

            "bm25_ms":
            total_ms,

            "collection":
            collection_name,

            "cache_hit":
            cache_hit,
        },
    }


# ==================================================
# CACHE MANAGEMENT
# ==================================================

def clear_bm25_cache():

    global _bm25_cache

    _bm25_cache = {}

    print(
        "BM25 cache cleared."
    )


def invalidate_collection_cache(
    collection_name: str,
):

    global _bm25_cache

    _bm25_cache.pop(
        collection_name,
        None,
    )


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    response = (
        bm25_retrieve(

            query=
            "What is LoRA?",

            collection_name=
            "research_papers",

            k=5,
        )
    )

    print(
        response["metrics"]
    )

    print()

    for result in response[
        "results"
    ]:

        print(
            result["score"]
        )

        print(
            result["payload"]
            .get(
                "paper_name"
            )
        )

        print("-" * 50)