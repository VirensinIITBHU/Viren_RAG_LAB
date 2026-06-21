"""
reranker.py

Production Cross Encoder Reranker

Features
--------
✓ Singleton Loading
✓ GPU / CPU Detection
✓ Torch Inference Mode
✓ Batch Processing
✓ Empty Candidate Safety
✓ FastAPI Ready
✓ Latency Profiling
"""

import time
import torch

from sentence_transformers import (
    CrossEncoder
)

# ==================================================
# CONFIG
# ==================================================

MODEL_NAME = (
    "BAAI/bge-reranker-base"
)

DEFAULT_BATCH_SIZE = 32

# ==================================================
# DEVICE
# ==================================================

if torch.cuda.is_available():

    DEVICE = "cuda"

elif (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
):

    DEVICE = "mps"

else:

    DEVICE = "cpu"

# ==================================================
# SINGLETON
# ==================================================

_reranker = None


def get_reranker():

    global _reranker

    if _reranker is None:

        print(
            f"Loading reranker on {DEVICE}"
        )

        _reranker = CrossEncoder(
            MODEL_NAME,
            device=DEVICE,
        )

        print(
            "Reranker loaded."
        )

    return _reranker


# ==================================================
# RERANK
# ==================================================

def rerank(
    query: str,
    candidates,
    batch_size: int = DEFAULT_BATCH_SIZE,
):

    start = (
        time.perf_counter()
    )

    if not candidates:

        return {

            "results": [],

            "metrics": {

                "rerank_ms": 0.0,

                "candidate_count": 0,

                "device": DEVICE,

                "model": MODEL_NAME,
            },
        }

    reranker = (
        get_reranker()
    )

    print(f"Reranker device: {DEVICE}")
    

    pairs = []

    for (
        chunk_id,
        payload,
    ) in candidates:

        pairs.append(

            (
                query,
                payload.get(
                    "text",
                    "",
                ),
            )
        )

    with torch.inference_mode():

        scores = reranker.predict(

            pairs,

            batch_size=batch_size,

            show_progress_bar=False,
        )

    reranked = []

    for (
        chunk_id,
        payload,
    ), score in zip(
        candidates,
        scores,
    ):

        reranked.append(

            (
                chunk_id,
                payload,
                float(score),
            )
        )

    reranked.sort(
        key=lambda x: x[2],
        reverse=True,
    )

    rerank_ms = round(
        (
            time.perf_counter()
            - start
        ) * 1000,
        2,
    )

    return {

        "results":
        reranked,

        "metrics": {

            "rerank_ms":
            rerank_ms,

            "candidate_count":
            len(candidates),

            "device":
            DEVICE,

            "model":
            MODEL_NAME,
        },
    }


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    sample_candidates = [

        (
            1,
            {
                "text":
                "LoRA is a parameter efficient fine tuning method."
            },
        ),

        (
            2,
            {
                "text":
                "Transformers use self attention."
            },
        ),
    ]

    response = rerank(

        query=
        "What is LoRA?",

        candidates=
        sample_candidates,
    )

    print(
        response["metrics"]
    )

    print()

    for result in response["results"]:

        print(
            result
        )