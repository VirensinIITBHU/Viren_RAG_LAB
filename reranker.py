"""
reranker.py

Production Cross Encoder Reranker (Child-Chunk Aware)

Features
--------
✓ Thread-Safe Singleton Loading
✓ GPU / CPU Auto-Detection
✓ Torch Inference Mode
✓ Graceful Degradation on Malformed Text
✓ Schema-Consistent Chunk Mapping
"""

import time
import threading
import torch

from sentence_transformers import CrossEncoder

# ==================================================
# CONFIG
# ==================================================

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_BATCH_SIZE = 32

# ==================================================
# DEVICE CONFIGURATION
# ==================================================

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# ==================================================
# SINGLETON RERANKER (THREAD-SAFE)
# ==================================================

_reranker = None
_reranker_lock = threading.Lock()

def get_reranker():
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                print(f"Loading reranker on {DEVICE}...")
                _reranker = CrossEncoder(
                    MODEL_NAME,
                    device=DEVICE,
                )
                print("Reranker loaded.")
    return _reranker

# ==================================================
# RERANK
# ==================================================

def rerank(
    query: str,
    candidates: list,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    start = time.perf_counter()

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

    reranker = get_reranker()
    
    pairs = []
    
    # --------------------------------------------------
    # 1. Prepare Text Pairs
    # --------------------------------------------------
    # The hybrid pipeline passes (chunk_id, payload) tuples.
    # We extract the precise child text for accurate Cross-Encoder scoring.
    for chunk_id, payload in candidates:
        text = payload.get("text", "").strip()
        pairs.append((query, text))

    # --------------------------------------------------
    # 2. Predict Scores
    # --------------------------------------------------
    try:
        with torch.inference_mode():
            scores = reranker.predict(
                pairs,
                batch_size=batch_size,
                show_progress_bar=False,
            )
    except Exception as e:
        print(f"WARNING: Reranker inference failed: {e}")
        # Graceful degradation: assign 0 score if tensor operation fails
        scores = [0.0] * len(pairs)

    # --------------------------------------------------
    # 3. Reconstruct & Sort
    # --------------------------------------------------
    reranked = []
    
    for (chunk_id, payload), score in zip(candidates, scores):
        reranked.append(
            (
                chunk_id,
                payload,
                float(score),
            )
        )

    # Sort descending by Cross-Encoder score
    reranked.sort(
        key=lambda x: x[2],
        reverse=True,
    )

    rerank_ms = round((time.perf_counter() - start) * 1000, 2)

    return {
        "results": reranked,
        "metrics": {
            "rerank_ms": rerank_ms,
            "candidate_count": len(candidates),
            "device": DEVICE,
            "model": MODEL_NAME,
        },
    }

# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":
    # Mocking the (chunk_id, payload) schema from the hybrid pipeline
    sample_candidates = [
        (
            1001, 
            {
                "paper_name": "Attention Is All You Need",
                "text": "LoRA is a parameter efficient fine tuning method."
            },
        ),
        (
            1002,
            {
                "paper_name": "Attention Is All You Need",
                "text": "Transformers use self attention mechanisms."
            },
        ),
    ]

    response = rerank(
        query="What is LoRA?",
        candidates=sample_candidates,
    )

    print("\nMetrics:")
    print(response["metrics"])
    print()

    for result in response["results"]:
        print(f"Chunk ID: {result[0]} | Score: {result[2]:.4f} | Text: {result[1]['text']}")