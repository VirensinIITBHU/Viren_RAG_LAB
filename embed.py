"""
embed.py

Production Streaming Embedding Service

Architecture

PDF
 ↓
Chunk Stream
 ↓
Embedding Batch (Main Thread) -> SQLite (Parents)
 ↓
Queue
 ↓
Qdrant Upload (Background Thread)
"""

import os
import time
import uuid
import queue
import threading

from dotenv import load_dotenv

from FlagEmbedding import FlagAutoModel
from fastembed import SparseTextEmbedding

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    HnswConfigDiff,
    SparseVectorParams,
    SparseVector,
)
from tqdm import tqdm

from ingest import chunk_batches
from database import SessionLocal, save_parent_chunks_batch

load_dotenv()

# ==================================================
# CONFIG
# ==================================================

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_BATCH_SIZE = 128

# ==================================================
# SINGLETON MODELS (THREAD-SAFE)
# ==================================================

_model = None
_model_lock = threading.Lock()

_sparse_model = None
_sparse_lock = threading.Lock()

def get_embedding_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                print("Loading dense embedding model...")
                _model = FlagAutoModel.from_finetuned(
                    EMBED_MODEL,
                    query_instruction_for_retrieval=
                    (
                        "Represent this sentence "
                        "for searching relevant passages:"
                    ),
                    use_fp16=False,  
                )
                print("Dense embedding model loaded.")
    return _model

def get_sparse_model():
    """Loads the FastEmbed Sparse model for Qdrant BM25 equivalent."""
    global _sparse_model
    if _sparse_model is None:
        with _sparse_lock:
            if _sparse_model is None:
                print("Loading sparse BM25 model...")
                _sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
                print("Sparse model loaded.")
    return _sparse_model


# ==================================================
# SINGLETON CLIENT (THREAD-SAFE)
# ==================================================

_client = None
_client_lock = threading.Lock()

def get_qdrant_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = QdrantClient(
                    url=os.getenv("QDRANT_URL"),
                    api_key=os.getenv("QDRANT_API_KEY"),
                    timeout=60
                )
                print("Connected to Qdrant.")
    return _client


# ==================================================
# VECTOR SIZE
# ==================================================

_vector_size = None

def get_vector_size():
    global _vector_size
    if _vector_size is None:
        model = get_embedding_model()
        vector = model.encode(["hello world"])[0]
        _vector_size = len(vector)
    return _vector_size


# ==================================================
# COLLECTION
# ==================================================

def collection_exists(collection_name: str):
    client = get_qdrant_client()
    collections = [
        c.name
        for c in client.get_collections().collections
    ]
    return collection_name in collections

def create_collection(collection_name: str):
    client = get_qdrant_client()
    if collection_exists(collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=get_vector_size(),
            distance=Distance.COSINE,
        ),
        sparse_vectors_config={
            "bm25": SparseVectorParams()
        },
        hnsw_config=HnswConfigDiff(
            m=16,
            ef_construct=100,
            full_scan_threshold=1000,
        ),
    )
    print(f"Created collection {collection_name}")


# ==================================================
# BACKGROUND UPLOAD WORKER
# ==================================================

def qdrant_upload_worker(upload_queue, collection_name):
    """Pulls encoded vectors from the queue and uploads them to avoid blocking the CPU."""
    client = get_qdrant_client()
    while True:
        item = upload_queue.get()
        if item is None:  # Sentinel value to terminate thread
            upload_queue.task_done()
            break
            
        points = item
        # upsert naturally overwrites existing deterministic IDs
        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=False  # Fire and forget to maximize throughput
        )
        upload_queue.task_done()


# ==================================================
# STREAM EMBEDDING
# ==================================================

def embed_collection(
    collection_name: str,
    pdf_directory: str,
):
    start = time.perf_counter()
    create_collection(collection_name)

    model = get_embedding_model()
    sparse_model = get_sparse_model()
    client = get_qdrant_client()
    db = SessionLocal()

    total_vectors = 0
    upload_queue = queue.Queue(maxsize=10) # Bounded queue to prevent RAM spikes
    
    uploader = threading.Thread(
        target=qdrant_upload_worker, 
        args=(upload_queue, collection_name),
        daemon=True
    )
    uploader.start()

    # True Generator Stream - No list() wrapper
    batch_generator = chunk_batches(
        pdf_directory=pdf_directory,
        batch_size=EMBED_BATCH_SIZE,
    )

    progress = tqdm(desc="Embedding Stream", unit="batch")

    try:
        for batch in batch_generator:
            
            # Route Chunks: Parents -> SQLite | Children -> Embedding Model
            parents = [doc for doc in batch if doc.metadata.get("chunk_type") == "parent"]
            children = [doc for doc in batch if doc.metadata.get("chunk_type") == "child"]

            # 1. Save Parents to SQLite
            if parents:
                parent_records = [{
                    "collection_name": collection_name,
                    "parent_id": p.metadata.get("parent_id"),
                    "text": p.page_content,
                    "paper_name": p.metadata.get("paper_name")
                } for p in parents]
                # Utilizing the idempotent upsert we wrote in database.py
                save_parent_chunks_batch(db, parent_records)

            if not children:
                progress.update(1)
                continue

            # 2. Embed Children (Dense & Sparse)
            texts = [doc.page_content for doc in children]
            embed_start = time.perf_counter()
            
            # Generate both dense and sparse vectors
            embeddings = model.encode(texts)
            # list() forces the fastembed generator to evaluate immediately
            sparse_embeddings = list(sparse_model.embed(texts))
            
            embed_ms = (time.perf_counter() - embed_start) * 1000

            points = []
            for i, doc in enumerate(children):
                
                chunk_id = doc.metadata.get("chunk_id")
                # Deterministic ID based on collection + chunk_id
                # This guarantees that re-running embed.py overwrites old vectors instead of duplicating
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{collection_name}_chunk_{chunk_id}"))

                points.append(
                    PointStruct(
                        id=point_id,
                        # Send both vectors as a dictionary
                        vector={
                            "": embeddings[i].tolist(),
                            "bm25": SparseVector(
                                indices=sparse_embeddings[i].indices.tolist(),
                                values=sparse_embeddings[i].values.tolist()
                            )
                        },
                        payload={
                            "text": doc.page_content,
                            "paper_name": doc.metadata.get("paper_name"),
                            "source": doc.metadata.get("source"),
                            "page": doc.metadata.get("page"),
                            "chunk_id": chunk_id,
                            "parent_id": doc.metadata.get("parent_id"),
                            "chunk_type": doc.metadata.get("chunk_type"),
                        },
                    )
                )

            # 3. Push to Background Network Queue
            upload_queue.put(points)
            
            total_vectors += len(points)
            progress.update(1)
            
            progress.set_postfix({
                "Vectors": total_vectors,
                "Embed(ms)": f"{embed_ms:.0f}",
                "Vec/sec": f"{len(points)/(embed_ms/1000):.1f}",
                "Q(size)": upload_queue.qsize()
            })
            
    finally:
        # Graceful shutdown of queues and database session
        upload_queue.put(None)
        upload_queue.join()
        uploader.join()
        progress.close()
        db.close()

    count = client.count(collection_name=collection_name)
    total_ms = round((time.perf_counter() - start) * 1000, 2)
    throughput = round(total_vectors / (total_ms / 1000), 2) if total_ms > 0 else 0

    return {
        "collection": collection_name,
        "vector_count": count.count,
        "embedded": total_vectors,
        "embedding_ms": total_ms,
        "throughput": throughput,
        "batch_size": EMBED_BATCH_SIZE,
        "embedding_model": EMBED_MODEL,
    }

# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":
    result = embed_collection(
        collection_name="research_papers",
        pdf_directory="user_uploads",
    )

    print("\n========== SUMMARY ==========")
    print(f"Collection : {result['collection']}")
    print(f"Vectors    : {result['vector_count']}")
    print(f"Embedded   : {result['embedded']}")
    print(f"Time       : {result['embedding_ms'] / 1000:.2f} sec")
    print(f"Throughput : {result['throughput']:.1f} vec/sec")