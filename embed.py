"""
embed.py

Production Streaming Embedding Service

Architecture

PDF
 ↓
Chunk Stream
 ↓
Embedding Batch
 ↓
Qdrant Batch

Features
--------
✓ Constant memory
✓ Streaming ingestion
✓ Multi-user collections
✓ Singleton model
✓ Singleton Qdrant client
✓ Batch embedding
✓ Batch upserts
✓ Large corpus ready
"""

import os
import time
import uuid

from dotenv import load_dotenv

from FlagEmbedding import (
    FlagAutoModel,
)

from qdrant_client import (
    QdrantClient,
)

from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from ingest import (
    chunk_batches,
)

load_dotenv()

# ==================================================
# CONFIG
# ==================================================

EMBED_MODEL = (
    "BAAI/bge-small-en-v1.5"
)

EMBED_BATCH_SIZE = 128

# ==================================================
# SINGLETON MODEL
# ==================================================

_model = None


def get_embedding_model():

    global _model

    if _model is None:

        print(
            "Loading embedding model..."
        )

        _model = (
            FlagAutoModel
            .from_finetuned(
                EMBED_MODEL,
                query_instruction_for_retrieval=
                (
                    "Represent this sentence "
                    "for searching relevant passages:"
                ),
                use_fp16=True,
            )
        )

        print(
            "Embedding model loaded."
        )

    return _model


# ==================================================
# SINGLETON CLIENT
# ==================================================

_client = None


def get_qdrant_client():

    global _client

    if _client is None:

        _client = QdrantClient(
            url=os.getenv(
                "QDRANT_URL"
            ),
            api_key=os.getenv(
                "QDRANT_API_KEY"
            ),
            timeout= 60
        )

        print(
            "Connected to Qdrant."
        )

    return _client


# ==================================================
# VECTOR SIZE
# ==================================================

_vector_size = None


def get_vector_size():

    global _vector_size

    if _vector_size is None:

        model = (
            get_embedding_model()
        )

        vector = (
            model.encode(
                ["hello world"]
            )[0]
        )

        _vector_size = len(
            vector
        )

    return _vector_size


# ==================================================
# COLLECTION
# ==================================================

def collection_exists(
    collection_name: str
):

    client = (
        get_qdrant_client()
    )

    collections = [

        c.name

        for c in client
        .get_collections()
        .collections
    ]

    return (
        collection_name
        in collections
    )


def create_collection(
    collection_name: str
):

    client = (
        get_qdrant_client()
    )

    if collection_exists(
        collection_name
    ):

        return

    client.create_collection(

        collection_name=
        collection_name,

        vectors_config=
        VectorParams(

            size=
            get_vector_size(),

            distance=
            Distance.COSINE,
        ),
    )

    print(
        f"Created collection "
        f"{collection_name}"
    )


# ==================================================
# STREAM EMBEDDING
# ==================================================

def embed_collection(
    collection_name: str,
    pdf_directory: str,
):

    start = (
        time.perf_counter()
    )

    create_collection(
        collection_name
    )

    model = (
        get_embedding_model()
    )

    client = (
        get_qdrant_client()
    )

    total_vectors = 0

    for batch in chunk_batches(

        pdf_directory=
        pdf_directory,

        batch_size=
        EMBED_BATCH_SIZE,
    ):

        texts = [

            doc.page_content

            for doc in batch
        ]

        embeddings = (
            model.encode(
                texts
            )
        )

        points = []

        for doc, vector in zip(
            batch,
            embeddings,
        ):

            points.append(

                PointStruct(

                    id=str(
                        uuid.uuid4()
                    ),

                    vector=
                    vector.tolist(),

                    payload={

                        "text":
                        doc.page_content,

                        "paper_name":
                        doc.metadata.get(
                            "paper_name"
                        ),

                        "source":
                        doc.metadata.get(
                            "source"
                        ),

                        "page":
                        doc.metadata.get(
                            "page"
                        ),

                        "chunk_id":
                        doc.metadata.get(
                            "chunk_id"
                        ),

                        "parent_id":
                        doc.metadata.get(
                            "parent_id"
                        ),

                        "chunk_type":
                        doc.metadata.get(
                            "chunk_type"
                        ),
                    },
                )
            )

        client.upsert(

            collection_name=
            collection_name,

            points=
            points,

            wait=False,
        )

        total_vectors += len(
            points
        )

    count = client.count(
        collection_name=
        collection_name
    )

    total_ms = round(
        (
            time.perf_counter()
            - start
        ) * 1000,
        2,
    )

    return {

        "collection":
        collection_name,

        "vector_count":
        count.count,

        "embedded":
        total_vectors,

        "embedding_ms":
        total_ms,
    }


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    result = (
        embed_collection(

            collection_name=
            "research_papers",

            pdf_directory=
            "user_uploads",
        )
    )

    print(result)