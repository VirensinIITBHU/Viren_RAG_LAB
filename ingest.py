"""
ingest.py

Production Streaming Ingestion

Architecture

PDF
 ↓
Lazy Load
 ↓
Chunk Stream
 ↓
Embed Batch
 ↓
Qdrant Batch

No Pickle
Low Memory
Large Corpus Ready
"""

from pathlib import Path
from typing import Generator

from langchain_community.document_loaders import (
    PyMuPDFLoader,
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from langchain_core.documents import (
    Document,
)

# ==================================================
# CONFIG
# ==================================================

CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 80

PARENT_CHUNK_SIZE = 1500
PARENT_CHUNK_OVERLAP = 200

# ==================================================
# SPLITTERS
# ==================================================

parent_splitter = (
    RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
    )
)

child_splitter = (
    RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
    )
)

# ==================================================
# PDF DISCOVERY
# ==================================================

def discover_pdfs(
    pdf_directory: str,
):

    root = Path(pdf_directory)

    if not root.exists():

        raise FileNotFoundError(
            f"{pdf_directory} not found"
        )

    pdfs = list(
        root.rglob("*.pdf")
    )

    if not pdfs:

        raise ValueError(
            "No PDFs found"
        )

    return pdfs


# ==================================================
# STREAM PAGES
# ==================================================

def stream_pages(
    pdf_directory: str,
) -> Generator[Document, None, None]:

    pdf_files = discover_pdfs(
        pdf_directory
    )

    for pdf_path in pdf_files:

        loader = PyMuPDFLoader(
            str(pdf_path)
        )

        pages = loader.load()

        for page in pages:

            page.metadata[
                "paper_name"
            ] = pdf_path.stem

            yield page


# ==================================================
# STREAM CHUNKS
# ==================================================

def stream_chunks(
    pdf_directory: str,
):

    chunk_id = 0
    parent_id = 0

    for page in stream_pages(
        pdf_directory
    ):

        parents = (
            parent_splitter
            .split_documents(
                [page]
            )
        )

        for parent in parents:

            current_parent_id = (
                parent_id
            )

            parent.metadata[
                "parent_id"
            ] = current_parent_id

            parent.metadata[
                "chunk_type"
            ] = "parent"

            parent_id += 1

            children = (
                child_splitter
                .split_documents(
                    [parent]
                )
            )

            for child in children:

                child.metadata[
                    "parent_id"
                ] = (
                    current_parent_id
                )

                child.metadata[
                    "chunk_id"
                ] = chunk_id

                child.metadata[
                    "chunk_type"
                ] = "child"

                chunk_id += 1

                yield child


# ==================================================
# BATCH GENERATOR
# ==================================================

def chunk_batches(
    pdf_directory: str,
    batch_size: int = 100,
):

    batch = []

    for chunk in stream_chunks(
        pdf_directory
    ):

        batch.append(chunk)

        if len(batch) >= batch_size:

            yield batch

            batch = []

    if batch:

        yield batch


# ==================================================
# INGEST STATS
# ==================================================

def collection_stats(
    pdf_directory: str,
):

    documents = len(
        discover_pdfs(
            pdf_directory
        )
    )

    child_chunks = 0

    for batch in chunk_batches(
        pdf_directory,
        batch_size=500,
    ):

        child_chunks += len(batch)

    return {

        "documents":
        documents,

        "child_chunks":
        child_chunks,
    }