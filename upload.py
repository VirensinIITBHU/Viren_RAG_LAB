"""
upload.py

Production Upload Service
"""

import json
import re
import shutil
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session

from embed import embed_collection
from retrieve_bm25 import invalidate_collection_cache
from database import get_collection, create_collection_record, get_db

router = APIRouter()

UPLOAD_ROOT = Path("user_uploads")
UPLOAD_ROOT.mkdir(exist_ok=True)

def sanitize_collection_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    return re.sub(r"_+", "_", name)

def save_metadata(collection_dir: Path, metadata: dict):
    with open(collection_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

def background_process_and_update(collection_name: str, pdf_dir: str, collection_dir: Path):
    """Runs embedding in the background and updates metadata when finished."""
    try:
        embed_result = embed_collection(
            collection_name=collection_name,
            pdf_directory=pdf_dir,
        )
        invalidate_collection_cache(collection_name)
        
        metadata_file = collection_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            metadata["vector_count"] = embed_result.get("vector_count", 0)
            metadata["status"] = "ready"
            save_metadata(collection_dir, metadata)
            
    except Exception as e:
        print(f"Background embedding failed for {collection_name}: {e}")
        # Optionally update metadata status to "failed" here

@router.post("/upload", status_code=202)
def upload_documents(
    background_tasks: BackgroundTasks,
    collection_name: str = Form(...),
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    collection_name = sanitize_collection_name(collection_name)

    if get_collection(db, collection_name):
        raise HTTPException(status_code=409, detail=f"Collection {collection_name} already exists")

    collection_dir = UPLOAD_ROOT / collection_name
    pdf_dir = collection_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    uploaded_files = []

    try:
        for file in files:
            if not file.filename or not file.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF")

            destination = pdf_dir / file.filename
            with open(destination, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_files.append(file.filename)

        create_collection_record(
            db=db,
            name=collection_name,
            chunk_dir=str(pdf_dir),
            description=f"{len(uploaded_files)} PDF(s)",
        )

        # Initial metadata with "processing" status
        metadata = {
            "collection_name": collection_name,
            "created_at": datetime.utcnow().isoformat(),
            "documents": uploaded_files,
            "document_count": len(uploaded_files),
            "vector_count": 0,
            "status": "processing"
        }
        save_metadata(collection_dir, metadata)

        # Trigger background task (CPU intensive work)
        background_tasks.add_task(
            background_process_and_update,
            collection_name=collection_name,
            pdf_dir=str(pdf_dir),
            collection_dir=collection_dir
        )

        return {
            "status": "processing",
            "collection": collection_name,
            "documents": uploaded_files,
            "document_count": len(uploaded_files),
            "message": "Documents saved and are currently being embedded in the background."
        }

    except Exception:
        if collection_dir.exists():
            shutil.rmtree(collection_dir, ignore_errors=True)
        raise


@router.post("/collections/{collection_name}/add", status_code=202)
def add_documents_to_collection(
    collection_name: str,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    collection_name = sanitize_collection_name(collection_name)
    collection_dir = UPLOAD_ROOT / collection_name

    if not collection_dir.exists():
        raise HTTPException(status_code=404, detail="Collection not found")

    pdf_dir = collection_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    metadata_file = collection_dir / "metadata.json"
    if not metadata_file.exists():
        save_metadata(collection_dir, {
            "collection_name": collection_name,
            "created_at": datetime.utcnow().isoformat(),
            "documents": [],
            "document_count": 0,
            "vector_count": 0,
            "status": "ready"
        })

    uploaded_files = []
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            continue

        destination = pdf_dir / file.filename
        if destination.exists():
            continue

        with open(destination, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        uploaded_files.append(file.filename)

    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if uploaded_files:
        existing_docs = set(metadata.get("documents", []))
        existing_docs.update(uploaded_files)
        metadata["documents"] = sorted(existing_docs)
        metadata["document_count"] = len(metadata["documents"])
        metadata["status"] = "processing"
        save_metadata(collection_dir, metadata)

        background_tasks.add_task(
            background_process_and_update,
            collection_name=collection_name,
            pdf_dir=str(pdf_dir),
            collection_dir=collection_dir
        )
        
        message = "New files saved and are being processed in the background."
    else:
        message = "No new files uploaded."

    return {
        "status": "processing" if uploaded_files else "success",
        "collection": collection_name,
        "new_files": uploaded_files,
        "total_documents": metadata["document_count"],
        "message": message
    }