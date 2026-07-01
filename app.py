"""
app.py

FirstRAG API Server

Run:
python -m uvicorn app:app --reload

Docs:
http://localhost:8000/docs
"""

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import joinedload, Session

from upload import router as upload_router
from generate import generate_answer
from retrieve import get_qdrant_client
from database import (
    init_db,
    get_db,
    Collection,
    ChatSession,
    Message,
    get_collection,
    create_chat_session,
    add_message,
    build_chat_history,
    get_chat_session,
    update_session_title
)

# ==================================================
# APP & CORS
# ==================================================

app = FastAPI(title="FirstRAG", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================================================
# STARTUP & ROUTERS
# ==================================================

from concurrent.futures import ThreadPoolExecutor

from retrieve import warmup
from reranker import get_reranker
from database import SessionLocal, list_collections
from retrieve_bm25 import get_bm25_index

def warmup_bm25():

    print("\nBuilding BM25 caches...")

    db = SessionLocal()

    try:

        collections = list_collections(db)

        if not collections:
            print("No collections found.")
            return

        for collection in collections:

            print(
                f"→ {collection.name}"
            )

            get_bm25_index(
                collection.name
            )

        print("BM25 cache ready.")

    finally:

        db.close()


@app.on_event("startup")
def startup():

    init_db()

    print("=" * 60)
    print("Starting FirstRAG...")

    with ThreadPoolExecutor(max_workers=2) as executor:

        executor.submit(warmup)

        executor.submit(get_reranker)

    # Build BM25 cache AFTER models are loaded
    warmup_bm25()

    print("Startup Complete")
    print("=" * 60)

app.include_router(upload_router, tags=["Upload"])

# ==================================================
# REQUEST MODELS
# ==================================================

class ChatRequest(BaseModel):
    session_id: int
    query: str
    dense_k: int = 20
    bm25_k: int = 20
    rerank_candidates: int = 20
    final_k: int = 5

class CreateSessionRequest(BaseModel):
    collection_name: str
    title: str = "New Chat"

class CreateCollectionRequest(BaseModel):
    name: str
    description: str = ""

# ==================================================
# UTILS & ROOT
# ==================================================

def sanitize_collection_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    return re.sub(r"_+", "_", name)

@app.get("/")
def root():
    return {"name": "FirstRAG", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

# ==================================================
# COLLECTIONS
# ==================================================

@app.post("/collections")
def create_collection(request: CreateCollectionRequest, db: Session = Depends(get_db)):
    collection_name = sanitize_collection_name(request.name)
    existing = db.query(Collection).filter(Collection.name == collection_name).first()

    if existing:
        raise HTTPException(status_code=409, detail="Collection already exists")

    collection_dir = Path("user_uploads") / collection_name
    upload_dir = collection_dir / "pdfs"
    upload_dir.mkdir(parents=True, exist_ok=True)

    metadata_file = collection_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump({
            "collection_name": collection_name,
            "created_at": datetime.utcnow().isoformat(),
            "documents": [],
            "document_count": 0,
            "vector_count": 0,
        }, f, indent=4)

    collection = Collection(
        name=collection_name,
        description=request.description,
        chunk_dir=str(upload_dir),
    )

    db.add(collection)
    db.commit()
    db.refresh(collection)

    return {
        "success": True,
        "id": collection.id,
        "name": collection.name,
        "description": collection.description,
    }

@app.get("/collections")
def collections(db: Session = Depends(get_db)):
    rows = db.query(Collection).order_by(Collection.created_at.desc()).all()
    return {
        "count": len(rows),
        "collections": [
            {
                "id": row.id,
                "name": row.name,
                "description": row.description,
                "chunk_dir": row.chunk_dir,
                "created_at": row.created_at,
            }
            for row in rows
        ]
    }

@app.get("/collections/{collection_name}")
def collection_details(collection_name: str, db: Session = Depends(get_db)):
    collection = get_collection(db, collection_name)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    return {
        "id": collection.id,
        "name": collection.name,
        "description": collection.description,
        "chunk_dir": collection.chunk_dir,
        "created_at": collection.created_at,
    }

@app.get("/collections/{collection_name}/stats")
def collection_stats(collection_name: str, db: Session = Depends(get_db)):
    collection = db.query(Collection).filter(Collection.name == collection_name).first()
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    # 1. Query sessions and messages from the database
    session_count = db.query(ChatSession).filter(ChatSession.collection_id == collection.id).count()
    message_count = (
        db.query(Message)
        .join(ChatSession, Message.session_id == ChatSession.id)
        .filter(ChatSession.collection_id == collection.id)
        .count()
    )

    # 2. BULLETPROOF DOCUMENT COUNT: Physically check the PDFs on the hard drive
    document_count = 0
    document_list = []
    pdf_dir = Path(collection.chunk_dir)
    
    if pdf_dir.exists():
        pdfs = list(pdf_dir.glob("*.pdf"))
        document_count = len(pdfs)
        document_list = [pdf.name for pdf in pdfs]

    # 3. Return ALL stats (including documents list)
    return {
        "collection": collection.name,
        "description": collection.description,
        "chunk_dir": collection.chunk_dir,
        "created_at": collection.created_at,
        "sessions": session_count,
        "messages": message_count,
        "documents": document_count,
        "document_list": document_list, # <--- Sending the list to the frontend!
    }

@app.delete("/collections/{collection_name}")
def delete_collection(collection_name: str, db: Session = Depends(get_db)):
    collection = db.query(Collection).filter(Collection.name == collection_name).first()
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    # 1. Safely delete Qdrant vectors
    try:
        client = get_qdrant_client()
        client.delete_collection(collection_name)
    except Exception as e:
        print(f"Qdrant delete warning: {e}")

    # 2. Safely delete local files (ignore_errors prevents Windows file lock crashes!)
    upload_dir = Path("user_uploads") / collection_name
    if upload_dir.exists():
        shutil.rmtree(upload_dir, ignore_errors=True)

    # 3. Manually sweep the database to prevent SQLite lockups
    sessions = db.query(ChatSession).filter(ChatSession.collection_id == collection.id).all()
    for s in sessions:
        # Delete all messages inside the session first
        db.query(Message).filter(Message.session_id == s.id).delete()
        # Then delete the session
        db.delete(s)

    # 4. Finally, delete the parent collection
    db.delete(collection)
    db.commit()

    return {
        "success": True,
        "collection": collection_name,
        "message": "Collection and all associated data deleted successfully",
    }
# ==================================================
# SESSIONS
# ==================================================

@app.post("/sessions")
def create_session(request: CreateSessionRequest, db: Session = Depends(get_db)):
    collection = get_collection(db, request.collection_name)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    session = create_chat_session(
        db=db,
        collection_id=collection.id,
        title=request.title,
    )

    return {
        "session_id": session.id,
        "collection": collection.name,
    }

@app.get("/sessions")
def list_sessions(db: Session = Depends(get_db)):
    sessions = (
        db.query(ChatSession)
        .options(joinedload(ChatSession.collection))
        .order_by(ChatSession.updated_at.desc())
        .all()
    )

    return {
        "count": len(sessions),
        "sessions": [
            {
                "id": s.id,
                "title": s.title,
                "collection_id": s.collection_id,
                "collection_name": s.collection.name if s.collection else None,
                "updated_at": s.updated_at,
                "created_at": s.created_at,
            }
            for s in sessions
        ]
    }

# ==================================================
# HISTORY
# ==================================================

@app.get("/history/{session_id}")
def history(session_id: int, db: Session = Depends(get_db)):
    history_data = build_chat_history(db, session_id)
    return {
        "session_id": session_id,
        "messages": history_data,
    }

# ==================================================
# CHAT
# ==================================================

@app.post("/chat")
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    session = get_chat_session(db, request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # ----------------------------------
    # AUTO RENAME "New Chat"
    # ----------------------------------
    if session.title and session.title.strip().lower() == "new chat":
        update_session_title(
            db=db,
            session_id=request.session_id,
            title=request.query[:60]
        )
        session = get_chat_session(db, request.session_id)

    collection = session.collection
    chat_history = build_chat_history(db, request.session_id)

    add_message(
        db=db,
        session_id=request.session_id,
        role="user",
        content=request.query,
    )

    try:
        response = generate_answer(
            query=request.query,
            collection_name=collection.name,
            chat_history=chat_history,
            dense_k=request.dense_k,
            bm25_k=request.bm25_k,
            rerank_candidates=request.rerank_candidates,
            final_k=request.final_k,
        )
    except Exception:
        import traceback
        print("\n" + "=" * 80)
        print("CHAT ERROR")
        print("=" * 80)
        traceback.print_exc()
        print("=" * 80 + "\n")
        raise

    answer = response["answer"]
    sources_data = response.get("sources", []) # Get the sources

    add_message(
        db=db,
        session_id=request.session_id,
        role="assistant",
        content=answer,
        sources=json.dumps(sources_data) # <--- Convert to string and save!
    )

    return {

        "answer": answer,

        "route": response.get("route"),

        "sources": response.get("sources", []),

        # -----------------------------
        # Legacy (keep frontend compatibility)
        # -----------------------------
        "retrieval_metrics": (
            response.get("retrieval_metrics")
            or response.get("retrieval", {})
        ),

        "routing_metrics": response.get("routing_metrics", {}),

        # -----------------------------
        # New Observability API
        # -----------------------------
        "retrieval": response.get("retrieval", {}),

        "observability": response.get("observability", {}),

        "session_title": session.title,
    }

# ==================================================
# SESSION ACTIONS (Add this to app.py)
# ==================================================

class RenameSessionRequest(BaseModel):
    title: str

@app.put("/sessions/{session_id}")
def rename_session(session_id: int, request: RenameSessionRequest, db: Session = Depends(get_db)):
    session = get_chat_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    update_session_title(db, session_id, title=request.title)
    return {"success": True, "new_title": request.title}

@app.delete("/sessions/{session_id}")
def delete_session(session_id: int, db: Session = Depends(get_db)):
    session = get_chat_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    db.delete(session) # SQLAlchemy automatically deletes associated messages!
    db.commit()
    
    return {"success": True, "message": "Session deleted"}