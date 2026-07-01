"""
database.py

Persistence Layer
"""

from datetime import datetime
import json
from sqlalchemy.orm import joinedload, Session
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import (
    declarative_base,
    sessionmaker,
    relationship,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DATABASE_URL = "sqlite:///firstrag.db"

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)

# Enable Write-Ahead Logging (WAL) for concurrent read/writes
with engine.connect() as conn:
    conn.execute(text("PRAGMA journal_mode=WAL;"))

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)

Base = declarative_base()

# --------------------------------------------------
# DEPENDENCY INJECTION
# --------------------------------------------------

def get_db():
    """Yields a single database session per web request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --------------------------------------------------
# COLLECTIONS
# --------------------------------------------------

class Collection(Base):
    __tablename__ = "collections"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    chunk_dir = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    sessions = relationship(
        "ChatSession",
        back_populates="collection",
        cascade="all, delete-orphan",
    )

# --------------------------------------------------
# CHAT SESSIONS
# --------------------------------------------------

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, default="New Chat", nullable=False)
    collection_id = Column(Integer, ForeignKey("collections.id"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    collection = relationship("Collection", back_populates="sessions")
    messages = relationship(
        "Message",
        back_populates="session",
        cascade="all, delete-orphan",
    )

# --------------------------------------------------
# MESSAGES
# --------------------------------------------------

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False, index=True)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    sources = Column(Text, nullable=True) 
    session = relationship("ChatSession", back_populates="messages")

# --------------------------------------------------
# PARENT CHUNKS (NEW: For Advanced RAG)
# --------------------------------------------------

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    collection_name = Column(String, index=True, nullable=False)
    parent_id = Column(Integer, index=True, nullable=False)
    text = Column(Text, nullable=False)
    paper_name = Column(String, nullable=True)

    # Prevents duplicate parent rows when a collection is re-ingested
    # (retry after partial failure, manual re-embed, etc.) and is the
    # conflict target for the upsert in save_parent_chunks_batch().
    __table_args__ = (
        UniqueConstraint(
            "collection_name",
            "parent_id",
            name="uq_document_chunks_collection_parent",
        ),
    )

# --------------------------------------------------
# DB INIT
# --------------------------------------------------

def init_db():
    Base.metadata.create_all(bind=engine)
    print("Database initialized (WAL mode active).")

# --------------------------------------------------
# COLLECTION HELPERS
# --------------------------------------------------

def create_collection_record(db: Session, name: str, chunk_dir: str, description: str = None):
    existing = db.query(Collection).filter(Collection.name == name).first()
    if existing:
        return existing

    collection = Collection(
        name=name,
        chunk_dir=chunk_dir,
        description=description,
    )
    db.add(collection)
    db.commit()
    db.refresh(collection)
    return collection

def get_collection(db: Session, collection_name: str):
    return db.query(Collection).filter(Collection.name == collection_name).first()

def list_collections(db: Session):
    return db.query(Collection).order_by(Collection.created_at.desc()).all()

# --------------------------------------------------
# SESSION HELPERS
# --------------------------------------------------

def create_chat_session(db: Session, collection_id: int, title: str = "New Chat"):
    session = ChatSession(
        title=title,
        collection_id=collection_id,
        updated_at=datetime.utcnow(),
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session

def get_chat_session(db: Session, session_id: int):
    return (
        db.query(ChatSession)
        .options(joinedload(ChatSession.collection))
        .filter(ChatSession.id == session_id)
        .first()
    )

def get_session_messages(db: Session, session_id: int):
    return (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
        .all()
    )

def touch_chat_session(db: Session, session_id: int):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session:
        session.updated_at = datetime.utcnow()
        db.commit()
    return session

def update_session_title(db: Session, session_id: int, title: str):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session:
        session.title = title
        db.commit()

# --------------------------------------------------
# MESSAGE HELPERS
# --------------------------------------------------

def add_message(db: Session, session_id: int, role: str, content: str, sources: str = None):
    message = Message(
        session_id=session_id,
        role=role,
        content=content,
        sources=sources
    )
    db.add(message)

    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session:
        session.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(message)
    return message

# --------------------------------------------------
# CHUNK HELPERS
# --------------------------------------------------

def save_parent_chunks_batch(db: Session, chunks_data: list[dict]):
    """
    Bulk upsert parent chunks into SQLite.

    Uses INSERT ... ON CONFLICT DO UPDATE keyed on
    (collection_name, parent_id) so re-running embed.py over the same
    collection (retry after a partial failure, manual re-embed) updates
    existing parent rows in place instead of duplicating them.
    """
    if not chunks_data:
        return

    stmt = sqlite_insert(DocumentChunk).values(chunks_data)
    stmt = stmt.on_conflict_do_update(
        index_elements=["collection_name", "parent_id"],
        set_={
            "text": stmt.excluded.text,
            "paper_name": stmt.excluded.paper_name,
        },
    )
    db.execute(stmt)
    db.commit()


def get_parent_chunks_by_ids(
    db: Session,
    collection_name: str,
    parent_ids: list[int],
) -> dict:
    """
    Fetches parent chunk text for small-to-big retrieval.

    Given the parent_ids surfaced by the reranked child chunks, returns
    a {parent_id: {"text": ..., "paper_name": ...}} lookup so the
    generation layer can expand each winning child chunk to its full
    parent context. Deduplicates automatically since callers may pass
    the same parent_id more than once (multiple child chunks can share
    a parent) and a single IN query covers all of them.
    """
    if not parent_ids:
        return {}

    unique_ids = list(set(parent_ids))

    rows = (
        db.query(DocumentChunk)
        .filter(
            DocumentChunk.collection_name == collection_name,
            DocumentChunk.parent_id.in_(unique_ids),
        )
        .all()
    )

    return {
        row.parent_id: {
            "text": row.text,
            "paper_name": row.paper_name,
        }
        for row in rows
    }


def delete_parent_chunks_for_collection(db: Session, collection_name: str):
    """
    Deletes all parent chunk rows for a collection.

    Needed alongside the existing Qdrant collection delete in app.py's
    DELETE /collections/{name} route — today that route drops the
    Qdrant collection and local files but leaves orphaned rows in
    document_chunks, which silently grows the table and can bleed into
    a future collection that reuses the same name.
    """
    db.query(DocumentChunk).filter(
        DocumentChunk.collection_name == collection_name
    ).delete()
    db.commit()

# --------------------------------------------------
# CHAT HISTORY FORMATTER
# --------------------------------------------------

def build_chat_history(db: Session, session_id: int):
    messages = get_session_messages(db, session_id)
    history = []
    for msg in messages:
        history.append({
            "role": msg.role,
            "content": msg.content,
            "sources": json.loads(msg.sources) if msg.sources else []
        })
    return history

if __name__ == "__main__":
    init_db()
    print("Database ready.")