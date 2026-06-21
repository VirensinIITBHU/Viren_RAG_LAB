"""
database.py

Persistence Layer
"""

from datetime import datetime
from sqlalchemy.orm import joinedload, Session
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
)
from sqlalchemy.orm import (
    declarative_base,
    sessionmaker,
    relationship,
)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DATABASE_URL = "sqlite:///firstrag.db"

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)

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

    session = relationship("ChatSession", back_populates="messages")

# --------------------------------------------------
# DB INIT
# --------------------------------------------------

def init_db():
    Base.metadata.create_all(bind=engine)
    print("Database initialized.")

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

def add_message(db: Session, session_id: int, role: str, content: str):
    message = Message(
        session_id=session_id,
        role=role,
        content=content,
    )
    db.add(message)

    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session:
        session.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(message)
    return message

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
        })
    return history

if __name__ == "__main__":
    init_db()
    print("Database ready.")