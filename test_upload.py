from database import SessionLocal
from database import Collection

db = SessionLocal()

collection = Collection(
    name="research_papers",
    chunk_dir="user_uploads/research_papers"
)

db.add(collection)
db.commit()

print("Done")