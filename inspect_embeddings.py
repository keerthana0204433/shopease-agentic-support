
from pathlib import Path
from langchain_chroma import Chroma

PROJECT_DIR = Path(__file__).resolve().parent
DATABASE_PATH = PROJECT_DIR / "data" / "chroma_db"

# Open the existing database without an embedding model.
# We only want to inspect stored records.
db = Chroma(
    collection_name="shop_policy",
    persist_directory=str(DATABASE_PATH),
)

# Retrieve stored records, including their embeddings
data = db.get(include=["documents", "metadatas", "embeddings"])

print("Number of stored chunks:", len(data["ids"]))

for i, record_id in enumerate(data["ids"]):
    print("\n" + "=" * 50)
    print("Record ID:", record_id)
    print("Metadata:", data["metadatas"][i])
    print("Stored text:", data["documents"][i])

    vector = data["embeddings"][i]

    if vector is not None:
        print("Embedding dimensions:", len(vector))
        print("First 10 embedding values:", vector[:10])
    else:
        print("No embedding returned for this record.")
