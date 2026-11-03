
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from load_policies import (
    load_policy_documents,
    split_policy_documents,
)

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY is missing from .env")

# 1. Load the original policy documents
policies = load_policy_documents()

# 2. Reuse the existing chunking function
chunks = split_policy_documents(policies)

if not chunks:
    raise ValueError("No policy chunks found.")

# 3. Convert the existing chunks into LangChain Documents
documents = [
    Document(
        page_content=chunk["text"],
        metadata={
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
        },
    )
    for chunk in chunks
]

print(f"Policies loaded: {len(policies)}")
print(f"Existing chunks reused: {len(documents)}")

# 4. Create embeddings using Gemini
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

# 5. Store the embeddings in Chroma
database_path = PROJECT_DIR / "data" / "chroma_db"

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=str(database_path),
    collection_name="shop_policy",
)

print("\nPolicy index created successfully!")

# 6. Test semantic search
question = "How long do I have to return an item?"
results = vector_store.similarity_search(question, k=2)

print("\nSearch results:")
for result in results:
    print(f"\nSource: {result.metadata['source']}")
    print(f"Chunk: {result.metadata['chunk_id']}")
    print(result.page_content)
