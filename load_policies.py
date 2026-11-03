
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Find the policy folder relative to this Python file
PROJECT_DIR = Path(__file__).resolve().parent
POLICY_DIR = PROJECT_DIR / "data" / "policies"


def load_policy_documents():
    documents = []

    for file_path in sorted(POLICY_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            print(f"Skipping empty file: {file_path.name}")
            continue

        documents.append({
            "source": file_path.name,
            "text": text,
        })

    return documents


def split_policy_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for document in documents:
        pieces = splitter.split_text(document["text"])

        for index, piece in enumerate(pieces):
            chunks.append({
                "source": document["source"],
                "chunk_id": index,
                "text": piece,
            })

    return chunks


if __name__ == "__main__":
    documents = load_policy_documents()
    chunks = split_policy_documents(documents)

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")

    for chunk in chunks:
        print(
            f"\nSource: {chunk['source']}"
            f"\nChunk: {chunk['chunk_id']}"
            f"\nText: {chunk['text']}"
        )
        print("-" * 50)
