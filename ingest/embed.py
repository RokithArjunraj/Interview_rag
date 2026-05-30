"""
ingest/embed.py
---------------
Embeds chunks into ChromaDB using the structured embed_text field.
The embed_text includes company + batch + topics + questions asked,
so retrieval captures all three dimensions simultaneously.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, CHROMA_DIR, EMBEDDING_MODEL, CHROMA_COLLECTION


def load_chunks() -> list[dict]:
    path = DATA_DIR / "parsed_chunks.json"
    if not path.exists():
        print("Run `python -m ingest.parse` first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def embed_and_store(chunks: list[dict]):
    from sentence_transformers import SentenceTransformer
    import chromadb

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(CHROMA_COLLECTION)
        print("Cleared existing collection.")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"Embedding {len(chunks)} chunks...")

    # Use the structured embed_text — richer signal than raw full_text
    texts      = [c["embed_text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    metadatas = [
        {
            "company":    c["company"],
            "batch":      c["batch"],
            "source":     c["source"],
            "topics":     ",".join(c["topics"]),
            "num_rounds": c["num_rounds"],
        }
        for c in chunks
    ]

    collection.add(
        ids        = [c["chunk_id"] for c in chunks],
        embeddings = embeddings,
        documents  = [c["full_text"] for c in chunks],  # LLM reads full_text
        metadatas  = metadatas,
    )

    print(f"\nStored {collection.count()} chunks.")

    from collections import Counter
    batches = Counter(c["batch"] for c in chunks)
    print("\nBatches in store:")
    for b, n in sorted(batches.items()):
        print(f"  Batch {b}: {n} chunks")


if __name__ == "__main__":
    chunks = load_chunks()
    embed_and_store(chunks)
