"""
ingest/embed.py
---------------
Reads parsed_chunks.json → embeds each chunk → stores in ChromaDB.

RAG concept learned here:
  - Embeddings turn text into vectors so "semantic similarity" can be measured.
  - ChromaDB stores both the vector AND the metadata (company, batch, topics).
  - Metadata lets you filter BEFORE semantic search — much faster and more precise.

Run:  python -m ingest.embed
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, CHROMA_DIR, EMBEDDING_MODEL, CHROMA_COLLECTION


def load_chunks() -> list[dict]:
    path = DATA_DIR / "parsed_chunks.json"
    if not path.exists():
        print("parsed_chunks.json not found. Run `python -m ingest.parse` first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def build_embed_text(chunk: dict) -> str:
    """
    What we actually embed — a focused summary rather than the raw full text.
    Including company + topics in the embed text improves retrieval quality
    because the embedding captures both the content AND the context.

    RAG insight: what you embed is a design decision.
    Embedding raw text vs a structured summary gives different retrieval behaviour.
    """
    topic_str = ", ".join(chunk["topics"]) if chunk["topics"] else "general"
    round_names = ", ".join(r["round_name"] for r in chunk["rounds"])
    return (
        f"Company: {chunk['company']}. "
        f"Topics covered: {topic_str}. "
        f"Rounds: {round_names}. "
        f"Details: {chunk['full_text'][:600]}"
    )


def embed_and_store(chunks: list[dict]):
    # Import here so the file is importable even without deps installed
    from sentence_transformers import SentenceTransformer
    import chromadb

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Connecting to ChromaDB at {CHROMA_DIR}")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete existing collection to avoid duplicates on re-run
    try:
        client.delete_collection(CHROMA_COLLECTION)
        print(f"Cleared existing collection '{CHROMA_COLLECTION}'")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}   # cosine similarity for sentence embeddings
    )

    print(f"Embedding {len(chunks)} chunks...")

    # Batch embed for speed
    texts      = [build_embed_text(c) for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    # Prepare metadata — ChromaDB only accepts str/int/float/bool values
    # Lists (like topics) must be serialised to a comma-separated string
    metadatas = [
        {
            "company":    c["company"],
            "batch":      c["batch"],
            "source":     c["source"],
            "topics":     ",".join(c["topics"]),      # stored as string
            "num_rounds": c["num_rounds"],
        }
        for c in chunks
    ]

    ids       = [c["chunk_id"] for c in chunks]
    documents = [c["full_text"] for c in chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"\nStored {collection.count()} chunks in ChromaDB.")
    print(f"Location: {CHROMA_DIR}")
    print("\nRAG concept: each chunk now has a vector AND metadata.")
    print("At query time you can filter by company/batch/topics BEFORE semantic search.")


if __name__ == "__main__":
    chunks = load_chunks()
    embed_and_store(chunks)
