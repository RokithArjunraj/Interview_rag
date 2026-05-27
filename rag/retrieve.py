"""
rag/retrieve.py
---------------
Retrieves relevant chunks from ChromaDB given a query + optional filters.

RAG concept learned here:
  - Semantic search finds chunks that are *meaningfully similar*, not just keyword matches.
  - Metadata filters pre-narrow the search space before similarity is computed.
  - Order matters: filter first, then search → faster and more relevant results.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHROMA_DIR, CHROMA_COLLECTION, EMBEDDING_MODEL, TOP_K


def get_collection():
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(CHROMA_COLLECTION)


def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


def retrieve(
    query: str,
    model,
    collection,
    company: str | None = None,
    batch: int | None = None,
    topic_filter: str | None = None,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Semantic search with optional metadata pre-filters.

    Args:
        query        : natural language query
        model        : SentenceTransformer instance
        collection   : ChromaDB collection
        company      : filter to a specific company (e.g. "BCG")
        batch        : filter to a specific batch (e.g. 10)
        topic_filter : filter to chunks containing this topic tag
        top_k        : how many chunks to return

    Returns:
        List of result dicts with keys: document, company, batch, topics, distance
    """
    # Build the metadata filter (ChromaDB "where" clause)
    # Rules:
    #   - Empty dict       → pass None  (ChromaDB rejects {})
    #   - Single filter    → {"key": value}
    #   - Multiple filters → {"$and": [{k:v}, {k:v}, ...]}
    filters = {}
    if company:
        filters["company"] = company.upper()
    if batch:
        filters["batch"] = batch
    if topic_filter:
        filters["topics"] = {"$contains": topic_filter}

    if len(filters) == 0:
        where_clause = None
    elif len(filters) == 1:
        where_clause = filters
    else:
        where_clause = {"$and": [{k: v} for k, v in filters.items()]}

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where=where_clause,
        include=["documents", "metadatas", "distances"],
    )

    # Flatten into a clean list of dicts
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "document":   doc,
            "company":    meta["company"],
            "batch":      meta["batch"],
            "topics":     meta["topics"].split(",") if meta["topics"] else [],
            "num_rounds": meta.get("num_rounds", 0),
            "similarity": round(1 - dist, 3),
        })

    return chunks


def retrieve_for_topic(
    topic: str,
    model,
    collection,
    company: str | None = None,
    top_k: int = 8,
) -> list[dict]:
    """
    Used for book-index mode: search for a single specific topic.
    Returns results sorted by similarity.
    """
    return retrieve(
        query=topic,
        model=model,
        collection=collection,
        company=company,
        top_k=top_k,
    )