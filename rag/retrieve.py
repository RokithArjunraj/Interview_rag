"""
rag/retrieve.py
---------------
Retrieval with recency-weighted re-ranking.

After semantic search, each chunk's similarity score is boosted
based on how recent the batch is. This ensures recent batches
surface higher without completely suppressing older ones.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHROMA_DIR, CHROMA_COLLECTION, EMBEDDING_MODEL, TOP_K, RECENCY_BONUS


def get_collection():
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(CHROMA_COLLECTION)


def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


def _build_where(company=None, batch=None, topic_filter=None):
    """Build ChromaDB where clause safely — never passes empty dict."""
    filters = {}
    if company:
        filters["company"] = company.upper()
    if batch:
        filters["batch"] = batch
    if topic_filter:
        filters["topics"] = {"$contains": topic_filter}

    if len(filters) == 0:
        return None
    if len(filters) == 1:
        return filters
    return {"$and": [{k: v} for k, v in filters.items()]}


def _recency_score(batch: int, max_batch: int) -> float:
    """
    Recency bonus: chunks from the most recent batch get the highest boost.
    batch 10 with max_batch=10 → bonus = RECENCY_BONUS * 10
    batch 3  with max_batch=10 → bonus = RECENCY_BONUS * 3
    """
    return RECENCY_BONUS * batch


def retrieve(
    query: str,
    model,
    collection,
    company: str | None = None,
    batch: int | None = None,
    topic_filter: str | None = None,
    top_k: int = TOP_K,
    recency_weight: bool = True,
) -> list[dict]:
    """
    Semantic search + recency re-ranking.
    Fetches 2x top_k from ChromaDB, re-ranks by (similarity + recency bonus),
    returns top_k after re-ranking.
    """
    where_clause    = _build_where(company, batch, topic_filter)
    query_embedding = model.encode(query).tolist()

    fetch_k = min(top_k * 2, collection.count())

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_k,
        where=where_clause,
        include=["documents", "metadatas", "distances"],
    )

    # Get max batch in this result set for normalisation
    all_batches = [m["batch"] for m in results["metadatas"][0]]
    max_batch   = max(all_batches) if all_batches else 1

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        base_sim = round(1 - dist, 4)
        bonus    = _recency_score(meta["batch"], max_batch) if recency_weight else 0
        chunks.append({
            "document":      doc,
            "company":       meta["company"],
            "batch":         meta["batch"],
            "topics":        meta["topics"].split(",") if meta["topics"] else [],
            "num_rounds":    meta.get("num_rounds", 0),
            "base_sim":      base_sim,
            "recency_bonus": round(bonus, 3),
            "final_score":   round(base_sim + bonus, 4),
        })

    # Re-rank by final_score (similarity + recency)
    chunks.sort(key=lambda x: x["final_score"], reverse=True)
    return chunks[:top_k]


def retrieve_for_topic(topic, model, collection, company=None, top_k=10):
    return retrieve(
        query=topic,
        model=model,
        collection=collection,
        company=company,
        top_k=top_k,
        recency_weight=True,
    )


def retrieve_multi_company(query, model, collection, companies, top_k_per_company=8):
    """
    For comparison queries — fetch separately per company so
    each company is fairly represented regardless of data volume.
    """
    results = {}
    for co in companies:
        chunks = retrieve(query, model, collection, company=co,
                          top_k=top_k_per_company, recency_weight=True)
        if chunks:
            results[co] = chunks
    return results
