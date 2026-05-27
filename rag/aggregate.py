"""
rag/aggregate.py
----------------
Takes raw retrieved chunks and produces structured summaries:
  - Company-wise breakdown
  - Topic frequency ranking
  - Book-index cross-reference table

RAG concept learned here:
  - Retrieval returns raw chunks — aggregation is pure Python, not LLM.
  - Knowing when to use Python vs LLM is a core RAG design skill.
    Python: counting, grouping, sorting.
    LLM: language synthesis, explanation, connecting ideas.
"""

from collections import defaultdict, Counter


def group_by_company(chunks: list[dict]) -> dict[str, list[dict]]:
    """Group retrieved chunks by company."""
    groups = defaultdict(list)
    for chunk in chunks:
        groups[chunk["company"]].append(chunk)
    return dict(groups)


def group_by_batch(chunks: list[dict]) -> dict[int, list[dict]]:
    """Group retrieved chunks by batch (proxy for year)."""
    groups = defaultdict(list)
    for chunk in chunks:
        groups[chunk["batch"]].append(chunk)
    return dict(groups)


def topic_frequency(chunks: list[dict]) -> list[tuple[str, int]]:
    """
    Count how many chunks each topic tag appears in.
    Returns sorted list of (topic, count) descending.
    """
    all_topics = [t for chunk in chunks for t in chunk["topics"] if t]
    return Counter(all_topics).most_common()


def company_topic_matrix(chunks: list[dict]) -> dict[str, dict[str, int]]:
    """
    Build a matrix: {company → {topic → count}}.
    Useful for comparing what different companies focus on.
    """
    matrix = defaultdict(Counter)
    for chunk in chunks:
        for topic in chunk["topics"]:
            matrix[chunk["company"]][topic] += 1
    return {co: dict(counts) for co, counts in matrix.items()}


def summarise_freeform(chunks: list[dict]) -> str:
    """
    Format retrieved chunks as a readable context string for the LLM prompt.
    Each entry shows: [COMPANY | Batch X | topics] text snippet
    """
    lines = []
    for i, c in enumerate(chunks, 1):
        topic_str = ", ".join(c["topics"]) if c["topics"] else "—"
        snippet   = c["document"][:400].replace("\n", " ")
        lines.append(
            f"[{i}] {c['company']} | Batch {c['batch']} | Topics: {topic_str}\n"
            f"     {snippet}..."
        )
    return "\n\n".join(lines)


def summarise_book_index(results_by_topic: dict[str, list[dict]]) -> str:
    """
    Format book-index query results as context for the LLM.

    results_by_topic: {topic_name → list of retrieved chunks}
    """
    lines = []
    for topic, chunks in results_by_topic.items():
        if not chunks:
            lines.append(f"• {topic}: No matches found in interview data.")
            continue

        companies = Counter(c["company"] for c in chunks)
        batches   = Counter(c["batch"] for c in chunks)
        co_str    = ", ".join(f"{co}({n})" for co, n in companies.most_common())
        batch_str = ", ".join(f"Batch {b}({n})" for b, n in sorted(batches.items()))
        lines.append(
            f"• {topic}: found in {len(chunks)} entries | "
            f"Companies: {co_str} | {batch_str}"
        )

    return "\n".join(lines)
