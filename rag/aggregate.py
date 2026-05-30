"""
rag/aggregate.py
----------------
Aggregates retrieved chunks into structured context for the LLM.
Each formatter is tailored to its query mode so the LLM
gets exactly the right structure to answer precisely.
"""

from collections import defaultdict, Counter


def _batch_label(batch: int) -> str:
    return f"Batch {batch}"


def format_concept_context(chunks: list[dict]) -> str:
    """
    For concept queries: group by company, then by batch within each company.
    Shows what angle each company takes on the concept.
    """
    by_company = defaultdict(list)
    for c in chunks:
        by_company[c["company"]].append(c)

    lines = []
    for company, co_chunks in sorted(by_company.items()):
        # Sort batches descending (most recent first)
        co_chunks.sort(key=lambda x: x["batch"], reverse=True)
        batches_seen = sorted({c["batch"] for c in co_chunks}, reverse=True)
        batch_str = ", ".join(_batch_label(b) for b in batches_seen)

        # Extract actual questions from the text
        questions = _extract_questions(co_chunks)
        q_str = "\n    ".join(f"- {q}" for q in questions[:6])

        lines.append(
            f"[{company}] — seen in {batch_str}\n"
            f"  Focus areas: {_topic_summary(co_chunks)}\n"
            f"  Sample questions:\n    {q_str}"
        )
    return "\n\n".join(lines)


def format_company_context(chunks: list[dict], company: str) -> str:
    """
    For company queries: group by batch (most recent first), list questions per batch.
    """
    by_batch = defaultdict(list)
    for c in chunks:
        by_batch[c["batch"]].append(c)

    lines = [f"Interview data for {company}:\n"]
    for batch in sorted(by_batch.keys(), reverse=True):
        batch_chunks = by_batch[batch]
        questions    = _extract_questions(batch_chunks)
        topics       = _topic_summary(batch_chunks)
        q_str        = "\n  ".join(f"- {q}" for q in questions[:8])
        lines.append(
            f"Batch {batch}:\n"
            f"  Topics covered: {topics}\n"
            f"  Questions asked:\n  {q_str}"
        )
    return "\n\n".join(lines)


def format_comparison_context(results_by_company: dict[str, list[dict]]) -> str:
    """
    For comparison queries: side-by-side per company with recent batches highlighted.
    """
    lines = []
    for company, chunks in results_by_company.items():
        if not chunks:
            continue
        recent = [c for c in chunks if c["batch"] == max(c["batch"] for c in chunks)]
        all_topics  = _topic_summary(chunks)
        rec_topics  = _topic_summary(recent)
        questions   = _extract_questions(chunks)
        batches     = sorted({c["batch"] for c in chunks}, reverse=True)

        lines.append(
            f"[{company}] (data from batches: {', '.join(str(b) for b in batches)})\n"
            f"  Overall focus  : {all_topics}\n"
            f"  Recent focus   : {rec_topics}\n"
            f"  Sample questions:\n" +
            "\n".join(f"    - {q}" for q in questions[:6])
        )
    return "\n\n".join(lines)


def format_book_index_context(results_by_topic: dict[str, list[dict]]) -> str:
    """
    For book-index queries: per topic, which companies asked it and in which batches.
    """
    lines = []
    not_found = []
    for topic, chunks in results_by_topic.items():
        if not chunks:
            not_found.append(topic)
            continue
        companies = Counter(c["company"] for c in chunks)
        batches   = sorted({c["batch"] for c in chunks}, reverse=True)
        co_str    = ", ".join(f"{co}({n}x)" for co, n in companies.most_common())
        b_str     = ", ".join(f"Batch {b}" for b in batches)
        lines.append(f"• {topic}: {co_str} | {b_str}")

    if not_found:
        lines.append(f"\nNot found in interview data: {', '.join(not_found)}")
    return "\n".join(lines)


def format_freeform_context(chunks: list[dict]) -> str:
    """Generic fallback — rich context with all metadata visible."""
    lines = []
    for c in chunks:
        snippet  = c["document"][:500].replace("\n", " ")
        topics   = ", ".join(c["topics"][:5]) if c["topics"] else "—"
        lines.append(
            f"[{c['company']} | Batch {c['batch']} | sim={c['base_sim']}]\n"
            f"Topics: {topics}\n{snippet}..."
        )
    return "\n\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_questions(chunks: list[dict]) -> list[str]:
    """Pull actual question lines from chunk documents."""
    questions = []
    for c in chunks:
        for line in c["document"].split("\n"):
            line = line.strip()
            if (re.match(r"^[\d\.\-\•●\*]", line)
                    and len(line) > 20
                    and "?" in line or len(line.split()) > 5):
                clean = re.sub(r"^[\d\.\-\•●\*\s]+", "", line).strip()
                if clean and clean not in questions:
                    questions.append(clean)
    return questions[:15]


def _topic_summary(chunks: list[dict]) -> str:
    all_topics = [t for c in chunks for t in c["topics"]]
    top = Counter(all_topics).most_common(5)
    return ", ".join(f"{t}({n}x)" for t, n in top) if top else "general"


import re
