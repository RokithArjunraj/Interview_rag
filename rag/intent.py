"""
rag/intent.py
-------------
Classifies user input into one of three query modes:

  "freeform"    → a plain question about interviews
  "book_index"  → user pastes a list of topics (from ISLP, PRML, etc.)
  "comparison"  → user asks to compare two or more companies

RAG concept learned here:
  - Not all queries should be handled the same way.
  - A query router lets you pick the right retrieval strategy per intent.
  - Start with simple keyword rules; upgrade to LLM classification later.
"""

import re


# Signals that the user is pasting a reference topic list
BOOK_INDEX_SIGNALS = [
    r"\bhere (is|are)\b",
    r"\bthese topics\b",
    r"\bthis list\b",
    r"\bchapter \d+\b",
    r"\bislp\b",
    r"\bprml\b",
    r"\bfrom (this|the) (book|index|chapter|syllabus)\b",
    r"\bwhich of these\b",
    r"\bfollowing topics\b",
]

COMPARISON_SIGNALS = [
    r"\bcompare\b",
    r"\bdifference between\b",
    r"\b(bcg|deshaw|qrt|jpmc|swiggy|png|kenvue|mastercard|piramal) (vs|versus|and)\b",
    r"\bboth companies\b",
    r"\bacross companies\b",
]


def detect_intent(user_message: str) -> str:
    """
    Returns: "book_index" | "comparison" | "freeform"

    For book_index queries, also extracts the topic list if present.
    """
    msg = user_message.lower()

    if any(re.search(p, msg) for p in BOOK_INDEX_SIGNALS):
        return "book_index"

    if any(re.search(p, msg) for p in COMPARISON_SIGNALS):
        return "comparison"

    return "freeform"


def extract_topic_list(user_message: str) -> list[str]:
    """
    When intent is "book_index", try to extract the list of topics.

    Handles common formats:
      - "Topic A, Topic B, Topic C"
      - "1. Topic A\n2. Topic B"
      - "- Topic A\n- Topic B"
    """
    # Strip the preamble (everything before the first topic indicator)
    # Look for content after a colon or newline following trigger phrases
    text = user_message

    # Remove numbered list markers (1. / 1) )
    text = re.sub(r"^\s*\d+[\.\)]\s*", "", text, flags=re.MULTILINE)
    # Remove bullet markers
    text = re.sub(r"^\s*[-•●*]\s*", "", text, flags=re.MULTILINE)

    # Split by commas or newlines
    candidates = re.split(r"[,\n]+", text)

    topics = []
    for c in candidates:
        c = c.strip().strip('"').strip("'")
        # Keep if it looks like a topic (2–60 chars, not a full sentence)
        if 2 < len(c) < 60 and len(c.split()) <= 6:
            topics.append(c)

    return topics


def extract_company_filter(user_message: str) -> str | None:
    """
    If the query mentions a specific company, return it for use as a metadata filter.
    """
    from config import COMPANIES
    msg = user_message.upper()
    for company in COMPANIES:
        # Match whole word
        if re.search(r"\b" + re.escape(company) + r"\b", msg):
            return company
    return None


def extract_batch_filter(user_message: str) -> int | None:
    """
    If the query mentions a batch number, return it.
    e.g. "batch 10", "batch10", "2023 batch"
    """
    m = re.search(r"batch\s*(\d+)", user_message, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None
