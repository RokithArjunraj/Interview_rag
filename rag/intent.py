"""
rag/intent.py
-------------
Classifies user query into one of four modes:
  concept   → asking about a topic/concept across all companies
  company   → asking about a specific company's interview pattern
  batch     → asking about a specific batch
  comparison→ comparing two or more companies
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import COMPANIES


BOOK_INDEX_SIGNALS = [
    r"\bhere (is|are)\b", r"\bthese topics\b", r"\bthis list\b",
    r"\bchapter \d+\b", r"\bislp\b", r"\bprml\b",
    r"\bfrom (this|the) (book|index|chapter|syllabus)\b",
    r"\bwhich of these\b", r"\bfollowing topics\b",
]

COMPARISON_SIGNALS = [
    r"\bcompare\b", r"\bvs\b", r"\bversus\b",
    r"\bdifference between\b", r"\bboth companies\b",
    r"\bacross companies\b", r"\bsimilar\b.*\bcompan",
]

BATCH_SIGNALS = [
    r"\bbatch\s*\d+\b", r"\blast\s*(year|batch)\b",
    r"\brecent\s*batch\b",
]

CONCEPT_SIGNALS = [
    r"\bwhat (topics|concepts|questions)\b",
    r"\bwhich (topics|concepts)\b",
    r"\bhow (often|frequently)\b",
    r"\bfrequen",
    r"\bcommonly asked\b",
    r"\bacross (all|companies|batches)\b",
]


def detect_intent(user_message: str) -> str:
    msg = user_message.lower()

    if any(re.search(p, msg) for p in BOOK_INDEX_SIGNALS):
        return "book_index"

    # Comparison: mentions 2+ companies OR explicit compare words
    mentioned_companies = [c for c in COMPANIES if c.lower() in msg]
    if len(mentioned_companies) >= 2:
        return "comparison"
    if any(re.search(p, msg) for p in COMPARISON_SIGNALS):
        return "comparison"

    if any(re.search(p, msg) for p in BATCH_SIGNALS) and not mentioned_companies:
        return "batch"

    # Single company mentioned → company mode
    if len(mentioned_companies) == 1:
        return "company"

    # Concept/topic queries
    if any(re.search(p, msg) for p in CONCEPT_SIGNALS):
        return "concept"

    return "freeform"


def extract_topic_list(user_message: str) -> list[str]:
    text = re.sub(r"^\s*\d+[\.\)]\s*", "", user_message, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-•●*]\s*", "", text, flags=re.MULTILINE)
    candidates = re.split(r"[,\n]+", text)
    return [
        c.strip().strip('"').strip("'")
        for c in candidates
        if 2 < len(c.strip()) < 60 and len(c.strip().split()) <= 6
    ]


def extract_company_filter(user_message: str) -> str | None:
    msg = user_message.upper()
    for company in COMPANIES:
        if re.search(r"\b" + re.escape(company) + r"\b", msg):
            return company
    return None


def extract_mentioned_companies(user_message: str) -> list[str]:
    msg = user_message.upper()
    return [c for c in COMPANIES if re.search(r"\b" + re.escape(c) + r"\b", msg)]


def extract_batch_filter(user_message: str) -> int | None:
    m = re.search(r"batch\s*(\d+)", user_message, re.IGNORECASE)
    return int(m.group(1)) if m else None
