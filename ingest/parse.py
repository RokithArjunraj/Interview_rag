"""
ingest/parse.py
---------------
Parses interview experience PDFs into structured chunks.

Each chunk represents ONE person's experience at ONE company.
We deliberately omit person names — the unit of analysis is
the (company, batch, topics, rounds) combination.

Output: data/raw/parsed_chunks.json

Run:  python -m ingest.parse
"""

import fitz  # PyMuPDF
import re
import json
import sys
from pathlib import Path
from collections import Counter

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, TOPIC_MAP, COMPANIES


# ── Helpers ──────────────────────────────────────────────────────────────────

def is_company_header(text: str) -> str | None:
    """Return company name if this page is a section divider, else None."""
    t = text.strip().upper()
    for company in COMPANIES:
        if t == company:
            return company
    # Short page that contains a company name
    if len(t) < 40:
        for company in COMPANIES:
            if company in t:
                return company
    return None


SKIP_NAME_PATTERNS = re.compile(
    r"^(round|test|pre.interview|presentation|questions|mainly|group|"
    r"question\s*\d|ml concepts|r\d+[:\s]|ann.related|hr questions|"
    r"conflict|kolmogorov|situation|note:|tips:|suggestion)",
    re.IGNORECASE
)

def looks_like_entry_start(line: str) -> bool:
    """
    Heuristic: a new person's entry starts with a short, title-cased line.
    We use this to split company text into individual entries.
    """
    line = line.strip()
    if not line or len(line) > 55:
        return False
    if SKIP_NAME_PATTERNS.match(line):
        return False
    words = line.split()
    if len(words) < 1 or len(words) > 5:
        return False
    # At least one word starting with uppercase
    return any(w[0].isupper() for w in words if w)


def detect_topics(text: str) -> list[str]:
    """Return sorted list of topic tags found in text."""
    text_lower = text.lower()
    return sorted(
        tag for tag, keywords in TOPIC_MAP.items()
        if any(kw in text_lower for kw in keywords)
    )


ROUND_PATTERN = re.compile(
    r"((?:Round[\s\-]+\d+|Written\s+[Tt]est|Test\s+[Rr]ound|Online\s+[Tt]est|"
    r"PPT\s+[Rr]ound|DCL|Pre[\-\s]+interview|Final\s+[Rr]ound|"
    r"Second\s+[Rr]ound|Third\s+[Rr]ound|First\s+[Rr]ound|"
    r"R\d+[\s:]|Presentation\s+[Rr]ound)[\s\S]*?)"
    r"(?=(?:Round[\s\-]+\d+|Written\s+[Tt]est|Test\s+[Rr]ound|Online\s+[Tt]est|"
    r"PPT\s+[Rr]ound|DCL|Pre[\-\s]+interview|Final\s+[Rr]ound|"
    r"Second\s+[Rr]ound|Third\s+[Rr]ound|First\s+[Rr]ound|"
    r"R\d+[\s:]|Presentation\s+[Rr]ound)|\Z)",
    re.IGNORECASE
)

def extract_rounds(text: str) -> list[dict]:
    """Split entry text into round-level dicts."""
    rounds = []
    for m in ROUND_PATTERN.finditer(text):
        content = m.group(0).strip()
        if len(content) > 20:
            rounds.append({
                "round_name": content.split("\n")[0].strip(),
                "content": content
            })
    return rounds or [{"round_name": "general", "content": text}]


# ── Main parser ───────────────────────────────────────────────────────────────

def parse_pdf(pdf_path: Path, batch: int) -> list[dict]:
    """
    Parse one PDF → list of chunk dicts.
    Each chunk = one interview entry (one person at one company).
    """
    doc = fitz.open(str(pdf_path))

    # Step 1: assign pages to companies
    company_pages: dict[str, list[str]] = {}
    current_company = None

    for page in doc:
        text = page.get_text().strip()
        if not text:
            continue
        company = is_company_header(text)
        if company:
            current_company = company
            company_pages.setdefault(current_company, [])
        elif current_company:
            company_pages[current_company].append(text)

    # Step 2: split each company's pages into individual entries
    chunks = []
    for company, pages in company_pages.items():
        full_text = "\n".join(pages)
        lines = full_text.split("\n")

        # Split by entry boundaries
        entries: list[list[str]] = []
        current_entry: list[str] = []

        for line in lines:
            if looks_like_entry_start(line) and current_entry:
                entries.append(current_entry)
                current_entry = [line]
            else:
                current_entry.append(line)
        if current_entry:
            entries.append(current_entry)

        # Build a chunk per entry
        for idx, entry_lines in enumerate(entries):
            entry_text = "\n".join(entry_lines).strip()
            if len(entry_text) < 80:   # skip near-empty entries
                continue

            topics  = detect_topics(entry_text)
            rounds  = extract_rounds(entry_text)

            chunk = {
                "chunk_id":  f"{company.lower().replace(' ', '_')}_batch{batch}_{idx}",
                "company":   company,
                "batch":     batch,
                "source":    pdf_path.name,
                "topics":    topics,
                "rounds":    rounds,
                "num_rounds": len(rounds),
                "full_text": entry_text,
            }
            chunks.append(chunk)

    return chunks


def parse_all(batch_map: dict[str, int] | None = None) -> list[dict]:
    """
    Parse all PDFs in DATA_DIR.
    batch_map: {"filename.pdf": batch_number}  — defaults to batch 10 for all.
    """
    pdfs = list(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {DATA_DIR}. Add your interview docs there.")
        return []

    batch_map = batch_map or {}
    all_chunks = []

    for pdf_path in pdfs:
        batch = batch_map.get(pdf_path.name, 10)
        print(f"Parsing {pdf_path.name} (batch {batch})...")
        chunks = parse_pdf(pdf_path, batch)
        print(f"  → {len(chunks)} entries across "
              f"{len(set(c['company'] for c in chunks))} companies")
        all_chunks.extend(chunks)

    return all_chunks


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    chunks = parse_all()

    if not chunks:
        sys.exit(1)

    # Save
    out_path = DATA_DIR / "parsed_chunks.json"
    with open(out_path, "w") as f:
        json.dump(chunks, f, indent=2)

    # Summary
    print(f"\n{'─'*50}")
    print(f"  Total chunks  : {len(chunks)}")
    print(f"  Companies     : {len(set(c['company'] for c in chunks))}")
    print(f"  Saved to      : {out_path}")

    print("\nCompany breakdown:")
    for co, n in Counter(c["company"] for c in chunks).most_common():
        print(f"  {co:<22} {n} entries")

    print("\nTopic frequency:")
    all_topics = [t for c in chunks for t in c["topics"]]
    for topic, n in Counter(all_topics).most_common():
        print(f"  {topic:<25} {n}")
