
import fitz
import re
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, TOPIC_MAP, COMPANIES, COMPANY_ALIASES, extract_batch_from_filename


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Content signals (format-independent)
# ═══════════════════════════════════════════════════════════════════════════════

# Round markers — every format uses some variant of these
ROUND_RE = re.compile(
    r"^(round[\s\-_]*\d+|r\d+[\s:\-]|written\s+test|test\s+round|online\s+test|"
    r"technical\s+round|hr\s+round|ppt\s+round|dcl|pre[\s\-]*interview|"
    r"final\s+round|first\s+round|second\s+round|third\s+round|"
    r"presentation\s+round|aptitude|coding\s+round|case\s+round|"
    r"group\s+discussion|gd\s+round|managerial\s+round|partner\s+round)",
    re.IGNORECASE
)

# Question lines — numbered, bulleted, or starting with Q:
QUESTION_RE = re.compile(
    r"^(\d+[\.\)]\s+|[\-\•\●\*\►]\s+|q\d*[\.\):\s]+|[a-z][\.\)]\s+)",
    re.IGNORECASE
)

# Lines to always skip — page numbers, decorators, whitespace artefacts
NOISE_RE = re.compile(
    r"^(\d{1,3}$|page\s+\d+|---+|===+|___+|\.\.\.|©|interview experience|"
    r"pgdba|batch\s+\d+|alumni|compiled by|index$|table of contents)",
    re.IGNORECASE
)


def normalise_company(raw: str) -> str | None:
    """
    Try to match a string to a known company.
    Returns canonical name or None if no match.
    """
    cleaned = raw.strip().upper()
    cleaned = re.sub(r"[^A-Z0-9&\s]", " ", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Direct alias
    if cleaned in COMPANY_ALIASES:
        return COMPANY_ALIASES[cleaned]

    # Exact match
    if cleaned in COMPANIES:
        return cleaned

    # Substring match — company name appears within the line
    for company in COMPANIES:
        if company in cleaned:
            return company
        # also try alias values
        for alias, canonical in COMPANY_ALIASES.items():
            if alias in cleaned:
                return canonical

    return None


def detect_topics(text: str) -> list[str]:
    text_lower = text.lower()
    return sorted(
        tag for tag, keywords in TOPIC_MAP.items()
        if any(kw in text_lower for kw in keywords)
    )


def build_embed_text(company, batch, topics, rounds, full_text):
    topic_str    = ", ".join(topics) if topics else "general"
    round_summary = "; ".join(
        f"{r['round_name']} [{', '.join(r['topics'])}]" for r in rounds
    )
    question_lines = [
        ln.strip() for ln in full_text.split("\n")
        if QUESTION_RE.match(ln.strip()) and len(ln.strip()) > 15
    ]
    q_preview = " | ".join(question_lines[:8])
    return (
        f"Company: {company}. Batch: {batch}. "
        f"Topics: {topic_str}. Rounds: {round_summary}. "
        f"Questions: {q_preview}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Page-level text extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_pages(pdf_path: Path) -> list[str]:
    """Extract text from every page. Returns list of page strings."""
    doc  = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        text = page.get_text().strip()
        pages.append(text)
    return pages


def is_likely_scanned(pages: list[str]) -> bool:
    """If >60% of pages are empty, the PDF is likely scanned/image-based."""
    empty = sum(1 for p in pages if len(p.strip()) < 20)
    return empty / max(len(pages), 1) > 0.6


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Company signal detection (format-agnostic)
# ═══════════════════════════════════════════════════════════════════════════════

def find_company_in_line(line: str) -> str | None:
    """
    Check if a line is primarily a company name.
    Works whether it's a dedicated header page OR inline mention.
    """
    line = line.strip()
    if not line or NOISE_RE.match(line):
        return None

    # A dedicated company-header line is typically short (< 50 chars)
    # and matches a known company
    if len(line) <= 60:
        match = normalise_company(line)
        if match:
            return match

    return None


def find_company_in_text(text: str) -> str | None:
    """
    Scan all lines in a text block for a company name signal.
    Returns first match found.
    """
    for line in text.split("\n"):
        company = find_company_in_line(line)
        if company:
            return company
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Three parsing strategies (tried in order)
# ═══════════════════════════════════════════════════════════════════════════════

def strategy_header_pages(pages: list[str], batch: int, source: str) -> list[dict]:
    """
    Strategy A: Company on its own page (Batch 9, 10 format).
    A page with ONLY a company name → marks start of that company's section.
    """
    chunks        = []
    cur_company   = None
    cur_lines     = []

    def flush(company, lines):
        text = "\n".join(lines).strip()
        if company and len(text) > 80:
            return make_chunks_from_block(text, company, batch, source)
        return []

    for page_text in pages:
        if not page_text.strip():
            continue

        page_lines    = [l for l in page_text.split("\n") if l.strip()]
        non_noise     = [l for l in page_lines if not NOISE_RE.match(l.strip())]

        # Is this page purely a company header?
        if len(non_noise) <= 3:
            possible = " ".join(non_noise)
            company  = normalise_company(possible)
            if company:
                chunks.extend(flush(cur_company, cur_lines))
                cur_company = company
                cur_lines   = []
                continue

        cur_lines.extend(page_text.split("\n"))

    chunks.extend(flush(cur_company, cur_lines))
    return chunks


def strategy_inline_headers(pages: list[str], batch: int, source: str) -> list[dict]:
    """
    Strategy B: Company name appears as a heading line within normal pages
    (many older batch formats). Splits on any line that matches a company name.
    """
    full_text   = "\n".join(pages)
    lines       = full_text.split("\n")

    chunks      = []
    cur_company = None
    cur_lines   = []

    def flush(company, lines):
        text = "\n".join(lines).strip()
        if company and len(text) > 80:
            return make_chunks_from_block(text, company, batch, source)
        return []

    for line in lines:
        company = find_company_in_line(line)
        if company:
            chunks.extend(flush(cur_company, cur_lines))
            cur_company = company
            cur_lines   = []
        else:
            cur_lines.append(line)

    chunks.extend(flush(cur_company, cur_lines))
    return chunks


def strategy_sliding_window(pages: list[str], batch: int, source: str) -> list[dict]:
    """
    Strategy C: Last resort — no clear company headers anywhere.
    Slide a window over all text, assign company based on nearest mention.
    Groups content by detected company, produces one chunk per company.
    """
    full_text = "\n".join(pages)
    lines     = full_text.split("\n")

    # Find all positions where a company is mentioned
    company_positions = []
    for i, line in enumerate(lines):
        company = find_company_in_line(line)
        if company:
            company_positions.append((i, company))

    if not company_positions:
        # No companies found at all — make one big chunk with whatever we have
        text   = full_text.strip()
        topics = detect_topics(text)
        if len(text) > 80:
            return [make_single_chunk("UNKNOWN", batch, source, text, 0)]
        return []

    # Assign each line to the nearest preceding company mention
    company_blocks = defaultdict(list)
    pos_idx = 0
    for i, line in enumerate(lines):
        while (pos_idx + 1 < len(company_positions)
               and company_positions[pos_idx + 1][0] <= i):
            pos_idx += 1
        company = company_positions[pos_idx][1]
        company_blocks[company].append(line)

    chunks = []
    for company, block_lines in company_blocks.items():
        text = "\n".join(block_lines).strip()
        if len(text) > 80:
            chunks.extend(make_chunks_from_block(text, company, batch, source))

    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Chunk builder (shared by all strategies)
# ═══════════════════════════════════════════════════════════════════════════════

def make_chunks_from_block(text: str, company: str, batch: int, source: str) -> list[dict]:
    """
    Given a block of text belonging to one company, split it into
    experience-level chunks by detecting entry boundaries.

    Tries three sub-strategies in order:
      1. Split on round markers (most reliable)
      2. Split on paragraph breaks
      3. Keep as single chunk
    """
    # First try: split into individual round segments, group consecutive
    # rounds into one experience chunk
    experiences = split_into_experiences(text)

    if not experiences:
        experiences = [text]

    chunks = []
    for idx, exp_text in enumerate(experiences):
        exp_text = exp_text.strip()
        if len(exp_text) < 60:
            continue
        chunk = make_single_chunk(company, batch, source, exp_text, idx)
        chunks.append(chunk)

    return chunks


def split_into_experiences(text: str) -> list[str]:
    """
    Split a company's text block into individual interview experiences.

    Heuristic: a new experience starts when we see a round marker
    AFTER some non-round content (i.e. after an experience has already begun).
    We also split on large blank gaps (3+ newlines).
    """
    # Split on 3+ consecutive newlines first (strong paragraph break)
    big_breaks = re.split(r"\n{3,}", text)
    if len(big_breaks) > 1:
        # Filter out very short fragments
        return [b.strip() for b in big_breaks if len(b.strip()) > 60]

    # Otherwise split on round markers
    lines   = text.split("\n")
    entries = []
    current = []
    seen_content = False  # have we seen non-round content yet?

    for line in lines:
        is_round = bool(ROUND_RE.match(line.strip()))

        if is_round and seen_content and current:
            # New round after content → could be new experience
            # But only split if current block is substantial
            if len("\n".join(current)) > 150:
                entries.append("\n".join(current))
                current = [line]
                seen_content = False
                continue

        if not is_round and not NOISE_RE.match(line.strip()) and len(line.strip()) > 10:
            seen_content = True

        current.append(line)

    if current:
        entries.append("\n".join(current))

    return [e.strip() for e in entries if len(e.strip()) > 60] or [text]


def make_single_chunk(company: str, batch: int, source: str,
                      text: str, idx: int) -> dict:
    topics = detect_topics(text)
    rounds = extract_rounds_from_text(text)
    import hashlib
    text_hash = hashlib.md5(text.encode()).hexdigest()[:6]
    return {
        "chunk_id":   f"{company.lower().replace(' ','_')}_b{batch}_{idx}_{text_hash}",
        "company":    company,
        "batch":      batch,
        "source":     source,
        "topics":     topics,
        "rounds":     rounds,
        "num_rounds": len(rounds),
        "embed_text": build_embed_text(company, batch, topics, rounds, text),
        "full_text":  text,
    }


def extract_rounds_from_text(text: str) -> list[dict]:
    """Extract round-level blocks from any text format."""
    ROUND_SPLIT = re.compile(
        r"((?:Round[\s\-]+\d+|R\d+[\s:\-]|Written\s+[Tt]est|Test\s+[Rr]ound|"
        r"Online\s+[Tt]est|PPT\s+[Rr]ound|DCL|Pre[\s\-]*[Ii]nterview|"
        r"Final\s+[Rr]ound|Second\s+[Rr]ound|Third\s+[Rr]ound|First\s+[Rr]ound|"
        r"Technical\s+[Rr]ound|HR\s+[Rr]ound|Coding\s+[Rr]ound|"
        r"Presentation\s+[Rr]ound)[\s\S]*?)"
        r"(?=(?:Round[\s\-]+\d+|R\d+[\s:\-]|Written\s+[Tt]est|Test\s+[Rr]ound|"
        r"Online\s+[Tt]est|PPT\s+[Rr]ound|DCL|Pre[\s\-]*[Ii]nterview|"
        r"Final\s+[Rr]ound|Second\s+[Rr]ound|Third\s+[Rr]ound|First\s+[Rr]ound|"
        r"Technical\s+[Rr]ound|HR\s+[Rr]ound|Coding\s+[Rr]ound|"
        r"Presentation\s+[Rr]ound)|\Z)",
        re.IGNORECASE
    )
    rounds = []
    for m in ROUND_SPLIT.finditer(text):
        content = m.group(0).strip()
        if len(content) > 20:
            rounds.append({
                "round_name": content.split("\n")[0].strip()[:80],
                "content":    content,
                "topics":     detect_topics(content),
            })
    return rounds or [{"round_name": "general", "content": text,
                       "topics": detect_topics(text)}]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Smart dispatcher: pick best strategy per PDF
# ═══════════════════════════════════════════════════════════════════════════════

def parse_pdf(pdf_path: Path, batch: int) -> list[dict]:
    """
    Parse one PDF using the best-fit strategy, auto-detected from content.
    Falls back through strategies until chunks are produced.
    """
    pages  = extract_pages(pdf_path)
    source = pdf_path.name

    if is_likely_scanned(pages):
        print(f"  ⚠  Looks like a scanned PDF — text extraction may be limited")

    # Count how many pages look like pure company headers
    header_page_count = 0
    for page_text in pages:
        non_noise = [l for l in page_text.split("\n")
                     if l.strip() and not NOISE_RE.match(l.strip())]
        if 0 < len(non_noise) <= 3 and normalise_company(" ".join(non_noise)):
            header_page_count += 1

    total_pages = len([p for p in pages if p.strip()])

    # Strategy selection logic
    if total_pages > 0 and header_page_count / total_pages > 0.05:
        # >5% of pages are company headers → Strategy A
        strategy_name = "header_pages"
        chunks = strategy_header_pages(pages, batch, source)
    else:
        # Try inline headers first
        strategy_name = "inline_headers"
        chunks = strategy_inline_headers(pages, batch, source)

    # If still no chunks, fall back to sliding window
    if not chunks:
        strategy_name = "sliding_window"
        chunks = strategy_sliding_window(pages, batch, source)

    # If still nothing, one big chunk per detected company (last resort)
    if not chunks:
        strategy_name = "full_text_fallback"
        full_text = "\n".join(pages)
        topics    = detect_topics(full_text)
        if len(full_text) > 100:
            chunks = [make_single_chunk("UNKNOWN", batch, source, full_text, 0)]

    print(f"  strategy: {strategy_name}")
    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Parse all files
# ═══════════════════════════════════════════════════════════════════════════════

def parse_all() -> list[dict]:
    """
    Parse all PDFs and XLSX files in DATA_DIR.
    Batch number auto-extracted from filename.
    No configuration needed for new batches — just drop files in data/raw/.
    """
    from ingest.parse_xlsx import parse_xlsx

    all_chunks = []

    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        batch = extract_batch_from_filename(pdf_path.name)
        print(f"\n[PDF ] {pdf_path.name}  →  batch {batch}")
        chunks = parse_pdf(pdf_path, batch)
        companies = set(c["company"] for c in chunks)
        print(f"  {len(chunks)} chunks | {len(companies)} companies: {', '.join(sorted(companies)[:6])}{'...' if len(companies)>6 else ''}")
        all_chunks.extend(chunks)

    for xlsx_path in sorted(DATA_DIR.glob("*.xlsx")):
        batch = extract_batch_from_filename(xlsx_path.name)
        print(f"\n[XLSX] {xlsx_path.name}  →  batch {batch}")
        chunks = parse_xlsx(xlsx_path, batch)
        companies = set(c["company"] for c in chunks)
        print(f"  {len(chunks)} chunks | {len(companies)} companies")
        all_chunks.extend(chunks)

    if not all_chunks:
        print(f"\nNo files found in {DATA_DIR}")
    return all_chunks


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Run
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    chunks = parse_all()

    if not chunks:
        sys.exit(1)

    out_path = DATA_DIR / "parsed_chunks.json"
    with open(out_path, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"\n{'─'*55}")
    print(f"  Total chunks : {len(chunks)}")
    print(f"  Companies    : {len(set(c['company'] for c in chunks))}")
    print(f"  Saved to     : {out_path}")

    print("\nBatch breakdown:")
    for b, n in sorted(Counter(c["batch"] for c in chunks).items()):
        cos = set(c["company"] for c in chunks if c["batch"] == b)
        print(f"  Batch {b:<3} {n:>4} chunks | {len(cos)} companies")

    print("\nTop topics across all batches:")
    all_topics = [t for c in chunks for t in c["topics"]]
    for topic, n in Counter(all_topics).most_common(12):
        print(f"  {topic:<25} {n}")

