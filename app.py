"""
app.py — entry point
--------------------
Run this to start the chatbot:  python app.py

Checks that ingestion has been done, then launches the chat loop.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, CHROMA_DIR, ANTHROPIC_API_KEY


def check_setup():
    errors = []

    if not ANTHROPIC_API_KEY:
        errors.append(
            "ANTHROPIC_API_KEY is not set.\n"
            "  Export it:  export ANTHROPIC_API_KEY=sk-ant-..."
        )

    parsed = DATA_DIR / "parsed_chunks.json"
    if not parsed.exists():
        errors.append(
            "parsed_chunks.json not found.\n"
            "  Run first:  python -m ingest.parse"
        )

    chroma = CHROMA_DIR / "chroma.sqlite3"
    if not chroma.exists():
        errors.append(
            "ChromaDB not found.\n"
            "  Run first:  python -m ingest.embed"
        )

    if errors:
        print("\n── Setup errors ──────────────────────────────")
        for e in errors:
            print(f"  ✗  {e}")
        print("──────────────────────────────────────────────\n")
        sys.exit(1)


if __name__ == "__main__":
    check_setup()
    from chatbot.chat import run_chat
    run_chat()
