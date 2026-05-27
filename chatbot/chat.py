"""
chatbot/chat.py
---------------
Multi-turn conversation loop. Ties together:
  intent detection → retrieval → aggregation → LLM synthesis

RAG concept learned here:
  - Chat history is just a list you prepend to every LLM call.
  - There is no magic "memory" — you pass the history explicitly each time.
  - Context window limits mean you can't keep history forever; we trim to last 4 turns.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ANTHROPIC_API_KEY, LLM_MODEL, LLM_MAX_TOKENS
from rag.retrieve import get_collection, get_embedding_model, retrieve, retrieve_for_topic
from rag.aggregate import summarise_freeform, summarise_book_index
from rag.intent import detect_intent, extract_topic_list, extract_company_filter, extract_batch_filter
from chatbot.prompts import (
    SYSTEM_PROMPT, freeform_prompt, book_index_prompt,
    comparison_prompt, format_history
)


def call_llm(prompt: str) -> str:
    import urllib.request, json
    payload = {
        "model":  LLM_MODEL,          # "llama3.2" from config
        "prompt": SYSTEM_PROMPT + "\n\n" + prompt,
        "stream": False
    }
    data = json.dumps(payload).encode()
    req  = urllib.request.urlopen("http://localhost:11434/api/generate", data)
    return json.loads(req.read())["response"]


def handle_freeform(user_input: str, model, collection, history: list) -> str:
    company = extract_company_filter(user_input)
    batch   = extract_batch_filter(user_input)

    chunks  = retrieve(user_input, model, collection, company=company, batch=batch)

    if not chunks:
        return "No relevant interview experiences found for that query. Try broadening the search."

    context = summarise_freeform(chunks)
    prompt  = freeform_prompt(user_input, context, format_history(history))
    return call_llm(prompt)


def handle_book_index(user_input: str, model, collection, history: list) -> str:
    topics  = extract_topic_list(user_input)
    company = extract_company_filter(user_input)

    if not topics:
        return ("Could not extract a topic list from your message. "
                "Try formatting as: 'Here are ISLP Ch.3 topics: Ridge, Lasso, PCR, PLS'")

    # Search each topic individually — this is the loop-per-topic pattern
    results_by_topic: dict[str, list] = {}
    for topic in topics:
        hits = retrieve_for_topic(topic, model, collection, company=company, top_k=8)
        results_by_topic[topic] = hits

    context = summarise_book_index(results_by_topic)
    prompt  = book_index_prompt(topics, context, format_history(history))
    return call_llm(prompt)


def handle_comparison(user_input: str, model, collection, history: list) -> str:
    chunks  = retrieve(user_input, model, collection, top_k=20)

    if not chunks:
        return "No relevant interview experiences found for that comparison."

    context = summarise_freeform(chunks)
    prompt  = comparison_prompt(user_input, context, format_history(history))
    return call_llm(prompt)


# ── Dispatch table ────────────────────────────────────────────────────────────

HANDLERS = {
    "freeform":   handle_freeform,
    "book_index": handle_book_index,
    "comparison": handle_comparison,
}


# ── Main chat loop ─────────────────────────────────────────────────────────────

def run_chat():
    try:
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.panel import Panel
        console = Console()
        USE_RICH = True
    except ImportError:
        USE_RICH = False

    def print_response(text: str):
        if USE_RICH:
            console.print(Panel(Markdown(text), title="[bold green]Bot[/bold green]",
                                border_style="green"))
        else:
            print(f"\nBot:\n{text}\n")

    def print_info(text: str):
        if USE_RICH:
            console.print(f"[dim]{text}[/dim]")
        else:
            print(text)

    # Load model and collection once at startup
    print_info("Loading embedding model...")
    model      = get_embedding_model()
    collection = get_collection()
    print_info(f"Ready. {collection.count()} interview chunks loaded.\n")

    if USE_RICH:
        console.print(Panel(
            "[bold]Interview Prep RAG Chatbot[/bold]\n\n"
            "Ask anything about past interview experiences.\n"
            "Paste a book index topic list to cross-reference.\n"
            "Type [bold red]quit[/bold red] to exit.",
            border_style="blue"
        ))
    else:
        print("=" * 55)
        print("  Interview Prep RAG Chatbot")
        print("  Type 'quit' to exit")
        print("=" * 55)

    history: list[dict] = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        intent  = detect_intent(user_input)
        print_info(f"[intent: {intent}]")

        handler  = HANDLERS[intent]
        response = handler(user_input, model, collection, history)

        print_response(response)

        # Store turn in history
        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": response})
