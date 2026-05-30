"""
chatbot/chat.py — dispatch layer
Routes each query to the right retrieval + prompt based on intent.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import GROQ_API_KEY, LLM_MODEL, LLM_MAX_TOKENS
from rag.retrieve import (get_collection, get_embedding_model, retrieve,
                           retrieve_for_topic, retrieve_multi_company)
from rag.aggregate import (format_concept_context, format_company_context,
                            format_comparison_context, format_book_index_context,
                            format_freeform_context)
from rag.intent import (detect_intent, extract_topic_list, extract_company_filter,
                         extract_mentioned_companies, extract_batch_filter)
from chatbot.prompts import (SYSTEM_PROMPT, concept_prompt, company_prompt,
                              comparison_prompt, batch_prompt, book_index_prompt,
                              freeform_prompt, format_history)


def call_llm(prompt: str) -> str:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )
    return response.choices[0].message.content


def handle_query(user_input: str, model, collection, history: list) -> tuple[str, str, list]:
    """Returns (response, intent, sources_list)."""

    intent    = detect_intent(user_input)
    company   = extract_company_filter(user_input)
    companies = extract_mentioned_companies(user_input)
    batch     = extract_batch_filter(user_input)

    # ── Concept query ─────────────────────────────────────────────────────────
    if intent == "concept":
        chunks  = retrieve(user_input, model, collection, top_k=30, recency_weight=True)
        if not chunks:
            return _no_data_response(), intent, []
        context = format_concept_context(chunks)
        prompt  = concept_prompt(user_input, context, format_history(history))
        sources = list({f"{c['company']} Batch {c['batch']}" for c in chunks})

    # ── Company query ─────────────────────────────────────────────────────────
    elif intent == "company" and company:
        chunks  = retrieve(user_input, model, collection, company=company,
                           top_k=25, recency_weight=True)
        if not chunks:
            return _no_data_response(company), intent, []
        context = format_company_context(chunks, company)
        prompt  = company_prompt(user_input, context, company, format_history(history))
        sources = [f"Batch {c['batch']}" for c in chunks]

    # ── Comparison query ──────────────────────────────────────────────────────
    elif intent == "comparison":
        target_companies = companies if companies else []
        if not target_companies:
            chunks  = retrieve(user_input, model, collection, top_k=30)
            context = format_freeform_context(chunks)
            prompt  = freeform_prompt(user_input, context, format_history(history))
            sources = list({f"{c['company']} Batch {c['batch']}" for c in chunks})
        else:
            results = retrieve_multi_company(user_input, model, collection,
                                             target_companies, top_k_per_company=10)
            if not results:
                return _no_data_response(), intent, []
            context = format_comparison_context(results)
            prompt  = comparison_prompt(user_input, context, target_companies,
                                        format_history(history))
            sources = [f"{co} Batch {c['batch']}"
                       for co, chunks in results.items() for c in chunks]

    # ── Batch query ───────────────────────────────────────────────────────────
    elif intent == "batch" and batch:
        chunks  = retrieve(user_input, model, collection, batch=batch,
                           top_k=25, recency_weight=False)
        if not chunks:
            return _no_data_response(f"Batch {batch}"), intent, []
        context = format_freeform_context(chunks)
        prompt  = batch_prompt(user_input, context, batch, format_history(history))
        sources = [f"{c['company']}" for c in chunks]

    # ── Book index query ──────────────────────────────────────────────────────
    elif intent == "book_index":
        topics = extract_topic_list(user_input)
        if not topics:
            return ("Could not extract topic list. Format as:\n"
                    "'Here are ISLP Ch.3 topics: Ridge, Lasso, PCR, PLS'",
                    intent, [])
        results_by_topic = {
            t: retrieve_for_topic(t, model, collection, company=company, top_k=10)
            for t in topics
        }
        context = format_book_index_context(results_by_topic)
        prompt  = book_index_prompt(topics, context, format_history(history))
        sources = list({
            f"{c['company']} Batch {c['batch']}"
            for chunks in results_by_topic.values() for c in chunks
        })

    # ── Freeform fallback ─────────────────────────────────────────────────────
    else:
        chunks  = retrieve(user_input, model, collection,
                           company=company, batch=batch,
                           top_k=20, recency_weight=True)
        if not chunks:
            return _no_data_response(), intent, []
        context = format_freeform_context(chunks)
        prompt  = freeform_prompt(user_input, context, format_history(history))
        sources = [f"{c['company']} Batch {c['batch']}" for c in chunks[:6]]

    return call_llm(prompt), intent, sorted(set(sources))


def _no_data_response(entity: str = "") -> str:
    msg = f"for {entity} " if entity else ""
    return (f"Not enough data in the retrieved interviews {msg}to answer this confidently. "
            "Try rephrasing, or check that the relevant batch PDFs have been ingested.")
