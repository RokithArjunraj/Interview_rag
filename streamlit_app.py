"""
streamlit_app.py
----------------
Streamlit UI for the Interview Prep RAG Chatbot.

Run:  streamlit run streamlit_app.py

Share with friends:
  - Local network : streamlit run streamlit_app.py --server.address 0.0.0.0
  - Public link   : use 'streamlit run streamlit_app.py' then enable tunnel
                    via ngrok or Streamlit Community Cloud
"""

import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from config import LLM_MODEL
from rag.retrieve import get_collection, get_embedding_model, retrieve, retrieve_for_topic
from rag.aggregate import summarise_freeform, summarise_book_index
from rag.intent import detect_intent, extract_topic_list, extract_company_filter, extract_batch_filter
from chatbot.prompts import (
    SYSTEM_PROMPT, freeform_prompt, book_index_prompt,
    comparison_prompt, format_history
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Interview Prep RAG",
    page_icon="🎯",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .intent-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .badge-freeform   { background: #e8f4fd; color: #1a73e8; }
    .badge-book_index { background: #e8f8f0; color: #1e7e34; }
    .badge-comparison { background: #fef3e2; color: #e67e00; }
    .source-box {
        background: #f8f9fa;
        border-left: 3px solid #6c757d;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 13px;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ── Load model + collection (cached — runs only once) ─────────────────────────
@st.cache_resource
def load_resources():
    model      = get_embedding_model()
    collection = get_collection()
    return model, collection


# ── LLM call ─────────────────────────────────────────────────────────────────
def call_llm(prompt: str) -> str:
    from groq import Groq
    from config import GROQ_API_KEY, LLM_MODEL, LLM_MAX_TOKENS
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

# ── Query handlers ────────────────────────────────────────────────────────────
def handle_query(user_input: str, model, collection, history: list) -> tuple[str, str, list]:
    """Returns (response, intent, source_chunks_summary)."""

    intent  = detect_intent(user_input)
    company = extract_company_filter(user_input)
    batch   = extract_batch_filter(user_input)

    if intent == "book_index":
        topics = extract_topic_list(user_input)
        if not topics:
            return ("Could not extract topics. Format as:\n"
                    "'Here are ISLP Ch.3 topics: Ridge, Lasso, PCR'",
                    intent, [])
        results_by_topic = {
            t: retrieve_for_topic(t, model, collection, company=company, top_k=8)
            for t in topics
        }
        context  = summarise_book_index(results_by_topic)
        prompt   = book_index_prompt(topics, context, format_history(history))
        # Build source summary for display
        sources = []
        for topic, chunks in results_by_topic.items():
            if chunks:
                cos = list({c["company"] for c in chunks})
                sources.append(f"**{topic}** → {', '.join(cos)}")

    elif intent == "comparison":
        chunks   = retrieve(user_input, model, collection, top_k=20)
        context  = summarise_freeform(chunks)
        prompt   = comparison_prompt(user_input, context, format_history(history))
        sources  = list({c["company"] for c in chunks})

    else:  # freeform
        chunks   = retrieve(user_input, model, collection, company=company, batch=batch)
        context  = summarise_freeform(chunks)
        prompt   = freeform_prompt(user_input, context, format_history(history))
        sources  = [f"**{c['company']}** (batch {c['batch']}, sim {c['similarity']})"
                    for c in chunks[:5]]

    response = call_llm(prompt)
    return response, intent, sources


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎯 Interview Prep RAG")
    st.caption("Built with ChromaDB + Llama3.2 + Streamlit")

    st.divider()

    try:
        model, collection = load_resources()
        chunk_count = collection.count()
        st.success(f"✅ {chunk_count} interview chunks loaded")
    except Exception as e:
        st.error(f"❌ Could not load ChromaDB: {e}")
        st.stop()

    st.divider()
    st.markdown("**Query modes detected automatically:**")
    st.markdown("🔵 **Freeform** — plain questions")
    st.markdown("🟢 **Book index** — paste topic lists")
    st.markdown("🟠 **Comparison** — compare companies")

    st.divider()
    st.markdown("**Example queries:**")
    example_queries = [
        "What topics were asked at BCG?",
        "Most frequent ML topics across all companies?",
        "Here are ISLP Ch.6 topics: Ridge, Lasso, PCR, PLS — which were asked?",
        "Compare DE Shaw and QRT interviews",
        "What Python/coding questions came up at JPMC?",
        "What time series concepts were asked and where?",
    ]
    for q in example_queries:
        if st.button(q, use_container_width=True, key=q):
            st.session_state.prefill = q

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history  = []
        st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────
st.title("Interview Prep Chatbot")
st.caption("Ask anything about past interview experiences from the PGDBA batch docs.")

# Initialise session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "prefill" not in st.session_state:
    st.session_state.prefill = ""

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            badge_class = f"badge-{msg.get('intent', 'freeform')}"
            st.markdown(
                f'<span class="intent-badge {badge_class}">{msg.get("intent","freeform")}</span>',
                unsafe_allow_html=True
            )
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources used", expanded=False):
                for s in msg["sources"]:
                    st.markdown(s)

# Chat input — prefill from sidebar button if set
user_input = st.chat_input("Ask about interview experiences, or paste a topic list...")

# Handle sidebar example button prefill
if st.session_state.prefill and not user_input:
    user_input = st.session_state.prefill
    st.session_state.prefill = ""

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate response
    with st.chat_message("assistant"):
        badge_placeholder = st.empty()
        with st.spinner("Searching interview data..."):
            response, intent, sources = handle_query(
                user_input, model, collection, st.session_state.history
            )

        badge_class = f"badge-{intent}"
        badge_placeholder.markdown(
            f'<span class="intent-badge {badge_class}">{intent}</span>',
            unsafe_allow_html=True
        )
        st.markdown(response)
        if sources:
            with st.expander("📚 Sources used", expanded=False):
                for s in sources:
                    st.markdown(s)

    # Store in session
    st.session_state.messages.append({
        "role":    "assistant",
        "content": response,
        "intent":  intent,
        "sources": sources,
    })
    st.session_state.history.append({"role": "user",      "content": user_input})
    st.session_state.history.append({"role": "assistant", "content": response})
