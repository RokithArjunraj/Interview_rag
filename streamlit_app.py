"""
streamlit_app.py — Streamlit UI
Run: streamlit run streamlit_app.py
"""

import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from chatbot.chat import handle_query, call_llm
from rag.retrieve import get_collection, get_embedding_model

st.set_page_config(page_title="Interview Prep RAG", page_icon="🎯", layout="wide")

st.markdown("""
<style>
.badge { display:inline-block; padding:2px 10px; border-radius:12px;
         font-size:12px; font-weight:600; margin-bottom:8px; }
.b-concept    { background:#e8f4fd; color:#1a73e8; }
.b-company    { background:#fce8fd; color:#7b1ae8; }
.b-comparison { background:#fef3e2; color:#e67e00; }
.b-batch      { background:#e8fdf0; color:#1ae85a; }
.b-book_index { background:#e8f8f0; color:#1e7e34; }
.b-freeform   { background:#f0f0f0; color:#555; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_resources():
    model      = get_embedding_model()
    collection = get_collection()
    return model, collection


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎯 Interview Prep RAG")
    st.caption("PGDBA Batch Interview Experiences")
    st.divider()

    try:
        model, collection = load_resources()
        st.success(f"✅ {collection.count()} chunks loaded")
    except Exception as e:
        st.error(f"❌ ChromaDB error: {e}")
        st.stop()

    st.divider()
    st.markdown("**Query modes:**")
    st.markdown("🔵 **Concept** — topic across all companies")
    st.markdown("🟣 **Company** — one company's pattern")
    st.markdown("🟠 **Comparison** — compare companies")
    st.markdown("🟢 **Book index** — paste topic list")
    st.markdown("⚪ **Batch** — specific batch")

    st.divider()
    st.markdown("**Try these:**")
    examples = [
        "What ML topics were asked across all companies?",
        "What does BCG focus on in interviews?",
        "Compare DE Shaw and QRT",
        "Here are ISLP Ch.6 topics: Ridge, Lasso, PCR — which were asked?",
        "What time series concepts are most frequently asked?",
        "What changed in recent batches at JPMC?",
        "Which companies ask about RAG and LLMs?",
        "What Python/DSA questions came up at DE Shaw?",
    ]
    for q in examples:
        if st.button(q, use_container_width=True, key=q):
            st.session_state.prefill = q

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history  = []
        st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("Interview Prep Chatbot")
st.caption("Powered by past PGDBA batch interview experiences · Recency-weighted · No hallucination")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "prefill" not in st.session_state:
    st.session_state.prefill = ""

BADGE_CLASS = {
    "concept": "b-concept", "company": "b-company",
    "comparison": "b-comparison", "batch": "b-batch",
    "book_index": "b-book_index", "freeform": "b-freeform",
}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            intent = msg.get("intent", "freeform")
            st.markdown(
                f'<span class="badge {BADGE_CLASS.get(intent,"b-freeform")}">{intent}</span>',
                unsafe_allow_html=True
            )
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 {len(msg['sources'])} sources", expanded=False):
                for s in msg["sources"]:
                    st.markdown(f"• {s}")

user_input = st.chat_input("Ask about interview experiences...")

if st.session_state.prefill and not user_input:
    user_input = st.session_state.prefill
    st.session_state.prefill = ""

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        badge_slot = st.empty()
        with st.spinner("Searching interview data..."):
            response, intent, sources = handle_query(
                user_input, model, collection, st.session_state.history
            )
        badge_slot.markdown(
            f'<span class="badge {BADGE_CLASS.get(intent,"b-freeform")}">{intent}</span>',
            unsafe_allow_html=True
        )
        st.markdown(response)
        if sources:
            with st.expander(f"📚 {len(sources)} sources", expanded=False):
                for s in sources:
                    st.markdown(f"• {s}")

    st.session_state.messages.append({
        "role": "assistant", "content": response,
        "intent": intent, "sources": sources,
    })
    st.session_state.history.append({"role": "user",      "content": user_input})
    st.session_state.history.append({"role": "assistant", "content": response})
