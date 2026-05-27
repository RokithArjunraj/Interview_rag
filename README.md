# Interview Prep RAG Chatbot

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about
past interview experiences. You can query it with plain questions, book index
topic lists (ISLP, PRML, etc.), or any reference material you bring at query time.

---

## Project Structure

```
interview_rag/
│
├── data/
│   └── raw/                  ← Drop your PDF interview docs here
│
├── ingest/
│   ├── parse.py              ← Parses PDF → structured chunks with metadata
│   └── embed.py              ← Embeds chunks → stores in ChromaDB
│
├── rag/
│   ├── retrieve.py           ← Semantic search + metadata filters
│   ├── aggregate.py          ← Groups results by company, counts topic frequency
│   └── intent.py             ← Detects query type (freeform vs book-index)
│
├── chatbot/
│   ├── chat.py               ← Multi-turn conversation loop
│   └── prompts.py            ← Prompt templates for each query mode
│
├── app.py                    ← Entry point — run this
├── config.py                 ← All settings in one place
└── requirements.txt
```

---

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key to config.py (or set as env variable)
export ANTHROPIC_API_KEY=your_key_here

# 4. Drop interview PDF(s) into data/raw/

# 5. Run ingestion (parse + embed) — do this once
python -m ingest.parse
python -m ingest.embed

# 6. Start chatbot
python app.py
```

---

## Example Queries

```
You: What topics were asked at DE Shaw?

You: Here are ISLP Chapter 3 topics — Linear Regression, Subset Selection,
     Ridge Regression, Lasso, PCR, PLS. Which were asked and in which companies?

You: What time series concepts came up most frequently?

You: Compare what BCG and Oliver Wyman asked — any overlap?

You: What Python/coding questions were asked across all companies?
```

---

## How RAG Works Here (Learning Notes)

1. **Ingestion (offline, once)**
   - PDF is parsed into chunks — one chunk per person per company
   - Each chunk gets metadata: company, batch, topics, round names
   - Chunks are embedded using sentence-transformers → stored in ChromaDB

2. **Query (every conversation turn)**
   - Your message is classified: freeform question OR book-index list
   - Freeform → single semantic search over ChromaDB
   - Book index → loop: search each topic separately, aggregate hits
   - Retrieved chunks are passed as context to the LLM
   - LLM synthesises a structured answer (year-wise table, frequency rank)

3. **Multi-turn**
   - Chat history is maintained in memory
   - Each new query includes prior conversation as context
   - Lets you refine: "now filter only 2023" after a broad query
