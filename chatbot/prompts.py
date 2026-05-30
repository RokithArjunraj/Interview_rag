"""
chatbot/prompts.py
------------------
Prompt templates — one per query mode.
Each prompt explicitly tells the LLM:
  - What data it has
  - What format to reply in
  - To ONLY use what's in the context (no hallucination)
  - How to weight recency
"""

SYSTEM_PROMPT = """You are an interview preparation assistant for PGDBA students.
You help current batch students prepare by analysing past interview experiences.

STRICT RULES — follow these always:
1. Only use information present in the retrieved context. Do NOT invent questions, companies, or trends.
2. If the context does not contain enough information to answer, say exactly: "Not enough data in the retrieved interviews to answer this confidently."
3. Always mention which batch(es) the information is from.
4. Give more weight to recent batches — they reflect current interview patterns better.
5. Be specific and precise. Avoid vague statements like "various topics were asked."
"""


def concept_prompt(question: str, context: str, history: str) -> str:
    return f"""The student is asking about a CONCEPT or TOPIC across all companies and batches.

Your task:
1. List which companies asked about this concept, with batch numbers
2. For each company, describe what specific angle they take on this concept
3. Highlight if the same concept appears across 3+ batches (mark it as "frequently asked")
4. Give more prominence to the most recent 2-3 batches
5. At the end, add a "Preparation tip" based on the patterns

Retrieved interview data:
{context}

{f"Previous conversation:{chr(10)}{history}" if history else ""}

Question: {question}

Answer (use only the data above, cite batch numbers):"""


def company_prompt(question: str, context: str, company: str, history: str) -> str:
    return f"""The student is asking about {company}'s interview pattern.

Your task:
1. List important questions/topics asked at {company}, organised by batch (most recent first)
2. Highlight concepts that appear in multiple batches — these are consistently tested
3. Note if the focus has shifted in recent batches vs older ones
4. Mention the interview structure (rounds, style) if visible in the data
5. End with "Key areas to focus for {company}" based on the pattern

Retrieved interview data for {company}:
{context}

{f"Previous conversation:{chr(10)}{history}" if history else ""}

Question: {question}

Answer (cite batch numbers, flag frequently recurring topics):"""


def comparison_prompt(question: str, context: str, companies: list, history: str) -> str:
    co_str = " vs ".join(companies)
    return f"""The student wants to COMPARE interviews at: {co_str}

Your task:
1. For each company: summarise what they focus on (technical depth, domain, coding, cases)
2. Topics that appear in BOTH/ALL companies — list these as "Common ground"
3. Topics unique to each company — list as "Differentiators"
4. Which company has harder technical rounds? Which is more domain-focused?
5. Preparation strategy: what to study if targeting both

IMPORTANT: Only compare what's actually in the data below. If a company has limited data, say so.

Retrieved data:
{context}

{f"Previous conversation:{chr(10)}{history}" if history else ""}

Question: {question}

Comparison (be specific, cite batches):"""


def batch_prompt(question: str, context: str, batch: int, history: str) -> str:
    return f"""The student is asking specifically about Batch {batch} interview experiences.

Your task:
1. Summarise what was asked in Batch {batch} across companies
2. Group by company, list actual questions asked
3. Note any patterns specific to Batch {batch}

Retrieved Batch {batch} data:
{context}

{f"Previous conversation:{chr(10)}{history}" if history else ""}

Question: {question}

Answer:"""


def book_index_prompt(topic_list: list, context: str, history: str) -> str:
    topics_str = "\n".join(f"  - {t}" for t in topic_list)
    return f"""The student has provided a reference topic list (e.g. from ISLP, PRML, or a syllabus).

Topics provided:
{topics_str}

Your task:
1. For each topic: which companies asked it, in which batches
2. Mark topics asked in 3+ batches as "HIGH FREQUENCY ⭐"
3. Mark topics only in recent batches as "TRENDING 📈"
4. Mark topics with no data as "Not seen in interviews"
5. End with: "Top 5 topics to prioritise" based on frequency + recency

Retrieved data:
{context}

{f"Previous conversation:{chr(10)}{history}" if history else ""}

Answer:"""


def freeform_prompt(question: str, context: str, history: str) -> str:
    return f"""Answer the student's interview prep question using only the data below.
Be specific. Cite companies and batch numbers. Do not add information not present in the context.

Retrieved data:
{context}

{f"Previous conversation:{chr(10)}{history}" if history else ""}

Question: {question}
Answer:"""


def format_history(history: list[dict]) -> str:
    if not history:
        return ""
    lines = []
    for turn in history[-4:]:
        role = "Student" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content'][:300]}")
    return "\n".join(lines)
