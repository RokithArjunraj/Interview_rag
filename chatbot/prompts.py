"""
chatbot/prompts.py
------------------
Prompt templates for each query mode.

RAG concept learned here:
  - The prompt is where you control the LLM's output format.
  - Structured prompts → structured answers.
  - Separating "context" (retrieved data) from "instructions" (what to do with it)
    is a core RAG prompt pattern.
"""


SYSTEM_PROMPT = """You are an interview preparation assistant for students of the PGDBA program.
You have access to interview experiences shared by past batches.
Answer questions clearly and concisely. Be specific about which companies and batches the information comes from.
If the retrieved context does not contain enough information to answer confidently, say so honestly."""


def freeform_prompt(question: str, context: str, history: str) -> str:
    return f"""Using the interview experiences below, answer the question.
Mention which companies the information is from.
If asked for frequency or trends, highlight the most commonly asked topics.

Retrieved interview context:
{context}

{f"Previous conversation:{chr(10)}{history}{chr(10)}" if history else ""}
Question: {question}

Answer:"""


def book_index_prompt(topic_list: list[str], context: str, history: str) -> str:
    topics_str = "\n".join(f"  - {t}" for t in topic_list)
    return f"""The user has provided the following reference topics (e.g. from a book index or syllabus):
{topics_str}

Based on the interview data below, tell them:
1. Which of these topics actually appeared in interviews
2. Which companies asked about each topic
3. An overall frequency ranking — which topics were asked most

Format your answer as:
- A bullet list: topic → companies where it appeared (and how often)
- A brief "Top topics to prioritise" section at the end

Retrieved interview data:
{context}

{f"Previous conversation:{chr(10)}{history}{chr(10)}" if history else ""}
Answer:"""


def comparison_prompt(question: str, context: str, history: str) -> str:
    return f"""The user wants to compare interview experiences across companies.
Using the retrieved data below, provide a structured comparison.

Show:
- What each company focuses on (technical depth, domain knowledge, case studies, coding)
- Any topics that appear in both / all companies
- Key differences in interview style

Retrieved interview context:
{context}

{f"Previous conversation:{chr(10)}{history}{chr(10)}" if history else ""}
Question: {question}

Comparison:"""


def format_history(history: list[dict]) -> str:
    """Format chat history for inclusion in prompts."""
    if not history:
        return ""
    lines = []
    for turn in history[-4:]:   # only last 4 turns to stay within context limits
        role = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content'][:300]}")
    return "\n".join(lines)
