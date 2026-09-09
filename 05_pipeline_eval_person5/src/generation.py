"""
generation.py — Uses OpenAI now.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))
from config import LLM_MODEL, MAX_TOKENS_GENERATION
from logger import get_logger

log = get_logger("generation")

GENERATION_SYSTEM_PROMPT = """You are a government information assistant for Egyptian citizens.
Answer the citizen's question in Arabic, using ONLY the provided official context below.
Do not use any outside knowledge. If the answer is not supported by the context,
say clearly that the information is unavailable in the retrieved sources — do not guess.
Always end your answer with a "Sources:" line listing the [chunk ids] you actually used."""


def _format_context(retrieved_chunks: List) -> str:
    blocks = []
    for i, rc in enumerate(retrieved_chunks, start=1):
        law_name = rc.chunk.law_name or "unknown"
        blocks.append(f"[{rc.chunk.id}] ({law_name})\n{rc.chunk.text}")
    return "\n\n".join(blocks) if blocks else "(no context retrieved)"


def build_prompt(question: str, retrieved_chunks: List) -> str:
    context = _format_context(retrieved_chunks)
    return f"Question:\n{question}\n\nRetrieved Context:\n{context}"


def generate_answer(client, question: str, retrieved_chunks: List) -> Tuple[str, List[str]]:
    """
    client: an openai.OpenAI() instance.
    Returns (answer_text, source_chunk_ids).
    """
    if not retrieved_chunks:
        log.warning("No chunks retrieved for question: %r", question)
        return ("المعلومات غير متوفرة في المصادر المسترجعة لهذا السؤال.", [])

    prompt = build_prompt(question, retrieved_chunks)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS_GENERATION,
        temperature=0.1,
        messages=[
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    answer_text = response.choices[0].message.content.strip()
    source_ids = [rc.chunk.id for rc in retrieved_chunks]
    return answer_text, source_ids