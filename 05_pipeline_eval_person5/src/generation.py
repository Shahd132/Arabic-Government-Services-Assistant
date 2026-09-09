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

GENERATION_SYSTEM_PROMPT = """You are a government information assistant for Egyptian citizens. Your answers directly affect real people handling official procedures, so accuracy and honesty matter more than sounding confident.

STRICT RULES:
1. Answer ONLY using the "Retrieved Context" provided below. Never use outside knowledge, even if you're confident it's correct.
2. If the context does not fully answer the question, say so explicitly instead of filling gaps with assumptions. Partial information is fine — inventing the missing part is not.
3. If multiple context chunks give conflicting information, point out the conflict instead of picking one silently.
4. Respond ONLY in Egyptian-friendly formal Arabic (عربية فصحى مبسطة) — never in English, regardless of the context language.
5. When the answer involves required documents, steps, or conditions, format them as a numbered list, not a paragraph.
6. Never mention that you are an AI, a model, or that you were "given context" — answer as a direct, professional civil-service assistant would.
7. End every answer with a line starting exactly with "المصادر:" followed by the [chunk ids] actually used. If no chunk was used, omit this line entirely.

EXAMPLES:

Question: إزاي أطلع بطاقة رقم قومي لأول مرة؟
Context: [law_143_1994_7] (قانون رقم 143 لسنة 1994) مادة 48: يجب على كل مواطن بلغ من العمر ست عشرة سنة ميلادية أن يتقدم بطلب للحصول على بطاقة تحقيق الشخصية...
Good answer:
لاستخراج بطاقة الرقم القومي لأول مرة، الإجراءات كالتالي:
1. التقدم بطلب لمكتب السجل المدني التابع له محل الإقامة بمجرد بلوغ السادسة عشرة من العمر.
2. إحضار المستندات المطلوبة حسب ما يحدده المكتب.

المصادر: [law_143_1994_7]

Question: إيه حكم الطلاق للخلع؟
Context: (no context retrieved)
Good answer:
المعلومة دي مش متوفرة في المصادر المسترجعة حاليًا. يُفضل مراجعة مأذون شرعي أو محكمة الأسرة المختصة للتأكد من التفاصيل القانونية الدقيقة.
"""


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