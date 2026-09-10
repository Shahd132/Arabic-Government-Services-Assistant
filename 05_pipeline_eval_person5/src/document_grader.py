"""
document_grader.py
Corrective RAG: يقيّم كل chunk مسترجع قبل تمريره للـ LLM.

FIX: 
- عند وجود مستند، قارن الـ chunk بمحتوى المستند (وليس بالسؤال).
- كشف أكثر مرونة لإجابة LLM.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))

from config import LLM_MODEL
from logger import get_logger

log = get_logger("document_grader")


# ============================================================
# Prompt التقييم (محسّن)
# ============================================================

GRADER_SYSTEM_PROMPT = """You are a relevance grader for an Egyptian government assistant.

Your task: Decide if a retrieved document chunk is RELEVANT to the reference text.

The reference text can be either:
- A user question (e.g., "إزاي أطلع بطاقة رقم قومي؟"), OR
- The content of an uploaded document (e.g., "مصلحة الضرائب المصرية / إقرار ضريبة الدخل...")

Rules:
- RELEVANT = the chunk talks about the SAME TOPIC as the reference.
- IRRELEVANT = the chunk talks about a DIFFERENT TOPIC.

CRITICAL: Match on TOPIC, not on generic words like "إجراءات" or "المستند".

Examples:

Example 1 (document upload):
Reference: "مصلحة الضرائب المصرية / إقرار ضريبة الدخل السنوي..."
Chunk: "إجراءات الطلاق تتطلب حضور الطرفين أمام المأذون..."
→ irrelevant

Example 2 (document upload):
Reference: "مصلحة الضرائب المصرية / إقرار ضريبة الدخل السنوي..."
Chunk: "قانون الضريبة على الدخل يحدد شرائح الضريبة..."
→ relevant

Example 3 (question only):
Reference: "إزاي أطلع بطاقة رقم قومي؟"
Chunk: "إجراءات استخراج بطاقة الرقم القومي تتطلب شهادة ميلاد..."
→ relevant

Example 4 (question only):
Reference: "إزاي أطلع بطاقة رقم قومي؟"
Chunk: "إجراءات الطلاق تتطلب حضور الطرفين أمام المأذون..."
→ irrelevant

Respond with ONLY one word: "relevant" or "irrelevant". No explanations.
"""


def grade_chunk(
    client,
    reference_text: str,
    chunk_text: str,
) -> bool:
    """
    يقيّم ما إذا كان الـ chunk مرتبطاً بالنص المرجعي.

    Args:
        reference_text: إما السؤال أو محتوى المستند

    Returns:
        True إذا كان مرتبطاً، False إذا لم يكن
    """
    snippet = chunk_text[:500]

    prompt = f"""Reference:
{reference_text[:600]}

Retrieved Chunk:
{snippet}

Is this chunk relevant to the reference? Answer with one word only (relevant / irrelevant):"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=5,
            temperature=0.0,
            messages=[
                {"role": "system", "content": GRADER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        verdict = response.choices[0].message.content.strip().lower()

        # ✨ FIX: More robust detection
        verdict_clean = verdict.replace(".", "").replace(",", "").strip()

        # If "irrelevant" appears anywhere → False
        if "irrelevant" in verdict_clean:
            return False
        # If "relevant" appears anywhere → True
        if "relevant" in verdict_clean:
            return True

        # Fallback: assume relevant (safety)
        log.warning("Ambiguous grader verdict: %r -> defaulting to relevant", verdict)
        return True

    except Exception as e:
        log.warning("Grading failed: %s. Defaulting to relevant.", e)
        return True


def grade_chunks(
    client,
    reference_text: str,
    chunks: List,
) -> Tuple[List, int]:
    """
    يقيّم قائمة من الـ chunks ويعيد المرتبط منها فقط.

    Returns:
        (relevant_chunks, num_irrelevant)
    """
    if not chunks:
        return [], 0

    relevant = []
    irrelevant_count = 0

    for rc in chunks:
        chunk_text = rc.chunk.text if hasattr(rc, "chunk") else rc.text
        chunk_id = getattr(rc.chunk, "id", "?") if hasattr(rc, "chunk") else "?"

        is_relevant = grade_chunk(client, reference_text, chunk_text)

        if is_relevant:
            relevant.append(rc)
            log.info("Chunk %s: RELEVANT", chunk_id)
        else:
            irrelevant_count += 1
            log.info("Chunk %s: IRRELEVANT", chunk_id)

    log.info(
        "Grading complete: %d relevant, %d irrelevant",
        len(relevant),
        irrelevant_count,
    )
    return relevant, irrelevant_count