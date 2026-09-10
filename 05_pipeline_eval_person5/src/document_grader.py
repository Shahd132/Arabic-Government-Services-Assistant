"""
document_grader.py
Corrective RAG: يقيّم كل chunk مسترجع قبل تمريره للـ LLM.

الفكرة:
- بعد Retrieval، لا نثق بالـ chunks مباشرة.
- نسأل LLM: هل هذا الـ chunk مرتبط فعلاً بالسؤال/المستند؟
- نحتفظ فقط بالـ chunks المرتبطة.
- إذا كانت كل الـ chunks غير مرتبطة، نعيدها كـ fallback.

الفائدة:
- يمنع الهلوسة الناتجة عن chunks غير مرتبطة.
- يحل مشكلة "رفعت PDF ضرائب ورد عن الطلاق".
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
# Prompt التقييم
# ============================================================

GRADER_SYSTEM_PROMPT = """You are a relevance grader for an Egyptian government assistant.

Your task: Decide if a retrieved document chunk is RELEVANT to the user's question.

Rules:
- RELEVANT = the chunk contains information that helps answer the question.
- IRRELEVANT = the chunk is about a completely different topic.

Respond with ONLY one word: "relevant" or "irrelevant". No explanations.

Examples:

Example 1:
Question: "المستند ده خاص بإيه؟" (user uploaded a tax return document)
Chunk: "ضريبة القيمة المضافة هي ضريبة على السلع والخدمات..."
Answer: irrelevant

Example 2:
Question: "المستند ده خاص بإيه؟" (user uploaded a tax return document)
Chunk: "إقرار ضريبة الدخل السنوي يجب تقديمه قبل 31 مارس..."
Answer: relevant

Example 3:
Question: "إزاي أطلع بطاقة رقم قومي؟"
Chunk: "إجراءات استخراج بطاقة الرقم القومي تتطلب شهادة ميلاد..."
Answer: relevant

Example 4:
Question: "إزاي أطلع بطاقة رقم قومي؟"
Chunk: "إجراءات الطلاق تتطلب حضور الطرفين أمام المأذون..."
Answer: irrelevant
"""


def grade_chunk(
    client,
    question: str,
    chunk_text: str,
) -> bool:
    """
    يقيّم ما إذا كان الـ chunk مرتبطاً بالسؤال.

    Returns:
        True إذا كان الـ chunk مرتبطاً (relevant)
        False إذا لم يكن مرتبطاً (irrelevant)
    """
    # قص الـ chunk لطول معقول لتسريع التقييم وتقليل التكلفة
    snippet = chunk_text[:500]

    prompt = f"""Question:
{question}

Retrieved Chunk:
{snippet}

Is this chunk relevant to the question? Answer with one word only (relevant / irrelevant):"""

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

        # التحقق من الإجابة
        is_relevant = verdict.startswith("relevant") and not verdict.startswith("irrelevant")
        return is_relevant

    except Exception as e:
        log.warning("Grading failed: %s. Defaulting to relevant.", e)
        # في حالة الخطأ، نعتبره مرتبطاً (لعدم إيقاف النظام)
        return True


def grade_chunks(
    client,
    question: str,
    chunks: List,
) -> Tuple[List, int]:
    """
    يقيّم قائمة من الـ chunks ويعيد المرتبط منها فقط.

    Args:
        client: openai.OpenAI instance
        question: السؤال أو محتوى المستند
        chunks: List of RetrievedChunk

    Returns:
        (relevant_chunks, num_irrelevant)
    """
    if not chunks:
        return [], 0

    relevant = []
    irrelevant_count = 0

    for rc in chunks:
        # استخراج النص من RetrievedChunk
        chunk_text = rc.chunk.text if hasattr(rc, "chunk") else rc.text
        chunk_id = getattr(rc.chunk, "id", "?") if hasattr(rc, "chunk") else "?"

        is_relevant = grade_chunk(client, question, chunk_text)

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