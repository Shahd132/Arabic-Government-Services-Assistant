"""
generation.py
Generates answers using retrieved chunks + OCR text.
Strips SOURCES_USED marker before returning to the caller.
"""

from __future__ import annotations
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))

from config import LLM_MODEL, MAX_TOKENS_GENERATION
from logger import get_logger

log = get_logger("generation")


GENERATION_SYSTEM_PROMPT = """You are a government information assistant for Egyptian citizens. Your answers directly affect real people handling official procedures, so accuracy and honesty matter more than sounding confident.

STRICT RULES:
1. The context has TWO sections:
   - "نص المستند المرفوع" (PRIMARY — the actual uploaded document)
   - "المعلومات المسترجعة" (SECONDARY — retrieved legal chunks)

2. For document-analysis questions (like "هل المستند مكتمل؟" or "ما هي المعلومات المفقودة؟"):
   → Read the document text CAREFULLY.
   → Look for these Arabic markers:
      • "ملاحظات" / "ملاحظة"       = notes section
      • "لم يتم" / "غير مرفق"       = not attached / not done
      • "ناقص" / "مفقود"            = missing
      • "المستندات المرفقة"          = attached documents list
      • "التوقيع"                    = signature

3. If the document contains ANY of these phrases:
   - "لم يتم إرفاق" (not attached)
   - "لم يتم التوقيع" (not signed)
   - "ناقص" / "مفقود" (missing)
   - "غير مرفق" (not included)
   
   → THEN ANSWER: "المستند غير مكتمل" and LIST the missing items explicitly.

4. Do NOT say "المعلومات غير كافية" if the answer is literally written in the document text.
   The document text is the SOURCE OF TRUTH for document questions.

5. Answer in Arabic only. Use numbered lists for steps/items.

6. Do NOT include a "المصادر:" line inside the answer text.

EXAMPLES:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Example 1 — Document completeness (INCOMPLETE):

Document Text:
"...ملاحظات:
- الطلب مقدم لاستخراج بطاقة رقم قومي بدل فاقد.
- لم يتم إرفاق صورة شخصية.
- لم يتم التوقيع على الطلب."

Question: هل هذا المستند مكتمل؟

GOOD Answer:
المستند غير مكتمل. ينقصه:
1. صورة شخصية (لم يتم إرفاقها).
2. توقيع مقدم الطلب (لم يتم التوقيع على الطلب).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Example 2 — Missing info:

Question: ما هي المعلومات المفقودة؟

GOOD Answer:
المعلومات المفقودة في المستند:
1. صورة شخصية.
2. توقيع مقدم الطلب.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Example 3 — Document completeness (COMPLETE):

Document Text:
"...المستندات المرفقة:
✓ صورة شخصية حديثة
✓ شهادة ميلاد
✓ بطاقة الرقم القومي القديمة
التوقيع: أحمد محمد"

Question: هل المستند مكتمل؟

GOOD Answer:
نعم، المستند مكتمل. يحتوي على جميع المستندات المطلوبة والتوقيع.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Example 4 — Legal question (not about document):

Context:
[law_143_1994] مادة 48: يجب على كل مواطن بلغ 16 سنة أن يتقدم بطلب للحصول على بطاقة تحقيق الشخصية...

Question: إزاي أطلع بطاقة رقم قومي لأول مرة؟

GOOD Answer:
لاستخراج بطاقة الرقم القومي لأول مرة:
1. التقدم بطلب لمكتب السجل المدني التابع لمحل الإقامة بمجرد بلوغ السادسة عشرة.
2. إحضار المستندات المطلوبة حسب ما يحدده المكتب.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SOURCES_USED marker:
After the answer, on a new line by itself:
SOURCES_USED: ["chunk_id_1", "chunk_id_2"]
If no chunks were used: SOURCES_USED: []
"""


# ============================================================
# Context formatting
# ============================================================

def _format_context(retrieved_chunks: List) -> str:
    blocks = []
    for rc in retrieved_chunks:
        law_name = getattr(rc.chunk, "law_name", None) or "unknown"
        blocks.append(f"[{rc.chunk.id}] ({law_name})\n{rc.chunk.text}")
    return "\n\n".join(blocks) if blocks else "(لا يوجد سياق مسترجع)"


def build_prompt(question: str, retrieved_chunks: List, ocr_text: Optional[str] = None) -> str:
    context_parts = []

    if ocr_text and ocr_text.strip():
        context_parts.append(f"=== نص المستند المرفوع ===\n{ocr_text[:3000]}")

    if retrieved_chunks:
        context_parts.append(f"=== المعلومات المسترجعة ===\n{_format_context(retrieved_chunks)}")

    context = "\n\n".join(context_parts) if context_parts else "(لا يوجد سياق)"

    return f"""السؤال:
{question}

السياق:
{context}

الإجابة (بالعربية فقط):"""


# ============================================================
# Strip SOURCES_USED marker
# ============================================================

_SOURCES_PATTERN = re.compile(
    r"SOURCES_USED\s*:\s*(\[.*?\])\s*$",
    re.DOTALL | re.IGNORECASE,
)


def _strip_sources_marker(raw: str) -> Tuple[str, List[str]]:
    """
    Removes the SOURCES_USED line from the raw LLM output.
    Returns (clean_answer, parsed_source_ids).
    """
    if not raw:
        return "", []

    match = _SOURCES_PATTERN.search(raw)
    if not match:
        # Also try a more permissive removal (line-based)
        lines = raw.split("\n")
        kept_lines = []
        found_sources = []
        for line in lines:
            if "SOURCES_USED" in line.upper():
                # Try to extract JSON from this line
                try:
                    json_part = line.split(":", 1)[1].strip()
                    parsed = json.loads(json_part)
                    if isinstance(parsed, list):
                        found_sources = parsed
                except Exception:
                    pass
                continue
            kept_lines.append(line)
        return "\n".join(kept_lines).strip(), found_sources

    clean_answer = raw[:match.start()].strip()
    sources_raw = match.group(1)

    try:
        source_ids = json.loads(sources_raw)
        if not isinstance(source_ids, list):
            source_ids = []
    except Exception as e:
        log.warning("Could not parse SOURCES_USED JSON: %r (%s)", sources_raw, e)
        source_ids = []

    return clean_answer, source_ids


# ============================================================
# Main generate
# ============================================================

def generate_answer(
    client,
    question: str,
    retrieved_chunks: List,
    ocr_text: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """
    Generates the answer, strips SOURCES_USED marker,
    returns (clean_answer, source_chunk_ids).
    """
    if not retrieved_chunks and not ocr_text:
        log.warning("No chunks and no OCR text for question: %r", question)
        return ("المعلومات غير متوفرة في المصادر الرسمية.", [])

    prompt = build_prompt(question, retrieved_chunks, ocr_text)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS_GENERATION,
        temperature=0.1,
        messages=[
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    raw = response.choices[0].message.content.strip()

    clean_answer, llm_sources = _strip_sources_marker(raw)

    if llm_sources:
        log.info("LLM-reported sources: %s", llm_sources)

    source_ids = llm_sources if llm_sources else (
        [rc.chunk.id for rc in retrieved_chunks] if retrieved_chunks else []
    )

    return clean_answer, source_ids


if __name__ == "__main__":
    print("generation.py loaded — call generate_answer(client, question, chunks, ocr_text=None)")
