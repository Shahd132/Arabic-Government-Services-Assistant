"""
verification.py — Uses OpenAI now.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))
from config import LLM_MODEL, MAX_TOKENS_VERIFICATION
from logger import get_logger
from generation import _format_context

log = get_logger("verification")

VERIFICATION_SYSTEM_PROMPT = """You are a strict fact-checker for a government information assistant. You will be given a citizen's question, the retrieved official context, and a draft answer. Decide how well the draft answer is supported by the context.

Respond with ONLY a JSON object, no other text:
{"verdict": "SUPPORTED" | "PARTIALLY_SUPPORTED" | "CONTRADICTION" | "UNSUPPORTED", "explanation": "<one short sentence in Arabic>"}

VERDICT DEFINITIONS:
- SUPPORTED: every claim in the answer is backed by the context.
- PARTIALLY_SUPPORTED: some claims are backed by the context, but the answer also includes at least one claim the context does not cover.
- CONTRADICTION: the answer states something that conflicts with the context.
- UNSUPPORTED: the answer's claims are not covered by the context at all.

EXAMPLE:
Question: إزاي أطلع بطاقة رقم قومي لأول مرة؟
Context: [law_143_1994_7] مادة 48: يجب على كل مواطن بلغ من العمر ست عشرة سنة أن يتقدم بطلب للحصول على بطاقة تحقيق الشخصية خلال ستة أشهر من بلوغ هذا السن.
Draft Answer: تقدر تطلع البطاقة من عمر 16 سنة، والرسوم 50 جنيه.
Output: {"verdict": "PARTIALLY_SUPPORTED", "explanation": "السن مذكور في السياق لكن الرسوم غير مذكورة."}
"""


@dataclass
class VerificationResult:
    is_supported: bool
    verdict: str
    explanation: str


def _build_prompt(question: str, retrieved_chunks: List, draft_answer: str) -> str:
    context = _format_context(retrieved_chunks)
    return (
        f"Question:\n{question}\n\n"
        f"Retrieved Context:\n{context}\n\n"
        f"Draft Answer:\n{draft_answer}"
    )


def verify_answer(client, question: str, retrieved_chunks: List, draft_answer: str) -> VerificationResult:
    """client: an openai.OpenAI() instance."""
    prompt = _build_prompt(question, retrieved_chunks, draft_answer)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS_VERIFICATION,
        temperature=0.0,
        messages=[
            {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    raw = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
        verdict = parsed.get("verdict", "UNSUPPORTED")
        explanation = parsed.get("explanation", "")
    except json.JSONDecodeError:
        log.warning("Verifier returned non-JSON, defaulting to UNSUPPORTED: %r", raw)
        verdict, explanation = "UNSUPPORTED", "تعذر التحقق من الإجابة تلقائيًا."

    is_supported = verdict == "SUPPORTED"
    return VerificationResult(is_supported=is_supported, verdict=verdict, explanation=explanation)