"""
query_rewriter.py

Responsible for converting follow-up questions into
self-contained questions using short-term conversation memory.

Example:

Previous:
    إزاي أطلع بطاقة الرقم القومي؟

Current:
    طب لو ضاعت؟

Rewritten:
    ما الإجراءات المطلوبة في حالة فقدان بطاقة الرقم القومي؟

Important:
    If the current question is already self-contained,
    it should NOT be rewritten.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------
# Import shared configuration
# ---------------------------------------------------------

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent.parent.parent
        / "shared"
    )
)

from config import LLM_MODEL, MAX_TOKENS_REWRITE
from logger import get_logger


# ---------------------------------------------------------
# Logger
# ---------------------------------------------------------

log = get_logger("query_rewriter")


# ---------------------------------------------------------
# System Prompt
# ---------------------------------------------------------

REWRITE_SYSTEM_PROMPT = """You rewrite follow-up questions from Egyptian citizens into fully self-contained questions, using the prior conversation only to resolve what the follow-up refers to.

STRICT RULES:
1. If the current question is already self-contained and clear, return it exactly as-is.
2. Use the prior conversation ONLY to resolve pronouns or implicit references (ده، دي، الموضوع ده) in the current question.
3. Never add facts, assumptions, or details the citizen never stated.
4. Never change the topic or intent of the question.
5. Never answer the question — output the rewritten question only, nothing else.
6. If the current question is about a different topic than the prior conversation, return it as-is (do not force a connection).
7. Output must be in Arabic, matching the citizen's original phrasing style.

EXAMPLES:

Prior: إزاي أطلع بطاقة الرقم القومي؟
Current: طب لو ضاعت؟
Rewritten: ما الإجراءات المطلوبة في حالة فقدان بطاقة الرقم القومي؟

Prior: إزاي أدفع الضريبة العقارية؟
Current: طب المرور عايز اجدد رخصة عربيتي إزاي؟
Rewritten: إزاي أجدد رخصة قيادة عربيتي؟

Prior: (none)
Current: إيه هي مستندات استخراج شهادة ميلاد؟
Rewritten: إيه هي مستندات استخراج شهادة ميلاد؟
"""

# ---------------------------------------------------------
# Follow-up markers
# ---------------------------------------------------------

_FOLLOW_UP_MARKERS = [
    "طب",
    "طيب",
    "لو",
    "ده",
    "دي",
    "دول",
    "هو",
    "هي",
    "هما",
    "كمان",
    "برضو",
    "برضه",
    "وماذا",
    "وإيه",
    "وايه",
]


# ---------------------------------------------------------
# Helper: detect follow-up marker
# ---------------------------------------------------------

def _has_follow_up_marker(
    question: str
) -> bool:
    """
    Check whether the question starts with
    a common Egyptian Arabic follow-up marker.
    """

    normalized = question.strip()

    for marker in _FOLLOW_UP_MARKERS:

        if normalized.startswith(marker):
            return True

    return False


# ---------------------------------------------------------
# Helper: detect pronouns/references
# ---------------------------------------------------------

_REFERENCE_WORDS = [
    "ده",
    "دي",
    "دول",
    "ده؟",
    "دي؟",
    "دول؟",
    "دي بتاعته",
    "بتاعه",
    "بتاعها",
    "بتاعهم",
    "منه",
    "منها",
    "فيها",
    "فيه",
    "عليها",
    "عليه",
    "بيها",
    "به",
    "بها",
]


def _has_reference_word(
    question: str
) -> bool:
    """
    Detect words that usually refer to something
    mentioned earlier in the conversation.
    """

    words = question.strip().split()

    return any(
        word in _REFERENCE_WORDS
        for word in words
    )


# ---------------------------------------------------------
# Decide whether rewriting is needed
# ---------------------------------------------------------

def needs_rewrite(
    current_question: str,
    history: List[Tuple[str, str]]
) -> bool:
    """
    Decide whether the current question is likely
    a follow-up question.

    Important:
        A short question is NOT automatically a follow-up.

    Example:

        "إزاي أدفع الضريبة العقارية؟"

    is short but self-contained, so it should not
    automatically be rewritten.
    """

    if not history:
        return False

    question = current_question.strip()

    if not question:
        return False

    # -----------------------------------------------------
    # Explicit follow-up marker
    # -----------------------------------------------------

    if _has_follow_up_marker(question):
        return True

    # -----------------------------------------------------
    # References such as:
    # ده / دي / منه / فيها / عليه ...
    # -----------------------------------------------------

    if _has_reference_word(question):
        return True

    # -----------------------------------------------------
    # Very short questions
    #
    # Example:
    # "محتاج إيه؟"
    # "أعمل إيه؟"
    # "فين؟"
    #
    # These are highly dependent on previous context.
    # -----------------------------------------------------

    words = question.split()

    if len(words) <= 3:
        return True

    return False


# ---------------------------------------------------------
# Build conversation history for the LLM
# ---------------------------------------------------------

def _build_prompt(
    current_question: str,
    history: List[Tuple[str, str]]
) -> str:
    """
    Build the prompt sent to the query-rewriting LLM.
    """

    recent_history = history[-3:]

    history_parts = []

    for question, answer in recent_history:

        history_parts.append(
            f"المواطن: {question}\n"
            f"المساعد: {answer}"
        )

    history_block = "\n\n".join(
        history_parts
    )

    return (
        f"المحادثة السابقة:\n"
        f"{history_block}\n\n"
        f"السؤال الحالي:\n"
        f"{current_question}\n\n"
        f"السؤال المستقل بعد إعادة الصياغة:"
    )


# ---------------------------------------------------------
# Rewrite question
# ---------------------------------------------------------

def rewrite_query(
    client,
    current_question: str,
    history: List[Tuple[str, str]]
) -> str:
    """
    Rewrite a follow-up question into a self-contained
    question.

    Parameters
    ----------
    client:
        OpenAI client instance.

    current_question:
        Current user question.

    history:
        List of (question, answer) conversation turns.

    Returns
    -------
    str:
        Self-contained question.
    """

    current_question = current_question.strip()

    # -----------------------------------------------------
    # Empty question
    # -----------------------------------------------------

    if not current_question:
        return current_question

    # -----------------------------------------------------
    # No history
    # -----------------------------------------------------

    if not history:
        return current_question

    # -----------------------------------------------------
    # Check if rewriting is actually needed
    # -----------------------------------------------------

    if not needs_rewrite(
        current_question,
        history
    ):
        log.info(
            "No rewrite needed: %r",
            current_question
        )

        return current_question

    # -----------------------------------------------------
    # Build prompt
    # -----------------------------------------------------

    prompt = _build_prompt(
        current_question,
        history
    )

    # -----------------------------------------------------
    # Call OpenAI
    # -----------------------------------------------------

    response = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS_REWRITE,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": REWRITE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    # -----------------------------------------------------
    # Extract rewritten question
    # -----------------------------------------------------

    rewritten = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )

    # -----------------------------------------------------
    # Safety fallback
    # -----------------------------------------------------

    if not rewritten:
        log.warning(
            "Rewriter returned empty result. "
            "Using original question."
        )

        return current_question

    # -----------------------------------------------------
    # Log
    # -----------------------------------------------------

    log.info(
        "Rewrote %r -> %r",
        current_question,
        rewritten
    )

    return rewritten