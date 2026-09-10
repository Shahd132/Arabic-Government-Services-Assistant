"""
generation.py — Uses OpenAI now.
"""
 
from __future__ import annotations
 
import json
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
2. The retrieved context may include excerpts that are NOT actually relevant to the question, even if they come from the same law or department. Before answering, judge each excerpt on its own: does it directly address what the user asked? Silently ignore any excerpt that does not, even if it is topically nearby. Do not let irrelevant excerpts influence, dilute, or contradict the answer.
3. If, after filtering, no excerpt sufficiently answers the question, say so explicitly instead of filling gaps with assumptions or leaning on a weak/partial match. Partial information is fine to share — inventing or guessing the missing part is not.
4. If multiple RELEVANT context chunks give conflicting information, point out the conflict instead of picking one silently.
5. Respond ONLY in Egyptian-friendly formal Arabic (عربية فصحى مبسطة) — never in English, regardless of the context language.
6. Write for a citizen who wants the practical information, not a legal reference. Do NOT mention law numbers, article numbers, or "according to law X" inside the answer text itself — just state the requirement, step, or fact directly and plainly.
7. When the answer involves required documents, steps, or conditions, format them as a numbered list, not a paragraph.
8. Never mention that you are an AI, a model, or that you were "given context" — answer as a direct, professional civil-service assistant would.
9. After the user-facing answer, on a new line, output the exact marker "SOURCES_USED:" followed by a JSON array of the chunk ids you actually relied on (only the ones you judged relevant and used — not every chunk you were given). If you used none (e.g. you said the information isn't available), output "SOURCES_USED: []". This line is for internal logging only — it will be stripped before the citizen sees the answer, so it does not need to read naturally and must not be mentioned anywhere in the answer text above it.
10. Never combine two separate context excerpts to produce a conclusion, condition, or number that neither excerpt states on its own — even if the combination seems logical or likely true. If connecting two excerpts requires an inferential leap you were not explicitly given, state each excerpt's fact separately (or omit the connection) rather than presenting the inferred conclusion as fact.
 
EXAMPLES:
 
Question: إزاي أطلع بطاقة رقم قومي لأول مرة؟
Context:
[law_143_1994_7] (قانون رقم 143 لسنة 1994) مادة 48: يجب على كل مواطن بلغ من العمر ست عشرة سنة ميلادية أن يتقدم بطلب للحصول على بطاقة تحقيق الشخصية...
[traffic_121_2008_3] (قانون المرور 121 لسنة 2008) مادة تتعلق بمخالفات السرعة والغرامات المقررة لها...
Good answer:
لاستخراج بطاقة الرقم القومي لأول مرة، الإجراءات كالتالي:
1. التقدم بطلب لمكتب السجل المدني التابع له محل الإقامة بمجرد بلوغ السادسة عشرة من العمر.
2. إحضار المستندات المطلوبة حسب ما يحدده المكتب.
SOURCES_USED: ["law_143_1994_7"]
 
(Note: the traffic-violation excerpt was ignored entirely — it has nothing to do with the question, even though it was retrieved alongside the correct one.)
 
Question: إيه حكم الطلاق للخلع؟
Context: (no context retrieved)
Good answer:
المعلومة دي مش متوفرة حاليًا. يُفضل مراجعة مأذون شرعي أو محكمة الأسرة المختصة للتأكد من التفاصيل القانونية الدقيقة.
SOURCES_USED: []
"""
 
SOURCES_MARKER = "SOURCES_USED:"
 
 
def _format_context(retrieved_chunks: List) -> str:
    blocks = []
    for i, rc in enumerate(retrieved_chunks, start=1):
        law_name = rc.chunk.law_name or "unknown"
        blocks.append(f"[{rc.chunk.id}] ({law_name})\n{rc.chunk.text}")
    return "\n\n".join(blocks) if blocks else "(no context retrieved)"
 
 
def build_prompt(question: str, retrieved_chunks: List) -> str:
    context = _format_context(retrieved_chunks)
    return f"Question:\n{question}\n\nRetrieved Context:\n{context}"
 
 
def _parse_answer_and_sources(raw_text: str, fallback_chunk_ids: List[str]) -> Tuple[str, List[str]]:
    """Splits the model's raw output into (citizen-facing answer, source ids).
 
    The model is instructed to end its output with a 'SOURCES_USED: [...]'
    line -- this function strips that line out so the citizen never sees
    chunk ids or internal bookkeeping, while still giving the caller
    (verification.py, logging, evaluation) the list of chunks the model
    actually says it relied on, as opposed to every chunk that happened
    to be retrieved.
    """
    if SOURCES_MARKER not in raw_text:
        # Model didn't follow the format -- fail safe: show the full text
        # as-is and fall back to "every retrieved chunk" for source tracking
        # rather than silently losing the answer.
        log.warning("Model output missing %r marker; using fallback sources", SOURCES_MARKER)
        return raw_text.strip(), fallback_chunk_ids
 
    answer_part, _, sources_part = raw_text.partition(SOURCES_MARKER)
    answer_text = answer_part.strip()
 
    sources_part = sources_part.strip()
    try:
        source_ids = json.loads(sources_part)
        if not isinstance(source_ids, list):
            raise ValueError("SOURCES_USED did not contain a JSON list")
    except (json.JSONDecodeError, ValueError):
        log.warning("Could not parse SOURCES_USED content %r; using fallback sources", sources_part)
        source_ids = fallback_chunk_ids
 
    return answer_text, source_ids
 
 
def generate_answer(client, question: str, retrieved_chunks: List) -> Tuple[str, List[str]]:
    """
    client: an openai.OpenAI() instance.
    Returns (answer_text, source_chunk_ids).
 
    answer_text is the clean, citizen-facing answer only -- no law numbers,
    no chunk ids, no internal markers. source_chunk_ids is for internal use
    (logging, verification, evaluation) and reflects only the chunks the
    model actually says it used, not every chunk that was retrieved.
    """
    if not retrieved_chunks:
        log.warning("No chunks retrieved for question: %r", question)
        return ("المعلومات غير متوفرة حاليًا لهذا السؤال.", [])
 
    all_chunk_ids = [rc.chunk.id for rc in retrieved_chunks]
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
    raw_text = response.choices[0].message.content.strip()
    answer_text, source_ids = _parse_answer_and_sources(raw_text, fallback_chunk_ids=all_chunk_ids)
    return answer_text, source_ids
