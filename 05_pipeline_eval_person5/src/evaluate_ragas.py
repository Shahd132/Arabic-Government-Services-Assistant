from __future__ import annotations

"""
evaluate_ragas.py
Runs the full end-to-end graph (graph.py) against a labeled QA set and
scores it with RAGAS: Faithfulness, Answer Relevancy, Context Precision,
Context Recall — the exact metrics the project spec calls for.

Usage:
    python evaluate_ragas.py --source local --limit 20
    python evaluate_ragas.py --source qa_law_egyptian --limit 50
    python evaluate_ragas.py --source qa_law_egyptian --department traffic --limit 30
    python evaluate_ragas.py --source egyptian_legal_v2 --limit 50

Default source is Omar-youssef/QA_LAW_Egyptian_dataset (3,725 Q&A pairs,
columns: question/answer/source_topics) — the project's chosen evaluation
set. It has no department field, so --department filters by keyword
matching over the question text + source_topics instead. Useful for
isolating the traffic department specifically, since that's the one with
the thinnest knowledge base (see the earlier retrieval review) — a low
score there tells you whether it's a generation problem or, more likely,
just not enough traffic-law text has been indexed yet.

NOTE ON LEAKAGE: this QA set and Person 2's knowledge base
(dataflare/egypt-legal-corpus) are both Egyptian legal text and likely
overlap in source. Report these numbers as a domain-in benchmark, not an
unseen test set.
"""

# ---------- FIX: Load .env from the project root ----------
from dotenv import load_dotenv
from pathlib import Path

# The project root is three levels up from this file:
#   evaluate_ragas.py  -> src/ -> 05_pipeline_eval_person5/ -> project_root/
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(env_path)
# -----------------------------------------------------------

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent.parent
sys.path.insert(0, str(_SRC_DIR))
sys.path.insert(0, str(_ROOT_DIR / "shared"))

from config import RAGAS_REPORT_PATH, PIPELINE_DIR  # noqa: E402
from logger import get_logger  # noqa: E402
from graph import run as run_graph  # noqa: E402

log = get_logger("evaluate_ragas")

LOCAL_SAMPLE_PATH = PIPELINE_DIR / "data" / "eval_sample.csv"


def load_local_sample() -> List[Dict]:
    import csv

    if not LOCAL_SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"{LOCAL_SAMPLE_PATH} not found. Create it with columns: question,ground_truth "
            "(a handful of Q&A pairs is enough for a smoke test)."
        )
    with open(LOCAL_SAMPLE_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# Keyword lists for the department filter — same spirit as Person 2's
# metadata_tagger.py category mapping, applied here to free-text questions
# and source_topics instead of law categories.
_DEPARTMENT_KEYWORDS = {
    "civil_affairs": ["بطاقة", "ميلاد", "أحوال مدنية", "زواج", "طلاق", "وفاة", "هوية"],
    "tax": ["ضريب", "جمارك", "إقرار ضريبي", "قيمة مضافة"],
    "traffic": ["مرور", "رخصة قيادة", "مخالفة", "ترخيص مركبة", "سيارة", "رخصة العربية"],
}


def _matches_department(question: str, source_topics: str, department: str) -> bool:
    keywords = _DEPARTMENT_KEYWORDS.get(department, [])
    haystack = f"{question} {source_topics}"
    return any(kw in haystack for kw in keywords)


def load_hf_dataset(name: str, limit: int, department: str = None) -> List[Dict]:
    from datasets import load_dataset

    if name == "qa_law_egyptian":
        ds = load_dataset("Omar-youssef/QA_LAW_Egyptian_dataset", split="train")
        examples = []
        for r in ds:
            if department and not _matches_department(r["question"], r.get("source_topics", ""), department):
                continue
            examples.append({"question": r["question"], "ground_truth": r["answer"]})
            if len(examples) >= limit:
                break
        if department and not examples:
            log.warning("No questions matched department=%s — try widening _DEPARTMENT_KEYWORDS "
                        "in this file, or drop --department to evaluate the full set.", department)
        return examples

    if name == "egyptian_legal_v2":
        ds = load_dataset("tarekys5/egyptian_legal_v2", split="train")
        rows = ds.select(range(min(limit, len(ds))))
        return [{"question": r["input"], "ground_truth": r["output"]} for r in rows]

    raise ValueError(f"Unknown source: {name}")


def run_pipeline_on_examples(examples: List[Dict]) -> Dict[str, List]:
    """Runs graph.run() for every example and assembles RAGAS's expected columns."""
    questions, answers, contexts, ground_truths = [], [], [], []

    for i, ex in enumerate(examples):
        question = ex["question"]
        log.info("[%d/%d] %s", i + 1, len(examples), question[:60])
        try:
            result = run_graph(question)
        except Exception as exc:  # keep the eval run alive even if one question fails
            log.error("Pipeline failed on %r: %s", question, exc)
            continue

        questions.append(question)
        answers.append(result.get("final_answer", ""))
        contexts.append([rc.chunk.text for rc in result.get("retrieved_chunks", [])])
        ground_truths.append(ex.get("ground_truth", ""))

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }


def run_ragas_evaluation(data: Dict[str, List]) -> Dict:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

    dataset = Dataset.from_dict(data)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return result.to_pandas().mean(numeric_only=True).to_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="qa_law_egyptian",
                         choices=["local", "qa_law_egyptian", "egyptian_legal_v2"])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--department", default=None, choices=["civil_affairs", "tax", "traffic"],
                         help="Filter qa_law_egyptian to questions matching this department "
                              "(keyword-based, since the dataset has no department field).")
    args = parser.parse_args()

    if args.source == "local":
        examples = load_local_sample()
    else:
        examples = load_hf_dataset(args.source, args.limit, department=args.department)

    log.info("Loaded %d evaluation examples from %s", len(examples), args.source)

    data = run_pipeline_on_examples(examples)
    if not data["question"]:
        log.error("No examples were successfully run through the pipeline. Aborting.")
        return

    scores = run_ragas_evaluation(data)
    log.info("RAGAS scores: %s", scores)

    RAGAS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Keep per-department reports side by side instead of overwriting each other
    report_path = RAGAS_REPORT_PATH
    if args.department:
        report_path = RAGAS_REPORT_PATH.with_name(f"ragas_report_{args.department}.json")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "source": args.source,
            "department": args.department,
            "n_examples": len(data["question"]),
            "scores": scores,
        }, f, ensure_ascii=False, indent=2)
    log.info("Saved report -> %s", report_path)


if __name__ == "__main__":
    main()
