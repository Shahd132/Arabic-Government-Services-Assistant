"""
integration/end_to_end_test.py
A minimal sanity test: question in -> answer out, with basic assertions
that every stage of the graph actually produced something. Not a replacement
for evaluate_ragas.py's real metrics — just a fast "did I break the wiring"
check to run after any change to Persons 1-5's code.

Usage:
    python integration/end_to_end_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "05_pipeline_eval_person5" / "src"))
sys.path.insert(0, str(ROOT_DIR / "shared"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT_DIR / ".env")

from graph import run  # noqa: E402

TEST_QUESTIONS = [
    ("إزاي أطلع بطاقة رقم قومي لأول مرة؟", "civil_affairs"),
    ("إزاي أدفع الضريبة العقارية؟", "tax"),
    ("عندي مخالفة مرور، أعمل إيه؟", "traffic"),
]


def main():
    failures = 0
    for question, expected_department in TEST_QUESTIONS:
        print(f"\n> {question}")
        result = run(question)

        department = result.get("department")
        chunks = result.get("retrieved_chunks", [])
        answer = result.get("final_answer")

        ok_department = department == expected_department
        ok_retrieval = len(chunks) > 0
        ok_answer = bool(answer)

        status = "PASS" if (ok_department and ok_retrieval and ok_answer) else "FAIL"
        if status == "FAIL":
            failures += 1

        print(f"  [{status}] department={department} (expected {expected_department}), "
              f"chunks_retrieved={len(chunks)}, answer_length={len(answer or '')}")

    print(f"\n{len(TEST_QUESTIONS) - failures}/{len(TEST_QUESTIONS)} passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()