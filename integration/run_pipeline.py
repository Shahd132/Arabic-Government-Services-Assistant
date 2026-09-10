"""
integration/run_pipeline.py
The CLI entry point calling all 5 modules in sequence, via Person 5's
graph. This is the file to run for a quick end-to-end sanity check, or
to use interactively from the terminal.

Usage:
    python integration/run_pipeline.py "إزاي أطلع بطاقة رقم قومي؟"
    python integration/run_pipeline.py "المستند ده خاص بإيه؟" --document path/to/scan.pdf
    python integration/run_pipeline.py   # interactive mode, supports multi-turn follow-ups
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "05_pipeline_eval_person5" / "src"))
sys.path.insert(0, str(ROOT_DIR / "shared"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT_DIR / ".env")

from graph import run, reset_conversation  # noqa: E402


def print_answer(result: dict) -> None:
    print(f"\nDepartment: {result.get('department')} "
          f"(confidence: {result.get('router_confidence', 0):.2%})")
    print(f"Sources: {result.get('source_ids')}")
    print("\n--- Answer ---")
    print(result.get("final_answer"))
    print()


def interactive_mode():
    print("Smart Citizen Assistant — interactive mode.")
    print("Type 'new' to start a fresh conversation, 'exit' to quit.\n")
    session_id = "cli-session"
    while True:
        question = input("سؤالك: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question.lower() == "new":
            reset_conversation(session_id)
            print("(started a new conversation — short-term memory cleared)\n")
            continue
        result = run(question, session_id=session_id)
        print_answer(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default=None)
    parser.add_argument("--document", default=None, help="Path to a PDF/image to analyze")
    args = parser.parse_args()

    if args.question is None:
        interactive_mode()
        return

    result = run(args.question, document_path=args.document)
    print_answer(result)


if __name__ == "__main__":
    main()