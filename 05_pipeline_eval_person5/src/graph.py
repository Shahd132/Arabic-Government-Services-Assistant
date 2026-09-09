"""
graph.py – LangGraph workflow with OpenAI, short-term memory (tuples), and query rewriting.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

# --- path bootstrap ---
_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent.parent
_SHARED_DIR = _ROOT_DIR / "shared"
for p in (_SRC_DIR, _SHARED_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from langgraph.graph import StateGraph, END
import openai

from config import ROUTER_LABEL_MAP, TOP_K, LLM_MODEL
from logger import get_logger

from module_loader import (
    get_ocr_process_document,
    get_retrieve_fn,
    get_router_predict_fn,
)
from query_rewriter import rewrite_query
from generation import generate_answer
from verification import verify_answer
from short_term_memory import memory

log = get_logger("graph")
MAX_REGENERATION_ATTEMPTS = 1


class GraphState(TypedDict, total=False):
    user_input: str
    document_path: Optional[str]
    conversation_history: List[Tuple[str, str]]   # changed to tuples
    ocr_text: Optional[str]
    query_for_pipeline: str
    department: Optional[str]
    router_confidence: Optional[float]
    retrieved_chunks: List
    draft_answer: Optional[str]
    source_ids: List[str]
    verification_verdict: Optional[str]
    verification_explanation: Optional[str]
    regeneration_attempts: int
    final_answer: str


# ---------- Nodes ----------
def check_document_node(state: GraphState) -> GraphState:
    has_doc = bool(state.get("document_path"))
    log.info("Document check: %s", "has document" if has_doc else "text-only")
    return {}


def ocr_node(state: GraphState) -> GraphState:
    process_document = get_ocr_process_document()
    text = process_document(state["document_path"])
    log.info("OCR extracted %d chars", len(text))
    return {"ocr_text": text}


def route_after_document_check(state: GraphState) -> str:
    return "ocr_node" if state.get("document_path") else "rewrite_query_node"


def rewrite_query_node(state: GraphState, client: openai.OpenAI) -> GraphState:
    base_query = state["user_input"]
    if state.get("ocr_text"):
        base_query = f"{state['user_input']}\n\nOCR Text:\n{state['ocr_text']}"

    # history is now List[Tuple[str, str]]
    history = state.get("conversation_history", [])
    rewritten = rewrite_query(client, base_query, history)
    return {"query_for_pipeline": rewritten}


def router_node(state: GraphState) -> GraphState:
    predict = get_router_predict_fn()
    raw_label, confidence = predict(state["query_for_pipeline"])
    department = ROUTER_LABEL_MAP.get(raw_label, "civil_affairs")
    log.info("Router: %r -> %s (raw=%s, conf=%.2f)", state["query_for_pipeline"][:50],
              department, raw_label, confidence)
    return {"department": department, "router_confidence": confidence}


def retrieval_node(state: GraphState) -> GraphState:
    retrieve = get_retrieve_fn()
    results = retrieve(state["query_for_pipeline"], top_k=TOP_K, department=state["department"])
    log.info("Retrieved %d chunks for dept=%s", len(results), state["department"])
    return {"retrieved_chunks": results}


def generation_node(state: GraphState, client: openai.OpenAI) -> GraphState:
    answer, source_ids = generate_answer(client, state["query_for_pipeline"], state["retrieved_chunks"])
    return {"draft_answer": answer, "source_ids": source_ids}


def verification_node(state: GraphState, client: openai.OpenAI) -> GraphState:
    result = verify_answer(client, state["query_for_pipeline"], state["retrieved_chunks"], state["draft_answer"])
    log.info("Verification: %s (%s)", result.verdict, result.explanation)
    return {"verification_verdict": result.verdict, "verification_explanation": result.explanation}


def route_after_verification(state: GraphState) -> str:
    if state.get("verification_verdict") in ("SUPPORTED", "PARTIALLY_SUPPORTED"):
        return "finalize_node"
    attempts = state.get("regeneration_attempts", 0)
    if attempts < MAX_REGENERATION_ATTEMPTS:
        return "regenerate_node"
    return "finalize_node"


def regenerate_node(state: GraphState) -> GraphState:
    return {"regeneration_attempts": state.get("regeneration_attempts", 0) + 1}


def finalize_node(state: GraphState) -> GraphState:
    verdict = state.get("verification_verdict")
    answer = state["draft_answer"]
    if verdict != "SUPPORTED":
        caveat = f"\n\n⚠️ ملاحظة: لم يتم التحقق الكامل من هذه الإجابة ({state.get('verification_explanation', '')}). يُنصح بمراجعة الجهة الرسمية."
        answer += caveat
    return {"final_answer": answer}


# ---------- Build Graph ----------
def build_graph(client: openai.OpenAI):
    graph = StateGraph(GraphState)

    graph.add_node("check_document_node", check_document_node)
    graph.add_node("ocr_node", ocr_node)
    graph.add_node("rewrite_query_node", lambda s: rewrite_query_node(s, client))
    graph.add_node("router_node", router_node)
    graph.add_node("retrieval_node", retrieval_node)
    graph.add_node("generation_node", lambda s: generation_node(s, client))
    graph.add_node("verification_node", lambda s: verification_node(s, client))
    graph.add_node("regenerate_node", regenerate_node)
    graph.add_node("finalize_node", finalize_node)

    graph.set_entry_point("check_document_node")
    graph.add_conditional_edges(
        "check_document_node",
        route_after_document_check,
        {"ocr_node": "ocr_node", "rewrite_query_node": "rewrite_query_node"}
    )
    graph.add_edge("ocr_node", "rewrite_query_node")
    graph.add_edge("rewrite_query_node", "router_node")
    graph.add_edge("router_node", "retrieval_node")
    graph.add_edge("retrieval_node", "generation_node")
    graph.add_edge("generation_node", "verification_node")
    graph.add_conditional_edges(
        "verification_node",
        route_after_verification,
        {"finalize_node": "finalize_node", "regenerate_node": "regenerate_node"}
    )
    graph.add_edge("regenerate_node", "generation_node")
    graph.add_edge("finalize_node", END)

    return graph.compile()


@lru_cache(maxsize=1)
def _get_compiled_graph():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Check .env")
    client = openai.OpenAI(api_key=api_key)
    return build_graph(client)


# ---------- Public API ----------
def run(
    question: str,
    document_path: Optional[str] = None,
    session_id: str = "default",
    history: Optional[List[Tuple[str, str]]] = None
) -> dict:
    """
    Run the pipeline. If `history` is given, it bypasses memory.
    Otherwise, it pulls history from short-term memory using session_id.
    """
    compiled = _get_compiled_graph()

    use_memory = history is None
    conversation_history = memory.get_history(session_id) if use_memory else history

    initial_state: GraphState = {
        "user_input": question,
        "document_path": document_path,
        "conversation_history": conversation_history,
        "regeneration_attempts": 0,
    }
    result = compiled.invoke(initial_state)

    if use_memory:
        memory.add_turn(session_id, question, result.get("final_answer", ""))

    return result


def reset_conversation(session_id: str = "default") -> None:
    memory.clear(session_id)


    # ---------- Demo ----------
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(_ROOT_DIR / ".env")

    session_id = "demo-session"

    q1 = "إيه الأوراق المطلوبة لاستخراج بطاقة الرقم القومي لأول مرة؟"
    print(f"\nQuestion 1: {q1}\n")
    result1 = run(q1, session_id=session_id)
    print("Department:", result1.get("department"))
    print("\n--- ANSWER 1 ---\n")
    print(result1.get("final_answer"))

    q2 = "طب لو ضاعت البطاقة؟"
    print(f"\n\nQuestion 2 (follow-up): {q2}\n")
    result2 = run(q2, session_id=session_id)
    print("Rewritten query:", result2.get("query_for_pipeline"))
    print("Department:", result2.get("department"))
    print("\n--- ANSWER 2 ---\n")
    print(result2.get("final_answer"))