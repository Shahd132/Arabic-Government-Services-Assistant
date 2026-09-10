"""
graph.py – LangGraph workflow with OpenAI, short-term memory, and Corrective RAG.
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
from document_grader import grade_chunks

log = get_logger("graph")
MAX_REGENERATION_ATTEMPTS = 1


class GraphState(TypedDict, total=False):
    user_input: str
    document_path: Optional[str]
    conversation_history: List[Tuple[str, str]]
    ocr_text: Optional[str]
    query_for_pipeline: str
    department: Optional[str]
    router_confidence: Optional[float]
    retrieved_chunks: List
    grading_result: Optional[str]
    num_irrelevant: int
    draft_answer: Optional[str]
    source_ids: List[str]
    verification_verdict: Optional[str]
    verification_explanation: Optional[str]
    regeneration_attempts: int
    final_answer: str
    is_verified: bool


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
    """
    If a document was uploaded, skip rewriting (preserve OCR context).
    """
    question = state["user_input"]

    if state.get("ocr_text"):
        log.info("Document present -> skipping query rewrite")
        return {"query_for_pipeline": question}

    history = state.get("conversation_history", [])
    rewritten = rewrite_query(client, question, history)
    return {"query_for_pipeline": rewritten}


def router_node(state: GraphState) -> GraphState:
    """
    If a document was uploaded, use OCR text for routing.
    """
    predict = get_router_predict_fn()

    if state.get("ocr_text"):
        routing_text = state["ocr_text"][:300]
        log.info("Using OCR text for routing")
    else:
        routing_text = state["query_for_pipeline"]

    raw_label, confidence = predict(routing_text)
    department = ROUTER_LABEL_MAP.get(raw_label, "civil_affairs")
    log.info(
        "Router: %r -> %s (raw=%s, conf=%.2f)",
        routing_text[:50],
        department,
        raw_label,
        confidence,
    )
    return {"department": department, "router_confidence": confidence}


def retrieval_node(state: GraphState) -> GraphState:
    """
    If a document was uploaded, use OCR text for retrieval.
    """
    retrieve = get_retrieve_fn()

    if state.get("ocr_text"):
        retrieval_query = state["ocr_text"][:500]
        log.info("Document retrieval using OCR text")
    else:
        retrieval_query = state["query_for_pipeline"]

    results = retrieve(
        retrieval_query, top_k=TOP_K, department=state["department"]
    )
    log.info(
        "Retrieved %d chunks for dept=%s", len(results), state["department"]
    )
    return {"retrieved_chunks": results}


def grade_documents_node(state: GraphState, client: openai.OpenAI) -> GraphState:
    """
    ✨ Corrective RAG node:
    - Evaluate each retrieved chunk.
    - Keep only relevant chunks.
    - Fallback to all chunks if everything is filtered out.
    """
    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        log.warning("No chunks to grade")
        return {"grading_result": "no_chunks", "num_irrelevant": 0}

    # Build grading question (include OCR content if available)
    if state.get("ocr_text"):
        grading_question = (
            f"{state['query_for_pipeline']}\n\n"
            f"Document Content:\n{state['ocr_text'][:300]}"
        )
    else:
        grading_question = state["query_for_pipeline"]

    relevant, num_irrelevant = grade_chunks(client, grading_question, chunks)

    if not relevant and chunks:
        log.warning("All chunks graded as IRRELEVANT -> using all as fallback")
        relevant = chunks
        grading_result = "all_irrelevant_fallback"
    elif num_irrelevant > 0:
        grading_result = "some_irrelevant"
    else:
        grading_result = "all_relevant"

    return {
        "retrieved_chunks": relevant,
        "grading_result": grading_result,
        "num_irrelevant": num_irrelevant,
    }


def generation_node(state: GraphState, client: openai.OpenAI) -> GraphState:
    answer, source_ids = generate_answer(
        client, state["query_for_pipeline"], state["retrieved_chunks"]
    )
    return {"draft_answer": answer, "source_ids": source_ids}


def verification_node(state: GraphState, client: openai.OpenAI) -> GraphState:
    result = verify_answer(
        client,
        state["query_for_pipeline"],
        state["retrieved_chunks"],
        state["draft_answer"],
    )
    log.info("Verification: %s (%s)", result.verdict, result.explanation)

    return {
        "verification_verdict": result.verdict,
        "verification_explanation": result.explanation,
        "is_verified": result.is_supported,
    }


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

    if verdict not in ("SUPPORTED", "PARTIALLY_SUPPORTED"):
        caveat = (
            f"\n\n⚠️ ملاحظة: لم يتم التحقق الكامل من هذه الإجابة "
            f"({state.get('verification_explanation', '')}). "
            "يُنصح بمراجعة الجهة الرسمية."
        )
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
    graph.add_node("grade_documents_node", lambda s: grade_documents_node(s, client))
    graph.add_node("generation_node", lambda s: generation_node(s, client))
    graph.add_node("verification_node", lambda s: verification_node(s, client))
    graph.add_node("regenerate_node", regenerate_node)
    graph.add_node("finalize_node", finalize_node)

    graph.set_entry_point("check_document_node")
    graph.add_conditional_edges(
        "check_document_node",
        route_after_document_check,
        {"ocr_node": "ocr_node", "rewrite_query_node": "rewrite_query_node"},
    )
    graph.add_edge("ocr_node", "rewrite_query_node")
    graph.add_edge("rewrite_query_node", "router_node")
    graph.add_edge("router_node", "retrieval_node")
    graph.add_edge("retrieval_node", "grade_documents_node")
    graph.add_edge("grade_documents_node", "generation_node")
    graph.add_edge("generation_node", "verification_node")
    graph.add_conditional_edges(
        "verification_node",
        route_after_verification,
        {
            "finalize_node": "finalize_node",
            "regenerate_node": "regenerate_node",
        },
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
    history: Optional[List[Tuple[str, str]]] = None,
) -> dict:
    compiled = _get_compiled_graph()

    use_memory = history is None
    conversation_history = (
        memory.get_history(session_id) if use_memory else history
    )

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
    print("Verified:", result1.get("is_verified"))
    print("Grading:", result1.get("grading_result"))
    print("\n--- ANSWER 1 ---\n")
    print(result1.get("final_answer"))
