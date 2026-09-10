import os
from pathlib import Path

SHARED_DIR = Path(__file__).resolve().parent
ROOT_DIR = SHARED_DIR.parent

OCR_DIR = ROOT_DIR / "01_ocr_person1"
KB_DIR = ROOT_DIR / "02_knowledge_base_person2"
RETRIEVAL_DIR = ROOT_DIR / "03_retrieval_person3"
ROUTER_DIR = ROOT_DIR / "04_router_person4"
PIPELINE_DIR = ROOT_DIR / "05_pipeline_eval_person5"
INTEGRATION_DIR = ROOT_DIR / "integration"

RETRIEVAL_SRC_DIR = RETRIEVAL_DIR / "src"
ROUTER_MODEL_DIR = ROUTER_DIR / "models" / "router_model"

DEPARTMENTS = ["civil_affairs", "tax", "traffic"]

ROUTER_LABEL_MAP = {
    "Civil Affairs": "civil_affairs",
    "Personal Status": "civil_affairs",
    "Taxes": "tax",
    "Traffic": "traffic",
}

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
MAX_TOKENS_GENERATION = 800
MAX_TOKENS_VERIFICATION = 300
MAX_TOKENS_REWRITE = 100

TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "2"))
RAGAS_REPORT_PATH = PIPELINE_DIR / "results" / "ragas_report.json"
