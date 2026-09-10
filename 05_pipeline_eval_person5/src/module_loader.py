"""
module_loader.py - Dynamically imports Persons 1, 3, 4
"""
from __future__ import annotations
 
import importlib
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable
 
_THIS_DIR = Path(__file__).resolve().parent
_SHARED_DIR_BOOTSTRAP = _THIS_DIR.parent.parent / "shared"
if str(_SHARED_DIR_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR_BOOTSTRAP))
 
from config import OCR_DIR, RETRIEVAL_SRC_DIR, ROUTER_DIR, SHARED_DIR
 
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))
 
def _add_to_path(path: Path) -> None:
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)
 
def _import_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
 
# ---------- Person 1 OCR ----------
@lru_cache(maxsize=1)
def get_ocr_process_document() -> Callable[[str], str]:
    _add_to_path(OCR_DIR)
    run_ocr = _import_from_path("run_ocr", OCR_DIR / "run_ocr.py")
    return run_ocr.process_document
 
# ---------- Person 3 Retrieval ----------
@lru_cache(maxsize=1)
def get_retrieve_fn() -> Callable:
    _add_to_path(RETRIEVAL_SRC_DIR)
    retrieve_module = _import_from_path("retrieve", RETRIEVAL_SRC_DIR / "retrieve.py")
    return retrieve_module.retrieve
 
# ---------- Person 4 Router ----------
@lru_cache(maxsize=1)
def get_router_predict_fn() -> Callable[[str], tuple]:
    original_cwd = os.getcwd()
    try:
        os.chdir(ROUTER_DIR)
        _add_to_path(ROUTER_DIR)
        router_module = _import_from_path("predict_router", ROUTER_DIR / "predict_router.py")
    finally:
        os.chdir(original_cwd)
    return router_module.predict
