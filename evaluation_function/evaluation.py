from __future__ import annotations

import os
import traceback
from typing import Any, List, Tuple, Optional
from urllib.parse import urlparse, unquote

import cv2
import numpy as np
import requests

from lf_toolkit.evaluation import Result, Params


def _pget(params: Params, key: str, default: Any) -> Any:
    try:
        return params.get(key, default)  # type: ignore
    except Exception:
        try:
            return params[key]  # type: ignore
        except Exception:
            return default


def _items_to_html(items: List[Tuple[Any, Any]]) -> str:
    lines: List[str] = []
    for k, v in items:
        k = "" if k is None else str(k).strip()
        v = "" if v is None else str(v).strip()
        lines.append(f"{k}: {v}" if k else v)
    return "<br>".join(lines)


def _escape_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _result(is_correct: bool, items: List[Tuple[str, str]]) -> Result:
    """Return Result in a version-tolerant way (keep your working pattern)."""
    html = _items_to_html(items)
    try:
        return Result(is_correct=is_correct, feedback=html, feedback_items=items)
    except TypeError:
        return Result(is_correct=is_correct, feedback_items=items)


def file_url_to_local_path(url: str) -> str:
    parsed = urlparse(url)
    path = unquote(parsed.path)
    if os.name == "nt" and len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return path


def _load_bgr_image_from_url(url: str, timeout: int = 15) -> Tuple[Optional[np.ndarray], Optional[str]]:
    try:
        if not isinstance(url, str) or not url:
            return None, "Empty URL."

        if url.startswith("file://"):
            local_path = file_url_to_local_path(url)
            img = cv2.imread(local_path, cv2.IMREAD_COLOR)
            if img is None:
                return None, f"cv2.imread failed for local_path='{local_path}'"
            return img, None

        if url.startswith("http://") or url.startswith("https://"):
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            data = np.frombuffer(resp.content, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                return None, "cv2.imdecode returned None (not a valid image)."
            return img, None

        return None, f"Unsupported URL scheme: {url}"
    except Exception as e:
        return None, str(e)


# ---------- NEW: ABCDE helpers (minimal & safe) ----------

def _candidate_model_paths() -> List[str]:
    """
    Matches your repo layout:
      evaluation_function/gear_model.pt
      evaluation_function/shaft_model.pt
    In container:
      /app/evaluation_function/gear_model.pt
      /app/evaluation_function/shaft_model.pt
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.join(here, "gear_model.pt"),
        os.path.join(here, "shaft_model.pt"),
        "/app/evaluation_function/gear_model.pt",
        "/app/evaluation_function/shaft_model.pt",
    ]


def evaluation_function(response: Any, answer: Any, params: Params) -> Result:
    """
    Smoke-test base + ABCDE diagnostics.
    Keeps unit-test behaviour:
      - empty response -> is_correct=False
      - bad url -> feedback contains LOAD_FAIL
      - local file:// url works
    Diagnostics via params["diag"] (or Params.diag):
      diag="torch" | "ultralytics" | "model_exists" | "load_model" | "infer_once"
    """
    items: List[Tuple[str, str]] = []

    try:
        fast_return: bool = bool(_pget(params, "fast_return", True))
        echo: bool = bool(_pget(params, "echo", True))
        try_fetch: bool = bool(_pget(params, "try_fetch", False))
        debug: bool = bool(_pget(params, "debug", True))
        skip_load_check: bool = bool(_pget(params, "skip_load_check", False))

        # NEW: diag switch
        diag: str = str(_pget(params, "diag", "none") or "none").strip().lower()

        items.append(("SMOKE", "Hello / evaluation_function reached ✅"))
        items.append(("diag", diag))
        items.append(("fast_return", str(fast_return)))
        items.append(("echo", str(echo)))
        items.append(("try_fetch", str(try_fetch)))
        items.append(("skip_load_check", str(skip_load_check)))
        items.append(("debug", str(debug)))

        # ---- A) torch import (no image needed, but we keep flow consistent) ----
        if diag == "torch":
            try:
                import torch  # noqa
                import torch
                items.append(("A_torch", "import OK"))
                items.append(("torch_version", str(torch.__version__)))
                items.append(("cuda_available", str(torch.cuda.is_available())))
                return _result(False, items)
            except Exception as e:
                items.append(("A_torch_FAIL", f"{type(e).__name__}: {e}"))
                items.append(("TRACEBACK", _escape_html(traceback.format_exc()).replace("\n", "<br>")))
                return _result(False, items)

        # ---- B) ultralytics import ----
        if diag == "ultralytics":
            try:
                from ultralytics import YOLO  # noqa: F401
                items.append(("B_ultralytics", "import OK"))
                return _result(False, items)
            except Exception as e:
                items.append(("B_ultralytics_FAIL", f"{type(e).__name__}: {e}"))
                items.append(("TRACEBACK", _escape_html(traceback.format_exc()).replace("\n", "<br>")))
                return _result(False, items)

        # ---- C) model existence ----
        if diag == "model_exists":
            paths = _candidate_model_paths()
            any_found = False
            for p in paths:
                if os.path.exists(p):
                    any_found = True
                    try:
                        sz = os.path.getsize(p)
                    except Exception:
                        sz = -1
                    items.append(("C_FOUND", f"{p} (size={sz})"))
            if not any_found:
                items.append(("C_FAIL", "No model files found in candidate paths"))
                items.append(("C_candidates", " | ".join(paths)))
            return _result(False, items)

        # ---- D/E require an image -> we continue below to load image first ----

        # 1) Validate input (unit test requirement)
        if not isinstance(response, list) or len(response) == 0:
            items.append(("BAD_INPUT", "No images uploaded."))
            return _result(False, items)

        # 2) Extract first URL
        first = response[0] if isinstance(response, list) and len(response) > 0 else None
        url = first.get("url") if isinstance(first, dict) else None

        if echo:
            items.append(("response_type", type(response).__name__))
            items.append(("response_len", str(len(response))))
            items.append(("first_item_type", type(first).__name__))
            if isinstance(first, dict):
                items.append(("first_item_keys", ", ".join(sorted([str(k) for k in first.keys()]))))
                items.append(("first_url", str(url)))

        if not url:
            items.append(("LOAD_FAIL", "LOAD_FAIL: first image has no url field"))
            return _result(False, items)

        # 3) Load-check (keep your existing CI-safe behaviour)
        #    D/E need an image anyway, so we must load here unless explicitly skipped.
        if (not skip_load_check) or try_fetch or diag in ("load_model", "infer_once"):
            img, err = _load_bgr_image_from_url(str(url))
            if img is None:
                items.append(("LOAD_FAIL", f"LOAD_FAIL: Failed to load image. ({err})"))
                items.append(("url", str(url)))
                return _result(False, items)

            h, w = img.shape[:2]
            items.append(("image_loaded", "OK"))
            items.append(("shape", f"{w}x{h}"))
        else:
            img = None  # type: ignore

        # ---- D) load model only (no inference) ----
        if diag == "load_model":
            try:
                from ultralytics import YOLO
                model_path = next((p for p in _candidate_model_paths() if os.path.exists(p)), None)
                if not model_path:
                    items.append(("D_FAIL", "No model file found to load"))
                    items.append(("D_candidates", " | ".join(_candidate_model_paths())))
                    return _result(False, items)
                items.append(("D_model_path", model_path))
                _ = YOLO(model_path)
                items.append(("D_load", "model loaded ✅"))
                return _result(False, items)
            except Exception as e:
                items.append(("D_FAIL", f"{type(e).__name__}: {e}"))
                items.append(("TRACEBACK", _escape_html(traceback.format_exc()).replace("\n", "<br>")))
                return _result(False, items)

        # ---- E) infer once (minimal) ----
        if diag == "infer_once":
            try:
                from ultralytics import YOLO
                model_path = next((p for p in _candidate_model_paths() if os.path.exists(p)), None)
                if not model_path:
                    items.append(("E_FAIL", "No model file found for inference"))
                    items.append(("E_candidates", " | ".join(_candidate_model_paths())))
                    return _result(False, items)

                if img is None:
                    items.append(("E_FAIL", "Image not loaded (img is None)"))
                    return _result(False, items)

                items.append(("E_model_path", model_path))
                model = YOLO(model_path)
                _ = model.predict(source=img, imgsz=640, conf=0.25, verbose=False)
                items.append(("E_infer", "predict done ✅"))
                return _result(False, items)
            except Exception as e:
                items.append(("E_FAIL", f"{type(e).__name__}: {e}"))
                items.append(("TRACEBACK", _escape_html(traceback.format_exc()).replace("\n", "<br>")))
                return _result(False, items)

        # 4) Optional early exit (keep your original platform smoke behaviour)
        if fast_return and not try_fetch:
            items.append(("note", "fast_return=True (no YOLO). Load-check already done."))
            return _result(False, items)

        # 5) Default end (still no YOLO in this build unless you add it)
        items.append(("note", "No YOLO executed in default path. Use diag=... to pinpoint failures."))
        return _result(False, items)

    except Exception as e:
        tb = _escape_html(traceback.format_exc())
        items.append(("UNHANDLED", f"{type(e).__name__}: {e}"))
        items.append(("TRACEBACK", tb.replace("\n", "<br>")))
        return _result(False, items)
