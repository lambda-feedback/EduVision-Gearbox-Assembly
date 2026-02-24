# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
import traceback
from typing import Any, List, Tuple, Optional, Callable
from urllib.parse import urlparse, unquote

import cv2
import numpy as np
import requests

from lf_toolkit.evaluation import Result, Params

# Lazy loading
from evaluation_function.lazy_load import LazyModule

torch = LazyModule("torch")
ultralytics = LazyModule("ultralytics")
_MODULE_IMPORT_T0 = time.perf_counter()

# ---- Small safety caps to avoid UI freeze ----
_MAX_ITEMS = int(os.environ.get("LF_MAX_ITEMS", "40"))
_MAX_TB_CHARS = int(os.environ.get("LF_MAX_TB_CHARS", "1200"))


def _pget(params: Params, key: str, default: Any) -> Any:
    try:
        return params.get(key, default)  # type: ignore
    except Exception:
        try:
            return params[key]  # type: ignore
        except Exception:
            return default


def _escape_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _tb_short() -> str:
    tb = traceback.format_exc()
    if len(tb) > _MAX_TB_CHARS:
        tb = tb[:_MAX_TB_CHARS] + "\n... (truncated)"
    return _escape_html(tb).replace("\n", "<br>")


def _items_to_html(items: List[Tuple[Any, Any]]) -> str:
    # cap output to avoid frontend freeze
    show = items[:_MAX_ITEMS]
    lines: List[str] = []
    for k, v in show:
        k = "" if k is None else str(k).strip()
        v = "" if v is None else str(v).strip()
        lines.append(f"{k}: {v}" if k else v)
    if len(items) > _MAX_ITEMS:
        lines.append(f"... (items truncated: showing {_MAX_ITEMS}/{len(items)})")
    return "<br>".join(lines)


def _result(is_correct: bool, items: List[Tuple[str, str]]) -> Result:
    """Return Result in a version-tolerant way (keep your working pattern)."""
    html = _items_to_html(items)
    safe_items = items[:_MAX_ITEMS]
    try:
        return Result(is_correct=is_correct, feedback=html, feedback_items=safe_items)
    except TypeError:
        return Result(is_correct=is_correct, feedback_items=safe_items)


def _timeit(fn: Callable[[], Any]) -> Tuple[Any, float]:
    """Measure wall time of fn() using perf_counter()."""
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    return out, dt


def _stage(items: List[Tuple[str, str]], name: str) -> None:
    # short stage marker for locating hang points
    items.append(("stage", name))


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
            # explicit (connect, read) timeouts reduce "endless wait"
            resp = requests.get(url, timeout=(5, timeout))
            resp.raise_for_status()
            data = np.frombuffer(resp.content, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                return None, "cv2.imdecode returned None (not a valid image)."
            return img, None

        return None, f"Unsupported URL scheme: {url}"
    except Exception as e:
        return None, str(e)


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


def _add_common_timing(items: List[Tuple[str, str]], t_handler0: float) -> None:
    now = time.perf_counter()
    items.append(("t_module_import_to_handler_s", f"{now - _MODULE_IMPORT_T0:.4f}"))
    items.append(("t_handler_elapsed_s", f"{now - t_handler0:.4f}"))


def evaluation_function(response: Any, answer: Any, params: Params) -> Result:
    """
    Minimal diagnostics, minimal changes:
      - keeps original tests behaviour: bad url => LOAD_FAIL
      - adds stage markers and truncates output to avoid UI freeze
    diag:
      "ping" | "mem" | "torch_min" | "ultra_min" | "torch" | "ultralytics"
      "model_exists" | "load_model" | "infer_once"
    """
    items: List[Tuple[str, str]] = []
    t_handler0 = time.perf_counter()

    try:
        fast_return: bool = bool(_pget(params, "fast_return", True))
        echo: bool = bool(_pget(params, "echo", True))
        try_fetch: bool = bool(_pget(params, "try_fetch", False))
        debug: bool = bool(_pget(params, "debug", True))
        skip_load_check: bool = bool(_pget(params, "skip_load_check", False))

        diag: str = str(_pget(params, "diag", "none") or "none").strip().lower()

        items.append(("SMOKE", "Hello / evaluation_function reached ✅"))
        items.append(("diag", diag))
        items.append(("fast_return", str(fast_return)))
        items.append(("echo", str(echo)))
        items.append(("try_fetch", str(try_fetch)))
        items.append(("skip_load_check", str(skip_load_check)))
        items.append(("debug", str(debug)))

        _add_common_timing(items, t_handler0)
        _stage(items, "entered")

        # ----------------------------
        # ping
        # ----------------------------
        if diag == "ping":
            _stage(items, "ping_ok")
            _add_common_timing(items, t_handler0)
            return _result(False, items)

        # ----------------------------
        # mem
        # ----------------------------
        if diag == "mem":
            try:
                import platform
                import resource

                items.append(("platform", platform.platform()))
                items.append(("python_version", platform.python_version()))
                items.append(("pid", str(os.getpid())))

                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                items.append(("ru_maxrss_kb", str(rss)))
                items.append(("ru_maxrss_mb_est", f"{(float(rss) / 1024.0):.2f}"))

                _stage(items, "mem_ok")
                _add_common_timing(items, t_handler0)
                return _result(False, items)
            except Exception:
                items.append(("MEM_FAIL", "see TRACEBACK"))
                items.append(("TRACEBACK", _tb_short()))
                _add_common_timing(items, t_handler0)
                return _result(False, items)

        # ----------------------------
        # torch minimal import
        # ----------------------------
        if diag == "torch_min":
            try:
                _stage(items, "torch_min_begin")
                _, dt = _timeit(lambda: __import__("torch"))
                items.append(("t_torch_import_s", f"{dt:.4f}"))
                _stage(items, "torch_min_done")
                _add_common_timing(items, t_handler0)
                return _result(False, items)
            except Exception:
                items.append(("A0_torch_min_FAIL", "see TRACEBACK"))
                items.append(("TRACEBACK", _tb_short()))
                _add_common_timing(items, t_handler0)
                return _result(False, items)

        # ----------------------------
        # ultralytics minimal import
        # ----------------------------
        if diag == "ultra_min":
            try:
                _stage(items, "ultra_min_begin")
                _, dt = _timeit(lambda: __import__("ultralytics"))
                items.append(("t_ultralytics_import_s", f"{dt:.4f}"))
                _stage(items, "ultra_min_done")
                _add_common_timing(items, t_handler0)
                return _result(False, items)
            except Exception:
                items.append(("B0_ultra_min_FAIL", "see TRACEBACK"))
                items.append(("TRACEBACK", _tb_short()))
                _add_common_timing(items, t_handler0)
                return _result(False, items)

        # ----------------------------
        # torch lazy import
        # ----------------------------
        if diag == "torch":
            try:
                _stage(items, "torch_lazy_begin")
                _, dt = _timeit(lambda: torch.__version__)
                items.append(("t_torch_import_s", f"{dt:.4f}"))
                items.append(("torch_version", str(torch.__version__)))
                items.append(("cuda_check", "skipped (CPU-only build)"))
                _stage(items, "torch_lazy_done")
                _add_common_timing(items, t_handler0)
                return _result(False, items)
            except Exception:
                items.append(("A_torch_FAIL", "see TRACEBACK"))
                items.append(("TRACEBACK", _tb_short()))
                _add_common_timing(items, t_handler0)
                return _result(False, items)

        # ----------------------------
        # ultralytics lazy import
        # ----------------------------
        if diag == "ultralytics":
            try:
                _stage(items, "ultra_lazy_begin")
                YOLO, dt = _timeit(lambda: ultralytics.YOLO)
                items.append(("t_ultralytics_symbol_s", f"{dt:.4f}"))
                items.append(("YOLO_symbol", str(YOLO)))
                _stage(items, "ultra_lazy_done")
                _add_common_timing(items, t_handler0)
                return _result(False, items)
            except Exception:
                items.append(("B_ultralytics_FAIL", "see TRACEBACK"))
                items.append(("TRACEBACK", _tb_short()))
                _add_common_timing(items, t_handler0)
                return _result(False, items)

        # ----------------------------
        # model existence
        # ----------------------------
        if diag == "model_exists":
            _stage(items, "model_exists_begin")
            paths = _candidate_model_paths()
            any_found = False
            for p in paths:
                if os.path.exists(p):
                    any_found = True
                    try:
                        sz = os.path.getsize(p)
                    except Exception:
                        sz = -1
                    items.append(("FOUND", f"{p} (size={sz})"))
            if not any_found:
                items.append(("C_FAIL", "No model files found in candidate paths"))
                items.append(("C_candidates", " | ".join(paths)))
            _stage(items, "model_exists_done")
            _add_common_timing(items, t_handler0)
            return _result(False, items)

        # ----------------------------
        # From here: need response/url to do load-check or infer.
        # Keep original CI behaviour: bad URL => LOAD_FAIL.
        # ----------------------------
        if not isinstance(response, list) or len(response) == 0:
            items.append(("BAD_INPUT", "No images uploaded."))
            _add_common_timing(items, t_handler0)
            return _result(False, items)

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
            _add_common_timing(items, t_handler0)
            return _result(False, items)

        # ----------------------------
        # Load-check (minimal change): if skip_load_check=False, always validate url.
        # This preserves your original unit tests.
        # If you want to skip in the UI, set skip_load_check=True.
        # ----------------------------
        img: Optional[np.ndarray] = None
        if (not skip_load_check) or try_fetch or diag in ("load_model", "infer_once"):
            try:
                _stage(items, "image_load_begin")
                (img, err), dt_img = _timeit(lambda: _load_bgr_image_from_url(str(url)))
                items.append(("t_image_load_s", f"{dt_img:.4f}"))
                if img is None:
                    items.append(("LOAD_FAIL", f"LOAD_FAIL: Failed to load image. ({err})"))
                    items.append(("url", str(url)))
                    _add_common_timing(items, t_handler0)
                    return _result(False, items)

                h, w = img.shape[:2]
                items.append(("image_loaded", "OK"))
                items.append(("shape", f"{w}x{h}"))
                _stage(items, "image_load_done")
            except Exception:
                items.append(("LOAD_FAIL", "LOAD_FAIL: Exception during image load"))
                items.append(("TRACEBACK", _tb_short()))
                _add_common_timing(items, t_handler0)
                return _result(False, items)

        # ----------------------------
        # D) load model only (no inference)
        # ----------------------------
        if diag == "load_model":
            try:
                _stage(items, "ultralytics_symbol_begin")
                YOLO, dt_ul = _timeit(lambda: ultralytics.YOLO)
                items.append(("t_ultralytics_symbol_s", f"{dt_ul:.4f}"))
                _stage(items, "ultralytics_symbol_done")

                model_path = next((p for p in _candidate_model_paths() if os.path.exists(p)), None)
                if not model_path:
                    items.append(("D_FAIL", "No model file found to load"))
                    items.append(("D_candidates", " | ".join(_candidate_model_paths())))
                    _add_common_timing(items, t_handler0)
                    return _result(False, items)

                items.append(("D_model_path", model_path))

                # Optional: limit threads (small + safe)
                try:
                    _ = torch.__version__
                    torch.set_num_threads(1)
                    torch.set_num_interop_threads(1)
                    items.append(("torch_threads", "set to 1"))
                except Exception:
                    pass

                _stage(items, "yolo_load_begin")
                _, dt_load = _timeit(lambda: YOLO(model_path))
                items.append(("t_model_load_s", f"{dt_load:.4f}"))
                _stage(items, "yolo_load_done")
                items.append(("D_load", "model loaded ✅"))

                _add_common_timing(items, t_handler0)
                return _result(False, items)
            except Exception:
                items.append(("D_FAIL", "see TRACEBACK"))
                items.append(("TRACEBACK", _tb_short()))
                _add_common_timing(items, t_handler0)
                return _result(False, items)

        # ----------------------------
        # E) infer once (minimal)
        # ----------------------------
        if diag == "infer_once":
            try:
                _stage(items, "ultralytics_symbol_begin")
                YOLO, dt_ul = _timeit(lambda: ultralytics.YOLO)
                items.append(("t_ultralytics_symbol_s", f"{dt_ul:.4f}"))
                _stage(items, "ultralytics_symbol_done")

                model_path = next((p for p in _candidate_model_paths() if os.path.exists(p)), None)
                if not model_path:
                    items.append(("E_FAIL", "No model file found for inference"))
                    items.append(("E_candidates", " | ".join(_candidate_model_paths())))
                    _add_common_timing(items, t_handler0)
                    return _result(False, items)

                if img is None:
                    items.append(("E_FAIL", "Image not loaded (img is None)"))
                    _add_common_timing(items, t_handler0)
                    return _result(False, items)

                items.append(("E_model_path", model_path))

                try:
                    _ = torch.__version__
                    torch.set_num_threads(1)
                    torch.set_num_interop_threads(1)
                    items.append(("torch_threads", "set to 1"))
                except Exception:
                    pass

                _stage(items, "yolo_load_begin")
                model, dt_load = _timeit(lambda: YOLO(model_path))
                items.append(("t_model_load_s", f"{dt_load:.4f}"))
                _stage(items, "yolo_load_done")

                _stage(items, "predict_begin")
                _, dt_pred = _timeit(lambda: model.predict(source=img, imgsz=640, conf=0.25, device="cpu", verbose=False))
                items.append(("t_predict_s", f"{dt_pred:.4f}"))
                _stage(items, "predict_done")

                items.append(("E_infer", "predict done ✅"))
                _add_common_timing(items, t_handler0)
                return _result(False, items)
            except Exception:
                items.append(("E_FAIL", "see TRACEBACK"))
                items.append(("TRACEBACK", _tb_short()))
                _add_common_timing(items, t_handler0)
                return _result(False, items)

        # ----------------------------
        # Optional early exit (preserve your behaviour)
        # NOTE: By the time we get here, load-check has already run (unless skip_load_check=True).
        # ----------------------------
        if fast_return and not try_fetch:
            items.append(("note", "fast_return=True (no YOLO). Load-check already done unless skip_load_check=True."))
            _add_common_timing(items, t_handler0)
            return _result(False, items)

        items.append(("note", "No YOLO executed in default path. Use diag=... to pinpoint failures."))
        _add_common_timing(items, t_handler0)
        return _result(False, items)

    except Exception as e:
        items.append(("UNHANDLED", f"{type(e).__name__}: {e}"))
        items.append(("TRACEBACK", _tb_short()))
        _add_common_timing(items, t_handler0)
        return _result(False, items)