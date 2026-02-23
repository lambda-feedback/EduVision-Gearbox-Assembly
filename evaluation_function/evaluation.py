from __future__ import annotations

import os
import json
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


def _timeit(fn: Callable[[], Any]) -> Tuple[Any, float]:
    """Measure wall time of fn() using perf_counter()."""
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    return out, dt


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


# ABCDE helpers (minimal & safe)
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
    """Append common timing fields."""
    now = time.perf_counter()
    items.append(("t_module_import_to_handler_s", f"{now - _MODULE_IMPORT_T0:.4f}"))
    items.append(("t_handler_elapsed_s", f"{now - t_handler0:.4f}"))


def evaluation_function(response: Any, answer: Any, params: Params) -> Result:
    """
    Smoke-test base + ABCDE diagnostics.
    Keeps unit-test behaviour:
      - empty response -> is_correct=False
      - bad url -> feedback contains LOAD_FAIL
      - local file:// url works
    Diagnostics via params["diag"] (or Params.diag):
      diag="ping" | "torch" | "ultralytics" | "model_exists" | "load_model" | "infer_once"
    """
    items: List[Tuple[str, str]] = []
    t_handler0 = time.perf_counter()

    try:
        fast_return: bool = bool(_pget(params, "fast_return", True))
        echo: bool = bool(_pget(params, "echo", True))
        try_fetch: bool = bool(_pget(params, "try_fetch", False))
        debug: bool = bool(_pget(params, "debug", True))
        skip_load_check: bool = bool(_pget(params, "skip_load_check", False))

        # diag switch
        diag: str = str(_pget(params, "diag", "none") or "none").strip().lower()

        items.append(("SMOKE", "Hello / evaluation_function reached ✅"))
        items.append(("diag", diag))
        items.append(("fast_return", str(fast_return)))
        items.append(("echo", str(echo)))
        items.append(("try_fetch", str(try_fetch)))
        items.append(("skip_load_check", str(skip_load_check)))
        items.append(("debug", str(debug)))

        # Always include a cold-start proxy timing snapshot early
        _add_common_timing(items, t_handler0)

        # ----------------------------
        # P) ping: absolute minimum path (no image, no torch/ultralytics)
        # ----------------------------
        if diag == "ping":
            items.append(("PING", "OK ✅"))
            _add_common_timing(items, t_handler0)
            return _result(False, items)

        # MEM) Environment & memory diagnostics (no torch/ultralytics)
        # ----------------------------
        if diag == "mem":
            try:
                import platform
                import resource
                import sys

                items.append(("MEM", "env/memory diagnostics"))
                items.append(("platform", platform.platform()))
                items.append(("python_version", platform.python_version()))
                items.append(("python_implementation", platform.python_implementation()))
                items.append(("pid", str(os.getpid())))

                # sys.path (truncate to avoid huge output)
                try:
                    sp = "\n".join(sys.path[:20])
                    items.append(("sys_path_head", _escape_html(sp).replace("\n", "<br>")))
                except Exception as e:
                    items.append(("sys_path_head_FAIL", f"{type(e).__name__}: {e}"))

                # ru_maxrss: max resident set size so far
                # On Linux: typically KB; on macOS: bytes. Platform is Linux here.
                try:
                    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    items.append(("ru_maxrss_raw", str(rss)))
                    # Best-effort human-friendly conversion assuming Linux KB.
                    try:
                        rss_mb = float(rss) / 1024.0
                        items.append(("ru_maxrss_mb_est", f"{rss_mb:.2f}"))
                    except Exception:
                        pass
                except Exception as e:
                    items.append(("ru_maxrss_FAIL", f"{type(e).__name__}: {e}"))

                # cgroup memory limits / usage
                # We ALWAYS emit FOUND/NOT_FOUND for each path.
                cgroup_files = [
                    # cgroup v2 common files
                    "/sys/fs/cgroup/memory.max",
                    "/sys/fs/cgroup/memory.high",
                    "/sys/fs/cgroup/memory.current",
                    "/sys/fs/cgroup/memory.swap.max",
                    "/sys/fs/cgroup/cpu.max",
                    # cgroup v1 common files
                    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
                    "/sys/fs/cgroup/memory/memory.soft_limit_in_bytes",
                    "/sys/fs/cgroup/memory/memory.usage_in_bytes",
                    "/sys/fs/cgroup/memory/memory.max_usage_in_bytes",
                ]

                for p in cgroup_files:
                    key = f"cgroup:{os.path.basename(p)}"
                    if os.path.exists(p):
                        try:
                            with open(p, "r", encoding="utf-8") as f:
                                val = f.read().strip()
                            items.append((key, val))
                        except Exception as e:
                            items.append((key + "_READ_FAIL", f"{type(e).__name__}: {e}"))
                    else:
                        items.append((key, "NOT_FOUND"))

                _add_common_timing(items, t_handler0)
                return _result(False, items)

            except Exception as e:
                items.append(("MEM_FAIL", f"{type(e).__name__}: {e}"))
                items.append(("TRACEBACK", _escape_html(traceback.format_exc()).replace("\n", "<br>")))
                _add_common_timing(items, t_handler0)
                return _result(False, items)

        # ----------------------------
        # A0) torch minimal import probe (as light as possible)
        # ----------------------------
        if diag == "torch_min":
            try:
                # Minimal: import only (avoid extra probing)
                _, dt = _timeit(lambda: __import__("torch"))
                items.append(("A0_torch_min", "import OK"))
                items.append(("t_torch_import_s", f"{dt:.4f}"))
                _add_common_timing(items, t_handler0)
                return _result(False, items)
            except Exception as e:
                items.append(("A0_torch_min_FAIL", f"{type(e).__name__}: {e}"))
                items.append(("TRACEBACK", _escape_html(traceback.format_exc()).replace("\n", "<br>")))
                _add_common_timing(items, t_handler0)
                return _result(False, items)

        # ----------------------------
        # B0) ultralytics minimal import probe (import package only, don't touch YOLO)
        # ----------------------------
        if diag == "ultra_min":
            try:
                _, dt = _timeit(lambda: __import__("ultralytics"))
                items.append(("B0_ultra_min", "import OK"))
                items.append(("t_ultralytics_import_s", f"{dt:.4f}"))
                _add_common_timing(items, t_handler0)
                return _result(False, items)
            except Exception as e:
                items.append(("B0_ultra_min_FAIL", f"{type(e).__name__}: {e}"))
                items.append(("TRACEBACK", _escape_html(traceback.format_exc()).replace("\n", "<br>")))
                _add_common_timing(items, t_handler0)
                return _result(False, items)

        if diag == "alloc":
            try:
                step_mb = int(_pget(params, "alloc_step_mb", 64))
                max_mb = int(_pget(params, "alloc_max_mb", 1024))
                chunks = []
                allocated = 0
                while allocated + step_mb <= max_mb:
                    chunks.append(bytearray(step_mb * 1024 * 1024))
                    allocated += step_mb
                    items.append(("alloc_mb", str(allocated)))
                    _add_common_timing(items, t_handler0)
                items.append(("ALLOC_DONE", f"{allocated}MB"))
                _add_common_timing(items, t_handler0)
                return _result(False, items)
            except Exception as e:
                items.append(("ALLOC_FAIL", f"{type(e).__name__}: {e}"))
                items.append(("TRACEBACK", _escape_html(traceback.format_exc()).replace("\n", "<br>")))
                _add_common_timing(items, t_handler0)
                return _result(False, items)

        # ----------------------------
        # A) torch lazy import timing (CPU-only friendly)
        # ----------------------------
        if diag == "torch":
            try:
                # Trigger actual import by touching a torch attribute
                _, dt = _timeit(lambda: torch.__version__)
                items.append(("A_torch", "lazy import OK"))
                items.append(("t_torch_import_s", f"{dt:.4f}"))
                items.append(("torch_version", str(torch.__version__)))

                # NOTE: removed torch.cuda.is_available() on purpose (CPU-only wheel)
                items.append(("cuda_check", "skipped (CPU-only build)"))

                _add_common_timing(items, t_handler0)
                return _result(False, items)
            except Exception as e:
                items.append(("A_torch_FAIL", f"{type(e).__name__}: {e}"))
                items.append(("TRACEBACK", _escape_html(traceback.format_exc()).replace("\n", "<br>")))
                _add_common_timing(items, t_handler0)
                return _result(False, items)

        # ----------------------------
        # B) ultralytics lazy import timing
        # ----------------------------
        if diag == "ultralytics":
            try:
                # Trigger ultralytics import by accessing YOLO symbol
                YOLO, dt = _timeit(lambda: ultralytics.YOLO)
                items.append(("B_ultralytics", "lazy import OK"))
                items.append(("t_ultralytics_import_s", f"{dt:.4f}"))
                items.append(("YOLO_symbol", str(YOLO)))
                _add_common_timing(items, t_handler0)
                return _result(False, items)
            except Exception as e:
                items.append(("B_ultralytics_FAIL", f"{type(e).__name__}: {e}"))
                items.append(("TRACEBACK", _escape_html(traceback.format_exc()).replace("\n", "<br>")))
                _add_common_timing(items, t_handler0)
                return _result(False, items)

        # ----------------------------
        # C) model existence
        # ----------------------------
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
            _add_common_timing(items, t_handler0)
            return _result(False, items)

        # D/E require an image -> continue to load image first

        # 1) Validate input (unit test requirement)
        if not isinstance(response, list) or len(response) == 0:
            items.append(("BAD_INPUT", "No images uploaded."))
            _add_common_timing(items, t_handler0)
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
            _add_common_timing(items, t_handler0)
            return _result(False, items)

        # 3) Load-check (keep your existing CI-safe behaviour)
        if (not skip_load_check) or try_fetch or diag in ("load_model", "infer_once"):
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
        else:
            img = None  # type: ignore

        # D) load model only (no inference)
        if diag == "load_model":
            try:
                YOLO, dt_ul = _timeit(lambda: ultralytics.YOLO)
                items.append(("t_ultralytics_import_s", f"{dt_ul:.4f}"))

                model_path = next((p for p in _candidate_model_paths() if os.path.exists(p)), None)
                if not model_path:
                    items.append(("D_FAIL", "No model file found to load"))
                    items.append(("D_candidates", " | ".join(_candidate_model_paths())))
                    _add_common_timing(items, t_handler0)
                    return _result(False, items)

                items.append(("D_model_path", model_path))

                _, dt_load = _timeit(lambda: YOLO(model_path))
                items.append(("t_model_load_s", f"{dt_load:.4f}"))
                items.append(("D_load", "model loaded ✅"))

                _add_common_timing(items, t_handler0)
                return _result(False, items)
            except Exception as e:
                items.append(("D_FAIL", f"{type(e).__name__}: {e}"))
                items.append(("TRACEBACK", _escape_html(traceback.format_exc()).replace("\n", "<br>")))
                _add_common_timing(items, t_handler0)
                return _result(False, items)

        # E) infer once (minimal)
        if diag == "infer_once":
            try:
                YOLO, dt_ul = _timeit(lambda: ultralytics.YOLO)
                items.append(("t_ultralytics_import_s", f"{dt_ul:.4f}"))

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

                model, dt_load = _timeit(lambda: YOLO(model_path))
                items.append(("t_model_load_s", f"{dt_load:.4f}"))

                _, dt_pred = _timeit(lambda: model.predict(source=img, imgsz=640, conf=0.25, verbose=False))
                items.append(("t_predict_s", f"{dt_pred:.4f}"))

                items.append(("E_infer", "predict done ✅"))
                _add_common_timing(items, t_handler0)
                return _result(False, items)
            except Exception as e:
                items.append(("E_FAIL", f"{type(e).__name__}: {e}"))
                items.append(("TRACEBACK", _escape_html(traceback.format_exc()).replace("\n", "<br>")))
                _add_common_timing(items, t_handler0)
                return _result(False, items)

        # 4) Optional early exit (keep your original platform smoke behaviour)
        if fast_return and not try_fetch:
            items.append(("note", "fast_return=True (no YOLO). Load-check already done."))
            _add_common_timing(items, t_handler0)
            return _result(False, items)

        # 5) Default end
        items.append(("note", "No YOLO executed in default path. Use diag=... to pinpoint failures."))
        _add_common_timing(items, t_handler0)
        return _result(False, items)

    except Exception as e:
        tb = _escape_html(traceback.format_exc())
        items.append(("UNHANDLED", f"{type(e).__name__}: {e}"))
        items.append(("TRACEBACK", tb.replace("\n", "<br>")))
        _add_common_timing(items, t_handler0)
        return _result(False, items)