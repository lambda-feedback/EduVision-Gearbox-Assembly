from __future__ import annotations

import os
import sys
import time
import traceback
import faulthandler
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

# ---- Old caps (kept, but UI output is now minimal by default) ----
_MAX_ITEMS = int(os.environ.get("LF_MAX_ITEMS", "40"))
_MAX_TB_CHARS = int(os.environ.get("LF_MAX_TB_CHARS", "1200"))

# ---- NEW: hard cap for feedback line ----
_MAX_FEEDBACK_CHARS = int(os.environ.get("LF_MAX_FEEDBACK_CHARS", "160"))
# If true, force minimal response even if params don't request it
_MINIMAL_DEFAULT = os.environ.get("LF_UI_MINIMAL_DEFAULT", "1").strip().lower() not in ("0", "false", "no")


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
    # cap output to avoid frontend freeze (kept for completeness)
    show = items[:_MAX_ITEMS]
    lines: List[str] = []
    for k, v in show:
        k = "" if k is None else str(k).strip()
        v = "" if v is None else str(v).strip()
        lines.append(f"{k}: {v}" if k else v)
    if len(items) > _MAX_ITEMS:
        lines.append(f"... (items truncated: showing {_MAX_ITEMS}/{len(items)})")
    return "<br>".join(lines)


def _short_feedback(is_correct: bool, items: List[Tuple[Any, Any]]) -> str:
    """
    Build a tiny single-line message for UI.
    Priority: explicit FAIL/TIMEOUT/UNHANDLED keys (from the end), else OK/FAIL.
    """
    if is_correct:
        return "OK"

    # Scan from the end for useful error markers
    for k, v in reversed(items):
        ks = ("" if k is None else str(k)).strip()
        vs = ("" if v is None else str(v)).strip()
        upper_k = ks.upper()

        if (
            "FAIL" in upper_k
            or "TIMEOUT" in upper_k
            or upper_k in ("UNHANDLED", "ERROR", "EXCEPTION")
        ):
            msg = ks if not vs else f"{ks}: {vs}"
            msg = msg.replace("\n", " ").strip()
            if len(msg) > _MAX_FEEDBACK_CHARS:
                msg = msg[:_MAX_FEEDBACK_CHARS] + "..."
            return msg

    # fallback if nothing obvious
    return "FAIL"


def _result(
    is_correct: bool,
    items: List[Tuple[str, str]],
    *,
    ui_minimum: bool = False,
    no_items: bool = False,
) -> Result:
    """
    Always return a tiny payload to avoid UI freeze.
    - feedback: single short line
    - feedback_items: omitted
    """
    if _MINIMAL_DEFAULT:
        ui_minimum = True
        no_items = True

    # Always minimal line
    msg = _short_feedback(is_correct, items)

    try:
        return Result(is_correct=is_correct, feedback=msg)
    except TypeError:
        # older toolkit versions: feedback may not be supported
        return Result(is_correct=is_correct)


def _timeit(fn: Callable[[], Any]) -> Tuple[Any, float]:
    """Measure wall time of fn() using perf_counter()."""
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    return out, dt


def _stage(items: List[Tuple[str, str]], name: str) -> None:
    # short stage marker for locating hang points
    items.append(("stage", name))


# ----------------------------
# Watchdog for hangs (CloudWatch-friendly)
# ----------------------------
def _watchdog_start(seconds: int = 6) -> None:
    """
    Dump traceback of ALL threads to stderr after `seconds`.
    In AWS Lambda, stderr goes to CloudWatch, so even if the function times out,
    you'll still see where it was stuck.
    """
    try:
        faulthandler.enable(all_threads=True, file=sys.stderr)
        faulthandler.dump_traceback_later(seconds, repeat=True, file=sys.stderr, exit=False)
    except Exception:
        pass


def _watchdog_stop() -> None:
    try:
        faulthandler.cancel_dump_traceback_later()
    except Exception:
        pass


def _torch_load_worker(model_path: str, q) -> None:
    """
    Run torch.load in a child process so the parent can enforce a timeout.
    NOTE: Import torch INSIDE the worker to avoid increasing cold-start time.
    """
    import time as _t
    t0 = _t.perf_counter()
    try:
        import torch as _torch

        try:
            _torch.set_num_threads(1)
            _torch.set_num_interop_threads(1)
        except Exception:
            pass

        try:
            obj = _torch.load(model_path, map_location="cpu", weights_only=True)
        except TypeError:
            obj = _torch.load(model_path, map_location="cpu")

        dt = _t.perf_counter() - t0
        q.put(("OK", dt, type(obj).__name__))
    except Exception as e:
        dt = _t.perf_counter() - t0
        q.put(("ERR", dt, f"{type(e).__name__}: {e}"))


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
    Minimal UI payload (single line), keep diagnostics and stage markers.
    diag:
      "ping" | "mem" | "torch_min" | "ultra_min" | "torch" | "ultralytics"
      "model_exists" | "stat_model" | "read_head" | "torch_load_only"
      "load_model_only" | "load_model" | "infer_once"
    """
    items: List[Tuple[str, str]] = []
    t_handler0 = time.perf_counter()

    # Defaults switched to True to prevent UI freeze even if caller doesn't pass params
    ui_minimum: bool = bool(_pget(params, "ui_minimum", True))
    no_items: bool = bool(_pget(params, "no_items", True))

    try:
        fast_return: bool = bool(_pget(params, "fast_return", True))
        echo: bool = bool(_pget(params, "echo", False))          # default False now
        try_fetch: bool = bool(_pget(params, "try_fetch", False))
        debug: bool = bool(_pget(params, "debug", True))
        skip_load_check: bool = bool(_pget(params, "skip_load_check", False))

        diag: str = str(_pget(params, "diag", "none") or "none").strip().lower()

        # Keep a couple of tiny markers (safe)
        items.append(("diag", diag))
        items.append(("debug", str(debug)))
        _add_common_timing(items, t_handler0)
        _stage(items, "entered")

        # ----------------------------
        # ping
        # ----------------------------
        if diag == "ping":
            _stage(items, "ping_ok")
            _add_common_timing(items, t_handler0)
            return _result(True, items, ui_minimum=ui_minimum, no_items=no_items)

        # ----------------------------
        # Sleep diagnostic (time limit test)
        # ----------------------------
        if diag == "sleep":
            try:
                t = float(_pget(params, "t", 5.0))
                _stage(items, "sleep_begin")
                t0 = time.perf_counter()
                time.sleep(t)
                dt = time.perf_counter() - t0
                items.append(("sleep_actual_s", f"{dt:.4f}"))
                _stage(items, "sleep_done")
                return _result(True, items, ui_minimum=ui_minimum, no_items=no_items)
            except Exception:
                items.append(("SLEEP_FAIL", "see TRACEBACK"))
                items.append(("TRACEBACK", _tb_short()))
                _add_common_timing(items, t_handler0)
                return _result(False, items, ui_minimum=ui_minimum, no_items=no_items)

        # ----------------------------
        # mem
        # ----------------------------
        if diag == "mem":
            try:
                import platform
                import resource

                # keep it short
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                items.append(("platform", platform.system()))
                items.append(("python", platform.python_version()))
                items.append(("ru_maxrss_kb", str(rss)))
                _stage(items, "mem_ok")
                _add_common_timing(items, t_handler0)
                return _result(True, items, ui_minimum=ui_minimum, no_items=no_items)
            except Exception:
                items.append(("MEM_FAIL", "see TRACEBACK"))
                items.append(("TRACEBACK", _tb_short()))
                _add_common_timing(items, t_handler0)
                return _result(False, items, ui_minimum=ui_minimum, no_items=no_items)

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
                return _result(True, items, ui_minimum=ui_minimum, no_items=no_items)
            except Exception:
                items.append(("A0_torch_min_FAIL", "see TRACEBACK"))
                items.append(("TRACEBACK", _tb_short()))
                _add_common_timing(items, t_handler0)
                return _result(False, items, ui_minimum=ui_minimum, no_items=no_items)

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
                return _result(True, items, ui_minimum=ui_minimum, no_items=no_items)
            except Exception:
                items.append(("B0_ultra_min_FAIL", "see TRACEBACK"))
                items.append(("TRACEBACK", _tb_short()))
                _add_common_timing(items, t_handler0)
                return _result(False, items, ui_minimum=ui_minimum, no_items=no_items)

        # ----------------------------
        # torch lazy import
        # ----------------------------
        if diag == "torch":
            try:
                _stage(items, "torch_lazy_begin")
                _, dt = _timeit(lambda: torch.__version__)
                items.append(("t_torch_import_s", f"{dt:.4f}"))
                _stage(items, "torch_lazy_done")
                _add_common_timing(items, t_handler0)
                return _result(True, items, ui_minimum=ui_minimum, no_items=no_items)
            except Exception:
                items.append(("A_torch_FAIL", "see TRACEBACK"))
                items.append(("TRACEBACK", _tb_short()))
                _add_common_timing(items, t_handler0)
                return _result(False, items, ui_minimum=ui_minimum, no_items=no_items)

        # ----------------------------
        # ultralytics lazy import
        # ----------------------------
        if diag == "ultralytics":
            try:
                _stage(items, "ultra_lazy_begin")
                YOLO, dt = _timeit(lambda: ultralytics.YOLO)
                _ = YOLO  # keep lint quiet
                items.append(("t_ultralytics_symbol_s", f"{dt:.4f}"))
                _stage(items, "ultra_lazy_done")
                _add_common_timing(items, t_handler0)
                return _result(True, items, ui_minimum=ui_minimum, no_items=no_items)
            except Exception:
                items.append(("B_ultralytics_FAIL", "see TRACEBACK"))
                items.append(("TRACEBACK", _tb_short()))
                _add_common_timing(items, t_handler0)
                return _result(False, items, ui_minimum=ui_minimum, no_items=no_items)

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
                    items.append(("FOUND", os.path.basename(p)))
            if not any_found:
                items.append(("C_FAIL", "No model files found"))
            _stage(items, "model_exists_done")
            _add_common_timing(items, t_handler0)
            return _result(any_found, items, ui_minimum=ui_minimum, no_items=no_items)

        # ----------------------------
        # model stat (size/mtime)
        # ----------------------------
        if diag == "stat_model":
            _stage(items, "stat_model_begin")
            paths = _candidate_model_paths()
            found = [p for p in paths if os.path.exists(p)]
            if not found:
                items.append(("S_FAIL", "No model file found"))
                _add_common_timing(items, t_handler0)
                return _result(False, items, ui_minimum=ui_minimum, no_items=no_items)

            p = found[0]
            st = os.stat(p)
            items.append(("size_bytes", str(st.st_size)))
            items.append(("mtime", str(st.st_mtime)))
            _stage(items, "stat_model_done")
            _add_common_timing(items, t_handler0)
            return _result(True, items, ui_minimum=ui_minimum, no_items=no_items)

        # ----------------------------
        # read first N bytes (pure file I/O speed)
        # ----------------------------
        if diag == "read_head":
            _stage(items, "read_head_begin")
            paths = _candidate_model_paths()
            found = [p for p in paths if os.path.exists(p)]
            if not found:
                items.append(("R_FAIL", "No model file found"))
                _add_common_timing(items, t_handler0)
                return _result(False, items, ui_minimum=ui_minimum, no_items=no_items)

            p = found[0]

            def _read():
                with open(p, "rb") as f:
                    return f.read(1024 * 1024)  # 1MB

            data, dt = _timeit(_read)
            items.append(("t_read_1mb_s", f"{dt:.4f}"))
            items.append(("read_len", str(len(data))))
            _stage(items, "read_head_done")
            _add_common_timing(items, t_handler0)
            return _result(True, items, ui_minimum=ui_minimum, no_items=no_items)

        # ----------------------------
        # subprocess smoke
        # ----------------------------
        if diag == "subprocess_smoke":
            try:
                import subprocess
                _stage(items, "subprocess_smoke_begin")
                t0 = time.perf_counter()
                cp = subprocess.run(
                    [sys.executable, "-c", "print('OK')"],
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                )
                items.append(("rc", str(cp.returncode)))
                dt = time.perf_counter() - t0
                items.append(("t_smoke_s", f"{dt:.4f}"))
                _stage(items, "subprocess_smoke_done")
                _add_common_timing(items, t_handler0)
                return _result(cp.returncode == 0, items, ui_minimum=ui_minimum, no_items=no_items)
            except Exception:
                items.append(("SUBPROC_FAIL", "see TRACEBACK"))
                items.append(("TRACEBACK", _tb_short()))
                _add_common_timing(items, t_handler0)
                return _result(False, items, ui_minimum=ui_minimum, no_items=no_items)

        # ----------------------------
        # torch.load only (bypass ultralytics) WITH HARD TIMEOUT (robust)
        # ----------------------------
        if diag == "torch_load_only":
            try:
                _stage(items, "torch_load_only_begin")

                paths = _candidate_model_paths()
                found = [p for p in paths if os.path.exists(p)]
                if not found:
                    items.append(("T_FAIL", "No model file found"))
                    _add_common_timing(items, t_handler0)
                    return _result(False, items, ui_minimum=ui_minimum, no_items=no_items)

                p = found[0]

                load_timeout_s = float(os.environ.get("LF_TORCHLOAD_TIMEOUT_S", "6.5"))
                items.append(("torchload_timeout_s", str(load_timeout_s)))

                import subprocess
                import json
                import uuid

                out_path = f"/tmp/torchload_{uuid.uuid4().hex}.json"

                code = r"""
import time, json, sys, os
t0 = time.perf_counter()
path = sys.argv[1]
out_path = sys.argv[2]
try:
    import torch
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    t_import_done = time.perf_counter()

    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")

    dt = time.perf_counter() - t0
    payload = {
        "status": "OK",
        "dt": dt,
        "type": type(obj).__name__,
        "t_import_s": (t_import_done - t0),
        "t_load_s": (time.perf_counter() - t_import_done),
    }
except Exception as e:
    dt = time.perf_counter() - t0
    payload = {"status": "ERR", "dt": dt, "err": f"{type(e).__name__}: {e}"}

with open(out_path, "w") as f:
    json.dump(payload, f)
"""

                _stage(items, "subprocess_popen_begin")
                t0 = time.perf_counter()
                proc = subprocess.Popen(
                    [sys.executable, "-c", code, p, out_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                _stage(items, "subprocess_popen_done")

                try:
                    _stage(items, "subprocess_wait_begin")
                    proc.wait(timeout=load_timeout_s)
                    _stage(items, "subprocess_wait_done")
                except subprocess.TimeoutExpired:
                    _stage(items, "torch_load_timeout")
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    dt_wait = time.perf_counter() - t0
                    items.append(("t_wait_s", f"{dt_wait:.4f}"))
                    items.append(("T_TIMEOUT", f"torch.load exceeded {load_timeout_s}s; subprocess killed"))
                    _add_common_timing(items, t_handler0)
                    return _result(False, items, ui_minimum=ui_minimum, no_items=no_items)

                dt_wait = time.perf_counter() - t0
                items.append(("t_wait_s", f"{dt_wait:.4f}"))

                if not os.path.exists(out_path):
                    items.append(("T_FAIL", "Subprocess finished but output missing"))
                    _add_common_timing(items, t_handler0)
                    return _result(False, items, ui_minimum=ui_minimum, no_items=no_items)

                with open(out_path, "r") as f:
                    payload = json.load(f)

                status = str(payload.get("status", "ERR"))
                items.append(("torch_load_status", status))
                items.append(("t_torch_total_s", f"{float(payload.get('dt', 0.0)):.4f}"))

                if status != "OK":
                    items.append(("T_FAIL", str(payload.get("err", "unknown error"))))
                    _add_common_timing(items, t_handler0)
                    return _result(False, items, ui_minimum=ui_minimum, no_items=no_items)

                _stage(items, "torch_load_done")
                _add_common_timing(items, t_handler0)
                return _result(True, items, ui_minimum=ui_minimum, no_items=no_items)

            except Exception:
                items.append(("T_FAIL", "see TRACEBACK"))
                items.append(("TRACEBACK", _tb_short()))
                _add_common_timing(items, t_handler0)
                return _result(False, items, ui_minimum=ui_minimum, no_items=no_items)

        # ----------------------------
        # From here: need response/url to do load-check or infer.
        # Keep original CI behaviour: bad URL => LOAD_FAIL.
        # ----------------------------
        if not isinstance(response, list) or len(response) == 0:
            items.append(("BAD_INPUT", "No images uploaded."))
            _add_common_timing(items, t_handler0)
            return _result(False, items, ui_minimum=ui_minimum, no_items=no_items)

        first = response[0] if isinstance(response, list) and len(response) > 0 else None
        url = first.get("url") if isinstance(first, dict) else None

        if echo:
            # still safe: echo is off by default
            items.append(("response_type", type(response).__name__))
            items.append(("response_len", str(len(response))))
            items.append(("first_url", str(url)))

        if not url:
            items.append(("LOAD_FAIL", "first image has no url"))
            _add_common_timing(items, t_handler0)
            return _result(False, items, ui_minimum=ui_minimum, no_items=no_items)

        img: Optional[np.ndarray] = None
        if (not skip_load_check) or try_fetch or diag in ("load_model", "infer_once"):
            try:
                _stage(items, "image_load_begin")
                (img, err), dt_img = _timeit(lambda: _load_bgr_image_from_url(str(url)))
                items.append(("t_image_load_s", f"{dt_img:.4f}"))
                if img is None:
                    items.append(("LOAD_FAIL", f"Failed to load image ({err})"))
                    _add_common_timing(items, t_handler0)
                    return _result(False, items, ui_minimum=ui_minimum, no_items=no_items)

                _stage(items, "image_load_done")
            except Exception:
                items.append(("LOAD_FAIL", "Exception during image load"))
                items.append(("TRACEBACK", _tb_short()))
                _add_common_timing(items, t_handler0)
                return _result(False, items, ui_minimum=ui_minimum, no_items=no_items)

        # ----------------------------
        # D0) load model ONLY (no image required)
        # ----------------------------
        if diag == "load_model_only":
            try:
                _stage(items, "load_model_only_begin")

                _stage(items, "ultralytics_symbol_begin")
                YOLO, dt_ul = _timeit(lambda: ultralytics.YOLO)
                _ = YOLO
                items.append(("t_ultralytics_symbol_s", f"{dt_ul:.4f}"))
                _stage(items, "ultralytics_symbol_done")

                model_path = next((p for p in _candidate_model_paths() if os.path.exists(p)), None)
                if not model_path:
                    items.append(("D_FAIL", "No model file found to load"))
                    _add_common_timing(items, t_handler0)
                    return _result(False, items, ui_minimum=ui_minimum, no_items=no_items)

                try:
                    _ = torch.__version__
                    torch.set_num_threads(1)
                    torch.set_num_interop_threads(1)
                except Exception:
                    pass

                _stage(items, "yolo_load_begin")
                _watchdog_start(6)
                _, dt_load = _timeit(lambda: ultralytics.YOLO(model_path))
                _watchdog_stop()
                items.append(("t_model_load_s", f"{dt_load:.4f}"))
                _stage(items, "yolo_load_done")

                _add_common_timing(items, t_handler0)
                return _result(True, items, ui_minimum=ui_minimum, no_items=no_items)

            except Exception:
                _watchdog_stop()
                items.append(("D_FAIL", "see TRACEBACK"))
                items.append(("TRACEBACK", _tb_short()))
                _add_common_timing(items, t_handler0)
                return _result(False, items, ui_minimum=ui_minimum, no_items=no_items)

        # ----------------------------
        # D) load model only (no inference)
        # ----------------------------
        if diag == "load_model":
            try:
                _stage(items, "ultralytics_symbol_begin")
                YOLO, dt_ul = _timeit(lambda: ultralytics.YOLO)
                _ = YOLO
                items.append(("t_ultralytics_symbol_s", f"{dt_ul:.4f}"))
                _stage(items, "ultralytics_symbol_done")

                model_path = next((p for p in _candidate_model_paths() if os.path.exists(p)), None)
                if not model_path:
                    items.append(("D_FAIL", "No model file found to load"))
                    _add_common_timing(items, t_handler0)
                    return _result(False, items, ui_minimum=ui_minimum, no_items=no_items)

                try:
                    _ = torch.__version__
                    torch.set_num_threads(1)
                    torch.set_num_interop_threads(1)
                except Exception:
                    pass

                _stage(items, "yolo_load_begin")
                _watchdog_start(6)
                _, dt_load = _timeit(lambda: ultralytics.YOLO(model_path))
                _watchdog_stop()
                items.append(("t_model_load_s", f"{dt_load:.4f}"))
                _stage(items, "yolo_load_done")

                _add_common_timing(items, t_handler0)
                return _result(True, items, ui_minimum=ui_minimum, no_items=no_items)

            except Exception:
                _watchdog_stop()
                items.append(("D_FAIL", "see TRACEBACK"))
                items.append(("TRACEBACK", _tb_short()))
                _add_common_timing(items, t_handler0)
                return _result(False, items, ui_minimum=ui_minimum, no_items=no_items)

        # ----------------------------
        # E) infer once (minimal)
        # ----------------------------
        if diag == "infer_once":
            try:
                _stage(items, "ultralytics_symbol_begin")
                YOLO, dt_ul = _timeit(lambda: ultralytics.YOLO)
                _ = YOLO
                items.append(("t_ultralytics_symbol_s", f"{dt_ul:.4f}"))
                _stage(items, "ultralytics_symbol_done")

                model_path = next((p for p in _candidate_model_paths() if os.path.exists(p)), None)
                if not model_path:
                    items.append(("E_FAIL", "No model file found for inference"))
                    _add_common_timing(items, t_handler0)
                    return _result(False, items, ui_minimum=ui_minimum, no_items=no_items)

                if img is None:
                    items.append(("E_FAIL", "Image not loaded (img is None)"))
                    _add_common_timing(items, t_handler0)
                    return _result(False, items, ui_minimum=ui_minimum, no_items=no_items)

                try:
                    _ = torch.__version__
                    torch.set_num_threads(1)
                    torch.set_num_interop_threads(1)
                except Exception:
                    pass

                _stage(items, "yolo_load_begin")
                _watchdog_start(6)
                model, dt_load = _timeit(lambda: ultralytics.YOLO(model_path))
                _watchdog_stop()
                items.append(("t_model_load_s", f"{dt_load:.4f}"))
                _stage(items, "yolo_load_done")

                _stage(items, "predict_begin")
                _, dt_pred = _timeit(lambda: model.predict(source=img, imgsz=640, conf=0.25, device="cpu", verbose=False))
                items.append(("t_predict_s", f"{dt_pred:.4f}"))
                _stage(items, "predict_done")

                _add_common_timing(items, t_handler0)
                return _result(True, items, ui_minimum=ui_minimum, no_items=no_items)
            except Exception:
                _watchdog_stop()
                items.append(("E_FAIL", "see TRACEBACK"))
                items.append(("TRACEBACK", _tb_short()))
                _add_common_timing(items, t_handler0)
                return _result(False, items, ui_minimum=ui_minimum, no_items=no_items)

        # ----------------------------
        # Optional early exit
        # ----------------------------
        if fast_return and not try_fetch:
            items.append(("note", "fast_return=True (no YOLO)"))
            _add_common_timing(items, t_handler0)
            return _result(True, items, ui_minimum=ui_minimum, no_items=no_items)

        items.append(("note", "No YOLO executed in default path"))
        _add_common_timing(items, t_handler0)
        return _result(True, items, ui_minimum=ui_minimum, no_items=no_items)

    except Exception as e:
        _watchdog_stop()
        items.append(("UNHANDLED", f"{type(e).__name__}: {e}"))
        items.append(("TRACEBACK", _tb_short()))
        _add_common_timing(items, t_handler0)
        return _result(False, items, ui_minimum=ui_minimum, no_items=no_items)