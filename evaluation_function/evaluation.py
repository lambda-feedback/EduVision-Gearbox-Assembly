from __future__ import annotations

import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, unquote

import cv2
import numpy as np
import requests

from lf_toolkit.evaluation import Result, Params

# --- Optional image upload support (depends on lf_toolkit version) ---
try:
    from lf_toolkit.evaluation.image_upload import upload_image, ImageUploadError  # type: ignore
except Exception:
    upload_image = None  # type: ignore

    class ImageUploadError(Exception):  # type: ignore
        pass


# ----------------------------
# Pipeline import guard
# ----------------------------
PIPELINE_IMPORT_ERROR: Optional[Dict[str, str]] = None
run_yolo_pipeline = None

try:
    from .yolo_pipeline import run_yolo_pipeline  # type: ignore
except Exception as e:
    PIPELINE_IMPORT_ERROR = {
        "stage": "IMPORT",
        "error_code": "E_PIPELINE_IMPORT",
        "exc_type": type(e).__name__,
        "message": str(e),
        "traceback": traceback.format_exc(),
    }
    run_yolo_pipeline = None


# ----------------------------
# Output caps (avoid UI freeze)
# ----------------------------
_MAX_FEEDBACK_CHARS = int(os.environ.get("LF_MAX_FEEDBACK_CHARS", "500000"))
_MAX_LINES = int(os.environ.get("LF_MAX_LINES", "20000"))


def _pget(params: Params, key: str, default: Any) -> Any:
    """Params in lf_toolkit is dict-like; keep safe across versions."""
    try:
        return params.get(key, default)  # type: ignore
    except Exception:
        try:
            return params[key]  # type: ignore
        except Exception:
            return default


def file_url_to_local_path(url: str) -> str:
    parsed = urlparse(url)
    path = unquote(parsed.path)
    if os.name == "nt" and len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return path


def _load_bgr_image_from_url(url: str, timeout: int = 15) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Load image into OpenCV BGR ndarray from:
      - http(s) URL (platform-hosted)
      - file:// URL (local dev only)
    Returns (img_bgr, err_message).
    """
    try:
        if not isinstance(url, str) or not url:
            return None, "Empty URL."

        if url.startswith("file://"):
            local_path = file_url_to_local_path(url)
            img = cv2.imread(local_path, cv2.IMREAD_COLOR)
            if img is None:
                return None, f"cv2.imread failed: {local_path}"
            return img, None

        if url.startswith("http://") or url.startswith("https://"):
            # safer timeouts: (connect, read)
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


def _cv2_bgr_to_png_bytes(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("Failed to encode image as PNG.")
    return buf.tobytes()


def _truncate_lines(lines: List[str], max_lines: int) -> List[str]:
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... (truncated, showing {max_lines} lines)"]
    return lines


def _truncate_text(s: str, max_chars: int) -> str:
    s = s.strip()
    if len(s) > max_chars:
        s = s[:max_chars] + " ... (truncated)"
    return s


def _result_minimal(is_correct: bool, message: str, *, max_chars: int = _MAX_FEEDBACK_CHARS) -> Result:
    """
    Return Result in a version-tolerant minimal way:
    - Prefer Result(feedback="...") if supported.
    - Fallback to Result(feedback_items=[(...)] ) for older lf_toolkit versions.
    """
    msg = _truncate_text(message, max_chars=max_chars)

    key = "OK" if is_correct else "FAIL"
    if ":" in msg:
        k0 = msg.split(":", 1)[0].strip()
        if k0:
            key = k0

    try:
        return Result(is_correct=is_correct, feedback=msg)
    except TypeError:
        try:
            return Result(is_correct=is_correct, feedback_items=[(key, msg)])
        except TypeError:
            return Result(is_correct=is_correct)


def _build_hud_from_pipeline_output(out: Dict[str, Any]) -> List[str]:
    """
    Convert pipeline output into HUD-like lines.
    """
    lines: List[str] = []

    summary = out.get("summary") if isinstance(out.get("summary"), dict) else {}
    ratio = out.get("ratio") if isinstance(out.get("ratio"), dict) else {}
    errors = out.get("errors") if isinstance(out.get("errors"), list) else []

    gears_total = summary.get("gears_detected", summary.get("gear_count", summary.get("gears", None)))
    shafts_total = summary.get("shafts_detected", summary.get("shaft_count", summary.get("shafts", None)))
    spacers_total = summary.get("spacers_detected", summary.get("spacer_count", summary.get("spacers", None)))
    mesh_count = summary.get("mesh_count", None)
    mismesh_count = summary.get("mismesh_count", None)

    big_count = summary.get("gear_big_count", summary.get("big_gears", None))
    small_count = summary.get("gear_small_count", summary.get("small_gears", None))

    stages = ratio.get("stages_computed", ratio.get("num_stages", None))
    total_ratio = ratio.get("total_ratio", ratio.get("R_total", None))
    out_rpm = ratio.get("output_rpm", ratio.get("out_rpm", None))
    motor_rpm = ratio.get("motor_rpm", ratio.get("MOTOR_RPM", None))

    per_stage = ratio.get("per_stage", ratio.get("stages", None))

    if gears_total is not None:
        lines.append(f"Gears detected: {gears_total}")
    if shafts_total is not None:
        lines.append(f"Shafts detected: {shafts_total}")
    if spacers_total is not None:
        lines.append(f"Spacers detected: {spacers_total}")
    if big_count is not None or small_count is not None:
        if big_count is not None:
            lines.append(f"Big gears: {big_count}")
        if small_count is not None:
            lines.append(f"Small gears: {small_count}")

    if mesh_count is not None or mismesh_count is not None:
        if mesh_count is not None:
            lines.append(f"Mesh count: {mesh_count}")
        if mismesh_count is not None:
            lines.append(f"Mismesh count: {mismesh_count}")

    if stages is not None:
        lines.append(f"Stages (computed): {stages}")

    if total_ratio is None or out_rpm is None:
        lines.append("Total gear ratio (slowdown): N/A")
        lines.append("Output shaft speed: N/A")
    else:
        try:
            lines.append(f"Total gear ratio (slowdown): {float(total_ratio):.3f}")
        except Exception:
            lines.append(f"Total gear ratio (slowdown): {total_ratio}")

        if motor_rpm is not None:
            try:
                lines.append(f"Output shaft speed: {float(out_rpm):.1f} RPM (motor={float(motor_rpm):.0f})")
            except Exception:
                lines.append(f"Output shaft speed: {out_rpm} RPM (motor={motor_rpm})")
        else:
            try:
                lines.append(f"Output shaft speed: {float(out_rpm):.1f} RPM")
            except Exception:
                lines.append(f"Output shaft speed: {out_rpm} RPM")

        if isinstance(per_stage, list):
            for s in per_stage:
                if isinstance(s, (tuple, list)) and len(s) >= 4:
                    stg, R, z1, z2 = s[0], s[1], s[2], s[3]
                    try:
                        lines.append(f"Stage{int(stg)}: {int(z2)}/{int(z1)} = {float(R):.2f}")
                    except Exception:
                        lines.append(f"Stage{stg}: {z2}/{z1} = {R}")
                elif isinstance(s, dict):
                    stg = s.get("stage", s.get("s", "?"))
                    z1 = s.get("z1", "?")
                    z2 = s.get("z2", "?")
                    R = s.get("ratio", s.get("R", "?"))
                    lines.append(f"Stage{stg}: {z2}/{z1} = {R}")

    if isinstance(errors, list) and errors:
        lines.append("---- ISSUES ----")
        for e in errors:
            if isinstance(e, dict):
                msg = str(e.get("message", "")).strip()
                if msg:
                    lines.append(msg)
                else:
                    code = str(e.get("code", "E_ERR"))
                    lines.append(f"{code}")
            else:
                lines.append(str(e))
    else:
        lines.append("Issues: None")

    return lines


def _candidate_model_paths(gear_model_rel: str, shaft_model_rel: str) -> Dict[str, List[str]]:
    """
    Try common locations for model files:
      - alongside this file (evaluation_function/...)
      - /app/evaluation_function/... inside the container
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return {
        "gear": [
            os.path.join(here, gear_model_rel),
            os.path.join(here, "gear_model.pt"),
            f"/app/evaluation_function/{gear_model_rel}",
            "/app/evaluation_function/gear_model.pt",
        ],
        "shaft": [
            os.path.join(here, shaft_model_rel),
            os.path.join(here, "shaft_model.pt"),
            f"/app/evaluation_function/{shaft_model_rel}",
            "/app/evaluation_function/shaft_model.pt",
        ],
    }


def _pick_existing_path(cands: List[str]) -> Optional[str]:
    for p in cands:
        if p and os.path.exists(p):
            return p
    return None


def evaluation_function(response: Any, answer: Any, params: Params) -> Result:
    """
    Platform entry:
      response = [{"url": "...", ...}, ...]
    """
    t0 = time.perf_counter()

    # ----------------------------
    # Diagnostics switches (mainly for frontend testing)
    # ----------------------------
    diag = str(_pget(params, "diag", "") or "").strip().lower()

    # If diag_unbounded=True, allow long outputs (for UI freeze repro / stress)
    diag_unbounded = bool(_pget(params, "diag_unbounded", False))
    diag_chars = int(_pget(params, "diag_chars", 200_000))
    diag_lines = int(_pget(params, "diag_lines", 5000))

    max_chars = diag_chars if diag_unbounded else _MAX_FEEDBACK_CHARS
    max_lines = diag_lines if diag_unbounded else _MAX_LINES

    # Controls
    return_images: bool = bool(_pget(params, "return_images", True))
    gear_model_rel = str(_pget(params, "gear_model_rel", "gear_model.pt"))
    shaft_model_rel = str(_pget(params, "shaft_model_rel", "shaft_model.pt"))

    # ----------------------------
    # Input validation (needed by infer_once + normal pipeline)
    # ----------------------------
    if not isinstance(response, list) or len(response) == 0:
        return _result_minimal(False, "BAD_INPUT: Please upload at least one image.", max_chars=max_chars)

    first = response[0] if isinstance(response[0], dict) else None
    url = first.get("url") if isinstance(first, dict) else None
    if not url:
        return _result_minimal(False, "LOAD_FAIL: first image has no url field", max_chars=max_chars)

    # Load image (we do it early because infer_once needs it)
    img_bgr, err = _load_bgr_image_from_url(str(url))
    if img_bgr is None:
        return _result_minimal(False, f"LOAD_FAIL: Failed to load image ({err})", max_chars=max_chars)

    # ----------------------------
    # DIAG 1) load_model_only  (matches your old smoke test name)
    # ----------------------------
    if diag == "load_model_only":
        lines: List[str] = []
        try:
            model_to_load = str(_pget(params, "model_to_load", "gear") or "gear").strip().lower()
            if model_to_load not in ("gear", "shaft"):
                model_to_load = "gear"

            lines.append("---- DIAG load_model_only ----")
            lines.append(f"model_to_load: {model_to_load}")

            # Import inside diagnostic (so normal path stays lighter)
            t_imp0 = time.perf_counter()
            from ultralytics import YOLO  # type: ignore
            t_imp = time.perf_counter() - t_imp0
            lines.append(f"t_import_ultralytics_s: {t_imp:.4f}")

            # pick model file
            cands = _candidate_model_paths(gear_model_rel, shaft_model_rel)[model_to_load]
            model_path = _pick_existing_path(cands)
            if not model_path:
                lines.append("E_MODEL_NOT_FOUND: no model file found in candidates")
                lines.append("candidates:")
                lines.extend([f"  - {p}" for p in cands])
                lines = _truncate_lines(lines, max_lines=max_lines)
                return _result_minimal(False, "\n".join(lines), max_chars=max_chars)

            lines.append(f"model_path: {model_path}")
            try:
                lines.append(f"model_size_bytes: {os.path.getsize(model_path)}")
            except Exception:
                pass

            # Load model (this is the “hangy” part you want to reproduce)
            t_load0 = time.perf_counter()
            _ = YOLO(model_path)
            t_load = time.perf_counter() - t_load0
            lines.append(f"t_model_load_s: {t_load:.4f}")
            lines.append("status: model loaded ✅")

            dt = time.perf_counter() - t0
            lines.append(f"runtime_s: {dt:.3f}")

            lines = _truncate_lines(lines, max_lines=max_lines)
            return _result_minimal(True, "\n".join(lines), max_chars=max_chars)

        except Exception as e:
            lines.append(f"E_DIAG_LOAD_MODEL_ONLY: {type(e).__name__}: {e}")
            dt = time.perf_counter() - t0
            lines.append(f"runtime_s: {dt:.3f}")
            lines = _truncate_lines(lines, max_lines=max_lines)
            return _result_minimal(False, "\n".join(lines), max_chars=max_chars)

    # ----------------------------
    # DIAG 2) infer_once (matches your old smoke test name)
    # ----------------------------
    if diag == "infer_once":
        lines: List[str] = []
        try:
            model_to_load = str(_pget(params, "model_to_load", "gear") or "gear").strip().lower()
            if model_to_load not in ("gear", "shaft"):
                model_to_load = "gear"

            imgsz = int(_pget(params, "imgsz", 640))
            conf = float(_pget(params, "conf", 0.25))
            device = str(_pget(params, "device", "cpu"))
            verbose = bool(_pget(params, "verbose", False))

            lines.append("---- DIAG infer_once ----")
            lines.append(f"model_to_load: {model_to_load}")
            lines.append(f"imgsz: {imgsz}")
            lines.append(f"conf: {conf}")
            lines.append(f"device: {device}")
            lines.append(f"verbose: {verbose}")

            t_imp0 = time.perf_counter()
            from ultralytics import YOLO  # type: ignore
            t_imp = time.perf_counter() - t_imp0
            lines.append(f"t_import_ultralytics_s: {t_imp:.4f}")

            cands = _candidate_model_paths(gear_model_rel, shaft_model_rel)[model_to_load]
            model_path = _pick_existing_path(cands)
            if not model_path:
                lines.append("E_MODEL_NOT_FOUND: no model file found in candidates")
                lines.append("candidates:")
                lines.extend([f"  - {p}" for p in cands])
                lines = _truncate_lines(lines, max_lines=max_lines)
                return _result_minimal(False, "\n".join(lines), max_chars=max_chars)

            lines.append(f"model_path: {model_path}")

            t_load0 = time.perf_counter()
            model = YOLO(model_path)
            t_load = time.perf_counter() - t_load0
            lines.append(f"t_model_load_s: {t_load:.4f}")

            # Predict once (this can also hang / be slow)
            t_pred0 = time.perf_counter()
            _ = model.predict(
                source=img_bgr,
                imgsz=imgsz,
                conf=conf,
                device=device,
                verbose=verbose,
            )
            t_pred = time.perf_counter() - t_pred0
            lines.append(f"t_predict_s: {t_pred:.4f}")
            lines.append("status: predict done ✅")

            dt = time.perf_counter() - t0
            lines.append(f"runtime_s: {dt:.3f}")

            lines = _truncate_lines(lines, max_lines=max_lines)
            return _result_minimal(True, "\n".join(lines), max_chars=max_chars)

        except Exception as e:
            lines.append(f"E_DIAG_INFER_ONCE: {type(e).__name__}: {e}")
            dt = time.perf_counter() - t0
            lines.append(f"runtime_s: {dt:.3f}")
            lines = _truncate_lines(lines, max_lines=max_lines)
            return _result_minimal(False, "\n".join(lines), max_chars=max_chars)

    # ----------------------------
    # Normal path: your YOLO pipeline
    # ----------------------------
    if run_yolo_pipeline is None:
        erri = PIPELINE_IMPORT_ERROR or {}
        msg = f"E_PIPELINE_IMPORT: {erri.get('exc_type', 'ImportError')}: {erri.get('message', 'pipeline import failed')}"
        return _result_minimal(False, msg, max_chars=max_chars)

    try:
        out: Dict[str, Any] = run_yolo_pipeline(  # type: ignore[misc]
            img_bgr=img_bgr,
            gear_model_rel=gear_model_rel,
            shaft_model_rel=shaft_model_rel,
            return_images=return_images,
        )
    except Exception as e:
        return _result_minimal(False, f"E_PIPELINE_RUNTIME: {type(e).__name__}: {e}", max_chars=max_chars)

    lines = _build_hud_from_pipeline_output(out)

    if return_images:
        imgs = out.get("images") if isinstance(out.get("images"), dict) else {}
        det_img = imgs.get("det_img") if isinstance(imgs.get("det_img"), np.ndarray) else None
        label_img = imgs.get("label_img") if isinstance(imgs.get("label_img"), np.ndarray) else None

        if upload_image is not None:
            try:
                if det_img is not None:
                    png = _cv2_bgr_to_png_bytes(det_img)
                    det_url = upload_image(png, "eduvision")  # type: ignore[misc]
                    lines.append(f"det_img_url: {det_url}")
                if label_img is not None:
                    png = _cv2_bgr_to_png_bytes(label_img)
                    lab_url = upload_image(png, "eduvision")  # type: ignore[misc]
                    lines.append(f"label_img_url: {lab_url}")
            except ImageUploadError as e:
                lines.append(f"E_UPLOAD_FAIL: {e}")
            except Exception as e:
                lines.append(f"E_UPLOAD_FAIL: {type(e).__name__}: {e}")

    dt = time.perf_counter() - t0
    lines.append(f"runtime_s: {dt:.3f}")

    lines = _truncate_lines(lines, max_lines=_MAX_LINES)
    msg = "\n".join(lines)

    errors = out.get("errors") if isinstance(out.get("errors"), list) else []
    has_E = any(isinstance(e, dict) and str(e.get("code", "")).startswith("E_") for e in errors)
    is_correct = not has_E

    return _result_minimal(is_correct, msg, max_chars=_MAX_FEEDBACK_CHARS)