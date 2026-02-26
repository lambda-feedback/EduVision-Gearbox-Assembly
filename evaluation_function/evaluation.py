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
_MAX_FEEDBACK_CHARS = int(os.environ.get("LF_MAX_FEEDBACK_CHARS", "1200"))  # keep moderate
_MAX_LINES = int(os.environ.get("LF_MAX_LINES", "40"))  # cap lines shown


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


def _truncate_lines(lines: List[str]) -> List[str]:
    if len(lines) > _MAX_LINES:
        lines = lines[:_MAX_LINES] + [f"... (truncated, showing {_MAX_LINES} lines)"]
    return lines


def _truncate_text(s: str) -> str:
    s = s.strip()
    if len(s) > _MAX_FEEDBACK_CHARS:
        s = s[:_MAX_FEEDBACK_CHARS] + " ... (truncated)"
    return s


def _result_minimal(is_correct: bool, message: str) -> Result:
    """
    Return Result in a version-tolerant minimal way:
    - Prefer Result(feedback="...") if supported.
    - Fallback to Result(feedback_items=[(...)] ) for older lf_toolkit versions.
      (keeps unit tests that look for LOAD_FAIL etc.)
    """
    msg = _truncate_text(message)

    # Extract a key for legacy feedback_items
    key = "OK" if is_correct else "FAIL"
    if ":" in msg:
        k0 = msg.split(":", 1)[0].strip()
        if k0:
            key = k0

    try:
        return Result(is_correct=is_correct, feedback=msg)
    except TypeError:
        # Older lf_toolkit: feedback not supported
        try:
            return Result(is_correct=is_correct, feedback_items=[(key, msg)])
        except TypeError:
            return Result(is_correct=is_correct)


def _build_hud_from_pipeline_output(out: Dict[str, Any]) -> List[str]:
    """
    Convert pipeline output into the HUD-like lines you want.
    We try multiple possible keys to be robust across your pipeline versions.
    """
    lines: List[str] = []

    # --- summary / counts ---
    summary = out.get("summary") if isinstance(out.get("summary"), dict) else {}
    ratio = out.get("ratio") if isinstance(out.get("ratio"), dict) else {}
    errors = out.get("errors") if isinstance(out.get("errors"), list) else []

    # Try to pull common fields (robust)
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

    # Header lines
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

    # Mesh/mismesh
    if mesh_count is not None or mismesh_count is not None:
        if mesh_count is not None:
            lines.append(f"Mesh count: {mesh_count}")
        if mismesh_count is not None:
            lines.append(f"Mismesh count: {mismesh_count}")

    # Ratio section
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

        # motor rpm optional
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

        # per-stage details
        if isinstance(per_stage, list):
            for s in per_stage:
                # accept tuple/list like (stage, R, z1, z2) or dict
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

    # Issues
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


def evaluation_function(response: Any, answer: Any, params: Params) -> Result:
    """
    Platform entry:
      response = [{"url": "...", ...}, ...]
    Output:
      - Plain text feedback (multi-line, capped)
      - Optional: uploads det/label images (returns URLs as plain text lines)
    """
    t0 = time.perf_counter()

    # If pipeline failed to import, return minimal error (short, non-HTML)
    if run_yolo_pipeline is None:
        err = PIPELINE_IMPORT_ERROR or {}
        msg = f"E_PIPELINE_IMPORT: {err.get('exc_type', 'ImportError')}: {err.get('message', 'pipeline import failed')}"
        return _result_minimal(False, msg)

    # Validate input
    if not isinstance(response, list) or len(response) == 0:
        return _result_minimal(False, "BAD_INPUT: Please upload at least one image.")

    first = response[0] if isinstance(response[0], dict) else None
    url = first.get("url") if isinstance(first, dict) else None
    if not url:
        return _result_minimal(False, "LOAD_FAIL: first image has no url field")

    # Controls (keep minimal; no debug spam)
    return_images: bool = bool(_pget(params, "return_images", True))  # you want detection image
    gear_model_rel = str(_pget(params, "gear_model_rel", "gear_model.pt"))
    shaft_model_rel = str(_pget(params, "shaft_model_rel", "shaft_model.pt"))

    # Load image
    img_bgr, err = _load_bgr_image_from_url(str(url))
    if img_bgr is None:
        return _result_minimal(False, f"LOAD_FAIL: Failed to load image ({err})")

    # Run pipeline (single image)
    try:
        out: Dict[str, Any] = run_yolo_pipeline(  # type: ignore[misc]
            img_bgr=img_bgr,
            gear_model_rel=gear_model_rel,
            shaft_model_rel=shaft_model_rel,
            return_images=return_images,
        )
    except Exception as e:
        # minimal, no traceback to UI
        return _result_minimal(False, f"E_PIPELINE_RUNTIME: {type(e).__name__}: {e}")

    # Build HUD-like feedback lines
    lines = _build_hud_from_pipeline_output(out)

    # Optional: attach uploaded image links as plain text lines (no HTML)
    if return_images:
        imgs = out.get("images") if isinstance(out.get("images"), dict) else {}
        # expected keys like "det_img", "label_img"
        det_img = imgs.get("det_img") if isinstance(imgs.get("det_img"), np.ndarray) else None
        label_img = imgs.get("label_img") if isinstance(imgs.get("label_img"), np.ndarray) else None

        # Only if upload_image is available
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
                # keep short
                lines.append(f"E_UPLOAD_FAIL: {e}")
            except Exception as e:
                lines.append(f"E_UPLOAD_FAIL: {type(e).__name__}: {e}")
        else:
            # If upload isn't available, still OK — don't spam
            pass

    # Add a tiny runtime line (1 line only)
    dt = time.perf_counter() - t0
    lines.append(f"runtime_s: {dt:.3f}")

    # Cap output
    lines = _truncate_lines(lines)
    msg = "\n".join(lines)

    # Decide correctness: any E_* in errors => incorrect
    errors = out.get("errors") if isinstance(out.get("errors"), list) else []
    has_E = any(isinstance(e, dict) and str(e.get("code", "")).startswith("E_") for e in errors)
    is_correct = not has_E

    return _result_minimal(is_correct, msg)