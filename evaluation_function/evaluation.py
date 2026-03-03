from __future__ import annotations

import os
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
_MAX_FEEDBACK_CHARS = int(os.environ.get("LF_MAX_FEEDBACK_CHARS", "1200"))
_MAX_LINES = int(os.environ.get("LF_MAX_LINES", "40"))


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


# ----------------------------
# Precheck: image quality only (NOT assembly correctness)
# ----------------------------
def _image_quality_checks(img_bgr: np.ndarray) -> List[str]:
    """
    Photo/setup quality checks: resolution, blur, exposure.
    This is intentionally separate from assembly correctness logic (which stays in yolo_pipeline).
    """
    lines: List[str] = []
    h, w = img_bgr.shape[:2]

    # Resolution gate (tune on your dataset)
    min_w, min_h = 900, 700
    if w < min_w or h < min_h:
        lines.append(f"W_RES_LOW: {w}x{h} (aim >= {min_w}x{min_h}). Move closer; avoid digital zoom.")
    else:
        lines.append(f"OK_RES: {w}x{h}")

    # Blur (variance of Laplacian)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_score < 80.0:
        lines.append(f"W_BLUR: blurry (score={blur_score:.1f}). Hold steady; tap-to-focus.")
    else:
        lines.append(f"OK_SHARP: blur_score={blur_score:.1f}")

    # Exposure (mean intensity heuristic)
    mean_intensity = float(gray.mean())
    if mean_intensity < 60.0:
        lines.append(f"W_DARK: mean={mean_intensity:.1f}. Increase lighting; reduce shadows.")
    elif mean_intensity > 200.0:
        lines.append(f"W_GLARE: mean={mean_intensity:.1f}. Avoid reflections; change angle/diffuse light.")
    else:
        lines.append(f"OK_EXPOSURE: mean={mean_intensity:.1f}")

    return lines


def _select_errors_by_task(errors: List[Dict[str, Any]], task: str) -> List[Dict[str, Any]]:
    """
    Filter pipeline-produced errors for each step.
    No new judgement is introduced here; we only choose what to display.
    """
    task = (task or "full").strip().lower()

    def keep(e: Dict[str, Any]) -> bool:
        code = str(e.get("code", "")).upper()

        if task in ("full", "all"):
            return True

        if task == "precheck":
            # Show only errors that suggest missed detections / sanity issues
            return code in {
                "E_NO_GEARS",
                "E_CONTACT_COUNT_MISMATCH",
                "E_GEAR_COUNT_UNSUPPORTED",
            }

        if task == "shaft":
            # pipeline may return E_NO_SHAFTS (new) or E_SHAFT* codes (legacy)
            return code.startswith("E_SHAFT") or code == "E_NO_SHAFTS"

        if task == "spacer":
            return code.startswith("E_SPACER")

        if task == "gear_inventory":
            return code in {"E_NO_GEARS", "E_GEAR_COUNT_MISMATCH"}

        if task == "mesh_ratio":
            # mesh/mismesh + contact sanity
            return ("MESH" in code) or code.startswith("E_CONTACT_COUNT") or code == "E_NO_GEARS"

        return True

    return [e for e in errors if isinstance(e, dict) and keep(e)]


def _render_summary_lines(out: Dict[str, Any], task: str) -> List[str]:
    """
    Render compact summary from yolo_pipeline keys.
    Your pipeline uses: gears/spacers/shafts/mesh/mismesh/stages.
    Ratio uses: num_stages/R_total/out_rpm/per_stage.
    """
    summary = out.get("summary") if isinstance(out.get("summary"), dict) else {}
    ratio = out.get("ratio") if isinstance(out.get("ratio"), dict) else {}

    task = (task or "full").strip().lower()

    gears = summary.get("gears", None)
    shafts = summary.get("shafts", None)
    spacers = summary.get("spacers", None)
    mesh = summary.get("mesh", None)
    mismesh = summary.get("mismesh", None)
    stages = summary.get("stages", None)
    big_cnt = summary.get("gear_big", None)
    small_cnt = summary.get("gear_small", None)

    lines: List[str] = []

    if task in ("precheck", "full", "all"):
        if gears is not None:
            lines.append(f"INFO_GEARS: {gears}")
        if shafts is not None:
            lines.append(f"INFO_SHAFTS: {shafts}")
        if spacers is not None:
            lines.append(f"INFO_SPACERS: {spacers}")
        if mesh is not None or mismesh is not None:
            lines.append(f"INFO_MESH: {mesh} (mismesh={mismesh})")
        if stages is not None:
            lines.append(f"INFO_STAGES: {stages}")

    elif task == "shaft":
        if shafts is not None:
            lines.append(f"Shafts detected: {shafts}")

    elif task == "spacer":
        if spacers is not None:
            lines.append(f"Spacers detected: {spacers}")

    elif task == "gear_inventory":
        if gears is not None:
            if big_cnt is not None or small_cnt is not None:
                lines.append(f"Gears detected: {gears} (big={big_cnt}, small={small_cnt})")
            else:
                lines.append(f"Gears detected: {gears}")
        if spacers is not None:
            lines.append(f"Spacers detected: {spacers}")

    elif task == "mesh_ratio":
        if mesh is not None or mismesh is not None:
            lines.append(f"Mesh count: {mesh} (mismesh={mismesh})")

        num_stages = ratio.get("num_stages", None)
        R_total = ratio.get("R_total", None)
        out_rpm = ratio.get("out_rpm", None)

        if num_stages is not None:
            lines.append(f"Stages (computed): {num_stages}")

        if R_total is None or out_rpm is None:
            lines.append("Total ratio: N/A")
            lines.append("Output speed: N/A")
        else:
            try:
                lines.append(f"Total ratio (slowdown): {float(R_total):.3f}")
            except Exception:
                lines.append(f"Total ratio (slowdown): {R_total}")
            try:
                lines.append(f"Output speed: {float(out_rpm):.1f} RPM")
            except Exception:
                lines.append(f"Output speed: {out_rpm}")

    else:
        if gears is not None:
            lines.append(f"Gears: {gears}")
        if shafts is not None:
            lines.append(f"Shafts: {shafts}")
        if spacers is not None:
            lines.append(f"Spacers: {spacers}")

    return lines


def _attach_overlay_urls(lines: List[str], out: Dict[str, Any]) -> List[str]:
    """
    Upload det_img / label_img if present in pipeline output.
    """
    imgs = out.get("images") if isinstance(out.get("images"), dict) else {}
    det_img = imgs.get("det_img") if isinstance(imgs.get("det_img"), np.ndarray) else None
    label_img = imgs.get("label_img") if isinstance(imgs.get("label_img"), np.ndarray) else None

    if upload_image is None:
        return lines

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

    return lines


def _render_task_result_lines(task_result: Dict[str, Any]) -> List[str]:
    """
    Render out["task_result"] into user-facing lines.
    """
    lines: List[str] = []
    t = str(task_result.get("task", "")).strip()
    st = str(task_result.get("status", "")).strip()
    nxt = str(task_result.get("recommended_next_task", "")).strip()
    ready = task_result.get("is_ready_for_next", None)

    if t:
        lines.append(f"TASK: {t}")
    if st:
        lines.append(f"STATUS: {st}")
    if ready is not None:
        lines.append(f"READY_FOR_NEXT: {bool(ready)}")

    msgs = task_result.get("messages")
    if isinstance(msgs, list) and msgs:
        lines.append("---- FEEDBACK ----")
        for m in msgs[:12]:
            lines.append(str(m))

    if nxt:
        lines.append(f"NEXT_TASK: {nxt}")

    return lines


def evaluation_function(response: Any, answer: Any, params: Params) -> Result:
    """
    Platform entry:
      response = [{"url": "...", ...}, ...]
    """
    t0 = time.perf_counter()

    # ----------------------------
    # Task routing (reads from question params)
    # ----------------------------
    task = str(_pget(params, "task", "full") or "full").strip().lower()

    # Controls
    # precheck: default no image returns (to reduce UI / payload / time)
    return_images: bool = bool(_pget(params, "return_images", task != "precheck"))
    gear_model_rel = str(_pget(params, "gear_model_rel", "gear_model.pt"))
    shaft_model_rel = str(_pget(params, "shaft_model_rel", "shaft_model.pt"))
    expected_gears = _pget(params, "expected_gears", None)

    # ----------------------------
    # Input validation
    # ----------------------------
    if not isinstance(response, list) or len(response) == 0:
        return _result_minimal(False, "BAD_INPUT: Please upload at least one image.", max_chars=_MAX_FEEDBACK_CHARS)

    first = response[0] if isinstance(response[0], dict) else None
    url = first.get("url") if isinstance(first, dict) else None
    if not url:
        return _result_minimal(False, "LOAD_FAIL: first image has no url field", max_chars=_MAX_FEEDBACK_CHARS)

    # Load image
    img_bgr, err = _load_bgr_image_from_url(str(url))
    if img_bgr is None:
        return _result_minimal(False, f"LOAD_FAIL: Failed to load image ({err})", max_chars=_MAX_FEEDBACK_CHARS)

    # ----------------------------
    # Pipeline availability
    # ----------------------------
    if run_yolo_pipeline is None:
        # For precheck, we can still return image-quality tips even if pipeline is unavailable.
        if task == "precheck":
            lines = ["W_PIPELINE_UNAVAILABLE: detection sanity checks unavailable."] + _image_quality_checks(img_bgr)
            msg = "\n".join(_truncate_lines(lines, _MAX_LINES))
            return _result_minimal(False, msg, max_chars=_MAX_FEEDBACK_CHARS)

        erri = PIPELINE_IMPORT_ERROR or {}
        msg = f"E_PIPELINE_IMPORT: {erri.get('exc_type', 'ImportError')}: {erri.get('message', 'pipeline import failed')}"
        return _result_minimal(False, msg, max_chars=_MAX_FEEDBACK_CHARS)

    # ----------------------------
    # Run pipeline (correctness logic stays in yolo_pipeline)
    # - pass task + expected_gears through
    # ----------------------------
    try:
        out: Dict[str, Any] = run_yolo_pipeline(  # type: ignore[misc]
            img_bgr=img_bgr,
            gear_model_rel=gear_model_rel,
            shaft_model_rel=shaft_model_rel,
            return_images=(return_images and task != "precheck"),
            task=task,
            expected_gears=expected_gears,
        )
    except TypeError:
        # Backward compatible: if pipeline doesn't accept new args
        out = run_yolo_pipeline(  # type: ignore[misc]
            img_bgr=img_bgr,
            gear_model_rel=gear_model_rel,
            shaft_model_rel=shaft_model_rel,
            return_images=(return_images and task != "precheck"),
        )
    except Exception as e:
        return _result_minimal(False, f"E_PIPELINE_RUNTIME: {type(e).__name__}: {e}", max_chars=_MAX_FEEDBACK_CHARS)

    errors = out.get("errors") if isinstance(out.get("errors"), list) else []
    selected_errors = _select_errors_by_task(errors, task)

    task_result = out.get("task_result") if isinstance(out.get("task_result"), dict) else None

    # ----------------------------
    # Render output per task (display only; no new judgement)
    # ----------------------------
    lines: List[str] = []

    if task == "precheck":
        lines.append("---- PRECHECK (Photo + detection sanity) ----")
        lines.extend(_image_quality_checks(img_bgr))

        # Prefer task_result if present, otherwise fallback to summary + selected errors
        if task_result is not None:
            lines.extend(_render_task_result_lines(task_result))
        else:
            lines.extend(_render_summary_lines(out, task="precheck"))
            if selected_errors:
                lines.append("---- WARNINGS ----")
                for e in selected_errors[:8]:
                    msg = str(e.get("message", "")).strip()
                    code = str(e.get("code", "E_ERR")).strip()
                    lines.append(msg if msg else code)

        # Simple pass/fail gate for precheck:
        # - Fail if image quality warnings exist OR pipeline sanity warnings exist
        is_correct_precheck = True
        if any(s.startswith(("W_RES_LOW", "W_BLUR", "W_DARK", "W_GLARE")) for s in lines):
            is_correct_precheck = False
        if selected_errors:
            is_correct_precheck = False

        if is_correct_precheck:
            lines.append("OK_PRECHECK: Photo looks acceptable. Proceed to next step.")
        else:
            lines.append("WARN_PRECHECK: Please improve photo setup and re-run precheck.")

        dt = time.perf_counter() - t0
        lines.append(f"runtime_s: {dt:.3f}")
        msg = "\n".join(_truncate_lines(lines, _MAX_LINES))
        return _result_minimal(is_correct_precheck, msg, max_chars=_MAX_FEEDBACK_CHARS)

    # Non-precheck tasks
    # Prefer task_result if present
    if task_result is not None:
        lines.extend(_render_task_result_lines(task_result))
    else:
        lines.extend(_render_summary_lines(out, task=task))
        if selected_errors:
            lines.append("---- ISSUES ----")
            for e in selected_errors[:10]:
                msg = str(e.get("message", "")).strip()
                code = str(e.get("code", "E_ERR")).strip()
                lines.append(msg if msg else code)
        else:
            lines.append("Issues: None")

    # Attach images if enabled
    if return_images and task != "precheck":
        lines = _attach_overlay_urls(lines, out)

    dt = time.perf_counter() - t0
    lines.append(f"runtime_s: {dt:.3f}")

    msg = "\n".join(_truncate_lines(lines, _MAX_LINES))

    # Correctness: pipeline drives correctness (E_ codes)
    has_E = any(isinstance(e, dict) and str(e.get("code", "")).startswith("E_") for e in errors)
    is_correct = not has_E

    return _result_minimal(is_correct, msg, max_chars=_MAX_FEEDBACK_CHARS)