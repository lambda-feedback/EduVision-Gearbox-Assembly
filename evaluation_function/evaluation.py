from __future__ import annotations

import os
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


# Pipeline import guard
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

# URL / path helpers
def file_url_to_local_path(url: str) -> str:
    """
    Convert file:// URL to local path.
    Windows fix: file:///C:/x/y.jpg -> C:/x/y.jpg (remove leading '/')
    """
    parsed = urlparse(url)
    path = unquote(parsed.path)

    # Windows: "/C:/..." -> "C:/..."
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


def _cv2_bgr_to_png_bytes(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("Failed to encode image as PNG.")
    return buf.tobytes()


def _pget(params: Params, key: str, default: Any) -> Any:
    """
    Params in lf_toolkit is dict-like, but keep it safe across versions.
    """
    try:
        return params.get(key, default)  # type: ignore
    except Exception:
        try:
            return params[key]  # type: ignore
        except Exception:
            return default


def _items_to_feedback_html(items: List[Tuple[Any, Any]]) -> str:
    lines: List[str] = []
    for k, v in items:
        k = str(k).strip() if k is not None else ""
        v = str(v).strip() if v is not None else ""
        if k:
            lines.append(f"{k}: {v}")
        else:
            lines.append(v)
    return "<br>".join(lines)


def _escape_html(s: str) -> str:
    # minimal safe escaping for traceback readability
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _error_dict_to_items(err: Dict[str, str]) -> List[Tuple[str, str]]:
    """
    Convert a structured error dict into readable feedback_items.
    Includes two traceback renderings:
      - <pre> (nice if allowed)
      - <br> version (if <pre> is stripped by sanitizer)
    """
    items: List[Tuple[str, str]] = []
    items.append(("Stage", err.get("stage", "UNKNOWN")))
    items.append(("ErrorCode", err.get("error_code", "E_UNKNOWN")))
    items.append(("ExceptionType", err.get("exc_type", "")))
    items.append(("Message", err.get("message", "")))

    tb = err.get("traceback", "")
    if tb:
        safe_tb = _escape_html(tb)
        items.append(("Traceback", f"<pre>{safe_tb}</pre>"))
        items.append(("Traceback(html)", safe_tb.replace("\n", "<br>")))

    return items

# Main entry
def evaluation_function(response: Any, answer: Any, params: Params) -> Result:
    try:
        # 0) Pipeline import guard (MOST IMPORTANT)
        if run_yolo_pipeline is None:
            if isinstance(PIPELINE_IMPORT_ERROR, dict):
                items = _error_dict_to_items(PIPELINE_IMPORT_ERROR)
            else:
                items = [
                    ("Stage", "IMPORT"),
                    ("ErrorCode", "E_PIPELINE_IMPORT"),
                    ("Message", f"Pipeline import failed: {PIPELINE_IMPORT_ERROR}"),
                ]

            feedback_html = _items_to_feedback_html(items)
            try:
                return Result(is_correct=False, feedback=feedback_html, feedback_items=items)
            except TypeError:
                return Result(is_correct=False, feedback_items=items)

        # 1) Validate input
        if not isinstance(response, list) or len(response) == 0:
            items = [("Response", "Please upload at least one image.")]
            feedback_html = _items_to_feedback_html(items)
            try:
                return Result(is_correct=False, feedback=feedback_html, feedback_items=items)
            except TypeError:
                return Result(is_correct=False, feedback_items=items)

        # 2) Optional controls
        return_images: bool = bool(_pget(params, "return_images", False))
        debug: bool = bool(_pget(params, "debug", False))

        gear_model_rel = str(_pget(params, "gear_model_rel", "gear_model.pt"))
        shaft_model_rel = str(_pget(params, "shaft_model_rel", "shaft_model.pt"))

        # 3) Process images
        merged_errors: List[Dict[str, str]] = []
        merged_summaries: List[Dict[str, Any]] = []
        merged_ratios: List[Dict[str, Any]] = []
        feedback_items: List[Tuple[str, str]] = []

        for idx, item in enumerate(response):
            url = item.get("url") if isinstance(item, dict) else None
            if not url:
                merged_errors.append({"code": "E_NO_URL", "message": f"Image [{idx}] has no 'url' field."})
                continue

            img_bgr, err = _load_bgr_image_from_url(url)
            if img_bgr is None:
                merged_errors.append({
                    "code": "E_LOAD_FAIL",
                    "message": f"Failed to load image [{idx}] from URL. ({err})"
                })
                if debug:
                    feedback_items.append((f"Input URL [{idx}]", str(url)))
                continue

            # ---- Run YOLO pipeline safely per-image ----
            try:
                out = run_yolo_pipeline(
                    img_bgr=img_bgr,
                    gear_model_rel=gear_model_rel,
                    shaft_model_rel=shaft_model_rel,
                    return_images=return_images,
                )
            except Exception as e:
                msg = f"Pipeline failed on image[{idx}]: {type(e).__name__}: {e}"
                if debug:
                    msg += "\n" + traceback.format_exc()
                merged_errors.append({
                    "code": "E_PIPELINE_RUNTIME",
                    "message": msg
                })
                if debug:
                    feedback_items.append((f"Input URL [{idx}]", str(url)))
                continue

            # Collect outputs safely
            summary = out.get("summary", {})
            ratio = out.get("ratio", {})
            errors = out.get("errors", [])

            if isinstance(summary, dict):
                merged_summaries.append(summary)
            if isinstance(ratio, dict):
                merged_ratios.append(ratio)
            if isinstance(errors, list):
                merged_errors.extend(errors)

            # Optional annotated images upload (off by default)
            if return_images:
                imgs = out.get("images", None)
                if isinstance(imgs, dict) and upload_image is not None:
                    for key in ("det_img", "label_img"):
                        if key in imgs and isinstance(imgs[key], np.ndarray):
                            try:
                                png_bytes = _cv2_bgr_to_png_bytes(imgs[key])
                                img_url = upload_image(png_bytes, "eduvision")
                                feedback_items.append(
                                    (f"{key} [{idx}]", f"<a href=\"{img_url}\" target=\"_blank\">{key}</a>")
                                )
                            except ImageUploadError as e:
                                merged_errors.append({
                                    "code": "E_UPLOAD_FAIL",
                                    "message": f"Failed to upload {key} for image[{idx}]: {e}"
                                })
                            except Exception as e:
                                merged_errors.append({
                                    "code": "E_UPLOAD_FAIL",
                                    "message": f"Failed to encode/upload {key} for image[{idx}]: {e}"
                                })
                elif upload_image is None and debug:
                    feedback_items.append(("Images", "return_images=True but upload_image() is not available in this lf_toolkit version."))

            if debug:
                feedback_items.append((f"Input URL [{idx}]", str(url)))

        # 4) Decide correctness
        has_E = any(str(e.get("code", "")).startswith("E_") for e in merged_errors)
        is_correct = (not has_E)

        # 5) Text feedback
        if merged_summaries:
            feedback_items.append(("Summary", str(merged_summaries[-1])))

        if merged_ratios:
            feedback_items.append(("Ratio", str(merged_ratios[-1])))

        if merged_errors:
            lines = [f"- {e.get('code', 'E_ERR')}: {e.get('message', '')}" for e in merged_errors]
            feedback_items.append(("Issues", "\n".join(lines)))

        if not feedback_items:
            feedback_items = [("Result", "No valid images could be processed.")]

        feedback_html = _items_to_feedback_html(feedback_items)

        try:
            return Result(is_correct=is_correct, feedback=feedback_html, feedback_items=feedback_items)
        except TypeError:
            return Result(is_correct=is_correct, feedback_items=feedback_items)

    except Exception as e:
        # Absolute last-resort: never crash the platform UI
        tb = traceback.format_exc()
        safe_tb = _escape_html(tb)
        items = [
            ("Stage", "UNHANDLED"),
            ("ErrorCode", "E_UNHANDLED"),
            ("ExceptionType", type(e).__name__),
            ("Message", str(e)),
            ("Traceback", f"<pre>{safe_tb}</pre>"),
            ("Traceback(html)", safe_tb.replace("\n", "<br>")),
        ]
        feedback_html = _items_to_feedback_html(items)
        try:
            return Result(is_correct=False, feedback=feedback_html, feedback_items=items)
        except TypeError:
            return Result(is_correct=False, feedback_items=items)
