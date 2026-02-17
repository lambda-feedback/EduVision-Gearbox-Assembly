from __future__ import annotations

import os
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

from .yolo_pipeline import run_yolo_pipeline


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

def _items_to_feedback_html(items):
    lines = []
    for k, v in items:
        k = str(k).strip() if k is not None else ""
        v = str(v).strip() if v is not None else ""
        if k:
            lines.append(f"{k}: {v}")
        else:
            lines.append(v)
    return "<br>".join(lines)


# Main entry
def evaluation_function(response: Any, answer: Any, params: Params) -> Result:
    try:
        # Validate input
        if not isinstance(response, list) or len(response) == 0:
            items = [("Response", "Please upload at least one image.")]
            feedback_html = _items_to_feedback_html(items)
            try:
                return Result(is_correct=False, feedback=feedback_html, feedback_items=items)
            except TypeError:
                return Result(is_correct=False, feedback_items=items)

        # Optional controls (safe defaults)
        # Default: NO image upload (text only), to match your current goal
        return_images: bool = bool(_pget(params, "return_images", False))
        debug: bool = bool(_pget(params, "debug", False))

        # Relative model filenames stored in evaluation_function/
        gear_model_rel = str(_pget(params, "gear_model_rel", "gear_model.pt"))
        shaft_model_rel = str(_pget(params, "shaft_model_rel", "shaft_model.pt"))


        # Process images
        merged_errors: List[Dict[str, str]] = []
        merged_summaries: List[Dict[str, Any]] = []
        merged_ratios: List[Dict[str, Any]] = []

        feedback_items: List[Tuple[str, str]] = []

        for idx, item in enumerate(response):
            url = item.get("url") if isinstance(item, dict) else None
            if not url:
                merged_errors.append({
                    "code": "NO_URL",
                    "message": f"Image [{idx}] has no 'url' field."
                })
                continue

            img_bgr, err = _load_bgr_image_from_url(url)
            if img_bgr is None:
                merged_errors.append({
                    "code": "LOAD_FAIL",
                    "message": f"Failed to load image [{idx}] from URL. ({err})"
                })
                if debug:
                    feedback_items.append((f"Input URL [{idx}]", str(url)))
                continue

            # Run pipeline (your external YOLO pipeline)
            out = run_yolo_pipeline(
                img_bgr=img_bgr,
                gear_model_rel=gear_model_rel,
                shaft_model_rel=shaft_model_rel,
                return_images=return_images,
            )

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

            # Optional annotated images upload (disabled by default)
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
                                    "code": "UPLOAD_FAIL",
                                    "message": f"Failed to upload {key} for image[{idx}]: {e}"
                                })
                            except Exception as e:
                                merged_errors.append({
                                    "code": "UPLOAD_FAIL",
                                    "message": f"Failed to encode/upload {key} for image[{idx}]: {e}"
                                })
                elif upload_image is None and debug:
                    feedback_items.append(("Images", "return_images=True but upload_image() is not available in this lf_toolkit version."))

            if debug:
                feedback_items.append((f"Input URL [{idx}]", str(url)))

        # Decide correctness
        # Your rule: incorrect if any error code starts with "E_"
        has_E = any(str(e.get("code", "")).startswith("E_") for e in merged_errors)
        is_correct = (not has_E)

        # Text feedback
        if merged_summaries:
            feedback_items.append(("Summary", str(merged_summaries[-1])))

        if merged_ratios:
            feedback_items.append(("Ratio", str(merged_ratios[-1])))

        if merged_errors:
            lines = [f"- {e.get('code', 'ERR')}: {e.get('message', '')}" for e in merged_errors]
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
        items = [("Error", f"{type(e).__name__}: {e}")]
        feedback_html = _items_to_feedback_html(items)
        try:
            return Result(is_correct=False, feedback=feedback_html, feedback_items=items)
        except TypeError:
            return Result(is_correct=False, feedback_items=items)
