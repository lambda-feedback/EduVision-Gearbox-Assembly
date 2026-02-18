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


def file_url_to_local_path(url: str) -> str:
    parsed = urlparse(url)
    path = unquote(parsed.path)
    # Windows: "/C:/..." -> "C:/..."
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


def evaluation_function(response: Any, answer: Any, params: Params) -> Result:
    """
    SMOKE TEST VERSION (NO YOLO)
    - Designed to satisfy evaluation_test.py expectations:
      * empty response -> is_correct=False
      * bad url -> feedback contains LOAD_FAIL
    """
    items: List[Tuple[str, str]] = []

    try:
        fast_return: bool = bool(_pget(params, "fast_return", True))
        echo: bool = bool(_pget(params, "echo", True))
        try_fetch: bool = bool(_pget(params, "try_fetch", False))
        debug: bool = bool(_pget(params, "debug", True))

        items.append(("SMOKE", "Hello / evaluation_function reached ✅"))
        items.append(("fast_return", str(fast_return)))
        items.append(("echo", str(echo)))
        items.append(("try_fetch", str(try_fetch)))
        items.append(("debug", str(debug)))

        # 1) Validate input
        if not isinstance(response, list) or len(response) == 0:
            items.append(("BAD_INPUT", "No images uploaded."))
            html = _items_to_html(items)
            try:
                return Result(is_correct=False, feedback=html, feedback_items=items)
            except TypeError:
                return Result(is_correct=False, feedback_items=items)

        # 2) Extract first URL (tests use response[0]["url"])
        first = response[0] if isinstance(response, list) and len(response) > 0 else None
        url = first.get("url") if isinstance(first, dict) else None

        if echo:
            items.append(("response_type", type(response).__name__))
            items.append(("response_len", str(len(response))))
            items.append(("first_item_type", type(first).__name__))
            if isinstance(first, dict):
                items.append(("first_item_keys", ", ".join(sorted([str(k) for k in first.keys()]))))
                items.append(("first_url", str(url)))

        # 3) IMPORTANT:
        #    To satisfy unit tests, if a URL exists we ALWAYS attempt a lightweight load check
        #    (so bad file:// paths produce LOAD_FAIL).
        if not url:
            items.append(("LOAD_FAIL", "LOAD_FAIL: first image has no url field"))
            html = _items_to_html(items)
            try:
                return Result(is_correct=False, feedback=html, feedback_items=items)
            except TypeError:
                return Result(is_correct=False, feedback_items=items)

        # If user explicitly wants "fast_return" behavior on platform,
        # they can set try_fetch=False AND also set skip_load_check=True.
        # Default is to run the load-check for CI correctness.
        skip_load_check: bool = bool(_pget(params, "skip_load_check", False))

        if (not skip_load_check) or try_fetch:
            img, err = _load_bgr_image_from_url(str(url))
            if img is None:
                items.append(("LOAD_FAIL", f"LOAD_FAIL: Failed to load image. ({err})"))
                items.append(("url", str(url)))
                html = _items_to_html(items)
                try:
                    return Result(is_correct=False, feedback=html, feedback_items=items)
                except TypeError:
                    return Result(is_correct=False, feedback_items=items)

            h, w = img.shape[:2]
            items.append(("image_loaded", "OK"))
            items.append(("shape", f"{w}x{h}"))

        # 4) Now optionally exit early (platform smoke)
        if fast_return and not try_fetch:
            items.append(("note", "fast_return=True (no YOLO). Load-check already done."))
            html = _items_to_html(items)
            try:
                return Result(is_correct=False, feedback=html, feedback_items=items)
            except TypeError:
                return Result(is_correct=False, feedback_items=items)

        # 5) Final message
        items.append(("note", "This build does NOT run YOLO. If you see this, platform execution is fine."))
        html = _items_to_html(items)
        try:
            return Result(is_correct=False, feedback=html, feedback_items=items)
        except TypeError:
            return Result(is_correct=False, feedback_items=items)

    except Exception as e:
        tb = _escape_html(traceback.format_exc())
        items.append(("UNHANDLED", f"{type(e).__name__}: {e}"))
        items.append(("TRACEBACK", tb.replace("\n", "<br>")))
        html = _items_to_html(items)
        try:
            return Result(is_correct=False, feedback=html, feedback_items=items)
        except TypeError:
            return Result(is_correct=False, feedback_items=items)
