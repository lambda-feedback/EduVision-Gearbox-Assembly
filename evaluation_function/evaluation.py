# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Any, List, Tuple, Optional
from urllib.parse import urlparse, unquote

import cv2

from lf_toolkit.evaluation import Result, Params


def _mk_result(is_correct: bool, items: List[Tuple[str, str]]) -> Result:
    # feedback can be str or list in toolkit; tests accept both.
    feedback = "<br>".join([f"{k}: {v}" for k, v in items])
    try:
        return Result(is_correct=is_correct, feedback=feedback, feedback_items=items)
    except TypeError:
        return Result(is_correct=is_correct, feedback_items=items)


def _file_url_to_local_path(url: str) -> str:
    """
    Convert file:// URL to local path.
    Linux example: file:///tmp/a.jpg -> /tmp/a.jpg
    Windows example: file:///C:/x/y.jpg -> C:/x/y.jpg
    """
    parsed = urlparse(url)
    path = unquote(parsed.path)

    # Windows fix: "/C:/..." -> "C:/..."
    if os.name == "nt" and len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return path


def _load_bgr_image_from_url(url: str) -> Tuple[Optional[Any], Optional[str]]:
    """
    Minimal loader for tests:
    - supports file:// for local tests
    - supports http/https (not used in tests but harmless)
    Returns (img_bgr, error_message)
    """
    try:
        if not isinstance(url, str) or not url:
            return None, "Empty URL"

        if url.startswith("file://"):
            local_path = _file_url_to_local_path(url)
            img = cv2.imread(local_path, cv2.IMREAD_COLOR)
            if img is None:
                return None, f"cv2.imread failed for local_path='{local_path}'"
            return img, None

        # For this minimal version, we won't actually fetch remote.
        # If you want, you can add requests later.
        if url.startswith("http://") or url.startswith("https://"):
            return None, "Remote URL not supported in minimal test version"

        return None, f"Unsupported URL scheme: {url}"

    except Exception as e:
        return None, str(e)


def evaluation_function(response: Any, answer: Any, params: Params) -> Result:
    debug = False
    try:
        debug = bool(getattr(params, "debug", False) or (isinstance(params, dict) and params.get("debug")))
    except Exception:
        debug = False

    # ---- Test requirement 1: missing image must be is_correct=False ----
    if not isinstance(response, list) or len(response) == 0:
        items = [
            ("MISSING_IMAGE", "Please upload at least one image."),
        ]
        return _mk_result(is_correct=False, items=items)

    # Must be list[dict] with "url"
    first = response[0]
    url = first.get("url") if isinstance(first, dict) else None
    if not isinstance(url, str) or not url:
        items = [
            ("LOAD_FAIL", "Image item has no valid 'url' field."),
        ]
        return _mk_result(is_correct=False, items=items)

    # ---- Load image (this is what the tests exercise) ----
    img_bgr, err = _load_bgr_image_from_url(url)

    # ---- Test requirement 2: bad url must include LOAD_FAIL in feedback ----
    if img_bgr is None:
        items = [
            ("LOAD_FAIL", f"Failed to load image from URL. ({err})"),
        ]
        if debug:
            items.append(("InputURL", url))
        return _mk_result(is_correct=False, items=items)

    # ---- Success path for local file url test ----
    h, w = img_bgr.shape[:2]
    items = [
        ("OK", "smoke test OK (no YOLO)"),
        ("ImageShape", f"{w}x{h}"),
    ]
    if debug:
        items.append(("InputURL", url))
    return _mk_result(is_correct=True, items=items)
