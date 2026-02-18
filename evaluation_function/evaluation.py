# -*- coding: utf-8 -*-
"""
Minimal smoke-test evaluation function for Lambda Feedback.
Always returns "hello" and echoes basic input info.

Purpose:
- Verify deployment works
- Verify Result serialization works
- Verify UI can display feedback (avoid "general error")
"""

from __future__ import annotations

from typing import Any, List, Tuple

from lf_toolkit.evaluation import Result, Params


def _items_to_feedback_html(items: List[Tuple[Any, Any]]) -> str:
    # Simple HTML with <br> (most robust against sanitizers)
    lines: List[str] = []
    for k, v in items:
        k = "" if k is None else str(k)
        v = "" if v is None else str(v)
        if k:
            lines.append(f"{k}: {v}")
        else:
            lines.append(v)
    return "<br>".join(lines)


def evaluation_function(response: Any, answer: Any, params: Params) -> Result:
    items: List[Tuple[str, str]] = []

    # Core message
    items.append(("hello", "world"))
    items.append(("status", "smoke test OK"))

    # Echo a few things for debugging (keep it small/robust)
    try:
        items.append(("response_type", type(response).__name__))
        items.append(("answer_type", type(answer).__name__))
    except Exception:
        pass

    # If response is the image-upload format, show how many
    try:
        if isinstance(response, list):
            items.append(("response_len", str(len(response))))
            if len(response) > 0 and isinstance(response[0], dict):
                items.append(("first_keys", ", ".join(list(response[0].keys())[:10])))
                if "url" in response[0]:
                    items.append(("first_url", str(response[0]["url"])[:200]))
    except Exception:
        pass

    feedback_html = _items_to_feedback_html(items)

    # Return BOTH feedback and feedback_items for max compatibility
    try:
        return Result(is_correct=True, feedback=feedback_html, feedback_items=items)
    except TypeError:
        # Fallback for older toolkit versions
        return Result(is_correct=True, feedback_items=items)
