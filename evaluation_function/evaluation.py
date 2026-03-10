#Update precheck to use gear-contact count consistency 2
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, unquote

import cv2
import numpy as np
import requests

from lf_toolkit.evaluation import Result, Params

# ----------------------------
# Pipeline import
# ----------------------------
run_yolo_pipeline = None

try:
    from .yolo_pipeline import run_yolo_pipeline  # type: ignore
except Exception:
    run_yolo_pipeline = None


# ----------------------------
# Output caps
# ----------------------------
_MAX_FEEDBACK_CHARS = int(os.environ.get("LF_MAX_FEEDBACK_CHARS", "1200"))

# Stage target for experiment goal
TARGET_STAGE_COUNT = int(os.environ.get("LF_TARGET_STAGE_COUNT", "6"))


# ----------------------------
# Student-facing message policy
# ----------------------------
MESSAGE_POLICY: Dict[str, Dict[str, str]] = {
    "precheck": {
        "pass": "Good photo. You can proceed.",
        "fail": "Please retake the photo. The image is not suitable for reliable checking.",
    },
    "shaft": {
        "pass": "Good. The shaft setup looks correct.",
        "count_fail": "A shaft may be missing or not detected. Please check whether both shafts are installed and retake the photo if needed.",
        "type_confusion": "The shaft types could not be identified reliably. Please retake the photo from a clearer angle and make sure both shafts are fully visible.",
        "position_swap": "The shaft positions appear to be incorrect. Please make sure the short shaft is closer to the white gear and the long shaft is farther away.",
        "fail": "Please check the shaft setup again.",
    },
    "spacer": {
        "pass": "Good. The spacer setup looks correct.",
        "short_missing": "The short spacer may be missing or not detected. Please check whether the short spacer is installed and retake the photo if needed.",
        "long_missing": "The long spacer may be missing or not detected. Please check whether the long spacer is installed and retake the photo if needed.",
        "count_fail": "A spacer may be missing or not detected. Please check whether both spacers are installed and retake the photo if needed.",
        "type_confusion": "The spacer types could not be identified reliably. Please retake the photo from a clearer angle and make sure both spacers are fully visible.",
        "position_mismatch": "The spacer positions appear to be incorrect. Please make sure the short spacer is on the short shaft and the long spacer is on the long shaft.",
        "distance_order": "The spacer order appears to be incorrect. Please check whether the short spacer is closer to gear 1 than the long spacer.",
        "fail": "Please check the spacer setup again.",
    },
    "gear_inventory": {
        "pass": "Good. The gear count looks correct.",
        "fail": "Please check the gears again. The number or type of detected gears does not match the expected assembly.",
    },
    "mesh_ratio": {
        "fail": "Please check the gear meshing again. The detected gear contacts do not match the expected setup.",
        "calc_fail": "The assembly was detected, but the gear ratio could not be calculated reliably. Please check the gear arrangement again.",
        "stage_below_target": (
            "Detected stages: {num_stages}.\n"
            "Total gear ratio: {R_total}.\n"
            "Output speed: {out_rpm} RPM.\n"
            "Your current number of stages may not be enough to achieve the experiment goal of lifting the 1 kg bottle. Please consider adding more gears."
        ),
        "stage_at_or_above_target": (
            "Detected stages: {num_stages}.\n"
            "Total gear ratio: {R_total}.\n"
            "Output speed: {out_rpm} RPM.\n"
            "Good. Your gearbox has reached the target stage count. You can now test whether it can lift the 1 kg bottle."
        ),
    },
}


def _pget(params: Params, key: str, default: Any) -> Any:
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


def _truncate_text(s: str, max_chars: int) -> str:
    s = s.strip()
    if len(s) > max_chars:
        s = s[:max_chars] + " ... (truncated)"
    return s


def _result_minimal(is_correct: bool, message: str, *, max_chars: int = _MAX_FEEDBACK_CHARS) -> Result:
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


def _select_errors_by_task(errors: List[Dict[str, Any]], task: str) -> List[Dict[str, Any]]:
    task = (task or "full").strip().lower()

    if task == "full":
        task = "mesh_ratio"

    def keep(e: Dict[str, Any]) -> bool:
        code = str(e.get("code", "")).upper()

        if task == "precheck":
            return code in {
                "E_PRECHECK_COUNT_RULE_FAIL",
            }

        if task == "shaft":
            return code.startswith("E_SHAFT") or code == "E_NO_SHAFTS" or code == "E_NO_GEAR11"

        if task == "spacer":
            return (
                code.startswith("E_SPACER")
                or code == "E_NO_SHAFTS"
                or code == "E_NO_GEARS"
                or code == "E_NO_GEAR11"
            )

        if task == "gear_inventory":
            return code in {"E_NO_GEARS", "E_GEAR_COUNT_MISMATCH", "E_GEAR_COUNT_UNSUPPORTED"}

        if task == "mesh_ratio":
            return (
                ("MESH" in code)
                or code.startswith("E_CONTACT_COUNT")
                or code == "E_NO_GEARS"
                or code == "E_GEAR_COUNT_UNSUPPORTED"
                or code.startswith("E_SHAFT")
                or code.startswith("E_SPACER")
            )

        return True

    return [e for e in errors if isinstance(e, dict) and keep(e)]


def _has_pipeline_error(errors: List[Dict[str, Any]]) -> bool:
    return any(
        isinstance(e, dict) and str(e.get("code", "")).startswith("E_")
        for e in errors
    )


def _format_stage_value(num_stages: Any) -> str:
    try:
        return str(int(num_stages))
    except Exception:
        return str(num_stages)


def _format_ratio_value(r_total: Any) -> str:
    try:
        return f"{float(r_total):.3f}"
    except Exception:
        return str(r_total)


def _format_rpm_value(out_rpm: Any) -> str:
    try:
        return f"{float(out_rpm):.1f}"
    except Exception:
        return str(out_rpm)


def _build_student_message(
    *,
    task: str,
    img_bgr: np.ndarray,
    out: Dict[str, Any],
    errors: List[Dict[str, Any]],
    selected_errors: List[Dict[str, Any]],
) -> Tuple[bool, str]:
    task = (task or "full").strip().lower()

    if task == "full":
        task = "mesh_ratio"

    if task not in ("precheck", "shaft", "spacer", "gear_inventory", "mesh_ratio"):
        task = "mesh_ratio"

    if task == "precheck":
        has_fail = any(
            str(e.get("code", "")).upper() == "E_PRECHECK_COUNT_RULE_FAIL"
            for e in selected_errors
            if isinstance(e, dict)
        )
        if has_fail:
            return False, MESSAGE_POLICY["precheck"]["fail"]
        return True, MESSAGE_POLICY["precheck"]["pass"]

    is_correct = not _has_pipeline_error(errors)

    if task == "shaft":
        codes = {
            str(e.get("code", "")).upper()
            for e in selected_errors
            if isinstance(e, dict)
        }

        if (
            "E_SHAFT_COUNT_MISMATCH" in codes
            or "E_NO_SHAFTS" in codes
            or "E_SHAFT2_NOT_FOUND" in codes
        ):
            return False, MESSAGE_POLICY["shaft"]["count_fail"]

        if (
            "E_SHAFT_TYPE_CONFUSION" in codes
            or "E_SHAFT2_CLASS_MISMATCH" in codes
        ):
            return False, MESSAGE_POLICY["shaft"]["type_confusion"]

        if "E_SHAFT_POSITION_SWAP" in codes:
            return False, MESSAGE_POLICY["shaft"]["position_swap"]

        if not is_correct:
            return False, MESSAGE_POLICY["shaft"]["fail"]

        return True, MESSAGE_POLICY["shaft"]["pass"]

    if task == "spacer":
        codes = {
            str(e.get("code", "")).upper()
            for e in selected_errors
            if isinstance(e, dict)
        }

        if "E_SPACER_SHORT_MISSING" in codes:
            return False, MESSAGE_POLICY["spacer"]["short_missing"]

        if "E_SPACER_LONG_MISSING" in codes:
            return False, MESSAGE_POLICY["spacer"]["long_missing"]

        if "E_SPACER_COUNT_MISMATCH" in codes:
            return False, MESSAGE_POLICY["spacer"]["count_fail"]

        if "E_SPACER_TYPE_CONFUSION" in codes:
            return False, MESSAGE_POLICY["spacer"]["type_confusion"]

        if (
            "E_SPACER_POSITION_MISMATCH" in codes
            or "E_SPACER2_TYPE_MISMATCH" in codes
            or "E_SPACER3_TYPE_MISMATCH" in codes
        ):
            return False, MESSAGE_POLICY["spacer"]["position_mismatch"]

        if "E_SPACER_DISTANCE_ORDER" in codes:
            return False, MESSAGE_POLICY["spacer"]["distance_order"]

        if not is_correct:
            return False, MESSAGE_POLICY["spacer"]["fail"]

        return True, MESSAGE_POLICY["spacer"]["pass"]

    if task == "gear_inventory":
        codes = {
            str(e.get("code", "")).upper()
            for e in selected_errors
            if isinstance(e, dict)
        }

        if (
            "E_NO_GEARS" in codes
            or "E_GEAR_COUNT_MISMATCH" in codes
            or "E_GEAR_COUNT_UNSUPPORTED" in codes
        ):
            return False, MESSAGE_POLICY["gear_inventory"]["fail"]

        if not is_correct:
            return False, MESSAGE_POLICY["gear_inventory"]["fail"]

        return True, MESSAGE_POLICY["gear_inventory"]["pass"]

    if task == "mesh_ratio":
        if not is_correct:
            return False, MESSAGE_POLICY["mesh_ratio"]["fail"]

        ratio = out.get("ratio") if isinstance(out.get("ratio"), dict) else {}
        num_stages = ratio.get("num_stages")
        r_total = ratio.get("R_total")
        out_rpm = ratio.get("out_rpm")

        if num_stages is None or r_total is None or out_rpm is None:
            return False, MESSAGE_POLICY["mesh_ratio"]["calc_fail"]

        stage_text = _format_stage_value(num_stages)
        ratio_text = _format_ratio_value(r_total)
        rpm_text = _format_rpm_value(out_rpm)

        try:
            stage_num = int(num_stages)
        except Exception:
            stage_num = None

        if stage_num is not None and stage_num >= TARGET_STAGE_COUNT:
            msg = MESSAGE_POLICY["mesh_ratio"]["stage_at_or_above_target"].format(
                num_stages=stage_text,
                R_total=ratio_text,
                out_rpm=rpm_text,
            )
            return True, msg

        msg = MESSAGE_POLICY["mesh_ratio"]["stage_below_target"].format(
            num_stages=stage_text,
            R_total=ratio_text,
            out_rpm=rpm_text,
        )
        return True, msg

    key = "pass" if is_correct else "fail"
    return is_correct, MESSAGE_POLICY[task][key]


def evaluation_function(response: Any, answer: Any, params: Params) -> Result:
    task = str(_pget(params, "task", "full") or "full").strip().lower()

    if task == "full":
        pipeline_task = "full"
        message_task = "mesh_ratio"
    else:
        pipeline_task = task
        message_task = task

    return_images: bool = bool(_pget(params, "return_images", pipeline_task != "precheck"))
    gear_model_rel = str(_pget(params, "gear_model_rel", "gear_model.pt"))
    shaft_model_rel = str(_pget(params, "shaft_model_rel", "shaft_model.pt"))
    expected_gears = _pget(params, "expected_gears", None)

    if not isinstance(response, list) or len(response) == 0:
        return _result_minimal(False, "Please upload at least one image.")

    first = response[0] if isinstance(response[0], dict) else None
    url = first.get("url") if isinstance(first, dict) else None
    if not url:
        return _result_minimal(False, "The uploaded image could not be read. Please upload the image again.")

    img_bgr, err = _load_bgr_image_from_url(str(url))
    if img_bgr is None:
        return _result_minimal(False, "The image could not be loaded. Please upload the image again.")

    if run_yolo_pipeline is None:
        return _result_minimal(
            False,
            "The system is temporarily unavailable. Please try again later.",
        )

    try:
        out: Dict[str, Any] = run_yolo_pipeline(  # type: ignore[misc]
            img_bgr=img_bgr,
            gear_model_rel=gear_model_rel,
            shaft_model_rel=shaft_model_rel,
            return_images=(return_images and pipeline_task != "precheck"),
            task=pipeline_task,
            expected_gears=expected_gears,
        )
    except TypeError:
        out = run_yolo_pipeline(  # type: ignore[misc]
            img_bgr=img_bgr,
            gear_model_rel=gear_model_rel,
            shaft_model_rel=shaft_model_rel,
            return_images=(return_images and pipeline_task != "precheck"),
        )
    except Exception:
        return _result_minimal(
            False,
            "The image could not be checked successfully. Please upload the image again.",
        )

    errors = out.get("errors") if isinstance(out.get("errors"), list) else []
    selected_errors = _select_errors_by_task(errors, message_task)

    is_correct, msg = _build_student_message(
        task=message_task,
        img_bgr=img_bgr,
        out=out,
        errors=errors,
        selected_errors=selected_errors,
    )
    return _result_minimal(is_correct, msg)