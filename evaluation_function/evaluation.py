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
TARGET_STAGE_COUNT = int(os.environ.get("LF_TARGET_STAGE_COUNT", "6"))


# ----------------------------
# Student-facing message policy
# ----------------------------
MESSAGE_POLICY: Dict[str, Dict[str, str]] = {
    "parts_inventory": {
        "fail": "Some parts could not be detected clearly. Please separate the parts and retake the photo.",
    },
    "precheck": {
        "report": (
            "Photo quality score: {quality_score}/100.\n"
            "Brightness: {brightness_score}/100\n"
            "Contrast: {contrast_score}/100\n"
            "Sharpness: {sharpness_score}/100\n"
            "Noise control: {noise_score}/100\n"
            "{quality_advice}"
        ),
        "fail": (
            "Photo quality score: {quality_score}/100.\n"
            "Brightness: {brightness_score}/100\n"
            "Contrast: {contrast_score}/100\n"
            "Sharpness: {sharpness_score}/100\n"
            "Noise control: {noise_score}/100\n"
            "{quality_advice}"
        ),
    },
    "single_stage": {
        "pass": (
            "Good. A valid single-stage setup was detected.\n"
            "Detected stages: {num_stages}\n"
            "Gear ratio: {R_total}\n"
            "Output speed: {out_rpm} RPM"
        ),
        "no_gears": (
            "No gears were detected clearly. Please make sure the gears are installed correctly "
            "and clearly visible in the photo."
        ),
        "gear_missing": (
            "Some gears may be missing or not clearly visible. Please check the gear setup and "
            "retake the photo from a clearer angle."
        ),
        "gear_overloaded": (
            "The gear setup seems to contain extra gears. Please try reducing the number of gears, "
            "and remove any gears that are not being used or are not part of the assembly from the photo."
        ),
        "no_spacer": (
            "No spacer was detected. Please make sure the spacer is installed correctly and clearly visible in the photo."
        ),
        "extra_spacer": (
            "Too many spacers were detected. Please remove any spacer that is not needed from the assembly or from the photo."
        ),
        "no_shaft": (
            "No shaft was detected. Please make sure the shaft is installed correctly and clearly visible in the photo."
        ),
        "extra_shaft": (
            "Too many shafts were detected. Please remove any shaft that is not needed from the assembly or from the photo."
        ),
        "mismesh": (
            "The gears do not appear to mesh correctly. Please check the gear engagement and remove any gear "
            "that is not properly participating in the transmission."
        ),
        "stage_fail": (
            "A clear single-stage transmission could not be confirmed. Please simplify the setup and "
            "make sure only the intended transmission parts are visible."
        ),
        "hint_long_spacer": (
            "The setup works, but think about whether a short spacer or a long spacer is more appropriate here."
        ),
        "hint_short_shaft": (
            "The setup works, but think about whether a long shaft or a short shaft is the better choice here."
        ),
        "fail": "Please check the single-stage setup again. A valid one-stage transmission could not be confirmed.",
    },
    "shaft": {
        "pass": "Good. The shaft setup looks correct.",
        "none_detected": "No shafts were detected clearly. Please make sure one short shaft and one long shaft are installed and clearly visible in the photo.",
        "short_missing": "The short shaft may be missing or not detected. Please check whether the short shaft is installed and clearly visible in the photo.",
        "long_missing": "The long shaft may be missing or not detected. Please check whether the long shaft is installed and clearly visible in the photo.",
        "too_many": "Too many shafts were detected. Please make sure only the required short shaft and long shaft are visible in the photo.",
        "count_fail": "A shaft may be missing or not detected. Please check whether both shafts are installed and retake the photo if needed.",
        "type_confusion": "The shaft types could not be identified reliably. Please retake the photo from a clearer angle and make sure both shafts are fully visible.",
        "position_swap": "The shaft positions appear to be incorrect. Please make sure the short shaft is closer to the white gear and the long shaft is farther away.",
        "fail": "Please check the shaft setup again.",
    },
    "spacer": {
        "pass": "Good. The spacer setup looks correct.",
        "none_detected": "No spacers were detected clearly. Please make sure both spacers are installed and clearly visible in the photo.",
        "short_missing": "The short spacer may be missing or not detected. Please check whether the short spacer is installed and retake the photo if needed.",
        "long_missing": "The long spacer may be missing or not detected. Please check whether the long spacer is installed and retake the photo if needed.",
        "too_many": "Too many spacers were detected. Please make sure only the required short spacer and long spacer are visible in the photo.",
        "count_fail": "A spacer may be missing or not detected. Please check whether both spacers are installed and retake the photo if needed.",
        "type_confusion": "The spacer types could not be identified reliably. Please retake the photo from a clearer angle and make sure both spacers are fully visible.",
        "assignment_fail": "The spacers were detected, but at least one spacer could not be reliably assigned to a shaft. Please make sure both spacers are placed on the shafts. If they are already on the shafts, retake the photo from a clear top view with both shafts and spacers fully visible.",
        "position_mismatch": "The spacer positions appear to be incorrect. Please make sure the short spacer is on the short shaft and the long spacer is on the long shaft.",
        "distance_order": "The spacer order appears to be incorrect. Please check whether the short spacer is closer to white gear than the long spacer.",
        "fail": "Please check the spacer setup again.",
    },
    "gear_inventory": {
        "pass": (
            "Detected gears:\n"
            "- driving gear: {driving_gear}\n"
            "- small gear: {smallgear}\n"
            "- big gear: {biggear}\n"
            "Good. No obvious gear mismatch was detected."
        ),
        "contact_consistency_fail": (
            "Detected gears:\n"
            "- driving gear: {driving_gear}\n"
            "- small gear: {smallgear}\n"
            "- big gear: {biggear}\n"
            "The system cannot reliably evaluate the gear meshing due to unclear or inconsistent detection results. "
            "Please ensure that only the required gears are visible in the image, and that the gear meshing areas "
            "are clearly shown before retaking the photo."
        ),
        "big_small_inconsistent": (
            "Detected gears:\n"
            "- driving gear: {driving_gear}\n"
            "- small gear: {smallgear}\n"
            "- big gear: {biggear}\n"
            "The detected numbers of big gears and small gears are not consistent. "
            "Please check the gear setup and retake the photo."
        ),
        "mismatch_fail": (
            "Detected gears:\n"
            "- driving gear: {driving_gear}\n"
            "- small gear: {smallgear}\n"
            "- big gear: {biggear}\n"
            "A gear mismatch was detected. Please adjust the gear positions so that the gears mesh correctly."
        ),
        "no_gears": (
            "Detected gears:\n"
            "- driving gear: {driving_gear}\n"
            "- small gear: {smallgear}\n"
            "- big gear: {biggear}\n"
            "No gears were detected clearly. Please retake the photo."
        ),
        "fail": (
            "Detected gears:\n"
            "- driving gear: {driving_gear}\n"
            "- small gear: {smallgear}\n"
            "- big gear: {biggear}\n"
            "Please check the gears again."
        ),
    },
    "mesh_ratio": {
        "fail": "Please check the assembly again. The detected setup is not correct enough for reliable gear-ratio calculation.",
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


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default


def _select_errors_by_task(errors: List[Dict[str, Any]], task: str) -> List[Dict[str, Any]]:
    task = (task or "full").strip().lower()

    if task == "full":
        task = "mesh_ratio"

    def keep(e: Dict[str, Any]) -> bool:
        code = str(e.get("code", "")).upper()

        if task == "parts_inventory":
            return code in {
                "E_PARTS_COUNT_MISMATCH",
                "E_BAD_PART_TYPE",
                "E_NO_TARGET_PARTS",
            }

        if task == "precheck":
            return code in {
                "E_PRECHECK_BIG_SMALL_INCONSISTENT",
                "E_PHOTO_QUALITY_LOW",
                "E_NO_GEARS",
            }

        if task == "single_stage":
            return code.startswith("E_SINGLE_STAGE") or code == "E_NO_GEARS"

        if task == "shaft":
            return (
                code.startswith("E_SHAFT")
                or code == "E_NO_SHAFTS"
                or code == "E_NO_GEAR11"
                or code == "E_NO_GEARS"
            )

        if task == "spacer":
            return (
                code.startswith("E_SPACER")
                or code.startswith("E_ASSIGN")
                or code.startswith("E_PARTS")
                or code == "E_BAD_PART_TYPE"
                or code == "E_NO_TARGET_PARTS"
                or code == "E_NO_SHAFTS"
                or code == "E_NO_GEAR11"
                or code == "E_NO_GEARS"
            )

        if task == "gear_inventory":
            return code in {
                "E_NO_GEARS",
                "E_MESH_MISMATCH",
                "E_GEAR_CONTACT_INCONSISTENT",
                "E_GEAR_BIG_SMALL_INCONSISTENT",
            }

        if task == "mesh_ratio":
            return (
                ("MESH" in code)
                or code.startswith("E_CONTACT_COUNT")
                or code == "E_NO_GEARS"
                or code == "E_NO_SHAFTS"
                or code == "E_NO_GEAR11"
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


def _has_selected_pipeline_error(selected_errors: List[Dict[str, Any]]) -> bool:
    return any(
        isinstance(e, dict) and str(e.get("code", "")).startswith("E_")
        for e in selected_errors
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


def _get_counts_dict(out: Dict[str, Any]) -> Dict[str, Any]:
    return out.get("counts", {}) if isinstance(out.get("counts"), dict) else {}


def _get_gear_counts(out: Dict[str, Any]) -> Tuple[int, int, int]:
    counts = _get_counts_dict(out)

    driving_gear = _safe_int(
        counts.get("driving_gear", counts.get("drivinggear", counts.get("gear11", 0)))
    )
    smallgear = _safe_int(counts.get("smallgear", 0))
    biggear = _safe_int(counts.get("biggear", 0))

    return driving_gear, smallgear, biggear


def _get_shaft_counts(out: Dict[str, Any]) -> Tuple[int, int, int]:
    counts = out.get("shaft_counts", {}) if isinstance(out.get("shaft_counts"), dict) else {}
    if not counts:
        counts = out.get("counts", {}) if isinstance(out.get("counts"), dict) else {}

    n_long = _safe_int(counts.get("shaft_long", 0))
    n_short = _safe_int(counts.get("shaft_short", 0))
    n_total = n_long + n_short

    return n_long, n_short, n_total


def _get_spacer_counts(out: Dict[str, Any]) -> Tuple[int, int, int]:
    counts = out.get("spacer_counts", {}) if isinstance(out.get("spacer_counts"), dict) else {}
    if not counts:
        counts = out.get("counts", {}) if isinstance(out.get("counts"), dict) else {}

    n_long = _safe_int(counts.get("spacer_long", 0))
    n_short = _safe_int(counts.get("spacer_short", 0))
    n_total = n_long + n_short

    return n_long, n_short, n_total


def _build_parts_inventory_message(out: Dict[str, Any], part_type: str) -> Tuple[bool, str]:
    counts = _get_counts_dict(out)
    errors = out.get("errors", []) if isinstance(out.get("errors"), list) else []

    is_correct = len(errors) == 0
    part_type = str(part_type or "").strip().lower()

    if part_type == "gear":
        driving_count = _safe_int(
            counts.get("Driving_Gear", counts.get("driving_gear", counts.get("drivinggear", 0)))
        )
        big_count = _safe_int(
            counts.get("Gear_big", counts.get("biggear", 0))
        )
        small_count = _safe_int(
            counts.get("Gear_small", counts.get("smallgear", 0))
        )
        summary = out.get("summary", {}) if isinstance(out.get("summary"), dict) else {}
        shaft_count = _safe_int(summary.get("shafts", 0))

        counts_text = (
            f"Driving gear: {driving_count}\n"
            f"Big gears: {big_count}\n"
            f"Small gears: {small_count}"
        )

        if shaft_count > 0:
            feedback = (
                "The image contains shafts, which are not required for this task. "
                "Please remove the shafts and any attached gear assemblies (e.g. orange gears) "
                "and upload a photo showing only the gears."
            )
            return False, f"{counts_text}\n{feedback}"

        if driving_count == 0:
            feedback = (
                "No driving gear was detected. Please include the white driving gear clearly in the photo."
            )
            return False, f"{counts_text}\n{feedback}"

        if driving_count > 1:
            feedback = (
                "More than one driving gear was detected. Please upload an image containing only one white driving gear."
            )
            return False, f"{counts_text}\n{feedback}"

        if big_count != small_count:
            feedback = (
                "The detected green idler gears are incomplete or unclear. "
                "Each green idler gear unit should contain one big gear and one small gear. "
                "Please retake a clearer photo."
            )
            return False, f"{counts_text}\n{feedback}"

        idler_units = big_count

        if idler_units == 2:
            feedback = (
                "Good. You have met the task requirement. "
                "Detected counts: 1 driving gear and 2 green idler gear units."
            )
            return True, f"{counts_text}\n{feedback}"

        if idler_units < 2:
            feedback = (
                "Not enough green idler gear units were detected. "
                "This task requires two green idler gear units. "
                "Please include both units clearly in the photo."
            )
            return False, f"{counts_text}\n{feedback}"

        feedback = (
            "Too many green idler gear units were detected. "
            "This task requires exactly two green idler gear units. "
            "Please remove the extra gears and upload a new photo."
        )
        return False, f"{counts_text}\n{feedback}"

    if part_type == "shaft":
        msg = (
            "Detected shafts:\n"
            f"- long shaft: {counts.get('shaft_long', 0)}\n"
            f"- short shaft: {counts.get('shaft_short', 0)}"
        )
        if not is_correct:
            msg += "\nPlease separate the shafts and ensure both ends are clearly visible."
        return is_correct, msg

    if part_type == "spacer":
        msg = (
            "Detected spacers:\n"
            f"- long spacer: {counts.get('spacer_long', 0)}\n"
            f"- short spacer: {counts.get('spacer_short', 0)}"
        )
        if not is_correct:
            msg += "\nPlease place the spacers separately and retake the photo."
        return is_correct, msg

    return False, "Unsupported part type."


def _build_student_message(
    *,
    task: str,
    img_bgr: np.ndarray,
    out: Dict[str, Any],
    errors: List[Dict[str, Any]],
    selected_errors: List[Dict[str, Any]],
    part_type: str = "",
) -> Tuple[bool, str]:
    task = (task or "full").strip().lower()

    if task == "full":
        task = "mesh_ratio"

    if task == "parts_inventory":
        return _build_parts_inventory_message(out, part_type)

    if task not in ("precheck", "single_stage", "shaft", "spacer", "gear_inventory", "mesh_ratio"):
        task = "mesh_ratio"

    task_has_error = _has_selected_pipeline_error(selected_errors)

    if task == "precheck":
        quality = out.get("quality", {}) if isinstance(out.get("quality"), dict) else {}
        codes = {
            str(e.get("code", "")).upper()
            for e in selected_errors
            if isinstance(e, dict)
        }

        def qint(key: str, default: int = 0) -> str:
            try:
                return str(int(round(float(quality.get(key, default)))))
            except Exception:
                return str(default)

        def advice_text() -> str:
            advice = quality.get("quality_advice", [])
            if isinstance(advice, list):
                clean = [str(item).strip() for item in advice if str(item).strip()]
            else:
                clean = [str(advice).strip()] if str(advice).strip() else []
            if not clean:
                clean = ["The photo is clear enough for the next check."]
            return "\n".join(f"- {item}" for item in clean[:3])

        policy_key = "fail" if ("E_PHOTO_QUALITY_LOW" in codes or task_has_error) else "report"
        return policy_key == "report", MESSAGE_POLICY["precheck"][policy_key].format(
            quality_score=qint("quality_score"),
            brightness_score=qint("brightness_score"),
            contrast_score=qint("contrast_score"),
            sharpness_score=qint("sharpness_score_100"),
            noise_score=qint("noise_score_100"),
            quality_advice=advice_text(),
        )

    if task == "single_stage":
        codes = {
            str(e.get("code", "")).upper()
            for e in selected_errors
            if isinstance(e, dict)
        }
        driving_gear, smallgear, biggear = _get_gear_counts(out)
        shaft_long, shaft_short, shaft_total = _get_shaft_counts(out)
        spacer_long, spacer_short, spacer_total = _get_spacer_counts(out)

        if "E_NO_GEARS" in codes:
            return False, MESSAGE_POLICY["single_stage"]["no_gears"]

        if "E_SINGLE_STAGE_SHAFT" in codes:
            if shaft_total <= 0:
                return False, MESSAGE_POLICY["single_stage"]["no_shaft"]
            if shaft_total > 1:
                return False, MESSAGE_POLICY["single_stage"]["extra_shaft"]

        if "E_SINGLE_STAGE_SPACER_COUNT" in codes:
            if spacer_total <= 0:
                return False, MESSAGE_POLICY["single_stage"]["no_spacer"]
            if spacer_total > 1:
                return False, MESSAGE_POLICY["single_stage"]["extra_spacer"]

        if "E_SINGLE_STAGE_MISMESH" in codes:
            return False, MESSAGE_POLICY["single_stage"]["mismesh"]

        gear_codes = {
            "E_SINGLE_STAGE_DRIVING_GEAR",
            "E_SINGLE_STAGE_SMALL_GEAR",
            "E_SINGLE_STAGE_BIG_GEAR",
        }
        if codes & gear_codes:
            if (
                driving_gear > 1
                or smallgear > 1
                or biggear > 1
                or biggear != smallgear
            ):
                return False, MESSAGE_POLICY["single_stage"]["gear_overloaded"]
            return False, MESSAGE_POLICY["single_stage"]["gear_missing"]

        if "E_SINGLE_STAGE_STAGE_COUNT" in codes or "E_SINGLE_STAGE_RATIO" in codes:
            return False, MESSAGE_POLICY["single_stage"]["stage_fail"]

        is_correct = not _has_pipeline_error(errors)
        if not is_correct:
            return False, MESSAGE_POLICY["single_stage"]["fail"]

        ratio = out.get("ratio") if isinstance(out.get("ratio"), dict) else {}
        num_stages = ratio.get("num_stages")
        r_total = ratio.get("R_total")
        out_rpm = ratio.get("out_rpm")

        if num_stages is None or r_total is None or out_rpm is None:
            return False, MESSAGE_POLICY["single_stage"]["fail"]

        msg = MESSAGE_POLICY["single_stage"]["pass"].format(
            num_stages=_format_stage_value(num_stages),
            R_total=_format_ratio_value(r_total),
            out_rpm=_format_rpm_value(out_rpm),
        )
        hints = out.get("single_stage_hints", [])
        if isinstance(hints, list):
            cleaned = [str(h).strip() for h in hints if str(h).strip()]
            if cleaned:
                msg = f"{msg}\n" + "\n".join(cleaned)
        return True, msg

    if task == "shaft":
        codes = {
            str(e.get("code", "")).upper()
            for e in selected_errors
            if isinstance(e, dict)
        }

        n_long, n_short, n_total = _get_shaft_counts(out)

        if n_total > 0:
            if n_total > 2:
                return False, MESSAGE_POLICY["shaft"]["too_many"]

            if n_total == 1 and n_long == 1 and n_short == 0:
                return False, MESSAGE_POLICY["shaft"]["short_missing"]

            if n_total == 1 and n_short == 1 and n_long == 0:
                return False, MESSAGE_POLICY["shaft"]["long_missing"]

            if n_total != 2:
                return False, MESSAGE_POLICY["shaft"]["count_fail"]

            if n_short != 1 or n_long != 1:
                return False, MESSAGE_POLICY["shaft"]["type_confusion"]

        if (
            "E_SHAFT_COUNT_MISMATCH" in codes
            or "E_NO_SHAFTS" in codes
            or "E_SHAFT2_NOT_FOUND" in codes
        ):
            if n_total == 0:
                return False, MESSAGE_POLICY["shaft"]["none_detected"]
            if n_total > 2:
                return False, MESSAGE_POLICY["shaft"]["too_many"]
            if n_total == 1 and n_long == 1 and n_short == 0:
                return False, MESSAGE_POLICY["shaft"]["short_missing"]
            if n_total == 1 and n_short == 1 and n_long == 0:
                return False, MESSAGE_POLICY["shaft"]["long_missing"]
            return False, MESSAGE_POLICY["shaft"]["count_fail"]

        if (
            "E_SHAFT_POSITION_SWAP" in codes
            or "E_SHAFT2_CLASS_MISMATCH" in codes
            or "E_SHAFT3_CLASS_MISMATCH" in codes
        ):
            return False, MESSAGE_POLICY["shaft"]["position_swap"]

        if "E_SHAFT_TYPE_CONFUSION" in codes:
            return False, MESSAGE_POLICY["shaft"]["type_confusion"]

        if task_has_error:
            return False, MESSAGE_POLICY["shaft"]["fail"]

        return True, MESSAGE_POLICY["shaft"]["pass"]

    if task == "spacer":
        codes = {
            str(e.get("code", "")).upper()
            for e in selected_errors
            if isinstance(e, dict)
        }

        n_long, n_short, n_total = _get_spacer_counts(out)
        counts_dict = _get_counts_dict(out)

        if "E_SPACER_SHORT_MISSING" in codes:
            return False, MESSAGE_POLICY["spacer"]["short_missing"]

        if "E_SPACER_LONG_MISSING" in codes:
            return False, MESSAGE_POLICY["spacer"]["long_missing"]

        if "E_SPACER_TYPE_CONFUSION" in codes:
            return False, MESSAGE_POLICY["spacer"]["type_confusion"]

        if ("spacer_long" in counts_dict) or ("spacer_short" in counts_dict):
            if n_total == 0:
                return False, MESSAGE_POLICY["spacer"]["none_detected"]

            if n_total > 2:
                return False, MESSAGE_POLICY["spacer"]["too_many"]

            if n_short == 0 and n_long >= 1:
                return False, MESSAGE_POLICY["spacer"]["short_missing"]

            if n_long == 0 and n_short >= 1:
                return False, MESSAGE_POLICY["spacer"]["long_missing"]

            if n_short != 1 or n_long != 1 or n_total != 2:
                return False, MESSAGE_POLICY["spacer"]["count_fail"]

        if "E_SPACER_COUNT_MISMATCH" in codes:
            if n_total == 0:
                return False, MESSAGE_POLICY["spacer"]["none_detected"]
            if n_total > 2:
                return False, MESSAGE_POLICY["spacer"]["too_many"]
            return False, MESSAGE_POLICY["spacer"]["count_fail"]

        if (
            "E_SPACER_ASSIGNMENT_FAIL" in codes
            or "E_SPACER2_MISSING" in codes
            or "E_SPACER3_MISSING" in codes
        ):
            return False, MESSAGE_POLICY["spacer"]["assignment_fail"]

        if (
            "E_SPACER_POSITION_MISMATCH" in codes
            or "E_SPACER2_TYPE_MISMATCH" in codes
            or "E_SPACER3_TYPE_MISMATCH" in codes
        ):
            return False, MESSAGE_POLICY["spacer"]["position_mismatch"]

        if "E_SPACER_DISTANCE_ORDER" in codes:
            return False, MESSAGE_POLICY["spacer"]["distance_order"]

        if task_has_error:
            return False, MESSAGE_POLICY["spacer"]["fail"]

        return True, MESSAGE_POLICY["spacer"]["pass"]

    if task == "gear_inventory":
        codes = {
            str(e.get("code", "")).upper()
            for e in selected_errors
            if isinstance(e, dict)
        }

        driving_gear, smallgear, biggear = _get_gear_counts(out)

        if "E_NO_GEARS" in codes:
            return False, MESSAGE_POLICY["gear_inventory"]["no_gears"].format(
                driving_gear=driving_gear,
                smallgear=smallgear,
                biggear=biggear,
            )

        if "E_MESH_MISMATCH" in codes:
            return False, MESSAGE_POLICY["gear_inventory"]["mismatch_fail"].format(
                driving_gear=driving_gear,
                smallgear=smallgear,
                biggear=biggear,
            )

        if "E_GEAR_BIG_SMALL_INCONSISTENT" in codes:
            return False, MESSAGE_POLICY["gear_inventory"]["big_small_inconsistent"].format(
                driving_gear=driving_gear,
                smallgear=smallgear,
                biggear=biggear,
            )

        if "E_GEAR_CONTACT_INCONSISTENT" in codes:
            return False, MESSAGE_POLICY["gear_inventory"]["contact_consistency_fail"].format(
                driving_gear=driving_gear,
                smallgear=smallgear,
                biggear=biggear,
            )

        if task_has_error:
            return False, MESSAGE_POLICY["gear_inventory"]["fail"].format(
                driving_gear=driving_gear,
                smallgear=smallgear,
                biggear=biggear,
            )

        return True, MESSAGE_POLICY["gear_inventory"]["pass"].format(
            driving_gear=driving_gear,
            smallgear=smallgear,
            biggear=biggear,
        )

    if task == "mesh_ratio":
        codes = {
            str(e.get("code", "")).upper()
            for e in selected_errors
            if isinstance(e, dict)
        }
        driving_gear, smallgear, biggear = _get_gear_counts(out)
        shaft_long, shaft_short, shaft_total = _get_shaft_counts(out)
        spacer_long, spacer_short, spacer_total = _get_spacer_counts(out)

        # Highest-priority prerequisite errors
        if "E_NO_GEARS" in codes:
            return False, MESSAGE_POLICY["gear_inventory"]["no_gears"].format(
                driving_gear=driving_gear,
                smallgear=smallgear,
                biggear=biggear,
            )

        if "E_NO_SHAFTS" in codes:
            return False, MESSAGE_POLICY["shaft"]["none_detected"]

        if "E_NO_GEAR11" in codes:
            return False, MESSAGE_POLICY["mesh_ratio"]["fail"]

        # Shaft checks
        if (
            "E_SHAFT_COUNT_MISMATCH" in codes
            or "E_SHAFT2_NOT_FOUND" in codes
        ):
            if shaft_total == 0:
                return False, MESSAGE_POLICY["shaft"]["none_detected"]
            if shaft_total > 2:
                return False, MESSAGE_POLICY["shaft"]["too_many"]
            if shaft_total == 1 and shaft_long == 1 and shaft_short == 0:
                return False, MESSAGE_POLICY["shaft"]["short_missing"]
            if shaft_total == 1 and shaft_short == 1 and shaft_long == 0:
                return False, MESSAGE_POLICY["shaft"]["long_missing"]
            return False, MESSAGE_POLICY["shaft"]["count_fail"]

        if "E_SHAFT_TYPE_CONFUSION" in codes:
            return False, MESSAGE_POLICY["shaft"]["type_confusion"]

        if (
            "E_SHAFT_POSITION_SWAP" in codes
            or "E_SHAFT2_CLASS_MISMATCH" in codes
            or "E_SHAFT3_CLASS_MISMATCH" in codes
        ):
            return False, MESSAGE_POLICY["shaft"]["position_swap"]

        # Spacer checks
        if "E_SPACER_SHORT_MISSING" in codes:
            return False, MESSAGE_POLICY["spacer"]["short_missing"]

        if "E_SPACER_LONG_MISSING" in codes:
            return False, MESSAGE_POLICY["spacer"]["long_missing"]

        if "E_SPACER_TYPE_CONFUSION" in codes:
            return False, MESSAGE_POLICY["spacer"]["type_confusion"]

        if "E_SPACER_COUNT_MISMATCH" in codes:
            if spacer_total == 0:
                return False, MESSAGE_POLICY["spacer"]["none_detected"]
            if spacer_total > 2:
                return False, MESSAGE_POLICY["spacer"]["too_many"]
            return False, MESSAGE_POLICY["spacer"]["count_fail"]

        if (
            "E_SPACER_ASSIGNMENT_FAIL" in codes
            or "E_SPACER2_MISSING" in codes
            or "E_SPACER3_MISSING" in codes
        ):
            return False, MESSAGE_POLICY["spacer"]["assignment_fail"]

        if (
            "E_SPACER_POSITION_MISMATCH" in codes
            or "E_SPACER2_TYPE_MISMATCH" in codes
            or "E_SPACER3_TYPE_MISMATCH" in codes
        ):
            return False, MESSAGE_POLICY["spacer"]["position_mismatch"]

        if "E_SPACER_DISTANCE_ORDER" in codes:
            return False, MESSAGE_POLICY["spacer"]["distance_order"]

        # Mesh and consistency checks
        if "E_GEAR_BIG_SMALL_INCONSISTENT" in codes:
            return False, MESSAGE_POLICY["gear_inventory"]["big_small_inconsistent"].format(
                driving_gear=driving_gear,
                smallgear=smallgear,
                biggear=biggear,
            )

        if "E_GEAR_CONTACT_INCONSISTENT" in codes:
            return False, MESSAGE_POLICY["gear_inventory"]["contact_consistency_fail"].format(
                driving_gear=driving_gear,
                smallgear=smallgear,
                biggear=biggear,
            )

        if (
            "E_MISMESH_DETECTED" in codes
            or "E_MESH_MISMATCH" in codes
        ):
            return False, MESSAGE_POLICY["gear_inventory"]["mismatch_fail"].format(
                driving_gear=driving_gear,
                smallgear=smallgear,
                biggear=biggear,
            )

        if (
            "E_CONTACT_COUNT_MISMATCH" in codes
            or "E_GEAR_COUNT_UNSUPPORTED" in codes
        ):
            return False, MESSAGE_POLICY["mesh_ratio"]["fail"]

        # Final ratio output only if all checks pass
        is_correct = not _has_pipeline_error(errors)
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

    return False, "Unsupported task."


def _call_pipeline_with_fallbacks(
    *,
    img_bgr: np.ndarray,
    model_a_rel: str,
    model_b_rel: str,
    model_c_rel: str,
    return_images: bool,
    task: str,
    part_type: str,
    expected_gears: Any,
) -> Dict[str, Any]:
    """
    Preferred new signature:
        run_yolo_pipeline(
            img_bgr=...,
            model_a_rel=...,
            model_b_rel=...,
            model_c_rel=...,
            return_images=...,
            task=...,
            part_type=...,
            expected_gears=...,
        )

    Backward-compatibility fallback:
        run_yolo_pipeline(
            img_bgr=...,
            gear_model_rel=...,
            shaft_model_rel=...,
            return_images=...,
            task=...,
            part_type=...,
            expected_gears=...,
        )
    """
    try:
        return run_yolo_pipeline(  # type: ignore[misc]
            img_bgr=img_bgr,
            model_a_rel=model_a_rel,
            model_b_rel=model_b_rel,
            model_c_rel=model_c_rel,
            return_images=return_images,
            task=task,
            part_type=part_type,
            expected_gears=expected_gears,
        )
    except TypeError:
        pass

    try:
        return run_yolo_pipeline(  # type: ignore[misc]
            img_bgr=img_bgr,
            model_a_rel=model_a_rel,
            model_b_rel=model_b_rel,
            model_c_rel=model_c_rel,
            return_images=return_images,
            task=task,
            expected_gears=expected_gears,
        )
    except TypeError:
        pass

    try:
        return run_yolo_pipeline(  # type: ignore[misc]
            img_bgr=img_bgr,
            gear_model_rel=model_a_rel,
            shaft_model_rel=model_b_rel,
            return_images=return_images,
            task=task,
            part_type=part_type,
            expected_gears=expected_gears,
        )
    except TypeError:
        pass

    return run_yolo_pipeline(  # type: ignore[misc]
        img_bgr=img_bgr,
        gear_model_rel=model_a_rel,
        shaft_model_rel=model_b_rel,
        return_images=return_images,
        task=task,
        expected_gears=expected_gears,
    )


def evaluation_function(response: Any, answer: Any, params: Params) -> Result:
    task = str(_pget(params, "task", "full") or "full").strip().lower()
    part_type = str(_pget(params, "part_type", "") or "").strip().lower()

    if task == "full":
        pipeline_task = "full"
        message_task = "mesh_ratio"
    else:
        pipeline_task = task
        message_task = task

    return_images: bool = bool(
        _pget(params, "return_images", pipeline_task not in ("precheck", "parts_inventory", "single_stage"))
    )

    model_a_rel = str(_pget(params, "model_a_rel", _pget(params, "gear_model_rel", "modelA.pt")))
    model_b_rel = str(_pget(params, "model_b_rel", _pget(params, "shaft_model_rel", "modelB.pt")))
    model_c_rel = str(_pget(params, "model_c_rel", "modelC.pt"))

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
        out: Dict[str, Any] = _call_pipeline_with_fallbacks(
            img_bgr=img_bgr,
            model_a_rel=model_a_rel,
            model_b_rel=model_b_rel,
            model_c_rel=model_c_rel,
            return_images=return_images,
            task=pipeline_task,
            part_type=part_type,
            expected_gears=expected_gears,
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
        part_type=part_type,
    )
    return _result_minimal(is_correct, msg)
