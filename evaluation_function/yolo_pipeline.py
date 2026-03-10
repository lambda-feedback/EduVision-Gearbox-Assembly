"""
YOLO inference pipeline (gear model + shaft OBB model) — TASK-AWARE VERSION

Final streamlined version:
- Keeps core detection and task logic
- Removes overlay image generation and drawing utilities
- Adds parts inventory logic:
    1) supports part_type = gear / shaft / spacer
    2) outputs counts only for:
       - gear: biggear / smallgear
       - shaft: shaft_long / shaft_short
       - spacer: spacer_long / spacer_short
    3) excludes white driving gear from parts inventory gear counts
- Adds precheck logic:
    1) checks only the consistency rule between gear count and (mesh + mismesh) count
    2) does not separately decide whether gears or contacts are wrong
- Adds single-stage logic:
    1) expects 1 driving gear
    2) expects 1 short spacer
    3) expects 1 small gear and 1 big gear
    4) expects 1 shaft (long or short)
    5) expects no mismesh
    6) expects exactly 1 stage and a valid ratio
- Adds improved shaft-step logic:
    1) missing shaft / wrong shaft count
    2) two shafts of the same detected type
    3) shaft_short / shaft_long position swap relative to gear11
- Adds improved spacer-step logic:
    1) missing spacer / wrong spacer count
    2) missing short spacer / missing long spacer
    3) two spacers of the same detected type
    4) spacer position mismatch relative to expected shafts
    5) spacer distance order mismatch relative to gear11
"""

from __future__ import annotations

import math
import os
import time
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .lazy_load import LazyModule

_ultralytics = LazyModule("ultralytics")
_COLD_START_FLAG = True

if TYPE_CHECKING:
    from ultralytics import YOLO as _YOLO  # pragma: no cover
else:
    _YOLO = Any


# =========================
# CONFIG
# =========================
CONF_GEAR: float = 0.50
CONF_SHAFT: float = 0.50

MOTOR_RPM: float = 8000.0
TEETH_BIG: int = 48
TEETH_SMALL: int = 12

DRIVING_GEAR_CLASS_NAMES = {"Driving_Gear", "driving_gear", "DrivingGear"}
GEAR_BIG_NAME = "Gear_big"
GEAR_SMALL_NAME = "Gear_small"
MESH_CLASS_NAME = "Mesh"
MISMESH_CLASS_NAME = "Mismesh"

SPACER_CLASSES = {
    "spacer_long", "spacer_short",
    "Long spacer tube", "Short spacer tube",
    "spacer tube long", "spacer tube short"
}
TARGET_SHAFT_CLASSES = {"shaft_long", "shaft_short"}

OBB_ASSIGN_SCALE: float = 1.10

LINE_SAMPLES: int = 25
LINE_HIT_RATIO_TH: float = 0.25

REQUIRE_DIFFERENT_SHAFT: bool = True
REQUIRE_BIG_SMALL_PAIR: bool = True

FORCE_STAGE1_BY_SHORT_SPACER: bool = True
STAGE1_MATE_MUST_BE_GEAR_BIG: bool = True

ONE_TO_ONE_CONTACT_BOX: bool = True
MAX_STAGE: int = 6

W_HIT: float = 1.0
W_DMS: float = 0.6
W_GAP: float = 0.4
NORM_DMS: float = 200.0
NORM_GAP: float = 200.0

ENABLE_ERROR_CHECKS: bool = True
SPACER_DIST_TOL_PX: float = 5.0


# -------------------------
# Task constants
# -------------------------
TASK_PARTS_INVENTORY = "parts_inventory"
TASK_PRECHECK = "precheck"
TASK_SINGLE_STAGE = "single_stage"
TASK_SHAFT = "shaft"
TASK_SPACER = "spacer"
TASK_GEAR_INV = "gear_inventory"
TASK_MESH_RATIO = "mesh_ratio"
TASK_FULL = "full"

_VALID_TASKS = {
    TASK_PARTS_INVENTORY,
    TASK_PRECHECK,
    TASK_SINGLE_STAGE,
    TASK_SHAFT,
    TASK_SPACER,
    TASK_GEAR_INV,
    TASK_MESH_RATIO,
    TASK_FULL,
}


# =========================
# Paths
# =========================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GEAR_MODEL_REL = "gear_model.pt"
DEFAULT_SHAFT_MODEL_REL = "shaft_model.pt"


def _abs_model_path(rel_name: str) -> str:
    return os.path.join(_THIS_DIR, rel_name)


# =========================
# Model cache
# =========================
@lru_cache(maxsize=2)
def _load_yolo_model(abs_path: str) -> Any:
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Model not found: {abs_path}")

    YOLO_cls = _ultralytics.YOLO
    return YOLO_cls(abs_path)


def get_models(
    gear_model_rel: str = DEFAULT_GEAR_MODEL_REL,
    shaft_model_rel: str = DEFAULT_SHAFT_MODEL_REL,
    timing: Optional[Dict[str, float]] = None,
) -> Tuple[Any, Any]:
    t0 = time.perf_counter()
    gear_model = _load_yolo_model(_abs_model_path(gear_model_rel))
    t1 = time.perf_counter()
    shaft_model = _load_yolo_model(_abs_model_path(shaft_model_rel))
    t2 = time.perf_counter()

    if timing is not None:
        timing["t_load_gear_model_s"] = float(t1 - t0)
        timing["t_load_shaft_model_s"] = float(t2 - t1)
        timing["t_get_models_total_s"] = float(t2 - t0)

    return gear_model, shaft_model


# =========================
# Helpers
# =========================
def bbox_center(b: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_wh(b: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (max(0.0, x2 - x1), max(0.0, y2 - y1))


def est_radius_from_bbox(b: Tuple[float, float, float, float]) -> float:
    w, h = bbox_wh(b)
    return 0.5 * min(w, h)


def poly_center(pts4: np.ndarray) -> Tuple[float, float]:
    pts = np.asarray(pts4, dtype=np.float32)
    return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))


def scale_poly_about_center(pts4: np.ndarray, scale: float = 1.1) -> np.ndarray:
    pts = np.asarray(pts4, dtype=np.float32)
    c = pts.mean(axis=0, keepdims=True)
    pts2 = c + (pts - c) * float(scale)
    return pts2.astype(np.float32)


def point_in_poly(pt: Tuple[float, float], poly4: np.ndarray) -> bool:
    contour = poly4.reshape((-1, 1, 2)).astype(np.float32)
    return cv2.pointPolygonTest(contour, (float(pt[0]), float(pt[1])), False) >= 0


def line_hit_ratio(
    line_p1: Tuple[float, float],
    line_p2: Tuple[float, float],
    bbox: Tuple[float, float, float, float],
    samples: int = 25,
) -> float:
    x1, y1, x2, y2 = bbox
    hit = 0
    for i in range(samples):
        t = i / max(1, samples - 1)
        x = line_p1[0] * (1 - t) + line_p2[0] * t
        y = line_p1[1] * (1 - t) + line_p2[1] * t
        if x1 <= x <= x2 and y1 <= y <= y2:
            hit += 1
    return hit / float(samples)


def center_dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return float((dx * dx + dy * dy) ** 0.5)


def dist_point_to_segment(p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab2 = abx * abx + aby * aby + 1e-9
    t = (apx * abx + apy * aby) / ab2
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    cx, cy = ax + t * abx, ay + t * aby
    dx, dy = px - cx, py - cy
    return float((dx * dx + dy * dy) ** 0.5)


# =========================
# Gear ratio helpers
# =========================
def teeth_from_cls(cls_name: str) -> Optional[int]:
    c = str(cls_name)
    if c == GEAR_BIG_NAME:
        return TEETH_BIG
    if c == GEAR_SMALL_NAME:
        return TEETH_SMALL
    if c in DRIVING_GEAR_CLASS_NAMES or ("Driving" in c) or ("driving" in c):
        return TEETH_SMALL
    return None


def compute_ratio_and_rpm_from_stage_labels(
    gears: List[Dict[str, Any]],
    gear_names: Dict[int, str],
    motor_rpm: float = MOTOR_RPM,
    max_stage: int = MAX_STAGE,
) -> Tuple[int, Optional[float], Optional[float], List[Tuple[int, float, int, int]]]:
    name_to_gear: Dict[str, Dict[str, Any]] = {}
    for g in gears:
        gid = g["gid"]
        if gid in gear_names:
            name_to_gear[gear_names[gid]] = g

    ratios: List[float] = []
    per_stage: List[Tuple[int, float, int, int]] = []

    for s in range(1, max_stage + 1):
        k1 = f"gear{s}1"
        k2 = f"gear{s}2"
        if (k1 not in name_to_gear) and (k2 not in name_to_gear):
            if len(per_stage) > 0:
                break
            continue

        g1 = name_to_gear.get(k1)
        g2 = name_to_gear.get(k2)
        if g1 is None or g2 is None:
            break

        z1 = teeth_from_cls(g1["cls"])
        z2 = teeth_from_cls(g2["cls"])
        if z1 is None or z2 is None:
            break

        R = float(z2) / float(z1)
        ratios.append(R)
        per_stage.append((s, R, z1, z2))

    if not ratios:
        return 0, None, None, per_stage

    R_total = 1.0
    for r in ratios:
        R_total *= r

    out_rpm = float(motor_rpm) / float(R_total)
    return len(ratios), R_total, out_rpm, per_stage


# =========================
# Spacer helpers
# =========================
def spacer_is_long(sp: Dict[str, Any]) -> bool:
    t = sp["cls"].lower().replace("_", " ")
    return ("long" in t) and ("spacer" in t)


def spacer_is_short(sp: Dict[str, Any]) -> bool:
    t = sp["cls"].lower().replace("_", " ")
    return ("short" in t) and ("spacer" in t)


def expected_contact_boxes_from_gear_count(gear_count: int) -> Tuple[Optional[int], bool]:
    if gear_count <= 1:
        return 0, True
    if (gear_count - 1) % 2 != 0:
        return None, False
    return (gear_count - 1) // 2, True


def evaluate_precheck_count_rule(
    gears: List[Dict[str, Any]],
    mesh_boxes: List[Tuple[float, float, float, float]],
    mismesh_boxes: List[Tuple[float, float, float, float]],
) -> List[Dict[str, str]]:
    errs: List[Dict[str, str]] = []

    gear_count = len(gears)
    total_contact = len(mesh_boxes) + len(mismesh_boxes)

    expected_contact, ok = expected_contact_boxes_from_gear_count(gear_count)

    if (not ok) or (expected_contact is None):
        errs.append({
            "code": "E_PRECHECK_COUNT_RULE_FAIL",
            "message": (
                f"Precheck failed: detected gear_count={gear_count}, which does not fit "
                f"the expected contact rule."
            ),
        })
        return errs

    if total_contact != expected_contact:
        errs.append({
            "code": "E_PRECHECK_COUNT_RULE_FAIL",
            "message": (
                f"Precheck failed: mesh+mismesh={total_contact}, expected={expected_contact} "
                f"for gear_count={gear_count}."
            ),
        })

    return errs


# =========================
# Parts inventory helpers
# =========================
def evaluate_parts_inventory(
    *,
    part_type: str,
    gear_dets: List[Dict[str, Any]],
    shaft_dets: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Returns counts only for the requested part type.

    part_type:
      - "gear"   -> biggear / smallgear
      - "shaft"  -> shaft_long / shaft_short
      - "spacer" -> spacer_long / spacer_short

    Notes:
      - white driving gear is excluded from gear counts
      - color is ignored
    """
    errors: List[Dict[str, str]] = []
    counts: Dict[str, int] = {}
    part_type = str(part_type or "").strip().lower()

    if part_type == "gear":
        counts = {
            "biggear": 0,
            "smallgear": 0,
        }

        for d in gear_dets:
            cls = str(d.get("cls", ""))
            if cls == GEAR_BIG_NAME:
                counts["biggear"] += 1
            elif cls == GEAR_SMALL_NAME:
                counts["smallgear"] += 1

        if (counts["biggear"] + counts["smallgear"]) == 0:
            errors.append({
                "code": "E_NO_TARGET_PARTS",
                "message": "No target gears were detected.",
            })

        return {
            "counts": counts,
            "errors": errors,
        }

    if part_type == "shaft":
        counts = {
            "shaft_long": 0,
            "shaft_short": 0,
        }

        for d in shaft_dets:
            cls = str(d.get("cls", ""))
            if cls == "shaft_long":
                counts["shaft_long"] += 1
            elif cls == "shaft_short":
                counts["shaft_short"] += 1

        if (counts["shaft_long"] + counts["shaft_short"]) == 0:
            errors.append({
                "code": "E_NO_TARGET_PARTS",
                "message": "No target shafts were detected.",
            })

        return {
            "counts": counts,
            "errors": errors,
        }

    if part_type == "spacer":
        counts = {
            "spacer_long": 0,
            "spacer_short": 0,
        }

        for d in gear_dets:
            cls = str(d.get("cls", "")).lower().replace(" ", "_")
            if ("spacer" in cls) and ("long" in cls):
                counts["spacer_long"] += 1
            elif ("spacer" in cls) and ("short" in cls):
                counts["spacer_short"] += 1

        if (counts["spacer_long"] + counts["spacer_short"]) == 0:
            errors.append({
                "code": "E_NO_TARGET_PARTS",
                "message": "No target spacers were detected.",
            })

        return {
            "counts": counts,
            "errors": errors,
        }

    return {
        "counts": {},
        "errors": [{
            "code": "E_BAD_PART_TYPE",
            "message": f"Unsupported part_type: {part_type}",
        }],
    }


# =========================
# Single-stage helpers
# =========================
def evaluate_single_stage_errors(
    gears: List[Dict[str, Any]],
    spacers: List[Dict[str, Any]],
    shafts: List[Dict[str, Any]],
    mismesh_boxes: List[Tuple[float, float, float, float]],
    ratio: Dict[str, Any],
) -> List[Dict[str, str]]:
    errs: List[Dict[str, str]] = []

    driving_cnt = sum(1 for g in gears if str(g["cls"]) in DRIVING_GEAR_CLASS_NAMES)
    big_cnt = sum(1 for g in gears if str(g["cls"]) == GEAR_BIG_NAME)
    small_cnt = sum(1 for g in gears if str(g["cls"]) == GEAR_SMALL_NAME)
    short_spacer_cnt = sum(1 for sp in spacers if spacer_is_short(sp))
    shaft_cnt = len(shafts)

    if driving_cnt != 1:
        errs.append({
            "code": "E_SINGLE_STAGE_DRIVING_GEAR",
            "message": f"Expected 1 driving gear, but detected {driving_cnt}.",
        })

    if short_spacer_cnt != 1:
        errs.append({
            "code": "E_SINGLE_STAGE_SHORT_SPACER",
            "message": f"Expected 1 short spacer, but detected {short_spacer_cnt}.",
        })

    if small_cnt != 1:
        errs.append({
            "code": "E_SINGLE_STAGE_SMALL_GEAR",
            "message": f"Expected 1 small gear, but detected {small_cnt}.",
        })

    if big_cnt != 1:
        errs.append({
            "code": "E_SINGLE_STAGE_BIG_GEAR",
            "message": f"Expected 1 big gear, but detected {big_cnt}.",
        })

    if shaft_cnt != 1:
        errs.append({
            "code": "E_SINGLE_STAGE_SHAFT",
            "message": f"Expected 1 shaft, but detected {shaft_cnt}.",
        })

    if len(mismesh_boxes) > 0:
        errs.append({
            "code": "E_SINGLE_STAGE_MISMESH",
            "message": f"Mismesh detected (count={len(mismesh_boxes)}).",
        })

    num_stages = ratio.get("num_stages")
    if num_stages != 1:
        errs.append({
            "code": "E_SINGLE_STAGE_STAGE_COUNT",
            "message": f"Expected 1 stage, but detected {num_stages}.",
        })

    if ratio.get("R_total") is None or ratio.get("out_rpm") is None:
        errs.append({
            "code": "E_SINGLE_STAGE_RATIO",
            "message": "The single-stage gear ratio could not be calculated reliably.",
        })

    return errs


# =========================
# Shaft helpers
# =========================
def get_expected_shaft_indices_for_step(
    gears: List[Dict[str, Any]],
    shafts: List[Dict[str, Any]],
    gear11_gid: int,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Return:
      - short_shaft_idx: shaft expected to be closer to gear11
      - long_shaft_idx: shaft expected to be farther from gear11
    """
    if not gears or not shafts or len(shafts) < 2:
        return None, None

    g11 = next((g for g in gears if g["gid"] == gear11_gid), None)
    if g11 is None:
        return None, None

    c11 = g11["center"]
    dlist: List[Tuple[float, int]] = []
    for i, s in enumerate(shafts):
        dlist.append((center_dist(s["center"], c11), i))

    dlist.sort(key=lambda x: x[0])

    if len(dlist) < 2:
        return None, None

    short_shaft_idx = dlist[0][1]
    long_shaft_idx = dlist[1][1]
    return short_shaft_idx, long_shaft_idx


def pick_shaft2_and_shaft3_by_distance(
    shafts: List[Dict[str, Any]],
    gear11_gid: int,
    gear_to_si: Dict[int, int],
    gears: List[Dict[str, Any]],
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Legacy helper kept for compatibility with assembly checks.
    shaft1: shaft containing gear11
    shaft2/shaft3: by distance to gear11 (closest, 2nd closest), excluding shaft1
    """
    if not shafts or not gears:
        return None, None, None

    g11 = next((g for g in gears if g["gid"] == gear11_gid), None)
    if g11 is None:
        return None, None, None

    shaft1 = gear_to_si.get(gear11_gid)
    c11 = g11["center"]

    dlist: List[Tuple[float, int]] = []
    for i, s in enumerate(shafts):
        if shaft1 is not None and i == shaft1:
            continue
        dlist.append((center_dist(s["center"], c11), i))

    dlist.sort(key=lambda x: x[0])
    shaft2 = dlist[0][1] if len(dlist) >= 1 else None
    shaft3 = dlist[1][1] if len(dlist) >= 2 else None
    return shaft1, shaft2, shaft3


def evaluate_shaft_step_errors(
    gears: List[Dict[str, Any]],
    shafts: List[Dict[str, Any]],
    gear11_gid: int,
) -> List[Dict[str, str]]:
    """
    Shaft-step specific rules:
    1) expected exactly two shafts
    2) expected one shaft_short and one shaft_long
    3) shaft_short should be closer to gear11 than shaft_long
    """
    errs: List[Dict[str, str]] = []

    if not gears:
        errs.append({"code": "E_NO_GEARS", "message": "No gears detected."})
        return errs

    if len(shafts) == 0:
        errs.append({"code": "E_SHAFT_COUNT_MISMATCH", "message": "No shafts detected."})
        return errs

    if len(shafts) == 1:
        errs.append({"code": "E_SHAFT_COUNT_MISMATCH", "message": "Only one shaft detected."})
        return errs

    if len(shafts) != 2:
        errs.append({
            "code": "E_SHAFT_COUNT_MISMATCH",
            "message": f"Expected 2 shafts, but detected {len(shafts)}."
        })
        return errs

    g11 = next((g for g in gears if g["gid"] == gear11_gid), None)
    if g11 is None:
        errs.append({"code": "E_NO_GEAR11", "message": "Cannot determine gear11."})
        return errs

    c11 = g11["center"]

    shaft_info: List[Dict[str, Any]] = []
    for i, s in enumerate(shafts):
        shaft_info.append({
            "index": i,
            "cls": str(s["cls"]),
            "dist": center_dist(s["center"], c11),
        })

    short_shafts = [s for s in shaft_info if s["cls"] == "shaft_short"]
    long_shafts = [s for s in shaft_info if s["cls"] == "shaft_long"]

    if len(short_shafts) != 1 or len(long_shafts) != 1:
        errs.append({
            "code": "E_SHAFT_TYPE_CONFUSION",
            "message": "The shaft types could not be identified reliably."
        })
        return errs

    short_dist = short_shafts[0]["dist"]
    long_dist = long_shafts[0]["dist"]

    if long_dist < short_dist:
        errs.append({
            "code": "E_SHAFT_POSITION_SWAP",
            "message": "The shaft positions appear to be swapped."
        })

    return errs


# =========================
# Spacer helpers for step logic
# =========================
def pick_spacer_on_shaft_as(
    spacers: List[Dict[str, Any]],
    spacer_to_si: Dict[int, int],
    shaft_idx: int,
    ref_pt: Tuple[float, float],
) -> Optional[Dict[str, Any]]:
    candidates: List[Tuple[float, Dict[str, Any]]] = []
    for sp in spacers:
        sid = sp["sid"]
        si = spacer_to_si.get(sid)
        if si is None or si != shaft_idx:
            continue
        candidates.append((center_dist(sp["center"], ref_pt), sp))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def evaluate_spacer_step_errors(
    gears: List[Dict[str, Any]],
    spacers: List[Dict[str, Any]],
    shafts: List[Dict[str, Any]],
    gear11_gid: int,
    spacer_to_si: Dict[int, int],
) -> List[Dict[str, str]]:
    """
    Spacer-step specific rules:
    1) expected exactly two spacers
    2) expected one spacer_short and one spacer_long
    3) spacer_short should be on the shaft closer to gear11
    4) spacer_long should be on the shaft farther from gear11
    5) spacer_short should be closer to gear11 than spacer_long
    """
    errs: List[Dict[str, str]] = []

    if not gears:
        errs.append({"code": "E_NO_GEARS", "message": "No gears detected."})
        return errs

    if not shafts:
        errs.append({"code": "E_NO_SHAFTS", "message": "No shafts detected."})
        return errs

    g11 = next((g for g in gears if g["gid"] == gear11_gid), None)
    if g11 is None:
        errs.append({"code": "E_NO_GEAR11", "message": "Cannot determine gear11."})
        return errs

    c11 = g11["center"]

    if len(spacers) == 0:
        errs.append({"code": "E_SPACER_COUNT_MISMATCH", "message": "No spacers detected."})
        return errs

    if len(spacers) == 1:
        sp = spacers[0]
        cls_name = str(sp["cls"]).lower().replace(" ", "_")

        if "short" in cls_name:
            errs.append({"code": "E_SPACER_LONG_MISSING", "message": "Only the short spacer was detected."})
        elif "long" in cls_name:
            errs.append({"code": "E_SPACER_SHORT_MISSING", "message": "Only the long spacer was detected."})
        else:
            errs.append({"code": "E_SPACER_COUNT_MISMATCH", "message": "Only one spacer was detected."})
        return errs

    if len(spacers) != 2:
        errs.append({
            "code": "E_SPACER_COUNT_MISMATCH",
            "message": f"Expected 2 spacers, but detected {len(spacers)}."
        })
        return errs

    short_spacers = [sp for sp in spacers if spacer_is_short(sp)]
    long_spacers = [sp for sp in spacers if spacer_is_long(sp)]

    if len(short_spacers) == 0 and len(long_spacers) == 2:
        errs.append({"code": "E_SPACER_TYPE_CONFUSION", "message": "Two long spacers were detected."})
        return errs

    if len(short_spacers) == 2 and len(long_spacers) == 0:
        errs.append({"code": "E_SPACER_TYPE_CONFUSION", "message": "Two short spacers were detected."})
        return errs

    if len(short_spacers) != 1 or len(long_spacers) != 1:
        errs.append({
            "code": "E_SPACER_TYPE_CONFUSION",
            "message": "The spacer types could not be identified reliably."
        })
        return errs

    short_sp = short_spacers[0]
    long_sp = long_spacers[0]

    short_shaft_idx, long_shaft_idx = get_expected_shaft_indices_for_step(
        gears=gears,
        shafts=shafts,
        gear11_gid=gear11_gid,
    )

    if short_shaft_idx is None or long_shaft_idx is None:
        errs.append({
            "code": "E_SPACER_POSITION_MISMATCH",
            "message": "Cannot determine the expected shaft positions for spacers."
        })
        return errs

    short_sp_si = spacer_to_si.get(short_sp["sid"])
    long_sp_si = spacer_to_si.get(long_sp["sid"])

    if short_sp_si != short_shaft_idx or long_sp_si != long_shaft_idx:
        errs.append({
            "code": "E_SPACER_POSITION_MISMATCH",
            "message": "The spacer positions appear to be incorrect."
        })
        return errs

    d_short = center_dist(short_sp["center"], c11)
    d_long = center_dist(long_sp["center"], c11)

    if d_short > d_long + float(SPACER_DIST_TOL_PX):
        errs.append({
            "code": "E_SPACER_DISTANCE_ORDER",
            "message": f"The short spacer is not closer to gear11 than the long spacer (tol={SPACER_DIST_TOL_PX}px)."
        })

    return errs


# =========================
# Assembly checks for full path
# =========================
def evaluate_assembly_errors(
    gears: List[Dict[str, Any]],
    spacers: List[Dict[str, Any]],
    shafts: List[Dict[str, Any]],
    mesh_boxes: List[Tuple[float, float, float, float]],
    mismesh_boxes: List[Tuple[float, float, float, float]],
    gear11_gid: int,
    gear_to_si: Dict[int, int],
    spacer_to_si: Dict[int, int],
) -> List[Dict[str, str]]:
    errs: List[Dict[str, str]] = []
    if not gears:
        return errs

    if len(mismesh_boxes) > 0:
        errs.append({
            "code": "E_MISMESH_DETECTED",
            "message": f"Assembly issue: MISMESH detected (count={len(mismesh_boxes)})."
        })

    g11 = next((g for g in gears if g["gid"] == gear11_gid), None)
    if g11 is None:
        return errs
    c11 = g11["center"]

    shaft1, shaft2, shaft3 = pick_shaft2_and_shaft3_by_distance(shafts, gear11_gid, gear_to_si, gears)

    if shaft2 is not None and 0 <= shaft2 < len(shafts):
        if str(shafts[shaft2]["cls"]) == "shaft_long":
            errs.append({
                "code": "E_SHAFT2_CLASS_MISMATCH",
                "message": "Assembly issue: shaft2 (closest shaft to gear11) is classified as 'shaft_long' (expected 'shaft_short')."
            })

    spacer2 = pick_spacer_on_shaft_as(spacers, spacer_to_si, shaft2, c11) if shaft2 is not None else None
    spacer3 = pick_spacer_on_shaft_as(spacers, spacer_to_si, shaft3, c11) if shaft3 is not None else None

    if shaft2 is not None:
        if spacer2 is None:
            errs.append({
                "code": "E_SPACER2_MISSING",
                "message": "Assembly issue: no spacer found on shaft2 (spacer2 missing or not assigned to shaft2)."
            })
        elif spacer_is_long(spacer2):
            errs.append({
                "code": "E_SPACER2_TYPE_MISMATCH",
                "message": "Assembly issue: spacer2 (spacer on shaft2) is classified as 'long spacer' (expected 'short spacer')."
            })

    if shaft3 is not None:
        if spacer3 is None:
            errs.append({
                "code": "E_SPACER3_MISSING",
                "message": "Assembly issue: no spacer found on shaft3 (spacer3 missing or not assigned to shaft3)."
            })
        elif spacer_is_short(spacer3):
            errs.append({
                "code": "E_SPACER3_TYPE_MISMATCH",
                "message": "Assembly issue: spacer3 (spacer on shaft3) is classified as 'short spacer' (expected 'long spacer')."
            })

    if spacer2 is not None and spacer3 is not None:
        d2 = center_dist(spacer2["center"], c11)
        d3 = center_dist(spacer3["center"], c11)
        if d2 > d3 + float(SPACER_DIST_TOL_PX):
            errs.append({
                "code": "E_SPACER_DISTANCE_ORDER",
                "message": f"Consistency check: spacer2 is not closer to gear11 than spacer3 (tol={SPACER_DIST_TOL_PX}px)."
            })

    total_contact = len(mesh_boxes) + len(mismesh_boxes)
    expected, ok = expected_contact_boxes_from_gear_count(len(gears))
    if not ok:
        errs.append({
            "code": "E_GEAR_COUNT_UNSUPPORTED",
            "message": f"Consistency check: gear_count={len(gears)} does not fit the expected contact rule (requires odd gear count)."
        })
    else:
        if expected is not None and total_contact != expected:
            errs.append({
                "code": "E_CONTACT_COUNT_MISMATCH",
                "message": f"Consistency check: mesh+mismesh={total_contact}, expected={expected} for gear_count={len(gears)} (likely missing Mesh/Mismesh detections)."
            })

    return errs


# =========================
# Detection
# =========================
def run_detection_gear(img_bgr: np.ndarray, gear_model: Any) -> List[Dict[str, Any]]:
    res = gear_model(img_bgr, verbose=False)[0]
    names = res.names
    dets: List[Dict[str, Any]] = []

    if res.boxes is not None and res.boxes.data is not None:
        for x1, y1, x2, y2, conf, cls_id in res.boxes.data.tolist():
            conf = float(conf)
            if conf < CONF_GEAR:
                continue
            cls_id = int(cls_id)
            cls = names.get(cls_id, str(cls_id))
            dets.append({
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "score": conf,
                "cls": cls,
            })
    return dets


def run_detection_shaft_obb(img_bgr: np.ndarray, shaft_model: Any) -> List[Dict[str, Any]]:
    res = shaft_model(img_bgr, verbose=False)[0]
    names = res.names
    dets: List[Dict[str, Any]] = []

    if hasattr(res, "obb") and res.obb is not None:
        data = getattr(res.obb, "data", None)
        if data is None:
            return dets

        rows = data.tolist()
        for row in rows:
            pts: Optional[np.ndarray] = None

            if len(row) >= 10:
                poly8 = row[:8]
                conf = float(row[8])
                cls_id = int(row[9])
                if conf < CONF_SHAFT:
                    continue
                cls = names.get(cls_id, str(cls_id))
                if cls not in TARGET_SHAFT_CLASSES:
                    continue
                pts = np.array([
                    [poly8[0], poly8[1]],
                    [poly8[2], poly8[3]],
                    [poly8[4], poly8[5]],
                    [poly8[6], poly8[7]],
                ], dtype=np.float32)

            elif len(row) >= 7:
                cx, cy, w, h, ang = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
                conf = float(row[5])
                cls_id = int(row[6])
                if conf < CONF_SHAFT:
                    continue
                cls = names.get(cls_id, str(cls_id))
                if cls not in TARGET_SHAFT_CLASSES:
                    continue
                rect = ((cx, cy), (w, h), ang * 180.0 / math.pi)
                pts = cv2.boxPoints(rect).astype(np.float32)
            else:
                continue

            c = poly_center(pts)

            e01 = pts[1] - pts[0]
            e12 = pts[2] - pts[1]
            d = e01 if np.linalg.norm(e01) >= np.linalg.norm(e12) else e12
            n = float(np.linalg.norm(d) + 1e-9)
            d = (d / n).astype(np.float32)

            dets.append({
                "cls": cls,
                "score": conf,
                "poly4": pts,
                "poly4_scaled": scale_poly_about_center(pts, scale=OBB_ASSIGN_SCALE),
                "center": c,
                "axis_dir": d,
            })

    return dets


# =========================
# Build objects
# =========================
def build_objects(gear_dets: List[Dict[str, Any]]) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Tuple[float, float, float, float]],
    List[Tuple[float, float, float, float]],
]:
    gears: List[Dict[str, Any]] = []
    spacers: List[Dict[str, Any]] = []
    mesh_boxes: List[Tuple[float, float, float, float]] = []
    mismesh_boxes: List[Tuple[float, float, float, float]] = []

    for d in gear_dets:
        cls = d["cls"]
        if cls == MESH_CLASS_NAME:
            mesh_boxes.append(d["bbox"])
        elif cls == MISMESH_CLASS_NAME:
            mismesh_boxes.append(d["bbox"])
        elif cls in SPACER_CLASSES:
            b = d["bbox"]
            spacers.append({
                "sid": len(spacers),
                "cls": cls,
                "score": d["score"],
                "bbox": b,
                "center": bbox_center(b),
            })
        elif (cls in DRIVING_GEAR_CLASS_NAMES) or (cls in (GEAR_BIG_NAME, GEAR_SMALL_NAME)):
            b = d["bbox"]
            gears.append({
                "gid": len(gears),
                "cls": cls,
                "score": d["score"],
                "bbox": b,
                "center": bbox_center(b),
                "r": est_radius_from_bbox(b),
            })

    return gears, spacers, mesh_boxes, mismesh_boxes


def pick_driving_gear(gears: List[Dict[str, Any]]) -> Dict[str, Any]:
    cands = [g for g in gears if g["cls"] in DRIVING_GEAR_CLASS_NAMES]
    if cands:
        return max(cands, key=lambda x: x["score"])
    return min(gears, key=lambda x: x["r"])


# =========================
# Assignment to shafts
# =========================
def assign_items_to_shafts(
    gears: List[Dict[str, Any]],
    spacers: List[Dict[str, Any]],
    shafts: List[Dict[str, Any]],
) -> Tuple[Dict[int, int], Dict[int, List[int]], Dict[int, int], Dict[int, List[int]]]:
    gear_to_si: Dict[int, int] = {}
    si_to_gids: Dict[int, List[int]] = {i: [] for i in range(len(shafts))}
    spacer_to_si: Dict[int, int] = {}
    si_to_spacers: Dict[int, List[int]] = {i: [] for i in range(len(shafts))}

    for g in gears:
        c = g["center"]
        candidates = []
        for i, s in enumerate(shafts):
            if point_in_poly(c, s["poly4_scaled"]):
                candidates.append(i)
        if candidates:
            best = max(candidates, key=lambda i: shafts[i]["score"])
            gear_to_si[g["gid"]] = best
            si_to_gids[best].append(g["gid"])

    for sp in spacers:
        c = sp["center"]
        candidates = []
        for i, s in enumerate(shafts):
            if point_in_poly(c, s["poly4_scaled"]):
                candidates.append(i)
        if candidates:
            best = max(candidates, key=lambda i: shafts[i]["score"])
            spacer_to_si[sp["sid"]] = best
            si_to_spacers[best].append(sp["sid"])

    return gear_to_si, si_to_gids, spacer_to_si, si_to_spacers


def gears_by_gid(gears: List[Dict[str, Any]], gid: int) -> Dict[str, Any]:
    return next(g for g in gears if g["gid"] == gid)


# =========================
# Size inference
# =========================
def compute_median_radius(gears: List[Dict[str, Any]]) -> float:
    rs = [g["r"] for g in gears if g["r"] > 1e-6]
    if not rs:
        return 0.0
    rs.sort()
    return float(rs[len(rs) // 2])


def is_big(gear: Dict[str, Any], median_r: float) -> bool:
    if gear["cls"] == GEAR_BIG_NAME:
        return True
    if gear["cls"] == GEAR_SMALL_NAME:
        return False
    return gear["r"] >= median_r


# =========================
# Contact-box scoring + chain naming
# =========================
def score_pair_by_contact_box(
    gA: Dict[str, Any],
    gB: Dict[str, Any],
    box: Tuple[float, float, float, float],
) -> Optional[Tuple[float, float, float, float]]:
    p1 = gA["center"]
    p2 = gB["center"]
    hit = line_hit_ratio(p1, p2, box, samples=LINE_SAMPLES)
    if hit < LINE_HIT_RATIO_TH:
        return None

    cd = center_dist(p1, p2)
    gap = cd - (gA["r"] + gB["r"])

    mx, my = bbox_center(box)
    dms = dist_point_to_segment((mx, my), p1, p2)

    score = (W_HIT * hit) - (W_DMS * (dms / NORM_DMS)) - (W_GAP * (abs(gap) / NORM_GAP))
    return score, hit, dms, gap


def find_best_mate_for(
    gidA: int,
    gears: List[Dict[str, Any]],
    contact_boxes: List[Tuple[float, float, float, float]],
    gear_to_si: Dict[int, int],
    median_r: float,
    used_gids: set,
    used_contact_idx: set,
) -> Tuple[Optional[int], float, Optional[int], Optional[Tuple[float, float, float, float]]]:
    gA = gears_by_gid(gears, gidA)
    siA = gear_to_si.get(gidA)

    best = None

    for gB in gears:
        gidB = gB["gid"]
        if gidB == gidA or gidB in used_gids:
            continue

        siB = gear_to_si.get(gidB)

        if REQUIRE_DIFFERENT_SHAFT and (siA is not None) and (siB is not None) and (siA == siB):
            continue

        if REQUIRE_BIG_SMALL_PAIR and (is_big(gA, median_r) == is_big(gB, median_r)):
            continue

        best_local = None
        for cidx, box in enumerate(contact_boxes):
            if ONE_TO_ONE_CONTACT_BOX and (cidx in used_contact_idx):
                continue
            ret = score_pair_by_contact_box(gA, gB, box)
            if ret is None:
                continue
            sc, hit, dms, gap = ret
            if (best_local is None) or (sc > best_local[0]):
                best_local = (sc, hit, cidx, box)

        if best_local is None:
            continue

        sc, hit, cidx, box = best_local
        if (best is None) or (sc > best[0]):
            best = (sc, gidB, hit, cidx, box)

    if best is None:
        return None, 0.0, None, None
    return best[1], best[2], best[3], best[4]


def pick_compound_on_same_shaft(
    gid_base: int,
    gears: List[Dict[str, Any]],
    gear_to_si: Dict[int, int],
    used_gids: set,
) -> Optional[int]:
    si = gear_to_si.get(gid_base)
    if si is None:
        return None

    g0 = gears_by_gid(gears, gid_base)
    candidates: List[Tuple[float, int]] = []
    for g in gears:
        gid = g["gid"]
        if gid == gid_base or gid in used_gids:
            continue
        if gear_to_si.get(gid) != si:
            continue
        candidates.append((center_dist(g["center"], g0["center"]), gid))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def find_stage2_shaft_index_by_short_spacer(
    spacers: List[Dict[str, Any]],
    spacer_to_si: Dict[int, int],
) -> Tuple[Optional[int], Optional[int]]:
    short_sps = []
    for sp in spacers:
        if spacer_is_short(sp):
            sid = sp["sid"]
            si = spacer_to_si.get(sid)
            if si is None:
                continue
            short_sps.append((sp["score"], si, sid))
    if not short_sps:
        return None, None
    short_sps.sort(key=lambda x: x[0], reverse=True)
    _, si, sid = short_sps[0]
    return si, sid


def pick_closest_gear_to_spacer_on_shaft(
    si_target: int,
    spacer_center: Tuple[float, float],
    gears: List[Dict[str, Any]],
    gear_to_si: Dict[int, int],
    median_r: float,
    require_big: bool = False,
    forbid_gid: Optional[int] = None,
) -> Optional[int]:
    best = None
    for g in gears:
        gid = g["gid"]
        if forbid_gid is not None and gid == forbid_gid:
            continue
        if gear_to_si.get(gid) != si_target:
            continue
        if require_big:
            if not is_big(g, median_r):
                continue
            if g["cls"] != GEAR_BIG_NAME:
                continue
        d = center_dist(g["center"], spacer_center)
        if (best is None) or (d < best[0]):
            best = (d, gid)
    return None if best is None else best[1]


def stage_role_naming_chain(
    gears: List[Dict[str, Any]],
    spacers: List[Dict[str, Any]],
    contact_boxes: List[Tuple[float, float, float, float]],
    gear_to_si: Dict[int, int],
    spacer_to_si: Dict[int, int],
    gear11_gid: int,
) -> Tuple[Dict[int, str], Dict[int, int], List[Tuple[int, int, float, Optional[Tuple[float, float, float, float]]]]]:
    median_r = compute_median_radius(gears)

    labels: Dict[int, str] = {}
    stage_of: Dict[int, int] = {}
    used_gids = set()
    used_contact_idx = set()
    chain_pairs: List[Tuple[int, int, float, Optional[Tuple[float, float, float, float]]]] = []

    labels[gear11_gid] = "gear11"
    stage_of[gear11_gid] = 1
    used_gids.add(gear11_gid)

    mate12 = None
    hit12 = 0.0
    box12 = None
    cidx12 = None

    if FORCE_STAGE1_BY_SHORT_SPACER:
        si2, sid_short = find_stage2_shaft_index_by_short_spacer(spacers, spacer_to_si)
        if si2 is not None and sid_short is not None:
            sp_short = next(sp for sp in spacers if sp["sid"] == sid_short)

            mate12 = pick_closest_gear_to_spacer_on_shaft(
                si_target=si2,
                spacer_center=sp_short["center"],
                gears=gears,
                gear_to_si=gear_to_si,
                median_r=median_r,
                require_big=STAGE1_MATE_MUST_BE_GEAR_BIG,
                forbid_gid=gear11_gid,
            )
            if mate12 is None:
                mate12 = pick_closest_gear_to_spacer_on_shaft(
                    si_target=si2,
                    spacer_center=sp_short["center"],
                    gears=gears,
                    gear_to_si=gear_to_si,
                    median_r=median_r,
                    require_big=False,
                    forbid_gid=gear11_gid,
                )

            if mate12 is not None:
                gA = gears_by_gid(gears, gear11_gid)
                gB = gears_by_gid(gears, mate12)
                best_local = None
                for cidx, box in enumerate(contact_boxes):
                    if ONE_TO_ONE_CONTACT_BOX and (cidx in used_contact_idx):
                        continue
                    ret = score_pair_by_contact_box(gA, gB, box)
                    if ret is None:
                        continue
                    sc, hit, dms, gap = ret
                    if (best_local is None) or (sc > best_local[0]):
                        best_local = (sc, hit, cidx, box)
                if best_local is not None:
                    _, hit12, cidx12, box12 = best_local

    if mate12 is None:
        mate12, hit12, cidx12, box12 = find_best_mate_for(
            gidA=gear11_gid,
            gears=gears,
            contact_boxes=contact_boxes,
            gear_to_si=gear_to_si,
            median_r=median_r,
            used_gids=used_gids,
            used_contact_idx=used_contact_idx,
        )

    if mate12 is None:
        return labels, stage_of, chain_pairs

    labels[mate12] = "gear12"
    stage_of[mate12] = 1
    used_gids.add(mate12)
    if ONE_TO_ONE_CONTACT_BOX and cidx12 is not None:
        used_contact_idx.add(cidx12)

    chain_pairs.append((gear11_gid, mate12, hit12, box12))
    prev_driven = mate12

    for stage in range(2, MAX_STAGE + 1):
        driving = pick_compound_on_same_shaft(prev_driven, gears, gear_to_si, used_gids)
        if driving is None:
            break

        labels[driving] = f"gear{stage}1"
        stage_of[driving] = stage
        used_gids.add(driving)

        driven, hit, cidx, box = find_best_mate_for(
            gidA=driving,
            gears=gears,
            contact_boxes=contact_boxes,
            gear_to_si=gear_to_si,
            median_r=median_r,
            used_gids=used_gids,
            used_contact_idx=used_contact_idx,
        )
        if driven is None:
            break

        labels[driven] = f"gear{stage}2"
        stage_of[driven] = stage
        used_gids.add(driven)
        if ONE_TO_ONE_CONTACT_BOX and cidx is not None:
            used_contact_idx.add(cidx)

        chain_pairs.append((driving, driven, hit, box))
        prev_driven = driven

    return labels, stage_of, chain_pairs


# =========================
# Task-result helpers
# =========================
def _has_E(errors: List[Dict[str, Any]]) -> bool:
    return any(isinstance(e, dict) and str(e.get("code", "")).startswith("E_") for e in errors)


def _task_result(task: str, errors: List[Dict[str, Any]], focus: List[str], next_task: str) -> Dict[str, Any]:
    fail = _has_E(errors)
    return {
        "task": task,
        "status": "FAIL" if fail else "PASS",
        "is_ready_for_next": (not fail),
        "focus": focus,
        "messages": [str(e.get("message", "")) for e in errors[:6] if isinstance(e, dict)],
        "recommended_next_task": next_task,
    }


def _filter_errors_by_prefix(errors: List[Dict[str, str]], prefixes: Tuple[str, ...]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for e in errors:
        if not isinstance(e, dict):
            continue
        code = str(e.get("code", "")).upper()
        if any(code.startswith(p.upper()) for p in prefixes):
            out.append(e)
    return out


# =========================
# Public API
# =========================
def run_yolo_pipeline(
    img_bgr: np.ndarray,
    gear_model_rel: str = DEFAULT_GEAR_MODEL_REL,
    shaft_model_rel: str = DEFAULT_SHAFT_MODEL_REL,
    return_images: bool = False,
    *,
    task: str = TASK_FULL,
    part_type: Optional[str] = None,
    expected_gears: Optional[int] = None,
) -> Dict[str, Any]:
    if img_bgr is None or not hasattr(img_bgr, "shape"):
        raise ValueError("img_bgr must be a valid OpenCV image (BGR).")

    task = str(task or TASK_FULL).strip().lower()
    if task not in _VALID_TASKS:
        task = TASK_FULL

    part_type = str(part_type or "").strip().lower()

    global _COLD_START_FLAG
    is_cold_start = _COLD_START_FLAG
    _COLD_START_FLAG = False

    t0_total = time.perf_counter()
    timing: Dict[str, float] = {}

    gear_model, shaft_model = get_models(gear_model_rel, shaft_model_rel, timing=timing)

    t_g0 = time.perf_counter()
    gear_dets = run_detection_gear(img_bgr, gear_model)
    timing["t_infer_gear_s"] = float(time.perf_counter() - t_g0)

    t_s0 = time.perf_counter()
    shaft_obbs = run_detection_shaft_obb(img_bgr, shaft_model)
    timing["t_infer_shaft_s"] = float(time.perf_counter() - t_s0)

    gears, spacers, mesh_boxes, mismesh_boxes = build_objects(gear_dets)
    contact_boxes = list(mesh_boxes) + list(mismesh_boxes)

    gear11_gid: Optional[int] = None
    gear_to_si: Dict[int, int] = {}
    spacer_to_si: Dict[int, int] = {}

    if gears:
        driving = pick_driving_gear(gears)
        gear11_gid = int(driving["gid"])

    if gears and shaft_obbs:
        gear_to_si, _si_to_gids, spacer_to_si, _si_to_spacers = assign_items_to_shafts(gears, spacers, shaft_obbs)

    errors_all: List[Dict[str, str]] = []

    def _need_gears() -> bool:
        return task in (
            TASK_PARTS_INVENTORY,
            TASK_PRECHECK,
            TASK_SINGLE_STAGE,
            TASK_SHAFT,
            TASK_SPACER,
            TASK_MESH_RATIO,
            TASK_FULL,
            TASK_GEAR_INV,
        )

    def _need_shafts() -> bool:
        return task in (
            TASK_PARTS_INVENTORY,
            TASK_SINGLE_STAGE,
            TASK_SHAFT,
            TASK_SPACER,
            TASK_MESH_RATIO,
            TASK_FULL,
        )

    if _need_gears() and not gears:
        errors_all.append({"code": "E_NO_GEARS", "message": "No gears detected."})
    if _need_shafts() and not shaft_obbs:
        errors_all.append({"code": "E_NO_SHAFTS", "message": "No shafts detected."})

    # --- task: parts_inventory ---
    if task == TASK_PARTS_INVENTORY:
        inv = evaluate_parts_inventory(
            part_type=part_type,
            gear_dets=gear_dets,
            shaft_dets=shaft_obbs,
        )

        out: Dict[str, Any] = {
            "summary": {
                "gears": len(gears),
                "spacers": len(spacers),
                "shafts": len(shaft_obbs),
                "mesh": len(mesh_boxes),
                "mismesh": len(mismesh_boxes),
                "stages": 0,
                "part_type": part_type,
            },
            "counts": inv.get("counts", {}),
            "errors": inv.get("errors", []),
            "ratio": {"num_stages": 0, "R_total": None, "out_rpm": None, "per_stage": []},
            "cold_start": bool(is_cold_start),
            "task_result": _task_result(
                TASK_PARTS_INVENTORY,
                inv.get("errors", []),
                focus=[part_type or "parts_inventory"],
                next_task=TASK_PRECHECK,
            ),
            "timing": {},
        }
        timing["t_total_pipeline_s"] = float(time.perf_counter() - t0_total)
        out["timing"] = timing
        return out

    # --- task: precheck ---
    if task == TASK_PRECHECK:
        errors = evaluate_precheck_count_rule(
            gears=gears,
            mesh_boxes=mesh_boxes,
            mismesh_boxes=mismesh_boxes,
        )

        out: Dict[str, Any] = {
            "summary": {
                "gears": len(gears),
                "spacers": len(spacers),
                "shafts": len(shaft_obbs),
                "mesh": len(mesh_boxes),
                "mismesh": len(mismesh_boxes),
                "stages": 0,
            },
            "errors": errors,
            "ratio": {"num_stages": 0, "R_total": None, "out_rpm": None, "per_stage": []},
            "cold_start": bool(is_cold_start),
            "task_result": _task_result(
                TASK_PRECHECK,
                errors,
                focus=["precheck"],
                next_task=TASK_SINGLE_STAGE,
            ),
            "timing": {},
        }
        timing["t_total_pipeline_s"] = float(time.perf_counter() - t0_total)
        out["timing"] = timing
        return out

    # --- task: single_stage ---
    if task == TASK_SINGLE_STAGE:
        if not gears:
            out = {
                "summary": {
                    "gears": 0,
                    "spacers": len(spacers),
                    "shafts": len(shaft_obbs),
                    "mesh": len(mesh_boxes),
                    "mismesh": len(mismesh_boxes),
                    "stages": 0,
                },
                "errors": [{"code": "E_NO_GEARS", "message": "No gears detected."}],
                "ratio": {"num_stages": 0, "R_total": None, "out_rpm": None, "per_stage": []},
                "cold_start": bool(is_cold_start),
                "task_result": _task_result(
                    TASK_SINGLE_STAGE,
                    [{"code": "E_NO_GEARS", "message": "No gears detected."}],
                    focus=["single_stage"],
                    next_task=TASK_SHAFT,
                ),
                "timing": {},
            }
            timing["t_total_pipeline_s"] = float(time.perf_counter() - t0_total)
            out["timing"] = timing
            return out

        if gear11_gid is None:
            gear11_gid = pick_driving_gear(gears)["gid"]

        if gears and shaft_obbs and not gear_to_si:
            gear_to_si, _si_to_gids, spacer_to_si, _si_to_spacers = assign_items_to_shafts(
                gears, spacers, shaft_obbs
            )

        gear_names, gear_stage, chain_pairs = stage_role_naming_chain(
            gears=gears,
            spacers=spacers,
            contact_boxes=contact_boxes,
            gear_to_si=gear_to_si,
            spacer_to_si=spacer_to_si,
            gear11_gid=int(gear11_gid),
        )

        num_stages, R_total, out_rpm, per_stage = compute_ratio_and_rpm_from_stage_labels(
            gears=gears,
            gear_names=gear_names,
            motor_rpm=MOTOR_RPM,
            max_stage=MAX_STAGE,
        )

        ratio = {
            "num_stages": num_stages,
            "R_total": R_total,
            "out_rpm": out_rpm,
            "per_stage": per_stage,
        }

        errors = evaluate_single_stage_errors(
            gears=gears,
            spacers=spacers,
            shafts=shaft_obbs,
            mismesh_boxes=mismesh_boxes,
            ratio=ratio,
        )

        out = {
            "summary": {
                "gears": len(gears),
                "spacers": len(spacers),
                "shafts": len(shaft_obbs),
                "mesh": len(mesh_boxes),
                "mismesh": len(mismesh_boxes),
                "stages": num_stages,
            },
            "gear_names": gear_names,
            "gear_stage": gear_stage,
            "chain_pairs": chain_pairs,
            "ratio": ratio,
            "errors": errors,
            "cold_start": bool(is_cold_start),
            "task_result": _task_result(
                TASK_SINGLE_STAGE,
                errors,
                focus=["single_stage"],
                next_task=TASK_SHAFT,
            ),
            "timing": {},
        }
        timing["t_total_pipeline_s"] = float(time.perf_counter() - t0_total)
        out["timing"] = timing
        return out

    # --- task: gear_inventory ---
    if task == TASK_GEAR_INV:
        big_cnt = 0
        small_cnt = 0
        if gears:
            med = compute_median_radius(gears)
            for g in gears:
                if is_big(g, med):
                    big_cnt += 1
                else:
                    small_cnt += 1

        if expected_gears is not None and gears:
            try:
                exp = int(expected_gears)
                if len(gears) != exp:
                    errors_all.append({
                        "code": "E_GEAR_COUNT_MISMATCH",
                        "message": f"Gear inventory: detected {len(gears)}, expected {exp}."
                    })
            except Exception:
                pass

        errors = _filter_errors_by_prefix(errors_all, prefixes=("E_NO_GEARS", "E_GEAR_COUNT", "E_GEAR"))

        out: Dict[str, Any] = {
            "summary": {
                "gears": len(gears),
                "spacers": len(spacers),
                "shafts": len(shaft_obbs),
                "mesh": len(mesh_boxes),
                "mismesh": len(mismesh_boxes),
                "stages": 0,
                "gear_big": big_cnt,
                "gear_small": small_cnt,
            },
            "errors": errors,
            "ratio": {"num_stages": 0, "R_total": None, "out_rpm": None, "per_stage": []},
            "cold_start": bool(is_cold_start),
            "task_result": _task_result(
                TASK_GEAR_INV,
                errors,
                focus=["gears"],
                next_task=TASK_MESH_RATIO,
            ),
            "timing": {},
        }
        timing["t_total_pipeline_s"] = float(time.perf_counter() - t0_total)
        out["timing"] = timing
        return out

    # --- task: shaft ---
    if task == TASK_SHAFT:
        if errors_all:
            errors = _filter_errors_by_prefix(errors_all, prefixes=("E_NO_GEARS", "E_NO_SHAFTS"))
        else:
            if gear11_gid is None:
                errors_all_local: List[Dict[str, str]] = [{
                    "code": "E_NO_GEAR11",
                    "message": "Cannot determine gear11 (driving gear) from detections."
                }]
            else:
                errors_all_local = evaluate_shaft_step_errors(
                    gears=gears,
                    shafts=shaft_obbs,
                    gear11_gid=gear11_gid,
                )

            errors = _filter_errors_by_prefix(
                errors_all_local,
                prefixes=("E_SHAFT", "E_NO_GEAR11")
            )

        out = {
            "summary": {
                "gears": len(gears),
                "spacers": len(spacers),
                "shafts": len(shaft_obbs),
                "mesh": len(mesh_boxes),
                "mismesh": len(mismesh_boxes),
                "stages": 0,
            },
            "errors": errors,
            "ratio": {"num_stages": 0, "R_total": None, "out_rpm": None, "per_stage": []},
            "cold_start": bool(is_cold_start),
            "task_result": _task_result(
                TASK_SHAFT,
                errors,
                focus=["shafts"],
                next_task=TASK_SPACER,
            ),
            "timing": {},
        }
        timing["t_total_pipeline_s"] = float(time.perf_counter() - t0_total)
        out["timing"] = timing
        return out

    # --- task: spacer ---
    if task == TASK_SPACER:
        if errors_all:
            errors = _filter_errors_by_prefix(errors_all, prefixes=("E_NO_GEARS", "E_NO_SHAFTS"))
        else:
            if gear11_gid is None:
                errors_all_local: List[Dict[str, str]] = [{
                    "code": "E_NO_GEAR11",
                    "message": "Cannot determine gear11 (driving gear) from detections."
                }]
            else:
                errors_all_local = evaluate_spacer_step_errors(
                    gears=gears,
                    spacers=spacers,
                    shafts=shaft_obbs,
                    gear11_gid=gear11_gid,
                    spacer_to_si=spacer_to_si,
                )

            errors = _filter_errors_by_prefix(
                errors_all_local,
                prefixes=("E_SPACER", "E_NO_GEAR11")
            )

        out = {
            "summary": {
                "gears": len(gears),
                "spacers": len(spacers),
                "shafts": len(shaft_obbs),
                "mesh": len(mesh_boxes),
                "mismesh": len(mismesh_boxes),
                "stages": 0,
            },
            "errors": errors,
            "ratio": {"num_stages": 0, "R_total": None, "out_rpm": None, "per_stage": []},
            "cold_start": bool(is_cold_start),
            "task_result": _task_result(
                TASK_SPACER,
                errors,
                focus=["spacers"],
                next_task=TASK_GEAR_INV,
            ),
            "timing": {},
        }
        timing["t_total_pipeline_s"] = float(time.perf_counter() - t0_total)
        out["timing"] = timing
        return out

    # --- task: mesh_ratio / full ---
    if not gears:
        out = {
            "summary": {
                "gears": 0,
                "spacers": len(spacers),
                "shafts": len(shaft_obbs),
                "mesh": len(mesh_boxes),
                "mismesh": len(mismesh_boxes),
                "stages": 0,
            },
            "errors": [{"code": "E_NO_GEARS", "message": "No gears detected."}],
            "gear_names": {},
            "gear_stage": {},
            "chain_pairs": [],
            "ratio": {"num_stages": 0, "R_total": None, "out_rpm": None, "per_stage": []},
            "cold_start": bool(is_cold_start),
            "task_result": _task_result(
                TASK_MESH_RATIO if task == TASK_MESH_RATIO else TASK_FULL,
                [{"code": "E_NO_GEARS", "message": "No gears detected."}],
                focus=["mesh_ratio"],
                next_task=TASK_SHAFT,
            ),
            "timing": {},
        }
        timing["t_total_pipeline_s"] = float(time.perf_counter() - t0_total)
        out["timing"] = timing
        return out

    if gear11_gid is None:
        gear11_gid = pick_driving_gear(gears)["gid"]

    if gears and shaft_obbs and not gear_to_si:
        gear_to_si, _si_to_gids, spacer_to_si, _si_to_spacers = assign_items_to_shafts(gears, spacers, shaft_obbs)

    errors: List[Dict[str, str]] = []
    if ENABLE_ERROR_CHECKS and shaft_obbs:
        errors = evaluate_assembly_errors(
            gears=gears,
            spacers=spacers,
            shafts=shaft_obbs,
            mesh_boxes=mesh_boxes,
            mismesh_boxes=mismesh_boxes,
            gear11_gid=int(gear11_gid),
            gear_to_si=gear_to_si,
            spacer_to_si=spacer_to_si,
        )

    gear_names, gear_stage, chain_pairs = stage_role_naming_chain(
        gears=gears,
        spacers=spacers,
        contact_boxes=contact_boxes,
        gear_to_si=gear_to_si,
        spacer_to_si=spacer_to_si,
        gear11_gid=int(gear11_gid),
    )

    num_stages, R_total, out_rpm, per_stage = compute_ratio_and_rpm_from_stage_labels(
        gears=gears,
        gear_names=gear_names,
        motor_rpm=MOTOR_RPM,
        max_stage=MAX_STAGE,
    )

    out = {
        "summary": {
            "gears": len(gears),
            "spacers": len(spacers),
            "shafts": len(shaft_obbs),
            "mesh": len(mesh_boxes),
            "mismesh": len(mismesh_boxes),
            "stages": num_stages,
        },
        "detections": {
            "gear_dets": gear_dets,
            "shaft_obbs": shaft_obbs,
        },
        "objects": {
            "gears": gears,
            "spacers": spacers,
            "mesh_boxes": mesh_boxes,
            "mismesh_boxes": mismesh_boxes,
            "shafts": shaft_obbs,
            "gear_to_si": gear_to_si,
            "spacer_to_si": spacer_to_si,
        },
        "gear_names": gear_names,
        "gear_stage": gear_stage,
        "chain_pairs": chain_pairs,
        "ratio": {
            "num_stages": num_stages,
            "R_total": R_total,
            "out_rpm": out_rpm,
            "per_stage": per_stage,
        },
        "errors": errors,
        "cold_start": bool(is_cold_start),
        "task_result": _task_result(
            TASK_MESH_RATIO if task == TASK_MESH_RATIO else TASK_FULL,
            errors,
            focus=["mesh_ratio"],
            next_task=TASK_FULL,
        ),
        "timing": {},
    }

    timing["t_total_pipeline_s"] = float(time.perf_counter() - t0_total)
    out["timing"] = timing
    return out