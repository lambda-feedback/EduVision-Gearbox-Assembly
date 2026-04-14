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
CONF_MODEL_A: float = 0.50
CONF_MODEL_B: float = 0.50
CONF_MODEL_C: float = 0.50

MOTOR_RPM: float = 8000.0
TEETH_BIG: int = 48
TEETH_SMALL: int = 12

DRIVING_GEAR_CLASS_NAMES = {"Driving_Gear", "driving_gear", "DrivingGear"}
GEAR_BIG_NAME = "Gear_big"
GEAR_SMALL_NAME = "Gear_small"
MESH_CLASS_NAME = "Mesh"
MISMESH_CLASS_NAME = "Mismesh"

SPACER_CLASSES = {
    "spacer_long",
    "spacer_short",
    "Long spacer tube",
    "Short spacer tube",
    "spacer tube long",
    "spacer tube short",
}
TARGET_SHAFT_CLASSES = {"shaft_long", "shaft_short"}

OBB_ASSIGN_SCALE: float = 1.10

LINE_SAMPLES: int = 25
LINE_HIT_RATIO_TH: float = 0.25

REQUIRE_DIFFERENT_SHAFT: bool = True
REQUIRE_BIG_SMALL_PAIR: bool = True

ONE_TO_ONE_CONTACT_BOX: bool = True
MAX_STAGE: int = 6

W_HIT: float = 1.0
W_DMS: float = 0.6
W_GAP: float = 0.4
NORM_DMS: float = 200.0
NORM_GAP: float = 200.0

ENABLE_ERROR_CHECKS: bool = True

# Image-quality precheck thresholds. Scores are normalized to 0-100 for display.
# The fail threshold is intentionally permissive so borderline usable photos are
# not rejected while obviously dark/blurry/noisy ones are sent back for retake.
QUALITY_ACCEPT_SCORE: int = int(os.environ.get("QUALITY_ACCEPT_SCORE", "40"))
QUALITY_MIN_COMPONENT_SCORE: int = int(os.environ.get("QUALITY_MIN_COMPONENT_SCORE", "20"))
QUALITY_MIN_SHARPNESS_SCORE: int = int(os.environ.get("QUALITY_MIN_SHARPNESS_SCORE", "10"))
QUALITY_WARNING_COMPONENT_SCORE: int = int(os.environ.get("QUALITY_WARNING_COMPONENT_SCORE", "50"))
QUALITY_WARNING_SHARPNESS_SCORE: int = int(os.environ.get("QUALITY_WARNING_SHARPNESS_SCORE", "25"))
QUALITY_BRIGHTNESS_USABLE_MIN: float = 35.0
QUALITY_BRIGHTNESS_IDEAL_MIN: float = 60.0
QUALITY_BRIGHTNESS_IDEAL_MAX: float = 210.0
QUALITY_BRIGHTNESS_USABLE_MAX: float = 235.0
QUALITY_CONTRAST_USABLE_MIN: float = 10.0
QUALITY_CONTRAST_IDEAL_MIN: float = 25.0
QUALITY_SHARPNESS_USABLE_MIN: float = 40.0
QUALITY_SHARPNESS_IDEAL_MIN: float = 250.0
QUALITY_NOISE_IDEAL_MAX: float = 12.0
QUALITY_NOISE_USABLE_MAX: float = 35.0

# Geometry / ambiguity thresholds
SHAFT_DISTANCE_AMBIG_RATIO: float = 0.08
SPACER_ASSIGN_AXIS_DIST_RATIO: float = 0.90
SPACER_ASSIGN_CENTER_DIST_RATIO: float = 0.90
SPACER_DIST_TOL_RATIO: float = 0.12

# Drawing config
BOX_THICK = 2
CENTER_R = 4

LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.85
LABEL_THICK = 2
LABEL_PAD = 3
LEADER_THICK = 2

HUD_SCALE = 0.85
HUD_THICK = 2
HUD_LINE_GAP = 28
HUD_X = 20
HUD_Y0 = 30
HUD_COLOR = (0, 255, 255)


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

DEFAULT_MODEL_A_REL = "modelA.pt"
DEFAULT_MODEL_B_REL = "modelB.pt"
DEFAULT_MODEL_C_REL = "modelC.pt"

# Backward-compatible aliases
DEFAULT_GEAR_MODEL_REL = DEFAULT_MODEL_A_REL
DEFAULT_SHAFT_MODEL_REL = DEFAULT_MODEL_B_REL


def _abs_model_path(rel_name: str) -> str:
    return os.path.join(_THIS_DIR, rel_name)


# =========================
# Model cache
# =========================
@lru_cache(maxsize=3)
def _load_yolo_model(abs_path: str) -> Any:
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Model not found: {abs_path}")

    YOLO_cls = _ultralytics.YOLO
    return YOLO_cls(abs_path)


def get_models(
    model_a_rel: str = DEFAULT_MODEL_A_REL,
    model_b_rel: str = DEFAULT_MODEL_B_REL,
    model_c_rel: str = DEFAULT_MODEL_C_REL,
    timing: Optional[Dict[str, float]] = None,
) -> Tuple[Any, Any, Any]:
    t0 = time.perf_counter()
    model_a = _load_yolo_model(_abs_model_path(model_a_rel))
    t1 = time.perf_counter()
    model_b = _load_yolo_model(_abs_model_path(model_b_rel))
    t2 = time.perf_counter()
    model_c = _load_yolo_model(_abs_model_path(model_c_rel))
    t3 = time.perf_counter()

    if timing is not None:
        timing["t_load_model_a_s"] = float(t1 - t0)
        timing["t_load_model_b_s"] = float(t2 - t1)
        timing["t_load_model_c_s"] = float(t3 - t2)
        timing["t_get_models_total_s"] = float(t3 - t0)

    return model_a, model_b, model_c


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


def dist_point_to_segment(
    p: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
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


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default


def shaft_length_width_from_poly(pts4: np.ndarray) -> Tuple[float, float, np.ndarray]:
    pts = np.asarray(pts4, dtype=np.float32)
    e01 = pts[1] - pts[0]
    e12 = pts[2] - pts[1]
    len01 = float(np.linalg.norm(e01))
    len12 = float(np.linalg.norm(e12))

    if len01 >= len12:
        major = len01
        minor = len12
        axis = e01 / (len01 + 1e-9)
    else:
        major = len12
        minor = len01
        axis = e12 / (len12 + 1e-9)

    return major, minor, axis.astype(np.float32)


def dist_point_to_shaft_axis(pt: Tuple[float, float], shaft: Dict[str, Any]) -> float:
    c = shaft["center"]
    major = float(shaft.get("major_len", 0.0))
    axis = shaft["axis_dir"]

    half = 0.5 * major
    p1 = (float(c[0] - axis[0] * half), float(c[1] - axis[1] * half))
    p2 = (float(c[0] + axis[0] * half), float(c[1] + axis[1] * half))
    return dist_point_to_segment(pt, p1, p2)


def relative_spacer_distance_tol(shafts: List[Dict[str, Any]], gears: List[Dict[str, Any]]) -> float:
    shaft_lengths = [float(s.get("major_len", 0.0)) for s in shafts if float(s.get("major_len", 0.0)) > 1e-6]
    gear_rs = [float(g.get("r", 0.0)) for g in gears if float(g.get("r", 0.0)) > 1e-6]

    ref = 0.0
    if shaft_lengths:
        ref = max(ref, float(np.median(np.array(shaft_lengths, dtype=np.float32))))
    if gear_rs:
        ref = max(ref, 2.0 * float(np.median(np.array(gear_rs, dtype=np.float32))))

    if ref <= 1e-6:
        ref = 40.0

    return max(5.0, SPACER_DIST_TOL_RATIO * ref)


# =========================
# Drawing helpers
# =========================
def put_text_outline(
    img: np.ndarray,
    text: str,
    org: Tuple[float, float],
    scale: float = 0.6,
    thick: int = 2,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> None:
    x, y = int(org[0]), int(org[1])
    cv2.putText(img, text, (x, y), LABEL_FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), LABEL_FONT, scale, color, thick, cv2.LINE_AA)


def draw_bbox(
    img: np.ndarray,
    b: Tuple[float, float, float, float],
    color: Tuple[int, int, int],
    thick: int = 2,
) -> None:
    x1, y1, x2, y2 = map(int, b)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)


def draw_hud_lines(
    img: np.ndarray,
    lines: List[str],
    x: int = HUD_X,
    y0: int = HUD_Y0,
    scale: float = HUD_SCALE,
    thick: int = HUD_THICK,
    color: Tuple[int, int, int] = HUD_COLOR,
    line_gap: int = HUD_LINE_GAP,
) -> None:
    y = int(y0)
    for s in lines:
        put_text_outline(img, str(s), (x, y), scale=scale, thick=thick, color=color)
        y += int(line_gap)


def text_box_size(text: str, scale: float, thick: int) -> Tuple[int, int, int]:
    (w, h), baseline = cv2.getTextSize(text, LABEL_FONT, scale, thick)
    return w, h, baseline


def rect_intersect(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def place_label_no_overlap(
    img: np.ndarray,
    text: str,
    anchor_xy: Tuple[float, float],
    occupied_rects: List[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (0, 255, 0),
    scale: float = LABEL_SCALE,
    thick: int = LABEL_THICK,
    pad: int = LABEL_PAD,
    leader: bool = True,
) -> Tuple[Tuple[int, int], Tuple[int, int, int, int]]:
    H, W = img.shape[:2]
    ax, ay = int(anchor_xy[0]), int(anchor_xy[1])
    w, h, baseline = text_box_size(text, scale, thick)

    offsets = [
        (8, -8),
        (8, 14),
        (-w - 8, -8),
        (-w - 8, 14),
        (10, -h - 10),
        (-w - 10, -h - 10),
        (12, 0),
        (-w - 12, 0),
    ]

    best = None
    best_collisions = 10**9

    for dx, dy in offsets:
        tx = ax + dx
        ty = ay + dy

        tx = max(pad, min(tx, W - w - pad))
        ty = max(h + pad, min(ty, H - pad))

        rect = (tx - pad, ty - h - pad, tx + w + pad, ty + baseline + pad)

        collisions = 0
        for r in occupied_rects:
            if rect_intersect(rect, r):
                collisions += 1

        if collisions == 0:
            put_text_outline(img, text, (tx, ty), scale=scale, thick=thick, color=color)
            occupied_rects.append(rect)

            if leader:
                label_anchor = (tx, ty - h // 2)
                dist = ((label_anchor[0] - ax) ** 2 + (label_anchor[1] - ay) ** 2) ** 0.5
                if dist > 10:
                    cv2.line(img, (ax, ay), label_anchor, (255, 255, 255), LEADER_THICK, cv2.LINE_AA)

            return (tx, ty), rect

        if collisions < best_collisions:
            best_collisions = collisions
            best = (tx, ty, rect)

    tx, ty, rect = best
    put_text_outline(img, text, (tx, ty), scale=scale, thick=thick, color=color)
    occupied_rects.append(rect)

    if leader:
        label_anchor = (tx, ty - h // 2)
        dist = ((label_anchor[0] - ax) ** 2 + (label_anchor[1] - ay) ** 2) ** 0.5
        if dist > 10:
            cv2.line(img, (ax, ay), label_anchor, (255, 255, 255), LEADER_THICK, cv2.LINE_AA)

    return (tx, ty), rect


def highlight_gear_by_name(
    img: np.ndarray,
    gears: List[Dict[str, Any]],
    gear_names: Dict[int, str],
    target_name: str,
    color: Tuple[int, int, int] = (0, 0, 255),
    thick: int = 4,
) -> None:
    for g in gears:
        gid = g["gid"]
        if gear_names.get(gid, None) == target_name:
            draw_bbox(img, g["bbox"], color, thick)
            cx, cy = int(g["center"][0]), int(g["center"][1])
            cv2.circle(img, (cx, cy), CENTER_R + 2, color, -1)
            put_text_outline(img, f"{target_name} ({g['cls']})", (cx + 12, cy - 12), 0.75, 2, color)
            break


def build_output_images(
    img_bgr: np.ndarray,
    gears: List[Dict[str, Any]],
    spacers: List[Dict[str, Any]],
    shaft_obbs: List[Dict[str, Any]],
    mesh_boxes: List[Tuple[float, float, float, float]],
    mismesh_boxes: List[Tuple[float, float, float, float]],
    gear_names: Optional[Dict[int, str]] = None,
    gear_stage: Optional[Dict[int, int]] = None,
    chain_pairs: Optional[List[Tuple[int, int, float, Optional[Tuple[float, float, float, float]]]]] = None,
    ratio: Optional[Dict[str, Any]] = None,
    errors: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, np.ndarray]:
    det_img = img_bgr.copy()
    label_img = img_bgr.copy()

    gear_names = gear_names or {}
    gear_stage = gear_stage or {}
    chain_pairs = chain_pairs or []
    ratio = ratio or {}
    errors = errors or []

    # Detection image
    for s in shaft_obbs:
        if "poly4" in s and s["poly4"] is not None:
            poly = np.asarray(s["poly4"], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(det_img, [poly], True, (255, 255, 255), BOX_THICK, cv2.LINE_AA)
            put_text_outline(
                det_img,
                f"{s['cls']} {s['score']:.2f}",
                (s["center"][0] + 6, s["center"][1] + 6),
                0.55,
                2,
                (255, 255, 255),
            )

    for b in mesh_boxes:
        draw_bbox(det_img, b, (255, 255, 0), 2)
        put_text_outline(det_img, "Mesh", (b[0] + 3, b[1] - 6), 0.55, 2, (255, 255, 0))

    for b in mismesh_boxes:
        draw_bbox(det_img, b, (255, 0, 255), 2)
        put_text_outline(det_img, "Mismesh", (b[0] + 3, b[1] - 6), 0.55, 2, (255, 0, 255))

    for sp in spacers:
        draw_bbox(det_img, sp["bbox"], (0, 255, 255), 2)
        put_text_outline(det_img, f"{sp['cls']} {sp['score']:.2f}", (sp["bbox"][0] + 3, sp["bbox"][1] - 6), 0.55, 2, (0, 255, 255))

    for g in gears:
        draw_bbox(det_img, g["bbox"], (0, 255, 0), 2)
        put_text_outline(det_img, f"{g['cls']} {g['score']:.2f}", (g["bbox"][0] + 3, g["bbox"][1] - 6), 0.55, 2, (0, 255, 0))

    # Label image
    for g in gears:
        draw_bbox(label_img, g["bbox"], (0, 120, 0), 1)
        cx, cy = int(g["center"][0]), int(g["center"][1])
        cv2.circle(label_img, (cx, cy), CENTER_R, (0, 0, 255), -1)

    occupied: List[Tuple[int, int, int, int]] = []
    for g in gears:
        gid = g["gid"]
        if gid not in gear_names:
            continue
        nm = gear_names[gid]
        cx, cy = g["center"]
        place_label_no_overlap(
            label_img,
            nm,
            (cx, cy),
            occupied,
            color=(0, 255, 0),
            scale=0.90,
            thick=2,
            leader=True,
        )

    for (gidA, gidB, hit, box) in chain_pairs:
        try:
            gA = gears_by_gid(gears, gidA)
            gB = gears_by_gid(gears, gidB)
        except StopIteration:
            continue

        p1 = (int(gA["center"][0]), int(gA["center"][1]))
        p2 = (int(gB["center"][0]), int(gB["center"][1]))
        cv2.line(label_img, p1, p2, (255, 255, 255), 2, cv2.LINE_AA)

        mid = (int((p1[0] + p2[0]) * 0.5), int((p1[1] + p2[1]) * 0.5))
        put_text_outline(label_img, f"contact:{hit:.2f}", (mid[0] + 6, mid[1] + 6), 0.55, 2, (255, 255, 255))

    highlight_gear_by_name(label_img, gears, gear_names, "gear11", color=(0, 0, 255), thick=5)
    highlight_gear_by_name(label_img, gears, gear_names, "gear12", color=(0, 165, 255), thick=5)

    num_stages = ratio.get("num_stages", 0)
    R_total = ratio.get("R_total", None)
    out_rpm = ratio.get("out_rpm", None)
    per_stage = ratio.get("per_stage", [])

    hud_lines: List[str] = [
        f"Gears detected: {len(gears)}",
        f"Stages (computed): {num_stages}",
    ]

    if R_total is None or out_rpm is None:
        hud_lines += [
            "Total gear ratio (slowdown): N/A",
            "Output shaft speed: N/A",
        ]
    else:
        hud_lines += [
            f"Total gear ratio (slowdown): {R_total:.3f}",
            f"Output shaft speed: {out_rpm:.1f} RPM (motor={MOTOR_RPM:.0f})",
        ]
        for (s, R, z1, z2) in per_stage:
            hud_lines.append(f"Stage{s}: {z2}/{z1} = {R:.2f}")

    if errors:
        hud_lines.append("---- ISSUES ----")
        for e in errors[:8]:
            if isinstance(e, dict):
                hud_lines.append(str(e.get("message", "")))
    else:
        hud_lines.append("Issues: None")

    draw_hud_lines(label_img, hud_lines)

    return {
        "det_img": det_img,
        "label_img": label_img,
    }


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
    t = str(sp["cls"]).lower().replace("_", " ")
    return ("long" in t) and ("spacer" in t)


def spacer_is_short(sp: Dict[str, Any]) -> bool:
    t = str(sp["cls"]).lower().replace("_", " ")
    return ("short" in t) and ("spacer" in t)


def expected_contact_boxes_from_gear_count(gear_count: int) -> Tuple[Optional[int], bool]:
    if gear_count <= 1:
        return 0, True
    if (gear_count - 1) % 2 != 0:
        return None, False
    return (gear_count - 1) // 2, True


# =========================
# Count helpers
# =========================
def get_gear_counts(gears: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {
        "driving_gear": 0,
        "smallgear": 0,
        "biggear": 0,
    }

    for g in gears:
        cls = str(g.get("cls", ""))
        if cls in DRIVING_GEAR_CLASS_NAMES:
            counts["driving_gear"] += 1
        elif cls == GEAR_SMALL_NAME:
            counts["smallgear"] += 1
        elif cls == GEAR_BIG_NAME:
            counts["biggear"] += 1

    return counts


def get_shaft_counts(shafts: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {
        "shaft_long": 0,
        "shaft_short": 0,
    }

    for s in shafts:
        cls = str(s.get("cls", ""))
        if cls == "shaft_long":
            counts["shaft_long"] += 1
        elif cls == "shaft_short":
            counts["shaft_short"] += 1

    return counts


def get_spacer_counts(spacers: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {
        "spacer_long": 0,
        "spacer_short": 0,
    }

    for sp in spacers:
        if spacer_is_long(sp):
            counts["spacer_long"] += 1
        elif spacer_is_short(sp):
            counts["spacer_short"] += 1

    return counts


# =========================
# Precheck helpers
# =========================
def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _score_band(value: float, usable_min: float, ideal_min: float, ideal_max: float, usable_max: float) -> float:
    if ideal_min <= value <= ideal_max:
        return 1.0
    if usable_min <= value < ideal_min:
        return _clip01((value - usable_min) / (ideal_min - usable_min))
    if ideal_max < value <= usable_max:
        return _clip01((usable_max - value) / (usable_max - ideal_max))
    return 0.0


def _score_min(value: float, usable_min: float, ideal_min: float) -> float:
    if value >= ideal_min:
        return 1.0
    if value <= usable_min:
        return 0.0
    return _clip01((value - usable_min) / (ideal_min - usable_min))


def _score_max(value: float, ideal_max: float, usable_max: float) -> float:
    if value <= ideal_max:
        return 1.0
    if value >= usable_max:
        return 0.0
    return _clip01((usable_max - value) / (usable_max - ideal_max))


def _score100(component: float) -> int:
    return int(round(100.0 * _clip01(component)))


def _quality_advice(
    *,
    brightness_mean: float,
    brightness_component: float,
    contrast_component: float,
    sharpness_component: float,
    noise_component: float,
) -> List[str]:
    advice: List[str] = []

    brightness_score = _score100(brightness_component)
    contrast_score = _score100(contrast_component)
    sharpness_score = _score100(sharpness_component)
    noise_score = _score100(noise_component)

    is_dark = brightness_mean < QUALITY_BRIGHTNESS_IDEAL_MIN
    brightness_fail = (
        "The photo is too dark for reliable checking. Please retake it in brighter light."
        if is_dark
        else "The photo is overexposed for reliable checking. Please retake it with less glare or strong direct light."
    )
    brightness_warn = (
        "The photo is a little dark. Add more light next time if possible."
        if is_dark
        else "The photo is a little bright. Try reducing glare next time."
    )

    if brightness_score <= QUALITY_MIN_COMPONENT_SCORE:
        advice.append(brightness_fail)
    elif brightness_score <= QUALITY_WARNING_COMPONENT_SCORE:
        advice.append(brightness_warn)

    if contrast_score <= QUALITY_MIN_COMPONENT_SCORE:
        advice.append("The parts do not stand out clearly enough for reliable checking. Please retake the photo with a plain background and fewer shadows.")
    elif contrast_score <= QUALITY_WARNING_COMPONENT_SCORE:
        advice.append("The parts do not stand out very clearly. Use a plain background and avoid shadows next time.")

    if sharpness_score <= QUALITY_MIN_SHARPNESS_SCORE:
        advice.append("The photo is too blurry for reliable checking. Please retake it after holding the camera still and refocusing.")
    elif sharpness_score <= QUALITY_WARNING_SHARPNESS_SCORE:
        advice.append("The photo is a little blurry. For better results, hold the camera still and refocus next time.")

    if noise_score <= QUALITY_MIN_COMPONENT_SCORE:
        advice.append("The photo is too noisy or grainy for reliable checking. Please retake it with better lighting and avoid digital zoom.")
    elif noise_score <= QUALITY_WARNING_COMPONENT_SCORE:
        advice.append("The photo looks a little noisy or grainy. Use better lighting and avoid digital zoom next time.")

    if not advice:
        advice.append("The photo is clear enough for the next check.")

    return advice


def compute_image_quality_metrics(img_bgr: np.ndarray) -> Dict[str, Any]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    brightness_mean = float(np.mean(gray))
    contrast_std = float(np.std(gray))
    sharpness_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_residual = gray.astype(np.float32) - blur.astype(np.float32)
    noise_score = float(np.std(noise_residual))

    brightness_component = _score_band(
        brightness_mean,
        QUALITY_BRIGHTNESS_USABLE_MIN,
        QUALITY_BRIGHTNESS_IDEAL_MIN,
        QUALITY_BRIGHTNESS_IDEAL_MAX,
        QUALITY_BRIGHTNESS_USABLE_MAX,
    )
    contrast_component = _score_min(
        contrast_std,
        QUALITY_CONTRAST_USABLE_MIN,
        QUALITY_CONTRAST_IDEAL_MIN,
    )
    sharpness_component = _score_min(
        sharpness_score,
        QUALITY_SHARPNESS_USABLE_MIN,
        QUALITY_SHARPNESS_IDEAL_MIN,
    )
    noise_component = _score_max(
        noise_score,
        QUALITY_NOISE_IDEAL_MAX,
        QUALITY_NOISE_USABLE_MAX,
    )
    quality_score = int(round(100.0 * (
        0.25 * brightness_component
        + 0.25 * contrast_component
        + 0.25 * sharpness_component
        + 0.25 * noise_component
    )))
    if contrast_component <= 0.0 and sharpness_component <= 0.0:
        quality_score = min(quality_score, QUALITY_ACCEPT_SCORE - 1)
    brightness_score_100 = _score100(brightness_component)
    contrast_score_100 = _score100(contrast_component)
    sharpness_score_100 = _score100(sharpness_component)
    noise_score_100 = _score100(noise_component)
    component_pass = (
        brightness_score_100 > QUALITY_MIN_COMPONENT_SCORE
        and contrast_score_100 > QUALITY_MIN_COMPONENT_SCORE
        and sharpness_score_100 > QUALITY_MIN_SHARPNESS_SCORE
        and noise_score_100 > QUALITY_MIN_COMPONENT_SCORE
    )
    advice = _quality_advice(
        brightness_mean=brightness_mean,
        brightness_component=brightness_component,
        contrast_component=contrast_component,
        sharpness_component=sharpness_component,
        noise_component=noise_component,
    )

    return {
        "brightness_mean": brightness_mean,
        "contrast_std": contrast_std,
        "sharpness_score": sharpness_score,
        "noise_score": noise_score,
        "brightness_score": brightness_score_100,
        "contrast_score": contrast_score_100,
        "sharpness_score_100": sharpness_score_100,
        "noise_score_100": noise_score_100,
        "quality_score": quality_score,
        "quality_score_max": 100,
        "quality_accept_score": QUALITY_ACCEPT_SCORE,
        "quality_min_component_score": QUALITY_MIN_COMPONENT_SCORE,
        "quality_min_sharpness_score": QUALITY_MIN_SHARPNESS_SCORE,
        "quality_warning_component_score": QUALITY_WARNING_COMPONENT_SCORE,
        "quality_warning_sharpness_score": QUALITY_WARNING_SHARPNESS_SCORE,
        "quality_pass": quality_score >= QUALITY_ACCEPT_SCORE and component_pass,
        "quality_advice": advice,
    }


# =========================
# Parts inventory helpers
# =========================
def evaluate_parts_inventory(
    *,
    part_type: str,
    gear_dets: List[Dict[str, Any]],
    shaft_dets: List[Dict[str, Any]],
    aux_dets: List[Dict[str, Any]],
) -> Dict[str, Any]:
    errors: List[Dict[str, str]] = []
    counts: Dict[str, int] = {}
    part_type = str(part_type or "").strip().lower()

    if part_type == "gear":
        counts = {
            "biggear": 0,
            "smallgear": 0,
            "driving_gear": 0,
        }

        for d in gear_dets:
            cls = str(d.get("cls", ""))
            if cls == GEAR_BIG_NAME:
                counts["biggear"] += 1
            elif cls == GEAR_SMALL_NAME:
                counts["smallgear"] += 1
            elif cls in DRIVING_GEAR_CLASS_NAMES:
                counts["driving_gear"] += 1

        if (counts["biggear"] + counts["smallgear"] + counts["driving_gear"]) == 0:
            errors.append({
                "code": "E_NO_TARGET_PARTS",
                "message": "No target gears were detected.",
            })

        return {"counts": counts, "errors": errors}

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

        return {"counts": counts, "errors": errors}

    if part_type == "spacer":
        counts = {
            "spacer_long": 0,
            "spacer_short": 0,
        }

        for d in aux_dets:
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

        return {"counts": counts, "errors": errors}

    return {
        "counts": {},
        "errors": [{
            "code": "E_BAD_PART_TYPE",
            "message": f"Unsupported part_type: {part_type}",
        }],
    }


# =========================
# Gear inventory helpers
# =========================
def evaluate_gear_inventory_step(
    gears: List[Dict[str, Any]],
    mesh_boxes: List[Tuple[float, float, float, float]],
    mismesh_boxes: List[Tuple[float, float, float, float]],
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    errs: List[Dict[str, str]] = []
    counts = get_gear_counts(gears)

    if len(gears) == 0:
        errs.append({
            "code": "E_NO_GEARS",
            "message": "No gears detected.",
        })
        return errs, counts

    if counts["biggear"] != counts["smallgear"]:
        errs.append({
            "code": "E_GEAR_BIG_SMALL_INCONSISTENT",
            "message": (
                f"Gear inventory check failed: biggear={counts['biggear']} and "
                f"smallgear={counts['smallgear']} are not consistent."
            ),
        })

    total_contact = len(mesh_boxes) + len(mismesh_boxes)
    gear_count = len(gears)
    expected_contact, ok = expected_contact_boxes_from_gear_count(gear_count)
    if (not ok) or (expected_contact is None) or (total_contact != expected_contact):
        errs.append({
            "code": "E_GEAR_CONTACT_INCONSISTENT",
            "message": (
                f"Gear inventory consistency check failed: mesh+mismesh={total_contact}, "
                f"expected={expected_contact} for gear_count={gear_count}."
            ),
        })

    if len(mismesh_boxes) > 0:
        errs.append({
            "code": "E_MESH_MISMATCH",
            "message": f"Gear inventory check failed: mismesh detected (count={len(mismesh_boxes)}).",
        })

    return errs, counts


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
    spacer_cnt = len(spacers)
    shaft_cnt = len(shafts)

    if driving_cnt != 1:
        errs.append({
            "code": "E_SINGLE_STAGE_DRIVING_GEAR",
            "message": f"Expected 1 driving gear, but detected {driving_cnt}.",
        })

    if spacer_cnt != 1:
        errs.append({
            "code": "E_SINGLE_STAGE_SPACER_COUNT",
            "message": f"Expected 1 spacer, but detected {spacer_cnt}.",
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
    if not gears or not shafts:
        return None, None

    g11 = next((g for g in gears if g["gid"] == gear11_gid), None)
    if g11 is None:
        return None, None

    c11 = g11["center"]
    ranked = rank_shafts_from_gear11(shafts, c11)

    shaft2 = ranked[0] if len(ranked) >= 1 else None
    shaft3 = ranked[1] if len(ranked) >= 2 else None
    return shaft2, shaft3


def pick_shaft2_and_shaft3_by_distance(
    shafts: List[Dict[str, Any]],
    gear11_gid: int,
    gear_to_si: Dict[int, int],
    gears: List[Dict[str, Any]],
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    if not shafts or not gears:
        return None, None, None

    g11 = next((g for g in gears if g["gid"] == gear11_gid), None)
    if g11 is None:
        return None, None, None

    c11 = g11["center"]
    ranked = rank_shafts_from_gear11(shafts, c11)

    shaft1 = gear_to_si.get(gear11_gid)
    shaft2 = ranked[0] if len(ranked) >= 1 else None
    shaft3 = ranked[1] if len(ranked) >= 2 else None
    return shaft1, shaft2, shaft3


def evaluate_shaft_step_errors(
    gears: List[Dict[str, Any]],
    shafts: List[Dict[str, Any]],
    gear11_gid: int,
) -> List[Dict[str, str]]:
    errs: List[Dict[str, str]] = []

    if not gears:
        errs.append({"code": "E_NO_GEARS", "message": "No gears detected."})
        return errs

    if len(shafts) == 0:
        errs.append({"code": "E_SHAFT_COUNT_MISMATCH", "message": "No shafts detected."})
        return errs

    g11 = next((g for g in gears if g["gid"] == gear11_gid), None)
    if g11 is None:
        errs.append({"code": "E_NO_GEAR11", "message": "Cannot determine gear11."})
        return errs

    c11 = g11["center"]
    ranked = rank_shafts_from_gear11(shafts, c11)

    if len(ranked) >= 1:
        shaft2_idx = ranked[0]
        if str(shafts[shaft2_idx]["cls"]) == "shaft_long":
            errs.append({
                "code": "E_SHAFT_POSITION_SWAP",
                "message": "The closest shaft to gear11 is classified as shaft_long, but shaft2 is expected to be shaft_short.",
            })

    if len(ranked) >= 2:
        shaft3_idx = ranked[1]
        if str(shafts[shaft3_idx]["cls"]) == "shaft_short":
            errs.append({
                "code": "E_SHAFT_POSITION_SWAP",
                "message": "The second closest shaft to gear11 is classified as shaft_short, but shaft3 is expected to be shaft_long.",
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
            errs.append({
                "code": "E_SPACER_LONG_MISSING",
                "message": "Only the short spacer was detected."
            })
        elif "long" in cls_name:
            errs.append({
                "code": "E_SPACER_SHORT_MISSING",
                "message": "Only the long spacer was detected."
            })
        else:
            errs.append({
                "code": "E_SPACER_COUNT_MISMATCH",
                "message": "Only one spacer was detected."
            })
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
        errs.append({
            "code": "E_SPACER_TYPE_CONFUSION",
            "message": "Two long spacers were detected."
        })
        return errs

    if len(short_spacers) == 2 and len(long_spacers) == 0:
        errs.append({
            "code": "E_SPACER_TYPE_CONFUSION",
            "message": "Two short spacers were detected."
        })
        return errs

    if len(short_spacers) != 1 or len(long_spacers) != 1:
        errs.append({
            "code": "E_SPACER_TYPE_CONFUSION",
            "message": "The spacer types could not be identified reliably."
        })
        return errs

    short_sp = short_spacers[0]
    long_sp = long_spacers[0]

    # Strong direct geometric rule first
    d_short = center_dist(short_sp["center"], c11)
    d_long = center_dist(long_sp["center"], c11)
    tol_px = relative_spacer_distance_tol(shafts, gears)

    if d_short > d_long + tol_px:
        errs.append({
            "code": "E_SPACER_DISTANCE_ORDER",
            "message": f"The short spacer is not closer to gear11 than the long spacer (tol={tol_px:.1f}px)."
        })
        return errs

    shaft2_idx, shaft3_idx = get_expected_shaft_indices_for_step(
        gears=gears,
        shafts=shafts,
        gear11_gid=gear11_gid,
    )

    if shaft2_idx is None or shaft3_idx is None:
        errs.append({
            "code": "E_SPACER_ASSIGNMENT_FAIL",
            "message": "The expected shaft identities could not be determined reliably."
        })
        return errs

    short_sp_si = spacer_to_si.get(short_sp["sid"])
    long_sp_si = spacer_to_si.get(long_sp["sid"])

    if short_sp_si is None or long_sp_si is None:
        errs.append({
            "code": "E_SPACER_ASSIGNMENT_FAIL",
            "message": "The spacer-to-shaft assignment could not be determined reliably."
        })
        return errs

    if short_sp_si != shaft2_idx and long_sp_si != shaft3_idx:
        errs.append({
            "code": "E_SPACER_POSITION_MISMATCH",
            "message": "The spacers appear to be swapped between shaft2 and shaft3."
        })
        return errs

    if short_sp_si != shaft2_idx:
        errs.append({
            "code": "E_SPACER2_TYPE_MISMATCH",
            "message": "The spacer on shaft2 is not the short spacer."
        })
        return errs

    if long_sp_si != shaft3_idx:
        errs.append({
            "code": "E_SPACER3_TYPE_MISMATCH",
            "message": "The spacer on shaft3 is not the long spacer."
        })
        return errs

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

    shaft1, shaft2, shaft3 = pick_shaft2_and_shaft3_by_distance(
        shafts, gear11_gid, gear_to_si, gears
    )

    if shaft2 is not None and 0 <= shaft2 < len(shafts):
        if str(shafts[shaft2]["cls"]) == "shaft_long":
            errs.append({
                "code": "E_SHAFT2_CLASS_MISMATCH",
                "message": "Assembly issue: shaft2 (closest detected shaft to gear11) is classified as 'shaft_long' (expected 'shaft_short')."
            })

    if shaft3 is not None and 0 <= shaft3 < len(shafts):
        if str(shafts[shaft3]["cls"]) == "shaft_short":
            errs.append({
                "code": "E_SHAFT3_CLASS_MISMATCH",
                "message": "Assembly issue: shaft3 (second closest detected shaft to gear11) is classified as 'shaft_short' (expected 'shaft_long')."
            })

    spacer2 = pick_spacer_on_shaft_as(spacers, spacer_to_si, shaft2, c11) if shaft2 is not None else None
    spacer3 = pick_spacer_on_shaft_as(spacers, spacer_to_si, shaft3, c11) if shaft3 is not None else None

    if shaft2 is not None and spacer2 is None:
        errs.append({
            "code": "E_SPACER2_MISSING",
            "message": "Assembly issue: no spacer found on shaft2."
        })

    if shaft3 is not None and spacer3 is None:
        errs.append({
            "code": "E_SPACER3_MISSING",
            "message": "Assembly issue: no spacer found on shaft3."
        })

    # Strict spacer type-position checks
    if spacer2 is not None and not spacer_is_short(spacer2):
        errs.append({
            "code": "E_SPACER2_TYPE_MISMATCH",
            "message": "Assembly issue: the spacer on shaft2 is not the short spacer."
        })

    if spacer3 is not None and not spacer_is_long(spacer3):
        errs.append({
            "code": "E_SPACER3_TYPE_MISMATCH",
            "message": "Assembly issue: the spacer on shaft3 is not the long spacer."
        })

    if (
        spacer2 is not None
        and spacer3 is not None
        and (not spacer_is_short(spacer2))
        and (not spacer_is_long(spacer3))
    ):
        errs.append({
            "code": "E_SPACER_POSITION_MISMATCH",
            "message": "Assembly issue: the short and long spacers appear to be swapped between shaft2 and shaft3."
        })

    if spacer2 is not None and spacer3 is not None:
        d2 = center_dist(spacer2["center"], c11)
        d3 = center_dist(spacer3["center"], c11)
        tol_px = relative_spacer_distance_tol(shafts, gears)

        if d2 > d3 + tol_px:
            errs.append({
                "code": "E_SPACER_DISTANCE_ORDER",
                "message": f"Consistency check: the spacer on shaft2 is not closer to gear11 than the spacer on shaft3 (tol={tol_px:.1f}px)."
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
def _run_detection_xyxy(
    img_bgr: np.ndarray,
    model: Any,
    conf_th: float,
) -> List[Dict[str, Any]]:
    res = model(img_bgr, verbose=False)[0]
    names = res.names
    dets: List[Dict[str, Any]] = []

    if res.boxes is not None and res.boxes.data is not None:
        for x1, y1, x2, y2, conf, cls_id in res.boxes.data.tolist():
            conf = float(conf)
            if conf < conf_th:
                continue
            cls_id = int(cls_id)
            cls = names.get(cls_id, str(cls_id))
            dets.append({
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "score": conf,
                "cls": cls,
            })
    return dets


def run_detection_model_a(img_bgr: np.ndarray, model_a: Any) -> List[Dict[str, Any]]:
    dets = _run_detection_xyxy(img_bgr, model_a, CONF_MODEL_A)
    filtered: List[Dict[str, Any]] = []
    for d in dets:
        cls = str(d["cls"])
        if cls in DRIVING_GEAR_CLASS_NAMES or cls in (GEAR_BIG_NAME, GEAR_SMALL_NAME):
            filtered.append(d)
    return filtered


def run_detection_model_c(img_bgr: np.ndarray, model_c: Any) -> List[Dict[str, Any]]:
    dets = _run_detection_xyxy(img_bgr, model_c, CONF_MODEL_C)
    filtered: List[Dict[str, Any]] = []
    for d in dets:
        cls = str(d["cls"])
        if cls == MESH_CLASS_NAME or cls == MISMESH_CLASS_NAME or cls in SPACER_CLASSES:
            filtered.append(d)
    return filtered


def run_detection_shaft_obb(img_bgr: np.ndarray, model_b: Any) -> List[Dict[str, Any]]:
    res = model_b(img_bgr, verbose=False)[0]
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
                if conf < CONF_MODEL_B:
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
                if conf < CONF_MODEL_B:
                    continue
                cls = names.get(cls_id, str(cls_id))
                if cls not in TARGET_SHAFT_CLASSES:
                    continue
                rect = ((cx, cy), (w, h), ang * 180.0 / math.pi)
                pts = cv2.boxPoints(rect).astype(np.float32)
            else:
                continue

            c = poly_center(pts)
            major_len, minor_len, axis_dir = shaft_length_width_from_poly(pts)

            dets.append({
                "cls": cls,
                "score": conf,
                "poly4": pts,
                "poly4_scaled": scale_poly_about_center(pts, scale=OBB_ASSIGN_SCALE),
                "center": c,
                "axis_dir": axis_dir,
                "major_len": float(major_len),
                "minor_len": float(minor_len),
            })

    return dets


# =========================
# Build objects
# =========================
def build_objects(
    gear_dets: List[Dict[str, Any]],
    aux_dets: List[Dict[str, Any]],
) -> Tuple[
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
        if (cls in DRIVING_GEAR_CLASS_NAMES) or (cls in (GEAR_BIG_NAME, GEAR_SMALL_NAME)):
            b = d["bbox"]
            gears.append({
                "gid": len(gears),
                "cls": cls,
                "score": d["score"],
                "bbox": b,
                "center": bbox_center(b),
                "r": est_radius_from_bbox(b),
            })

    for d in aux_dets:
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

    # Assign gears
    for g in gears:
        c = g["center"]

        strict_candidates = []
        for i, s in enumerate(shafts):
            if point_in_poly(c, s["poly4_scaled"]):
                strict_candidates.append(i)

        if strict_candidates:
            best = max(strict_candidates, key=lambda i: shafts[i]["score"])
            gear_to_si[g["gid"]] = best
            si_to_gids[best].append(g["gid"])
            continue

        scored: List[Tuple[float, float, int]] = []
        for i, s in enumerate(shafts):
            axis_dist = dist_point_to_shaft_axis(c, s)
            center_d = center_dist(c, s["center"])
            scored.append((axis_dist, center_d, i))

        if scored:
            scored.sort(key=lambda x: (x[0], x[1]))
            best_axis_dist, best_center_d, best_i = scored[0]
            shaft = shafts[best_i]

            axis_thresh = max(
                12.0,
                1.2 * max(float(shaft.get("minor_len", 0.0)), 1.0),
            )
            center_thresh = max(
                40.0,
                1.2 * max(float(shaft.get("major_len", 0.0)), 1.0),
            )

            if best_axis_dist <= axis_thresh and best_center_d <= center_thresh:
                gear_to_si[g["gid"]] = best_i
                si_to_gids[best_i].append(g["gid"])

    # Assign spacers
    for sp in spacers:
        c = sp["center"]

        strict_candidates = []
        for i, s in enumerate(shafts):
            if point_in_poly(c, s["poly4_scaled"]):
                strict_candidates.append(i)

        if strict_candidates:
            best = max(strict_candidates, key=lambda i: shafts[i]["score"])
            spacer_to_si[sp["sid"]] = best
            si_to_spacers[best].append(sp["sid"])
            continue

        scored: List[Tuple[float, float, int]] = []
        for i, s in enumerate(shafts):
            axis_dist = dist_point_to_shaft_axis(c, s)
            center_d = center_dist(c, s["center"])
            scored.append((axis_dist, center_d, i))

        if scored:
            scored.sort(key=lambda x: (x[0], x[1]))
            best_axis_dist, best_center_d, best_i = scored[0]
            shaft = shafts[best_i]

            axis_thresh = max(
                8.0,
                SPACER_ASSIGN_AXIS_DIST_RATIO * max(float(shaft.get("minor_len", 0.0)), 1.0),
            )
            center_thresh = max(
                15.0,
                SPACER_ASSIGN_CENTER_DIST_RATIO * max(float(shaft.get("major_len", 0.0)), 1.0),
            )

            if best_axis_dist <= axis_thresh and best_center_d <= center_thresh:
                spacer_to_si[sp["sid"]] = best_i
                si_to_spacers[best_i].append(sp["sid"])

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
    si = gear_to_si.get(gid_base, None)
    g0 = gears_by_gid(gears, gid_base)

    # Strict same-shaft search
    if si is not None:
        candidates = []
        for g in gears:
            gid = g["gid"]
            if gid == gid_base or gid in used_gids:
                continue
            if gear_to_si.get(gid, None) != si:
                continue

            d = center_dist(g["center"], g0["center"])
            candidates.append((d, gid))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]

    # Fallback by proximity
    fallback = []
    for g in gears:
        gid = g["gid"]
        if gid == gid_base or gid in used_gids:
            continue

        d = center_dist(g["center"], g0["center"])
        r0 = float(g0.get("r", 20.0))
        rg = float(g.get("r", 20.0))
        dist_th = 2.4 * max(r0, rg, 20.0)

        if d <= dist_th:
            fallback.append((d, gid))

    if fallback:
        fallback.sort(key=lambda x: x[0])
        return fallback[0][1]

    return None


def rank_shafts_from_gear11(
    shafts: List[Dict[str, Any]],
    gear11_center: Tuple[float, float],
) -> List[int]:
    ranked: List[Tuple[float, int]] = []
    for i, s in enumerate(shafts):
        d = center_dist(s["center"], gear11_center)
        ranked.append((d, i))
    ranked.sort(key=lambda x: x[0])
    return [i for _, i in ranked]


def pick_nearest_spacer_on_shaft_to_gear11(
    spacers: List[Dict[str, Any]],
    spacer_to_si: Dict[int, int],
    shaft_idx: int,
    gear11_center: Tuple[float, float],
) -> Optional[Dict[str, Any]]:
    candidates: List[Tuple[float, Dict[str, Any]]] = []
    for sp in spacers:
        if spacer_to_si.get(sp["sid"]) != shaft_idx:
            continue
        d = center_dist(sp["center"], gear11_center)
        candidates.append((d, sp))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def pick_nearest_gear_on_shaft_to_point(
    gears: List[Dict[str, Any]],
    gear_to_si: Dict[int, int],
    shaft_idx: int,
    ref_pt: Tuple[float, float],
    forbid_gid: Optional[int] = None,
) -> Optional[int]:
    candidates: List[Tuple[float, int]] = []
    for g in gears:
        gid = g["gid"]
        if forbid_gid is not None and gid == forbid_gid:
            continue
        if gear_to_si.get(gid) != shaft_idx:
            continue
        d = center_dist(g["center"], ref_pt)
        candidates.append((d, gid))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def find_best_contact_box_for_pair(
    gA: Dict[str, Any],
    gB: Dict[str, Any],
    contact_boxes: List[Tuple[float, float, float, float]],
    used_contact_idx: set,
) -> Tuple[float, Optional[int], Optional[Tuple[float, float, float, float]]]:
    best = None
    for cidx, box in enumerate(contact_boxes):
        if ONE_TO_ONE_CONTACT_BOX and (cidx in used_contact_idx):
            continue
        ret = score_pair_by_contact_box(gA, gB, box)
        if ret is None:
            continue
        sc, hit, dms, gap = ret
        if (best is None) or (sc > best[0]):
            best = (sc, hit, cidx, box)

    if best is None:
        return 0.0, None, None
    return best[1], best[2], best[3]


def stage_role_naming_chain(
    gears: List[Dict[str, Any]],
    spacers: List[Dict[str, Any]],
    shafts: List[Dict[str, Any]],
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

    # Stage 1 anchor
    labels[gear11_gid] = "gear11"
    stage_of[gear11_gid] = 1
    used_gids.add(gear11_gid)

    g11 = gears_by_gid(gears, gear11_gid)
    c11 = g11["center"]

    # New deterministic rule:
    # 1) gear11 is fixed as the driving gear
    # 2) sort all detected shafts by distance to gear11 center
    # 3) the nearest shaft is shaft2
    # 4) choose the spacer on shaft2 that is nearest to gear11 center
    # 5) use that spacer center as the local reference point on shaft2
    # 6) choose the nearest gear on shaft2 to that local reference point as gear12
    ranked_shafts = rank_shafts_from_gear11(shafts, c11)
    shaft2 = ranked_shafts[0] if len(ranked_shafts) >= 1 else None

    mate12 = None
    hit12 = 0.0
    box12 = None
    cidx12 = None

    if shaft2 is not None:
        spacer2 = pick_nearest_spacer_on_shaft_to_gear11(
            spacers=spacers,
            spacer_to_si=spacer_to_si,
            shaft_idx=shaft2,
            gear11_center=c11,
        )

        if spacer2 is not None:
            mate12 = pick_nearest_gear_on_shaft_to_point(
                gears=gears,
                gear_to_si=gear_to_si,
                shaft_idx=shaft2,
                ref_pt=spacer2["center"],
                forbid_gid=gear11_gid,
            )

            if mate12 is not None:
                hit12, cidx12, box12 = find_best_contact_box_for_pair(
                    gA=g11,
                    gB=gears_by_gid(gears, mate12),
                    contact_boxes=contact_boxes,
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

    # Remaining stages keep the original chain logic
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


def _finalize_out(
    *,
    out: Dict[str, Any],
    timing: Dict[str, float],
    t0_total: float,
    return_images: bool,
    img_bgr: np.ndarray,
    gears: List[Dict[str, Any]],
    spacers: List[Dict[str, Any]],
    shaft_obbs: List[Dict[str, Any]],
    mesh_boxes: List[Tuple[float, float, float, float]],
    mismesh_boxes: List[Tuple[float, float, float, float]],
    gear_names: Optional[Dict[int, str]] = None,
    gear_stage: Optional[Dict[int, int]] = None,
    chain_pairs: Optional[List[Tuple[int, int, float, Optional[Tuple[float, float, float, float]]]]] = None,
) -> Dict[str, Any]:
    timing["t_total_pipeline_s"] = float(time.perf_counter() - t0_total)
    out["timing"] = timing

    if return_images:
        out["images"] = build_output_images(
            img_bgr=img_bgr,
            gears=gears,
            spacers=spacers,
            shaft_obbs=shaft_obbs,
            mesh_boxes=mesh_boxes,
            mismesh_boxes=mismesh_boxes,
            gear_names=gear_names or out.get("gear_names", {}) or {},
            gear_stage=gear_stage or out.get("gear_stage", {}) or {},
            chain_pairs=chain_pairs or out.get("chain_pairs", []) or [],
            ratio=out.get("ratio", {}),
            errors=out.get("errors", []),
        )

    return out


# =========================
# Public API
# =========================
def run_yolo_pipeline(
    img_bgr: np.ndarray,
    model_a_rel: str = DEFAULT_MODEL_A_REL,
    model_b_rel: str = DEFAULT_MODEL_B_REL,
    model_c_rel: str = DEFAULT_MODEL_C_REL,
    return_images: bool = False,
    *,
    task: str = TASK_FULL,
    part_type: Optional[str] = None,
    expected_gears: Optional[int] = None,
    gear_model_rel: Optional[str] = None,
    shaft_model_rel: Optional[str] = None,
) -> Dict[str, Any]:
    if img_bgr is None or not hasattr(img_bgr, "shape"):
        raise ValueError("img_bgr must be a valid OpenCV image (BGR).")

    task = str(task or TASK_FULL).strip().lower()
    if task not in _VALID_TASKS:
        task = TASK_FULL

    part_type = str(part_type or "").strip().lower()

    if gear_model_rel:
        model_a_rel = gear_model_rel
    if shaft_model_rel:
        model_b_rel = shaft_model_rel

    global _COLD_START_FLAG
    is_cold_start = _COLD_START_FLAG
    _COLD_START_FLAG = False

    t0_total = time.perf_counter()
    timing: Dict[str, float] = {}

    model_a, model_b, model_c = get_models(
        model_a_rel=model_a_rel,
        model_b_rel=model_b_rel,
        model_c_rel=model_c_rel,
        timing=timing,
    )

    t_a0 = time.perf_counter()
    gear_dets = run_detection_model_a(img_bgr, model_a)
    timing["t_infer_model_a_s"] = float(time.perf_counter() - t_a0)

    t_b0 = time.perf_counter()
    shaft_obbs = run_detection_shaft_obb(img_bgr, model_b)
    timing["t_infer_model_b_s"] = float(time.perf_counter() - t_b0)

    t_c0 = time.perf_counter()
    aux_dets = run_detection_model_c(img_bgr, model_c)
    timing["t_infer_model_c_s"] = float(time.perf_counter() - t_c0)

    gears, spacers, mesh_boxes, mismesh_boxes = build_objects(gear_dets, aux_dets)
    contact_boxes = list(mesh_boxes) + list(mismesh_boxes)

    gear11_gid: Optional[int] = None
    gear_to_si: Dict[int, int] = {}
    spacer_to_si: Dict[int, int] = {}

    if gears:
        driving = pick_driving_gear(gears)
        gear11_gid = int(driving["gid"])

    if gears and shaft_obbs:
        gear_to_si, _si_to_gids, spacer_to_si, _si_to_spacers = assign_items_to_shafts(
            gears, spacers, shaft_obbs
        )

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
            aux_dets=aux_dets,
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
        return _finalize_out(
            out=out,
            timing=timing,
            t0_total=t0_total,
            return_images=return_images,
            img_bgr=img_bgr,
            gears=gears,
            spacers=spacers,
            shaft_obbs=shaft_obbs,
            mesh_boxes=mesh_boxes,
            mismesh_boxes=mismesh_boxes,
        )

    # --- task: precheck ---
    if task == TASK_PRECHECK:
        quality = compute_image_quality_metrics(img_bgr)
        counts = get_gear_counts(gears)
        errors: List[Dict[str, Any]] = []
        if not bool(quality.get("quality_pass", True)):
            errors.append({
                "code": "E_PHOTO_QUALITY_LOW",
                "message": (
                    "The photo quality is too low for reliable checking. "
                    "Please retake it with clearer focus and better lighting."
                ),
            })

        out = {
            "summary": {
                "gears": len(gears),
                "spacers": len(spacers),
                "shafts": len(shaft_obbs),
                "mesh": len(mesh_boxes),
                "mismesh": len(mismesh_boxes),
                "stages": 0,
            },
            "counts": counts,
            "quality": quality,
            "errors": errors,
            "ratio": {"num_stages": 0, "R_total": None, "out_rpm": None, "per_stage": []},
            "cold_start": bool(is_cold_start),
            "task_result": _task_result(
                TASK_PRECHECK,
                errors,
                focus=["photo_quality"],
                next_task=TASK_SINGLE_STAGE,
            ),
            "timing": {},
        }
        return _finalize_out(
            out=out,
            timing=timing,
            t0_total=t0_total,
            return_images=return_images,
            img_bgr=img_bgr,
            gears=gears,
            spacers=spacers,
            shaft_obbs=shaft_obbs,
            mesh_boxes=mesh_boxes,
            mismesh_boxes=mismesh_boxes,
        )

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
                "counts": get_gear_counts(gears),
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
            return _finalize_out(
                out=out,
                timing=timing,
                t0_total=t0_total,
                return_images=return_images,
                img_bgr=img_bgr,
                gears=gears,
                spacers=spacers,
                shaft_obbs=shaft_obbs,
                mesh_boxes=mesh_boxes,
                mismesh_boxes=mismesh_boxes,
            )

        if gear11_gid is None:
            gear11_gid = pick_driving_gear(gears)["gid"]

        if gears and shaft_obbs and not gear_to_si:
            gear_to_si, _si_to_gids, spacer_to_si, _si_to_spacers = assign_items_to_shafts(
                gears, spacers, shaft_obbs
            )

        gear_names, gear_stage, chain_pairs = stage_role_naming_chain(
            gears=gears,
            spacers=spacers,
            shafts=shaft_obbs,
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
        spacer_counts = get_spacer_counts(spacers)
        shaft_counts = get_shaft_counts(shaft_obbs)
        single_stage_hints: List[str] = []
        if (
            spacer_counts.get("spacer_long", 0) == 1
            and spacer_counts.get("spacer_short", 0) == 0
        ):
            single_stage_hints.append(
                "The setup works, but think about whether a short spacer or a long spacer "
                "is more appropriate here."
            )
        if (
            shaft_counts.get("shaft_short", 0) == 1
            and shaft_counts.get("shaft_long", 0) == 0
        ):
            single_stage_hints.append(
                "The setup works, but think about whether a long shaft or a short shaft "
                "is the better choice here."
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
            "counts": get_gear_counts(gears),
            "shaft_counts": shaft_counts,
            "spacer_counts": spacer_counts,
            "gear_names": gear_names,
            "gear_stage": gear_stage,
            "chain_pairs": chain_pairs,
            "ratio": ratio,
            "errors": errors,
            "single_stage_hints": single_stage_hints,
            "cold_start": bool(is_cold_start),
            "task_result": _task_result(
                TASK_SINGLE_STAGE,
                errors,
                focus=["single_stage"],
                next_task=TASK_SHAFT,
            ),
            "timing": {},
        }
        return _finalize_out(
            out=out,
            timing=timing,
            t0_total=t0_total,
            return_images=return_images,
            img_bgr=img_bgr,
            gears=gears,
            spacers=spacers,
            shaft_obbs=shaft_obbs,
            mesh_boxes=mesh_boxes,
            mismesh_boxes=mismesh_boxes,
            gear_names=gear_names,
            gear_stage=gear_stage,
            chain_pairs=chain_pairs,
        )

    # --- task: gear_inventory ---
    if task == TASK_GEAR_INV:
        errors, counts = evaluate_gear_inventory_step(
            gears=gears,
            mesh_boxes=mesh_boxes,
            mismesh_boxes=mismesh_boxes,
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
            "counts": counts,
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
        return _finalize_out(
            out=out,
            timing=timing,
            t0_total=t0_total,
            return_images=return_images,
            img_bgr=img_bgr,
            gears=gears,
            spacers=spacers,
            shaft_obbs=shaft_obbs,
            mesh_boxes=mesh_boxes,
            mismesh_boxes=mismesh_boxes,
        )

    # --- task: shaft ---
    if task == TASK_SHAFT:
        shaft_counts = get_shaft_counts(shaft_obbs)

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
            "counts": shaft_counts,
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
        return _finalize_out(
            out=out,
            timing=timing,
            t0_total=t0_total,
            return_images=return_images,
            img_bgr=img_bgr,
            gears=gears,
            spacers=spacers,
            shaft_obbs=shaft_obbs,
            mesh_boxes=mesh_boxes,
            mismesh_boxes=mismesh_boxes,
        )

    # --- task: spacer ---
    if task == TASK_SPACER:
        spacer_counts = get_spacer_counts(spacers)

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
            "counts": spacer_counts,
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
        return _finalize_out(
            out=out,
            timing=timing,
            t0_total=t0_total,
            return_images=return_images,
            img_bgr=img_bgr,
            gears=gears,
            spacers=spacers,
            shaft_obbs=shaft_obbs,
            mesh_boxes=mesh_boxes,
            mismesh_boxes=mismesh_boxes,
        )

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
            "counts": get_gear_counts(gears),
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
        return _finalize_out(
            out=out,
            timing=timing,
            t0_total=t0_total,
            return_images=return_images,
            img_bgr=img_bgr,
            gears=gears,
            spacers=spacers,
            shaft_obbs=shaft_obbs,
            mesh_boxes=mesh_boxes,
            mismesh_boxes=mismesh_boxes,
        )

    if gear11_gid is None:
        gear11_gid = pick_driving_gear(gears)["gid"]

    if gears and shaft_obbs and not gear_to_si:
        gear_to_si, _si_to_gids, spacer_to_si, _si_to_spacers = assign_items_to_shafts(
            gears, spacers, shaft_obbs
        )

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
        shafts=shaft_obbs,
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
        "counts": get_gear_counts(gears),
        "detections": {
            "gear_dets": gear_dets,
            "aux_dets": aux_dets,
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

    return _finalize_out(
        out=out,
        timing=timing,
        t0_total=t0_total,
        return_images=return_images,
        img_bgr=img_bgr,
        gears=gears,
        spacers=spacers,
        shaft_obbs=shaft_obbs,
        mesh_boxes=mesh_boxes,
        mismesh_boxes=mismesh_boxes,
        gear_names=gear_names,
        gear_stage=gear_stage,
        chain_pairs=chain_pairs,
    )
