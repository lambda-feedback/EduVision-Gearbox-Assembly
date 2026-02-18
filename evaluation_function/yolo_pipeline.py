# -*- coding: utf-8 -*-
"""
YOLO inference pipeline (gear model + shaft OBB model)
- Relative model paths (gear_model.pt, shaft_model.pt in the same folder)
- Cached model loading (per process)
- No folder scanning / no CSV writing (Lambda-friendly)
- Keeps your core logic: object building, shaft assignment, stage chain naming,
  assembly error checks, and gear-ratio computation.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
#from ultralytics import YOLO
YOLO = None


# =========================
# CONFIG (edit if needed)
# =========================
CONF_GEAR: float = 0.50
CONF_SHAFT: float = 0.50

# gear ratio constants
MOTOR_RPM: float = 8000.0
TEETH_BIG: int = 48
TEETH_SMALL: int = 12

# class names
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

# assignment robustness
OBB_ASSIGN_SCALE: float = 1.10

# contact-line sampling
LINE_SAMPLES: int = 25
LINE_HIT_RATIO_TH: float = 0.25

# matching / naming config
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

# assembly checks
ENABLE_ERROR_CHECKS: bool = True
SPACER_DIST_TOL_PX: float = 5.0

# drawing
BOX_THICK: int = 2
CENTER_R: int = 4

LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE: float = 0.85
LABEL_THICK: int = 2
LABEL_PAD: int = 3
LEADER_THICK: int = 2

HUD_SCALE: float = 1.3
HUD_THICK: int = 2
HUD_LINE_GAP: int = 32
HUD_X: int = 20
HUD_Y0: int = 40
HUD_COLOR = (0, 255, 255)


# =========================
# Paths (relative)
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
def _load_yolo_model(abs_path: str) -> YOLO:
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Model not found: {abs_path}")
    return YOLO(abs_path)


def get_models(
    gear_model_rel: str = DEFAULT_GEAR_MODEL_REL,
    shaft_model_rel: str = DEFAULT_SHAFT_MODEL_REL,
) -> Tuple[YOLO, YOLO]:
    gear_model = _load_yolo_model(_abs_model_path(gear_model_rel))
    shaft_model = _load_yolo_model(_abs_model_path(shaft_model_rel))
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


def draw_bbox(img: np.ndarray, b: Tuple[float, float, float, float], color: Tuple[int, int, int], thick: int = 2) -> None:
    x1, y1, x2, y2 = map(int, b)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)


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
# HUD drawer
# =========================
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


# =========================
# anti-overlap label placer
# =========================
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
        (12, -12), (12, 18), (-w - 12, -12), (-w - 12, 18),
        (12, -h - 18), (-w - 12, -h - 18),
        (12, h + 24), (-w - 12, h + 24),
        (18, 0), (-w - 18, 0),
        (0, -18), (0, 30),
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
                cv2.line(img, (ax, ay), (tx, ty - h // 2), (255, 255, 255), LEADER_THICK, cv2.LINE_AA)
            return (tx, ty), rect

        if collisions < best_collisions:
            best_collisions = collisions
            best = (tx, ty, rect)

    tx, ty, rect = best
    put_text_outline(img, text, (tx, ty), scale=scale, thick=thick, color=color)
    occupied_rects.append(rect)
    if leader:
        cv2.line(img, (ax, ay), (tx, ty - h // 2), (255, 255, 255), LEADER_THICK, cv2.LINE_AA)
    return (tx, ty), rect


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
        if gear_names.get(gid) == target_name:
            draw_bbox(img, g["bbox"], color, thick)
            cx, cy = int(g["center"][0]), int(g["center"][1])
            cv2.circle(img, (cx, cy), CENTER_R + 2, color, -1)
            put_text_outline(img, f"{target_name} ({g['cls']})", (cx + 12, cy - 12), 0.85, 2, color)
            break


# =========================
# Spacer type helpers
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


def pick_shaft2_and_shaft3_by_distance(
    shafts: List[Dict[str, Any]],
    gear11_gid: int,
    gear_to_si: Dict[int, int],
    gears: List[Dict[str, Any]],
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
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

    # Rule 1: mismesh present
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

    # Rule 2: shaft2 should be shaft_short
    if shaft2 is not None and 0 <= shaft2 < len(shafts):
        if str(shafts[shaft2]["cls"]) == "shaft_long":
            errs.append({
                "code": "E_SHAFT2_CLASS_MISMATCH",
                "message": "Assembly issue: shaft2 (closest shaft to gear11) is classified as 'shaft_long' (expected 'shaft_short')."
            })

    # spacer2/spacer3 by containment assignment
    spacer2 = pick_spacer_on_shaft_as(spacers, spacer_to_si, shaft2, c11) if shaft2 is not None else None
    spacer3 = pick_spacer_on_shaft_as(spacers, spacer_to_si, shaft3, c11) if shaft3 is not None else None

    # Rule 3A: spacer2 exists and must be short
    if shaft2 is not None:
        if spacer2 is None:
            errs.append({
                "code": "E_SPACER2_MISSING",
                "message": "Assembly issue: no spacer found on shaft2 (spacer2 missing or not assigned to shaft2)."
            })
        else:
            if spacer_is_long(spacer2):
                errs.append({
                    "code": "E_SPACER2_TYPE_MISMATCH",
                    "message": "Assembly issue: spacer2 (spacer on shaft2) is classified as 'long spacer' (expected 'short spacer')."
                })

    # Rule 3B: spacer3 exists and should be long
    if shaft3 is not None:
        if spacer3 is None:
            errs.append({
                "code": "E_SPACER3_MISSING",
                "message": "Assembly issue: no spacer found on shaft3 (spacer3 missing or not assigned to shaft3)."
            })
        else:
            if spacer_is_short(spacer3):
                errs.append({
                    "code": "E_SPACER3_TYPE_MISMATCH",
                    "message": "Assembly issue: spacer3 (spacer on shaft3) is classified as 'short spacer' (expected 'long spacer')."
                })

    # Extra sanity: spacer2 closer to gear11 than spacer3
    if spacer2 is not None and spacer3 is not None:
        d2 = center_dist(spacer2["center"], c11)
        d3 = center_dist(spacer3["center"], c11)
        if d2 > d3 + float(SPACER_DIST_TOL_PX):
            errs.append({
                "code": "E_SPACER_DISTANCE_ORDER",
                "message": f"Consistency check: spacer2 is not closer to gear11 than spacer3 (tol={SPACER_DIST_TOL_PX}px)."
            })

    # Rule 4: contact boxes count
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
def run_detection_gear(img_bgr: np.ndarray, gear_model: YOLO) -> List[Dict[str, Any]]:
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


def run_detection_shaft_obb(img_bgr: np.ndarray, shaft_model: YOLO) -> List[Dict[str, Any]]:
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

            # case A: poly8 + conf + cls (len >= 10)
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

            # case B: (cx,cy,w,h,ang,conf,cls) (len >= 7)
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
    # fallback: smallest radius
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
# Contact-box scoring
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

        if REQUIRE_BIG_SMALL_PAIR:
            if is_big(gA, median_r) == is_big(gB, median_r):
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


# =========================
# Force stage1 mate by short spacer (for chain)
# =========================
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
# Public API
# =========================
def run_yolo_pipeline(
    img_bgr: np.ndarray,
    gear_model_rel: str = DEFAULT_GEAR_MODEL_REL,
    shaft_model_rel: str = DEFAULT_SHAFT_MODEL_REL,
    return_images: bool = False,
) -> Dict[str, Any]:
    """
    Main pipeline entry.

    Args:
        img_bgr: OpenCV BGR image (np.ndarray)
        gear_model_rel: relative model filename under evaluation_function/
        shaft_model_rel: relative model filename under evaluation_function/
        return_images: if True, returns annotated images (det_img, label_img)

    Returns:
        dict with:
            - summary
            - detections (gear model dets + shaft OBB dets)
            - gears/spacers/mesh/mismesh/shafts parsed objects
            - gear_names, gear_stage, chain_pairs
            - ratio info
            - errors (list of {code,message})
            - images (optional)
    """
    if img_bgr is None or not hasattr(img_bgr, "shape"):
        raise ValueError("img_bgr must be a valid OpenCV image (BGR).")

    gear_model, shaft_model = get_models(gear_model_rel, shaft_model_rel)

    gear_dets = run_detection_gear(img_bgr, gear_model)
    shaft_obbs = run_detection_shaft_obb(img_bgr, shaft_model)
    gears, spacers, mesh_boxes, mismesh_boxes = build_objects(gear_dets)
    contact_boxes = list(mesh_boxes) + list(mismesh_boxes)

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
        }
        if return_images:
            out["images"] = {"det_img": img_bgr.copy(), "label_img": img_bgr.copy()}
        return out

    driving = pick_driving_gear(gears)
    gear11_gid = driving["gid"]

    gear_to_si, si_to_gids, spacer_to_si, si_to_spacers = assign_items_to_shafts(gears, spacers, shaft_obbs)

    errors: List[Dict[str, str]] = []
    if ENABLE_ERROR_CHECKS:
        errors = evaluate_assembly_errors(
            gears=gears,
            spacers=spacers,
            shafts=shaft_obbs,
            mesh_boxes=mesh_boxes,
            mismesh_boxes=mismesh_boxes,
            gear11_gid=gear11_gid,
            gear_to_si=gear_to_si,
            spacer_to_si=spacer_to_si,
        )

    gear_names, gear_stage, chain_pairs = stage_role_naming_chain(
        gears=gears,
        spacers=spacers,
        contact_boxes=contact_boxes,
        gear_to_si=gear_to_si,
        spacer_to_si=spacer_to_si,
        gear11_gid=gear11_gid,
    )

    num_stages, R_total, out_rpm, per_stage = compute_ratio_and_rpm_from_stage_labels(
        gears=gears,
        gear_names=gear_names,
        motor_rpm=MOTOR_RPM,
        max_stage=MAX_STAGE,
    )

    out: Dict[str, Any] = {
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
    }

    if return_images:
        det_img = img_bgr.copy()
        label_img = img_bgr.copy()

        # draw shafts (poly)
        for s in shaft_obbs:
            poly = s["poly4"].astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(det_img, [poly], True, (255, 255, 255), BOX_THICK, cv2.LINE_AA)
            put_text_outline(det_img, f"{s['cls']} {s['score']:.2f}", (s["center"][0] + 6, s["center"][1] + 6), 0.55, 2, (255, 255, 255))

        # mesh / mismesh boxes
        for b in mesh_boxes:
            draw_bbox(det_img, b, (255, 255, 0), 2)
            put_text_outline(det_img, "Mesh", (b[0] + 3, b[1] - 6), 0.55, 2, (255, 255, 0))
        for b in mismesh_boxes:
            draw_bbox(det_img, b, (255, 0, 255), 2)
            put_text_outline(det_img, "Mismesh", (b[0] + 3, b[1] - 6), 0.55, 2, (255, 0, 255))

        # spacers
        for sp in spacers:
            draw_bbox(det_img, sp["bbox"], (0, 255, 255), 2)
            put_text_outline(det_img, f"{sp['cls']} {sp['score']:.2f}", (sp["bbox"][0] + 3, sp["bbox"][1] - 6), 0.55, 2, (0, 255, 255))
            cv2.circle(det_img, (int(sp["center"][0]), int(sp["center"][1])), CENTER_R, (0, 255, 255), -1)

        # gears
        for g in gears:
            draw_bbox(det_img, g["bbox"], (0, 255, 0), 2)
            cx, cy = int(g["center"][0]), int(g["center"][1])
            cv2.circle(det_img, (cx, cy), CENTER_R, (0, 0, 255), -1)
            put_text_outline(det_img, f"{g['cls']} {g['score']:.2f}", (g["bbox"][0] + 3, g["bbox"][1] - 6), 0.55, 2, (0, 255, 0))

        # labels with anti-overlap
        occupied: List[Tuple[int, int, int, int]] = []
        for g in gears:
            gid = g["gid"]
            if gid not in gear_names:
                continue
            nm = gear_names[gid]
            cx, cy = g["center"]
            put_text_outline(det_img, nm, (cx + 10, cy + 10), 0.75, 2, (0, 255, 0))
            place_label_no_overlap(label_img, nm, (cx, cy), occupied, color=(0, 255, 0), scale=0.90, thick=2, leader=True)

        # chain lines
        for (gidA, gidB, hit, _box) in chain_pairs:
            gA = gears_by_gid(gears, gidA)
            gB = gears_by_gid(gears, gidB)
            p1 = (int(gA["center"][0]), int(gA["center"][1]))
            p2 = (int(gB["center"][0]), int(gB["center"][1]))
            cv2.line(det_img, p1, p2, (255, 255, 255), 2, cv2.LINE_AA)
            mid = (int((p1[0] + p2[0]) * 0.5), int((p1[1] + p2[1]) * 0.5))
            put_text_outline(det_img, f"contact:{hit:.2f}", (mid[0] + 6, mid[1] + 6), 0.55, 2, (255, 255, 255))

        # HUD
        hud_lines: List[str] = [
            f"Gears: {len(gears)}  Spacers: {len(spacers)}  Shafts: {len(shaft_obbs)}",
            f"Stages: {num_stages}",
        ]
        if R_total is None or out_rpm is None:
            hud_lines += ["Total ratio: N/A", "Output speed: N/A"]
        else:
            hud_lines += [f"Total ratio: {R_total:.3f}", f"Output speed: {out_rpm:.1f} RPM"]
            for (s, R, z1, z2) in per_stage:
                hud_lines.append(f"Stage{s}: {z2}/{z1}={R:.2f}")

        if ENABLE_ERROR_CHECKS:
            if errors:
                hud_lines.append("---- ISSUES ----")
                hud_lines += [e["message"] for e in errors[:6]]
            else:
                hud_lines.append("Issues: None")

        draw_hud_lines(det_img, hud_lines)
        draw_hud_lines(label_img, hud_lines)

        highlight_gear_by_name(det_img, gears, gear_names, "gear11", color=(0, 0, 255), thick=5)
        highlight_gear_by_name(det_img, gears, gear_names, "gear12", color=(0, 165, 255), thick=5)
        highlight_gear_by_name(label_img, gears, gear_names, "gear11", color=(0, 0, 255), thick=5)
        highlight_gear_by_name(label_img, gears, gear_names, "gear12", color=(0, 165, 255), thick=5)

        out["images"] = {"det_img": det_img, "label_img": label_img}

    return out
