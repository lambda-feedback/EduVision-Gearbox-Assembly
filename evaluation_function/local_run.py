from urllib.parse import urlparse, unquote
import os
import cv2
import numpy as np

from evaluation_function.yolo_pipeline import run_yolo_pipeline

# ---- local test image ----
IMAGE_PATH = r"C:\Users\sheng\Desktop\Test.jpg"

# ---- model files ----
MODEL_A_REL = "modelA.pt"
MODEL_B_REL = "modelB.pt"
MODEL_C_REL = "modelC.pt"

# ---- task controls ----
TASK = "full"  # "parts_inventory" | "shaft" | "spacer" | "gear_inventory" | "mesh_ratio" | "full"
PART_TYPE = "gear"  # "gear" | "shaft" | "spacer" (only used when TASK=="parts_inventory")
EXPECTED_GEARS = None
RUN_ALL_TASKS = False

RETURN_IMAGES = False

OUT_DIR = os.path.join(os.path.dirname(__file__), "_local_out")
os.makedirs(OUT_DIR, exist_ok=True)


def load_bgr_image_from_url(url: str):
    if not isinstance(url, str) or not url:
        return None, "URL is empty or not a string."

    parsed = urlparse(url)

    if parsed.scheme == "file":
        path = unquote(parsed.path)

        if os.name == "nt" and path.startswith("/"):
            path = path[1:]

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None, f"Cannot read image from local path: {path}"

        return img, None

    return None, f"Unsupported URL scheme: {parsed.scheme}"


def _safe_task_dir(task_name: str) -> str:
    t = (task_name or "full").strip().lower()
    d = os.path.join(OUT_DIR, t)
    os.makedirs(d, exist_ok=True)
    return d


def _print_result(task_name: str, result: dict):
    print("\n==============================")
    print(f"===== TASK: {task_name} =====")
    print("==============================")

    tr = result.get("task_result")
    if isinstance(tr, dict):
        print("\n===== TASK_RESULT =====")
        for k in ["task", "status", "is_ready_for_next", "recommended_next_task", "focus"]:
            if k in tr:
                print(f"{k}: {tr.get(k)}")
        msgs = tr.get("messages")
        if isinstance(msgs, list) and msgs:
            print("messages:")
            for m in msgs[:12]:
                print(f"  - {m}")

    print("\n===== SUMMARY =====")
    print(result.get("summary", {}))

    print("\n===== COUNTS =====")
    print(result.get("counts", {}))

    print("\n===== RATIO =====")
    print(result.get("ratio", {}))

    print("\n===== ERRORS =====")
    errs = result.get("errors", [])
    if not errs:
        print("(none)")
    else:
        for e in errs:
            if isinstance(e, dict):
                print(f"- {e.get('code')}: {e.get('message')}")
            else:
                print(f"- {e}")

    print("\n===== TIMING =====")
    print(result.get("timing", {}))


def _save_images(task_name: str, result: dict):
    imgs = result.get("images", None)
    if not imgs:
        print("\n[INFO] No images returned (result['images'] missing).")
        return

    task_dir = _safe_task_dir(task_name)
    det_path = os.path.join(task_dir, "det.jpg")
    lab_path = os.path.join(task_dir, "labels.jpg")

    if "det_img" in imgs and isinstance(imgs["det_img"], np.ndarray):
        cv2.imwrite(det_path, imgs["det_img"])
        print(f"\nSaved: {det_path}")
    else:
        print("\n[WARN] det_img not found in result['images'].")

    if "label_img" in imgs and isinstance(imgs["label_img"], np.ndarray):
        cv2.imwrite(lab_path, imgs["label_img"])
        print(f"Saved: {lab_path}")
    else:
        print("[WARN] label_img not found in result['images'].")


def _run_one(task_name: str, img_bgr: np.ndarray, part_type: str | None = None):
    kwargs = {
        "model_a_rel": MODEL_A_REL,
        "model_b_rel": MODEL_B_REL,
        "model_c_rel": MODEL_C_REL,
        "return_images": RETURN_IMAGES,
        "task": task_name,
    }

    if task_name == "parts_inventory" and part_type:
        kwargs["part_type"] = part_type

    if task_name == "gear_inventory" and EXPECTED_GEARS is not None:
        kwargs["expected_gears"] = EXPECTED_GEARS

    result = run_yolo_pipeline(img_bgr, **kwargs)
    _print_result(task_name, result)

    if RETURN_IMAGES:
        _save_images(task_name, result)


def main():
    abs_path = os.path.abspath(IMAGE_PATH).replace("\\", "/")
    response = [{"url": f"file:///{abs_path}"}]

    url = response[0].get("url")
    img, err = load_bgr_image_from_url(url)
    if err:
        raise SystemExit(f"[ERROR] {err}")

    print("\n===== RESPONSE (URL) =====")
    print(url)

    if RUN_ALL_TASKS:
        _run_one("parts_inventory", img, "gear")
        _run_one("parts_inventory", img, "shaft")
        _run_one("parts_inventory", img, "spacer")
        _run_one("shaft", img)
        _run_one("spacer", img)
        _run_one("gear_inventory", img)
        _run_one("mesh_ratio", img)
        _run_one("full", img)
    else:
        if TASK == "parts_inventory":
            _run_one(TASK, img, PART_TYPE)
        else:
            _run_one(TASK, img)


if __name__ == "__main__":
    main()