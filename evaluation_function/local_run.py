from urllib.parse import urlparse, unquote
import os
import cv2
import json
import shutil
import numpy as np

from evaluation_function.yolo_pipeline import run_yolo_pipeline

# ---- local test image or folder ----
IMAGE_PATH = r"C:\Users\sheng\Desktop\Eastern week 3"

# ---- model files ----
MODEL_A_REL = "modelA.pt"
MODEL_B_REL = "modelB.pt"
MODEL_C_REL = "modelC.pt"

# ---- where you want to save outputs ----
SAVE_ROOT = r"C:\Users\sheng\Desktop\Pipeline_saved_results"

# ---- task controls ----
TASK = "full"  # "parts_inventory" | "shaft" | "spacer" | "gear_inventory" | "mesh_ratio" | "full"
#PART_TYPE = "gear"  # "gear" | "shaft" | "spacer" (only used when TASK=="parts_inventory")
EXPECTED_GEARS = None
RUN_ALL_TASKS = False

# must be True if you want det/label images from pipeline
RETURN_IMAGES = True

os.makedirs(SAVE_ROOT, exist_ok=True)


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


def find_image_files(path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    if os.path.isfile(path):
        return [path]

    if os.path.isdir(path):
        files = []
        for name in os.listdir(path):
            full = os.path.join(path, name)
            if os.path.isfile(full) and os.path.splitext(name)[1].lower() in exts:
                files.append(full)
        return sorted(files)

    return []


def safe_name(name: str) -> str:
    bad = '\\/:*?"<>|'
    for ch in bad:
        name = name.replace(ch, "_")
    return name


def make_output_dir(image_path: str, task_name: str) -> str:
    image_stem = os.path.splitext(os.path.basename(image_path))[0]
    folder_name = f"{safe_name(image_stem)}__{safe_name(task_name)}"
    out_dir = os.path.join(SAVE_ROOT, folder_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items() if k != "images"}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    return obj


def format_result_text(task_name: str, result: dict) -> str:
    lines = []
    lines.append("==============================")
    lines.append(f"===== TASK: {task_name} =====")
    lines.append("==============================")
    lines.append("")

    tr = result.get("task_result")
    if isinstance(tr, dict):
        lines.append("===== TASK_RESULT =====")
        for k in ["task", "status", "is_ready_for_next", "recommended_next_task", "focus"]:
            if k in tr:
                lines.append(f"{k}: {tr.get(k)}")
        msgs = tr.get("messages")
        if isinstance(msgs, list) and msgs:
            lines.append("messages:")
            for m in msgs:
                lines.append(f"  - {m}")
        lines.append("")

    lines.append("===== SUMMARY =====")
    lines.append(str(result.get("summary", {})))
    lines.append("")

    lines.append("===== COUNTS =====")
    lines.append(str(result.get("counts", {})))
    lines.append("")

    lines.append("===== RATIO =====")
    lines.append(str(result.get("ratio", {})))
    lines.append("")

    lines.append("===== ERRORS =====")
    errs = result.get("errors", [])
    if not errs:
        lines.append("(none)")
    else:
        for e in errs:
            if isinstance(e, dict):
                lines.append(f"- {e.get('code')}: {e.get('message')}")
            else:
                lines.append(f"- {e}")
    lines.append("")

    lines.append("===== TIMING =====")
    lines.append(str(result.get("timing", {})))
    lines.append("")

    return "\n".join(lines)


def save_outputs(image_path: str, task_name: str, img_bgr: np.ndarray, result: dict):
    out_dir = make_output_dir(image_path, task_name)

    # 1) save original image
    original_path = os.path.join(out_dir, "original.jpg")
    cv2.imwrite(original_path, img_bgr)

    # 2) save pipeline output images
    imgs = result.get("images", {})
    det_path = os.path.join(out_dir, "det.jpg")
    label_path = os.path.join(out_dir, "labels.jpg")

    if isinstance(imgs, dict):
        if "det_img" in imgs and isinstance(imgs["det_img"], np.ndarray):
            cv2.imwrite(det_path, imgs["det_img"])
        if "label_img" in imgs and isinstance(imgs["label_img"], np.ndarray):
            cv2.imwrite(label_path, imgs["label_img"])

    # 3) save staging / task result content as json
    json_result = to_jsonable(result)
    json_path = os.path.join(out_dir, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_result, f, ensure_ascii=False, indent=2)

    # 4) save readable text report
    txt_path = os.path.join(out_dir, "report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(format_result_text(task_name, result))

    # 5) optional: save a copy of the source file with original filename
    src_copy_path = os.path.join(out_dir, os.path.basename(image_path))
    if os.path.abspath(src_copy_path) != os.path.abspath(image_path):
        shutil.copy2(image_path, src_copy_path)

    print(f"\nSaved outputs to: {out_dir}")
    print(f"  - original image: {original_path}")
    if os.path.exists(det_path):
        print(f"  - det image:      {det_path}")
    if os.path.exists(label_path):
        print(f"  - label image:    {label_path}")
    print(f"  - report text:    {txt_path}")
    print(f"  - result json:    {json_path}")


def print_result(task_name: str, result: dict):
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
            for m in msgs:
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


def run_one(task_name: str, image_path: str, img_bgr: np.ndarray, part_type: str | None = None):
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
    print_result(task_name, result)
    save_outputs(image_path, task_name, img_bgr, result)


def main():
    image_files = find_image_files(IMAGE_PATH)
    if not image_files:
        raise SystemExit(f"[ERROR] No image file found in: {IMAGE_PATH}")

    for image_path in image_files:
        abs_path = os.path.abspath(image_path).replace("\\", "/")
        response = [{"url": f"file:///{abs_path}"}]

        url = response[0].get("url")
        img, err = load_bgr_image_from_url(url)
        if err:
            print(f"[ERROR] {err}")
            continue

        print("\n===== RESPONSE (URL) =====")
        print(url)

        if RUN_ALL_TASKS:
            run_one("parts_inventory", image_path, img, "gear")
            run_one("parts_inventory", image_path, img, "shaft")
            run_one("parts_inventory", image_path, img, "spacer")
            run_one("shaft", image_path, img)
            run_one("spacer", image_path, img)
            run_one("gear_inventory", image_path, img)
            run_one("mesh_ratio", image_path, img)
            run_one("full", image_path, img)
        else:
            if TASK == "parts_inventory":
                run_one(TASK, image_path, img, PART_TYPE)
            else:
                run_one(TASK, image_path, img)


if __name__ == "__main__":
    main()