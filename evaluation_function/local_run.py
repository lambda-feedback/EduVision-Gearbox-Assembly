# -*- coding: utf-8 -*-
from urllib.parse import urlparse, unquote
import os
import cv2
import numpy as np

from evaluation_function.yolo_pipeline import run_yolo_pipeline


# ----local test image ----
IMAGE_PATH = r"C:\Users\sheng\Desktop\Test.jpg"

OUT_DIR = os.path.join(os.path.dirname(__file__), "_local_out")
os.makedirs(OUT_DIR, exist_ok=True)


def load_bgr_image_from_url(url: str):
    """
    Correctly load image from:
      - file:// URL (Windows / Linux / macOS)
    """
    if not isinstance(url, str) or not url:
        return None, "URL is empty or not a string."

    parsed = urlparse(url)

    if parsed.scheme == "file":
        # parsed.path on Windows looks like: /C:/Users/...
        path = unquote(parsed.path)

        # Fix Windows leading slash: /C:/... -> C:/...
        if os.name == "nt" and path.startswith("/"):
            path = path[1:]

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None, f"Cannot read image from local path: {path}"

        return img, None

    return None, f"Unsupported URL scheme: {parsed.scheme}"


def main():
    # 1) Build a Lambda-like response payload
    # Use forward slashes to make file:// URL robust
    abs_path = os.path.abspath(IMAGE_PATH).replace("\\", "/")
    response = [{"url": f"file:///{abs_path}"}]

    # 2) Load image via URL (simulate platform URL flow)
    url = response[0].get("url")
    img, err = load_bgr_image_from_url(url)
    if err:
        raise SystemExit(f"[ERROR] {err}")

    # 3) Run your pipeline
    result = run_yolo_pipeline(img, return_images=True)

    # 4) Print outputs
    print("\n===== RESPONSE (URL) =====")
    print(url)

    print("\n===== SUMMARY =====")
    print(result.get("summary", {}))

    print("\n===== RATIO =====")
    print(result.get("ratio", {}))

    print("\n===== ERRORS =====")
    for e in result.get("errors", []):
        print(f"- {e.get('code')}: {e.get('message')}")

    # 5) Save images
    imgs = result.get("images", None)
    if imgs:
        det_path = os.path.join(OUT_DIR, "det.jpg")
        lab_path = os.path.join(OUT_DIR, "labels.jpg")

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


if __name__ == "__main__":
    main()
