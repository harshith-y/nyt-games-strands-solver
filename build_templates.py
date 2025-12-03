# build_templates.py
import os
from collections import defaultdict

import cv2
import numpy as np

from vision import StrandsOCRv2, AutoGridDetector
from ALL_GROUND_TRUTH import GROUND_TRUTH   # uses your attached file

# All screenshots live here
SCREENSHOT_DIR = "data/screenshots"
TEMPLATE_SIZE = (64, 64)  # (H, W) – canonical size for matching

def preprocess_for_templates(ocr: StrandsOCRv2, cell_image):
    """
    Use the same preprocessing as OCR, then resize to a fixed size.
    """
    processed = ocr.preprocess_for_ocr(cell_image)
    resized = cv2.resize(processed, TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)
    return resized


def extract_cells_for_templates(ocr: StrandsOCRv2, rows=8, cols=6):
    """
    Extract raw cell crops using the same grid detection + cell slicing
    logic used in the main pipeline.

    Returns:
        list of (row_idx, col_idx, raw_cell_image)
    """
    if ocr.image is None:
        ocr.load_image()
    if ocr.is_dark_mode is None:
        ocr.detect_mode()

    detector = AutoGridDetector(ocr.image, is_dark_mode=ocr.is_dark_mode)
    rows_detected, cols_detected, grid_region = detector.auto_detect()

    grid_top, grid_bottom, grid_left, grid_right = grid_region
    grid_image = ocr.image[grid_top:grid_bottom, grid_left:grid_right]

    H, W = grid_image.shape[:2]
    cell_h = H // rows
    cell_w = W // cols

    cell_crops = []

    for r in range(rows):
        for c in range(cols):
            y1 = r * cell_h
            y2 = (r + 1) * cell_h
            x1 = c * cell_w
            x2 = (c + 1) * cell_w

            # Use the same kind of inner padding as extract_grid_with_detection
            pad_y = int(0.10 * cell_h)
            pad_x = int(0.10 * cell_w)

            yy1 = max(y1 + pad_y, 0)
            yy2 = min(y2 - pad_y, H)
            xx1 = max(x1 + pad_x, 0)
            xx2 = min(x2 - pad_x, W)

            raw_cell = grid_image[yy1:yy2, xx1:xx2].copy()
            cell_crops.append((r, c, raw_cell))

    return cell_crops


def build_template_bank(output_path="templates/letter_templates_v1.npz", limit=100):
    """
    Build template bank using only the first `limit` ground-truth entries.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    bank = defaultdict(list)

    # Select only first `limit` items
    filenames = list(GROUND_TRUTH.keys())[:limit]

    print(f"[INFO] Using {len(filenames)} puzzles for template building")

    for filename in filenames:
        grid = GROUND_TRUTH[filename]
        image_path = os.path.join(SCREENSHOT_DIR, filename)
        if not os.path.exists(image_path):
            print(f"[WARN] Screenshot {image_path} missing, skipping")
            continue

        print(f"[INFO] Processing {image_path}")

        rows = len(grid)
        cols = len(grid[0])

        ocr = StrandsOCRv2(
            image_path,
            ocr_engine="pytesseract",  # doesn't matter
            output_dir="data/debug/templates",
        )

        cells = extract_cells_for_templates(ocr, rows=rows, cols=cols)

        for (r, c, raw_cell) in cells:
            true_letter = grid[r][c]
            if not isinstance(true_letter, str) or not true_letter.isalpha():
                continue
            true_letter = true_letter.upper()

            proc = preprocess_for_templates(ocr, raw_cell)
            bank[true_letter].append(proc)

    # Save to NPZ
    npz_payload = {}
    for letter, imgs in bank.items():
        arr = np.stack(imgs, axis=0).astype(np.uint8)
        npz_payload[letter] = arr
        print(f"[INFO] Letter {letter}: {arr.shape[0]} samples")

    np.savez(output_path, **npz_payload)
    print(f"[DONE] Saved templates to {output_path}")


if __name__ == "__main__":
    build_template_bank()
