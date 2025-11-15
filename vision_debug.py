"""
Debug OCR - Save preprocessed images to see what OCR actually sees
Helps identify why specific letters are failing
"""

import cv2
import numpy as np
from PIL import Image
import os
from vision import StrandsOCRv2


def debug_ocr_cells(image_path, save_dir="debug_ocr_cells"):
    """
    Extract and save preprocessed images of all cells
    Shows exactly what the OCR engine sees
    """
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*70)
    print("OCR DEBUG - Saving Preprocessed Cells")
    print("="*70)
    
    # Initialize OCR
    ocr = StrandsOCRv2(image_path, ocr_engine='easyocr')
    ocr.load_image()
    ocr.detect_mode()
    
    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Mode: {'dark' if ocr.is_dark_mode else 'light'}")
    
    # Extract grid
    print("\nExtracting grid...")
    grid = ocr.extract_grid_with_detection()  # This now includes auto-corrections!
    
    # Grid is already corrected by extract_grid_with_detection()
    
    # Also get the raw cells before OCR
    height, width = ocr.image.shape[:2]
    
    if height > 2000:
        grid_top = int(height * 0.38)
        grid_bottom = int(height * 0.85)
        grid_left = int(width * 0.10)
        grid_right = int(width * 0.90)
    else:
        grid_top = int(height * 0.35)
        grid_bottom = int(height * 0.85)
        grid_left = int(width * 0.08)
        grid_right = int(width * 0.92)
    
    # Add margin
    margin_v = int((grid_bottom - grid_top) * 0.03)
    margin_h = int((grid_right - grid_left) * 0.03)
    
    grid_top = max(0, grid_top - margin_v)
    grid_bottom = min(height, grid_bottom + margin_v)
    grid_left = max(0, grid_left - margin_h)
    grid_right = min(width, grid_right + margin_h)
    
    grid_image = ocr.image[grid_top:grid_bottom, grid_left:grid_right]
    grid_height, grid_width = grid_image.shape[:2]
    
    rows, cols = 8, 6
    cell_height = grid_height // rows
    cell_width = grid_width // cols
    
    padding = 0.12  # Same as vision.py for high-res
    
    print(f"\nSaving {rows}x{cols} cells...")
    print(f"Each cell: ~{cell_width}x{cell_height}px")
    
    # Extract and save each cell
    errors = []
    
    for row in range(rows):
        for col in range(cols):
            # Extract cell
            y1 = row * cell_height + int(cell_height * padding)
            y2 = (row + 1) * cell_height - int(cell_height * padding)
            x1 = col * cell_width + int(cell_width * padding)
            x2 = (col + 1) * cell_width - int(cell_width * padding)
            
            cell = grid_image[y1:y2, x1:x2]
            
            # Preprocess (same as OCR sees)
            processed = ocr.preprocess_for_ocr(cell)
            
            # Get OCR result
            letter = grid[row][col]
            
            # Save both raw and preprocessed
            raw_path = f"{save_dir}/cell_{row+1}_{col+1}_raw.png"
            proc_path = f"{save_dir}/cell_{row+1}_{col+1}_OCR_{letter}.png"
            
            cv2.imwrite(raw_path, cell)
            cv2.imwrite(proc_path, processed)
            
            if letter == '?':
                errors.append((row+1, col+1, raw_path, proc_path))
    
    print(f"\n✓ Saved all cells to: {save_dir}/")
    
    # Report errors
    if errors:
        print(f"\n❌ Found {len(errors)} OCR errors:")
        print("\nProblem cells:")
        for row, col, raw, proc in errors:
            print(f"  Position ({row},{col}): {proc}")
        
        print("\n💡 Open these images to see why OCR failed!")
        print("  • *_raw.png = original cell")
        print("  • *_OCR_?.png = preprocessed (what OCR sees)")
    else:
        print("\n✅ No errors! All letters recognized correctly!")
    
    # Summary grid
    print("\n" + "="*70)
    print("FINAL GRID")
    print("="*70)
    for row in grid:
        print(' '.join(row))
    
    return grid, errors


if __name__ == "__main__":
    import sys
    import glob
    
    # Find image
    sample_images = glob.glob("data/samples/*.PNG") + glob.glob("data/samples/*.png")
    
    if not sample_images and len(sys.argv) < 2:
        print("Usage: python debug_ocr.py <image_path>")
        print("\nOr place images in data/samples/")
        exit()
    
    image_path = "data/samples/IMG_1016.PNG"
    
    grid, errors = debug_ocr_cells(image_path)
    
    if errors:
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("\n1. Open the *_OCR_?.png files for error positions")
        print("2. Check if letter is visible in preprocessed image")
        print("3. If visible → OCR engine issue")
        print("4. If not visible → Preprocessing issue")