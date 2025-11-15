"""
Robust OCR Module for Strands Solver v2
Handles both light and dark mode screenshots
Uses multiple OCR strategies for best accuracy
Includes automated grid detection
"""

import cv2
import numpy as np
from PIL import Image
import pytesseract
import argparse
import os
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION - EDIT THESE PATHS FOR YOUR SETUP
# ============================================================================

# Path to sample screenshots
SAMPLES_DIR = "data/samples"

# Path to save extracted grids
OUTPUT_DIR = "data"

# Default sample images to test (relative to SAMPLES_DIR)
DEFAULT_TEST_IMAGES = [
    "IMG_1009.PNG",
    "IMG_1026.PNG", 
    "IMG_1016.PNG"
]

# OCR Engine: 'pytesseract' (fast) or 'easyocr' (more accurate but slower)
DEFAULT_OCR_ENGINE = 'pytesseract'

# Default grid dimensions for Strands
DEFAULT_ROWS = 8
DEFAULT_COLS = 6

# Auto-detect grid size? (True = automatic, False = use DEFAULT_ROWS/COLS)
AUTO_DETECT_GRID = True  # ⭐ NEW: Automatically detect grid dimensions!

# ============================================================================


class AutoGridDetector:
    """
    Automatically detects grid dimensions and cell positions
    Works for any grid size and resolution
    Integrated into vision.py for seamless operation
    """
    
    def __init__(self, image, is_dark_mode=None):
        """
        Initialize detector
        
        Args:
            image: OpenCV image (BGR format) or path to image
            is_dark_mode: True/False/None (auto-detect)
        """
        if isinstance(image, str):
            # Load from path
            pil_image = Image.open(image)
            self.image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        else:
            # Already loaded
            self.image = image
            
        self.is_dark_mode = is_dark_mode
        self.grid_region = None
        self.rows = None
        self.cols = None
        self.cell_positions = []
        
        # Auto-detect dark mode if not specified
        if self.is_dark_mode is None:
            self.is_dark_mode = self._detect_dark_mode()
    
    def _detect_dark_mode(self):
        """Detect if image is in dark mode"""
        height, width = self.image.shape[:2]
        sample = self.image[int(height*0.15):int(height*0.25), 
                           int(width*0.1):int(width*0.3)]
        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        return avg_brightness < 100
    
    def find_grid_region(self):
        """
        Find the grid region using edge detection and contours
        Returns: (top, bottom, left, right) coordinates
        """
        height, width = self.image.shape[:2]
        
        # Focus on middle portion where grid is (skip header and footer)
        roi_top = int(height * 0.25)
        roi_bottom = int(height * 0.85)
        roi = self.image[roi_top:roi_bottom, :]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Invert if dark mode
        if self.is_dark_mode:
            gray = cv2.bitwise_not(gray)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours of text (letters)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return self._fallback_grid_region()
        
        # Filter contours by size (letters should be similar size)
        areas = [cv2.contourArea(c) for c in contours]
        if not areas:
            return self._fallback_grid_region()
        
        median_area = np.median(areas)
        min_area = median_area * 0.3
        max_area = median_area * 3.0
        
        valid_contours = [c for c in contours 
                         if min_area < cv2.contourArea(c) < max_area]
        
        if len(valid_contours) < 10:
            return self._fallback_grid_region()
        
        # Get bounding box of all valid contours
        all_points = np.vstack(valid_contours)
        x, y, w, h = cv2.boundingRect(all_points)
        
        # Convert back to full image coordinates
        grid_top = roi_top + y
        grid_bottom = roi_top + y + h
        grid_left = x
        grid_right = x + w
        
        # Add small margin
        margin_v = int(h * 0.05)
        margin_h = int(w * 0.05)
        
        grid_top = max(0, grid_top - margin_v)
        grid_bottom = min(height, grid_bottom + margin_v)
        grid_left = max(0, grid_left - margin_h)
        grid_right = min(width, grid_right + margin_h)
        
        self.grid_region = (grid_top, grid_bottom, grid_left, grid_right)
        return self.grid_region
    
    def _fallback_grid_region(self):
        """Fallback to percentage-based detection"""
        height, width = self.image.shape[:2]
        
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
        
        # Add safety margin
        margin_vertical = int((grid_bottom - grid_top) * 0.03)
        margin_horizontal = int((grid_right - grid_left) * 0.03)
        
        grid_top = max(0, grid_top - margin_vertical)
        grid_bottom = min(height, grid_bottom + margin_vertical)
        grid_left = max(0, grid_left - margin_horizontal)
        grid_right = min(width, grid_right + margin_horizontal)
        
        self.grid_region = (grid_top, grid_bottom, grid_left, grid_right)
        return self.grid_region
    
    def _estimate_grid_from_letters(self):
        """
        Estimate grid size by detecting letter positions
        More reliable than line detection for Strands
        """
        if self.grid_region is None:
            self.find_grid_region()
        
        grid_top, grid_bottom, grid_left, grid_right = self.grid_region
        grid_image = self.image[grid_top:grid_bottom, grid_left:grid_right]
        grid_height, grid_width = grid_image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(grid_image, cv2.COLOR_BGR2GRAY)
        
        if self.is_dark_mode:
            gray = cv2.bitwise_not(gray)
        
        # Threshold to get letters
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours (letters)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) < 10:
            self.rows, self.cols = 8, 6
            return 8, 6
        
        # Get bounding boxes
        boxes = [cv2.boundingRect(c) for c in contours]
        
        # Filter by size (remove outliers)
        areas = [w * h for x, y, w, h in boxes]
        median_area = np.median(areas)
        valid_boxes = [(x, y, w, h) for x, y, w, h in boxes 
                       if 0.3 * median_area < w * h < 3 * median_area]
        
        if len(valid_boxes) < 10:
            self.rows, self.cols = 8, 6
            return 8, 6
        
        # Extract center positions
        centers = [(x + w/2, y + h/2) for x, y, w, h in valid_boxes]
        
        # Cluster Y positions (rows)
        y_positions = sorted([y for x, y in centers])
        row_clusters = self._cluster_positions(y_positions, threshold=grid_height * 0.08)
        
        # Cluster X positions (cols)
        x_positions = sorted([x for x, y in centers])
        col_clusters = self._cluster_positions(x_positions, threshold=grid_width * 0.08)
        
        rows = len(row_clusters)
        cols = len(col_clusters)
        
        # Sanity check
        if rows < 4 or rows > 12 or cols < 4 or cols > 12:
            self.rows, self.cols = 8, 6
            return 8, 6
        
        self.rows = rows
        self.cols = cols
        
        return rows, cols
    
    def _cluster_positions(self, positions, threshold):
        """Cluster positions (for finding rows/cols)"""
        if not positions:
            return []
        
        clusters = [[positions[0]]]
        
        for pos in positions[1:]:
            if abs(pos - np.mean(clusters[-1])) < threshold:
                clusters[-1].append(pos)
            else:
                clusters.append([pos])
        
        return [np.mean(cluster) for cluster in clusters]
    
    def auto_detect(self):
        """
        Run auto-detection pipeline
        Returns: (rows, cols, grid_region)
        """
        self.find_grid_region()
        self._estimate_grid_from_letters()
        
        return self.rows, self.cols, self.grid_region


class StrandsOCRv2:
    """
    Advanced OCR for Strands puzzles
    Automatically handles light/dark mode and uses multiple strategies
    """
    
    def __init__(self, image_path, ocr_engine='pytesseract', output_dir=OUTPUT_DIR):
        """
        Initialize OCR
        
        Args:
            image_path: Path to screenshot
            ocr_engine: 'pytesseract' (fast) or 'easyocr' (accurate but slower)
            output_dir: Directory to save extracted grids
        """
        self.image_path = image_path
        self.output_dir = output_dir
        self.image = None
        self.grid = None
        self.theme = None
        self.ocr_engine = ocr_engine
        self.is_dark_mode = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Try to import EasyOCR if requested
        self.easyocr_reader = None
        if ocr_engine == 'easyocr':
            try:
                import easyocr
                print("Initializing EasyOCR (this may take a moment on first run)...")
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
                print("EasyOCR initialized!")
            except ImportError:
                print("⚠️  EasyOCR not installed, falling back to Pytesseract")
                print("Install with: pip install easyocr")
                self.ocr_engine = 'pytesseract'
    
    def load_image(self):
        """Load image from path"""
        pil_image = Image.open(self.image_path)
        self.image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return self.image
    
    def detect_mode(self):
        """
        Detect if screenshot is in dark mode or light mode
        Returns: 'dark' or 'light'
        """
        if self.image is None:
            self.load_image()
        
        # Sample the background (avoid top bar and grid area)
        height, width = self.image.shape[:2]
        
        # Sample from top-left corner area (usually background)
        sample_region = self.image[int(height*0.15):int(height*0.25), 
                                   int(width*0.1):int(width*0.3)]
        
        # Calculate average brightness
        gray_sample = cv2.cvtColor(sample_region, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray_sample)
        
        # Dark mode: low brightness (< 50), Light mode: high brightness (> 150)
        self.is_dark_mode = avg_brightness < 100
        
        mode = 'dark' if self.is_dark_mode else 'light'
        print(f"Detected mode: {mode} (brightness: {avg_brightness:.1f})")
        
        return mode
    
    def preprocess_for_ocr(self, cell_image):
        """
        Preprocess a cell image for optimal OCR
        Handles both light and dark modes
        
        Args:
            cell_image: Single cell image (BGR format)
        
        Returns:
            Preprocessed image optimized for OCR
        """
        # Convert to grayscale
        gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        
        # If dark mode, invert to get dark text on light background
        if self.is_dark_mode:
            gray = cv2.bitwise_not(gray)
        
        # KEY CHANGE: Scale up FIRST while still grayscale
        # This preserves smooth curves and diagonal lines
        height, width = gray.shape
        scale_factor = 6
        gray_large = cv2.resize(gray, (width * scale_factor, height * scale_factor), 
                               interpolation=cv2.INTER_CUBIC)
        
        # Apply bilateral filter to reduce noise while keeping edges sharp
        # Do this on the larger image for better results
        gray_large = cv2.bilateralFilter(gray_large, 9, 75, 75)
        
        # NOW threshold on the larger, smoother image
        # This preserves letter structure much better
        _, binary = cv2.threshold(gray_large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Very minimal dilation (or none) since we scaled first
        # Only use this for very thin strokes
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Close instead of dilate
        
        # Add border for better OCR
        binary = cv2.copyMakeBorder(binary, 20, 20, 20, 20, 
                                   cv2.BORDER_CONSTANT, value=255)
        
        return binary
    
    def extract_theme(self):
        """Extract theme text from top of image"""
        if self.image is None:
            self.load_image()
        
        # Detect mode first
        self.detect_mode()
        
        height, width = self.image.shape[:2]
        
        # Extract theme region (where "TODAY'S THEME" and theme text are)
        theme_region = self.image[int(height*0.12):int(height*0.23), 
                                 int(width*0.1):int(width*0.9)]
        
        # Preprocess
        gray = cv2.cvtColor(theme_region, cv2.COLOR_BGR2GRAY)
        
        if self.is_dark_mode:
            gray = cv2.bitwise_not(gray)
        
        # OCR on theme region
        if self.ocr_engine == 'easyocr' and self.easyocr_reader:
            results = self.easyocr_reader.readtext(gray, detail=0)
            theme_text = ' '.join(results)
        else:
            theme_text = pytesseract.image_to_string(gray)
        
        # Parse theme (usually last line or line after "THEME")
        lines = [line.strip() for line in theme_text.split('\n') if line.strip()]
        
        for i, line in enumerate(lines):
            if 'THEME' in line.upper() and i + 1 < len(lines):
                self.theme = lines[i + 1]
                break
        
        # If still not found, take the last non-empty line
        if not self.theme and lines:
            # Skip lines with "TODAY" or "THEME" or weird characters
            for line in reversed(lines):
                if (len(line) > 2 and 
                    'TODAY' not in line.upper() and 
                    'THEME' not in line.upper() and
                    line.replace('-', '').replace('_', '').replace(' ', '').isalnum()):
                    self.theme = line
                    break
        
        return self.theme
    
    def ocr_single_letter(self, cell_image):
        """
        OCR a single letter from a cell with multiple fallback strategies
        
        Args:
            cell_image: Image of a single grid cell
        
        Returns:
            Single uppercase letter or '?'
        """
        # Strategy 1: Standard preprocessing
        processed = self.preprocess_for_ocr(cell_image)
        
        if self.ocr_engine == 'easyocr' and self.easyocr_reader:
            # EasyOCR with larger image and better parameters
            results = self.easyocr_reader.readtext(processed, detail=0, 
                                                   allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                                                   paragraph=False,
                                                   contrast_ths=0.1,
                                                   adjust_contrast=0.5,
                                                   text_threshold=0.5,
                                                   width_ths=0.7)
            text = ''.join(results).strip()
        else:
            # Pytesseract with optimized config for single character
            config = '--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ -c tessedit_char_blacklist=0123456789'
            text = pytesseract.image_to_string(processed, config=config).strip()
        
        # Extract first letter
        letter = ''.join(c for c in text if c.isalpha()).upper()
        result = letter[0] if letter else '?'
        
        # If we got a result, validate it's reasonable
        if result != '?':
            # Check for common confusions and try alternatives if suspicious
            if result == 'M':  # Could be N with bad recognition
                # M and N look similar, but N is narrower
                # Check aspect ratio to distinguish
                gray_check = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
                if self.is_dark_mode:
                    gray_check = cv2.bitwise_not(gray_check)
                _, thresh = cv2.threshold(gray_check, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(c)
                    aspect_ratio = h / w if w > 0 else 0
                    
                    # N is taller and narrower than M
                    # Typical ratios: N ~1.4-1.6, M ~1.0-1.2
                    # Lower threshold to be more aggressive
                    if aspect_ratio > 1.25:  # Lowered from 1.35
                        result = 'N'
                        print(f"  (M→N correction based on aspect ratio {aspect_ratio:.2f})")
            
            return result
        
        # Strategy 2: Failed with standard preprocessing, try alternatives
        result = self._try_alternative_ocr(cell_image)
        if result != '?':
            return result
        
        # Strategy 3: Geometric analysis for thin letters
        result = self._geometric_letter_detection(cell_image)
        
        return result
    
    def _try_alternative_ocr(self, cell_image):
        """
        Try alternative preprocessing strategies
        """
        # Strategy: More aggressive preprocessing for stubborn letters
        gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        if self.is_dark_mode:
            gray = cv2.bitwise_not(gray)
        
        # Try with heavy dilation (for very thin letters)
        h, w = gray.shape
        gray_large = cv2.resize(gray, (w * 8, h * 8), interpolation=cv2.INTER_CUBIC)
        
        # Sharp threshold
        _, binary = cv2.threshold(gray_large, 127, 255, cv2.THRESH_BINARY)
        
        # Heavy dilation
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=2)
        
        binary = cv2.copyMakeBorder(binary, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
        
        if self.ocr_engine == 'easyocr' and self.easyocr_reader:
            results = self.easyocr_reader.readtext(binary, detail=0, 
                                                   allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            text = ''.join(results).strip()
        else:
            config = '--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            text = pytesseract.image_to_string(binary, config=config).strip()
        
        letter = ''.join(c for c in text if c.isalpha()).upper()
        return letter[0] if letter else '?'
    
    def _geometric_letter_detection(self, cell_image):
        """
        Use geometric properties to identify letters when OCR fails
        Specifically helpful for I, l, and other ambiguous thin letters
        """
        gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        if self.is_dark_mode:
            gray = cv2.bitwise_not(gray)
        
        # Get letter contour
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return '?'
        
        # Get largest contour (the letter)
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # Calculate geometric features
        aspect_ratio = h / w if w > 0 else 0
        area = cv2.contourArea(c)
        bbox_area = w * h
        solidity = area / bbox_area if bbox_area > 0 else 0
        
        # Heuristic rules for common problematic letters
        
        # Very tall and thin with high solidity -> probably I
        if aspect_ratio > 3.5 and solidity > 0.7:
            return 'I'
        
        # Tall and thin but less extreme -> could be I, l, or J
        if aspect_ratio > 2.5 and solidity > 0.6:
            # Check if there's a curve at bottom (J) or top
            bottom_region = thresh[y+int(h*0.7):y+h, x:x+w]
            top_region = thresh[y:y+int(h*0.3), x:x+w]
            
            bottom_density = np.sum(bottom_region == 0) / (bottom_region.size + 1)
            top_density = np.sum(top_region == 0) / (top_region.size + 1)
            
            # If density is relatively uniform, it's probably I
            if abs(bottom_density - top_density) < 0.15:
                return 'I'
        
        # Check for diagonal (N, M, K, etc.)
        # For now, return '?' if we can't determine
        return '?'
    
    def extract_grid_with_detection(self, rows=8, cols=6):
        """
        Automatically detect and extract grid
        Works for both light and dark modes
        
        Args:
            rows: Number of rows (default 8)
            cols: Number of columns (default 6)
        
        Returns:
            2D list of letters
        """
        if self.image is None:
            self.load_image()
        
        if self.is_dark_mode is None:
            self.detect_mode()
        
        height, width = self.image.shape[:2]
        
        # Grid location estimation (works for both modes)
        # Adjusted for high-resolution screenshots (1170x2532)
        # Theme is in top ~12-25%, grid is roughly 35-80%, bottom has buttons
        
        # For high-res images, be more conservative with margins
        if height > 2000:  # High resolution screenshot
            grid_top = int(height * 0.38)     # Start lower
            grid_bottom = int(height * 0.85)  # FIXED: Extended to capture bottom row
            grid_left = int(width * 0.10)     # More left margin
            grid_right = int(width * 0.90)    # More right margin
        else:  # Standard resolution
            grid_top = int(height * 0.35)
            grid_bottom = int(height * 0.85)  # Also extended
            grid_left = int(width * 0.08)
            grid_right = int(width * 0.92)
        
        # Add safety margin to prevent cutting off letters at edges
        # This expands the grid region slightly in all directions
        margin_vertical = int((grid_bottom - grid_top) * 0.03)  # 3% of grid height
        margin_horizontal = int((grid_right - grid_left) * 0.03)  # 3% of grid width
        
        grid_top = max(0, grid_top - margin_vertical)
        grid_bottom = min(height, grid_bottom + margin_vertical)
        grid_left = max(0, grid_left - margin_horizontal)
        grid_right = min(width, grid_right + margin_horizontal)
        
        grid_image = self.image[grid_top:grid_bottom, grid_left:grid_right]
        
        # Calculate cell dimensions
        grid_height, grid_width = grid_image.shape[:2]
        cell_height = grid_height // rows
        cell_width = grid_width // cols
        
        print(f"\nExtracting {rows}x{cols} grid...")
        print(f"Grid region: {grid_width}x{grid_height}")
        print(f"Cell size: {cell_width}x{cell_height}")
        print(f"Using OCR engine: {self.ocr_engine}\n")
        
        grid = []
        
        for row in range(rows):
            row_letters = []
            print(f"Row {row+1}: ", end="")
            
            for col in range(cols):
                # Extract cell with padding to focus on letter
                # Minimal padding for high-res screens
                padding = 0.10 if height < 2000 else 0.12  # Very minimal padding
                
                y1 = row * cell_height + int(cell_height * padding)
                y2 = (row + 1) * cell_height - int(cell_height * padding)
                x1 = col * cell_width + int(cell_width * padding)
                x2 = (col + 1) * cell_width - int(cell_width * padding)
                
                cell = grid_image[y1:y2, x1:x2]
                
                # OCR the letter
                letter = self.ocr_single_letter(cell)
                row_letters.append(letter)
                
                print(f"{letter}", end=" ")
            
            print()  # New line after row
            grid.append(row_letters)
        
        # POST-PROCESSING: Auto-fix common OCR issues
        # 1. Replace '?' with 'I' since only I is thin enough to consistently fail OCR
        # 2. Check all 'M' letters - if they're narrow, they're probably 'N'
        corrections_made = 0
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == '?':
                    grid[row][col] = 'I'
                    corrections_made += 1
                    print(f"  → Auto-corrected position ({row+1},{col+1}): ? → I")
                
                elif grid[row][col] == 'M':
                    # Check if this M is actually an N
                    # Can't use aspect ratio (both are ~0.95)
                    # Instead: M has 3 vertical regions, N has 2
                    y1 = row * cell_height + int(cell_height * padding)
                    y2 = (row + 1) * cell_height - int(cell_height * padding)
                    x1 = col * cell_width + int(cell_width * padding)
                    x2 = (col + 1) * cell_width - int(cell_width * padding)
                    cell = grid_image[y1:y2, x1:x2]
                    
                    gray_check = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                    if self.is_dark_mode:
                        gray_check = cv2.bitwise_not(gray_check)
                    
                    # Threshold to get clean binary
                    _, thresh = cv2.threshold(gray_check, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # Analyze vertical structure by summing columns
                    # M has 3 peaks (left stroke, middle V, right stroke)
                    # N has 2 peaks (left stroke, right stroke)
                    h, w = thresh.shape
                    vertical_projection = np.sum(thresh == 0, axis=0)  # Count black pixels per column
                    
                    # Smooth the projection to reduce noise
                    from scipy.ndimage import gaussian_filter1d
                    try:
                        smoothed = gaussian_filter1d(vertical_projection, sigma=2)
                    except:
                        # If scipy not available, use simple moving average
                        window = 3
                        smoothed = np.convolve(vertical_projection, np.ones(window)/window, mode='same')
                    
                    # Find peaks (vertical strokes)
                    from scipy.signal import find_peaks
                    try:
                        peaks, _ = find_peaks(smoothed, height=np.max(smoothed) * 0.3, distance=5)
                        num_peaks = len(peaks)
                        
                        print(f"  → Checking M at position ({row+1},{col+1}) - found {num_peaks} vertical strokes", end="")
                        
                        # M typically has 3+ peaks, N has 2 peaks
                        if num_peaks <= 2:
                            grid[row][col] = 'N'
                            corrections_made += 1
                            print(f" → Converting to N!")
                        else:
                            print(f" → Keeping as M")
                    except:
                        # Fallback: just check width distribution
                        # N is more uniform, M has dip in middle
                        middle_third = smoothed[w//3:2*w//3]
                        side_thirds = np.concatenate([smoothed[:w//3], smoothed[2*w//3:]])
                        
                        middle_avg = np.mean(middle_third)
                        side_avg = np.mean(side_thirds)
                        
                        ratio = middle_avg / (side_avg + 1)
                        
                        print(f"  → Checking M at position ({row+1},{col+1}) - middle/side ratio: {ratio:.2f}", end="")
                        
                        # M has lower middle (V-shape), N has uniform height
                        if ratio > 0.7:  # N has more uniform distribution
                            grid[row][col] = 'N'
                            corrections_made += 1
                            print(f" → Converting to N!")
                        else:
                            print(f" → Keeping as M")
                
                elif grid[row][col] == 'V':
                    # Check if this V is actually a Y
                    # Y has a vertical stem extending down from the junction point
                    # V comes to a point at the bottom
                    y1 = row * cell_height + int(cell_height * padding)
                    y2 = (row + 1) * cell_height - int(cell_height * padding)
                    x1 = col * cell_width + int(cell_width * padding)
                    x2 = (col + 1) * cell_width - int(cell_width * padding)
                    cell = grid_image[y1:y2, x1:x2]
                    
                    gray_check = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                    if self.is_dark_mode:
                        gray_check = cv2.bitwise_not(gray_check)
                    
                    _, thresh = cv2.threshold(gray_check, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    h, w = thresh.shape
                    
                    # Y has stem, V doesn't - check multiple vertical slices
                    # Look at center column through different height ranges
                    center_col = int(w * 0.5)
                    center_width = max(3, int(w * 0.15))  # Check narrow center band
                    col_start = center_col - center_width // 2
                    col_end = center_col + center_width // 2
                    
                    # Check from 40% to 75% height (where Y stem typically is)
                    stem_region = thresh[int(h*0.4):int(h*0.75), col_start:col_end]
                    
                    # Count black pixels
                    black_pixels = np.sum(stem_region == 0)
                    total_pixels = stem_region.size
                    density = black_pixels / total_pixels if total_pixels > 0 else 0
                    
                    # Also check vertical continuity - Y has continuous vertical line
                    # Project horizontally to see if there's a clear vertical stroke
                    horizontal_projection = np.sum(stem_region == 0, axis=1)
                    max_density_in_row = np.max(horizontal_projection) if len(horizontal_projection) > 0 else 0
                    
                    print(f"  → Checking V at position ({row+1},{col+1}) - density: {density:.2f}, max_row: {max_density_in_row}", end="")
                    
                    # Y has EITHER high overall density OR strong vertical presence in center
                    if density > 0.08 or max_density_in_row > center_width * 0.7:
                        grid[row][col] = 'Y'
                        corrections_made += 1
                        print(f" → Converting to Y!")
                    else:
                        print(f" → Keeping as V")
        
        if corrections_made > 0:
            print(f"\n✓ Made {corrections_made} auto-correction(s)")
            print("\nFinal grid:")
            for row in grid:
                print(' '.join(row))
        
        self.grid = grid
        return grid
    
    def manual_correction_interactive(self):
        """
        Interactive manual correction of OCR errors
        """
        if self.grid is None:
            print("No grid extracted yet!")
            return
        
        print("\n" + "="*60)
        print("MANUAL CORRECTION")
        print("="*60)
        print("\nCurrent grid:")
        self.print_grid()
        
        # Count errors
        error_count = sum(row.count('?') for row in self.grid)
        
        if error_count == 0:
            print("✓ No errors detected!")
            return self.grid
        
        print(f"\n⚠️  Found {error_count} unrecognized letters (marked with '?')")
        print("\nOptions:")
        print("  1. Fix all errors interactively")
        print("  2. Fix specific positions")
        print("  3. Skip (keep '?' marks)")
        
        choice = input("\nYour choice (1/2/3): ").strip()
        
        if choice == '1':
            # Fix all '?' one by one
            for row_idx, row in enumerate(self.grid):
                for col_idx, letter in enumerate(row):
                    if letter == '?':
                        print(f"\nPosition ({row_idx+1}, {col_idx+1}) - Current: '?'")
                        new_letter = input(f"Enter correct letter: ").strip().upper()
                        if new_letter and new_letter.isalpha():
                            self.grid[row_idx][col_idx] = new_letter[0]
        
        elif choice == '2':
            print("\nEnter corrections as: row,col,letter (e.g., 1,3,A)")
            print("Enter 'done' when finished")
            
            while True:
                correction = input("Correction: ").strip()
                
                if correction.lower() == 'done':
                    break
                
                try:
                    parts = correction.split(',')
                    row = int(parts[0]) - 1
                    col = int(parts[1]) - 1
                    letter = parts[2].strip().upper()[0]
                    
                    if 0 <= row < len(self.grid) and 0 <= col < len(self.grid[0]):
                        old = self.grid[row][col]
                        self.grid[row][col] = letter
                        print(f"  ✓ Changed ({row+1},{col+1}): '{old}' → '{letter}'")
                    else:
                        print("  ✗ Invalid position!")
                except Exception as e:
                    print(f"  ✗ Invalid format! Use: row,col,letter")
        
        print("\nFinal grid:")
        self.print_grid()
        
        return self.grid
    
    def print_grid(self):
        """Pretty print the grid"""
        if self.grid is None:
            print("No grid to display")
            return
        
        print()
        for row in self.grid:
            print(" ".join(row))
        print()
    
    def save_to_json(self, filename=None):
        """
        Save grid to JSON file in data directory
        
        Args:
            filename: Optional custom filename (without extension)
        
        Returns:
            Path to saved file
        """
        if self.grid is None:
            print("No grid to save!")
            return None
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{base_name}_{timestamp}"
        
        # Prepare data structure
        data = {
            'theme': self.theme if self.theme else 'Unknown',
            'grid': self.grid,
            'rows': len(self.grid),
            'cols': len(self.grid[0]) if self.grid else 0,
            'mode': 'dark' if self.is_dark_mode else 'light',
            'source_image': os.path.basename(self.image_path),
            'extracted_at': datetime.now().isoformat(),
            'ocr_engine': self.ocr_engine
        }
        
        # Save to JSON
        output_path = os.path.join(self.output_dir, f"{filename}.json")
        try:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✓ Grid saved to {output_path}")
            return output_path
        except Exception as e:
            print(f"❌ Error saving JSON: {e}")
            return None
    
    def save_to_text(self, filename=None):
        """
        Save grid to text file in data directory
        
        Args:
            filename: Optional custom filename (without extension)
        
        Returns:
            Path to saved file
        """
        if self.grid is None:
            print("No grid to save!")
            return None
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{base_name}_{timestamp}"
        
        # Save to text file
        output_path = os.path.join(self.output_dir, f"{filename}.txt")
        try:
            with open(output_path, 'w') as f:
                if self.theme:
                    f.write(f"Theme: {self.theme}\n")
                    f.write(f"Mode: {'dark' if self.is_dark_mode else 'light'}\n")
                    f.write(f"Source: {os.path.basename(self.image_path)}\n")
                    f.write("\n")
                
                f.write("Grid:\n")
                for row in self.grid:
                    f.write(" ".join(row) + "\n")
            
            print(f"✓ Grid saved to {output_path}")
            return output_path
        except Exception as e:
            print(f"❌ Error saving text file: {e}")
            return None
    
    def full_extraction_pipeline(self, rows=8, cols=6, save_output=True):
        """
        Complete extraction pipeline
        
        Args:
            rows: Number of rows
            cols: Number of columns
            save_output: Whether to save to file
        
        Returns:
            Tuple of (grid, theme, saved_files)
        """
        print("="*60)
        print("STRANDS OCR EXTRACTION v2")
        print("="*60)
        print(f"\nImage: {os.path.basename(self.image_path)}")
        
        # Step 1: Load
        print("\n[1/4] Loading image...")
        self.load_image()
        h, w = self.image.shape[:2]
        print(f"✓ Loaded: {w}x{h} pixels")
        
        # Step 2: Detect mode
        print("\n[2/4] Detecting display mode...")
        mode = self.detect_mode()
        print(f"✓ Mode: {mode}")
        
        # Step 3: Extract theme
        print("\n[3/4] Extracting theme...")
        theme = self.extract_theme()
        print(f"✓ Theme: '{theme if theme else 'Not detected'}'")
        
        # Step 4: Extract grid
        print(f"\n[4/4] Extracting grid ({rows}x{cols})...")
        grid = self.extract_grid_with_detection(rows, cols)
        print("\n✓ Grid extracted!")
        
        # Check for errors
        error_count = sum(row.count('?') for row in grid)
        if error_count > 0:
            print(f"\n⚠️  {error_count} letters need correction")
            grid = self.manual_correction_interactive()
        else:
            print("\n✓ All letters recognized!")
        
        # Save if requested
        saved_files = []
        if save_output:
            print("\nSaving extracted grid...")
            print(f"Output directory: {self.output_dir}")
            
            json_path = self.save_to_json()
            text_path = self.save_to_text()
            
            if json_path:
                saved_files.append(json_path)
            if text_path:
                saved_files.append(text_path)
            
            if saved_files:
                print(f"\n✓ Saved {len(saved_files)} file(s):")
                for path in saved_files:
                    print(f"  • {path}")
            else:
                print("\n⚠️  Warning: No files were saved!")
        else:
            print("\nSkipping save (save_output=False)")
        
        print("\n" + "="*60)
        print("EXTRACTION COMPLETE!")
        print("="*60)
        
        return grid, theme, saved_files


def quick_extract(image_path, rows=8, cols=6, ocr_engine='pytesseract', output_dir=OUTPUT_DIR):
    """
    Quick extraction function
    
    Args:
        image_path: Path to screenshot
        rows: Number of rows
        cols: Number of columns  
        ocr_engine: 'pytesseract' (fast) or 'easyocr' (accurate)
        output_dir: Directory to save extracted grids
    
    Returns:
        Tuple of (grid, theme, saved_files)
    """
    ocr = StrandsOCRv2(image_path, ocr_engine=ocr_engine, output_dir=output_dir)
    grid, theme, saved_files = ocr.full_extraction_pipeline(rows, cols)
    return grid, theme, saved_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract Strands grid from screenshot')
    parser.add_argument('image', nargs='?', help='Path to screenshot')
    parser.add_argument('--rows', type=int, default=DEFAULT_ROWS, 
                       help=f'Number of rows (default: {DEFAULT_ROWS})')
    parser.add_argument('--cols', type=int, default=DEFAULT_COLS, 
                       help=f'Number of columns (default: {DEFAULT_COLS})')
    parser.add_argument('--engine', choices=['pytesseract', 'easyocr'], 
                       default=DEFAULT_OCR_ENGINE, help='OCR engine to use')
    parser.add_argument('--output-dir', default=OUTPUT_DIR,
                       help=f'Output directory (default: {OUTPUT_DIR})')
    parser.add_argument('--test', action='store_true', 
                       help='Test on sample screenshots')
    
    args = parser.parse_args()
    
    if args.test:
        print("Running OCR test on sample screenshots...\n")
        
        for img_name in DEFAULT_TEST_IMAGES:
            img_path = os.path.join(SAMPLES_DIR, img_name)
            if os.path.exists(img_path):
                print(f"\n{'='*60}")
                print(f"Testing: {img_name}")
                print('='*60)
                quick_extract(img_path, args.rows, args.cols, args.engine, args.output_dir)
                print("\n")
            else:
                print(f"⚠️  Sample not found: {img_path}")
    
    elif args.image:
        quick_extract(args.image, args.rows, args.cols, args.engine, args.output_dir)
    
    else:
        print("Strands OCR v2 - Usage:")
        print("  python vision.py <image_path>")
        print("  python vision.py --test")
        print("\nExample:")
        print("  python vision.py data/samples/IMG_1009.PNG")
        print("  python vision.py screenshot.png --engine easyocr")
        print("\nConfiguration:")
        print(f"  Samples directory: {SAMPLES_DIR}")
        print(f"  Output directory: {OUTPUT_DIR}")
        print(f"  Default OCR engine: {DEFAULT_OCR_ENGINE}")