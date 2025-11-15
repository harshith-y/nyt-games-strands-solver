"""
Main entry point for Strands Solver
Just press "Run" in VSCode!
"""

from dotenv import load_dotenv
load_dotenv()  # This must come before using os.environ

import os
import json
from vision import StrandsOCRv2, quick_extract
from solver import solve_from_json, list_available_grids

# ============================================================================
# CONFIGURATION - EDIT THESE TO CHANGE BEHAVIOR
# ============================================================================

# Input method: "auto" (ask each time), "ocr" (always OCR), "manual" (always manual)
INPUT_METHOD = "auto"  # Change to "ocr" or "manual" to skip the prompt

# What do you want to do? Options: "extract", "solve", "both"
MODE = "both"  # "extract" = OCR only, "solve" = solve only, "both" = do both

IMAGE_TO_EXTRACT = "IMG_1009.PNG" 

# OCR engine: "pytesseract" (fast) or "easyocr" (accurate)
OCR_ENGINE = "easyocr"

# Grid dimensions
ROWS = 8
COLS = 6

# Auto-correct OCR errors? (True = interactive, False = skip)
AUTO_CORRECT = True

# Use LLM theme matching? (requires theme to be entered)
USE_THEME_MATCHING = True

# ============================================================================


def get_word_count_input():
    """
    Prompt user to enter the number of words in the puzzle
    
    Returns:
        Integer number of words (default 8 if not specified)
    """
    print("\n" + "="*70)
    print("🔢 NUMBER OF WORDS")
    print("="*70)
    print("\nHow many words are in this puzzle?")
    print("Typical Strands puzzles have: 7-9 words (6-8 theme words + 1 spangram)")
    print("💡 Check the puzzle - it usually shows the word count")
    
    while True:
        user_input = input("\nNumber of words (press Enter for default 8): ").strip()
        
        # Default to 8 if empty
        if not user_input:
            print("  ✓ Using default: 8 words")
            return 8
        
        # Validate it's a number
        try:
            word_count = int(user_input)
            if 5 <= word_count <= 12:  # Reasonable range
                print(f"  ✓ Word count set to: {word_count}")
                return word_count
            else:
                print("  ⚠️  Word count seems unusual (should be 5-12). Try again.")
        except ValueError:
            print("  ⚠️  Please enter a valid number.")


def get_theme_input():
    """
    Prompt user to enter the puzzle theme
    
    Returns:
        String containing the theme
    """
    print("\n" + "="*70)
    print("🎯 PUZZLE THEME")
    print("="*70)
    
    print("\nEnter the puzzle theme/category shown in the game")
    print("Examples: 'Movie Musicals', 'Wee wee wee!', 'Encuentra el ritmo'")
    print("Tip: The theme is usually shown at the top of the Strands puzzle")
    print("\n💡 Claude AI will interpret cryptic themes for you!")
    
    while True:
        theme = input("\nTheme: ").strip()
        if theme:
            print(f"  ✓ Theme set to: '{theme}'")
            return theme
        else:
            print("  ⚠️  Theme cannot be empty. Please enter the puzzle theme.")


def update_json_theme_and_count(json_path, theme, word_count):
    """
    Update the theme and word count in an existing JSON file
    
    Args:
        json_path: Path to the JSON file
        theme: New theme to set
        word_count: Number of words in puzzle
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    data['theme'] = theme
    data['word_count'] = word_count
    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)


def update_json_theme(json_path, theme):
    """
    Update the theme in an existing JSON file (backwards compatibility)
    
    Args:
        json_path: Path to the JSON file
        theme: New theme to set
    """
    update_json_theme_and_count(json_path, theme, 8)  # Default to 8 words


def manual_grid_entry_inline(image_name, rows=8, cols=6):
    """Quick manual entry integrated into main.py"""
    
    print("\n" + "="*70)
    print("📝 MANUAL GRID ENTRY")
    print("="*70)
    print(f"\nImage: {image_name}")
    print(f"Enter {rows}x{cols} grid - type each row of letters")
    print("Example: SWFOML or S W F O M L\n")
    
    grid = []
    
    for row_num in range(1, rows + 1):
        while True:
            user_input = input(f"Row {row_num}: ").strip().upper()
            letters = [c for c in user_input if c.isalpha()]
            
            if len(letters) == cols:
                grid.append(letters)
                print(f"  ✓ {' '.join(letters)}")
                break
            else:
                print(f"  ⚠️  Expected {cols} letters, got {len(letters)}. Try again.")
    
    # Show final grid
    print("\n" + "="*70)
    print("FINAL GRID")
    print("="*70)
    for row in grid:
        print(' '.join(row))
    
    # Get theme from user
    theme = get_theme_input()
    
    # Get word count from user
    word_count = get_word_count_input()
    
    # Save to file
    output_dir = "data/inputs"
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_name))[0]
    output_file = f"{output_dir}/{base_name}_grid.json"
    
    data = {
        'grid': grid,
        'rows': rows,
        'cols': cols,
        'source': image_name,
        'method': 'manual',
        'theme': theme,
        'word_count': word_count
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✓ Grid saved to: {output_file}")
    
    return output_file


def main():
    """Main function - runs when you press Play in VSCode"""
    
    print("="*70)
    print("🎮 STRANDS SOLVER")
    print("="*70)
    
    # Check if theme matching is enabled
    if USE_THEME_MATCHING:
        print("\n💡 Theme matching is ENABLED (uses Claude API)")
        
        # Check if API key is set
        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            print(f"   ✓ API key is set ({api_key[:15]}...)")
        else:
            print("   ⚠️  API key NOT set!")
            print("   Set it: export ANTHROPIC_API_KEY='sk-ant-your-key-here'")
            print("   Get key from: https://console.anthropic.com/")
    else:
        print("\n💡 Theme matching is DISABLED")
        print("   You'll see all words without AI filtering")
    
    # Build full path to image
    image_path = os.path.join("data", "samples", IMAGE_TO_EXTRACT)
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"\n❌ Error: Image not found at {image_path}")
        print("\nAvailable images in data/samples/:")
        samples_dir = "data/samples"
        if os.path.exists(samples_dir):
            images = [f for f in os.listdir(samples_dir) 
                     if f.endswith(('.png', '.PNG', '.jpg', '.jpeg', '.JPG'))]
            if images:
                for img in images:
                    print(f"  • {img}")
                print(f"\n💡 Tip: Change IMAGE_TO_EXTRACT to one of these names")
            else:
                print("  (No images found)")
        else:
            print(f"  (Directory {samples_dir} not found)")
        return
    
    # ========================================================================
    # CHOOSE INPUT METHOD
    # ========================================================================
    
    # Determine input method
    use_method = INPUT_METHOD
    
    if INPUT_METHOD == "auto":
        print(f"\nImage: {IMAGE_TO_EXTRACT}")
        print("\n📋 Choose input method:")
        print("  1. 🤖 OCR (Automatic) - Fast, ~95-100% accurate")
        print("  2. ✏️  Manual Entry - 100% accurate, takes 30 seconds")
        
        while True:
            choice = input("\nYour choice (1 or 2): ").strip()
            if choice in ['1', '2']:
                use_method = "ocr" if choice == '1' else "manual"
                break
            else:
                print("  ⚠️  Please enter 1 or 2")
    
    # ========================================================================
    # EXTRACT GRID
    # ========================================================================
    if MODE in ["extract", "both"]:
        
        if use_method == "manual":
            # Manual entry (includes theme input)
            print("\n" + "="*70)
            print("STEP 1: MANUAL GRID ENTRY")
            print("="*70)
            
            latest_json = manual_grid_entry_inline(image_path, ROWS, COLS)
            
        else:
            # OCR extraction
            print("\n" + "="*70)
            print("STEP 1: EXTRACTING GRID FROM SCREENSHOT (OCR)")
            print("="*70)
            print(f"OCR Engine: {OCR_ENGINE}")
            
            try:
                grid, theme, saved_files = quick_extract(
                    image_path, 
                    rows=ROWS, 
                    cols=COLS, 
                    ocr_engine=OCR_ENGINE,
                    output_dir="data/inputs"
                )
                
                if saved_files:
                    print(f"\n✅ Extraction complete!")
                    print(f"   Saved {len(saved_files)} file(s)")
                    latest_json = [f for f in saved_files if f.endswith('.json')][0]
                    
                    # Now prompt for theme and word count, then update the JSON
                    theme = get_theme_input()
                    word_count = get_word_count_input()
                    update_json_theme_and_count(latest_json, theme, word_count)
                    
                else:
                    print(f"\n⚠️  No files were saved (but extraction completed)")
                    return
                    
            except Exception as e:
                print(f"\n❌ Error during extraction: {e}")
                import traceback
                traceback.print_exc()
                return
    
    # ========================================================================
    # SOLVE PUZZLE
    # ========================================================================
    if MODE in ["solve", "both"]:
        print("\n" + "="*70)
        print("STEP 2: SOLVING PUZZLE")
        print("="*70)
        
        # Find the JSON file to solve
        if MODE == "both":
            # Use the file we just created
            json_file = latest_json
        else:
            # Find most recent JSON in data/inputs
            grids = list_available_grids("data/inputs")
            if not grids:
                print("\n❌ No extracted grids found in data/inputs/")
                print("   Run in 'extract' or 'both' mode first!")
                return
            json_file = grids[-1]  # Most recent
            print(f"\nUsing most recent grid: {os.path.basename(json_file)}")
        
        try:
            solver = solve_from_json(
                json_file, 
                dictionary_path="wordlist.txt",
                use_theme_matching=USE_THEME_MATCHING
            )
            
            if solver and solver.found_words:
                print("\n" + "="*70)
                print("✅ COMPLETE!")
                print("="*70)
                
                theme = solver.puzzle_info.get('theme', 'Unknown')
                
                print(f"\n📊 Summary:")
                print(f"   • Theme: {theme}")
                print(f"   • Total words found: {len(solver.found_words)}")
                
                # Show spangrams
                spangrams = solver.find_spangrams()
                if spangrams:
                    print(f"   • Total spangrams: {len(spangrams)}")
                    print(f"     {', '.join(spangrams.keys())}")
                
                # Show theme-matched results
                if USE_THEME_MATCHING and solver.theme_matched_words:
                    print(f"   • Theme-relevant words: {len(solver.theme_matched_words)}")
                    
                    # Show theme spangrams
                    theme_spangrams = solver.find_theme_spangrams()
                    if theme_spangrams:
                        print(f"   • Theme spangrams: {', '.join(theme_spangrams.keys())}")
                    
                    print(f"\n💡 Top theme matches:")
                    sorted_matches = sorted(
                        solver.theme_matched_words.items(),
                        key=lambda x: x[1].get('relevance', 0),
                        reverse=True
                    )
                    for word, info in sorted_matches[:10]:
                        score = info.get('relevance', 0)
                        is_spangram = word in spangrams
                        marker = " 🎯 SPANGRAM" if is_spangram else ""
                        print(f"     • {word.upper():<15} ({score:.0%}){marker}")
                
        except Exception as e:
            print(f"\n❌ Error during solving: {e}")
            import traceback
            traceback.print_exc()
            return
    
    print("\n" + "="*70)
    print("🎉 ALL DONE!")
    print("="*70)
    print("\n💡 To change settings, edit the configuration at the top of main.py")


if __name__ == "__main__":
    main()