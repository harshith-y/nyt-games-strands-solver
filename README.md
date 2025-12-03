# NYT Strands Solver

Automated solver for NYT Strands puzzles using custom OCR, semantic reasoning, and constraint satisfaction.

**Read the full story**: [Solving NYT Strands: The Suffering Continues... With Words This Time](https://medium.com/@hyerraguntla/solving-nyt-strands-the-suffering-continues-with-words-this-time-79e87acb56d1)

## Installation

```bash
# Clone the repository
git clone https://github.com/harshith-y/strands-solver.git
cd strands-solver

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up Claude API key
export ANTHROPIC_API_KEY='your-api-key-here'
# Or create a .env file:
echo "ANTHROPIC_API_KEY=your-api-key-here" > .env
```

## Quick Start

### 1. Extract Grid from Screenshot

Take a screenshot of the NYT Strands puzzle and run:

```bash
python main.py
```

Choose **OCR (Automatic)** when prompted.

**Input**: PNG/JPG screenshot of the puzzle  
**Output**: JSON file in `data/inputs/` directory with extracted grid

Example output structure:
```json
{
  "grid": [
    ["M", "U", "J", "E", "R", "N"],
    ["I", "P", "S", "E", "T", "A"],
    ...
  ],
  "theme": "On the web",
  "word_count": 7,
  "rows": 8,
  "cols": 6
}
```

### 2. Verify and Solve

The solver runs automatically after extraction:
- Verifies grid (fix any OCR errors)
- Enter theme and word count
- Finds theme words and spangram
- Places all words on grid

Example solution:
```
✅ SOLUTION FOUND!

Theme words placed (7):
  1. JUMPING (7 cells) - Jumping spider
  2. RECLUSE (7 cells) - Brown recluse
  3. TARANTULA (9 cells) - Tarantula
  4. WIDOW (5 cells) - Black widow
  5. HUNTSMAN (8 cells) - Huntsman spider
  6. HOUSE (5 cells) - House spider
  7. SPIDERS (7 cells) - Spangram

Total: 48/48 cells (100%)
```

## Usage Examples

### Manual Grid Entry

If OCR fails or you want direct control:

```bash
python main.py
# Choose option 2 (Manual Entry)
# Enter each row when prompted
```

### Solve from Existing JSON

```bash
python solver.py data/inputs/IMG_1275.json "On the web" 7
```

### Batch Process Multiple Screenshots

```bash
# Place screenshots in project directory
for img in IMG_*.PNG; do
    python main.py --image "$img"
done
```

## Configuration

### Main Pipeline Settings

Edit flags in `main.py`:

```python
INPUT_METHOD = "auto"        # "auto", "ocr", or "manual"
MODE = "both"                # "extract", "solve", or "both"
OCR_ENGINE = "easyocr"       # "pytesseract" or "easyocr"
RECOGNIZER_MODE = "template" # "ocr", "template", or "hybrid"
USE_THEME_MATCHING = True    # Use Claude API for theme matching
```

### Solver Options

Edit parameters in `solver.py`:

```python
MAX_PATH_COMBOS_PER_COMBO = 7000  # Max combinations to try
CONFIDENCE_THRESHOLD = 6.0         # Min theme interpretation score
MAX_INTERPRETATIONS = 9            # Theme interpretations to try
```

## Troubleshooting

### Vision Pipeline Issues

**Problem**: Letters misrecognized (I vs J, O vs Q)
- Use interactive verification to correct errors
- System will save corrections automatically
- Template library improves with more labeled data

**Problem**: Theme not detected
- Enter theme manually when prompted
- Theme detection is optional, manual entry always works

### Solver Issues

**Problem**: No solution found
- Try different word count (±1 from expected)
- Some themes require multiple interpretation attempts
- Multi-word spangrams may not be detected

**Problem**: Wrong words selected
- Verify theme is entered correctly (punctuation matters)
- Some cryptic themes may need manual word selection
- API may prefer scientific terms over common words

**Problem**: Solver takes too long
- Hit Ctrl+C to stop
- Most puzzles solve in 10-30 seconds
- Complex themes with 9+ words may take 1-2 minutes

## Project Structure

```
strands-solver/
├── main.py                 # Main entry point
├── vision.py              # OCR and grid extraction
├── solver.py              # Core solver (DFS + Trie)
├── spangram_finder.py     # Spangram identification
├── wordlist.txt           # Dictionary (370k words)
├── templates/             # Letter templates for OCR
│   └── letter_templates_v1.npz
├── data/
│   ├── inputs/           # Extracted grids (JSON)
│   └── outputs/          # Debug files
└── .env                  # API key (create this)
```

## How It Works

**Vision Pipeline**: Custom template-based OCR using cosine similarity against 2,352 labeled letter samples. Achieves 97-99% accuracy on Franklin Gothic font. Interactive verification catches remaining errors.

**Word Search**: DFS with Trie (prefix tree) pruning. Searches 370k-word dictionary, finds ~1,500-2,000 valid words in 1-2 seconds through early termination of invalid prefixes.

**Theme Matching**: Claude API evaluates multiple semantic interpretations, scores each 0-10, selects words from actual grid (no hallucinations). Shows 150 real words to API for selection.

**Geometric Solver**: Places theme words on 8×6 grid without overlapping. Prunes impossible combinations, fills leftover regions with API assistance, validates spangram across opposite edges.

## Performance

- **OCR Accuracy**: 97-99%
- **Word Search**: 1-2 seconds for full grid
- **Theme Matching**: Variable (depends on clarity)
- **Success Rate**: High on single-word spangrams, lower on multi-word

## Limitations

- Multi-word spangrams (e.g., IN + IT + TOGETHER) not reliably detected
- Very cryptic themes may require manual word selection
- High word counts (9+) increase combinatorial complexity
- Overlapping word families (CIRCLE/INCIRCLE) can cause failures

## Requirements

- Python 3.8+
- Claude API key (Anthropic)
- Dependencies in `requirements.txt`:
  - `anthropic` - Claude API client
  - `easyocr` - OCR engine
  - `opencv-python` - Image processing
  - `numpy` - Array operations
  - `python-dotenv` - Environment variables
  - `Pillow` - Image handling

## Acknowledgments

This solver is for educational purposes. Support the NYT by playing their puzzles at [nytimes.com/puzzles](https://www.nytimes.com/puzzles).

Special thanks to Anthropic for Claude API and the Franklin Gothic typeface for being both clear and impossible to read.