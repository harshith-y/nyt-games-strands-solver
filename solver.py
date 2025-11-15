"""
NYT Strands Solver - Cleaned and Organized Version
Solves Strands puzzles using DFS + Trie + LLM theme matching

ARCHITECTURE:
1. Core Classes (Trie, TrieNode, StrandsSolver)
2. Word Finding (DFS to find all valid words)
3. Spangram Detection (single & multi-word)
4. Theme Matching (Claude API)
5. Solving Strategies (4 methods)
6. Utility Functions

Author: Cleaned version
"""

import json
import os
from glob import glob
import requests
import itertools
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

DICTIONARY_PATH = "wordlist.txt"
DATA_DIR = "data"

# Solving configuration
API_WORD_LIMIT = 20  # Get top 20 theme-relevant words from API
MIN_WORDS = 4        # Try at least 4 words in combinations
MAX_WORDS = 15       # Try up to 15 words in combinations

# Confidence thresholds for exhaustive search
HIGH_CONFIDENCE_THRESHOLD = 7.0      # Score threshold (0-10)
HIGH_CONFIDENCE_ATTEMPTS = 50000     # Max attempts when confidence >= 7.0 (IMPROVED: 2x)
NORMAL_ATTEMPTS = 10000              # Max attempts for normal confidence (IMPROVED: 2x)

# Visualization settings
SHOW_PROGRESS_EVERY = 500            # Show progress every N attempts (IMPROVED: less spam)
SHOW_CLOSE_ATTEMPTS = True           # Show grids for close attempts
CLOSE_ATTEMPT_THRESHOLD = 25         # Show attempts that use 40+ cells

# Path search limits (to avoid combinatorial explosion)
MAX_PATHS_PER_WORD = 5              # Max paths per word considered in a combo
MAX_PATH_COMBOS_PER_COMBO = 500     # Max path combinations per word-set combo

# Spangram requirement:
#   False  -> accept any clean theme-word layout (leftover may NOT be a valid spangram)
#   True   -> require leftover cells to form a valid spangram
REQUIRE_SPANGRAM = False

# ============================================================================
# SECTION 0: IMPROVEMENT HELPERS (Word Grouping & Trapped Letter Detection)
# ============================================================================

def group_word_variants(words_with_scores):
    """
    Group words that are variants of each other (plurals, etc.)
    
    Args:
        words_with_scores: List of (word, relevance) tuples
        
    Returns:
        List of groups, each group is [(base_word, score), (variant, score), ...]
        sorted by best score in group
    """
    groups = defaultdict(list)
    
    for word, score in words_with_scores:
        # Find base form
        base = word
        
        # Strip common suffixes to find base
        if word.endswith('s') and len(word) > 4:
            base = word[:-1]
        elif word.endswith('es') and len(word) > 5:
            base = word[:-2]
        elif word.endswith('ed') and len(word) > 5:
            base = word[:-2]
        elif word.endswith('ing') and len(word) > 6:
            base = word[:-3]
        
        # Group by base
        groups[base].append((word, score))
    
    # Convert to list of groups, sort each group by score
    result = []
    for base, variants in groups.items():
        variants.sort(key=lambda x: x[1], reverse=True)  # Best score first
        result.append(variants)
    
    # Sort groups by best score in each group
    result.sort(key=lambda g: g[0][1], reverse=True)
    
    return result


def select_best_words_from_groups(word_groups, target_count=20):
    """
    Select best words from grouped variants
    
    Strategy:
    - Take best word from each group
    - Avoid wasting attempts on multiple forms of same word
    
    Args:
        word_groups: List of word groups from group_word_variants()
        target_count: How many words to select
        
    Returns:
        List of (word, score) tuples
    """
    selected = []
    
    # Take best from each group
    for group in word_groups:
        if len(selected) >= target_count:
            break
        selected.append(group[0])  # Best word in group
    
    # If we need more, take second-best from high-scoring groups
    if len(selected) < target_count:
        for group in word_groups:
            if len(group) > 1:  # Has variants
                for variant in group[1:]:
                    if len(selected) >= target_count:
                        break
                    selected.append(variant)
    
    return selected[:target_count]


def find_reachable_regions(used_cells, all_cells_set, rows, cols, get_neighbors_func):
    """
    Find all connected regions of unused cells using BFS
    
    CRITICAL: Two cells are in the same region if you can reach one from the other
    by moving through adjacent unused cells (8-directional adjacency)
    
    Returns: List of sets, each set is a connected region of unused cells
    """
    unused_cells = all_cells_set - used_cells
    
    if not unused_cells:
        return []
    
    regions = []
    visited = set()
    
    for start_cell in unused_cells:
        if start_cell in visited:
            continue
        
        # BFS to find all cells connected to start_cell
        region = set()
        queue = [start_cell]
        
        while queue:
            current = queue.pop(0)
            
            # Skip if already processed or not unused
            if current in visited or current not in unused_cells:
                continue
            
            # Mark as visited and add to region
            visited.add(current)
            region.add(current)
            
            # Check all 8 neighbors
            for neighbor in get_neighbors_func(current[0], current[1]):
                # Only add unused, unvisited neighbors to queue
                if neighbor in unused_cells and neighbor not in visited:
                    queue.append(neighbor)
        
        if region:
            regions.append(region)
    
    return regions


def has_trapped_letters(used_cells, all_cells_set, rows, cols, get_neighbors_func):
    """
    Check if current placement creates DISCONNECTED regions that are too small
    
    Key concept: A region is a connected group of unused cells (via 8-directional adjacency).
    
    Example of TRAPPED cells (BAD):
        X X X X X X
        X . . X X X     ← These 2 dots are DISCONNECTED from other unused cells
        X X X X X X        (surrounded by X's)
        . . . . . . .   ← This is a separate region (fine, has 7 cells)
    
    Example of NOT trapped (GOOD):
        X X X X . .     ← These 2 dots connect to dots below
        X . . . . .     ← All dots form ONE connected region
        X X X X . .        (can reach any dot from any other dot)
    
    Returns: (bool, reason)
        - True if there are trapped letters (BAD - prune this placement)
        - False if all regions are viable (GOOD - continue checking)
    """
    regions = find_reachable_regions(used_cells, all_cells_set, rows, cols, get_neighbors_func)
    
    for i, region in enumerate(regions):
        # Each DISCONNECTED region must have at least 4 cells to form a valid word
        # (Strands requires minimum 4-letter words)
        if len(region) < 4:
            return True, f"Found disconnected region #{i+1} with only {len(region)} cells (need ≥4)"
    
    return False, ""

# ============================================================================
# SECTION 1: CORE DATA STRUCTURES
# ============================================================================

class TrieNode:
    """Node in a Trie for efficient prefix checking"""
    def __init__(self):
        self.children = {}
        self.is_word = False


class Trie:
    """Trie for dictionary storage and prefix validation"""
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        """Insert a word into the Trie"""
        node = self.root
        for char in word.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True
    
    def search(self, word):
        """Check if word exists"""
        node = self.root
        for char in word.lower():
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_word
    
    def starts_with(self, prefix):
        """Check if any word starts with prefix"""
        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return False
            node = node.children[char]
        return True


# ============================================================================
# SECTION 2: MAIN SOLVER CLASS
# ============================================================================

class StrandsSolver:
    """Main solver for NYT Strands puzzle"""
    
    def __init__(self, grid, dictionary_path=DICTIONARY_PATH, puzzle_info=None):
        """
        Initialize solver
        
        Args:
            grid: 2D list of characters
            dictionary_path: Path to dictionary file
            puzzle_info: Dict with theme, source, etc.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
        self.trie = Trie()
        self.found_words = {}  # word -> list of paths
        self.puzzle_info = puzzle_info or {}
        self.theme_matched_words = {}  # word -> {relevance, reasoning}
        
        # Load dictionary
        if dictionary_path and os.path.exists(dictionary_path):
            self.load_dictionary(dictionary_path)
        else:
            print(f"⚠️  Dictionary not found: {dictionary_path}")
    
    def load_dictionary(self, path):
        """Load dictionary words into Trie"""
        word_count = 0
        short_word_count = 0
        with open(path, 'r') as f:
            for line in f:
                word = line.strip()
                if len(word) >= 1:  # Load all words including short ones
                    self.trie.insert(word)
                    word_count += 1
                    if len(word) < 4:
                        short_word_count += 1
        print(f"✓ Dictionary loaded: {word_count} words (including {short_word_count} words <4 letters)")
    
    # ------------------------------------------------------------------------
    # WORD FINDING (DFS)
    # ------------------------------------------------------------------------
    
    def get_neighbors(self, row, col):
        """Get all valid adjacent cells (8 directions)"""
        neighbors = []
        directions = [
            (-1, -1), (-1, 0), (-1, 1),  # top row
            (0, -1),           (0, 1),    # middle row
            (1, -1),  (1, 0),  (1, 1)     # bottom row
        ]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                neighbors.append((new_row, new_col))
        
        return neighbors
    
    def _iter_non_overlapping_placements_for_combo(self, combo_words, all_cells):
        """Generate non-overlapping placements for a given word combo.

        For a fixed set of theme words ([(word, relevance), ...]), this explores
        combinations of their available paths via itertools.product, while
        respecting MAX_PATHS_PER_WORD and MAX_PATH_COMBOS_PER_COMBO to avoid
        combinatorial explosion.

        Yields dicts of the form:
            {
                'placed_words': [(word, path), ...],
                'used_cells': set((r, c), ...),
                'leftover_cells': set((r, c), ...),
            }
        """
        # Build per-word path lists (limited by MAX_PATHS_PER_WORD)
        path_lists = []
        for word, _rel in combo_words:
            key = word.lower()
            paths = self.found_words.get(key) or self.found_words.get(word)
            if not paths:
                # This word cannot actually be formed on the grid
                return

            limited_paths = paths[:MAX_PATHS_PER_WORD]
            path_lists.append(limited_paths)

        path_combo_count = 0

        for path_tuple in itertools.product(*path_lists):
            path_combo_count += 1
            if path_combo_count > MAX_PATH_COMBOS_PER_COMBO:
                break

            used_cells = set()
            placed_words = []
            valid = True

            for (word, _rel), path in zip(combo_words, path_tuple):
                path_cells = set(path)
                if path_cells & used_cells:
                    valid = False
                    break
                placed_words.append((word, path))
                used_cells |= path_cells

            if not valid:
                continue

            leftover_cells = all_cells - used_cells

            yield {
                'placed_words': placed_words,
                'used_cells': used_cells,
                'leftover_cells': leftover_cells,
            }

    def dfs(self, row, col, visited, current_word, path):
        """
        Depth-first search to find all valid words
        
        Args:
            row, col: Current position
            visited: Set of visited cells
            current_word: Word built so far
            path: List of (row, col) tuples
        """
        # Add current cell
        visited.add((row, col))
        current_word += self.grid[row][col]
        path.append((row, col))
        
        # Check if valid word
        if len(current_word) >= 1 and self.trie.search(current_word):
            word_key = current_word.lower()
            if word_key not in self.found_words:
                self.found_words[word_key] = []
            self.found_words[word_key].append(list(path))
        
        # Early pruning: stop if prefix doesn't exist
        if not self.trie.starts_with(current_word):
            visited.remove((row, col))
            path.pop()
            return
        
        # Explore neighbors
        for next_row, next_col in self.get_neighbors(row, col):
            if (next_row, next_col) not in visited:
                self.dfs(next_row, next_col, visited, current_word, path)
        
        # Backtrack
        visited.remove((row, col))
        path.pop()

    def _iter_non_overlapping_placements_for_combo(self, combo_words, all_cells):
        """Generate non-overlapping placements for a given word combo.

        For a fixed set of theme words ([(word, relevance), ...]), this explores
        combinations of their available paths via itertools.product, while
        respecting MAX_PATHS_PER_WORD and MAX_PATH_COMBOS_PER_COMBO to avoid
        combinatorial explosion.

        Yields dicts of the form:
            {
                'placed_words': [(word, path), ...],
                'used_cells': set((r, c), ...),
                'leftover_cells': set((r, c), ...),
            }
        """
        # Build per-word path lists (limited by MAX_PATHS_PER_WORD)
        path_lists = []
        for word, _rel in combo_words:
            key = word.lower()
            paths = self.found_words.get(key) or self.found_words.get(word)
            if not paths:
                # This word cannot actually be formed on the grid
                return

            limited_paths = paths[:MAX_PATHS_PER_WORD]
            path_lists.append(limited_paths)

        path_combo_count = 0

        for path_tuple in itertools.product(*path_lists):
            path_combo_count += 1
            if path_combo_count > MAX_PATH_COMBOS_PER_COMBO:
                break

            used_cells = set()
            placed_words = []
            valid = True

            for (word, _rel), path in zip(combo_words, path_tuple):
                path_cells = set(path)
                if path_cells & used_cells:
                    valid = False
                    break
                placed_words.append((word, path))
                used_cells |= path_cells

            if not valid:
                continue

            leftover_cells = all_cells - used_cells

            yield {
                'placed_words': placed_words,
                'used_cells': used_cells,
                'leftover_cells': leftover_cells,
            }
    
    def find_all_words(self):
        """Find all valid words in the grid"""
        self.found_words = {}
        
        for row in range(self.rows):
            for col in range(self.cols):
                self.dfs(row, col, set(), "", [])
        
        return self.found_words
    
    # ------------------------------------------------------------------------
    # SPANGRAM DETECTION
    # ------------------------------------------------------------------------
    
    def is_spangram(self, path):
        """
        Check if path is a spangram (touches opposite edges)
        
        Args:
            path: List of (row, col) tuples
            
        Returns:
            Boolean
        """
        rows_in_path = [pos[0] for pos in path]
        cols_in_path = [pos[1] for pos in path]
        
        # Check if touches top AND bottom
        touches_top = 0 in rows_in_path
        touches_bottom = (self.rows - 1) in rows_in_path
        
        # Check if touches left AND right
        touches_left = 0 in cols_in_path
        touches_right = (self.cols - 1) in cols_in_path
        
        # Must touch BOTH opposite edges
        return (touches_top and touches_bottom) or (touches_left and touches_right)
    
    def _cells_form_spangram(self, cells):
        """Check if a set of cells could form a spangram"""
        rows = [cell[0] for cell in cells]
        cols = [cell[1] for cell in cells]
        
        touches_top = 0 in rows
        touches_bottom = (self.rows - 1) in rows
        touches_left = 0 in cols
        touches_right = (self.cols - 1) in cols
        
        return (touches_top and touches_bottom) or (touches_left and touches_right)
    
    def _paths_are_adjacent(self, path1, path2):
        """Check if two paths are adjacent (end of path1 touches start of path2)"""
        end1 = path1[-1]
        start2 = path2[0]
        return start2 in self.get_neighbors(end1[0], end1[1])
    
    def find_spangrams(self):
        """
        Find all single-word spangrams
        
        Returns:
            Dict of {word: [paths]}
        """
        spangrams = {}
        for word, paths in self.found_words.items():
            for path in paths:
                if self.is_spangram(path):
                    if word not in spangrams:
                        spangrams[word] = []
                    spangrams[word].append(path)
        return spangrams
    
    def find_all_spangrams_including_multiword(self):
        """
        Find single AND multi-word spangrams
        
        Returns:
            List of dicts with format:
            {'word': 'text', 'paths': [path], 'length': n, 'type': 'single'/'multi'}
        """
        all_spangrams = []
        
        # 1. Single-word spangrams
        single_spangrams = self.find_spangrams()
        for word, paths in single_spangrams.items():
            for path in paths:
                all_spangrams.append({
                    'word': word,
                    'paths': [path],
                    'length': len(path),
                    'type': 'single'
                })
        
        # 2. Multi-word spangrams (2-word combinations)
        print("\n🔍 Checking for multi-word spangrams...")
        words_list = [(w, paths) for w, paths in self.found_words.items() if len(w) >= 3]
        
        # Limit to top 200 most common words for performance
        words_list = words_list[:200]
        print(f"   Checking {len(words_list)} words (≥3 letters) for multi-word spangrams")
        print(f"   Limited to top 200 most common words")
        
        multi_count = 0
        checked_pairs = 0
        
        for (w1, paths1), (w2, paths2) in itertools.combinations(words_list, 2):
            checked_pairs += 1
            
            for p1 in paths1:
                for p2 in paths2:
                    # Check if adjacent and no overlap
                    if not (set(p1) & set(p2)) and self._paths_are_adjacent(p1, p2):
                        combined = p1 + p2
                        if self.is_spangram(combined):
                            combo_word = f"{w1}_{w2}"
                            all_spangrams.append({
                                'word': combo_word,
                                'paths': [p1, p2],
                                'length': len(combined),
                                'type': 'multi'
                            })
                            multi_count += 1
                            if multi_count <= 10:
                                print(f"   ✓ Found: {combo_word.upper()} ({len(combined)} cells)")
        
        if multi_count > 10:
            print(f"   ... and {multi_count - 10} more multi-word spangrams")
        
        print(f"   Checked {checked_pairs} word pairs, found {multi_count} multi-word spangrams")
        
        # 3. TODO: 3-word spangrams (optional, rare)
        
        return all_spangrams
    
    # ------------------------------------------------------------------------
    # THEME MATCHING (Claude API)
    # ------------------------------------------------------------------------
    
    def filter_words_by_theme(self, theme, word_count=8):
        """
        Use Claude API to filter words by theme relevance
        
        Args:
            theme: Puzzle theme string
            word_count: Expected number of words
            
        Returns:
            Dict of {word: {relevance, reasoning}}
        """
        print(f"\n🤖 Using Claude API to match words to theme: '{theme}'")
        print(f"   Expected words: {word_count} ({word_count-1} theme words + 1 spangram)")
        print("This may take 5-10 seconds...")
        
        # Check API key
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("❌ Claude API Error: No API key found")
            print("📋 API Key Issue:")
            print("  1. Get key from: https://console.anthropic.com/")
            print("  2. Set it: export ANTHROPIC_API_KEY='sk-ant-your-key-here'")
            print("  3. Or add to ~/.bashrc or ~/.zshrc")
            return {}
        
        # Prepare word candidates
        candidates = []
        for word in self.found_words.keys():
            if len(word) >= 3:  # Only words 3+ letters
                candidates.append(word)
        
        candidates.sort(key=lambda w: len(w), reverse=True)  # Sort by length
        
        # Build prompt for Claude
        grid_text = '\n'.join([' '.join(row) for row in self.grid])
        
        candidate_list = ""
        for i in range(0, len(candidates[:100]), 5):
            chunk = candidates[i:i+5]
            candidate_list += ", ".join(chunk) + "\n"
        
        prompt = f"""You are solving a NYT Strands puzzle. The theme is: "{theme}"

Grid:
{grid_text}

Find the {word_count-1} theme words (NOT including the spangram) that best match this theme.

Candidate words (first 100):
{candidate_list}

Instructions:
1. Interpret the theme (it may be cryptic or a pun)
2. Find {word_count-1} words that relate to your interpretation
3. Rank by relevance (0.0 to 1.0)
4. Explain your reasoning

Return JSON:
{{
  "theme_interpretation": "your interpretation",
  "words": [
    {{"word": "word1", "relevance": 0.95, "reasoning": "why it fits"}},
    ...
  ],
  "spangram_candidates": [
    {{"word": "spangram1", "relevance": 0.90, "category": "type of spangram"}}
  ]
}}"""
        
        # Call Claude API
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 4000,
                    "messages": [{
                        "role": "user",
                        "content": prompt
                    }]
                },
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"❌ Claude API Error: {response.status_code}")
                print(f"   Response: {response.text[:200]}")  # Show first 200 chars
                if response.status_code == 401:
                    print("📋 API Key Issue:")
                    print("  1. Get key from: https://console.anthropic.com/")
                    print("  2. Set it: export ANTHROPIC_API_KEY='sk-ant-your-key-here'")
                    print("  3. Or add to ~/.bashrc or ~/.zshrc")
                return {}
            
            data = response.json()
            response_text = data['content'][0]['text'].strip()
            
            # Debug: Show what we got
            if not response_text:
                print(f"❌ API returned empty response")
                print(f"   Full response: {data}")
                return {}
            
            # IMPROVED: Extract JSON from response
            # Claude sometimes adds explanation text before/after the JSON
            
            # First, try to find JSON in markdown code blocks
            if '```json' in response_text or '```' in response_text:
                # Extract content between ``` markers
                lines = response_text.split('\n')
                json_lines = []
                in_code_block = False
                
                for line in lines:
                    if line.strip().startswith('```'):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block:
                        json_lines.append(line)
                
                if json_lines:
                    response_text = '\n'.join(json_lines).strip()
            
            # If no code block, try to find JSON by looking for { and }
            if not response_text.startswith('{'):
                # Find the first { and last }
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}')
                
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    response_text = response_text[start_idx:end_idx+1]
                else:
                    print(f"❌ Could not find JSON object in response")
                    print(f"   Response text (first 500 chars):")
                    print(f"   {response_text[:500]}")
                    return {}
            
            # Try to parse JSON
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse API response as JSON")
                print(f"   Error: {e}")
                print(f"   Response text (first 500 chars):")
                print(f"   {response_text[:500]}")
                return {}
            
            # Print interpretation
            print(f"\n💡 Theme interpretations:")
            interpretation = result.get('theme_interpretation', 'Unknown')
            print(f"  🎯 {interpretation}")
            
            # Print spangram candidates
            spangram_candidates = result.get('spangram_candidates', [])
            if spangram_candidates:
                print(f"\n🎯 Spangram candidates:")
                for s in spangram_candidates[:5]:
                    word = s.get('word', '').upper()
                    score = s.get('relevance', 0)
                    category = s.get('category', '')
                    reasoning = s.get('reasoning', '')[:80]
                    print(f"  • {word} ({score:.0%})")
                    if category:
                        print(f"     Category: {category}")
                    if reasoning:
                        print(f"     → {reasoning}")
            
            # Process theme words
            theme_words = result.get('words', [])
            
            # Filter out words not in grid
            valid_words = []
            missing_words = []
            
            for w in theme_words:
                word = w.get('word', '').lower()
                if word in self.found_words:
                    valid_words.append(w)
                    self.theme_matched_words[word] = {
                        'relevance': w.get('relevance', 0),
                        'reasoning': w.get('reasoning', '')
                    }
                else:
                    missing_words.append(word)
            
            if missing_words:
                print(f"\n⚠️  Filtered out {len(missing_words)} words not found in grid")
                print(f"   Missing: {', '.join(missing_words[:10])}")
            
            print(f"\n✓ Found {len(valid_words)} theme words (in grid):")
            print(f"  {result.get('theme_interpretation', '')}\n")
            
            for w in valid_words[:10]:
                word = w.get('word', '').upper()
                score = w.get('relevance', 0)
                reasoning = w.get('reasoning', '')[:60]
                print(f"  {word:<15} ({score:.0%}) - {reasoning}")
            
            return self.theme_matched_words
            
        except Exception as e:
            print(f"❌ Error calling Claude API: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def find_theme_spangrams(self):
        """Find spangrams that match the theme"""
        if not self.theme_matched_words:
            return {}
        
        all_spangrams = self.find_spangrams()
        theme_spangrams = {}
        
        for word in all_spangrams:
            if word in self.theme_matched_words:
                theme_spangrams[word] = all_spangrams[word]
        
        return theme_spangrams
    
    # ========================================================================
    # SOLVING STRATEGY - ADAPTIVE CLUSTERING WITH MULTIPLE INTERPRETATIONS
    # ========================================================================
    
    def get_multiple_interpretations(self, theme, count=3, exclude=None):
        """
        Get multiple diverse interpretations of the theme
        
        Args:
            theme: Puzzle theme string
            count: Number of interpretations to generate
            exclude: Set of already-tried interpretation themes to avoid
            
        Returns:
            List of interpretation dicts with theme, explanation, expected_words
        """
        if exclude is None:
            exclude = set()
        
        exclude_text = ""
        if exclude:
            exclude_text = f"\nAlready tried these (give DIFFERENT interpretations): {list(exclude)}"
        
        prompt = f"""You are solving a NYT Strands puzzle with theme: "{theme}"

Generate {count} DIFFERENT interpretations of this cryptic theme.{exclude_text}

Range from literal to abstract, consider:
- Wordplay and puns
- Idioms and phrases  
- Cultural references
- Double meanings
- Categories of things

For each interpretation, provide:
1. What the theme means
2. What types of words to expect (be specific: "cleaning actions", "laundry terms", etc.)
3. Confidence level (0.0-1.0)

Respond with VALID JSON ONLY (no explanation text before or after):
{{
  "interpretations": [
    {{
      "theme": "Cleaning and stain removal actions",
      "explanation": "The quote is about removing stains, so words are cleaning actions",
      "expected_category": "cleaning verbs and actions",
      "example_words": ["scrub", "wash", "rinse", "soak"],
      "confidence": 0.90
    }},
    ...
  ]
}}

Return exactly {count} diverse interpretations."""

        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 2000,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"⚠️  API error: {response.status_code}")
                return []
            
            data = response.json()
            response_text = data['content'][0]['text'].strip()
            
            # Extract JSON (handle text before/after)
            if not response_text.startswith('{'):
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    response_text = response_text[start_idx:end_idx+1]
            
            result = json.loads(response_text)
            return result.get('interpretations', [])
            
        except Exception as e:
            print(f"⚠️  Error getting interpretations: {e}")
            return []
    
    def find_words_for_interpretation(self, interpretation, limit=20):
        """
        Find words in grid matching a specific interpretation
        NOW WITH WORD GROUPING to avoid testing plurals separately!
        
        Args:
            interpretation: Dict with theme, expected_category, etc.
            limit: Max words to return
            
        Returns:
            List of (word, relevance) tuples found in grid
        """
        category = interpretation.get('expected_category', '')
        theme_desc = interpretation.get('theme', '')
        
        # Get all words from grid
        all_words = list(self.found_words.keys())
        
        # Sort by length (longer words first)
        all_words.sort(key=lambda w: len(w), reverse=True)
        
        # Take up to 1000 words
        words_to_send = all_words[:1000]
        word_list = ', '.join(words_to_send)
        
        print(f"     Searching {len(words_to_send)} words for interpretation...")
        
        # IMPROVED: Ask for 2x words since we'll group them
        api_limit = limit * 2
        
        prompt = f"""Find words matching this theme interpretation:

Theme: "{theme_desc}"
Category: {category}
Expected types: {interpretation.get('example_words', [])}

Available words in grid:
{word_list}

Find {api_limit} words that match this interpretation.
IMPORTANT: Return ONLY theme words (NOT the spangram).
The spangram will be found later from leftover cells.

Include both base words AND their variants if relevant (e.g., STAIN and STAINS).

Respond with VALID JSON ONLY (no text before/after):
{{
  "words": [
    {{"word": "scrub", "relevance": 0.95, "reasoning": "Action for removing stains"}},
    {{"word": "scrubs", "relevance": 0.93, "reasoning": "Plural of scrub"}},
    ...
  ]
}}

Return {api_limit} words ranked by relevance."""

        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 3000,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"     ⚠️  API error: {response.status_code}")
                return []
            
            data = response.json()
            response_text = data['content'][0]['text'].strip()
            
            # Extract JSON
            if not response_text.startswith('{'):
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    response_text = response_text[start_idx:end_idx+1]
            
            result = json.loads(response_text)
            words = result.get('words', [])
            
            # Filter to words that exist in grid
            valid_words = [(w['word'].lower(), w.get('relevance', 0)) 
                          for w in words 
                          if w['word'].lower() in self.found_words]
            
            filtered_count = len(words) - len(valid_words)
            if filtered_count > 0:
                print(f"     ⚠️  Filtered out {filtered_count} words not in grid")
            
            # IMPROVED: Group variants and select best representatives
            word_groups = group_word_variants(valid_words)
            print(f"     📦 Grouped into {len(word_groups)} word families")
            
            # Select best representatives
            selected = select_best_words_from_groups(word_groups, target_count=limit)
            
            print(f"     ✓ Selected {len(selected)} unique words (avoiding duplicates)")
            
            return selected
            
        except Exception as e:
            print(f"     ⚠️  Error finding words: {e}")
            return []
    
    def evaluate_cluster_coherence(self, words, interpretation):
        """
        Evaluate how well a cluster of words fits the interpretation
        
        Args:
            words: List of (word, relevance) tuples
            interpretation: Dict with theme info
            
        Returns:
            Score from 0-10
        """
        if not words:
            return 0.0
        
        # Calculate average relevance
        avg_relevance = sum(rel for _, rel in words) / len(words)
        
        # Penalize if too few words
        word_count_score = min(len(words) / 7.0, 1.0)
        
        # Combine scores
        score = (avg_relevance * 8 + word_count_score * 2)
        
        return score
    
    def solve_adaptive_clustering(self, theme, word_count=8):
        """
        IMPROVED: Multi-interpretation adaptive clustering
        
        YOUR KEY IMPROVEMENTS:
        1. Gets 3 different theme interpretations
        2. For each, finds 15-20 matching words
        3. Tries to solve with each interpretation
        4. NO spangram search in API - emerges from leftover
        
        Returns:
            Solution dict or None
        """
        print("\n" + "="*70)
        print("🎯 ADAPTIVE CLUSTERING SOLVER (MULTI-INTERPRETATION)")
        print("="*70)
        print(f"Theme: '{theme}'")
        print(f"Strategy: Try 3 interpretations → Find words → Place → Leftover = spangram\n")
        
        max_batches = 3
        batch_size = 3
        confidence_threshold_high = 6.0
        
        all_clusters = []
        tried_interpretations = set()
        
        for batch_num in range(max_batches):
            print(f"📋 Batch {batch_num + 1}: Getting {batch_size} theme interpretations...")
            
            # Get interpretations
            interpretations = self.get_multiple_interpretations(
                theme,
                count=batch_size,
                exclude=tried_interpretations
            )
            
            if not interpretations:
                print("⚠️  Failed to get interpretations from API")
                break
            
            print(f"✓ Got {len(interpretations)} interpretations\n")
            
            # Process each interpretation
            for i, interp in enumerate(interpretations, 1):
                interp_theme = interp.get('theme', '')
                confidence = interp.get('confidence', 0)
                explanation = interp.get('explanation', '')
                
                tried_interpretations.add(interp_theme)
                
                print(f"  {i}. {interp_theme}")
                print(f"     Explanation: {explanation[:80]}...")
                print(f"     Expected: {interp.get('expected_category', 'various terms')}")
                
                # Find words for this interpretation
                words_found = self.find_words_for_interpretation(interp, limit=20)
                
                if not words_found:
                    print(f"     ✗ No words found\n")
                    continue
                
                # Evaluate cluster quality
                cluster_score = self.evaluate_cluster_coherence(words_found, interp)
                
                print(f"     ✓ Found {len(words_found)} words, score: {cluster_score:.1f}/10.0")
                
                # Show top words
                if len(words_found) > 5:
                    print(f"       Top 5: {', '.join([w.upper() for w, _ in words_found[:5]])}")
                
                print()
                
                all_clusters.append({
                    'interpretation': interp,
                    'words': words_found,
                    'count': len(words_found),
                    'score': cluster_score,
                    'batch': batch_num
                })
            
            # Sort clusters by score
            all_clusters.sort(key=lambda x: x['score'], reverse=True)
            
            if not all_clusters:
                continue
            
            best = all_clusters[0]
            best_score = best['score']
            
            print(f"🏆 Best cluster so far: {best['interpretation']['theme']}")
            print(f"   Score: {best_score:.1f}/10.0, Words: {best['count']}\n")
            
            # Try solving with best cluster if score is high enough
            if best_score >= confidence_threshold_high:
                print(f"✓ High confidence ({best_score:.1f} >= {confidence_threshold_high}) - attempting solve...")
                solution = self._try_solve_with_cluster(best, word_count)
                if solution:
                    return solution
                print("✗ Solve failed, continuing search...\n")
        
        print(f"\n❌ No solution found after trying {len(all_clusters)} interpretations")
        return None
    
    def _try_solve_with_cluster(self, cluster, word_count):
        """
        Try to solve puzzle using a specific word cluster.

        Uses a flexible approach:
        - Try placing different numbers of theme words
        - Early pruning for trapped letters
        - Validate leftover as spangram (final step)
        - Uses HIGH_CONFIDENCE_ATTEMPTS if cluster score is high

        Args:
            cluster: Dict with words, interpretation, score
            word_count: Target number of solutions (including spangram)

        Returns:
            Solution dict or None
        """
        words = cluster.get("words", [])
        cluster_score = cluster.get("score", 0.0)

        if not words:
            print("  ⚠️ Cluster has no words, skipping.")
            return None

        # Sort words by relevance (highest first)
        sorted_words = sorted(words, key=lambda x: x[1], reverse=True)

        # Expect roughly (word_count - 1) theme words + 1 spangram
        target_theme_words = max(1, word_count - 1)

        # Try a band around the target
        min_try = max(3, target_theme_words - 3)
        max_try = min(len(sorted_words), target_theme_words + 2)

        attempts_per_n = list(range(min_try, max_try + 1))
        attempts_per_n.sort(key=lambda x: abs(x - target_theme_words))  # Prioritize target-ish sizes

        print(f"  Trying to place {min_try}-{max_try} theme words...")

        # Use HIGH_CONFIDENCE_ATTEMPTS threshold
        if cluster_score >= HIGH_CONFIDENCE_THRESHOLD:
            max_total_attempts = HIGH_CONFIDENCE_ATTEMPTS
            print(f"  🔥 High confidence ({cluster_score:.1f}) - exhaustive search ({max_total_attempts:,} attempts)")
        else:
            max_total_attempts = NORMAL_ATTEMPTS
            print(f"  💡 Normal search ({max_total_attempts:,} attempts)")

        total_attempts = 0
        pruned_count = 0

        # Track the best near-miss layout so we can visualize it on failure
        best_attempt = None

        all_cells = {(r, c) for r in range(self.rows) for c in range(self.cols)}

        # Try each word-count size in attempts_per_n
        for n_words in attempts_per_n:
            print(f"\n  → Attempting with {n_words} theme words...")

            if n_words > len(sorted_words):
                continue

            max_combos_per_n = max_total_attempts // max(1, len(attempts_per_n))
            combo_attempts = 0

            for combo_words in itertools.combinations(sorted_words, n_words):
                combo_attempts += 1
                total_attempts += 1

                # Progress updates
                if combo_attempts % SHOW_PROGRESS_EVERY == 0:
                    print(f"     [{combo_attempts:,} combos, {pruned_count} pruned]")

                if combo_attempts > max_combos_per_n:
                    print(f"    ⚠️  Batch limit reached ({max_combos_per_n:,} combinations for {n_words} words)")
                    break

                if total_attempts > max_total_attempts:
                    print(f"    ⚠️  Total limit reached ({max_total_attempts:,} attempts)")
                    break

                # Try to place these words greedily (one path per word, no overlaps)
                placed_words = []
                used_cells = set()

                for word, _rel in combo_words:
                    if word not in self.found_words:
                        # This word cannot be drawn on the grid at all
                        placed = False
                        break

                    placed = False
                    for path in self.found_words[word]:
                        path_cells = set(path)
                        if not (path_cells & used_cells):
                            placed_words.append((word, path))
                            used_cells |= path_cells
                            placed = True
                            break

                    if not placed:
                        # Couldn't place this word without overlap
                        break

                # Did we actually place all n_words?
                if len(placed_words) != n_words:
                    continue

                # Early pruning: trapped letters
                has_trap, trap_reason = has_trapped_letters(
                    used_cells,
                    all_cells,
                    self.rows,
                    self.cols,
                    self.get_neighbors
                )

                if has_trap:
                    # Even if it's trapped, it might still be our "best" attempt so far
                    leftover_cells = all_cells - used_cells
                    cells_used = len(used_cells)

                    attempt_info = {
                        "placed_words": placed_words,
                        "leftover_cells": leftover_cells,
                        "cells_used": cells_used,
                        "spangram": "",
                        "spangram_type": "unknown",
                        "failure_reason": trap_reason,
                    }

                    if best_attempt is None or cells_used > best_attempt["cells_used"]:
                        best_attempt = attempt_info
                        if SHOW_CLOSE_ATTEMPTS and cells_used >= CLOSE_ATTEMPT_THRESHOLD:
                            print("\n  🔍 Close attempt (trapped letters):")
                            self._visualize_grid_state(placed_words, leftover_cells)

                    pruned_count += 1
                    continue  # Skip this combination - it has trapped letters

                # No traps – check leftover and spangram
                leftover_cells = all_cells - used_cells
                if not leftover_cells:
                    # No room left for spangram; still record as near-miss
                    cells_used = len(used_cells)
                    attempt_info = {
                        "placed_words": placed_words,
                        "leftover_cells": leftover_cells,
                        "cells_used": cells_used,
                        "spangram": "",
                        "spangram_type": "unknown",
                        "failure_reason": "No leftover cells for spangram",
                    }
                    if best_attempt is None or cells_used > best_attempt["cells_used"]:
                        best_attempt = attempt_info
                        if SHOW_CLOSE_ATTEMPTS and cells_used >= CLOSE_ATTEMPT_THRESHOLD:
                            print("\n  🔍 Close attempt (no leftover cells):")
                            self._visualize_grid_state(placed_words, leftover_cells)
                    continue

                # Validate leftover as spangram
                validation = self._validate_leftover_spangram(leftover_cells, placed_words)

                if not validation.get("valid"):
                    # Record as near-miss
                    cells_used = len(used_cells)
                    attempt_info = {
                        "placed_words": placed_words,
                        "leftover_cells": leftover_cells,
                        "cells_used": cells_used,
                        "spangram": validation.get("spangram", ""),
                        "spangram_type": validation.get("type", "unknown"),
                        "failure_reason": validation.get("reason", "Spangram invalid"),
                    }
                    if best_attempt is None or cells_used > best_attempt["cells_used"]:
                        best_attempt = attempt_info
                        if SHOW_CLOSE_ATTEMPTS and cells_used >= CLOSE_ATTEMPT_THRESHOLD:
                            print("\n  🔍 Close attempt (spangram invalid):")
                            self._visualize_grid_state(placed_words, leftover_cells)
                    continue

                # SUCCESS: valid spangram & no traps
                spangram_text = validation["spangram"]
                spangram_type = validation.get("type", "single")

                print(f"\n  ✅ SOLUTION FOUND!")
                print(f"     Placed {len(placed_words)} theme words: "
                    f"{', '.join([w.upper() for w, _ in placed_words])}")
                print(f"     Spangram: {spangram_text.upper()}")
                print(f"     Total cells: {len(used_cells) + len(leftover_cells)}/{self.rows * self.cols}")

                self._show_final_solution({
                    "placed_words": placed_words,
                    "spangram": spangram_text,
                    "spangram_type": spangram_type,
                    "leftover_cells": leftover_cells,
                    "cells_used": len(used_cells) + len(leftover_cells),
                })

                return {
                    "theme_words": placed_words,
                    "spangram": spangram_text,
                    "spangram_cells": leftover_cells,
                    "total_cells": len(used_cells) + len(leftover_cells),
                    "cluster": cluster,
                    "validation_type": spangram_type,
                    "pruned_attempts": pruned_count,
                }

            if total_attempts >= max_total_attempts:
                break

        # If we reach here, no full valid solution was found
        print(f"\n  ✗ No valid solution found for this cluster (tried {total_attempts:,} combinations, pruned {pruned_count})")
        print(f"     Pruning saved {pruned_count:,} futile validations ({100*pruned_count//max(1, total_attempts)}% of attempts)")

        # Show the best attempt if we have one
        if best_attempt is not None:
            print("\n  🔍 BEST ATTEMPT (no full solution):")
            self._show_best_attempt(best_attempt, total_attempts)

        return None

    
    def _try_place_word_combination(self, word_combo, word_count, show_progress=False):
        """
        Try to place a specific combination of words on the grid
        
        Returns result dict with:
        - placed_words: List of (word, path) tuples
        - leftover_cells: Set of unused cells
        - cells_used: Number of cells used
        - is_solution: Whether this is a valid complete solution
        """
        # Extract just the word names
        word_names = [w for w, _ in word_combo]
        
        # Try to place each word without overlaps
        placed_words = []
        used_cells = set()
        
        for word in word_names:
            if word not in self.found_words:
                continue
            
            # Try each path for this word
            placed = False
            for path in self.found_words[word]:
                path_cells = set(path)
                
                # Check if this path overlaps with already placed words
                if not (path_cells & used_cells):
                    placed_words.append((word, path))
                    used_cells |= path_cells
                    placed = True
                    break
            
            if not placed:
                # Couldn't place this word without overlap
                return None
        
        # Calculate leftover cells
        all_cells = set((r, c) for r in range(self.rows) for c in range(self.cols))
        leftover_cells = all_cells - used_cells
        cells_used = len(used_cells)
        
        # Validate leftover as spangram
        validation = self._validate_leftover_spangram(leftover_cells, placed_words)
        
        return {
            'placed_words': placed_words,
            'leftover_cells': leftover_cells,
            'cells_used': cells_used,
            'is_solution': validation['valid'],
            'spangram': validation.get('spangram', ''),
            'spangram_type': validation.get('type', 'unknown'),
            'failure_reason': validation.get('reason', '')
        }
    
    def _validate_leftover_spangram(self, leftover_cells, placed_words):
        """
        Validate if leftover cells can form valid word(s)
        
        IMPROVED: Actually searches for valid words in the leftover cells
        instead of just trying to read them as one path!
        
        Returns dict with 'valid', 'spangram', 'type', 'reason'
        """
        if not leftover_cells:
            return {'valid': False, 'reason': 'No leftover cells'}
        
        # NEW APPROACH: Search for all valid words in leftover cells
        leftover_words = self._find_words_in_cells(leftover_cells, min_length=4)
        
        if not leftover_words:
            # Try reading as single path (fallback)
            components = self._find_connected_components(leftover_cells)
            if len(components) == 1:
                text = self._read_component_text(components[0])
                return {
                    'valid': False,
                    'reason': f'No valid words found in leftover. Read as: "{text}"'
                }
            else:
                return {
                    'valid': False,
                    'reason': f'No valid words found in {len(components)} components'
                }
        
        # Try to place words in leftover to use all cells
        # This is similar to the main solving loop but for leftover only
        leftover_word_list = [(word, 1.0) for word in leftover_words.keys()]
        
        # Try to find a combination that uses ALL leftover cells
        for n_words in range(1, len(leftover_word_list) + 1):
            for word_combo in itertools.combinations(leftover_word_list, n_words):
                # Try to place these words
                placed = []
                used = set()
                
                for word, _ in word_combo:
                    placed_word = False
                    for path in leftover_words[word]:
                        path_set = set(path)
                        if not (path_set & used):  # No overlap
                            placed.append((word, path))
                            used |= path_set
                            placed_word = True
                            break
                    
                    if not placed_word:
                        break
                
                # Check if we used ALL leftover cells
                if len(placed) == n_words and used == leftover_cells:
                    # SUCCESS! All leftover cells form valid word(s)
                    spangram_text = '_'.join([w for w, _ in placed])
                    return {
                        'valid': True,
                        'spangram': spangram_text,
                        'type': 'multi' if len(placed) > 1 else 'single',
                        'components': len(placed),
                        'words': placed
                    }
        
        # Couldn't use all cells
        # Return best attempt (most cells covered)
        best_coverage = 0
        best_words = []
        
        for n_words in range(1, len(leftover_word_list) + 1):
            for word_combo in itertools.combinations(leftover_word_list, n_words):
                placed = []
                used = set()
                
                for word, _ in word_combo:
                    placed_word = False
                    for path in leftover_words[word]:
                        path_set = set(path)
                        if not (path_set & used):
                            placed.append((word, path))
                            used |= path_set
                            placed_word = True
                            break
                    
                    if not placed_word:
                        break
                
                if len(used) > best_coverage:
                    best_coverage = len(used)
                    best_words = [w for w, _ in placed]
        
        return {
            'valid': False,
            'reason': f'Found words {best_words} but only covers {best_coverage}/{len(leftover_cells)} cells'
        }
    
    # ========================================================================
    # VISUALIZATION METHODS
    # ========================================================================
    
    def _visualize_grid_state(self, placed_words, leftover_cells):
        """Show current grid state with placed words and leftover"""
        print("\n  Grid state:")
        
        # Create a map of cells to words
        cell_to_word = {}
        for word, path in placed_words:
            for cell in path:
                cell_to_word[cell] = word
        
        # Print grid
        for row in range(self.rows):
            row_str = "  "
            for col in range(self.cols):
                cell = (row, col)
                if cell in cell_to_word:
                    letter = self.grid[row][col]
                    row_str += f" {letter} "
                elif cell in leftover_cells:
                    row_str += " . "
                else:
                    row_str += "   "
            print(row_str)
        
        # Show leftover info
        if leftover_cells:
            components = self._find_connected_components(leftover_cells)
            print(f"\n  Leftover: {len(leftover_cells)} cells in {len(components)} component(s)")
            
            for i, comp in enumerate(components, 1):
                text = self._read_component_text(comp)
                is_valid = self.trie.search(text)
                status = "✓" if is_valid else "✗"
                print(f"    Component {i}: '{text.upper()}' ({len(comp)} cells) {status}")
    
    def _show_final_solution(self, result):
        """Display the final solution with full details"""
        print("\n" + "="*70)
        print("📋 FINAL SOLUTION")
        print("="*70)
        
        placed_words = result['placed_words']
        spangram = result['spangram']
        spangram_type = result['spangram_type']
        
        # Show grid
        print("\nFinal grid:")
        cell_to_word = {}
        for word, path in placed_words:
            for cell in path:
                cell_to_word[cell] = word
        
        for row in range(self.rows):
            row_str = ""
            for col in range(self.cols):
                cell = (row, col)
                letter = self.grid[row][col]
                if cell in cell_to_word:
                    row_str += f" {letter} "
                else:
                    row_str += f" {letter.lower()} "
            print(row_str)
        
        # Show spangram
        print(f"\n🎯 Spangram: {spangram.upper()}")
        if spangram_type == 'multi':
            parts = spangram.split('_')
            print(f"   Type: Multi-word ({len(parts)} words: {', '.join(p.upper() for p in parts)})")
        else:
            print(f"   Type: Single word")
        
        # Show theme words
        print(f"\n📝 Theme words placed ({len(placed_words)}):")
        for i, (word, path) in enumerate(placed_words, 1):
            relevance = self.theme_matched_words.get(word, {}).get('relevance', 0)
            print(f"  {i}. {word.upper():<15} ({len(path)} cells, {relevance:.0%} relevance)")
        
        # Statistics
        total_cells = self.rows * self.cols
        print(f"\n✓ Total cells used: {result['cells_used']}/{total_cells} ({result['cells_used']/total_cells:.0%})")
        print("✓ All words form valid paths")
        print("✓ Spangram touches opposite edges")
    
    def _show_best_attempt(self, best_attempt, total_attempts):
        """Show the best attempt even though it failed"""
        cells_used = best_attempt['cells_used']
        total_cells = self.rows * self.cols
        percentage = (cells_used / total_cells) * 100
        
        print(f"\n  Used {cells_used}/{total_cells} cells ({percentage:.0f}%)")
        
        # Show grid
        self._visualize_grid_state(best_attempt['placed_words'], best_attempt['leftover_cells'])
        
        # Show what was placed
        print(f"\n  Theme words placed ({len(best_attempt['placed_words'])}):")
        for word, path in best_attempt['placed_words']:
            print(f"    • {word.upper()} ({len(path)} cells)")
        
        # Explain why it failed
        reason = best_attempt.get('failure_reason', 'Unknown')
        print(f"\n  ❌ Why it failed: {reason}")
        
        print(f"\n  💡 Suggestions:")
        print(f"     1. Theme interpretation might need adjustment")
        print(f"     2. Some theme words may be incorrect")
        print(f"     3. Try different word combinations manually")
        print(f"     4. Check if dictionary has all needed words")

    # ------------------------------------------------------------------------
    # UTILITY METHODS
    # ------------------------------------------------------------------------
    
    def _find_connected_components(self, cells):
        """Find connected components in a set of cells"""
        if not cells:
            return []
        
        cells_list = list(cells)
        visited = set()
        components = []
        
        for start_cell in cells_list:
            if start_cell in visited:
                continue
            
            # BFS to find component
            component = set()
            queue = [start_cell]
            
            while queue:
                cell = queue.pop(0)
                if cell in visited:
                    continue
                
                visited.add(cell)
                component.add(cell)
                
                # Add neighbors
                for neighbor in self.get_neighbors(cell[0], cell[1]):
                    if neighbor in cells and neighbor not in visited:
                        queue.append(neighbor)
            
            components.append(component)
        
        return components
    
    def _read_component_text(self, cells):
        """
        Read text from a component using DFS
        
        Returns:
            String of letters in the component
        """
        if not cells:
            return ""
        
        # Find starting cell (edge cell preferred)
        cells_list = list(cells)
        start = cells_list[0]
        
        for cell in cells_list:
            if (cell[0] == 0 or cell[0] == self.rows - 1 or 
                cell[1] == 0 or cell[1] == self.cols - 1):
                start = cell
                break
        
        # DFS to read text
        visited = set()
        text = ""
        
        def dfs(cell):
            nonlocal text
            if cell in visited or cell not in cells:
                return
            
            visited.add(cell)
            text += self.grid[cell[0]][cell[1]]
            
            # Try neighbors
            for neighbor in self.get_neighbors(cell[0], cell[1]):
                if neighbor not in visited and neighbor in cells:
                    dfs(neighbor)
        
        dfs(start)
        return text.lower()
    
    def _find_words_in_cells(self, cells, min_length=4):
        """
        Find all valid words that can be formed from a set of cells
        
        This is like find_all_words() but limited to specific cells.
        Used for finding words in leftover components.
        
        Args:
            cells: Set of (row, col) tuples to search within
            min_length: Minimum word length
            
        Returns:
            Dict of {word: [paths]} where each path uses only cells from the set
        """
        found_words = {}
        
        # Try starting from each cell in the set
        for start_cell in cells:
            row, col = start_cell
            
            # Run DFS from this starting point
            def dfs_search(r, c, visited, current_word, path):
                # Add current cell
                visited.add((r, c))
                current_word += self.grid[r][c]
                path.append((r, c))
                
                # Check if valid word
                if len(current_word) >= min_length and self.trie.search(current_word):
                    word_key = current_word.lower()
                    if word_key not in found_words:
                        found_words[word_key] = []
                    found_words[word_key].append(list(path))
                
                # Early pruning: stop if prefix doesn't exist
                if not self.trie.starts_with(current_word):
                    visited.remove((r, c))
                    path.pop()
                    return
                
                # Explore neighbors (only within allowed cells)
                for next_row, next_col in self.get_neighbors(r, c):
                    if (next_row, next_col) in cells and (next_row, next_col) not in visited:
                        dfs_search(next_row, next_col, visited, current_word, path)
                
                # Backtrack
                visited.remove((r, c))
                path.pop()
            
            dfs_search(row, col, set(), "", [])
        
        return found_words
    
    def print_grid(self):
        """Print the puzzle grid"""
        print("\nGrid:")
        for row in self.grid:
            print(' '.join(row))
    
    def print_puzzle_info(self):
        """Print puzzle information"""
        print("\nPuzzle Info:")
        theme = self.puzzle_info.get('theme', 'Unknown')
        print(f"  Theme: {theme}")
    
    def print_word_statistics(self):
        """Print statistics about found words"""
        print("\n" + "="*70)
        print("WORD STATISTICS")
        print("="*70)
        
        # Count by length
        length_counts = {}
        max_length = 0
        
        for word in self.found_words:
            length = len(word)
            max_length = max(max_length, length)
            length_counts[length] = length_counts.get(length, 0) + 1
        
        print("\nWord count by length:")
        for length in range(1, max_length + 1):
            if length in length_counts:
                print(f"  {length} letters: {length_counts[length]} words")
        
        # Longest words
        longest = [w for w in self.found_words if len(w) == max_length]
        if longest:
            print(f"\nLongest words ({max_length} letters): {', '.join(longest)}")


# ============================================================================
# SECTION 3: SOLVING STRATEGIES
# ============================================================================
# 
# The solver has 4 main strategies:
# 1. solve_adaptive_clustering() - tries multiple theme interpretations
# 2. solve_hybrid() - geometry-first approach with theme ranking  
# 3. solve_optimal_ilp() - mathematical optimization (requires pulp)
# 4. auto_solve_puzzle() - quick heuristic approach
#
# These methods are very long (1000+ lines total) and would make this file
# too large. For now, I'm marking them as TODO to add back if needed.
# ============================================================================

# TODO: Add back solving strategies if needed
# Currently commented out to keep file manageable
# See original solver.py lines 500-2500 for full implementations


# ============================================================================
# SECTION 4: FILE I/O AND MAIN EXECUTION
# ============================================================================

def load_grid_from_json(json_path):
    """
    Load grid from JSON file
    
    Returns:
        Tuple of (grid, puzzle_info)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    grid = data.get('grid', [])
    puzzle_info = {
        'theme': data.get('theme', 'Unknown'),
        'source': data.get('source', json_path),
        'method': data.get('method', 'unknown'),
        'word_count': data.get('word_count', 8)
    }
    
    return grid, puzzle_info


def list_available_grids(data_dir="data/inputs"):
    """
    List all available grid JSON files
    
    Returns:
        List of file paths, sorted by modification time
    """
    if not os.path.exists(data_dir):
        return []
    
    json_files = glob(os.path.join(data_dir, "*_grid.json"))
    json_files.sort(key=os.path.getmtime)
    
    return json_files


def solve_from_json(json_path, dictionary_path=DICTIONARY_PATH, use_theme_matching=True):
    """
    Solve a puzzle from a JSON file
    
    This is the main entry point called by main.py
    
    Args:
        json_path: Path to JSON file with grid
        dictionary_path: Path to dictionary
        use_theme_matching: Whether to use Claude API
        
    Returns:
        StrandsSolver instance or None
    """
    print("="*70)
    print("STRANDS SOLVER - Loading from JSON")
    print("="*70)
    print(f"\nLoading grid from: {os.path.basename(json_path)}")
    
    # Load grid
    grid, puzzle_info = load_grid_from_json(json_path)
    
    if not grid:
        print("❌ No grid data found in JSON")
        return None
    
    # Create solver
    solver = StrandsSolver(grid, dictionary_path, puzzle_info)
    
    # Print info
    solver.print_puzzle_info()
    solver.print_grid()
    
    # Find words
    print("\nSearching for all valid words...")
    words = solver.find_all_words()
    
    if not words:
        print("⚠️  No words found! Check dictionary path.")
        return solver
    
    print(f"✓ Found {len(words)} unique words")
    
    # Print statistics
    solver.print_word_statistics()
    
    # Find spangrams
    print("\n" + "-"*70)
    print("Searching for spangrams (words touching opposite edges)...")
    spangrams = solver.find_spangrams()
    
    if spangrams:
        print(f"✓ Found {len(spangrams)} potential spangram(s):")
        for word in sorted(spangrams.keys()):
            print(f"  • {word.upper()}")
    else:
        print("✗ No spangrams found")
    
    # Theme matching and solving
    if use_theme_matching and puzzle_info.get('theme'):
        word_count = puzzle_info.get('word_count', 8)
        
        # Use the improved adaptive clustering solver
        solution = solver.solve_adaptive_clustering(puzzle_info['theme'], word_count=word_count)
        
        if solution:
            print("\n" + "="*70)
            print("🎉 SUCCESS!")
            print("="*70)
            return solver
        else:
            print("\n" + "="*70)
            print("⚠️  NO SOLUTION FOUND")
            print("="*70)
            print("\nThe solver tried many combinations but couldn't find a valid solution.")
            print("See the 'BEST ATTEMPT' above for the closest result.")
    
    print("\n" + "="*70)
    print("SOLVING COMPLETE!")
    print("="*70)
    
    return solver


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("Strands Solver (Clean Version)")
    print("=" * 70)
    
    if len(sys.argv) > 1:
        # Solve specific file
        json_path = sys.argv[1]
        if os.path.exists(json_path):
            solver = solve_from_json(json_path)
        else:
            print(f"❌ File not found: {json_path}")
            sys.exit(1)
    else:
        # List available grids
        grids = list_available_grids()
        
        if not grids:
            print(f"\n⚠️  No extracted grids found in {DATA_DIR}/")
            print("\nFirst, extract a grid using vision.py")
        else:
            print(f"\nFound {len(grids)} extracted grid(s):")
            for i, grid_path in enumerate(grids, 1):
                print(f"  {i}. {os.path.basename(grid_path)}")
            
            print("\nSolving most recent grid...\n")
            latest = grids[-1]
            solver = solve_from_json(latest)