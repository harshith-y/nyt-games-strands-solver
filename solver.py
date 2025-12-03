"""
NYT Strands Solver - No Spangram Version
Solves Strands puzzles using DFS + Trie + LLM theme matching

ARCHITECTURE:
1. Core Classes (Trie, TrieNode, StrandsSolver)
2. Word Finding (DFS to find all valid words)
3. Theme Matching (Claude API)
4. Solving Strategy (adaptive clustering with multiple interpretations)
5. Utility Functions

Author: Cleaned version (spangram code removed)
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
HIGH_CONFIDENCE_ATTEMPTS = 50000     # Max attempts when confidence >= 7.0
NORMAL_ATTEMPTS = 10000              # Max attempts for normal confidence

# Visualization settings
SHOW_PROGRESS_EVERY = 500            # Show progress every N attempts
SHOW_CLOSE_ATTEMPTS = True           # Show grids for close attempts
CLOSE_ATTEMPT_THRESHOLD = 25         # Show attempts that use 40+ cells

# Path search limits (to avoid combinatorial explosion)
MAX_PATHS_PER_WORD = 5              # Max paths per word considered in a combo
MAX_PATH_COMBOS_PER_COMBO = 3000     # Max path combinations per word-set combo

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
    
    def find_all_words(self):
        """Find all valid words in the grid"""
        self.found_words = {}
        
        for row in range(self.rows):
            for col in range(self.cols):
                self.dfs(row, col, set(), "", [])
        
        return self.found_words
    
    # ------------------------------------------------------------------------
    # THEME MATCHING (Claude API)
    # ------------------------------------------------------------------------
    
    def filter_words_by_theme(self, theme, word_count=8):
        """
        Use Claude API to filter words by theme relevance
        
        IMPROVED VERSION: Shows API the actual words in grid and asks it to SELECT from those
        
        Args:
            theme: Puzzle theme string
            word_count: Expected number of words
            
        Returns:
            Dict of {word: {relevance, reasoning}}
        """
        print(f"\n🤖 Using Claude API to match words to theme: '{theme}'")
        print(f"   Expected words: {word_count} ({word_count-1} theme words + 1 spangram)")
        print("   Strategy: Show API actual grid words, ask it to SELECT from those")
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
        
        # Prepare word candidates (4+ letters, sorted by length)
        candidates = []
        for word in self.found_words.keys():
            if len(word) >= 4:  # Only words 4+ letters (Strands minimum)
                candidates.append(word)
        
        candidates.sort(key=lambda w: len(w), reverse=True)  # Sort by length
        
        # Take top 150 words for API (good coverage without overwhelming)
        top_candidates = candidates[:150]
        
        print(f"   Showing API {len(top_candidates)} actual words from grid...")
        
        # Format as comma-separated list
        candidate_list = ", ".join([w.upper() for w in top_candidates])
        
        # Build IMPROVED prompt - shows API actual words, asks it to SELECT
        prompt = f"""You are helping solve a NYT Strands word puzzle with theme: "{theme}"

Here are 150 words that ACTUALLY EXIST in this puzzle's grid:
{candidate_list}

Your task: Select approximately {word_count*2} words from this list that best match the theme "{theme}".

CRITICAL RULES:
1. ONLY select words from the list above (don't suggest words not in the list!)
2. Think about SPECIFIC categories the theme refers to
3. Look for SPECIFIC examples matching the theme, not general/abstract concepts
4. Avoid selecting multiple variants of the same word (e.g., don't pick both SPIDER and SPIDERS)
5. Prefer common/simple words over technical or scientific terms
6. Consider wordplay - the theme might be a pun, idiom, or have double meanings

Theme interpretation examples:
- "On the web" could mean:
  * Types of spiders (look for specific spider names: WIDOW, RECLUSE, HUNTSMAN, TARANTULA, JUMPING, HOUSE)
  * OR internet terms (look for: BROWSER, WEBSITE, ONLINE, STREAMING)
  
- "Things are starting to take shape" could mean:
  * Geometric shapes (look for: ANGLE, OVAL, TRIANGLE, RECTANGLE, PENTAGON, OCTAGON, POLYGON)
  * OR things forming/developing (look for: FORMING, BUILDING, GROWING)

For theme "{theme}", think step-by-step:
1. What does "{theme}" literally mean?
2. Is there wordplay involved? (puns, idioms, double meanings?)
3. What SPECIFIC category of things does this describe?
4. What are examples of that category?
5. Which words from the list match those examples?

Return ONLY valid JSON (no markdown, no code blocks):
{{
  "theme_interpretation": "Brief explanation of what '{theme}' means",
  "expected_category": "Specific category (e.g., 'types of spiders', NOT just 'spider-related')",
  "words": [
    {{"word": "WORD1", "relevance": 0.95, "reasoning": "Why this specific word fits the category"}},
    {{"word": "WORD2", "relevance": 0.90, "reasoning": "Why this specific word fits"}}
  ]
}}

Select approximately {word_count*2} words from the list, sorted by relevance (highest first).
Focus on SPECIFIC examples, not general terms or scientific vocabulary."""
        
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
                print(f"   Response: {response.text[:200]}")
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
            
            # Extract JSON from response
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
            print(f"\n💡 Theme interpretation:")
            interpretation = result.get('theme_interpretation', 'Unknown')
            expected_category = result.get('expected_category', '')
            print(f"  🎯 {interpretation}")
            if expected_category:
                print(f"  📂 Expected category: {expected_category}")
            
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
        Uses word grouping to avoid testing plurals separately
        
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
            
            # Group variants and select best representatives
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
    
    
    def solve_greedy_longest_first(self, target_word_count=8, min_word_length=4):
        """
        Fallback solver: Ignore theme, place longest words first
        
        This greedy approach often works when semantic matching fails because:
        - Long words are more specific (less ambiguous)
        - Long words have fewer placement options (less overlap potential)
        - Greedy placement is fast (seconds vs minutes)
        
        IMPORTANT: Enforces hard constraints:
        - Exactly target_word_count words
        - All words >= min_word_length letters
        - All cells used exactly once
        
        Args:
            target_word_count: Exact number of words to place (default 8)
            min_word_length: Minimum word length (default 4)
        
        Returns:
            List of (word, path) tuples or None if constraints can't be met
        """
        print(f"\n{'='*70}")
        print(f"🔧 FALLBACK: GREEDY LONGEST-FIRST SOLVER")
        print(f"{'='*70}")
        print(f"\nIgnoring theme - trying pure geometric approach...")
        print(f"Strategy: Place longest words first, then fill remaining cells")
        print(f"Constraints: Exactly {target_word_count} words, all ≥{min_word_length} letters\n")
        
        # Sort all words by length (4+ letters only)
        all_words_sorted = sorted(
            [(w, paths) for w, paths in self.found_words.items() if len(w) >= min_word_length],
            key=lambda x: len(x[0]),
            reverse=True
        )
        
        placed_words = []
        used_cells = set()
        total_cells = self.rows * self.cols
        
        # Try placing each word (longest first)
        for word, paths in all_words_sorted:
            # Stop if we've placed target number of words
            if len(placed_words) >= target_word_count:
                break
            
            # Try each path for this word (take first that works)
            for path in paths[:3]:  # Only try first 3 paths for speed
                path_cells = set(path)
                
                if not (path_cells & used_cells):  # No overlap
                    placed_words.append((word, path))
                    used_cells |= path_cells
                    
                    cells_pct = 100 * len(used_cells) / total_cells
                    
                    # Show first 10 placements
                    if len(placed_words) <= 10:
                        print(f"  ✓ {len(placed_words):2}. {word.upper():<15} "
                              f"({len(word)} letters) → {len(used_cells)}/{total_cells} cells "
                              f"({cells_pct:.0f}%)")
                    
                    break  # Placed this word, move to next
        
        # Check if we met all constraints
        cells_used = len(used_cells)
        words_placed = len(placed_words)
        
        print(f"\n{'='*70}")
        print(f"GREEDY RESULT")
        print(f"{'='*70}")
        print(f"  Words placed: {words_placed}/{target_word_count}")
        print(f"  Cells used: {cells_used}/{total_cells}")
        
        # Verify constraints
        if words_placed == target_word_count and cells_used == total_cells:
            print(f"  ✅ SUCCESS! All constraints met!")
            return placed_words
        elif words_placed == target_word_count:
            print(f"  ⚠️  Correct word count but only {cells_used}/{total_cells} cells used")
            print(f"     (Leftover cells: {total_cells - cells_used})")
            # Could try to fill leftover cells here
            return None
        elif cells_used == total_cells:
            print(f"  ⚠️  All cells used but wrong word count ({words_placed} vs {target_word_count})")
            return None
        else:
            print(f"  ❌ Failed - neither word count nor cells match")
            return None
    
    def solve_adaptive_clustering(self, theme, word_count=8):
        """
        Multi-interpretation adaptive clustering
        
        Strategy:
        1. Gets 3 different theme interpretations
        2. For each, finds 15-20 matching words
        3. Tries to place words on grid
        4. Uses early pruning for trapped letters
        
        Returns:
            Solution dict or None
        """
        print("\n" + "="*70)
        print("🎯 ADAPTIVE CLUSTERING SOLVER (MULTI-INTERPRETATION)")
        print("="*70)
        print(f"Theme: '{theme}'")
        print(f"Strategy: Try interpretations → Find words → Place on grid\n")
        
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
        
        # Try greedy fallback
        print(f"\n{'='*70}")
        print(f"🔧 ACTIVATING FALLBACK SOLVER")
        print(f"{'='*70}")
        print(f"\nTheme matching failed - trying geometry-first approach...")
        
        fallback_result = self.solve_greedy_longest_first(
            target_word_count=word_count,
            min_word_length=4
        )
        
        if fallback_result:
            print(f"\n{'='*70}")
            print(f"✅ FALLBACK SOLVER SUCCEEDED!")
            print(f"{'='*70}")
            print(f"\nFound solution using greedy placement (ignoring theme)")
            
            # Return in same format as main solver
            # Note: Spangram identification will happen in Phase 2
            return {
                "placed_words": fallback_result,
                "method": "greedy_fallback",
                "theme_interpretation": "Greedy placement (theme ignored)",
                "cells_used": sum(len(path) for _, path in fallback_result)
            }
        
        print(f"\n⚠️  Fallback solver also failed")
        return None
    
    def _try_solve_with_cluster(self, cluster, word_count):
        """
        Try to solve puzzle using a specific word cluster.

        Uses a flexible approach:
        - Try placing different numbers of theme words
        - Early pruning for trapped letters
        - Validate that all cells are used

        Args:
            cluster: Dict with words, interpretation, score
            word_count: Target number of words

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

        # Try a range of word counts
        min_try = max(3, word_count - 3)
        max_try = min(len(sorted_words), word_count + 3)

        attempts_per_n = list(range(min_try, max_try + 1))
        attempts_per_n.sort(key=lambda x: abs(x - word_count))  # Prioritize target size

        print(f"  Trying to place {min_try}-{max_try} words...")

        # Use HIGH_CONFIDENCE_ATTEMPTS threshold
        if cluster_score >= HIGH_CONFIDENCE_THRESHOLD:
            max_total_attempts = HIGH_CONFIDENCE_ATTEMPTS
            print(f"  🔥 High confidence ({cluster_score:.1f}) - exhaustive search ({max_total_attempts:,} attempts)")
        else:
            max_total_attempts = NORMAL_ATTEMPTS
            print(f"  💡 Normal search ({max_total_attempts:,} attempts)")

        total_attempts = 0
        pruned_count = 0

        # Track the best attempt
        best_attempt = None

        all_cells = {(r, c) for r in range(self.rows) for c in range(self.cols)}

        # Try each word-count size
        for n_words in attempts_per_n:
            print(f"\n  → Attempting with {n_words} words...")

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

                # Try to place these words (one path per word, no overlaps)
                placed_words = []
                used_cells = set()

                for word, _rel in combo_words:
                    if word not in self.found_words:
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
                        break

                # Did we place all n_words?
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
                    leftover_cells = all_cells - used_cells
                    cells_used = len(used_cells)

                    attempt_info = {
                        "placed_words": placed_words,
                        "leftover_cells": leftover_cells,
                        "cells_used": cells_used,
                        "failure_reason": trap_reason,
                    }

                    if best_attempt is None or cells_used > best_attempt["cells_used"]:
                        best_attempt = attempt_info
                        if SHOW_CLOSE_ATTEMPTS and cells_used >= CLOSE_ATTEMPT_THRESHOLD:
                            print("\n  🔍 Close attempt (trapped letters):")
                            self._visualize_grid_state(placed_words, leftover_cells)

                    pruned_count += 1
                    continue

                # No traps – check if we used all cells
                leftover_cells = all_cells - used_cells
                cells_used = len(used_cells)

                # Record as attempt
                attempt_info = {
                    "placed_words": placed_words,
                    "leftover_cells": leftover_cells,
                    "cells_used": cells_used,
                    "failure_reason": f"{len(leftover_cells)} cells unused" if leftover_cells else None,
                }

                if best_attempt is None or cells_used > best_attempt["cells_used"]:
                    best_attempt = attempt_info
                    if SHOW_CLOSE_ATTEMPTS and cells_used >= CLOSE_ATTEMPT_THRESHOLD:
                        print("\n  🔍 Close attempt:")
                        self._visualize_grid_state(placed_words, leftover_cells)

                # Check if we used ALL cells (success!)
                if len(leftover_cells) == 0:
                    print(f"\n  ✅ SOLUTION FOUND!")
                    print(f"     Placed {len(placed_words)} words: "
                        f"{', '.join([w.upper() for w, _ in placed_words])}")
                    print(f"     Total cells: {cells_used}/{self.rows * self.cols}")

                    self._show_final_solution({
                        "placed_words": placed_words,
                        "leftover_cells": leftover_cells,
                        "cells_used": cells_used,
                    })
                    
                    # Get theme for spangram identification
                    theme = self.puzzle_info.get('theme', '')
                    
                    # PHASE 2: Identify spangram
                    spangram_info = self.identify_spangram(
                        placed_words,
                        word_count,  # expected_count
                        theme
                    )

                    return {
                        "theme_words": spangram_info['theme_words'],
                        "spangram_words": spangram_info['spangram_words'],
                        "spangram_type": spangram_info['spangram_type'],
                        "spangram_reasoning": spangram_info['reasoning'],
                        "total_cells": cells_used,
                        "cluster": cluster,
                        "pruned_attempts": pruned_count,
                    }

            if total_attempts >= max_total_attempts:
                break

        # No solution found
        print(f"\n  ✗ No valid solution found for this cluster (tried {total_attempts:,} combinations, pruned {pruned_count})")
        print(f"     Pruning saved {pruned_count:,} futile validations ({100*pruned_count//max(1, total_attempts)}% of attempts)")

        # Show the best attempt
        if best_attempt is not None:
            print("\n  🔍 BEST ATTEMPT (no complete solution):")
            self._show_best_attempt(best_attempt, total_attempts)


        # Try to fill leftover regions with theme words
        if best_attempt and best_attempt['leftover_cells']:
            print(f"\n  💡 Attempting to fill leftover regions with theme words...")
            
            theme = self.puzzle_info.get('theme', '')
            leftover_words = self._analyze_leftover_regions(
                best_attempt['leftover_cells'],
                best_attempt['placed_words'],
                theme
            )
            
            if leftover_words:
                print(f"\n  📝 Selected {len(leftover_words)} word(s) from leftover regions")
                
                # Combine placed words with leftover words
                combined_words = best_attempt['placed_words'] + leftover_words
                
                # Check if this uses all cells
                used_cells = set()
                for word, path in combined_words:
                    used_cells.update(path)
                
                total_cells = self.rows * self.cols
                
                if len(used_cells) == total_cells:
                    print(f"\n  ✅ SOLUTION FOUND by filling leftover regions!")
                    print(f"     Total words: {len(combined_words)}")
                    print(f"     Cells used: {len(used_cells)}/{total_cells}")
                    
                    self._show_final_solution({
                        "placed_words": combined_words,
                        "leftover_cells": set(),
                        "cells_used": len(used_cells),
                    })
                    
                    # PHASE 2: Identify spangram
                    spangram_info = self.identify_spangram(
                        combined_words,
                        word_count,  # expected_count
                        theme
                    )
                    
                    return {
                        "theme_words": spangram_info['theme_words'],
                        "spangram_words": spangram_info['spangram_words'],
                        "spangram_type": spangram_info['spangram_type'],
                        "spangram_reasoning": spangram_info['reasoning'],
                        "total_cells": len(used_cells),
                        "cluster": cluster,
                        "pruned_attempts": pruned_count,
                    }
                else:
                    print(f"\n  ⚠️  Leftover words found but don't complete the grid")
                    print(f"     Cells used: {len(used_cells)}/{total_cells}")

        return None
    
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
            print(f"\n  Leftover: {len(leftover_cells)} cells")
            
            # Find valid words in leftover cells
            leftover_words = self._find_words_in_cells(leftover_cells, min_length=4)
            
            if leftover_words:
                print(f"  Valid words found in leftover: {len(leftover_words)}")
                # Show top 5 words by length
                sorted_words = sorted(leftover_words.keys(), key=len, reverse=True)
                for word in sorted_words[:5]:
                    print(f"    • {word.upper()} ({len(word)} letters)")
            else:
                print(f"  No valid words found in leftover cells")
            
            # Show components
            components = self._find_connected_components(leftover_cells)
            if len(components) > 1:
                print(f"  Split into {len(components)} separate regions:")
                for i, comp in enumerate(components, 1):
                    # Show what letters are in each component
                    text = self._read_component_text(comp)
                    print(f"    Region {i}: {len(comp)} cells - '{text.upper()}'")
    
    def _show_final_solution(self, result):
        """Display the final solution with full details"""
        print("\n" + "="*70)
        print("PHASE 1 RESULTS")
        print("="*70)
        
        placed_words = result['placed_words']
        
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
        
        # Show theme words
        print(f"\n📝 Theme words placed ({len(placed_words)}):")
        for i, (word, path) in enumerate(placed_words, 1):
            relevance = self.theme_matched_words.get(word, {}).get('relevance', 0)
            print(f"  {i}. {word.upper():<15} ({len(path)} cells, {relevance:.0%} relevance)")
        
        # Statistics
        total_cells = self.rows * self.cols
        print(f"\n✓ Total cells used: {result['cells_used']}/{total_cells} ({result['cells_used']/total_cells:.0%})")
        print("✓ All words form valid paths")
    
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
        if reason:
            print(f"\n  ❌ Why it failed: {reason}")
        
        print(f"\n  💡 Suggestions:")
        print(f"     1. Theme interpretation might need adjustment")
        print(f"     2. Some theme words may be incorrect")
        print(f"     3. Try different word combinations manually")

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
        Used for visualization/debugging
        
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


    def _analyze_leftover_regions(self, leftover_cells, placed_words, theme):
        """
        Analyze disconnected leftover regions and select theme-appropriate words
        
        For each disconnected region:
        1. Find all valid words via DFS
        2. Use API to select which word(s) fit the theme
        
        Args:
            leftover_cells: Set of unused cells
            placed_words: List of (word, path) tuples already placed
            theme: Puzzle theme string
            
        Returns:
            List of (word, path) tuples that should fill the leftover regions
        """
        if not leftover_cells:
            return []
        
        # Find disconnected regions
        regions = self._find_connected_components(leftover_cells)
        
        if not regions:
            return []
        
        print(f"\n  🔍 Analyzing {len(regions)} leftover region(s)...")
        
        selected_words = []
        
        # Process each region separately
        for i, region in enumerate(regions, 1):
            print(f"\n    Region {i}: {len(region)} cells")
            
            # Get the letters in this region
            region_letters = [self.grid[r][c] for r, c in region]
            print(f"      Letters: {', '.join(region_letters)}")
            
            # Find all valid words in this region
            region_words = self._find_words_in_cells(region, min_length=4)
            
            if not region_words:
                print(f"      ⚠️  No valid words found in region")
                continue
            
            # Show found words
            sorted_region_words = sorted(region_words.keys(), key=len, reverse=True)
            print(f"      Found {len(region_words)} valid words:")
            for word in sorted_region_words[:10]:
                print(f"        • {word.upper()} ({len(word)} letters)")
            
            # Ask API to select which word(s) fit the theme
            selected_region_words = self._select_words_for_region(
                region_words=region_words,
                region_letters=region_letters,
                placed_words=placed_words,
                theme=theme,
                region_size=len(region)
            )
            
            if selected_region_words:
                for word, path in selected_region_words:
                    print(f"      ✓ Selected: {word.upper()} ({len(path)} cells)")
                    selected_words.append((word, path))
            else:
                print(f"      ✗ API couldn't select theme-appropriate word(s)")
        
        return selected_words

    def _select_words_for_region(self, region_words, region_letters, placed_words, theme, region_size):
        """
        Use API to select which word(s) from a region fit the theme
        
        Args:
            region_words: Dict of {word: [paths]} found in the region
            region_letters: List of letters available in the region
            placed_words: List of (word, path) already placed
            theme: Puzzle theme string
            region_size: Number of cells in the region
            
        Returns:
            List of (word, path) tuples, or empty list
        """
        # Prepare word list with letter usage info
        word_list = list(region_words.keys())
        word_list.sort(key=len, reverse=True)
        
        # Build word descriptions showing letter usage
        word_descriptions = []
        for word in word_list[:20]:
            word_len = len(word)
            if word_len == region_size:
                word_descriptions.append(f"  - {word.upper()} ({word_len} letters) - uses all letters")
            else:
                unused_count = region_size - word_len
                word_descriptions.append(f"  - {word.upper()} ({word_len} letters) - leaves {unused_count} letters unused")
        
        # Get already placed word names
        placed_word_names = [w.upper() for w, _ in placed_words]
        
        # Build prompt
        prompt = f"""You are solving a NYT Strands puzzle with theme: "{theme}"

Already placed theme words: {', '.join(placed_word_names)}

I have a leftover region with {region_size} unused cells containing these letters:
{', '.join(region_letters)}

Valid words that can be formed from these letters:
{chr(10).join(word_descriptions)}

Which word(s) should fill this region to complete the theme?

Requirements:
- Selected word(s) must use ALL {region_size} letters in the region
- Words must fit the theme
- Letters cannot be reused

Respond with VALID JSON ONLY:
{{
"selected_words": ["word1", "word2"],
"reasoning": "why these words fit the theme and use all letters"
}}

If only one word is needed, return a list with one word.
If multiple words are needed to use all {region_size} letters, return multiple words."""

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
                print(f"        ⚠️  API error: {response.status_code}")
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
            selected_words = result.get('selected_words', [])
            reasoning = result.get('reasoning', '')
            
            print(f"        API reasoning: {reasoning}")
            
            # Validate and return selected words with their paths
            validated_words = []
            for word_str in selected_words:
                word_lower = word_str.lower()
                if word_lower in region_words:
                    # Take the first path for this word
                    validated_words.append((word_lower, region_words[word_lower][0]))
                else:
                    print(f"        ⚠️  API selected '{word_str}' which is not in region")
            
            return validated_words
            
        except Exception as e:
            print(f"        ⚠️  Error selecting words: {e}")
            import traceback
            traceback.print_exc()
            return []

    """
    COMPLETE INTEGRATION GUIDE FOR SPANGRAM IDENTIFICATION
    ======================================================

    This guide shows exactly where to add code to your solver.py file.
    """

    # ==============================================================================
    # STEP 1: Add new methods after _select_words_for_region() (around line 1550)
    # ==============================================================================

    # ADD THESE 6 METHODS RIGHT AFTER _select_words_for_region() ends:

    def identify_spangram(self, placed_words, expected_count, theme):
        """
        Identify which word(s) form the spangram from placed words
        
        A spangram must:
        1. Span opposite edges (top-to-bottom OR left-to-right)
        2. Be physically connected if multi-word
        
        Args:
            placed_words: List of (word, path) tuples - all placed words
            expected_count: Expected total word count (e.g., 8)
            theme: Puzzle theme string
            
        Returns:
            Dict with:
            - 'spangram_words': List of (word, path) forming spangram
            - 'theme_words': List of (word, path) that are NOT spangram
            - 'spangram_type': 'single' or 'multi-word'
            - 'reasoning': Why this is the spangram
        """
        print(f"\n{'='*70}")
        print("PHASE 2: SPANGRAM IDENTIFICATION")
        print(f"{'='*70}")
        
        total_placed = len(placed_words)
        
        # Calculate how many words form the spangram
        # If we placed 9 words but expected 8, then 2 words form a multi-word spangram
        # Formula: spangram_word_count = total_placed - expected_count + 1
        spangram_word_count = total_placed - expected_count + 1
        
        print(f"\n📊 Analysis:")
        print(f"   Total words placed: {total_placed}")
        print(f"   Expected word count: {expected_count}")
        print(f"   Spangram word count: {spangram_word_count}")
        
        if spangram_word_count < 1:
            print(f"\n⚠️  Error: Calculated spangram word count is {spangram_word_count}")
            print(f"   This shouldn't happen - returning first word as spangram")
            return {
                'spangram_words': [placed_words[0]],
                'theme_words': placed_words[1:],
                'spangram_type': 'single',
                'reasoning': 'Error in calculation - defaulted to first word'
            }
        
        # Find all valid spangram candidates
        print(f"\n🔍 Finding spangram candidates ({spangram_word_count} word(s))...")
        
        candidates = self._find_spangram_candidates(placed_words, spangram_word_count)
        
        print(f"   Found {len(candidates)} valid candidate(s)")
        
        if len(candidates) == 0:
            print(f"\n❌ No valid spangram found!")
            print(f"   This shouldn't happen if solution is correct")
            return {
                'spangram_words': [],
                'theme_words': placed_words,
                'spangram_type': 'unknown',
                'reasoning': 'No valid spangram candidates found'
            }
        
        if len(candidates) == 1:
            # Only one candidate - this is our spangram
            selected = candidates[0]
            spangram_words = selected['words']
            theme_words = [w for w in placed_words if w not in spangram_words]
            
            print(f"\n✅ Spangram identified (only one valid candidate):")
            for word, path in spangram_words:
                print(f"   • {word.upper()} ({len(path)} cells)")
            
            return {
                'spangram_words': spangram_words,
                'theme_words': theme_words,
                'spangram_type': selected['type'],
                'reasoning': selected['reasoning']
            }
        
        # Multiple candidates - use API to select the best one
        print(f"\n🤖 Multiple candidates found - using API to select best...")
        
        selected = self._select_best_spangram(candidates, theme, placed_words)
        
        if selected:
            spangram_words = selected['words']
            theme_words = [w for w in placed_words if w not in spangram_words]
            
            print(f"\n✅ Spangram identified via API:")
            for word, path in spangram_words:
                print(f"   • {word.upper()} ({len(path)} cells)")
            print(f"   Reasoning: {selected['reasoning']}")
            
            return {
                'spangram_words': spangram_words,
                'theme_words': theme_words,
                'spangram_type': selected['type'],
                'reasoning': selected['reasoning']
            }
        else:
            # API failed - default to first candidate
            print(f"\n⚠️  API selection failed - using first candidate")
            selected = candidates[0]
            spangram_words = selected['words']
            theme_words = [w for w in placed_words if w not in spangram_words]
            
            return {
                'spangram_words': spangram_words,
                'theme_words': theme_words,
                'spangram_type': selected['type'],
                'reasoning': 'API selection failed - defaulted to first candidate'
            }


    def _find_spangram_candidates(self, placed_words, spangram_word_count):
        """
        Find all valid spangram candidates
        
        Args:
            placed_words: List of (word, path) tuples
            spangram_word_count: How many words form the spangram
            
        Returns:
            List of candidate dicts with:
            - 'words': List of (word, path) tuples
            - 'type': 'single' or 'multi-word'
            - 'reasoning': Why this is a valid spangram
        """
        candidates = []
        
        # Try all combinations of N words
        for word_combo in itertools.combinations(placed_words, spangram_word_count):
            # Check if this combination forms a valid spangram
            is_valid, reasoning = self._is_valid_spangram(word_combo)
            
            if is_valid:
                candidates.append({
                    'words': list(word_combo),
                    'type': 'single' if spangram_word_count == 1 else 'multi-word',
                    'reasoning': reasoning
                })
        
        return candidates


    def _is_valid_spangram(self, word_combo):
        """
        Check if a combination of words forms a valid spangram
        
        Args:
            word_combo: Tuple of (word, path) tuples
            
        Returns:
            (is_valid, reasoning) tuple
        """
        # Get all cells used by these words
        all_cells = set()
        for word, path in word_combo:
            all_cells.update(path)
        
        # Check 1: If multi-word, are they physically connected?
        if len(word_combo) > 1:
            is_connected = self._are_words_connected(word_combo)
            if not is_connected:
                return False, "Words are not physically connected"
        
        # Check 2: Do they span opposite edges?
        spans_edges, edge_info = self._spans_opposite_edges(all_cells)
        
        if not spans_edges:
            return False, "Does not span opposite edges"
        
        # Valid spangram!
        word_names = " + ".join([w.upper() for w, _ in word_combo])
        return True, f"Spans {edge_info}, words: {word_names}"


    def _are_words_connected(self, word_combo):
        """
        Check if words in a multi-word combination are physically connected
        
        Multi-word spangrams must connect: last letter of word N is adjacent to 
        first letter of word N+1
        
        Args:
            word_combo: Tuple of (word, path) tuples
            
        Returns:
            Boolean - True if all words are connected in sequence
        """
        # For each pair of consecutive words
        for i in range(len(word_combo) - 1):
            word1, path1 = word_combo[i]
            word2, path2 = word_combo[i + 1]
            
            # Get last cell of word1 and first cell of word2
            last_cell_word1 = path1[-1]
            first_cell_word2 = path2[0]
            
            # Check if they're adjacent (8-directional)
            if not self._are_cells_adjacent(last_cell_word1, first_cell_word2):
                # Try the reverse - maybe word2 connects to word1
                last_cell_word2 = path2[-1]
                first_cell_word1 = path1[0]
                
                if not self._are_cells_adjacent(last_cell_word2, first_cell_word1):
                    return False
        
        return True


    def _are_cells_adjacent(self, cell1, cell2):
        """
        Check if two cells are adjacent (8-directional)
        
        Args:
            cell1, cell2: (row, col) tuples
            
        Returns:
            Boolean
        """
        r1, c1 = cell1
        r2, c2 = cell2
        
        # Adjacent if within 1 step in any direction
        return abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1 and (r1, c1) != (r2, c2)


    def _spans_opposite_edges(self, cells):
        """
        Check if a set of cells spans opposite edges of the grid
        
        Args:
            cells: Set of (row, col) tuples
            
        Returns:
            (spans, edge_description) tuple
        """
        # Get min/max positions
        rows = [r for r, c in cells]
        cols = [c for r, c in cells]
        
        min_row = min(rows)
        max_row = max(rows)
        min_col = min(cols)
        max_col = max(cols)
        
        # Check if spans top-to-bottom
        touches_top = (min_row == 0)
        touches_bottom = (max_row == self.rows - 1)
        spans_vertical = touches_top and touches_bottom
        
        # Check if spans left-to-right
        touches_left = (min_col == 0)
        touches_right = (max_col == self.cols - 1)
        spans_horizontal = touches_left and touches_right
        
        if spans_vertical and spans_horizontal:
            return True, "top-to-bottom AND left-to-right"
        elif spans_vertical:
            return True, "top-to-bottom"
        elif spans_horizontal:
            return True, "left-to-right"
        else:
            return False, "does not span opposite edges"


    def _select_best_spangram(self, candidates, theme, all_words):
        """
        Use API to select the best spangram from multiple candidates
        
        Args:
            candidates: List of candidate dicts
            theme: Puzzle theme
            all_words: All placed words
            
        Returns:
            Selected candidate dict or None
        """
        # Build candidate descriptions
        candidate_descriptions = []
        for i, cand in enumerate(candidates, 1):
            words = [w.upper() for w, _ in cand['words']]
            word_str = " + ".join(words) if len(words) > 1 else words[0]
            candidate_descriptions.append(f"{i}. {word_str} - {cand['reasoning']}")
        
        # Get all word names
        all_word_names = [w.upper() for w, _ in all_words]
        
        prompt = f"""You are solving a NYT Strands puzzle with theme: "{theme}"

All words in solution: {', '.join(all_word_names)}

Multiple valid spangram candidates were found. Which one best represents the theme as a spangram?

Candidates:
{chr(10).join(candidate_descriptions)}

The spangram should:
1. Be a word/phrase that encompasses or represents the theme
2. Span opposite edges of the grid (already verified)
3. Be distinct from the regular theme words

Respond with VALID JSON ONLY:
{{
  "selected_index": 1,
  "reasoning": "why this is the best spangram for this theme"
}}"""

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
                    "max_tokens": 1000,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"   ⚠️  API error: {response.status_code}")
                return None
            
            data = response.json()
            response_text = data['content'][0]['text'].strip()
            
            # Extract JSON
            if not response_text.startswith('{'):
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    response_text = response_text[start_idx:end_idx+1]
            
            result = json.loads(response_text)
            selected_idx = result.get('selected_index', 1)
            reasoning = result.get('reasoning', '')
            
            # Validate index
            if 1 <= selected_idx <= len(candidates):
                selected = candidates[selected_idx - 1]
                selected['reasoning'] = reasoning
                return selected
            else:
                print(f"   ⚠️  Invalid index {selected_idx}")
                return None
            
        except Exception as e:
            print(f"   ⚠️  Error selecting spangram: {e}")
            return None


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
# SECTION 3: FILE I/O AND MAIN EXECUTION
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
    
    # Theme matching and solving
    if use_theme_matching and puzzle_info.get('theme'):
        word_count = puzzle_info.get('word_count', 8)
        
        # Use the adaptive clustering solver
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
    
    print("Strands Solver (No Spangram Version)")
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