"""
Phase 2: Spangram Finder
Takes theme words identified in Phase 1 and finds valid spangrams

A valid spangram must:
1. Touch opposite edges (left AND right, OR top AND bottom)
2. Be composed of words with adjacent letters between them
3. Use all remaining cells after theme words are placed

Author: Spangram identification module
"""

import itertools
from typing import List, Set, Tuple, Dict, Optional


class SpangramFinder:
    """Finds multi-word spangrams from Phase 1 theme words"""
    
    def __init__(self, solver):
        """
        Initialize with a solver instance from Phase 1
        
        Args:
            solver: StrandsSolver instance with found_words populated
        """
        self.solver = solver
        self.grid = solver.grid
        self.rows = solver.rows
        self.cols = solver.cols
        self.found_words = solver.found_words
    
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
    
    def is_spangram(self, combined_cells: Set[Tuple[int, int]]) -> bool:
        """
        Check if a set of cells forms a spangram (touches opposite edges)
        
        Args:
            combined_cells: Set of (row, col) tuples
            
        Returns:
            True if touches opposite edges
        """
        if not combined_cells:
            return False
        
        rows = [cell[0] for cell in combined_cells]
        cols = [cell[1] for cell in combined_cells]
        
        # Check if touches top AND bottom
        touches_top = 0 in rows
        touches_bottom = (self.rows - 1) in rows
        
        # Check if touches left AND right
        touches_left = 0 in cols
        touches_right = (self.cols - 1) in cols
        
        # Must touch BOTH opposite edges
        return (touches_top and touches_bottom) or (touches_left and touches_right)
    
    def words_are_adjacent(self, word1_path: List[Tuple[int, int]], 
                          word2_path: List[Tuple[int, int]]) -> bool:
        """
        Check if two word paths are adjacent (end of word1 touches start of word2)
        
        Args:
            word1_path: Path of first word [(r,c), ...]
            word2_path: Path of second word [(r,c), ...]
            
        Returns:
            True if last letter of word1 is adjacent to first letter of word2
        """
        if not word1_path or not word2_path:
            return False
        
        end_of_word1 = word1_path[-1]
        start_of_word2 = word2_path[0]
        
        # Check if start_of_word2 is in the 8 neighbors of end_of_word1
        neighbors = self.get_neighbors(end_of_word1[0], end_of_word1[1])
        return start_of_word2 in neighbors
    
    def find_multiword_spangrams(self, theme_words: List[str], 
                                 max_words_in_spangram: int = 3) -> List[Dict]:
        """
        Find all valid multi-word spangrams from theme words
        
        Args:
            theme_words: List of word strings identified in Phase 1
            max_words_in_spangram: Maximum number of words to combine (default 3)
            
        Returns:
            List of dicts with spangram info:
            {
                'words': ['word1', 'word2'],
                'paths': [path1, path2],
                'combined_name': 'WORD1_WORD2',
                'total_cells': int,
                'touches': 'top-bottom' or 'left-right'
            }
        """
        print("\n" + "="*70)
        print("PHASE 2: SPANGRAM FINDER")
        print("="*70)
        print(f"Searching for multi-word spangrams from {len(theme_words)} theme words")
        print(f"Testing combinations of 2-{max_words_in_spangram} words\n")
        
        all_spangrams = []
        
        # Try different numbers of words (2, 3, etc.)
        for n_words in range(2, max_words_in_spangram + 1):
            print(f"🔍 Testing {n_words}-word combinations...")
            
            combos_tested = 0
            valid_found = 0
            
            # Try all combinations of n_words
            for word_combo in itertools.combinations(theme_words, n_words):
                combos_tested += 1
                
                # Get all possible path permutations for these words
                # (order matters: STAIN_REMOVAL != REMOVAL_STAIN)
                for word_order in itertools.permutations(word_combo):
                    # Try to find a valid path combination
                    result = self._try_word_combination(word_order)
                    
                    if result:
                        valid_found += 1
                        all_spangrams.append(result)
                        
                        # Show what we found
                        combined_name = result['combined_name']
                        total_cells = result['total_cells']
                        touches = result['touches']
                        print(f"   ✓ Found: {combined_name} ({total_cells} cells, {touches})")
            
            print(f"   Tested {combos_tested} combinations, found {valid_found} valid spangrams\n")
        
        print(f"="*70)
        print(f"Total spangrams found: {len(all_spangrams)}")
        print(f"="*70)
        
        return all_spangrams
    
    def _try_word_combination(self, word_order: Tuple[str, ...]) -> Optional[Dict]:
        """
        Try a specific ordering of words to see if they form a valid spangram
        
        Args:
            word_order: Tuple of words in order ('word1', 'word2', ...)
            
        Returns:
            Dict with spangram info if valid, None otherwise
        """
        # Get all paths for each word
        word_paths = []
        for word in word_order:
            word_lower = word.lower()
            if word_lower not in self.found_words:
                return None
            word_paths.append(self.found_words[word_lower])
        
        # Try all combinations of paths (one path per word)
        for path_combo in itertools.product(*word_paths):
            # Check constraints:
            # 1. No overlapping cells between words
            # 2. Adjacent connections between consecutive words
            # 3. Combined cells form a spangram
            
            all_cells = set()
            paths_list = []
            valid = True
            
            for i, path in enumerate(path_combo):
                path_cells = set(path)
                
                # Check for overlap with previous words
                if path_cells & all_cells:
                    valid = False
                    break
                
                # Check adjacency with previous word
                if i > 0:
                    prev_path = path_combo[i-1]
                    if not self.words_are_adjacent(prev_path, path):
                        valid = False
                        break
                
                all_cells.update(path_cells)
                paths_list.append(path)
            
            if not valid:
                continue
            
            # Check if combined cells form a spangram
            if not self.is_spangram(all_cells):
                continue
            
            # SUCCESS! Found a valid multi-word spangram
            combined_name = '_'.join([w.upper() for w in word_order])
            
            # Determine which edges it touches
            rows = [cell[0] for cell in all_cells]
            cols = [cell[1] for cell in all_cells]
            
            touches_top = 0 in rows
            touches_bottom = (self.rows - 1) in rows
            touches_left = 0 in cols
            touches_right = (self.cols - 1) in cols
            
            if touches_top and touches_bottom:
                touches = 'top-bottom'
            else:
                touches = 'left-right'
            
            return {
                'words': list(word_order),
                'paths': paths_list,
                'combined_name': combined_name,
                'total_cells': len(all_cells),
                'touches': touches,
                'cells': all_cells
            }
        
        return None
    
    def identify_spangram_from_phase1(self, phase1_words: List[str], 
                                      expected_theme_count: int = 8,
                                      max_spangram_words: int = None) -> Optional[Dict]:
        """
        Main method: Identify the spangram from Phase 1 words
        
        Logic:
        - Phase 1 identified N words (e.g., 9 words)
        - We need M solutions (e.g., 8 theme word solutions)
        - If N > M, then (N - M + 1) words must combine into the spangram
        - Example: 9 words, 8 solutions → 2 words form spangram
          (7 single-word solutions + 1 two-word spangram = 8 total)
        
        Args:
            phase1_words: Words identified by Phase 1 solver
            expected_theme_count: Number of expected theme word solutions
            max_spangram_words: Max words to try (if None, calculated from difference)
            
        Returns:
            Dict with the identified spangram, or None if not found
        """
        print("\n" + "="*70)
        print("IDENTIFYING SPANGRAM FROM PHASE 1 RESULTS")
        print("="*70)
        print(f"Phase 1 identified: {len(phase1_words)} words")
        print(f"Expected solutions: {expected_theme_count}")
        print(f"Words identified: {', '.join([w.upper() for w in phase1_words])}\n")
        
        # Calculate how many words should be in the spangram
        # Formula: If we have N words but need M solutions,
        #          then (N - M + 1) words must combine into 1 spangram
        words_in_spangram = len(phase1_words) - expected_theme_count + 1
        
        if words_in_spangram < 2:
            print("⚠️  Not enough extra words for a multi-word spangram")
            print(f"   Phase 1 found {len(phase1_words)} words, expected {expected_theme_count} solutions")
            if len(phase1_words) == expected_theme_count:
                print("   All words appear to be single-word theme words")
            elif len(phase1_words) < expected_theme_count:
                print("   ❌ Phase 1 didn't find enough words!")
            return None
        
        print(f"Calculation: {len(phase1_words)} words - {expected_theme_count} solutions + 1 = {words_in_spangram} words in spangram")
        print(f"This means: {expected_theme_count - 1} single-word solutions + 1 {words_in_spangram}-word spangram = {expected_theme_count} total\n")
        
        if max_spangram_words is None:
            max_spangram_words = words_in_spangram
        else:
            max_spangram_words = min(max_spangram_words, words_in_spangram)
        
        # Find all multi-word spangrams
        all_spangrams = self.find_multiword_spangrams(
            phase1_words, 
            max_words_in_spangram=words_in_spangram
        )
        
        # Filter to spangrams with the correct number of words
        correct_length_spangrams = [
            s for s in all_spangrams 
            if len(s['words']) == words_in_spangram
        ]
        
        if not correct_length_spangrams:
            print(f"\n❌ No {words_in_spangram}-word spangrams found")
            return None
        
        print(f"\n✅ Found {len(correct_length_spangrams)} candidate spangram(s):")
        for i, spangram in enumerate(correct_length_spangrams, 1):
            print(f"   {i}. {spangram['combined_name']} "
                  f"({spangram['total_cells']} cells, {spangram['touches']})")
        
        # Return the first valid one (or you could rank them)
        return correct_length_spangrams[0]
    
    def visualize_spangram(self, spangram: Dict):
        """
        Visualize the spangram on the grid
        
        Args:
            spangram: Dict with spangram info from find_multiword_spangrams()
        """
        print("\n" + "="*70)
        print(f"SPANGRAM: {spangram['combined_name']}")
        print("="*70)
        
        # Create cell-to-word mapping
        cell_to_word_idx = {}
        for i, path in enumerate(spangram['paths']):
            for cell in path:
                cell_to_word_idx[cell] = i
        
        # Print grid with word indicators
        print("\nGrid visualization:")
        for row in range(self.rows):
            row_str = ""
            for col in range(self.cols):
                cell = (row, col)
                letter = self.grid[row][col]
                
                if cell in cell_to_word_idx:
                    # Part of spangram - show with letter
                    row_str += f" {letter} "
                else:
                    # Not part of spangram - show lowercase
                    row_str += f" {letter.lower()} "
            print(row_str)
        
        # Show word breakdown
        print(f"\nWord breakdown:")
        for i, (word, path) in enumerate(zip(spangram['words'], spangram['paths']), 1):
            print(f"  {i}. {word.upper():<12} ({len(path)} cells)")
        
        print(f"\nTotal cells: {spangram['total_cells']}")
        print(f"Touches: {spangram['touches']}")
        
        # Show adjacency connections
        print(f"\nAdjacency connections:")
        for i in range(len(spangram['paths']) - 1):
            word1 = spangram['words'][i]
            word2 = spangram['words'][i + 1]
            path1 = spangram['paths'][i]
            path2 = spangram['paths'][i + 1]
            
            end_cell = path1[-1]
            start_cell = path2[0]
            
            end_letter = self.grid[end_cell[0]][end_cell[1]]
            start_letter = self.grid[start_cell[0]][start_cell[1]]
            
            print(f"  {word1.upper()}[{end_letter}] → {word2.upper()}[{start_letter}]")


def find_spangram_from_solver(solver, phase1_words: List[str], 
                               expected_theme_count: int = 8) -> Optional[Dict]:
    """
    Convenience function to find spangram from a solver instance
    
    Args:
        solver: StrandsSolver instance from Phase 1
        phase1_words: Words identified in Phase 1
        expected_theme_count: Expected number of theme words
        
    Returns:
        Spangram dict or None
    """
    finder = SpangramFinder(solver)
    spangram = finder.identify_spangram_from_phase1(phase1_words, expected_theme_count)
    
    if spangram:
        finder.visualize_spangram(spangram)
    
    return spangram


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example of how to use this with Phase 1 solver
    """
    print("Phase 2: Spangram Finder")
    print("="*70)
    print("\nThis module should be used after Phase 1 has identified theme words.")
    print("\nExample usage:")
    print("""
from solver_no_spangram import StrandsSolver, solve_from_json
from spangram_finder import find_spangram_from_solver

# Phase 1: Find theme words
solver = solve_from_json('puzzle.json')

# Assume Phase 1 identified these words:
phase1_words = ['spray', 'stain', 'removal', 'scrub', 'soak', 
                'bleach', 'steam', 'launder', 'blot']

# Phase 2: Find the spangram
spangram = find_spangram_from_solver(
    solver, 
    phase1_words, 
    expected_theme_count=8
)

if spangram:
    print(f"Spangram identified: {spangram['combined_name']}")
    print(f"Composed of: {', '.join(spangram['words'])}")
""")