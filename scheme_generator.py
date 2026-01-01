#!/usr/bin/env python3
"""
Crucigram Puzzle Generator
==========================
Generates dense crossword-style puzzles where:
- Every horizontal/vertical sequence of 2+ letters is a valid word
- All letters form one connected group
- 30-34 letters in a 7x7 grid
- Maximum 3 two-letter words
- No 3x3 black blocks
- No repeated words
"""

import random
from typing import Optional
from collections import Counter
from pathlib import Path
from wordfreq import zipf_frequency


# ============================================================================
# DICTIONARY SETUP
# ============================================================================

# Word rarity thresholds (Zipf scale: 0=rare, 7=extremely common)
RARITY_THRESHOLDS = {
    'easy': 3.0,    # Common words (Zipf 3-7)
    'medium': 2.0,  # Includes less common words like "abstruse"
    'hard': 1.0,    # Includes obscure words like "aalii"
}
DEFAULT_UNKNOWN_ZIPF = 1.5  # Words not in wordfreq - treated as hard-only

# Common 2-letter words that most people would know
COMMON_TWO_LETTER = {
    'AM', 'AN', 'AS', 'AT', 'BE', 'BY', 'DO', 'GO', 'HE', 'IF', 'IN', 'IS', 'IT',
    'ME', 'MY', 'NO', 'OF', 'ON', 'OR', 'SO', 'TO', 'UP', 'US', 'WE', 'AD', 'AH',
    'AX', 'HA', 'HI', 'HO', 'ID', 'LA', 'LO', 'MA', 'OH', 'OK', 'OX', 'PA', 'PI',
    'RE', 'MI', 'FA', 'SI', 'AA'
}


def load_blacklist() -> set[str]:
    """Load blacklisted words from file."""
    path = Path(__file__).parent / 'data' / 'blacklist.txt'
    if path.exists():
        with open(path) as f:
            return {line.strip().upper() for line in f if line.strip()}
    return set()


def load_dictionary(common_only: bool = False, blacklist: set[str] | None = None) -> set[str]:
    """Load and filter the word dictionary from data/words.txt."""
    words_path = Path(__file__).parent / 'data' / 'words.txt'
    blacklist = blacklist or set()

    with open(words_path) as f:
        all_words = {line.strip() for line in f if line.strip()}

    # Filter words
    valid_words = set()
    for word in all_words:
        # Skip blacklisted words
        if word in blacklist:
            continue
        # For 2-letter words, optionally filter to common ones
        if len(word) == 2:
            if common_only:
                if word in COMMON_TWO_LETTER:
                    valid_words.add(word)
            else:
                valid_words.add(word)
        else:
            valid_words.add(word)

    return valid_words


# ============================================================================
# GRID CLASS
# ============================================================================

class CrucigramGrid:
    def __init__(self, size: int = 7):
        self.size = size
        # None = undecided, letter = filled, '#' = black square
        self.grid: list[list[Optional[str]]] = [[None] * size for _ in range(size)]
        self.dictionary: set[str] = set()
        
    def copy(self) -> 'CrucigramGrid':
        """Create a deep copy of the grid."""
        new_grid = CrucigramGrid(self.size)
        new_grid.grid = [row[:] for row in self.grid]
        new_grid.dictionary = self.dictionary  # Share reference (immutable)
        return new_grid
    
    def get(self, row: int, col: int) -> Optional[str]:
        """Get cell value, or '#' if out of bounds."""
        if 0 <= row < self.size and 0 <= col < self.size:
            return self.grid[row][col]
        return '#'  # Treat out-of-bounds as black
    
    def set(self, row: int, col: int, value: Optional[str]):
        """Set cell value."""
        if 0 <= row < self.size and 0 <= col < self.size:
            self.grid[row][col] = value
    
    def count_letters(self) -> int:
        """Count total letters placed."""
        count = 0
        for row in self.grid:
            for cell in row:
                if cell is not None and cell != '#':
                    count += 1
        return count
    
    def get_letter_inventory(self) -> dict[str, int]:
        """Get count of each letter."""
        inventory = Counter()
        for row in self.grid:
            for cell in row:
                if cell is not None and cell != '#':
                    inventory[cell] += 1
        return dict(sorted(inventory.items()))
    
    # ------------------------------------------------------------------------
    # WORD EXTRACTION
    # ------------------------------------------------------------------------
    
    def get_all_words(self) -> list[tuple[str, int, int, str]]:
        """
        Extract all words (horizontal and vertical sequences of 2+ letters).
        Returns list of (word, row, col, direction).
        """
        words = []
        
        # Horizontal words
        for row in range(self.size):
            col = 0
            while col < self.size:
                # Skip blacks/empty
                while col < self.size and (self.grid[row][col] is None or self.grid[row][col] == '#'):
                    col += 1
                if col >= self.size:
                    break
                # Collect word
                start_col = col
                word = ""
                while col < self.size and self.grid[row][col] is not None and self.grid[row][col] != '#':
                    word += self.grid[row][col]
                    col += 1
                if len(word) >= 2:
                    words.append((word, row, start_col, "ACROSS"))
        
        # Vertical words
        for col in range(self.size):
            row = 0
            while row < self.size:
                # Skip blacks/empty
                while row < self.size and (self.grid[row][col] is None or self.grid[row][col] == '#'):
                    row += 1
                if row >= self.size:
                    break
                # Collect word
                start_row = row
                word = ""
                while row < self.size and self.grid[row][col] is not None and self.grid[row][col] != '#':
                    word += self.grid[row][col]
                    row += 1
                if len(word) >= 2:
                    words.append((word, start_row, col, "DOWN"))
        
        return words
    
    def get_word_strings(self) -> list[str]:
        """Get just the word strings."""
        return [w[0] for w in self.get_all_words()]
    
    # ------------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------------
    
    def all_words_valid(self) -> bool:
        """Check if all current words are in dictionary."""
        for word, _, _, _ in self.get_all_words():
            if word not in self.dictionary:
                return False
        return True
    
    def count_two_letter_words(self) -> int:
        """Count words of exactly 2 letters."""
        return sum(1 for w in self.get_word_strings() if len(w) == 2)
    
    def has_repeated_words(self) -> bool:
        """Check for duplicate words."""
        words = self.get_word_strings()
        return len(words) != len(set(words))
    
    def has_3x3_black_block(self) -> bool:
        """Check for any 3x3 region of all black squares."""
        for row in range(self.size - 2):
            for col in range(self.size - 2):
                all_black = True
                for dr in range(3):
                    for dc in range(3):
                        cell = self.grid[row + dr][col + dc]
                        if cell != '#':
                            all_black = False
                            break
                    if not all_black:
                        break
                if all_black:
                    return True
        return False
    
    def is_connected(self) -> bool:
        """Check if all letters form one connected group (4-directional)."""
        # Find first letter
        start = None
        letter_cells = set()
        for row in range(self.size):
            for col in range(self.size):
                cell = self.grid[row][col]
                if cell is not None and cell != '#':
                    letter_cells.add((row, col))
                    if start is None:
                        start = (row, col)
        
        if not letter_cells:
            return True  # Empty is technically connected
        
        # BFS from start
        visited = set()
        queue = [start]
        while queue:
            r, c = queue.pop(0)
            if (r, c) in visited:
                continue
            visited.add((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in letter_cells and (nr, nc) not in visited:
                    queue.append((nr, nc))
        
        return visited == letter_cells
    
    def has_both_directions(self) -> bool:
        """Check for at least one across and one down word."""
        words = self.get_all_words()
        has_across = any(d == "ACROSS" for _, _, _, d in words)
        has_down = any(d == "DOWN" for _, _, _, d in words)
        return has_across and has_down
    
    def is_valid_complete(self, min_letters: int = 30, max_letters: int = 34, max_two_letter: int = 3) -> bool:
        """Full validation for a completed grid."""
        letter_count = self.count_letters()
        if letter_count < min_letters or letter_count > max_letters:
            return False
        if not self.all_words_valid():
            return False
        if self.count_two_letter_words() > max_two_letter:
            return False
        if self.has_repeated_words():
            return False
        if self.has_3x3_black_block():
            return False
        if not self.is_connected():
            return False
        if not self.has_both_directions():
            return False
        return True
    
    # ------------------------------------------------------------------------
    # DISPLAY
    # ------------------------------------------------------------------------
    
    def to_string(self) -> str:
        """Generate ASCII art representation of the grid."""
        lines = []
        
        # Top border
        lines.append("┌" + "───┬" * (self.size - 1) + "───┐")
        
        for row_idx, row in enumerate(self.grid):
            # Cell contents
            cells = []
            for cell in row:
                if cell is None:
                    cells.append("   ")  # Empty/undecided
                elif cell == '#':
                    cells.append(" ■ ")  # Black square
                else:
                    cells.append(f" {cell} ")  # Letter
            lines.append("│" + "│".join(cells) + "│")
            
            # Row separator or bottom border
            if row_idx < self.size - 1:
                lines.append("├" + "───┼" * (self.size - 1) + "───┤")
            else:
                lines.append("└" + "───┴" * (self.size - 1) + "───┘")
        
        return "\n".join(lines)
    
    def print_summary(self):
        """Print the grid with word list and letter inventory."""
        print(self.to_string())
        print()
        
        # Word list grouped by direction
        words = self.get_all_words()
        across_words = [(w, r, c) for w, r, c, d in words if d == "ACROSS"]
        down_words = [(w, r, c) for w, r, c, d in words if d == "DOWN"]
        
        print("Words:")
        print("  ACROSS:", ", ".join(w for w, _, _ in sorted(across_words, key=lambda x: (x[1], x[2]))))
        print("  DOWN:  ", ", ".join(w for w, _, _ in sorted(down_words, key=lambda x: (x[1], x[2]))))
        print()
        
        # Letter inventory
        inventory = self.get_letter_inventory()
        print("Letters:")
        letter_parts = [f"{letter}×{count}" for letter, count in inventory.items()]
        print("  " + ", ".join(letter_parts))
        print(f"  Total: {self.count_letters()} letters")
        
        # Stats
        print()
        print(f"Stats:")
        print(f"  Total words: {len(words)}")
        print(f"  Two-letter words: {self.count_two_letter_words()}")


# ============================================================================
# GENERATOR
# ============================================================================

class CrucigramGenerator:
    def __init__(self, size: int = 7, min_letters: int = 30, max_letters: int = 34,
                 max_two_letter: int = 3, common_two_letter_only: bool = True,
                 word_rarity: str | None = None, use_blacklist: bool = True,
                 word_preference: str = 'common'):
        self.size = size
        self.min_letters = min_letters
        self.max_letters = max_letters
        self.max_two_letter = max_two_letter
        self.word_rarity = word_rarity
        self.word_preference = word_preference  # 'common', 'uncommon', or 'random'

        # Load blacklist and dictionary
        blacklist = load_blacklist() if use_blacklist else set()
        self.dictionary = load_dictionary(common_only=common_two_letter_only, blacklist=blacklist)

        # Compute word frequencies and filter by rarity threshold
        min_zipf = RARITY_THRESHOLDS.get(word_rarity, 0.0) if word_rarity else 0.0
        self.word_frequencies: dict[str, float] = {}

        filtered_dictionary = set()
        for word in self.dictionary:
            freq = zipf_frequency(word.lower(), 'en')
            if freq == 0:
                freq = DEFAULT_UNKNOWN_ZIPF
            self.word_frequencies[word] = freq
            if freq >= min_zipf:
                filtered_dictionary.add(word)

        self.dictionary = filtered_dictionary

        # Pre-compute words by length, sorted by frequency (descending)
        self.words_by_length: dict[int, list[str]] = {}
        for word in self.dictionary:
            length = len(word)
            if length not in self.words_by_length:
                self.words_by_length[length] = []
            self.words_by_length[length].append(word)

        # Sort/shuffle words based on preference
        for length in self.words_by_length:
            if word_preference == 'random':
                random.shuffle(self.words_by_length[length])
            else:
                # -1 = common first (descending), +1 = uncommon first (ascending)
                sign = 1 if word_preference == 'uncommon' else -1
                self.words_by_length[length].sort(
                    key=lambda w, s=sign: (s * self.word_frequencies.get(w, 0), random.random())
                )

        # Build letter->words index for faster crossing
        self.words_with_letter: dict[str, list[str]] = {}
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            words_list = [w for w in self.dictionary if letter in w and 3 <= len(w) <= 7]
            if word_preference == 'random':
                random.shuffle(words_list)
            else:
                sign = 1 if word_preference == 'uncommon' else -1
                words_list.sort(key=lambda w, s=sign: (s * self.word_frequencies.get(w, 0), random.random()))
            self.words_with_letter[letter] = words_list
    
    def generate(self, max_attempts: int = 1000, verbose: bool = True) -> Optional[CrucigramGrid]:
        """
        Generate a valid Crucigram puzzle.
        Uses a constructive approach with backtracking.
        """
        for attempt in range(max_attempts):
            grid = self._try_generate()
            if grid is not None:
                if verbose:
                    print(f"Success after {attempt + 1} attempts!")
                return grid
            if verbose and (attempt + 1) % 100 == 0:
                print(f"Attempt {attempt + 1}...")
        
        if verbose:
            print(f"Failed to generate after {max_attempts} attempts")
        return None
    
    def _try_generate(self) -> Optional[CrucigramGrid]:
        """Single attempt at generating a puzzle using iterative deepening."""
        grid = CrucigramGrid(self.size)
        grid.dictionary = self.dictionary
        
        # Start with a seed word (4-6 letters) placed in center, horizontal
        seed_length = random.randint(4, 6)
        seed_words = self.words_by_length.get(seed_length, [])
        if not seed_words:
            return None
        
        seed = random.choice(seed_words)
        start_row = self.size // 2
        start_col = (self.size - len(seed)) // 2
        
        for i, letter in enumerate(seed):
            grid.set(start_row, start_col + i, letter)
        
        # Iteratively expand by adding crossing words
        max_expansions = 100
        stall_count = 0
        
        for _ in range(max_expansions):
            expanded = self._try_expand(grid)
            
            if expanded:
                stall_count = 0
            else:
                stall_count += 1
                if stall_count > 10:
                    break
            
            # Check completion
            letter_count = grid.count_letters()
            if letter_count >= self.min_letters:
                # Try to finalize
                final = self._try_finalize(grid)
                if final:
                    return final
                if letter_count >= self.max_letters:
                    break
        
        return None
    
    def _try_expand(self, grid: CrucigramGrid) -> bool:
        """Try to add a crossing word to the grid."""
        # Find all potential crossing points (existing letters that could start/cross new words)
        crossing_points = []
        
        for row in range(self.size):
            for col in range(self.size):
                cell = grid.get(row, col)
                if cell is not None and cell != '#' and cell.isalpha():
                    # Can we add a vertical word through a horizontal letter?
                    if self._has_horizontal_word_at(grid, row, col):
                        # Check if vertical is free
                        above = grid.get(row - 1, col)
                        below = grid.get(row + 1, col)
                        if above is None or below is None:
                            crossing_points.append((row, col, cell, 'DOWN'))
                    
                    # Can we add a horizontal word through a vertical letter?
                    if self._has_vertical_word_at(grid, row, col):
                        left = grid.get(row, col - 1)
                        right = grid.get(row, col + 1)
                        if left is None or right is None:
                            crossing_points.append((row, col, cell, 'ACROSS'))
        
        random.shuffle(crossing_points)
        
        for row, col, letter, direction in crossing_points:
            if self._try_add_crossing_word(grid, row, col, letter, direction):
                return True
        
        return False
    
    def _has_horizontal_word_at(self, grid: CrucigramGrid, row: int, col: int) -> bool:
        """Check if this cell is part of a horizontal word (has letter neighbor left or right)."""
        left = grid.get(row, col - 1)
        right = grid.get(row, col + 1)
        return (left is not None and left != '#' and left.isalpha()) or \
               (right is not None and right != '#' and right.isalpha())
    
    def _has_vertical_word_at(self, grid: CrucigramGrid, row: int, col: int) -> bool:
        """Check if this cell is part of a vertical word (has letter neighbor above or below)."""
        above = grid.get(row - 1, col)
        below = grid.get(row + 1, col)
        return (above is not None and above != '#' and above.isalpha()) or \
               (below is not None and below != '#' and below.isalpha())
    
    def _try_add_crossing_word(self, grid: CrucigramGrid, row: int, col: int, letter: str, direction: str) -> bool:
        """Try to add a word that crosses through (row, col) with the given letter."""
        candidates = self.words_with_letter.get(letter, [])[:]  # Copy to avoid mutating
        # Sort/shuffle based on word preference
        if self.word_preference == 'random':
            random.shuffle(candidates)
        else:
            sign = 1 if self.word_preference == 'uncommon' else -1
            candidates.sort(key=lambda w: (sign * int(self.word_frequencies.get(w, 0)), random.random()))

        for word in candidates[:100]:  # Limit search
            # Find all positions where 'letter' appears in word
            positions = [i for i, c in enumerate(word) if c == letter]
            random.shuffle(positions)
            
            for pos in positions:
                if direction == 'DOWN':
                    start_row = row - pos
                    if self._can_place_vertical(grid, start_row, col, word):
                        if self._place_and_validate(grid, start_row, col, word, 'DOWN'):
                            return True
                else:  # ACROSS
                    start_col = col - pos
                    if self._can_place_horizontal(grid, row, start_col, word):
                        if self._place_and_validate(grid, row, start_col, word, 'ACROSS'):
                            return True
        
        return False
    
    def _can_place_horizontal(self, grid: CrucigramGrid, row: int, start_col: int, word: str) -> bool:
        """Check if word can be placed horizontally."""
        end_col = start_col + len(word)
        
        # Bounds check
        if start_col < 0 or end_col > self.size or row < 0 or row >= self.size:
            return False
        
        # Must have black/edge before and after
        before = grid.get(row, start_col - 1)
        after = grid.get(row, end_col)
        if before is not None and before != '#':
            return False
        if after is not None and after != '#':
            return False
        
        # Check each cell
        has_crossing = False
        for i, char in enumerate(word):
            col = start_col + i
            cell = grid.get(row, col)
            
            if cell == '#':
                return False
            if cell is not None and cell != char:
                return False
            if cell == char:
                has_crossing = True
            
            # Check for parallel adjacency issues
            if cell is None:
                above = grid.get(row - 1, col)
                below = grid.get(row + 1, col)
                # If there's a letter above or below, we'd be creating an adjacency
                # that needs to form valid words - we'll validate this later
        
        return has_crossing
    
    def _can_place_vertical(self, grid: CrucigramGrid, start_row: int, col: int, word: str) -> bool:
        """Check if word can be placed vertically."""
        end_row = start_row + len(word)
        
        # Bounds check
        if start_row < 0 or end_row > self.size or col < 0 or col >= self.size:
            return False
        
        # Must have black/edge before and after
        before = grid.get(start_row - 1, col)
        after = grid.get(end_row, col)
        if before is not None and before != '#':
            return False
        if after is not None and after != '#':
            return False
        
        # Check each cell
        has_crossing = False
        for i, char in enumerate(word):
            row = start_row + i
            cell = grid.get(row, col)
            
            if cell == '#':
                return False
            if cell is not None and cell != char:
                return False
            if cell == char:
                has_crossing = True
        
        return has_crossing
    
    def _place_and_validate(self, grid: CrucigramGrid, start_row: int, start_col: int, word: str, direction: str) -> bool:
        """Place word and check if all constraints still hold."""
        # Create backup
        backup = [row[:] for row in grid.grid]
        
        # Place the word
        for i, char in enumerate(word):
            if direction == 'ACROSS':
                grid.set(start_row, start_col + i, char)
            else:
                grid.set(start_row + i, start_col, char)
        
        # Validate
        valid = True
        
        # Check all words are valid
        if not grid.all_words_valid():
            valid = False
        
        # Check two-letter limit
        if valid and grid.count_two_letter_words() > self.max_two_letter:
            valid = False
        
        # Check no repeated words
        if valid and grid.has_repeated_words():
            valid = False
        
        if not valid:
            # Restore backup
            grid.grid = backup
            return False
        
        return True
    
    def _try_finalize(self, grid: CrucigramGrid) -> Optional[CrucigramGrid]:
        """Try to finalize the grid by filling blacks and validating."""
        final = grid.copy()
        
        # Fill all None with black
        for row in range(final.size):
            for col in range(final.size):
                if final.grid[row][col] is None:
                    final.grid[row][col] = '#'
        
        # Validate
        if final.is_valid_complete(self.min_letters, self.max_letters, self.max_two_letter):
            return final
        
        return None


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Crucigram puzzles')
    parser.add_argument('--size', type=int, default=7, help='Grid size (default: 7)')
    parser.add_argument('--min-letters', type=int, default=30, help='Minimum letters (default: 30)')
    parser.add_argument('--max-letters', type=int, default=34, help='Maximum letters (default: 34)')
    parser.add_argument('--max-two-letter', type=int, default=3, help='Max 2-letter words (default: 3)')
    parser.add_argument('--attempts', type=int, default=500, help='Max generation attempts (default: 500)')
    parser.add_argument('--allow-obscure-two-letter', action='store_true', help='Allow obscure 2-letter words')
    parser.add_argument('--word-rarity', choices=['easy', 'medium', 'hard'],
                        help='Word difficulty: easy (common), medium (standard), hard (obscure)')
    parser.add_argument('--word-preference', choices=['common', 'uncommon', 'random'], default='common',
                        help='Which words to try first: common (default), uncommon, or random')
    parser.add_argument('--no-blacklist', action='store_true', help='Disable offensive word filtering')
    args = parser.parse_args()
    
    print("Crucigram Puzzle Generator")
    print("=" * 50)
    print()
    print("Loading dictionary...")
    
    generator = CrucigramGenerator(
        size=args.size,
        min_letters=args.min_letters,
        max_letters=args.max_letters,
        max_two_letter=args.max_two_letter,
        common_two_letter_only=not args.allow_obscure_two_letter,
        word_rarity=args.word_rarity,
        use_blacklist=not args.no_blacklist,
        word_preference=args.word_preference
    )

    print(f"Dictionary loaded: {len(generator.dictionary)} words")
    print(f"Generating {args.size}×{args.size} puzzle with {args.min_letters}-{args.max_letters} letters...")
    print(f"Max 2-letter words: {args.max_two_letter}")
    print(f"Using common 2-letter words only: {not args.allow_obscure_two_letter}")
    if args.word_rarity:
        print(f"Word rarity: {args.word_rarity} (min Zipf: {RARITY_THRESHOLDS[args.word_rarity]})")
    print(f"Word preference: {args.word_preference}")
    print(f"Blacklist: {'disabled' if args.no_blacklist else 'enabled'}")
    print()
    
    grid = generator.generate(max_attempts=args.attempts)
    
    if grid:
        print()
        print("=" * 50)
        print("GENERATED PUZZLE:")
        print("=" * 50)
        print()
        grid.print_summary()
    else:
        print("Could not generate a valid puzzle. Try running again or increasing attempts.")