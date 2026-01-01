# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Crucigram Puzzle Generator - creates dense crossword-style puzzles where every horizontal and vertical sequence of 2+ letters forms a valid word. Read @RULES.md for the full set of rules.

## Development Setup

```bash
# Create virtual environment (Python 3.12+)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Generator

```bash
# Basic usage
python scheme_generator.py

# With options
python scheme_generator.py --size 7 --min-letters 30 --max-letters 34 --max-two-letter 3 --attempts 500

# Allow obscure 2-letter words (normally filtered to 39 common ones)
python scheme_generator.py --allow-obscure-two-letter
```

## Architecture

Single-file implementation (`scheme_generator.py`) with two main classes:

### CrucigramGrid (lines 58-306)
Represents the puzzle state. Key methods:
- `get_all_words()` - Extracts all horizontal/vertical word sequences
- `is_valid_complete()` - Validates all constraints
- `is_connected()` - BFS to verify letters form one connected component
- `to_string()` / `print_summary()` - ASCII display with Unicode box drawing

### CrucigramGenerator (lines 312-588)
Implements puzzle generation using **constructive expansion with backtracking**:
1. Place a seed word (4-6 letters) in center horizontally
2. Iteratively add crossing words via `_try_expand()` → `_try_add_crossing_word()`
3. Stop when stalled or target letter count reached
4. Finalize by filling empty cells with black squares

Pre-indexes words by length and by letter for O(1) lookup during expansion.

## Puzzle Constraints

A valid crucigram must satisfy:
1. **Letter count**: 30-34 letters in 7×7 grid
2. **Valid words**: Every 2+ letter sequence (horizontal/vertical) must be a dictionary word
3. **Two-letter limit**: Maximum 3 two-letter words
4. **No repeats**: Each word appears exactly once
5. **Connectivity**: All letters form one connected group (4-directional)
6. **No 3×3 blacks**: No 3×3 region of all black squares
