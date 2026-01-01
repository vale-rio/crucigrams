# Crucigram Puzzle Rules

## Overview

Crucigram is a crossword construction puzzle where players are given:
- A set of letter tiles (exact count required to fill the grid)
- A set of crossword-style clues (labeled Across/Down)
- An empty grid of known dimensions

The player must figure out:
1. Where the black squares go
2. How to arrange all the letter tiles to form valid intersecting words that answer the clues

---

## Grid Rules

| Rule | Description |
|------|-------------|
| **Grid size** | 7×7 (49 cells total) |
| **Letter density** | 30-34 letters (~65% of grid) |
| **Black squares** | 15-19 squares (~35% of grid) |
| **No 3×3 black blocks** | No region of 9 contiguous black squares allowed (2×2 is acceptable) |

---

## Word Rules

| Rule | Description |
|------|-------------|
| **Valid vocabulary** | All words must be in the SOWPODS Scrabble dictionary |
| **No proper nouns** | No names, cities, acronyms, or abbreviations |
| **Minimum length** | 2 letters (but limited — see below) |
| **Two-letter word limit** | Maximum 3 two-letter words per puzzle |
| **No repeated words** | Each word may appear only once in the grid |
| **Both directions required** | At least one Across word AND at least one Down word |

---

## Density Rule (Dense Grid Style)

Every horizontal and vertical sequence of 2+ adjacent letters must form a valid word.

**Example — Valid:**
```
C A T
A . .
R . .
```
- CAT (across) ✓
- CAR (down) ✓

**Example — Invalid:**
```
C A T
A R E
R . .
```
- CAT (across) ✓
- ARE (across) ✓
- CAR (down) ✓
- AR (down) — must be valid ✓
- TE (down) — NOT a valid word ✗

This means placing letters adjacent to existing words creates implicit words that must also be valid.

---

## Connectivity Rule

All letters must form **one connected group** (no isolated clusters).

**Valid:** All letters reachable by moving up/down/left/right through other letters.

**Invalid:**
```
C A T . . . .
A . . . . . .
R . . D O G .
. . . . . . .
```
CAT/CAR is disconnected from DOG — this is not allowed.

---

## Output Specification

### Solved Grid
Using box-drawing characters with padding:
```
┌───┬───┬───┬───┬───┬───┬───┐
│ C │ A │ T │ ■ │ D │ O │ G │
├───┼───┼───┼───┼───┼───┼───┤
│ O │ R │ E │ ■ │ ■ │ N │ E │
├───┼───┼───┼───┼───┼───┼───┤
│ W │ ■ │ A │ R │ T │ ■ │ W │
├───┼───┼───┼───┼───┼───┼───┤
...
└───┴───┴───┴───┴───┴───┴───┘
```

Characters used:
- Corners: `┌ ┐ └ ┘`
- Edges: `─ │`
- Intersections: `┬ ┴ ├ ┤ ┼`
- Black square: `■`
- Letters: Capital letters with single space padding on each side

### Word List
```
Words:
  ACROSS: CAT, ORE, ART, ...
  DOWN:   COW, ARE, TEN, ...
```

### Letter Inventory
```
Letters:
  A×4, C×2, D×1, E×3, G×1, ...
  Total: 32 letters
```

---

## Summary of Constraints

1. ✓ Words must be valid English (SOWPODS dictionary)
2. ✓ Every adjacent letter sequence (2+) must be a valid word
3. ✓ All letters must be connected (no isolated clusters)
4. ✓ 30-34 letters in a 7×7 grid
5. ✓ Maximum 3 two-letter words
6. ✓ No repeated words
7. ✓ No 3×3 black square blocks
8. ✓ At least one Across and one Down word
9. ✓ Player must use ALL provided letters to win