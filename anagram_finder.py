#!/usr/bin/env python3
"""
Anagram Finder - Find all anagrams of a given word.

Uses the english-words library for a comprehensive dictionary.
Primarily intended for generating crossword puzzle clues.
"""

import argparse
from collections import defaultdict

from english_words import get_english_words_set


def load_dictionary() -> set[str]:
    """Load comprehensive English dictionary from english-words package."""
    words = get_english_words_set(['web2', 'gcide'], alpha=True)
    # Normalize to lowercase to avoid duplicates (Act/act)
    return {w.lower() for w in words}


def build_anagram_index(words: set[str]) -> dict[str, list[str]]:
    """Build index mapping sorted letter signatures to words."""
    index: dict[str, list[str]] = defaultdict(list)
    for word in words:
        signature = ''.join(sorted(word.lower()))
        index[signature].append(word)
    return index


def find_anagrams(word: str, index: dict[str, list[str]], min_length: int = 0) -> list[str]:
    """Find all anagrams of the given word."""
    signature = ''.join(sorted(word.lower()))
    matches = index.get(signature, [])
    # Exclude the input word itself, apply minimum length filter
    return sorted([
        w for w in matches
        if w.lower() != word.lower() and len(w) >= min_length
    ])


def main():
    parser = argparse.ArgumentParser(
        description='Find all anagrams of a given word.'
    )
    parser.add_argument('word', help='The word to find anagrams for')
    parser.add_argument(
        '--min-length', '-ml',
        type=int,
        default=0,
        help='Minimum length of anagrams to return'
    )
    args = parser.parse_args()

    # Load dictionary and build index
    print('Loading dictionary...')
    words = load_dictionary()
    print(f'Loaded {len(words):,} words')

    print('Building anagram index...')
    index = build_anagram_index(words)

    # Find anagrams
    anagrams = find_anagrams(args.word, index, args.min_length)

    # Display results
    word_upper = args.word.upper()
    if anagrams:
        print(f'\nAnagrams of {word_upper} ({len(anagrams)} found):')
        print('  ' + ', '.join(anagrams))
    else:
        print(f'\nNo anagrams found for {word_upper}')


if __name__ == '__main__':
    main()
