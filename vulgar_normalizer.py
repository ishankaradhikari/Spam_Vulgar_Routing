"""
vulgar_normalizer.py
====================
Part 3 of the enhancement spec: Character-level normalization.

PURPOSE
-------
Converts obfuscated/leet-speak text into a canonical form BEFORE
feature extraction so the ML classifier sees "ass" and "@ss" as
the same character sequence.

This is PREPROCESSING, not rule-based filtering.
No word is blocked here - we just standardise characters.
The classifier then decides based on learned patterns.

SUBSTITUTION TABLE
------------------
Symbol → Latin letter mapping, chosen to match common obfuscation:
  @  →  a      (used in @ss, @ttack)
  $  →  s      (used in $hit, cla$$)
  !  →  i      (used in sh!t, b!tch, f!ck)
  1  →  i      (used in b1tch, sh1t, id1ot)
  3  →  e      (used in s3x, h3ll)
  0  →  o      (used in wh0re, m0ron)
  *  → ''      (used as wildcard: f**k, b*tch → removed so fk, btch remain)
  #  → ''      (used as: #ss → ss, f#ck → fck)

After normalization, "bi*ch" → "bich", "@ss" → "ass", "f!ck" → "fick"
These canonical forms cluster tightly in character-ngram feature space,
allowing Naive Bayes to learn the pattern from training examples.
"""

import re

# ── Substitution table ─────────────────────────────────────────────────────────
# Order matters: apply single-char subs before removing wildcards.
_SUBSTITUTIONS = [
    ('@', 'a'),
    ('$', 's'),
    ('!', 'i'),
    ('1', 'i'),
    ('3', 'e'),
    ('0', 'o'),
    ('*', ''),   # remove wildcard entirely: f**k → fk
    ('#', ''),   # remove hash: f#ck → fck
]


def normalize(text: str) -> str:
    """
    Apply leet-speak / obfuscation normalization to a text string.

    Parameters
    ----------
    text : str
        Raw input text, possibly containing obfuscation symbols.

    Returns
    -------
    str
        Normalized text with symbols replaced by their letter equivalents.
        Case is preserved; call .lower() afterward if needed.

    Examples
    --------
    >>> normalize("@ss")
    'ass'
    >>> normalize("bi*ch")
    'bich'
    >>> normalize("f!ck")
    'fick'           # classifier learns 'fick' ≈ vulgar from training data
    >>> normalize("sh1t")
    'shiit'          # 1→i, so 'shit' with double-i; still clusters with vulgar
    >>> normalize("wh0re")
    'whore'
    >>> normalize("b@$tard")
    'bastard'
    """
    for symbol, replacement in _SUBSTITUTIONS:
        text = text.replace(symbol, replacement)
    return text


def normalize_lower(text: str) -> str:
    """Normalize and lowercase in one step."""
    return normalize(text).lower()
