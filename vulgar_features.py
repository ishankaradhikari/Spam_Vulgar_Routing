"""
vulgar_features.py
==================
Feature extraction for the vulgar language classifier.

WHY CHARACTER N-GRAMS DETECT OBFUSCATED WORDS
----------------------------------------------
Word-level features fail on obfuscated text because "f*ck", "f!ck",
and "fick" are three different tokens that don't match "fuck".

Character n-grams solve this:

  "bitch"  → trigrams: ['bit','itc','tch']
  "bi*ch"  normalize→ "bich"  → trigrams: ['bic','ich']
  "b!tch"  normalize→ "bitch" → trigrams: ['bit','itc','tch']  ← identical!

After normalization, obfuscated variants share MOST of their character
n-grams with the clear form. The Naive Bayes classifier learns that
sequences like 'itc','tch','uck','ass','shi' appear overwhelmingly
in vulgar training examples, and classifies new messages accordingly.

FEATURE SET (4 groups)
----------------------
1. Character trigrams  (n=3)  - captures partial word roots
2. Character 4-grams   (n=4)  - captures longer discriminative roots
3. Word TF-IDF tokens  (word-level after normalization+tokenisation)
4. Special character features:
   - special_char_freq : fraction of non-alphanumeric chars in raw text
   - symbol_to_letter  : ratio of [!@#$%*] to total alphabetic chars
   - avg_word_length   : average length of tokens (short = obfuscated)

All features are returned as a flat dict { feature_name: float }.
"""

import re
import math
from vulgar_normalizer import normalize_lower


# ── Stop words (minimal - vulgar words are NOT in this list) ──────────────────
_STOP = {
    # Articles, prepositions, conjunctions
    "a","an","the","to","for","or","and","in","on","at","of","with","from","by","as","into","through",
    
    # Common verbs
    "is","are","am","was","were","be","being","been","have","has","had","do","does","did","will","would",
    "could","should","can","may","might","must","shall","ought","should",
    
    # Pronouns
    "i","me","my","we","our","ours","you","your","yours","he","him","his","she","her","hers","it","its",
    "they","them","their","theirs","what","which","who","whom","whose",
    
    # Common adverbs and question words
    "how","when","where","why","there","then","now","here","very","just","only","so","too","not","no",
    "yes","up","down","out","in","off","over","under","above","below","before","after","during","about",
    
    # Other common words
    "this","that","these","those","such","some","any","all","each","every","both","either","neither",
    "because","since","while","unless","if","than","as","also","again","still","even","but","thus","however",
    "therefore","moreover","however","meanwhile","instead","rather","quite","already","yet","again","more",
    "most","less","least","few","many","much","enough","more","other","another","same","different"
}


def _char_ngrams(text: str, n: int) -> list[str]:
    """
    Extract all character n-grams from text.

    Why this helps with obfuscation:
      "bich" (from bi*ch) shares trigrams ['bic','ich'] with "bitch" ['bit','itc','tch']
      They don't perfectly overlap, but the model still learns the region
      is high-probability vulgar because both forms appear in training data
      with overlapping n-grams.

    Parameters
    ----------
    text : str   Already normalized and lowercased.
    n    : int   N-gram size.
    """
    # Remove spaces for character n-grams (treat words as one stream)
    compact = text.replace(' ', '')
    return [compact[i:i+n] for i in range(len(compact) - n + 1)]


def _word_tokens(text: str) -> list[str]:
    """Tokenize normalized text, removing stop words and short tokens."""
    tokens = re.sub(r'[^a-z0-9\s]', ' ', text).split()
    return [t for t in tokens if t not in _STOP and len(t) > 1]


def _special_char_features(raw_text: str) -> dict[str, float]:
    """
    Compute three numeric features from the RAW (un-normalized) text.

    These catch obfuscation attempts even BEFORE normalization:
    '@ss' has a high special_char_freq and symbol_to_letter ratio.
    """
    if not raw_text:
        return {'special_char_freq': 0.0, 'symbol_to_letter': 0.0, 'avg_word_length': 0.0}

    # Fraction of characters that are non-alphanumeric (excluding spaces)
    non_space = [c for c in raw_text if not c.isspace()]
    special   = [c for c in non_space if not c.isalnum()]
    special_freq = len(special) / max(len(non_space), 1)

    # Ratio of obfuscation symbols [!@#$%*] to alphabetic characters
    obf_symbols = sum(1 for c in raw_text if c in '!@#$%*')
    alpha_chars  = sum(1 for c in raw_text if c.isalpha())
    sym_ratio    = obf_symbols / max(alpha_chars, 1)

    # Average word length (very short words can signal obfuscation fragments)
    words = raw_text.split()
    avg_len = sum(len(w) for w in words) / max(len(words), 1)

    return {
        'special_char_freq': special_freq,
        'symbol_to_letter':  sym_ratio,
        'avg_word_length':   avg_len,
    }


def extract_features(raw_text: str) -> dict[str, float]:
    """
    Full feature extraction pipeline for one text sample.

    Pipeline:
      1. Extract special-char stats from raw text (before normalization)
      2. Normalize (@ → a, ! → i, etc.)
      3. Extract character trigrams from normalized text
      4. Extract character 4-grams from normalized text
      5. Extract word tokens (TF-style: count each word once)

    Returns
    -------
    dict[str, float]
        Feature name → value.
        Binary features (n-gram presence) have value 1.0.
        Numeric features (ratios) are floats in [0, ∞).
    """
    features: dict[str, float] = {}

    # ── Group 4: numeric special-character features (from raw text) ──────────
    features.update(_special_char_features(raw_text))

    # ── Normalize text ────────────────────────────────────────────────────────
    norm = normalize_lower(raw_text)

    # ── Group 1: character trigrams ───────────────────────────────────────────
    for ngram in _char_ngrams(norm, 3):
        key = f'c3_{ngram}'
        features[key] = features.get(key, 0.0) + 1.0

    # ── Group 2: character 4-grams ────────────────────────────────────────────
    for ngram in _char_ngrams(norm, 4):
        key = f'c4_{ngram}'
        features[key] = features.get(key, 0.0) + 1.0

    # ── Group 3: word tokens ──────────────────────────────────────────────────
    for token in _word_tokens(norm):
        key = f'w_{token}'
        features[key] = features.get(key, 0.0) + 1.0

    return features
