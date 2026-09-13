"""
preprocessor.py — Lightweight Query Preprocessor
==================================================
Sharon's IR engine needs to normalise incoming *queries* before
retrieval.  The documents in the CSV are already preprocessed by Nysa
(stored in the `uses_processed`, `side_effects_processed`, and
`search_text` columns).

This module handles:
  1. Lower-casing
  2. Punctuation removal
  3. Stop-word filtering
  4. Simple suffix-stripping stemmer (no external dependency)

Why a custom stemmer?
  scikit-learn's TF-IDF sees the *already-stemmed* corpus text (Nysa's
  `search_text` column).  If the user types "fevers", we need to stem
  it to "fever" so it matches the index.  The NLTK PorterStemmer would
  be ideal; this fallback works when NLTK is not installed.
"""

import re
import unicodedata

# --------------------------------------------------------------------------- #
# English stop-words (curated for medicine retrieval — very common words that  #
# add no discriminative power for symptom/drug queries)                         #
# --------------------------------------------------------------------------- #
STOP_WORDS: set[str] = {
    "a", "an", "the", "and", "or", "not", "in", "of", "to", "for",
    "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "shall", "should",
    "may", "might", "must", "can", "could", "with", "by", "from",
    "at", "this", "that", "these", "those", "it", "its", "i", "me",
    "my", "we", "our", "you", "your", "he", "she", "his", "her",
    "they", "their", "them", "what", "which", "who", "when", "where",
    "how", "all", "some", "any", "each", "more", "also", "as", "on",
    "use", "used", "treatment", "treat",  # too broad in medicine context
}

# Boolean operator tokens — must NOT be stemmed or stripped
BOOLEAN_KEYWORDS: set[str] = {"AND", "OR", "NOT"}


# --------------------------------------------------------------------------- #
# Minimal suffix stemmer                                                        #
# --------------------------------------------------------------------------- #
_SUFFIX_RULES: list[tuple[str, str]] = [
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
    ("anci", "ance"), ("izer", "ize"), ("ising", "ise"),
    ("izing", "ize"), ("ised", "ise"), ("ized", "ize"),
    ("ational", "ate"), ("ing", ""), ("ings", ""), ("ness", ""),
    ("ment", ""), ("ments", ""), ("ful", ""), ("less", ""),
    ("tion", ""), ("sion", ""), ("ations", ""), ("ers", "r"),
    ("ies", "y"), ("ied", "y"), ("ing", ""), ("ness", ""),
    ("ment", ""), ("ed", ""), ("es", ""), ("s", ""),
]


def _stem(word: str) -> str:
    """Very lightweight suffix-stripping stemmer (fallback)."""
    if len(word) <= 3:
        return word
    for suffix, replacement in _SUFFIX_RULES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: len(word) - len(suffix)] + replacement
    return word


def _try_nltk_stem(word: str) -> str:
    """Use NLTK PorterStemmer if available, else fall back."""
    try:
        from nltk.stem import PorterStemmer  # type: ignore
        _try_nltk_stem._stemmer = getattr(
            _try_nltk_stem, "_stemmer", PorterStemmer()
        )
        return _try_nltk_stem._stemmer.stem(word)
    except ImportError:
        return _stem(word)

# ------------------------------------------------------------------ #
# Stemmer — uses NLTK PorterStemmer (hard dependency)                 #
# ------------------------------------------------------------------ #
try:
    from nltk.stem import PorterStemmer as _PorterStemmer  # type: ignore
    _STEMMER = _PorterStemmer()
    _USE_NLTK = True
except ImportError:  # pragma: no cover
    _USE_NLTK = False
    _STEMMER = None


def normalize_token(token: str) -> str:
    """
    Lower-case, strip accents, then stem a single token.

    Uses NLTK PorterStemmer if installed (recommended).
    Falls back to lightweight suffix-stripping if NLTK is absent.
    """
    token = token.lower().strip()
    if not token:
        return token
    token = unicodedata.normalize("NFKD", token)
    token = "".join(c for c in token if not unicodedata.combining(c))
    if _USE_NLTK:
        return _STEMMER.stem(token)
    return _stem(token)


def preprocess_query(raw_query: str, preserve_boolean: bool = True) -> list[str]:
    """
    Tokenise and normalise a raw user query string.

    Parameters
    ----------
    raw_query : str
        E.g. ``"fever AND headache NOT cough"``
    preserve_boolean : bool
        When True, AND / OR / NOT tokens are kept as-is (upper-cased)
        for the Boolean model to parse.

    Returns
    -------
    list[str]
        E.g. ``["fever", "AND", "headach", "NOT", "cough"]``
    """
    raw_query = raw_query.strip()

    # Normalise boolean operators to upper-case before splitting
    for op in ("AND", "OR", "NOT"):
        pattern = re.compile(rf"\b{op}\b", re.IGNORECASE)
        raw_query = pattern.sub(op, raw_query)

    # Tokenise on whitespace / punctuation (keep hyphens inside words)
    tokens: list[str] = re.findall(r"[A-Za-z0-9][\w\-]*", raw_query)

    processed: list[str] = []
    for token in tokens:
        upper = token.upper()
        if preserve_boolean and upper in BOOLEAN_KEYWORDS:
            processed.append(upper)          # keep as operator
            continue
        lower = token.lower()
        if lower in STOP_WORDS:
            continue
        stemmed = normalize_token(lower)
        if stemmed:
            processed.append(stemmed)

    return processed


def tokens_to_plain_string(tokens: list[str]) -> str:
    """
    Convert a token list (with boolean operators removed) to a space-
    separated string suitable for TF-IDF vectorisation.
    """
    return " ".join(t for t in tokens if t.upper() not in BOOLEAN_KEYWORDS)
