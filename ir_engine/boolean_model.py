"""
boolean_model.py — Boolean Information Retrieval Model
=======================================================


Implements strict AND / OR / NOT Boolean retrieval over the inverted
index.  Supports two modes:

  1. **Simple mode** — space-separated tokens are implicitly AND-ed.
       ``"fever headache"``  →  fever AND headache

  2. **Explicit mode** — user writes boolean operators explicitly.
       ``"fever AND headache NOT cough"``
       ``"(fever OR headache) AND NOT cough"``
       ``"category:RESPIRATORY AND fever"``

Special filter syntax
---------------------
  ``category:<value>``  — filters by therapeutic_class
  ``class:<value>``     — filters by chemical_class (partial match)

Query grammar (simplified BNF)
-------------------------------
    query  := expr
    expr   := term (('AND' | 'OR') term)*
    term   := ['NOT'] atom
    atom   := TOKEN | '(' expr ')'
    TOKEN  := word | 'category:' word | 'class:' word

Implementation
--------------
Uses a simple recursive-descent parser on the token stream.  Each node
evaluates to a set of document IDs using the inverted index.

Complexity: O(k × N) worst-case (k = number of terms, N = docs per
posting list).  In practice the index posting lists are much smaller
than N and set operations are fast.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from .inverted_index import InvertedIndex
from .preprocessor import normalize_token

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Tokeniser for boolean queries                                                 #
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(
    r"""
    (?P<LPAREN>  \(                        ) |
    (?P<RPAREN>  \)                        ) |
    (?P<AND>     \bAND\b                   ) |
    (?P<OR>      \bOR\b                    ) |
    (?P<NOT>     \bNOT\b                   ) |
    (?P<CAT>     category:[A-Za-z0-9_\s]+  ) |
    (?P<WORD>    [A-Za-z0-9][\w\-]*        )
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _tokenise(query: str) -> list[tuple[str, str]]:
    """
    Convert a raw query string into a list of ``(type, value)`` tokens.

    Examples
    --------
    >>> _tokenise("fever AND (headache OR migraine) NOT cough")
    [('WORD','fever'),('AND','AND'),('LPAREN','('),('WORD','headache'),
     ('OR','OR'),('WORD','migraine'),('RPAREN',')'),('NOT','NOT'),('WORD','cough')]
    """
    # Normalise boolean operators to uppercase
    for op in ("AND", "OR", "NOT"):
        query = re.sub(rf"\b{op}\b", op, query, flags=re.IGNORECASE)

    tokens: list[tuple[str, str]] = []
    for m in _TOKEN_RE.finditer(query):
        kind = m.lastgroup
        value = m.group().strip()
        if kind == "AND":
            tokens.append(("AND", "AND"))
        elif kind == "OR":
            tokens.append(("OR", "OR"))
        elif kind == "NOT":
            tokens.append(("NOT", "NOT"))
        elif kind == "LPAREN":
            tokens.append(("LPAREN", "("))
        elif kind == "RPAREN":
            tokens.append(("RPAREN", ")"))
        elif kind == "CAT":
            tokens.append(("CAT", value))
        elif kind == "WORD":
            tokens.append(("WORD", value))
    return tokens


# --------------------------------------------------------------------------- #
# Recursive-descent parser                                                      #
# --------------------------------------------------------------------------- #

class _Parser:
    """
    Parses a boolean query token stream and evaluates it against the
    inverted index, producing a ``set[int]`` of matching document IDs.
    """

    def __init__(self, tokens: list[tuple[str, str]], index: InvertedIndex) -> None:
        self._tokens = tokens
        self._pos = 0
        self._index = index

    # ---- Helper ---------------------------------------------------------- #

    def _peek(self) -> Optional[tuple[str, str]]:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _consume(self) -> tuple[str, str]:
        token = self._tokens[self._pos]
        self._pos += 1
        return token

    # ---- Grammar --------------------------------------------------------- #

    def parse_expr(self) -> set[int]:
        """expr := term (('AND' | 'OR') term)*"""
        result = self.parse_term()

        while True:
            peek = self._peek()
            if peek is None:
                break
            if peek[0] == "AND":
                self._consume()
                right = self.parse_term()
                result = result & right           # intersection
            elif peek[0] == "OR":
                self._consume()
                right = self.parse_term()
                result = result | right           # union
            else:
                break

        return result

    def parse_term(self) -> set[int]:
        """term := ['NOT'] atom"""
        peek = self._peek()
        if peek and peek[0] == "NOT":
            self._consume()
            atom_result = self.parse_atom()
            return self._index.get_all_doc_ids() - atom_result   # complement
        return self.parse_atom()

    def parse_atom(self) -> set[int]:
        """atom := '(' expr ')' | CAT_token | WORD_token"""
        peek = self._peek()
        if peek is None:
            return set()

        if peek[0] == "LPAREN":
            self._consume()               # consume '('
            result = self.parse_expr()
            if self._peek() and self._peek()[0] == "RPAREN":
                self._consume()           # consume ')'
            return result

        elif peek[0] == "CAT":
            self._consume()
            # "category:RESPIRATORY" → extract value after ':'
            cat_value = peek[1].split(":", 1)[1].strip()
            return self._index.get_docs_for_category(cat_value)

        elif peek[0] == "WORD":
            self._consume()
            stemmed = normalize_token(peek[1].lower())
            return self._index.get_docs_for_term(stemmed)

        else:
            # Unexpected token — skip it
            self._consume()
            return set()


# --------------------------------------------------------------------------- #
# Public API                                                                    #
# --------------------------------------------------------------------------- #

class BooleanModel:
    """
    Boolean IR retrieval component.

    Usage
    -----
    ::

        from ir_engine.inverted_index import InvertedIndex
        from ir_engine.boolean_model import BooleanModel

        idx = InvertedIndex()
        idx.build_from_csv("cleaned_medicines (4).csv")

        bm = BooleanModel(idx)

        # Simple implicit AND
        result_ids = bm.search("fever headache")

        # Explicit boolean
        result_ids = bm.search("fever AND headache NOT cough")

        # Category filter
        result_ids = bm.search("fever AND category:RESPIRATORY")

        # Complex expression
        result_ids = bm.search("(fever OR headache) AND NOT cough")
    """

    def __init__(self, index: InvertedIndex) -> None:
        if not index.is_built():
            raise RuntimeError(
                "InvertedIndex must be built before BooleanModel can be used."
            )
        self._index = index

    def search(self, query: str) -> set[int]:
        """
        Execute a Boolean query and return matching document IDs.

        Implicit AND is applied when no boolean operators are present:
        ``"fever headache"`` → ``fever AND headache``

        Parameters
        ----------
        query : str
            Raw query string (may contain AND / OR / NOT, parentheses,
            ``category:`` filters).

        Returns
        -------
        set[int]
            Set of document IDs satisfying the Boolean expression.
        """
        if not query.strip():
            return set()

        tokens = _tokenise(query.strip())

        if not tokens:
            return set()

        # Implicit AND: inject AND between consecutive WORD tokens that have
        # no explicit operator between them.
        tokens = self._inject_implicit_and(tokens)

        parser = _Parser(tokens, self._index)
        result = parser.parse_expr()

        logger.debug(
            "BooleanSearch('%s') → %d matching docs", query, len(result)
        )
        return result

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _inject_implicit_and(
        tokens: list[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        """
        Insert implicit AND between adjacent non-operator tokens.

        ``[WORD, WORD, WORD]``  →  ``[WORD, AND, WORD, AND, WORD]``
        ``[WORD, AND, WORD]``   → unchanged
        ``[WORD, RPAREN]``      → unchanged (no AND before closing paren)
        """
        OPERANDS = {"WORD", "CAT", "RPAREN"}
        OPERATORS = {"AND", "OR", "NOT", "LPAREN"}

        result: list[tuple[str, str]] = []
        for i, token in enumerate(tokens):
            result.append(token)
            if i + 1 < len(tokens):
                curr_type = token[0]
                next_type = tokens[i + 1][0]
                # Insert AND if current is an operand and next is a WORD/CAT
                # (but NOT right before NOT — that's a unary operator)
                if (
                    curr_type in OPERANDS
                    and next_type in {"WORD", "CAT", "LPAREN"}
                    and next_type != "NOT"
                ):
                    result.append(("AND", "AND"))
                elif curr_type in OPERANDS and next_type == "NOT":
                    result.append(("AND", "AND"))
        return result

    def explain(self, query: str) -> dict:
        """
        Return a human-readable explanation of how the query is parsed.
        Useful for debugging and viva demonstrations.

        Returns
        -------
        dict with keys:
          - ``raw_query`` (str)
          - ``tokens`` (list)
          - ``result_count`` (int)
          - ``sample_ids`` (list[int], up to 10)
        """
        tokens = _tokenise(query.strip())
        tokens_with_and = self._inject_implicit_and(tokens)
        result = self.search(query)
        return {
            "raw_query": query,
            "parsed_tokens": tokens_with_and,
            "result_count": len(result),
            "sample_ids": sorted(result)[:10],
        }
