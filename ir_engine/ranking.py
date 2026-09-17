"""
ranking.py — Manual Ranking Module
====================================
 Deliverable 3.

After Boolean filtering and TF-IDF scoring, this module applies layered
heuristic rules to produce the final ranked list presented to the user.

Ranking Formula
---------------
    Final Score = w1 × TF-IDF Score
                + w2 × Exact Symptom Match Bonus
                + w3 × Multi-Symptom Match Bonus
                + w4 × Name Match Bonus
                + w5 × Category Match Bonus

Default weights (tunable):
    w1 = 0.60   # primary relevance signal
    w2 = 0.20   # reward exact term hit in uses_processed
    w3 = 0.10   # reward covering multiple query terms
    w4 = 0.06   # drug name contains query term
    w5 = 0.04   # therapeutic category bonus (if user specifies)

Why these weights?
------------------
The TF-IDF cosine similarity is the dominant signal (60%).
Exact symptom match is awarded a strong 20% bonus because a medicine
whose *primary use* exactly matches the query symptom is more relevant
than one where the symptom appears only in side-effects.
The remaining 20% covers coverage and metadata signals.

Ranking Layers (from highest to lowest priority):
-------------------------------------------------
1. Exact symptom match: query term ∈ uses_processed
2. Multi-symptom coverage: more query terms matched → higher bonus
3. TF-IDF cosine similarity (from VectorModel)
4. Medicine name match: query term ∈ clean_name
5. Therapeutic category alignment

Score Normalisation
-------------------
Final scores are normalised to [0, 1] before being returned so that
the caller always gets a consistent confidence metric.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from .inverted_index import InvertedIndex, MedicineDoc
from .preprocessor import normalize_token, preprocess_query, BOOLEAN_KEYWORDS

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Ranking weights —                                 #
# --------------------------------------------------------------------------- #
@dataclass
class RankingWeights:
    tfidf_weight: float = 0.60
    exact_symptom_weight: float = 0.20
    multi_symptom_weight: float = 0.10
    name_match_weight: float = 0.06
    category_match_weight: float = 0.04

    def validate(self) -> None:
        total = (
            self.tfidf_weight
            + self.exact_symptom_weight
            + self.multi_symptom_weight
            + self.name_match_weight
            + self.category_match_weight
        )
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"RankingWeights must sum to 1.0 (got {total:.4f})"
            )


DEFAULT_WEIGHTS = RankingWeights()


# --------------------------------------------------------------------------- #
# Result dataclass                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class RankedResult:
    """A single ranked search result with breakdown for transparency."""
    doc_id: int
    clean_name: str
    uses: str
    side_effects: str
    dosage_info: str
    therapeutic_class: str
    action_class: str
    substitutes: str
    # Scores
    final_score: float
    tfidf_score: float
    exact_symptom_bonus: float
    multi_symptom_bonus: float
    name_match_bonus: float
    category_bonus: float

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "medicine_name": self.clean_name,
            "uses": self.uses,
            "side_effects": self.side_effects,
            "dosage_info": self.dosage_info,
            "therapeutic_class": self.therapeutic_class,
            "action_class": self.action_class,
            "substitutes": self.substitutes,
            "score": round(self.final_score, 4),
            "score_breakdown": {
                "tfidf": round(self.tfidf_score, 4),
                "exact_symptom_bonus": round(self.exact_symptom_bonus, 4),
                "multi_symptom_bonus": round(self.multi_symptom_bonus, 4),
                "name_match_bonus": round(self.name_match_bonus, 4),
                "category_bonus": round(self.category_bonus, 4),
            },
        }


# --------------------------------------------------------------------------- #
# Ranker                                                                        #
# --------------------------------------------------------------------------- #
class Ranker:
    """
    Applies manual ranking rules on top of TF-IDF scores.

    Usage
    -----
    ::

        ranker = Ranker(index)
        ranked = ranker.rank(
            query="fever headache",
            tfidf_scores={1: 0.82, 3: 0.61, 5: 0.44},
            top_k=10
        )
        for r in ranked:
            print(r.medicine_name, r.final_score)
    """

    def __init__(
        self,
        index: InvertedIndex,
        weights: Optional[RankingWeights] = None,
    ) -> None:
        self._index = index
        self._weights = weights or DEFAULT_WEIGHTS
        self._weights.validate()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def rank(
        self,
        query: str,
        tfidf_scores: dict[int, float],
        top_k: int = 10,
        target_category: Optional[str] = None,
    ) -> list[RankedResult]:
        """
        Combine TF-IDF scores with manual ranking rules.

        Parameters
        ----------
        query : str
            Raw user query string.
        tfidf_scores : dict[int, float]
            Mapping of doc_id → cosine similarity from VectorModel.
        top_k : int
            Number of final results to return.
        target_category : str | None
            Optional therapeutic category string for category bonus.

        Returns
        -------
        list[RankedResult]
            Sorted by final_score descending, length ≤ top_k.
        """
        if not tfidf_scores:
            return []

        # Extract plain query terms (no boolean operators)
        query_tokens = [
            t for t in preprocess_query(query, preserve_boolean=False)
            if t.upper() not in BOOLEAN_KEYWORDS
        ]

        # Normalise target category
        norm_category = target_category.lower().strip() if target_category else ""

        w = self._weights
        raw_results: list[tuple[float, RankedResult]] = []

        for doc_id, tfidf_score in tfidf_scores.items():
            doc = self._index.docs.get(doc_id)
            if doc is None:
                continue

            # --- Layer 1: Exact symptom match bonus ----------------------
            exact_bonus = self._exact_symptom_bonus(doc, query_tokens)

            # --- Layer 2: Multi-symptom coverage bonus --------------------
            multi_bonus = self._multi_symptom_bonus(doc, query_tokens)

            # --- Layer 3: Medicine name match bonus -----------------------
            name_bonus = self._name_match_bonus(doc, query_tokens)

            # --- Layer 4: Category alignment bonus -----------------------
            cat_bonus = self._category_bonus(doc, norm_category)

            # --- Weighted sum ---------------------------------------------
            final_score = (
                w.tfidf_weight          * tfidf_score
                + w.exact_symptom_weight  * exact_bonus
                + w.multi_symptom_weight  * multi_bonus
                + w.name_match_weight     * name_bonus
                + w.category_match_weight * cat_bonus
            )

            result = RankedResult(
                doc_id=doc_id,
                clean_name=doc.clean_name,
                uses=doc.uses_str,
                side_effects=doc.side_effects_str,
                dosage_info=doc.dosage_info,
                therapeutic_class=doc.therapeutic_class,
                action_class=doc.action_class,
                substitutes=doc.substitutes_str,
                final_score=final_score,
                tfidf_score=tfidf_score,
                exact_symptom_bonus=exact_bonus,
                multi_symptom_bonus=multi_bonus,
                name_match_bonus=name_bonus,
                category_bonus=cat_bonus,
            )
            raw_results.append((final_score, result))

        # Sort descending
        raw_results.sort(key=lambda x: x[0], reverse=True)

        # Normalise final scores to [0, 1]
        top_results = [r for _, r in raw_results[:top_k]]
        self._normalise_scores(top_results)

        return top_results

    # ------------------------------------------------------------------ #
    # Individual bonus calculators                                         #
    # ------------------------------------------------------------------ #

    def _exact_symptom_bonus(
        self, doc: MedicineDoc, query_tokens: list[str]
    ) -> float:
        """
        Returns 1.0 if ANY query token exactly matches a stemmed token
        in uses_processed.  Returns 0.0 otherwise.

        This gives preference to medicines whose *primary purpose*
        directly addresses the user's symptom.
        """
        uses_tokens: set[str] = set(
            doc.uses_processed.lower().split()
        )
        for qt in query_tokens:
            if qt in uses_tokens:
                return 1.0
        return 0.0

    def _multi_symptom_bonus(
        self, doc: MedicineDoc, query_tokens: list[str]
    ) -> float:
        """
        Returns a proportional score based on how many query tokens are
        found in the document's search_text.

        0 matches  → 0.0
        1/n matches → 1/n
        n/n matches → 1.0

        Example: query = "fever headache cough", doc matches 2 of 3
        → bonus = 2/3 ≈ 0.667
        """
        if not query_tokens:
            return 0.0

        search_tokens: set[str] = set(
            doc.search_text.lower().split()
        )
        matched = sum(1 for qt in query_tokens if qt in search_tokens)
        return matched / len(query_tokens)

    def _name_match_bonus(
        self, doc: MedicineDoc, query_tokens: list[str]
    ) -> float:
        """
        Returns 1.0 if any query token appears inside the medicine's
        clean_name (drug name match).  This helps when the user types
        part of a medicine name directly.
        """
        name_lower = doc.clean_name.lower()
        for qt in query_tokens:
            if qt in name_lower:
                return 1.0
        return 0.0

    def _category_bonus(
        self, doc: MedicineDoc, target_category: str
    ) -> float:
        """
        Returns 1.0 if the medicine's therapeutic_class contains the
        target category string (prefix match).

        Example: target_category = "resp"
        matches doc.therapeutic_class = "RESPIRATORY"
        """
        if not target_category:
            return 0.0
        doc_cat = doc.therapeutic_class.lower()
        return 1.0 if target_category in doc_cat else 0.0

    @staticmethod
    def _normalise_scores(results: list[RankedResult]) -> None:
        """
        Min-max normalise the final_score field to [0, 1] in-place.
        If all scores are equal, set them all to 1.0.
        """
        if not results:
            return
        max_score = max(r.final_score for r in results)
        min_score = min(r.final_score for r in results)
        span = max_score - min_score
        for r in results:
            if span == 0:
                r.final_score = 1.0
            else:
                r.final_score = (r.final_score - min_score) / span
