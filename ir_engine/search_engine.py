"""
search_engine.py — Unified Search Engine (Sharon's Deliverable 4)
=================================================================
This is the single public interface that Sam's Django API calls.

Pipeline
--------
    User Query (raw string)
          │
          ▼
    [1] Boolean Filtering  (BooleanModel)
          │  → set of doc_ids matching strict AND/OR/NOT conditions
          ▼
    [2] TF-IDF Scoring     (VectorModel)
          │  → dict of doc_id → cosine_similarity score
          │  (scored on Boolean-filtered subset for efficiency)
          ▼
    [3] Manual Ranking     (Ranker)
          │  → weighted final score per result
          ▼
    [4] Return top-k results as list[dict]

If Boolean filtering returns 0 results (too strict), the engine
automatically falls back to full-corpus TF-IDF search and returns
the top-k most similar documents.

Initialisation
--------------
The engine loads lazily — the CSV is parsed and the TF-IDF matrix is
fitted only when ``engine.load()`` is explicitly called.  This avoids
slow startup in unit tests.

Sam's Django integration
------------------------
::

    # settings/views.py
    from ir_engine import MedicineSearchEngine

    engine = MedicineSearchEngine("path/to/cleaned_medicines (4).csv")
    engine.load()            # call once at server startup

    # In the search view:
    results = engine.search("fever headache", top_k=10)
    return JsonResponse({"results": results})
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from .boolean_model import BooleanModel
from .inverted_index import InvertedIndex
from .ranking import RankedResult, Ranker, RankingWeights
from .vector_model import VectorModel

logger = logging.getLogger(__name__)


class MedicineSearchEngine:
    """
    Top-level search engine combining Boolean, VSM, and ranking.

    Parameters
    ----------
    csv_path : str | Path
        Path to ``cleaned_medicines (4).csv``.
    max_docs : int | None
        If given, only load the first *max_docs* rows.  Useful for
        development and testing.
    weights : RankingWeights | None
        Custom ranking weights.  Defaults to Sharon's defaults.
    """

    def __init__(
        self,
        csv_path: str | Path,
        max_docs: Optional[int] = None,
        weights: Optional[RankingWeights] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.max_docs = max_docs
        self._weights = weights

        # Core components (initialised in load())
        self._index: Optional[InvertedIndex] = None
        self._boolean_model: Optional[BooleanModel] = None
        self._vector_model: Optional[VectorModel] = None
        self._ranker: Optional[Ranker] = None
        self._loaded: bool = False

    # ------------------------------------------------------------------ #
    # Initialisation                                                       #
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """
        Parse CSV, build inverted index, fit TF-IDF model.
        Call this ONCE at application startup.
        """
        if self._loaded:
            logger.warning("Engine already loaded — skipping re-load.")
            return

        t0 = time.perf_counter()
        logger.info("=== MedicineSearchEngine loading ===")

        # Step 1 — Inverted Index
        self._index = InvertedIndex()
        self._index.build_from_csv(self.csv_path, max_docs=self.max_docs)

        # Step 2 — Boolean Model
        self._boolean_model = BooleanModel(self._index)

        # Step 3 — Vector Model (TF-IDF)
        self._vector_model = VectorModel(self._index)
        self._vector_model.fit()

        # Step 4 — Ranker
        self._ranker = Ranker(self._index, self._weights)

        self._loaded = True
        elapsed = time.perf_counter() - t0
        logger.info("Engine ready in %.1f seconds.", elapsed)

    # ------------------------------------------------------------------ #
    # Public search API                                                    #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        top_k: int = 10,
        boolean_filter: Optional[str] = None,
        target_category: Optional[str] = None,
    ) -> list[dict]:
        """
        Execute a search query and return ranked medicine results.

        Parameters
        ----------
        query : str
            The user's raw symptom / medicine query.
            Examples:
              - ``"fever headache"``
              - ``"fever AND headache NOT cough"``
              - ``"(fever OR cold) AND category:RESPIRATORY"``
        top_k : int
            Number of results to return (default 10).
        boolean_filter : str | None
            Separate explicit Boolean filter expression.  When provided
            this overrides implicit Boolean parsing of ``query``.
            Use this to support advanced search forms where the user
            builds a structured filter separately.
        target_category : str | None
            Therapeutic category string for ranking bonus.
            E.g. ``"RESPIRATORY"``.

        Returns
        -------
        list[dict]
            Each dict has:
              - ``doc_id`` (int)
              - ``medicine_name`` (str)
              - ``uses`` (str)
              - ``side_effects`` (str)
              - ``dosage_info`` (str)
              - ``therapeutic_class`` (str)
              - ``action_class`` (str)
              - ``substitutes`` (str)
              - ``score`` (float, 0–1)
              - ``score_breakdown`` (dict)
        """
        self._ensure_loaded()

        if not query.strip():
            return []

        # ---- Step 1: Boolean Filtering ---------------------------------- #
        bool_query = boolean_filter if boolean_filter else query
        bool_doc_ids: set[int] = self._boolean_model.search(bool_query)

        logger.debug(
            "Boolean filter '%s' → %d docs", bool_query, len(bool_doc_ids)
        )

        # ---- Step 2: TF-IDF Scoring ------------------------------------- #
        if bool_doc_ids:
            # Score only the Boolean-filtered subset (efficient)
            tfidf_scores = self._vector_model.score_subset(query, bool_doc_ids)
        else:
            # Fallback: Boolean was too strict → full corpus TF-IDF
            logger.info(
                "Boolean filter returned 0 results for '%s'. "
                "Falling back to full TF-IDF search.",
                query,
            )
            tfidf_scores = self._vector_model.score_all(query)

        if not tfidf_scores:
            logger.info("No TF-IDF results for query: '%s'", query)
            return []

        # Limit to top 3×top_k before ranking (performance optimisation)
        # to avoid ranking all 222k documents
        if len(tfidf_scores) > top_k * 3:
            sorted_tfidf = sorted(
                tfidf_scores.items(), key=lambda x: x[1], reverse=True
            )
            tfidf_scores = dict(sorted_tfidf[: top_k * 3])

        # ---- Step 3: Manual Ranking ------------------------------------- #
        ranked_results: list[RankedResult] = self._ranker.rank(
            query=query,
            tfidf_scores=tfidf_scores,
            top_k=top_k,
            target_category=target_category,
        )

        logger.debug(
            "Search('%s') → %d results (top score: %.4f)",
            query,
            len(ranked_results),
            ranked_results[0].final_score if ranked_results else 0,
        )

        return [r.to_dict() for r in ranked_results]

    def boolean_only(self, query: str) -> list[dict]:
        """
        Return documents satisfying only Boolean conditions (unranked).
        Useful for exact filtering queries.

        Parameters
        ----------
        query : str
            Boolean query string.

        Returns
        -------
        list[dict]
            List of medicine dicts (no scores).
        """
        self._ensure_loaded()
        doc_ids = self._boolean_model.search(query)
        results = []
        for doc_id in sorted(doc_ids)[:200]:   # cap at 200
            doc = self._index.docs.get(doc_id)
            if doc:
                results.append(doc.to_dict())
        return results

    def explain_search(self, query: str, top_k: int = 5) -> dict:
        """
        Return a detailed explanation of the search pipeline for a query.
        Useful for viva demonstrations and debugging.

        Returns
        -------
        dict with keys:
          - ``query`` (str)
          - ``boolean_result_count`` (int)
          - ``boolean_sample_ids`` (list[int])
          - ``tfidf_result_count`` (int)
          - ``top_results`` (list[dict])
        """
        self._ensure_loaded()
        bool_explanation = self._boolean_model.explain(query)
        ranked = self.search(query, top_k=top_k)

        return {
            "query": query,
            "boolean_result_count": bool_explanation["result_count"],
            "boolean_sample_ids": bool_explanation["sample_ids"],
            "boolean_parsed_tokens": bool_explanation["parsed_tokens"],
            "tfidf_result_count": len(
                self._vector_model.score_subset(
                    query,
                    set(bool_explanation["sample_ids"]),
                )
            ),
            "top_results": ranked,
        }

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def total_documents(self) -> int:
        self._ensure_loaded()
        return self._index.total_documents

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(
                "Engine not loaded. Call MedicineSearchEngine.load() first."
            )
