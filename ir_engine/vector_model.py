"""
vector_model.py — Vector Space Model (TF-IDF + Cosine Similarity)
=================================================================

Implements the Vector Space Model (VSM):
  1. Fit a TF-IDF matrix over the entire corpus (``search_text`` column).
  2. Given a query string, transform it into a TF-IDF query vector.
  3. Compute cosine similarity between the query vector and every
     document vector.
  4. Return a dictionary mapping doc_id → similarity score.

Design decisions
----------------
* ``TfidfVectorizer`` from scikit-learn is used (recommended in the
  project tech stack).
* The vectorizer is fitted on the pre-stemmed ``search_text`` column
  produced by Nysa's preprocessing.  We therefore set
  ``analyzer='word'`` and do NOT apply additional scikit-learn
  tokenisation/stemming to avoid double-stemming.
* ``sublinear_tf=True`` dampens the effect of very high term frequencies
  (replaces raw TF with 1 + log(TF)).
* ``ngram_range=(1, 2)`` allows the model to capture bigrams like
  ``"stomach pain"`` or ``"blood pressure"`` as single features.
* Sparse matrix representation keeps RAM usage manageable for 222 k docs.

Performance note
----------------
Building the TF-IDF matrix for all 222 k rows takes ~30–60 seconds.
The result is stored in ``self._tfidf_matrix`` as a scipy sparse CSR
matrix.  Subsequent searches are very fast (< 100 ms).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .inverted_index import InvertedIndex
from .preprocessor import tokens_to_plain_string, preprocess_query

logger = logging.getLogger(__name__)


class VectorModel:
    """
    TF-IDF Vector Space Model for the medicine corpus.

    Usage
    -----
    ::

        from ir_engine.inverted_index import InvertedIndex
        from ir_engine.vector_model import VectorModel

        idx = InvertedIndex()
        idx.build_from_csv("cleaned_medicines (4).csv")

        vm = VectorModel(idx)
        vm.fit()

        scores = vm.score_all("fever headache")
        # → {1: 0.82, 3: 0.61, 5: 0.55, ...}

        top = vm.search("fever headache", top_k=20)
        # → [(doc_id, score), ...]  sorted descending
    """

    def __init__(self, index: InvertedIndex) -> None:
        if not index.is_built():
            raise RuntimeError(
                "InvertedIndex must be built before VectorModel can be used."
            )
        self._index = index
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix: Optional[csr_matrix] = None
        self._fitted: bool = False

    # ------------------------------------------------------------------ #
    # Fitting                                                              #
    # ------------------------------------------------------------------ #

    def fit(self) -> None:
        """
        Fit the TF-IDF vectorizer on the corpus and build the document
        matrix.

        This is called once during engine initialisation.  Re-calling it
        will re-fit the model (useful if the index is rebuilt).
        """
        logger.info(
            "Fitting TF-IDF model on %d documents …",
            self._index.total_documents,
        )

        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            tokenizer=None,          # corpus already tokenised/stemmed
            preprocessor=None,       # no extra preprocessing
            token_pattern=r"(?u)\b[a-z0-9]\w+\b",
            sublinear_tf=True,       # 1 + log(tf) instead of raw tf
            ngram_range=(1, 2),      # unigrams + bigrams
            min_df=2,                # ignore terms in < 2 docs (noise)
            max_df=0.90,             # ignore terms in > 90% of docs
            max_features=80_000,     # vocabulary cap for memory
        )

        corpus = self._index.corpus_texts      # ordered list of search_text
        self._tfidf_matrix = self._vectorizer.fit_transform(corpus)

        self._fitted = True
        logger.info(
            "TF-IDF matrix shape: %s  |  vocabulary size: %d",
            self._tfidf_matrix.shape,
            len(self._vectorizer.vocabulary_),
        )

    # ------------------------------------------------------------------ #
    # Scoring                                                              #
    # ------------------------------------------------------------------ #

    def _build_query_vector(self, query: str) -> csr_matrix:
        """
        Convert a raw query string to its TF-IDF vector.

        1. Preprocess the query (stem, remove stop-words).
        2. Transform with the fitted vectorizer.
        """
        # Preprocess — strip boolean ops, stem tokens
        tokens = preprocess_query(query, preserve_boolean=False)
        plain = tokens_to_plain_string(tokens)

        if not plain.strip():
            # Fallback: use raw query if preprocessing empties it
            plain = query

        return self._vectorizer.transform([plain])

    def score_all(self, query: str) -> dict[int, float]:
        """
        Compute cosine similarity of the query against ALL documents.

        Parameters
        ----------
        query : str
            Raw user query.

        Returns
        -------
        dict[int, float]
            Mapping of ``doc_id`` → cosine similarity score (0.0 – 1.0).
            Only documents with score > 0 are included.
        """
        if not self._fitted:
            raise RuntimeError("VectorModel.fit() must be called first.")

        q_vec = self._build_query_vector(query)
        if q_vec.nnz == 0:
            logger.warning("Query vector is empty after preprocessing: '%s'", query)
            return {}

        # Shape: (1, num_docs)
        sim_scores: np.ndarray = cosine_similarity(q_vec, self._tfidf_matrix)[0]

        # Build {doc_id: score} for non-zero scores only
        scores: dict[int, float] = {}
        for pos, score in enumerate(sim_scores):
            if score > 0.0:
                doc_id = self._index.pos_to_doc_id[pos]
                scores[doc_id] = float(round(score, 6))

        return scores

    def score_subset(
        self, query: str, doc_ids: set[int]
    ) -> dict[int, float]:
        """
        Compute cosine similarity only for a pre-filtered subset of docs.

        Used after Boolean filtering: instead of scoring all 222 k docs,
        we score only the Boolean-matching subset.

        Parameters
        ----------
        query : str
            Raw user query.
        doc_ids : set[int]
            Subset of doc IDs to score (e.g., from Boolean model).

        Returns
        -------
        dict[int, float]
            Mapping of ``doc_id`` → cosine similarity score.
        """
        if not self._fitted:
            raise RuntimeError("VectorModel.fit() must be called first.")

        if not doc_ids:
            return {}

        q_vec = self._build_query_vector(query)
        if q_vec.nnz == 0:
            # Return uniform scores if query vector is empty
            return {d: 0.0 for d in doc_ids}

        # Gather positional indices for the subset
        positions = [
            self._index.doc_id_to_pos[d]
            for d in doc_ids
            if d in self._index.doc_id_to_pos
        ]
        valid_doc_ids = [
            d for d in doc_ids if d in self._index.doc_id_to_pos
        ]

        # Extract sub-matrix rows
        sub_matrix = self._tfidf_matrix[positions, :]

        # Cosine similarity against the subset
        sim_scores = cosine_similarity(q_vec, sub_matrix)[0]

        scores: dict[int, float] = {}
        for doc_id, score in zip(valid_doc_ids, sim_scores):
            scores[doc_id] = float(round(score, 6))

        return scores

    def search(
        self, query: str, top_k: int = 20
    ) -> list[tuple[int, float]]:
        """
        Full-corpus TF-IDF search.  Returns top-k (doc_id, score) pairs.

        Parameters
        ----------
        query : str
            Raw user query.
        top_k : int
            Number of results to return.

        Returns
        -------
        list[tuple[int, float]]
            Sorted descending by cosine similarity.
        """
        scores = self.score_all(query)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    @property
    def is_fitted(self) -> bool:
        return self._fitted
