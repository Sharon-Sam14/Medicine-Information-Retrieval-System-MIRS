"""
inverted_index.py — Inverted Index Builder
==========================================
 core data structure.

An inverted index maps every unique *term* (stemmed token) that appears
in the corpus to the set of document IDs that contain it:

    fever     → {1, 3, 5, 8, ...}
    headach   → {1, 4, 8, ...}
    cough     → {2, 5, 9, ...}

This allows O(1) term lookup and very fast set-intersection / union for
Boolean retrieval, instead of scanning every document for every query.

Architecture notes
------------------
* The index is built from the ``search_text`` column (Nysa's output).
* Additional partial indexes are kept for ``therapeutic_class`` and
  ``chemical_class`` so that Boolean filters like
  ``"fever AND category:RESPIRATORY"`` can be supported.
* The entire index is kept in RAM (Python dicts + sets).  For 222 k
  documents the memory footprint is manageable (~150–250 MB depending
  on vocabulary size).
"""

from __future__ import annotations

import csv
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .preprocessor import normalize_token

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data model for a single medicine document                                     #
# --------------------------------------------------------------------------- #
@dataclass
class MedicineDoc:
    """Lightweight representation of one row in the CSV."""
    doc_id: int
    clean_name: str
    dosage_info: str
    uses_str: str
    side_effects_str: str
    substitutes_str: str
    chemical_class: str
    habit_forming: str
    therapeutic_class: str
    action_class: str
    uses_processed: str
    side_effects_processed: str
    search_text: str

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "clean_name": self.clean_name,
            "dosage_info": self.dosage_info,
            "uses": self.uses_str,
            "side_effects": self.side_effects_str,
            "substitutes": self.substitutes_str,
            "chemical_class": self.chemical_class,
            "habit_forming": self.habit_forming,
            "therapeutic_class": self.therapeutic_class,
            "action_class": self.action_class,
        }


# --------------------------------------------------------------------------- #
# Inverted Index                                                                #
# --------------------------------------------------------------------------- #
class InvertedIndex:
    """
    Inverted index over the medicine corpus.

    Attributes
    ----------
    term_index : dict[str, set[int]]
        Maps stemmed term → set of doc_ids containing that term.
    category_index : dict[str, set[int]]
        Maps lowercase therapeutic_class → set of doc_ids.
    docs : dict[int, MedicineDoc]
        Maps doc_id → full document object (for result rendering).
    corpus_texts : list[str]
        Ordered list of ``search_text`` strings; index position equals
        the *positional* id used by the TF-IDF matrix.
    doc_id_to_pos : dict[int, int]
        Maps csv doc_id → positional index in corpus_texts.
    pos_to_doc_id : dict[int, int]
        Reverse mapping.
    """

    def __init__(self) -> None:
        self.term_index: dict[str, set[int]] = defaultdict(set)
        self.category_index: dict[str, set[int]] = defaultdict(set)
        self.docs: dict[int, MedicineDoc] = {}
        self.corpus_texts: list[str] = []
        self.doc_id_to_pos: dict[int, int] = {}
        self.pos_to_doc_id: dict[int, int] = {}
        self._built: bool = False

    # ------------------------------------------------------------------ #
    # Building                                                             #
    # ------------------------------------------------------------------ #
    def build_from_csv(
        self,
        csv_path: str | Path,
        max_docs: Optional[int] = None,
    ) -> None:
        """
        Parse the medicine CSV and populate all index structures.

        Parameters
        ----------
        csv_path : str | Path
            Absolute path to ``cleaned_medicines (4).csv``.
        max_docs : int | None
            If given, only index the first *max_docs* rows (useful for
            development / testing without loading all 222 k entries).
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        logger.info("Building inverted index from %s …", csv_path.name)
        pos = 0

        with open(csv_path, "r", encoding="utf-8", errors="ignore") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if max_docs is not None and pos >= max_docs:
                    break

                doc_id = int(row["id"])
                doc = MedicineDoc(
                    doc_id=doc_id,
                    clean_name=row.get("clean_name", "").strip(),
                    dosage_info=row.get("dosage_info", "").strip(),
                    uses_str=row.get("uses_str", "").strip(),
                    side_effects_str=row.get("side_effects_str", "").strip(),
                    substitutes_str=row.get("substitutes_str", "").strip(),
                    chemical_class=row.get("chemical_class", "").strip(),
                    habit_forming=row.get("habit_forming", "").strip(),
                    therapeutic_class=row.get("therapeutic_class", "").strip(),
                    action_class=row.get("action_class", "").strip(),
                    uses_processed=row.get("uses_processed", "").strip(),
                    side_effects_processed=row.get(
                        "side_effects_processed", ""
                    ).strip(),
                    search_text=row.get("search_text", "").strip(),
                )

                # --- Store document ----------------------------------------
                self.docs[doc_id] = doc

                # --- Positional mapping for TF-IDF matrix ------------------
                self.doc_id_to_pos[doc_id] = pos
                self.pos_to_doc_id[pos] = doc_id
                self.corpus_texts.append(doc.search_text)

                # --- Term index ---------------------------------------------
                # search_text is already pre-stemmed by Nysa's preprocessing.
                # We store these tokens as-is (no additional normalize_token),
                # which avoids double-stemming mismatches.
                raw_tokens = re.findall(r"[a-z0-9]+", doc.search_text.lower())
                for token in raw_tokens:
                    if token:
                        self.term_index[token].add(doc_id)

                # --- Category index -----------------------------------------
                tc = doc.therapeutic_class.lower().strip()
                if tc:
                    self.category_index[tc].add(doc_id)

                pos += 1

        self._built = True
        logger.info(
            "Index built: %d docs | %d unique terms | %d categories",
            len(self.docs),
            len(self.term_index),
            len(self.category_index),
        )

    # ------------------------------------------------------------------ #
    # Lookup helpers                                                        #
    # ------------------------------------------------------------------ #
    def get_docs_for_term(self, stemmed_term: str) -> set[int]:
        """Return the posting list for a single stemmed term."""
        return self.term_index.get(stemmed_term, set())

    def get_docs_for_category(self, category: str) -> set[int]:
        """
        Return doc ids for a therapeutic category.
        Supports partial prefix matching so ``"resp"`` matches
        ``"respiratory"``.
        """
        category = category.lower().strip()
        # Exact match first
        if category in self.category_index:
            return self.category_index[category]
        # Prefix match
        result: set[int] = set()
        for cat_key, ids in self.category_index.items():
            if cat_key.startswith(category):
                result |= ids
        return result

    def get_all_doc_ids(self) -> set[int]:
        return set(self.docs.keys())

    @property
    def total_documents(self) -> int:
        return len(self.docs)

    def is_built(self) -> bool:
        return self._built
