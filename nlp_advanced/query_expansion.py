import json
import os
import re

class QueryExpander:
    """
    Query Expansion Module (Phase 3).
    Expands layperson terms into structured Boolean synonym clauses.
    """
    def __init__(self, thesaurus_path: str = None):
        if thesaurus_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            thesaurus_path = os.path.join(base_dir, "thesaurus.json")
        
        self.thesaurus = self._load_thesaurus(thesaurus_path)

    def _load_thesaurus(self, path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Sort keys by length descending so longer phrases match before individual words
                return dict(sorted(data.items(), key=lambda x: len(x[0]), reverse=True))
        return {}

    def expand_query(self, query: str) -> str:
        if not query or not query.strip():
            return query

        clean_q = query.strip()

        # Preserve category directives like category:RESPIRATORY
        category_match = re.search(r"category:([A-Za-z0-9_\s]+)", clean_q, re.IGNORECASE)
        category_str = f" {category_match.group(0)}" if category_match else ""
        clean_q = re.sub(r"category:[A-Za-z0-9_\s]+", "", clean_q, flags=re.IGNORECASE).strip()

        # Replace matching phrases with parenthesized OR clauses
        for phrase, synonyms in self.thesaurus.items():
            pattern = re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
            if pattern.search(clean_q):
                unique_syns = list(dict.fromkeys(synonyms))
                expanded_clause = "(" + " OR ".join(unique_syns) + ")"
                clean_q = pattern.sub(expanded_clause, clean_q)

        return (clean_q + category_str).strip()