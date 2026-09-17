"""
test_ir_engine.py — Test Suite for IR Engine
======================================================
Run with:
    python test_ir_engine.py

Tests cover:
  - Preprocessor tokenisation
  - Inverted index construction
  - Boolean AND / OR / NOT
  - TF-IDF scoring
  - Manual ranking
  - Full pipeline search()
"""

import sys
import os
import time
import logging

# Setup logging so test output is informative
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_ir_engine")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ir_engine.preprocessor import preprocess_query, tokens_to_plain_string
from ir_engine.inverted_index import InvertedIndex
from ir_engine.boolean_model import BooleanModel
from ir_engine.vector_model import VectorModel
from ir_engine.ranking import Ranker
from ir_engine.search_engine import MedicineSearchEngine

# --- CONFIG -----------------------------------------------------------------
CSV_PATH = os.path.join(os.path.dirname(__file__), "cleaned_medicines (4).csv")
MAX_DOCS = 5000   # Use first 5000 rows for fast testing
# -----------------------------------------------------------------------------

PASS = "✅ PASS"
FAIL = "❌ FAIL"


def check(condition: bool, test_name: str, detail: str = "") -> None:
    status = PASS if condition else FAIL
    msg = f"  {status}  {test_name}"
    if detail:
        msg += f"\n         → {detail}"
    print(msg)
    if not condition:
        raise AssertionError(f"Test failed: {test_name}")


# --- TEST 1: Preprocessor ----------------------------------------------------

def test_preprocessor():
    print("\n--- TEST: Preprocessor ---")

    tokens = preprocess_query("Fever AND Headache NOT Cough")
    check("AND" in tokens, "Boolean AND operator preserved")
    check("NOT" in tokens, "Boolean NOT operator preserved")
    check("OR" not in tokens, "OR not injected spuriously")

    tokens2 = preprocess_query("fever headache")
    check(len(tokens2) >= 2, "Plain query tokenised", str(tokens2))

    plain = tokens_to_plain_string(["fever", "AND", "headach"])
    check("AND" not in plain, "Boolean ops stripped for TF-IDF string")
    check("fever" in plain, "Content tokens kept")


# --- TEST 2: Inverted Index --------------------------------------------------

def test_inverted_index(index: InvertedIndex):
    print("\n--- TEST: Inverted Index ---")

    check(index.is_built(), "Index reports built=True")
    check(index.total_documents == MAX_DOCS, f"Loaded {MAX_DOCS} docs")
    check(len(index.term_index) > 100, "Vocabulary > 100 terms")

    # "infect" is the stemmed form of "infection/infections"
    docs_for_infect = index.get_docs_for_term("infect")
    check(len(docs_for_infect) > 0, "'infect' found in index",
          f"{len(docs_for_infect)} docs")

    docs_for_cat = index.get_docs_for_category("respiratory")
    check(len(docs_for_cat) > 0, "Category index: 'respiratory' found",
          f"{len(docs_for_cat)} docs")


# --- TEST 3: Boolean Model ---------------------------------------------------

def test_boolean_model(bm: BooleanModel, index: InvertedIndex):
    print("\n--- TEST: Boolean Model ---")

    # Implicit AND (most common user pattern)
    # "bacterial" stems to "bacteri" and "infection" stems to "infect" via NLTK PorterStemmer
    r1 = bm.search("bacterial infection")
    check(len(r1) > 0, "Implicit AND: 'bacterial infection'", f"{len(r1)} results")

    # Explicit AND
    r2 = bm.search("bacterial AND infection")
    check(isinstance(r2, set), "Explicit AND returns set")

    # OR
    r3 = bm.search("fever OR cough")
    check(len(r3) > 0, "OR query: 'fever OR cough'", f"{len(r3)} results")

    # NOT — result should exclude documents containing the NOT term
    r4 = bm.search("infect NOT bacteri")
    r5 = bm.search("infect")
    check(len(r4) <= len(r5), "NOT reduces result set")

    # Category filter
    r6 = bm.search("category:RESPIRATORY")
    check(len(r6) > 0, "Category filter works", f"{len(r6)} docs in RESPIRATORY")

    # Complex parenthesised expression
    r7 = bm.search("(fever OR cough) AND category:respiratory")
    check(isinstance(r7, set), "Parenthesised expression executed")

    # Empty query
    r8 = bm.search("")
    check(r8 == set(), "Empty query returns empty set")

    print(f"  • 'fever OR cough' → {len(r3)} docs")
    print(f"  • 'category:RESPIRATORY' → {len(r6)} docs")


# --- TEST 4: Vector Model ----------------------------------------------------

def test_vector_model(vm: VectorModel, index: InvertedIndex):
    print("\n--- TEST: Vector Model (TF-IDF) ---")

    check(vm.is_fitted, "VectorModel is fitted")

    scores = vm.score_all("fever headache")
    check(len(scores) > 0, "score_all returns non-empty dict", f"{len(scores)} results")
    check(all(0.0 <= v <= 1.0 for v in scores.values()), "All scores in [0,1]")

    top = vm.search("fever headache", top_k=5)
    check(len(top) == 5, "search() returns top_k=5 results")
    scores_list = [s for _, s in top]
    check(scores_list == sorted(scores_list, reverse=True), "Results sorted descending")

    # Subset scoring
    bool_ids = {1, 2, 3, 4, 5}
    subset_scores = vm.score_subset("fever", bool_ids & set(index.docs.keys()))
    check(len(subset_scores) > 0, "score_subset works on subset")

    print(f"  • Top result: doc_id={top[0][0]}, score={top[0][1]:.4f}")


# --- TEST 5: Ranker ----------------------------------------------------------

def test_ranker(ranker: Ranker, vm: VectorModel):
    print("\n--- TEST: Manual Ranker ---")

    scores = vm.score_all("nausea vomiting")
    top_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:50])

    ranked = ranker.rank("nausea vomiting", top_scores, top_k=10)
    check(len(ranked) <= 10, f"Ranker returns ≤ 10 results (got {len(ranked)})")

    # All final scores should be in [0, 1] after normalisation
    check(
        all(0.0 <= r.final_score <= 1.0 for r in ranked),
        "All normalised scores in [0,1]"
    )

    # Scores should be descending
    fs = [r.final_score for r in ranked]
    check(fs == sorted(fs, reverse=True), "Ranked results sorted descending")

    # Score breakdown should have all keys
    breakdown_keys = {"tfidf", "exact_symptom_bonus", "multi_symptom_bonus",
                      "name_match_bonus", "category_bonus"}
    check(
        all(set(r.to_dict()["score_breakdown"].keys()) == breakdown_keys
            for r in ranked),
        "score_breakdown has all 5 components"
    )

    print(f"  • Top result: {ranked[0].clean_name}  score={ranked[0].final_score:.4f}")


# --- TEST 6: Full Pipeline ---------------------------------------------------

LIVE_QUERIES = [
    ("fever headache",               "Common symptom query"),
    ("nausea vomiting diarrhea",     "Multiple symptoms"),
    ("fever AND headache",           "Explicit AND"),
    ("cough OR cold",                "Explicit OR"),
    ("fever NOT cough",              "NOT exclusion"),
    ("headache migraine",            "Neurological"),
    ("blood pressure",               "Cardiovascular"),
    ("category:RESPIRATORY AND cough", "Category filter + symptom"),
    ("allergic reaction",            "Allergy query"),
    ("stomach pain",                 "GI query"),
]


def test_full_pipeline(engine: MedicineSearchEngine):
    print("\n--- TEST: Full Pipeline (search API) ---")

    for query, label in LIVE_QUERIES:
        t0 = time.perf_counter()
        results = engine.search(query, top_k=5)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        ok = len(results) > 0
        check(ok, f"[{label}] '{query}'",
              f"{len(results)} results in {elapsed_ms:.1f} ms"
              + (f" | top: {results[0]['medicine_name']}" if results else ""))

        # Verify result schema
        if results:
            r = results[0]
            check("medicine_name" in r, "Result has medicine_name")
            check("score" in r, "Result has score")
            check("score_breakdown" in r, "Result has score_breakdown")


def test_explain(engine: MedicineSearchEngine):
    print("\n--- TEST: explain_search() ---")
    expl = engine.explain_search("fever headache", top_k=3)
    check("boolean_result_count" in expl, "explain has boolean_result_count")
    check("top_results" in expl, "explain has top_results")
    print(f"  • Boolean matched: {expl['boolean_result_count']} docs")
    print(f"  • Parsed tokens: {expl['boolean_parsed_tokens']}")


# --- MAIN --------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Medicine IR Engine — Test Suite")
    print(f"  Dataset: {CSV_PATH}")
    print(f"  Using first {MAX_DOCS} documents for speed")
    print("=" * 60)

    if not os.path.exists(CSV_PATH):
        print(f"\n❌  CSV not found at: {CSV_PATH}")
        print("    Update CSV_PATH at the top of this file.")
        sys.exit(1)

    # -- Phase 1: Unit tests ----------------------------------------------
    test_preprocessor()

    # -- Phase 2: Build shared components --------------------------------
    print("\n--- SETUP: Building index and models ---")
    t0 = time.perf_counter()

    index = InvertedIndex()
    index.build_from_csv(CSV_PATH, max_docs=MAX_DOCS)

    bm = BooleanModel(index)
    vm = VectorModel(index)
    vm.fit()
    ranker = Ranker(index)

    print(f"  Setup completed in {time.perf_counter() - t0:.1f} s")

    # -- Phase 3: Component tests -----------------------------------------
    test_inverted_index(index)
    test_boolean_model(bm, index)
    test_vector_model(vm, index)
    test_ranker(ranker, vm)

    # -- Phase 4: Full pipeline test --------------------------------------
    print("\n--- SETUP: Loading full engine ---")
    engine = MedicineSearchEngine(CSV_PATH, max_docs=MAX_DOCS)
    engine.load()

    test_full_pipeline(engine)
    test_explain(engine)

    print("\n" + "=" * 60)
    print("  ✅  All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
