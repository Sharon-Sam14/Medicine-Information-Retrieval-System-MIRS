"""
MIRS Quick Search CLI & Demo Script (With Ronald's Query Expansion)
Usage:
    python demo_ronald.py "fever headache"
"""

import sys
import logging
from ir_engine.search_engine import MedicineSearchEngine
from nlp_advanced.query_expansion import QueryExpander

# Suppress detailed debug logs during demo
logging.basicConfig(level=logging.WARNING)

def print_results(raw_query, expanded_query, results):
    print("=" * 65)
    print(f"  RAW USER QUERY  : '{raw_query}'")
    print(f"  EXPANDED QUERY  : '{expanded_query}'")
    print(f"  Results Returned: {len(results)}")
    print("=" * 65)

    if not results:
        print("  No matching medicines found.")
        return

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r['medicine_name'].upper()}")
        print(f"    Match Score   : {r['score'] * 100:.1f}%")
        print(f"    Category      : {r['therapeutic_class']}")
        print(f"    Indications   : {r['uses']}")
        
        sb = r['score_breakdown']
        print(f"    Score Details : TF-IDF={sb['tfidf']:.2f} | SymptomMatch={sb['exact_symptom_bonus']:.1f}")

def main():
    print("Loading Medicine IR Engine & Query Expander...")
    engine = MedicineSearchEngine("cleaned_medicines (4).csv", max_docs=5000)
    engine.load()
    expander = QueryExpander()
    print("Engine ready!\n")

    if len(sys.argv) > 1:
        raw_query = " ".join(sys.argv[1:])
        expanded_query = expander.expand_query(raw_query)
        results = engine.search(expanded_query, top_k=5)
        print_results(raw_query, expanded_query, results)
    else:
        print("--- Ronald's Interactive Query Expansion Search ---")
        while True:
            try:
                query = input("Enter query > ").strip()
                if not query or query.lower() in ("exit", "quit", "q"):
                    break
                expanded_query = expander.expand_query(query)
                results = engine.search(expanded_query, top_k=5)
                print_results(query, expanded_query, results)
                print("\n" + "-" * 65 + "\n")
            except (KeyboardInterrupt, EOFError):
                break

if __name__ == "__main__":
    main()