"""
MIRS Quick Search CLI & Demo Script
Usage:
    python demo.py "fever headache"
    python demo.py "fever NOT cough"
    python demo.py "category:RESPIRATORY AND cough"
Or run without arguments for interactive mode:
    python demo.py
"""

import sys
import logging
from ir_engine import MedicineSearchEngine

# Suppress detailed debug logs during demo
logging.basicConfig(level=logging.WARNING)

def print_results(query, results):
    print("=" * 65)
    print(f"  SEARCH QUERY: '{query}'")
    print(f"  Total Results Returned: {len(results)}")
    print("=" * 65)

    if not results:
        print("  No matching medicines found.")
        return

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r['medicine_name'].upper()}")
        print(f"    Match Score   : {r['score'] * 100:.1f}%")
        print(f"    Category      : {r['therapeutic_class']}")
        print(f"    Dosage Form   : {r['dosage_info']}")
        print(f"    Indications   : {r['uses']}")
        print(f"    Side Effects  : {r['side_effects']}")
        if r['substitutes'] and r['substitutes'] != 'NA':
            print(f"    Substitutes   : {r['substitutes'][:80]}...")
        
        # Show score breakdown
        sb = r['score_breakdown']
        print(f"    Score Details : TF-IDF={sb['tfidf']:.2f} | SymptomMatch={sb['exact_symptom_bonus']:.1f} | MultiCoverage={sb['multi_symptom_bonus']:.2f}")

def main():
    print("Loading Medicine IR Engine (first 5,000 records for fast demo)...")
    engine = MedicineSearchEngine("cleaned_medicines (4).csv", max_docs=5000)
    engine.load()
    print("Engine ready!\n")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        results = engine.search(query, top_k=5)
        print_results(query, results)
    else:
        print("--- Interactive Search Mode ---")
        print("Type symptoms (e.g. 'fever headache', 'cough OR cold', 'fever NOT vomit')")
        print("Type 'exit' or 'quit' to stop.\n")
        while True:
            try:
                query = input("Enter query > ").strip()
                if not query or query.lower() in ("exit", "quit", "q"):
                    break
                results = engine.search(query, top_k=5)
                print_results(query, results)
                print("\n" + "-" * 65 + "\n")
            except (KeyboardInterrupt, EOFError):
                break

    print("\nDone.")

if __name__ == "__main__":
    main()
