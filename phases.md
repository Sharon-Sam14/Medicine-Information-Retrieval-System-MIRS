# PHASES.md — Project Development Phases & Roadmap
# Medicine Information Retrieval System (MIRS)

> **Document Version:** 2.0.0  
> **Lead Core IR Architect:** Sharon  
> **Current Date:** September 2026  
> **Repository Policy:** Local Development Only — DO NOT commit or push to GitHub/remote repositories without explicit instruction.

---

## ⚠️ Teammate Phase Execution Board & Missing Deliverables

> [!IMPORTANT]
> **Phase Execution Summary:**
> - **Sharon (Core IR Architect)**: **Phase 2 is 100% COMPLETED & VERIFIED ✅** (Inverted index, Boolean retrieval, TF-IDF vector space model, 5-factor clinical ranking, and full test suite passing 36/36 tests).
> - **Nysa, Ronald, and Sam**: Phases 1, 3, 4, and 5 contain **PENDING / MISSING** deliverables required to achieve full project completion.

```
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│        PHASE 1          │     │        PHASE 2          │     │        PHASE 3          │
│   Data Preprocessing    │ ──> │   Core IR Engine        │ ──> │   Query Expansion       │
│      Owner: Nysa        │     │      Owner: Sharon      │     │      Owner: Ronald      │
│   🟡 IN PROGRESS/MISSING│     │   🟢 COMPLETED & TESTED │     │   🔴 PENDING/MISSING    │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
                                                                             │
                                                                             ▼
┌─────────────────────────┐                                     ┌─────────────────────────┐
│        PHASE 5          │                                     │        PHASE 4          │
│   Full-Stack UI & API   │ <────────────────────────────────── │  Feedback Loop (Rocchio)│
│       Owner: Sam        │                                     │    Owner: Sam + Sharon  │
│   🔴 PENDING/MISSING    │                                     │    🔴 PENDING/MISSING   │
└─────────────────────────┘                                     └─────────────────────────┘
```

| Phase | Phase Name | Primary Owner | Status | Missing / Needs To Be Done by Teammate |
|---|---|---|---|---|
| **Phase 1** | Data Acquisition & Preprocessing | **Nysa** | 🟡 In Progress / Missing | • Dataset is provided (`cleaned_medicines (4).csv`).<br>• Automated Kaggle data ingestion script needed.<br>• Regex dosage pattern extraction module needed.<br>• Fuzzy string matching for drug name spelling needed. |
| **Phase 2** | Core Retrieval Engine Construction | **Sharon** *(User)* | 🟢 **COMPLETED & TESTED** | • **DONE:** Inverted index with posting lists.<br>• **DONE:** Boolean model with AND/OR/NOT parser.<br>• **DONE:** TF-IDF Vector Space Model & cosine similarity.<br>• **DONE:** 5-factor manual ranking with clinical primacy.<br>• **DONE:** 36/36 test cases passing in `test_ir_engine.py`. |
| **Phase 3** | Advanced Query Processing | **Ronald** | 🔴 **MISSING** | • Medical thesaurus mapping (colloquial → clinical synonyms).<br>• Query expansion module before Sharon's retrieval pipeline.<br>• Symptom-to-document ID index mapping fine-tuning. |
| **Phase 4** | Feedback Loop & Ranking Optimization | **Sam** (assisted by Sharon) | 🔴 **MISSING** | • User upvote/downvote interaction logger.<br>• Rocchio feedback algorithm modifying query vector weights.<br>• Evaluation of ranking changes based on feedback loops. |
| **Phase 5** | Application Interface & Local Deployment | **Sam** | 🔴 **MISSING** | • Django REST Framework API routing (`/search`, `/feedback`).<br>• Local JWT authentication with SQLite/PostgreSQL.<br>• High-contrast dark-themed React user interface.<br>• Clear display for dosage, indications, precautions, and side effects. |

---

## Phase 1: Data Acquisition & Preprocessing

- **Assigned Owner:** Nysa (Data Engineer)
- **Objective:** Ingest, inspect, clean, and standardize the raw pharmaceutical dataset, preparing normalized tokens for indexing and extraction.
- **Key Deliverables:**
  1. Download and manage local Kaggle dataset `cleaned_medicines (4).csv` (222,797 rows).
  2. Implement text preprocessing pipeline: tokenization, lowercasing, stop-word removal, and custom stemming.
  3. Develop pattern-matching regex utilities for dosage extraction (e.g., `500mg`, `10ml`, `tablets`).
  4. Build fuzzy string pattern matching for generic medicine spell-checking.
- **Completion Criteria:**
  - Standardized CSV with verified columns: `id`, `clean_name`, `dosage_info`, `uses_str`, `side_effects_str`, `therapeutic_class`, `uses_processed`, `side_effects_processed`, `search_text`.
  - Standalone script `data_pipeline/ingest.py` demonstrating zero data loss and reproducible preprocessing.

---

## Phase 2: Core Retrieval Engine Construction

- **Assigned Owner:** Sharon (Core IR Architect) — **COMPLETED & VALIDATED ✅**
- **Objective:** Construct the high-performance retrieval and ranking core capable of sub-10ms queries over 222,797 records.
- **Key Deliverables:**
  1. **Inverted Index (`ir_engine/inverted_index.py`):**
     - Term-to-doc_id posting list dictionary with $O(1)$ lookups.
     - Categorical index over `therapeutic_class`.
     - Bidirectional positional mapping for TF-IDF vector matrix synchronization.
  2. **Boolean Retrieval Model (`ir_engine/boolean_model.py`):**
     - Recursive-descent AST parser supporting `AND`, `OR`, `NOT`, and parenthesized expressions.
     - Implicit AND injection for space-delimited queries (`fever headache` → `fever AND headache`).
     - Category-directed filtering (`category:RESPIRATORY`).
  3. **Vector Space Model (`ir_engine/vector_model.py`):**
     - Scikit-Learn `TfidfVectorizer` with `sublinear_tf=True`, `ngram_range=(1,2)`, `min_df=2`, `max_df=0.90`.
     - Sub-matrix slicing `score_subset()` for Boolean candidate sets.
     - Full-corpus fallback scoring `score_all()`.
  4. **Domain Heuristic Ranker (`ir_engine/ranking.py`):**
     - 5-factor clinical primacy scoring formula:
       $$\text{Score} = 0.60 \cdot \text{TFIDF} + 0.20 \cdot \text{SymptomBonus} + 0.10 \cdot \text{CoverageBonus} + 0.06 \cdot \text{NameBonus} + 0.04 \cdot \text{CategoryBonus}$$
     - Min-max score normalization to $[0.0, 1.0]$.
     - Transparent `score_breakdown` diagnostic dictionary for every result.
  5. **Unified Search Engine Facade (`ir_engine/search_engine.py`):**
     - Single entry point `MedicineSearchEngine.search(query, top_k)`.
- **Completion Criteria:**
  - Full test suite `test_ir_engine.py` passes 36/36 tests with 0 failures and 0 errors.
  - Sub-15ms query execution on filtered subsets.

---

## Phase 3: Advanced Query Processing

- **Assigned Owner:** Ronald (Advanced IR Specialist)
- **Objective:** Bridge the semantic gap between layperson symptom queries and formal clinical documentation through query expansion and index optimization.
- **Key Deliverables:**
  1. Construct a medical thesaurus dictionary (`nlp_advanced/thesaurus.json`) mapping colloquial symptoms:
     - `fever` ↔ `high temperature`, `pyrexia`, `febrile`
     - `headache` ↔ `migraine`, `cephalalgia`
     - `loose motion` ↔ `diarrhea`, `gastroenteritis`
     - `stomach pain` ↔ `abdominal cramps`, `gastritis`
  2. Implement pre-retrieval query expansion module (`nlp_advanced/query_expansion.py`):
     - Function `expand_query(raw_query: str) -> str` that appends weighted synonyms.
  3. Optimize symptom-to-document ID inverted index mapping to support multi-word expansion terms.
- **Completion Criteria:**
  - Queries like "high temperature" retrieve fever-indicated medications even if the word "temperature" is absent from the medicine's primary uses.
  - Expanded queries remain 100% syntactically valid when fed into Sharon's Boolean parser.

---

## Phase 4: Feedback Loop & Ranking Optimization

- **Assigned Owners:** Sam (Full-Stack Lead) in collaboration with Sharon
- **Objective:** Implement relevance feedback mechanisms that enable the retrieval system to adapt to user interaction signals over time.
- **Key Deliverables:**
  1. Interaction tracking service capturing user click-throughs, upvotes (relevance confirmation), and downvotes (irrelevant penalty).
  2. Rocchio Algorithm implementation (`backend/search_api/rocchio.py`):
     - Reformulate query vectors dynamically:
       $$\vec{q}_{m} = \alpha \vec{q}_0 + \beta \frac{1}{|D_r|} \sum_{d \in D_r} \vec{d} - \gamma \frac{1}{|D_{nr}|} \sum_{d \in D_{nr}} \vec{d}$$
     - Tune baseline hyperparameters ($\alpha = 1.0, \beta = 0.75, \gamma = 0.25$).
  3. Query weight re-ranking mechanism integrating updated weights into Sharon's vector scoring pipeline.
- **Completion Criteria:**
  - Repeated search sessions show measurable rank elevation for frequently upvoted medicines.
  - Downvoted medicines decrease in rank position for identical query tokens.

---

## Phase 5: Application Interface & Deployment

- **Assigned Owner:** Sam (Full-Stack Lead)
- **Objective:** Construct a responsive, high-contrast web application backed by local Django APIs to present medicine information clearly and securely.
- **Key Deliverables:**
  1. **Django REST Framework Backend:**
     - Initialize `search_api` app and instantiate Sharon's `MedicineSearchEngine` as a singleton.
     - Endpoints: `GET /api/search/?q=...&top_k=...` and `POST /api/feedback/`.
     - Local JWT authentication and SQLite/PostgreSQL interaction storage.
  2. **React Frontend Application:**
     - High-contrast dark theme (Midnight Obsidian `#0B0F17`, Cyan `#00F0FF`, Emerald `#10B981`, Rose `#F43F5E`).
     - Interactive search bar with Boolean operator chips (`AND`, `OR`, `NOT`, `category:`).
     - Informative medicine cards clearly separating:
       - Medicine Name & Therapeutic Class
       - Recommended Dosage & Strength
       - Clinical Indications (Uses)
       - Adverse Side Effects with high-contrast warning badges
       - Generic Substitutes
     - Explainable AI drawer displaying Sharon's `score_breakdown`.
     - Interactive upvote/downvote buttons triggering Phase 4 feedback.
- **Completion Criteria:**
  - Fully accessible web app running on `http://localhost:3000` connected to `http://localhost:8000`.
  - Meets WCAG 2.1 AA contrast standards for all medical and dosage text.
  - End-to-end user journey verified from search input to feedback submission.
