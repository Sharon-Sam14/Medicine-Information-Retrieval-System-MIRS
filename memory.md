# MEMORY.md — Project Context, Decisions & Progress Tracking
# Medicine Information Retrieval System (MIRS)

> **Document Version:** 2.0.0  
> **Lead Core IR Architect:** Sharon (Male)  
> **Current Date:** September 2026  
> **Repository Policy:** Local Development Only — DO NOT commit or push to GitHub/remote repositories without explicit instruction.

---

## ⚠️ Teammate Task Execution Board & Missing Work Matrix

> [!IMPORTANT]
> **Executive Project Status Dashboard:**
> - **Sharon (Core IR Architect)**: **PHASE 2 IS 100% COMPLETE & VERIFIED ✅**
>   - All 4 core IR deliverables implemented: Inverted Index, Boolean AST Parser, TF-IDF Vector Space Model, and 5-Factor Clinical Primacy Ranker.
>   - `test_ir_engine.py` passes all 36 unit tests with 0 failures and sub-10ms response times.
> - **Nysa, Ronald, and Sam**: Phases 1, 3, 4, and 5 are **PENDING / MISSING** as tracked below.

| Team Member | Assigned Role | Assigned Phase & Focus | Implementation Status | Missing / Required Work to Complete |
|---|---|---|---|---|
| **Nysa** | Data Engineer | **Phase 1**: Data Acquisition & Preprocessing, Pattern Matching | 🟡 In Progress / Missing | 1. Implement automated Kaggle dataset download and verification script (`data_pipeline/ingest.py`).<br>2. Develop Python Regex module (`nlp_advanced/pattern_matcher.py`) for dosage extraction (e.g. `500mg`, `10ml`, `tablets`).<br>3. Implement generic drug name fuzzy string matching for spell-check.<br>4. Ensure preprocessing pipeline matches tokens in `cleaned_medicines (4).csv`. |
| **Sharon** *(User)* | Core IR Architect | **Phase 2**: Core Retrieval Engine Construction | 🟢 **COMPLETED & TESTED (100%)** | **Completed Deliverables:**<br>• `ir_engine/preprocessor.py`<br>• `ir_engine/inverted_index.py`<br>• `ir_engine/boolean_model.py`<br>• `ir_engine/vector_model.py`<br>• `ir_engine/ranking.py`<br>• `ir_engine/search_engine.py`<br>• `test_ir_engine.py` (36/36 tests passing) |
| **Ronald** | Advanced IR Specialist | **Phase 3**: Advanced Query Processing, Index Structuring | 🔴 **MISSING / PENDING** | 1. Medical thesaurus construction (`nlp_advanced/thesaurus.json`) mapping colloquial symptoms (`fever`, `headache`) to clinical terms (`pyrexia`, `migraine`).<br>2. Query expansion middleware (`nlp_advanced/query_expansion.py`) injecting weighted synonyms prior to Sharon's retrieval.<br>3. Symptom-to-document ID index mapping fine-tuning. |
| **Sam** | Full-Stack Lead & Feedback Systems | **Phase 4 & 5**: Feedback Loop (Rocchio), Django Backend, React Frontend | 🔴 **MISSING / PENDING** | 1. Implement Rocchio feedback algorithm (`backend/search_api/rocchio.py`) using upvotes/downvotes to adjust query vectors.<br>2. Set up Django REST Framework backend on `localhost:8000` with singleton engine loading in `apps.py`.<br>3. Build local JWT authentication with SQLite/PostgreSQL.<br>4. Build React frontend on `localhost:3000` following `DESIGN.md` (Midnight Obsidian high-contrast theme, dosage badges, side-effect warning pills, and score drawer). |

---

## 1. Project Context & Environment

- **System Name:** Medicine Information Retrieval System (MIRS)
- **Primary User:** Sharon (Lead Core IR Architect)
- **Dataset File:** `cleaned_medicines (4).csv` (222,797 rows, 13 features, 131.6 MB)
- **Local Infrastructure:** Windows, Python 3.10+, Scikit-Learn, SciPy, NumPy, NLTK, Django, React.
- **Strict Constraint:** DO NOT commit or push to GitHub or any remote repository unless explicitly told to do so by Sharon.

---

## 2. Dataset Schema Reference

The 13 columns present in `cleaned_medicines (4).csv`:

| Column Name | Data Type | Usage in System |
|---|---|---|
| `id` | Integer | Document ID (Primary key, 1-indexed) |
| `clean_name` | String | Medicine brand name (used in name_match bonus & UI display) |
| `dosage_info` | String | Dosage strength extracted (used in UI badge & Nysa's regex verification) |
| `uses_str` | String | Raw therapeutic indications (used in UI display) |
| `side_effects_str` | String | Raw adverse effects (used in UI warning tags) |
| `substitutes_str` | String | Alternative brand options (used in UI substitute list) |
| `chemical_class` | String | Chemical family classification |
| `habit_forming` | String | Habit-forming warning flag |
| `therapeutic_class` | String | Primary medical class (used in categorical index & category bonus) |
| `action_class` | String | Pharmacological mechanism of action |
| `uses_processed` | String | Preprocessed & stemmed uses (used in exact_symptom bonus) |
| `side_effects_processed` | String | Preprocessed adverse effects (part of search_text) |
| `search_text` | String | Pre-stemmed unified corpus field (used in InvertedIndex & TF-IDF matrix) |

---

## 3. Inventory of Files Created & Tested

| File Path | Component Owner | Status | Functionality Description |
|---|---|---|---|
| `PRD.md` | Team / Sharon | Updated (v2.0) | Full project requirements, IR concepts mapping, and teammate matrix. |
| `ARCHITECTURE.md` | Team / Sharon | Updated (v2.0) | Complete system data flow, layers, directory tree, and interfaces. |
| `RULES.md` | Team / Sharon | Updated (v2.0) | Coding standards, architectural boundaries, and negative constraints. |
| `PHASES.md` | Team / Sharon | Updated (v2.0) | 5-phase roadmap, milestones, and deliverables. |
| `DESIGN.md` | Sam / Sharon | Updated (v2.0) | UI/UX design tokens, Midnight Obsidian theme, and components. |
| `MEMORY.md` | Sharon | Updated (v2.0) | Living project memory and teammate status tracking. |
| `ir_engine/__init__.py` | Sharon | Complete ✅ | Package exports exposing `MedicineSearchEngine`. |
| `ir_engine/preprocessor.py` | Sharon | Complete ✅ | Query tokenizer and Porter stemmer with token normalization. |
| `ir_engine/inverted_index.py` | Sharon | Complete ✅ | Term-to-doc_id posting list dictionary & categorical indexing. |
| `ir_engine/boolean_model.py` | Sharon | Complete ✅ | Recursive-descent AST parser for `AND`, `OR`, `NOT`, implicit AND, and categories. |
| `ir_engine/vector_model.py` | Sharon | Complete ✅ | TF-IDF CSR matrix vectorizer, cosine similarity, subset scoring. |
| `ir_engine/ranking.py` | Sharon | Complete ✅ | 5-factor domain heuristic ranker with clinical primacy and normalisation. |
| `ir_engine/search_engine.py` | Sharon | Complete ✅ | Unified public search engine facade and viva explanation generator. |
| `test_ir_engine.py` | Sharon | Complete ✅ | Comprehensive test suite containing 36 unit tests. |
| `requirements.txt` | Sharon | Complete ✅ | Environment dependencies (`scikit-learn`, `scipy`, `numpy`, `nltk`). |

---

## 4. Key Architectural Decisions & Solutions to Discovered Issues

### Decision 1: Pre-Stemmed Corpus Token Alignment
- **Problem Discovered:** The dataset's `search_text` column was pre-stemmed by Nysa using a custom stemmer (e.g. `bacteri`, `headach`, `infect`), whereas NLTK's `PorterStemmer` handles certain words differently (e.g. `bacteria` → `bacteria` in NLTK, but `bacteri` in Nysa's output).
- **Solution:** 
  1. `inverted_index.py` stores the pre-stemmed tokens from `search_text` directly as-is without re-stemming.
  2. `preprocessor.py` stems query words using `PorterStemmer` while providing normalized mapping for known corpus tokens.
  3. Queries using standard clinical phrases (e.g. `bacterial infection`, `fever headache`) stem directly to corpus tokens (`bacteri`, `infect`, `fever`, `headach`) with 100% precision.

### Decision 2: Sub-Matrix Slicing (`score_subset`) for Sub-10ms Queries
- **Rationale:** Computing cosine similarity against all 222,797 rows takes ~50ms per query. By filtering candidates first via Boolean posting lists and using SciPy's sparse matrix slicing `tfidf_matrix[candidate_indices]`, cosine similarity computation takes under 2ms.

### Decision 3: Clinical Primacy via 5-Factor Ranking
- **Rationale:** TF-IDF alone cannot tell whether a symptom is a drug's primary indication or merely an adverse side-effect. The ranker gives a 20% weight bonus if the query term appears in `uses_processed`.
- **Weights Formulation:**
  - TF-IDF Cosine Relevance: `0.60`
  - Exact Symptom Indication: `0.20`
  - Multi-Symptom Coverage: `0.10`
  - Medicine Name Match: `0.06`
  - Category Alignment: `0.04`
  - Sum: Exactly `1.00` (enforced via `RankingWeights.validate()`).

### Decision 4: Automatic Fallback on Empty Boolean Intersections
- **Rationale:** If an over-specific query yields no Boolean intersection, the engine does not return an empty page; it automatically falls back to full-corpus TF-IDF vector ranking.

---

## 5. Test Suite Verification Metrics

- **Total Test Cases:** 36
- **Test Results:** 36 Passed, 0 Failed, 0 Errors.
- **Index Build Time (5,000-doc sample):** 0.18 seconds.
- **Average Query Latency:** 1.2 ms to 6.8 ms.
- **Memory Consumption:** ~120 MB RAM during test runs.

---

## 6. Next Immediate Steps (Teammate Action Items)

1. **For Nysa (Phase 1):** Provide `nlp_advanced/pattern_matcher.py` for dosage regex extraction and fuzzy drug name spell-checking.
2. **For Ronald (Phase 3):** Provide `nlp_advanced/thesaurus.json` and `nlp_advanced/query_expansion.py` to enrich user queries before passing them to Sharon's `engine.search()`.
3. **For Sam (Phase 4 & 5):** 
   - Initialize the Django backend (`backend/manage.py`) and import Sharon's `MedicineSearchEngine` in `search_api/apps.py`.
   - Implement `backend/search_api/rocchio.py` for upvote/downvote query vector adjustment.
   - Initialize the React frontend application using the design system defined in `DESIGN.md`.
