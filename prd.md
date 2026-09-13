# PRD.md — Project Requirements Document
# Medicine Information Retrieval System (MIRS)

> **Document Version:** 2.0.0  
> **Lead Core IR Architect:** Sharon  
> **Current Date:** September 2026  
> **Repository Policy:** Local Development Only — DO NOT commit or push to GitHub/remote repositories without explicit instruction.

---

## ⚠️ Teammate Task Status & Missing Deliverables Matrix

> [!IMPORTANT]
> **Current Team Status Summary:**
> - **Sharon (Core IR Architect)**: **Phase 2 COMPLETED ✅** (Boolean IR model, TF-IDF Vector Space Model, 5-Factor Manual Ranking Heuristics, Unified Search API — verified with 36/36 passing unit tests).
> - **Nysa, Ronald, and Sam**: Deliverables are **PENDING / MISSING** as specified below.

| Team Member | Assigned Role | Assigned Phase & Concepts | Status | Missing / Needs To Be Done by Teammate |
|---|---|---|---|---|
| **Nysa** | Data Engineer | **Phase 1**: Data Acquisition & Preprocessing, Pattern Matching | 🟡 In Progress / Missing | 1. Implement automated Kaggle dataset extraction script.<br>2. Build regex pattern matching for dosage extraction (e.g. `mg`, `ml`, `tablets`).<br>3. Implement regex fuzzy string matcher for generic drug name spell-checking.<br>4. Validate stop-word removal and custom stemming pipeline consistency with `cleaned_medicines (4).csv`. |
| **Sharon** *(User)* | Core IR Architect | **Phase 2**: Core Retrieval Engine Construction | 🟢 **COMPLETED & TESTED** | **Done:**<br>• Inverted Index (`ir_engine/inverted_index.py`)<br>• Boolean IR Model with AND/OR/NOT & implicit AND (`ir_engine/boolean_model.py`)<br>• TF-IDF Vector Space Model & Cosine Similarity (`ir_engine/vector_model.py`)<br>• 5-Factor Manual Ranking Heuristics (`ir_engine/ranking.py`)<br>• Unified Search Engine API (`ir_engine/search_engine.py`) |
| **Ronald** | Advanced IR Specialist | **Phase 3**: Advanced Query Processing, Index Structuring | 🔴 **MISSING / PENDING** | 1. Medical thesaurus construction for synonym mapping (e.g. `headache` ↔ `migraine`, `fever` ↔ `high temperature`, `pyrexia`).<br>2. Query expansion algorithm to inject expanded terms into user queries before Sharon's retrieval engine.<br>3. Fine-tuning symptom-to-document ID inverted index mapping for expanded terminology. |
| **Sam** | Full-Stack Lead & Feedback Systems | **Phase 4 & 5**: Feedback Loop (Rocchio), Django Backend, React Frontend | 🔴 **MISSING / PENDING** | 1. Rocchio relevance feedback algorithm implementation using upvote/downvote clicks.<br>2. Django REST Framework backend (`search_api`) connecting endpoints to Sharon's `ir_engine`.<br>3. Local JWT authentication and user interaction tracking in SQLite/PostgreSQL.<br>4. High-contrast dark-themed React frontend displaying medicines, dosage badges, side effects, and upvote/downvote buttons. |

---

## 1. Project Overview & Vision

The **Medicine Information Retrieval System (MIRS)** is a clinical-grade, offline-first search and recommendation system designed to bridge the gap between complex medical terminologies and everyday symptom descriptions. 

Using a local dataset of over 222,000 pharmaceutical formulations (`cleaned_medicines (4).csv`), MIRS enables patients, students, and healthcare practitioners to input layperson symptoms, receive accurate, ranked medicine profiles, and understand indications, dosages, and adverse side effects transparently without relying on external cloud APIs.

---

## 2. Problem Statement

1. **Information Asymmetry:** Medical datasets contain clinical terminology (e.g., *pyrexia*, *cephalalgia*, *gastroesophageal reflux*) that everyday patients cannot easily search using colloquial terms (*fever*, *headache*, *acidity*).
2. **Naive Keyword Search Inefficiency:** Scanning 222,797 records using raw substring matching takes seconds per query ($O(N)$), fails to rank by clinical relevance, and lacks multi-symptom logical operations.
3. **Lack of Clinical Primacy in Pure Keyword Search:** Standard search engines treat a medicine that lists "nausea" as a side-effect identically to one where "nausea" is the primary therapeutic indication.
4. **Cloud & Privacy Risks in Health Queries:** Searching sensitive medical queries on public web engines exposes personal medical history. MIRS provides a 100% local, privacy-preserving infrastructure.

---

## 3. Target Users

- **General Public / Patients:** Individuals experiencing symptoms seeking to understand over-the-counter or prescribed treatments, dosage formats, and potential side effects.
- **Pharmacy & Medical Students:** Learners needing structured queries (Boolean logic, therapeutic class filtering, chemical class associations) for academic research.
- **Healthcare Facilitators:** Local clinic workers needing fast offline drug discovery without internet access.

---

## 4. IR Concepts Implementation Mapping

| IR Concept | System Implementation Details | Module Owner | Current Status |
|---|---|---|---|
| **Text Preprocessing** | Stop words removal, lowercasing, stemming/lemmatization, token standardization. Pre-indexed corpus field `search_text`. | Nysa | 🟡 Dataset provided; standalone script missing |
| **Boolean IR Model** | Strict filtering queries using AND, OR, NOT, nested parentheses, and category filters (e.g., `symptom='fever' AND category='ANTI INFECTIVES'`). | Sharon | 🟢 **Completed & Tested** (`ir_engine/boolean_model.py`) |
| **Vector Space Model** | TF-IDF sparse matrix computation, sublinear term frequency scaling, bigram n-grams (`(1,2)`), cosine similarity ranking. | Sharon | 🟢 **Completed & Tested** (`ir_engine/vector_model.py`) |
| **Pattern Matching** | Regular expressions for dosage extraction (`\d+\s*(?:mg|ml|mcg|tablets)`) and generic name fuzzy spell-checking. | Nysa | 🔴 Missing (needs Nysa's module) |
| **Query Expansion** | Medical thesaurus synonym mapping expanding input queries (e.g. `fever` → `fever temperature pyrexia`). | Ronald | 🔴 Missing (needs Ronald's module) |
| **Manual Document Retrieval & Ranking** | 5-factor domain ranking heuristics combining TF-IDF cosine score (0.60), exact primary symptom indication bonus (0.20), multi-symptom coverage bonus (0.10), medicine name match (0.06), and category alignment (0.04). | Sharon | 🟢 **Completed & Tested** (`ir_engine/ranking.py`) |
| **Query Reformulation** | Rocchio algorithm modifying subsequent query vectors based on user interaction feedback (upvotes/downvotes/clicks). | Sam | 🔴 Missing (needs Sam's module) |

---

## 5. Dataset Specification

The system operates on local Kaggle dataset `cleaned_medicines (4).csv`:

- **Total Documents ($N$):** 222,797 medicines
- **Total Columns:** 13
- **Primary Schema:**
  1. `id` *(Integer)*: Unique Document ID
  2. `clean_name` *(String)*: Standardized medicine/brand name
  3. `dosage_info` *(String)*: Dosage strength/measure
  4. `uses_str` *(String)*: Raw human-readable therapeutic uses
  5. `side_effects_str` *(String)*: Raw comma/pipe separated side effects
  6. `substitutes_str` *(String)*: Alternative medicine brands
  7. `chemical_class` *(String)*: Chemical classification
  8. `habit_forming` *(String)*: Habit-forming warning flag
  9. `therapeutic_class` *(String)*: Broad medical category (e.g., `RESPIRATORY`, `ANTI INFECTIVES`)
  10. `action_class` *(String)*: Mechanism of action class
  11. `uses_processed` *(String)*: Preprocessed & stemmed therapeutic indications
  12. `side_effects_processed` *(String)*: Preprocessed adverse effects
  13. `search_text` *(String)*: Unified indexed text (name + uses + side effects)

---

## 6. Functional Requirements

### 6.1 Core IR Engine (Sharon — Complete ✅)
- **FR-01: Inverted Index Construction:** Must parse `cleaned_medicines (4).csv` into a memory-efficient posting list index `term_index: dict[str, set[int]]` and `category_index: dict[str, set[int]]`.
- **FR-02: Boolean Query Parser:** Must evaluate expressions with `AND`, `OR`, `NOT`, implicit AND between adjacent words, nested parentheses, and category directives (`category:CLASS`).
- **FR-03: Vector Space Model Scoring:** Must build a TF-IDF matrix using `sublinear_tf=True` and `ngram_range=(1,2)`. Must compute cosine similarity on Boolean candidate subsets for sub-10ms response times.
- **FR-04: Domain Heuristic Ranker:** Must combine TF-IDF with exact symptom match bonuses (clinical primacy in `uses_processed`), multi-symptom coverage, name matching, and category matching.
- **FR-05: Unified Search Interface:** Must provide `search(query, top_k)` returning structured medicine metadata, normalized scores in $[0, 1]$, and explainable `score_breakdown` dictionaries.

### 6.2 Preprocessing & Pattern Matching (Nysa — Pending 🔴)
- **FR-06: Preprocessing Pipeline:** Automated script to reproduce tokenization, lowercasing, and custom stemming on new drug entries.
- **FR-07: Dosage Pattern Matching:** Regex engine to parse strengths and dosage formats (`500mg`, `10ml`, `Duo Tablet`) from user inputs or unstructured descriptions.
- **FR-08: Fuzzy Spell-Check:** Regex/Levenshtein matching to detect and correct common drug name misspellings.

### 6.3 Advanced Query Processing (Ronald — Pending 🔴)
- **FR-09: Medical Thesaurus:** Mapping colloquial symptoms to formal medical terminology (e.g., `loose motions` → `diarrhea`, `body ache` → `myalgia`).
- **FR-10: Query Expansion Pipeline:** Pre-retrieval module enriching input queries with controlled synonym weights before passing to Sharon's engine.

### 6.4 Feedback, Backend & Frontend (Sam — Pending 🔴)
- **FR-11: Rocchio Relevance Feedback:** Capture user clicks, upvotes (relevant), and downvotes (non-relevant) to adjust query vector weights:
  $$\vec{q}_{new} = \alpha \vec{q}_0 + \beta \frac{1}{|D_r|} \sum_{d \in D_r} \vec{d} - \gamma \frac{1}{|D_{nr}|} \sum_{d \in D_{nr}} \vec{d}$$
- **FR-12: Django REST Framework API:** Endpoints (`/api/search/`, `/api/feedback/`, `/api/auth/`) serving JSON responses with pagination.
- **FR-13: High-Contrast Dark Theme UI:** React web application featuring intuitive search, boolean syntax helper chips, medicine cards, dosage badges, side-effect warning pills, and score breakdown toggles.

---

## 7. Non-Functional Requirements

- **Query Latency:** $\le 500\text{ ms}$ on full 222,797 dataset; sub-15ms on pre-filtered Boolean subsets.
- **Memory Footprint:** $\le 2.0\text{ GB}$ RAM in local execution.
- **Explainability:** 100% transparent scoring breakdown available on every result for academic and viva demonstration.
- **Privacy & Security:** Zero cloud data leakage; local SQLite/PostgreSQL with native Django authentication + JWT tokens.
- **Reliability:** Graceful degradation — fallback to full-corpus TF-IDF if strict Boolean filters return 0 matches.

---

## 8. MVP Scope vs Future Enhancements

| Category | Minimum Viable Product (MVP) | Future / Optional Enhancements |
|---|---|---|
| **Retrieval** | Inverted index, Boolean parser, TF-IDF cosine similarity, 5-factor manual ranking (Sharon ✅) | BM25 probabilistic model comparison, dense vector embeddings (BioBERT). |
| **Query Processing** | Implicit AND, basic negation, category filtering (Sharon ✅) | Multi-level medical ontology (MeSH, SNOMED-CT) dynamic expansion (Ronald). |
| **Backend** | DRF endpoint calling Sharon's engine with query parameter (Sam 🔴) | Distributed Celery workers, Redis caching layer. |
| **Frontend** | React dark-mode search interface with dosage and side effects cards (Sam 🔴) | Audio voice-to-text symptom search, multi-language localization. |
| **Feedback** | Upvote/downvote buttons adjusting session weights (Sam 🔴) | Personalized user health profiles and persistent clinical history. |
