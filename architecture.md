# ARCHITECTURE.md — System Architecture & Component Design
# Medicine Information Retrieval System (MIRS)

> **Document Version:** 2.0.0  
> **Lead Core IR Architect:** Sharon  
> **Current Date:** September 2026  
> **Repository Policy:** Local Development Only — DO NOT commit or push to GitHub/remote repositories without explicit instruction.

---

## ⚠️ Teammate Architecture Status & Missing Components Matrix

> [!IMPORTANT]
> **Component Ownership & Status:**
> - **Sharon (Core IR Architect)**: **Phase 2 COMPLETED & OPERATIONAL ✅** (`ir_engine/` package including Inverted Index, Boolean AST Parser, TF-IDF Vector Space Model, and 5-Factor Clinical Primacy Ranker).
> - **Nysa, Ronald, and Sam**: Architectural layers are **PENDING / MISSING** as detailed below.

| Architectural Layer | Component File / Path | Component Owner | Status | Missing / Required Teammate Implementation |
|---|---|---|---|---|
| **Data Ingestion & Pattern Matching** | `nlp_advanced/pattern_matcher.py`<br>`data_pipeline/ingest.py` | **Nysa** (Data Engineer) | 🔴 **MISSING** | 1. Implement regex dosage parser (`parse_dosage()`) to isolate numerical strength and unit.<br>2. Build fuzzy drug name matcher (`match_generic_spell()`) using Levenshtein/Regex distance.<br>3. Verify consistency between raw data transformations and `search_text` tokens. |
| **Core IR Engine & Retrieval** | `ir_engine/` (all 6 core modules) | **Sharon** (Core IR Architect) | 🟢 **COMPLETED & TESTED** | **Done & Verified:**<br>• `preprocessor.py` (tokenisation, NLTK Porter stemming)<br>• `inverted_index.py` (O(1) posting list lookups, category index)<br>• `boolean_model.py` (AND/OR/NOT parser, implicit AND)<br>• `vector_model.py` (TF-IDF CSR matrix, cosine similarity)<br>• `ranking.py` (5-factor clinical primacy scoring)<br>• `search_engine.py` (unified search & explain pipeline) |
| **Advanced Query Expansion** | `nlp_advanced/query_expansion.py`<br>`nlp_advanced/thesaurus.json` | **Ronald** (Advanced IR Specialist) | 🔴 **MISSING** | 1. Build synonym mapping dictionary (`thesaurus.json`) linking colloquial terms (`fever`, `headache`) to clinical terms (`pyrexia`, `migraine`).<br>2. Implement query pre-expansion middleware (`expand_query(q) -> str`) to enrich queries before passing to Sharon's `ir_engine`.<br>3. Optimize symptom-to-doc_id posting index access for expanded queries. |
| **Relevance Feedback (Rocchio)** | `backend/search_api/rocchio.py` | **Sam** (Full-Stack Lead) | 🔴 **MISSING** | 1. Implement Rocchio query vector reformulation based on user clickstream and upvotes/downvotes.<br>2. Store session vectors in memory/cache to adjust future query vector representations. |
| **API & Backend Layer** | `backend/search_api/views.py`<br>`backend/core/` | **Sam** (Full-Stack Lead) | 🔴 **MISSING** | 1. Django REST Framework server setup on `localhost:8000`.<br>2. Singleton instantiation of Sharon's `MedicineSearchEngine` inside Django `apps.py`.<br>3. Search & Feedback API routes (`/api/search/`, `/api/feedback/`).<br>4. Local JWT authentication endpoints. |
| **Frontend Web Application** | `frontend/src/` (React App) | **Sam** (Full-Stack Lead) | 🔴 **MISSING** | 1. React SPA running on `localhost:3000`.<br>2. High-contrast dark-themed medical UI (Slate #0B0F17, Cyan #00F0FF, Emerald #10B981).<br>3. Symptom search bar with Boolean helper badges (`AND`, `OR`, `NOT`).<br>4. Medicine cards displaying dosage, precautions, side effects, and score breakdowns.<br>5. Upvote/downvote action buttons feeding Sam's Rocchio service. |

---

## 1. High-Level End-to-End System Architecture

The following diagram illustrates the complete end-to-end flow from the user interface down to data persistence and retrieval:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        USER BROWSER / CLIENT                           │
│     React Single Page Application (Sam - PENDING 🔴)                   │
│     - Search Bar (Symptoms / Boolean syntax / Category filter)         │
│     - High-Contrast Medicine Cards (Dosage, Uses, Side Effects)        │
│     - Feedback Actions (Upvote / Downvote buttons)                     │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ HTTP REST (JSON / JWT)
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│               DJANGO REST FRAMEWORK BACKEND (Sam - PENDING 🔴)         │
│  ┌───────────────────────┐          ┌───────────────────────────────┐  │
│  │ /api/search/ View     │          │ /api/feedback/ View           │  │
│  └──────────┬────────────┘          └──────────────┬────────────────┘  │
│             │                                      │                   │
│             ▼                                      ▼                   │
│  ┌───────────────────────┐          ┌───────────────────────────────┐  │
│  │ Ronald Query Expander │          │ Sam Rocchio Algorithm         │  │
│  │ (Thesaurus Mapping)   │          │ (Vector Adjustment)           │  │
│  │ [PENDING 🔴]          │          │ [PENDING 🔴]                  │  │
│  └──────────┬────────────┘          └──────────────┬────────────────┘  │
└─────────────┼──────────────────────────────────────┼───────────────────┘
              │ Enriched Query                       │ Feedback Vectors
              ▼                                      ▼
┌────────────────────────────────────────────────────────────────────────┐
│            CORE IR RETRIEVAL & RANKING ENGINE (Sharon - COMPLETE ✅)   │
│                                                                        │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ 1. Query Normalization & Preprocessing (preprocessor.py)       │   │
│   │    - Tokenization, lowercasing, NLTK PorterStemmer             │   │
│   └───────────────────────────────┬────────────────────────────────┘   │
│                                   │ Normalized Tokens                  │
│                                   ▼                                    │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ 2. Boolean Model & AST Parser (boolean_model.py)               │   │
│   │    - AND, OR, NOT set intersections over Inverted Index        │   │
│   │    - Category directive parsing (category:RESPIRATORY)         │   │
│   │    - Output: Filtered candidate set {doc_id, ...}              │   │
│   └───────────────────────────────┬────────────────────────────────┘   │
│                                   │ Candidate Doc IDs (or full corpus) │
│                                   ▼                                    │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ 3. Vector Space Model (vector_model.py)                        │   │
│   │    - TF-IDF Sparse Matrix (sublinear TF, ngram_range=(1,2))    │   │
│   │    - Cosine Similarity computation over candidate subset       │   │
│   │    - Output: dict[doc_id -> tfidf_cosine_score]                │   │
│   └───────────────────────────────┬────────────────────────────────┘   │
│                                   │ Raw Relevance Scores               │
│                                   ▼                                    │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ 4. Domain Heuristic Ranker (ranking.py)                        │   │
│   │    - Exact Symptom Indication Bonus (uses_processed match)     │   │
│   │    - Multi-Symptom Coverage Fraction                           │   │
│   │    - Medicine Clean Name Match                                 │   │
│   │    - Therapeutic Category Match                                │   │
│   │    - Min-Max Normalization -> [0.0, 1.0]                       │   │
│   └───────────────────────────────┬────────────────────────────────┘   │
│                                   │ Top-K RankedResult Objects         │
│                                   ▼                                    │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ 5. Unified Facade (search_engine.py)                           │   │
│   │    - Metadata enrichment from InvertedIndex.docs               │   │
│   │    - JSON serializable list[dict] with score_breakdown         │   │
│   └────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   PERSISTENCE & DATA STORAGE                           │
│  - cleaned_medicines (4).csv (222,797 rows, 13 features)               │
│  - SQLite / Local PostgreSQL (User auth, interaction history)          │
│  - In-Memory Inverted Index Posting Lists (Sharon)                     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Communication & Data Contracts

### 2.1 Ronald's Query Expansion → Sharon's IR Engine
Ronald's module receives raw user text and enriches it using a clinical synonym thesaurus before Sharon's engine executes retrieval:

```python
# Ronald's expected contract:
def expand_query(raw_query: str) -> str:
    """
    Expands colloquial symptoms to medical terminology.
    Example: 'fever headache' -> 'fever pyrexia headache migraine'
    """
    pass

# Sharon's engine seamlessly ingests the resulting string:
results = engine.search(query=expanded_query, top_k=10)
```

### 2.2 Sharon's IR Engine → Sam's Django REST API
Sharon's engine returns self-contained, serializable medicine objects complete with transparent score breakdowns:

```python
[
    {
        "doc_id": 142,
        "medicine_name": "paracetamol 500mg tablet",
        "uses": "Treatment of Fever | Relief of mild to moderate pain",
        "side_effects": "Liver dysfunction (rare) | Allergic skin rash",
        "dosage_info": "500mg",
        "therapeutic_class": "ANALGESICS & ANTIPYRETICS",
        "action_class": "COX-Inhibitor",
        "substitutes": "Dolo 650 | Calpol 500",
        "score": 0.9421,
        "score_breakdown": {
            "tfidf": 0.8120,
            "exact_symptom_bonus": 1.0000,
            "multi_symptom_bonus": 1.0000,
            "name_match_bonus": 0.0000,
            "category_bonus": 0.0000
        }
    }
]
```

### 2.3 Sam's Frontend ↔ Django Feedback Route
When a user upvotes or downvotes a result, the frontend issues an asynchronous POST request:
```http
POST /api/feedback/
Content-Type: application/json
Authorization: Bearer <JWT_TOKEN>

{
    "query": "fever headache",
    "doc_id": 142,
    "action": "upvote"  // or "downvote"
}
```
Sam's Rocchio service updates the query profile vectors in memory, reinforcing positive document tokens and damping negative document tokens for subsequent searches.

---

## 3. Technology Stack Specification

| Subsystem | Technology | Version / Tooling | Purpose |
|---|---|---|---|
| **Frontend UI** | React.js | React 18+, Vanilla CSS, Axios | High-contrast accessible UI, responsive layout, state management. |
| **Backend Framework** | Python / Django | Django 5.x, Django REST Framework | RESTful routing, authentication, IR engine hosting. |
| **Local Database** | SQLite / PostgreSQL | SQLite3 (local embedded) | User profiles, audit logs, interaction records. |
| **Authentication** | Django Auth + SimpleJWT | `djangorestframework-simplejwt` | Token-based local authentication without external auth servers. |
| **Core IR Engine** | Python / Scikit-Learn | `scikit-learn>=1.3.0`, `scipy>=1.11.0`, `numpy>=1.24.0` | TF-IDF sparse matrix computation, cosine vector similarity. |
| **Text Processing** | NLTK & Python Regex | `nltk>=3.8.1`, `re` (built-in) | Porter stemming, token normalization, dosage regex pattern extraction. |

---

## 4. Complete Project Directory Structure

```
IR Mini project/
├── cleaned_medicines (4).csv       # Master dataset (222,797 records)
├── requirements.txt                # Python environment specifications
├── test_ir_engine.py               # Sharon's comprehensive test suite (36 tests)
│
├── PRD.md                          # Project Requirements Document
├── ARCHITECTURE.md                 # System Architecture (This file)
├── RULES.md                        # Development & Integration Guidelines
├── PHASES.md                       # Phase-by-Phase Roadmap & Milestones
├── DESIGN.md                       # UI/UX Design System (Sam)
├── MEMORY.md                       # Living Context & Progress Log
│
├── ir_engine/                      # Sharon's Core IR Engine (COMPLETE ✅)
│   ├── __init__.py                 # Package exports (MedicineSearchEngine)
│   ├── preprocessor.py             # Query tokenizer & Porter stemmer
│   ├── inverted_index.py           # Inverted index & posting lists
│   ├── boolean_model.py            # Boolean AST parser (AND/OR/NOT)
│   ├── vector_model.py             # TF-IDF sparse vector space model
│   ├── ranking.py                  # 5-factor domain heuristic ranker
│   └── search_engine.py            # Unified search orchestrator & facade
│
├── nlp_advanced/                   # Ronald's & Nysa's Modules (PENDING 🔴)
│   ├── __init__.py
│   ├── query_expansion.py          # Ronald: Medical thesaurus expander
│   ├── thesaurus.json              # Ronald: Clinical synonym mappings
│   └── pattern_matcher.py          # Nysa: Dosage & fuzzy spell-check regexes
│
├── backend/                        # Sam's Django Backend (PENDING 🔴)
│   ├── manage.py
│   ├── core/                       # Django project configuration & settings
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   ├── urls.py
│   │   └── wsgi.py
│   └── search_api/                 # Search application
│       ├── __init__.py
│       ├── apps.py                 # Engine singleton initialization on startup
│       ├── views.py                # REST endpoints (/search, /feedback)
│       ├── rocchio.py              # Sam: Rocchio relevance feedback
│       └── urls.py
│
└── frontend/                       # Sam's React Application (PENDING 🔴)
    ├── package.json
    ├── public/
    └── src/
        ├── index.js
        ├── App.jsx
        ├── styles/
        │   └── index.css           # Design tokens (Midnight Obsidian Theme)
        └── components/
            ├── SearchBar.jsx       # Query input with Boolean syntax chips
            ├── MedicineCard.jsx    # Card with dosage, uses, side-effects
            ├── ScoreDrawer.jsx     # Explainable score breakdown
            └── FeedbackButtons.jsx # Rocchio thumbs up/down actions
```

---

## 5. Scalability, Latency & Reliability Design

1. **Sub-10ms Scoring via Candidate Subset Filtering:**
   Instead of computing cosine similarity against all 222,797 rows for every query, Sharon's engine executes Boolean filtering first to produce a restricted candidate set (typically 50–500 documents). `VectorModel.score_subset()` then slices only the corresponding rows from the pre-computed TF-IDF sparse matrix, reducing compute by up to $300\times$.
2. **Graceful Fallback:**
   If a user supplies an over-constrained Boolean query (e.g. `fever AND headache AND NOT vomiting AND category:OPHTHALMOLOGICAL`) that yields an empty set, the system automatically degrades to full-corpus TF-IDF vector ranking, ensuring the user always receives the closest available therapeutic alternatives.
3. **Thread-Safe Read-Only Engine:**
   Sharon's `MedicineSearchEngine` loads the inverted index and fitted TF-IDF matrix into memory once during Django startup. All subsequent queries perform read-only lookups, enabling safe concurrent query serving across multiple Django worker threads without lock contention.
