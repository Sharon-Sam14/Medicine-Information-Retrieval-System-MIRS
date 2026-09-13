# RULES.md — Development & Engineering Rules
# Medicine Information Retrieval System (MIRS)

> **Document Version:** 2.0.0  
> **Lead Core IR Architect:** Sharon  
> **Current Date:** September 2026  
> **Repository Policy:** Local Development Only — DO NOT commit or push to GitHub/remote repositories without explicit instruction.

---

## ⚠️ Teammate Development Boundaries & Missing Implementation Rules

> [!IMPORTANT]
> **Strict Team Separation & Boundaries:**
> - **Sharon (Core IR Architect)** has **completed and validated** Phase 2 (`ir_engine/`). Do NOT modify, rewrite, or break Sharon's working IR engine without an explicit user prompt.
> - **Nysa, Ronald, and Sam** must adhere to the specific architectural boundaries outlined below for their pending implementations.

| Team Member | Functional Territory | Permitted Modules | Strict Restrictions & Missing Rules |
|---|---|---|---|
| **Nysa** | Preprocessing & Pattern Matching | `nlp_advanced/pattern_matcher.py`<br>`data_pipeline/` | • Must strictly use Python `re` for dosage patterns.<br>• Must NOT alter column definitions in `cleaned_medicines (4).csv`.<br>• Must NOT introduce heavy NLP models (like spaCy transformers) that exceed memory limits. |
| **Sharon** *(User)* | Core Retrieval & Ranking | `ir_engine/*`<br>`test_ir_engine.py` | • **COMPLETED & LOCKED**. All 36 unit tests pass.<br>• Inverted Index, Boolean AST, TF-IDF VSM, and 5-factor Ranker are stable and production-ready. |
| **Ronald** | Advanced Query Processing | `nlp_advanced/query_expansion.py`<br>`nlp_advanced/thesaurus.json` | • Must output a sanitized query string compatible with Sharon's parser.<br>• Must NOT bypass Sharon's Boolean parser or directly modify the inverted index.<br>• Synonym mappings must be clinically sound and verified. |
| **Sam** | Full-Stack & Feedback Systems | `backend/*`<br>`frontend/*` | • Must treat Sharon's `MedicineSearchEngine` as a read-only singleton service in Django.<br>• Must NOT re-instantiate the engine on every HTTP request.<br>• Must design UI according to `DESIGN.md` (high-contrast dark theme, accessible dosage/side-effect pills).<br>• Must implement Rocchio vector adjustments cleanly in `search_api/rocchio.py`. |

---

## 1. General AI Development Rules

1. **First Understand, Then Implement:** Always inspect existing code and documentation before suggesting changes.
2. **Never Blindly Rewrite Working Code:** Sharon's `ir_engine/` is fully tested and functioning. Do not refactor or replace it with hypothetical alternatives.
3. **Never Invent Files, APIs, or Schema Fields:** Every feature must strictly use the 13 columns present in `cleaned_medicines (4).csv`:
   - `id`, `clean_name`, `dosage_info`, `uses_str`, `side_effects_str`, `substitutes_str`, `chemical_class`, `habit_forming`, `therapeutic_class`, `action_class`, `uses_processed`, `side_effects_processed`, `search_text`.
   - Never reference fictional fields like `price`, `manufacturer`, `expiry_date`, or `doctor_rating`.
4. **Git Discipline:** Under no circumstances should git commands (`git add`, `git commit`, `git push`) be executed without explicit, direct user instructions.
5. **No Cloud Dependencies:** The entire application (React, Django, Scikit-learn, SQLite) must run 100% locally on `localhost`. Do not integrate OpenAI APIs, AWS services, or external cloud vector databases.

---

## 2. Core IR Engine Rules (Sharon's Standard)

### R-01: Zero Document Table Scanning During Queries
The inverted index exists to eliminate $O(N)$ scans. Boolean operations must execute exclusively via posting list operations (set intersections, unions, and complements). Never iterate across `index.docs` checking substring presence at query time.

### R-02: Query Token Normalization Consistency
All query tokens must pass through `preprocessor.normalize_token()` before querying `term_index`. The corpus was indexed using stemmed tokens; un-stemmed raw words will produce false negative misses.

### R-03: Separation of Document IDs and Document Bodies
`BooleanModel.search()` must return `set[int]` (Doc IDs only). Complete `MedicineDoc` instances must only be hydrated at the final ranking stage to preserve memory bandwidth and CPU cache locality.

### R-04: Single-Pass TF-IDF Matrix Fitting
`VectorModel.fit()` is a one-time startup operation (~30 seconds on the full 222k corpus). Guard this with `is_fitted` flags. It must never be re-fitted per query or per web request.

### R-05: Ranking Weight Conservation Law
Any modification to heuristic weights in `ir_engine/ranking.py` must strictly sum to $1.0$:
$$\text{Weight}_{TFIDF} (0.60) + \text{Weight}_{Symptom} (0.20) + \text{Weight}_{Coverage} (0.10) + \text{Weight}_{Name} (0.06) + \text{Weight}_{Category} (0.04) = 1.0$$

### R-06: Graceful Boolean Failure Fallback
If an over-constrained Boolean query produces an empty set, the engine must fall back automatically to full-corpus TF-IDF scoring. A user must never receive a zero-result page if relevant therapeutic matches exist in the dataset.

### R-07: Mandatory Explainability (`score_breakdown`)
Every search result dictionary returned to Django and React must contain the `score_breakdown` mapping displaying exact constituent contributions (TF-IDF, symptom bonus, coverage bonus, name bonus, category bonus).

---

## 3. Backend & API Rules (Sam's Domain)

### R-08: Singleton Engine Pattern in Django
Instantiate Sharon's engine once in `backend/search_api/apps.py` inside `ready()`:
```python
# CORRECT:
from django.apps import AppConfig
from ir_engine import MedicineSearchEngine

class SearchApiConfig(AppConfig):
    name = 'search_api'
    def ready(self):
        global ENGINE
        ENGINE = MedicineSearchEngine("cleaned_medicines (4).csv")
        ENGINE.load()
```
Never instantiate `MedicineSearchEngine()` inside a view function.

### R-09: Robust Query Validation & Error Handling
- Sanitize input strings against script injection and buffer overruns.
- Cap query lengths at 250 characters.
- If a user enters an unclosed parenthesis like `(fever AND headache`, the Boolean parser must either self-heal or degrade to implicit AND without crashing.

### R-10: Local Authentication & Secret Management
- Use Django native authentication combined with SimpleJWT.
- Store sensitive configuration variables (e.g., `SECRET_KEY`) in a local `.env` file; never hardcode secrets in source code.

---

## 4. Frontend Development Rules (Sam's Domain)

### R-11: Adherence to `DESIGN.md` Visual System
- Use the predefined high-contrast dark theme (Midnight Obsidian `#0B0F17`, Surface `#151D2C`, High-contrast text `#F8FAFC`, Accent Cyan `#00F0FF`, Alert Red `#F43F5E`).
- Avoid arbitrary inline colors. Use CSS design tokens defined in `styles/index.css`.
- Avoid TailwindCSS unless explicitly requested; use Vanilla CSS with modular class naming.

### R-12: Medical Accessibility & Usability (WCAG 2.1 AA)
- Ensure all text contrasts meet at least 4.5:1 ratio.
- Side effects must be clearly marked with danger badges and warning icons.
- Dosage numbers must use monospace typography (`JetBrains Mono` or `Courier`) for unambiguous numeric legibility.

---

## 5. Teammate Integration Verification Rules

1. Before merging any teammate code, run the full test suite:
   ```bash
   python test_ir_engine.py
   ```
2. All 36 tests must pass with 0 failures and 0 errors.
3. Memory consumption must remain under 2.0 GB RAM during test execution.
