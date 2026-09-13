# DESIGN.md — UI/UX Design System & Frontend Architecture
# Medicine Information Retrieval System (MIRS)

> **Document Version:** 2.0.0  
> **Lead Core IR Architect:** Sharon  
> **Frontend Lead & Designer:** Sam  
> **Current Date:** September 2026  
> **Repository Policy:** Local Development Only — DO NOT commit or push to GitHub/remote repositories without explicit instruction.

---

## ⚠️ Teammate UI/UX Implementation Responsibilities & Missing Work

> [!IMPORTANT]
> **Design Ownership & Execution:**
> - **Sam (Full-Stack Lead)** is responsible for implementing this complete UI/UX design system in React during **Phase 5**.
> - **Sharon (Core IR Architect)** has completed the backend engine data contract (`ir_engine/search_engine.py`) which delivers the exact JSON schema consumed by these UI components.

| Component / Layer | Responsible Member | Status | Missing / Needs To Be Done |
|---|---|---|---|
| **Data Contract & JSON Schema** | **Sharon** *(User)* | 🟢 **COMPLETED** | Provided: `clean_name`, `dosage_info`, `uses_str`, `side_effects_str`, `substitutes_str`, `therapeutic_class`, `score`, `score_breakdown`. |
| **Design Tokens & Theme Setup** | **Sam** | 🔴 **MISSING** | Create `frontend/src/styles/index.css` defining the Midnight Obsidian tokens, typography scales, and CSS custom properties. |
| **Search Bar & Boolean Chips** | **Sam** | 🔴 **MISSING** | Build `SearchBar.jsx` with input field, auto-suggestions, and quick-insert chips for `AND`, `OR`, `NOT`, and `category:`. |
| **Medicine Card Component** | **Sam** | 🔴 **MISSING** | Build `MedicineCard.jsx` displaying high-contrast dosage badges, expandable uses, and danger-styled side effect pills. |
| **Explainable AI Score Drawer** | **Sam** | 🔴 **MISSING** | Build `ScoreDrawer.jsx` to render visual progress bars representing Sharon's `score_breakdown` (TF-IDF, symptom bonus, etc.). |
| **Relevance Feedback Widget** | **Sam** | 🔴 **MISSING** | Build `FeedbackButtons.jsx` (thumbs up/down) calling Sam's Phase 4 Rocchio feedback API. |

---

## 1. Visual Identity: Clinical "Midnight Obsidian"

The visual identity of MIRS is modeled after clinical decision support dashboards. It uses an ultra-dark, high-contrast palette to ensure medical readability, prevent eye fatigue, and prominently surface critical health warnings (such as contraindications and adverse reactions).

- **Theme Paradigm:** High-Contrast Dark Mode ("Midnight Obsidian & Emerald/Teal Cyan").
- **Design Metaphor:** Precision laboratory diagnostics meets modern consumer search.
- **Accessibility Level:** Target WCAG 2.1 AA (Minimum 4.5:1 contrast ratio for body text, 7:1 for critical medical warnings).

---

## 2. Color Palette & Token System

```css
:root {
  /* Background Surfaces */
  --bg-primary: #070A0F;        /* Deep void black/blue */
  --bg-surface: #0F172A;        /* Slate card surface */
  --bg-surface-hover: #1E293B;  /* Elevated interactive surface */
  --bg-glass: rgba(15, 23, 42, 0.75); /* Glassmorphic backdrop */

  /* Borders & Dividers */
  --border-subtle: #334155;     /* Subtle container borders */
  --border-glow: #00F0FF;       /* Active focus glow */

  /* Text & Typography */
  --text-high-contrast: #F8FAFC; /* Primary headings and medicine titles */
  --text-body: #E2E8F0;          /* Readable body copy and descriptions */
  --text-muted: #94A3B8;         /* Subtitles, secondary metadata */
  --text-dim: #64748B;           /* Placeholders and hints */

  /* Accent & Category Colors */
  --accent-cyan: #00F0FF;        /* Primary interactive brand color */
  --accent-cyan-dim: rgba(0, 240, 255, 0.15);
  --accent-emerald: #10B981;     /* Primary therapeutic indications (Uses) */
  --accent-emerald-dim: rgba(16, 185, 129, 0.15);
  
  /* Clinical Status Alerts */
  --alert-danger: #F43F5E;       /* Adverse side effects & habit warnings */
  --alert-danger-dim: rgba(244, 63, 94, 0.15);
  --alert-warning: #F59E0B;      /* Precautions and dosage warnings */
  --alert-warning-dim: rgba(245, 158, 11, 0.15);
  --alert-info: #3B82F6;         /* Chemical class & category pills */
}
```

---

## 3. Typography Hierarchy

| Level | Font Family | Size | Weight | Line Height | Tracking | Usage |
|---|---|---|---|---|---|---|
| **Display / H1** | `Outfit`, sans-serif | 36px (2.25rem) | 700 (Bold) | 1.2 | -0.02em | Application Title / Hero header |
| **Heading / H2** | `Outfit`, sans-serif | 24px (1.5rem) | 600 (Semi-bold) | 1.3 | -0.01em | Medicine Brand Name (`clean_name`) |
| **Subheading / H3**| `Inter`, sans-serif | 18px (1.125rem) | 600 (Semi-bold) | 1.4 | normal | Section titles (Uses, Side Effects) |
| **Body Primary** | `Inter`, sans-serif | 15px (0.9375rem)| 400 (Regular) | 1.6 | normal | General medical descriptions |
| **Badge / Pill** | `Inter`, sans-serif | 12px (0.75rem) | 600 (Semi-bold) | 1.0 | +0.05em | Category, Therapeutic Class tags |
| **Code / Dosage** | `JetBrains Mono`, mono | 14px (0.875rem) | 500 (Medium) | 1.4 | normal | Dosage (`500mg`), Boolean tokens |

---

## 4. UI Component Specifications

### 4.1 Global Search Bar with Boolean Helper Chips
- **Input Container:** Full width, glassmorphism (`backdrop-filter: blur(12px)`), 56px height, subtle cyan glow on active focus (`box-shadow: 0 0 20px rgba(0, 240, 255, 0.2)`).
- **Helper Chips:** Positioned directly below search input for one-click Boolean query construction:
  - `[+ AND]` (Cyan pill) — Injects logical conjunction.
  - `[+ OR]` (Cyan pill) — Injects logical disjunction.
  - `[+ NOT]` (Rose pill) — Injects exclusion filter.
  - `[category:...]` (Blue pill) — Opens dropdown of available therapeutic classes (e.g. `RESPIRATORY`, `ANTI INFECTIVES`, `ANALGESICS & ANTIPYRETICS`).

### 4.2 High-Contrast Medicine Result Card
Every result from Sharon's engine renders as an elevated slate card (`background: var(--bg-surface)` with `border: 1px solid var(--border-subtle)`):

1. **Card Header:**
   - **Medicine Name:** Bold, 20px, `#F8FAFC`.
   - **Therapeutic Class Badge:** Pill in top-right corner (`background: var(--alert-info)`, color `#FFFFFF`).
   - **Overall Match Score:** Circular or linear progress meter with percentage badge (e.g. `94% Match`).
2. **Dosage & Form Pill:**
   - Monospace font (`JetBrains Mono`), amber outline badge: `💊 Dosage: 625 Duo Tablet`.
3. **Indications Section (Uses):**
   - Icon: Emerald check-circle (`✓`).
   - Highlight matched symptom query words in bright emerald bolding.
4. **Adverse Effects Section (Side Effects):**
   - High-contrast warning container (`border-left: 4px solid var(--alert-danger)`).
   - Side effect tags rendered as rose pills (`background: var(--alert-danger-dim)`, `color: #FDA4AF`).
5. **Score Breakdown Accordion / Drawer:**
   - Toggle button: "View Match Analysis" (viva demonstration & transparency feature).
   - Sliders showing individual factor contributions:
     - TF-IDF Relevance: `0.72`
     - Exact Symptom Indication: `+0.20`
     - Multi-Symptom Coverage: `+0.10`
     - Name Match: `+0.00`
     - Category Match: `+0.04`
6. **Card Footer & Feedback Actions:**
   - Quick view for generic substitutes (`substitutes_str`).
   - Upvote (👍) and Downvote (👎) interactive action buttons communicating with Sam's Rocchio feedback engine.

---

## 5. Layout & Responsive Behavior

| Viewport | Breakpoint | Layout Behavior |
|---|---|---|
| **Desktop** | $\ge 1024\text{px}$ | Two-column dashboard: Left sidebar (Boolean filters, category trees, feedback stats) + Right main column (Search bar + 1-column list of detailed medicine cards). |
| **Tablet** | $768\text{px} - 1023\text{px}$| Single-column stacked layout; collapsible filter accordion above search results. |
| **Mobile** | $< 768\text{px}$ | Compact single-column view; full-width search bar with sticky bottom action sheet for filter pills and score breakdown modal. |

---

## 6. Accessibility & Usability Standards

- **Strict Color Independence:** Never rely solely on color to convey meaning. Side-effects must display an icon (`⚠️`) alongside the red badge. Primary uses must display a checkmark (`✓`).
- **Keyboard Navigation:** Full accessibility support for `Tab`, `Enter`, and `Escape` across search inputs, filter chips, and medicine card accordions.
- **Screen Reader Support:** Use ARIA labels (`aria-label="Mark medicine as relevant"` on upvote buttons, `aria-expanded` on score breakdown drawers).
