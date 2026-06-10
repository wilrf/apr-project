# Repo Professionalization — Design Spec

- Doc Type: Design Spec
- Topic: repo-professionalization
- Topic Slug: repo-professionalization
- Date: 2026-06-10
- Status: Proposed

## Context

`apr-research` is a finished AP Research project (NFL upset prediction via multi-architecture
model disagreement). It is already well-structured: a clean `src/` tree, 256 tests mirroring it,
an honest README, and a working local dashboard. It now lives on a public GitHub profile and needs
to read as a serious, professional engineering project to three audiences at once:

1. **Technical reviewers** who read `src/` — they reward appropriate engineering, clean
   architecture, tests, types, and reproducibility (and penalize over-engineering).
2. **A live demo audience** — they should be visually engaged by the running dashboard.
3. **GitHub browsers** — they judge in ~30 seconds from the README's top third, the badge row,
   the root file list, and commit hygiene; they rarely run anything.

## Goal

Raise the repository's professional surface and engineering rigor without altering the submitted
research. Concretely: add the engineering infrastructure a serious project is expected to have,
clean the repo surface, and rebuild the dashboard into a polished, interactive analytics app served
locally.

## Guardrails

- **Numbers stay as the canonical artifacts; claims may be corrected by evidence.** No model is
  re-tuned and no result regenerated for the sake of nicer numbers. Every metric in `README.md`,
  the paper, the dashboard, and `results/` stays anchored to the existing `results/` artifacts —
  *unless* the work surfaces a genuine computation or reporting error, in which case the correct
  value is flagged, fixed, and propagated everywhere consistently. **Interpretive claims may be
  revised when what I find in the artifacts/code warrants it.** Every claim change must be grounded
  in a specific artifact and flagged for the user — not invented, not weakened or strengthened
  beyond what the evidence supports. (Existing audits — `audit_results.md`, `verified-numbers.md`,
  `docs/2026-03-15-paper-rewrite-audit-results.md` — are the starting point for this reconciliation.)
- **The science is not re-run for new findings.** This relaxation covers correcting and tightening
  claims against existing evidence, not training new models or generating new experimental results.
- **Runs locally, no cloud.** No Vercel, no hosting service. One command brings the dashboard up on
  `localhost`. (Optional GitHub Pages static export is a stretch item, explicitly opt-in.)
- **Tests stay green** at every step; all new code is tested and type-checked.
- **Respect existing repo conventions** (`AGENTS.md` nomenclature, `pipeline.py` canonical column
  lists, the documented invariants in `CLAUDE.md`).

## Out of Scope

- Re-tuning models or regenerating predictions to produce *new* numbers.
- Rewriting the paper's core argument or inventing findings the artifacts don't support.
  (Correcting or tightening claims to match existing evidence is in scope — see B7.)
- Any cloud deployment or CI/CD beyond GitHub Actions test/lint.
- New research features (no new models, no new feature families).

---

## Track A — Repo Surface (GitHub browser)

### A1. Continuous Integration
- `.github/workflows/ci.yml`: on push and PR to `main`, run on Python 3.10/3.11:
  - `ruff check`
  - `black --check`
  - `mypy` (see B3)
  - the fast test suite (`pytest`, excluding the slow LSTM model/trainer tests, matching the
    documented fast-test command).
- Optionally a second job that runs the slow LSTM tests (allowed to be slower / separate).
- **Rationale:** 256 tests currently run in no automated pipeline. A green CI badge is the single
  strongest "real engineer" signal and is genuine rigor, not theater.

### A2. License
- Add `LICENSE` — **MIT** (confirmed). Year + author line.

### A3. README badges + hero
- Badge row at the very top: CI status · license (MIT) · Python 3.10+ · test count.
- Embed a **screenshot or GIF of the rebuilt dashboard** near the top (created in Track C).
- Keep the existing honest, well-written prose. Add a concise "Quickstart" and a "Dashboard"
  section. Do not inflate any claim.

### A4. Root cleanup
The repository root is the first thing a visitor sees; internal/process docs and stray files dilute
it. **Target end-state root:** `README.md`, `LICENSE`, `Makefile`, `pyproject.toml`, `CLAUDE.md`,
`AGENTS.md`, `.gitignore`, plus the `src/ tests/ docs/ dashboard/ data/ results/` directories.

Relocations (moves, not deletions, except confirmed dead code):
- `PROJECT.md` → `docs/architecture-and-analysis.md`. This **fixes a broken link**: `CLAUDE.md`
  already instructs readers to open `docs/architecture-and-analysis.md`, which does not currently
  exist. Verify the move target matches the referenced path and update any references.
- `audit_results.md`, `slide_reconciliation.md`, `BUGHUNT.md` → `docs/development/`.
- Stray working-tree files (`Untitled document.md`, `Wil_Fowler_-_AP_Research_APA7 (1).docx`) →
  handled under the Paper Reconciliation item below. **Never deleted without explicit approval.**
- `.gitignore`: ensure relocated artifacts and the dashboard build output are handled correctly.

### A5. Makefile (one-command ergonomics)
A `Makefile` exposing the project's real commands so a newcomer never has to reconstruct them:
- `make install` — venv + deps.
- `make test` — fast suite. `make test-all` — including slow LSTM tests.
- `make lint` — ruff + black --check. `make format` — black + ruff --fix.
- `make typecheck` — mypy.
- `make features` / `make evaluate` / `make ab` — pipeline entry points.
- `make dashboard` — build frontend + serve on localhost. `make dashboard-dev` — HMR dev mode.
- `make reproduce` — deterministic end-to-end regeneration (see B4), documented as numbers-frozen.

---

## Track B — Code & Engineering Rigor (technical reviewer)

### B1. Delete confirmed dead code
- Remove `src/evaluation/comparison.py` and `tests/evaluation/test_comparison.py`. Confirmed unused
  in production (no `ModelComparison` import anywhere in `src/`/`tools/`; `CLAUDE.md` documents it as
  dead). Update `src/evaluation/__init__.py` and remove the dead-code note from `CLAUDE.md` and the
  Change-Impact Map.
- **Trace step:** grep for any remaining references after removal; confirm the fast suite still
  passes and the count drops by exactly the removed tests.

### B2. Dependency management via `pyproject.toml`
- Consolidate dependencies into PEP 621 `[project.dependencies]` and
  `[project.optional-dependencies]` groups (`dev` = pytest/black/ruff/mypy/pre-commit;
  `lstm` = torch; `dashboard` = fastapi/uvicorn). Keep a thin `requirements.txt` only if needed for
  compatibility, or replace it with documented `pip install -e ".[dev,lstm]"`.
- **Rationale:** modern, single-source dependency declaration reads as more current than a flat
  requirements file and makes optional heavy deps (torch) explicit.

### B3. Static typing
- Add `[tool.mypy]` config (pragmatic: start permissive, no untyped-def errors on data scripts;
  strict enough to catch real bugs). Close gaps so `mypy` passes in CI. Convention already mandates
  `from __future__ import annotations`.

### B4. Reproducibility
- A documented global seed mechanism (single helper, used by trainers/experiment scripts) and
  `make reproduce`. The output must match the frozen artifacts; this is a guard against drift, not a
  re-run that changes numbers. Document the expectation explicitly in the README/Make target.

### B5. Surface existing rigor (no new claims)
- **Model cards** under `docs/model-cards/` for LR, XGB, LSTM: representation, features, training
  data, intended use, evaluation, and limitations — all from existing, frozen results.
- Wire the **already-existing** SHAP analysis (`src/evaluation/shap_analysis.py`) and the
  significance/ablation artifacts (`results/ab_experiment/significance_and_analysis_2026-03-10.md`)
  into the dashboard and docs so the rigor that exists becomes visible.

### B6. Pre-commit
- `.pre-commit-config.yaml` with black + ruff (and trailing-whitespace/EOF hooks). Mentioned in
  CONTRIBUTING/README so the hygiene is visible.

### B7. Claim–evidence reconciliation
- Audit the interpretive claims in `README.md` and the canonical paper against the actual artifacts
  in `results/`, building on the existing audits (`audit_results.md`, `verified-numbers.md`,
  `docs/2026-03-15-paper-rewrite-audit-results.md`).
- Where a claim overstates, understates, or misreads the evidence, revise it to match — each change
  grounded in a named artifact and surfaced to the user for sign-off. Where a reported number is
  genuinely wrong, correct it and propagate consistently (README, paper, dashboard, `results/`
  summaries).
- This is correction-by-evidence, not re-running experiments. Net effect: a repo where every claim
  is defensible against the data on disk.

---

## Track C — Dashboard Rebuild (demo) — largest component

Replace `tools/dashboard/` with a new top-level `dashboard/`. Stack: **FastAPI** backend +
**Vite + React + TypeScript** frontend with **Recharts**. Built simply and elegantly — no state
library, no CSS framework bloat, no heavy viz stack.

### C1. Directory layout
```
dashboard/
  README.md                  # run/build instructions, Node + Python prerequisites
  backend/
    app.py                   # FastAPI app; serves /api/* and the built SPA on one port
    schemas.py               # Pydantic response models (typed API contract)
    data_access.py           # read results/, compute metrics (reuse src.evaluation.metrics)
  frontend/
    index.html
    package.json
    tsconfig.json
    vite.config.ts
    src/
      main.tsx
      App.tsx
      api/client.ts          # typed fetch client
      api/types.ts           # TS types mirroring Pydantic schemas (hand-written or generated)
      theme/                 # design tokens + global styles (distinctive, not generic)
      components/
        charts/              # RocCurve, CalibrationPlot, DisagreementBars, SeasonTrend
        ui/                  # StatCard, DataTable, Tabs, etc.
      views/                 # Overview, Predictions, Disagreement, Features, Experiments
```

### C2. Backend (FastAPI)
- Endpoints (typed Pydantic responses), reading the same frozen artifacts the current server reads
  (`results/test/predictions.csv`, `results/ab_experiment/*`), reusing `src/evaluation/metrics.py`
  (`safe_roc_auc_score`, `safe_log_loss`) for parity with the rest of the codebase:
  - `GET /api/summary` — per-dataset model metrics (AUC/Brier/LogLoss), games, upset rate.
  - `GET /api/predictions/{dataset}` — per-game rows for the test set / ablation sets.
  - `GET /api/disagreement/{dataset}` — category breakdown + actual upset rate per category.
  - `GET /api/features` — LR coefficients (with/without spread).
  - `GET /api/seasons/{dataset}` — per-season AUCs.
  - `GET /api/curves/{dataset}` — ROC and calibration curve points (computed from predictions).
- Auto OpenAPI docs at `/docs` (a real credibility/demo artifact).
- Serves the built frontend (`frontend/dist`) as static at `/`. Single origin, single port.
- **No metric is recomputed differently from the canonical pipeline** — dashboard values must match
  `results/` and the README, keeping every number anchored to the canonical artifacts.

### C3. Frontend (Vite + React + TS + Recharts)
- Views mirror the current dashboard's information architecture so no analysis is lost: Overview,
  Game Results (Predictions), Disagreement, Feature Weights, Experiments (ablation + CV-vs-test).
- Charts (Recharts, heavily themed to avoid the generic look): ROC curves (3 models), calibration
  reliability plot, disagreement category bars, per-season AUC trend.
- A typed API client; end-to-end types from the backend contract (hand-written `types.ts`, with an
  optional `openapi-typescript` codegen step as a stretch item).
- **Aesthetic:** the user chose a rich, modern, dark analytics look. To avoid the generic
  AI-dashboard cliché (which we explicitly flagged), the build will use the `frontend-design` skill
  and a small bespoke design system that carries the project's existing editorial/typographic
  identity into a considered dark theme — distinctive, restrained accent color, real typographic
  hierarchy, not a default component-library skin.

### C4. Run story
- `make dashboard`: `npm --prefix dashboard/frontend ci && npm --prefix dashboard/frontend run build`,
  then run the FastAPI backend serving the built assets on `localhost` (default port 8050).
- `make dashboard-dev`: Vite dev server (HMR) + backend with CORS for local development.
- Prerequisites documented: Node (LTS) + Python. `frontend/dist` is gitignored by default.
- **Stretch (opt-in, not default):** a static export + GitHub Pages workflow to give GitHub browsers
  a live link. Free, not Vercel, compatible with the "no cloud hosting service" guardrail. Will not
  be enabled without explicit approval.

### C5. Retire the old dashboard
- Remove `tools/dashboard/` after the new one reaches parity. Update README reproduce commands and
  any references. Keep `tools/presentation/` and `tools/numerical_audit_compute.py` (still useful).

---

## Open Reconciliation Item — Paper Artifacts

Three paper artifacts exist and must be reconciled **non-destructively**:
- `docs/paper.md` — older markdown draft (~51 KB).
- `Untitled document.md` (untracked) — final prose, titled "The Anatomy of NFL Upsets."
- `Wil_Fowler_-_AP_Research_APA7 (1).docx` (untracked) — the APA7 submission.

Plan: diff/compare the three, check which is most complete and consistent with the frozen numbers,
then **recommend a canonical choice for explicit approval** before any rename/move/delete. Likely
end-state: a clean `docs/paper/` holding the canonical readable paper (markdown) plus the official
`.docx` submission, with older drafts archived under `docs/development/` or removed only on approval.

---

## Build Sequence (each step is a commit; tests green throughout)

1. **Surface + dead code:** root cleanup/relocations, fix the architecture-doc link, remove
   `comparison.py` (+ test), add `LICENSE`. Low-risk, instantly tidier.
2. **Tooling + CI:** `pyproject` dependency consolidation, `mypy` config + fixes, `pre-commit`,
   `Makefile`, `.github/workflows/ci.yml`, README badges. Watch CI go green.
3. **Dashboard backend:** FastAPI app + Pydantic schemas + data access over frozen results, with
   tests asserting parity against `results/` and README numbers.
4. **Dashboard frontend:** Vite/React/TS scaffold, design system, views, Recharts charts; wire to
   the API; reach parity with the old dashboard, then add the new visualizations.
5. **Polish:** README hero screenshot/GIF, model cards, paper reconciliation (after approval),
   retire `tools/dashboard/`, final pass.

## Testing & Verification

- Python: every new backend module has tests; `make test` and `make lint`/`make typecheck` pass.
  Backend tests assert dashboard metrics equal the canonical `results/` values (numbers-frozen
  guard).
- Frontend: type-checks (`tsc --noEmit`) and builds clean; a smoke check that the built SPA loads
  and renders each view against a running backend.
- CI green on the fast suite before any "done" claim (per the verification-before-completion
  discipline).
- After each track, run the documented fast test command and report real output, not assertions.

## Risks & Mitigations

- **Number drift in the dashboard.** Mitigation: backend reuses `src/evaluation/metrics.py` and a
  parity test pins values to `results/`.
- **Two toolchains (Node + Python) raising the "over-engineered" read.** Mitigation: keep the
  frontend minimal and elegant; document a single `make dashboard`; backend stays small.
- **Removing files that matter.** Mitigation: relocations over deletions; the only deletion is
  confirmed dead code; paper artifacts touched only after explicit approval.
- **CI flakiness from slow LSTM/torch tests.** Mitigation: fast suite is the gating job; slow LSTM
  tests run in a separate, non-blocking job.
