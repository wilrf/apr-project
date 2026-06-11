# Track C — Dashboard Rebuild: Implementation Plan

- Doc Type: Implementation Plan
- Topic: repo-professionalization (Track C — dashboard)
- Topic Slug: repo-professionalization-dashboard
- Date: 2026-06-10
- Status: Proposed
- Authoritative spec: [`docs/plans/2026-06-10-repo-professionalization-design.md`](2026-06-10-repo-professionalization-design.md) — Track C (C1–C5) + Guardrails
- Branch: `feature/repo-professionalization` (no new branches)

---

## 0. Purpose & Scope

Replace the legacy `tools/dashboard/` (a single static `index.html` + a `http.server` shim in
`serve.py`) with a top-level `dashboard/` built as a **FastAPI** backend + **Vite + React +
TypeScript + Recharts** frontend, served locally on **one port (default 8050)**, no cloud. The
rebuild must reach **full information-architecture parity** with the old dashboard (lose no view,
no analysis), then add new charts (ROC curves, calibration reliability, disagreement bars, per-season
AUC trend). Every number the dashboard shows MUST equal the canonical `results/` artifacts and the
README; the backend reuses `src.evaluation.metrics` and never recomputes a metric differently.

**This document is a plan only. It does not build anything and does not touch git.**

---

## 1. Ground Truth Captured (read before building)

### 1.1 `src.evaluation.metrics` — exact callables available (MUST reuse for parity)

From [`src/evaluation/metrics.py`](../../src/evaluation/metrics.py):

| Callable | Signature | Use in backend |
|----------|-----------|----------------|
| `clip_probabilities(y_pred, eps=1e-7) -> np.ndarray` | clip probs off 0/1 | internal, before log-loss |
| `safe_roc_auc_score(y_true, y_pred) -> float` | NaN when one class | **AUC** for summary/seasons/curves |
| `safe_log_loss(y_true, y_pred) -> float` | clipped, labels `[0,1]` | **Log Loss** for summary |
| `safe_probability_correlation(preds_a, preds_b) -> float` | finite Pearson, 0 on degenerate | inter-model prob correlations (Overview/Experiments) |
| `safe_quantile_buckets(values, q=5) -> pd.Series` | quantile buckets, NA-safe | optional (top-K / decile views) |
| `calculate_calibration_metrics(y_true, y_pred, n_bins=10) -> dict` | returns `calibration_error`, `prob_true`, `prob_pred` | **calibration curve** endpoint (`/api/curves`) |
| `calculate_betting_metrics(y_true, y_pred, odds, threshold=0.5) -> dict` | ROI/win-rate/profit | not used (no odds in current views) |
| `calculate_baseline_brier(upset_rate) -> float` | `upset_rate*(1-upset_rate)` | **baseline Brier** for summary |

> Note: Brier itself (`sklearn.metrics.brier_score_loss`) is NOT in `metrics.py`. The old
> `serve.py` imports it directly from sklearn. The new backend does the same (`brier_score_loss`),
> matching the legacy computation exactly — do not invent a local Brier.

### 1.2 `results/` artifacts the backend reads (frozen — never regenerate to "improve" numbers)

| Path | Rows/Shape | Backend use |
|------|-----------|-------------|
| `results/test/predictions.csv` | 558 rows, 17 cols (incl. `category`, `*_pred`, `*_correct`) | `dataset=test`: summary, predictions, disagreement, seasons, curves |
| `results/ab_experiment/predictions_with_spread.csv` | 1162 rows, 13 cols (NO `category`) | `dataset=cv_with_spread`: summary, predictions, disagreement (derived), seasons, curves |
| `results/ab_experiment/predictions_without_spread.csv` | 1162 rows, 13 cols (NO `category`) | `dataset=cv_without_spread`: same as above |
| `results/ab_experiment/lr_coefs_with_spread.json` | 46 keys (feature → coef) | `/api/features` (with spread) |
| `results/ab_experiment/lr_coefs_without_spread.json` | 42 keys | `/api/features` (without spread) |
| `results/ab_experiment/significance_and_analysis_2026-03-10.md` | markdown | surfaced as text/links on Experiments view (Track B5 tie-in) |
| `results/test/report.md` | markdown | reference only (agreement %, prob-correlation matrix, top-K) |
| `results/audit_computed.json` | keys: metadata, dataset, cv, lstm_buckets, spread_ablation, test, taxonomy, calibration, features, lr_coefficients, source_refs | cross-check / parity reference (not a primary read; numbers must agree) |

**Test-CSV column order** (load-bearing): `game_id, season, week, underdog, favorite,
spread_magnitude, y_true, lr_prob, xgb_prob, lstm_prob, lr_pred, xgb_pred, lstm_pred, lr_correct,
xgb_correct, lstm_correct, category`. The ab CSVs are the same minus `lr_correct/xgb_correct/
lstm_correct/category`.

### 1.3 README / model-card canonical numbers the dashboard MUST match exactly

From [`README.md`](../../README.md) "Results At A Glance" and "Spread Ablation":

| Model | CV AUC | Test AUC | Test Brier | CV AUC no-spread | Delta |
|-------|--------|----------|------------|------------------|-------|
| LR | 0.6497 | 0.5622 | 0.2026 | 0.5707 | −0.0790 |
| XGB | 0.6377 | 0.5755 | 0.2013 | 0.5662 | −0.0715 |
| LSTM | 0.6372 | 0.5263 | 0.2089 | 0.5682 | −0.0690 |

- Train: 3,495 labeled, upset rate **30.36%**. Test: 558 labeled, upset rate **28.49%** (report.md
  rounds to 28.5%). Baseline Brier (test) = **0.2038**.
- Test prob correlations: LR-XGB `0.878`, LR-LSTM `0.373`, XGB-LSTM `0.309`.
- Agreement: LR-XGB 89.2%, LR-LSTM 70.3%, XGB-LSTM 68.1%, all-three 63.8%.
- XGB top-10 held-out: 6 real upsets (60% hit, ~2.1x lift over 28.49%).

> **Discrepancy flagged for the user (do NOT silently resolve):** repo memory (`MEMORY.md`) lists
> LSTM CV AUC = 0.6407 and Test AUC = 0.5202; README/model-card list **0.6372 / 0.5263**. The
> README/`results/` artifacts are canonical per the guardrail, so the dashboard and parity tests
> anchor to **0.6372 / 0.5263**. The memory figure is stale — surface this in the report, do not
> "fix" the README without the user's sign-off.

### 1.4 Old dashboard — IA, endpoints, and every analysis captured (parity target)

Legacy server: [`tools/dashboard/serve.py`](../../tools/dashboard/serve.py) (a `SimpleHTTPRequestHandler`
exposing **one** JSON endpoint `GET /api/data` returning the whole payload) +
[`tools/dashboard/index.html`](../../tools/dashboard/index.html) (vanilla-JS SPA, serif "newspaper"
theme, 5 tabs).

**Single legacy endpoint** `GET /api/data` → `build_payload()` returns:
`{ predictions, summaries, categories, seasons, coefs }` keyed by dataset
(`test`, `cv_with_spread`, `cv_without_spread`) — built fresh on each request from the CSVs/JSONs
in §1.2 with `safe_roc_auc_score`, `safe_log_loss`, `brier_score_loss`.

**Five tabs (the IA the rebuild must preserve):**

1. **Overview** (`renderOverview`) — reading guide; per-dataset (test + cv_with_spread) stat strip
   (Games / Upset Rate / AUC Leader), a model table (AUC, Brier, Log Loss, "reading"), and a
   per-season breakdown table (Season, Games, Upset Rate, LR/XGB/LSTM AUC).
2. **Game Results / Predictions** (`renderPredictions`) — **test set only**: full sortable, text-
   filterable table over columns `season, week, underdog, favorite, spread_magnitude, y_true,
   lr_prob, xgb_prob, lstm_prob, category, lr_correct, xgb_correct, lstm_correct`. Row count
   "X of 558". `y_true`→"Upset"/"Favorite held"; `category`→friendly label; `*_correct`→"correct/
   missed".
3. **Disagreement** (`renderDisagreement`) — per dataset (test + cv_with_spread): category table
   (Category, Games, Share %, Actual Upset Rate, Interpretation prose). 8 categories: `all_correct,
   all_wrong, only_lr, only_xgb, only_lstm, lr_xgb, lr_lstm, xgb_lstm`.
4. **Feature Weights** (`renderFeatures`) — LR coefficients ranked by |coef|, with_spread (46) and
   without_spread (42): Rank, Feature, Coefficient, Direction ("raises/lowers upset odds").
5. **Experiments** (`renderCV`) — four sections:
   - **Spread Ablation**: per-model CV AUC With / Without / Delta.
   - **Cross-Validation vs Test Set**: per-model CV AUC / Test AUC / Gap.
   - **Disagreement Shift After Removing Spread**: per-category share With / Without / Shift.
   - **Per-Season Cross-Validation**: Season, Games, Upset Rate, LR/XGB/LSTM AUC (cv_with_spread).

**Parity nuances discovered (must be handled, not copied blindly):**
- The ab CSVs (`predictions_with/without_spread.csv`) have **no `category` column**, so the legacy
  `_compute_categories` returns `[]` for the CV datasets — meaning the old "Disagreement (CV)" and
  "Disagreement Shift After Removing Spread" sections render **empty** in practice. The rebuild
  must **derive categories on the fly** for the ab datasets using the base-rate threshold (see
  invariants §1.5), which fixes a latent gap while staying numbers-frozen (categorization, not a
  new metric). Flag this as an improvement, not a number change.
- Legacy `_compute_metrics` reads `category` only if present; the rebuild centralizes category
  derivation in `data_access.py`.

### 1.5 CLAUDE.md invariants the backend MUST honor

- **Threshold is the base upset rate (~0.30), NEVER 0.5** for any binary categorization. When the
  backend derives `*_pred`/`category` for the ab datasets, it must threshold each model's prob at
  the dataset's base upset rate (`y_true.mean()`), matching `DisagreementAnalyzer` semantics — NOT
  0.5, and NOT the persisted `lr_pred/xgb_pred/lstm_pred` columns (those hardcode 0.5 and must not
  be used for categorization).
- For the **test** dataset, the frozen `category`/`*_correct` columns already exist and are
  authoritative — use them directly; do not recompute (avoids drift from the canonical pipeline).
- Spread threshold (`spread >= 3`) already applied upstream; the CSVs are pre-filtered. No
  re-labeling in the backend.

---

## 2. Conventions & Guardrails (apply to every task)

- **venv:** use `.venv/bin/python3` and `.venv/bin/pip` for all Python commands.
- **Branch:** stay on `feature/repo-professionalization`. Do not create branches.
- **Gate stays green:** backend Python must pass `ruff check`, `black --check`, `mypy`, and the
  fast `pytest`. Frontend TS/JS is **out of the Python gate's scope** (see §2.1).
- **Numbers frozen:** dashboard values MUST equal `results/` + README. Reuse `src.evaluation.metrics`.
  Parity tests (Task 2) enforce this.
- **Git hygiene:** scoped commits **per task**, never `git add -A`. Never commit `node_modules/`,
  `dist/`, the untracked root paper files (`Untitled document.md`, the `.docx`), or anything under
  `.claude/`. (This plan does not commit; these rules apply when the work is executed.)
- **Typing:** `from __future__ import annotations` in every backend module.

### 2.1 mypy / ruff / black scope decision (RECOMMENDATION — simplest correct option)

Current config: `[tool.mypy] files = ["src"]`; CI runs `ruff check src/ tests/` and
`black --check src/ tests/`; `[tool.ruff] exclude = ["tools/"]`. So **nothing checks
`dashboard/backend/` today**.

**Recommendation (do this):** bring the backend under the existing single gate rather than adding a
second tool invocation:
- mypy: change `files = ["src"]` → `files = ["src", "dashboard/backend"]` (one line; one `make
  typecheck`, one CI step covers everything).
- ruff/black/CI: change the three CI/Makefile invocations from `src/ tests/` →
  `src/ tests/ dashboard/backend/` (and the same in `make lint` / `make format`).
- Tests live in `tests/dashboard/` which is already inside the `tests/` glob — no change needed
  for the parity tests to be linted/typed and to run under the fast gate.
- Do **not** put backend code under `tools/` (ruff-excluded) — that would silently drop it from
  linting. Top-level `dashboard/backend/` + extended scope is the clean choice.

> Rejected alternative: a separate `mypy dashboard/backend` step / second ruff call. More moving
> parts, two failure surfaces, no benefit at this size. One gate, extended scope, is simpler and
> correct.

### 2.2 Missing dependency to add (blocker for Task 2)

FastAPI's `TestClient` requires **`httpx`**, which is NOT currently in `[project.optional-dependencies]
.dev`. Add `httpx` (pinned, e.g. `httpx==0.28.1` — confirm against installed version) to the `dev`
group so `from fastapi.testclient import TestClient` works in CI. The `dashboard` extra
(`fastapi>=0.110.0`, `uvicorn>=0.29.0`) already exists. CI already installs `.[dev,lstm]`; **add
`dashboard`** to the CI install (`.[dev,lstm,dashboard]`) so the backend imports and parity tests
run. `make install` should install `.[dev,lstm,dashboard]` too.

---

## 3. Prereqs Probe (do this first when execution starts)

- [ ] Confirm Node + npm: `node --version` (expect LTS ≥ 18; **verified v22.22.0 present**) and
      `npm --version` (**verified 10.9.4 present**). Record in `dashboard/README.md`.
- [ ] **Fallback note:** if a future machine lacks Node, `make dashboard` cannot build the SPA. The
      backend still runs standalone (`uvicorn`) and `/docs` (OpenAPI) + `/api/*` work without a
      build; document that the SPA at `/` requires a one-time `npm ci && npm run build`. Backend
      parity tests do NOT depend on Node (pure Python).
- [ ] Confirm venv interpreter resolves: `.venv/bin/python3 -c "import fastapi, uvicorn"` (after the
      Task-1 dep install) — fail early if the `dashboard` extra is not installed.

---

## 4. Tasks

Each task: **Files → Steps (`- [ ]`) → Verification (a real, failable command/check)**. Tasks are
ordered; backend before frontend; parity tests before frontend so the contract is pinned.

---

### Task 1 — Dependency wiring + gate-scope extension

Make the backend importable and put it under the Python gate before writing code.

**Files:** `pyproject.toml`, `Makefile`, `.github/workflows/ci.yml`

**Steps:**
- [ ] Add `httpx==<installed>` to `[project.optional-dependencies].dev` (for `TestClient`).
- [ ] In `[tool.mypy]`, set `files = ["src", "dashboard/backend"]`.
- [ ] In `Makefile`: `lint`/`format` targets → append `dashboard/backend/` to the ruff+black paths;
      `install` → `pip install -e ".[dev,lstm,dashboard]"`.
- [ ] In `.github/workflows/ci.yml`: install step → `.[dev,lstm,dashboard]`; ruff and black steps →
      `src/ tests/ dashboard/backend/`. (mypy step is unchanged — it reads `files` from config.)
- [ ] Run `.venv/bin/pip install -e ".[dev,lstm,dashboard]"` to materialize the deps.

**Verification:**
```bash
.venv/bin/python3 -c "import fastapi, uvicorn, httpx; print('deps ok')"
.venv/bin/python3 -m mypy --version && grep -q 'dashboard/backend' pyproject.toml && echo "mypy scope ok"
```
Both must succeed (deps import; `dashboard/backend` present in mypy `files`).

---

### Task 2 — Backend scaffold (FastAPI + Pydantic + data access)

**Files:**
`dashboard/backend/__init__.py`, `dashboard/backend/app.py`, `dashboard/backend/schemas.py`,
`dashboard/backend/data_access.py`

**Steps:**
- [ ] `data_access.py`: module-level `ROOT`/`RESULTS` path constants (resolve to repo root, not
      cwd). Loaders: `load_predictions(dataset)` for the three datasets (`test`,
      `cv_with_spread`, `cv_without_spread`) returning a typed `pd.DataFrame`; `load_lr_coefs(variant)`
      for `with_spread`/`without_spread`. Raise a clear error if a frozen artifact is missing
      (file-exists guard, matching repo convention).
- [ ] `data_access.py`: `compute_summary(df)` → AUC (`safe_roc_auc_score`), Brier
      (`sklearn.brier_score_loss`, matching legacy), Log Loss (`safe_log_loss`) per model; plus
      `n_games`, `upset_rate` (`y_true.mean()`), `baseline_brier` (`calculate_baseline_brier`).
- [ ] `data_access.py`: `derive_categories(df)` — for **test**, read frozen `category`/`*_correct`;
      for **ab** datasets, derive `*_pred = prob >= base_rate` (base rate = `y_true.mean()`,
      **never 0.5**), `*_correct = (pred == y_true)`, and the 8-way `category`. Centralize the
      taxonomy here. `compute_disagreement(df)` → per-category `{category, n, pct, upset_rate}`.
- [ ] `data_access.py`: `compute_seasons(df)` → per-season `{season, n, upset_rate, lr_auc, xgb_auc,
      lstm_auc}` (mirror legacy `_compute_seasons`). `compute_roc_points(df)` and
      `compute_calibration_points(df)` (use `calculate_calibration_metrics` for reliability;
      sklearn `roc_curve` for ROC) per model → `/api/curves`.
- [ ] `schemas.py`: Pydantic v2 response models — `ModelMetrics`, `DatasetSummary`,
      `SummaryResponse` (map of dataset → DatasetSummary), `PredictionRow`, `PredictionsResponse`,
      `DisagreementCategory`, `DisagreementResponse`, `FeatureCoef`, `FeaturesResponse`,
      `SeasonRow`, `SeasonsResponse`, `RocSeries`/`CalibrationSeries`/`CurvesResponse`. Field types
      explicit (`float`, `int`, `str`; AUC may be `float | None` for NaN seasons).
- [ ] `app.py`: `FastAPI(title="APR Research Dashboard", ...)`. Endpoints (all return the typed
      models, `response_model=`):
      - `GET /api/summary` → all datasets' metrics + games + upset rate.
      - `GET /api/predictions/{dataset}` → per-game rows (dataset ∈ test/cv_with_spread/
        cv_without_spread; 404 on unknown).
      - `GET /api/disagreement/{dataset}` → category breakdown + actual upset rate per category.
      - `GET /api/features` → LR coefs with/without spread (ranked by |coef|, sign).
      - `GET /api/seasons/{dataset}` → per-season AUCs.
      - `GET /api/curves/{dataset}` → ROC + calibration points (3 models).
- [ ] `app.py`: mount built SPA — if `frontend/dist` exists, `StaticFiles(html=True)` at `/`;
      otherwise a small JSON stub at `/` telling the user to run `make dashboard`. Single origin,
      single port. OpenAPI auto-served at `/docs`.
- [ ] `app.py`: `__main__` / a `main()` that runs `uvicorn` on `127.0.0.1:8050` (port overridable
      via env `DASHBOARD_PORT`).

**Verification:**
```bash
.venv/bin/python3 -m ruff check dashboard/backend/ && .venv/bin/black --check dashboard/backend/
.venv/bin/python3 -m mypy   # files now include dashboard/backend
.venv/bin/python3 - <<'PY'
from fastapi.testclient import TestClient
from dashboard.backend.app import app
c = TestClient(app)
for url in ["/api/summary","/api/predictions/test","/api/disagreement/test",
            "/api/features","/api/seasons/test","/api/curves/test","/openapi.json"]:
    assert c.get(url).status_code == 200, url
print("all endpoints 200")
PY
```
All three must pass (lint clean, types clean, every endpoint 200 + OpenAPI present).

---

### Task 3 — Backend parity tests (numbers-frozen guard)

**Files:** `tests/dashboard/__init__.py`, `tests/dashboard/test_endpoints.py`,
`tests/dashboard/test_parity.py` (mirrors the `tests/<area>/` convention; runs under the fast gate).

**Steps:**
- [ ] `test_endpoints.py`: a `TestClient(app)` fixture; assert each endpoint returns 200 and its
      Pydantic-validated shape (FastAPI validates on serialize; tests assert key presence + types,
      and 404 on a bad `{dataset}`).
- [ ] `test_parity.py` — **the numbers-frozen guard**. Assert backend output EQUALS canonical
      values (use `pytest.approx(..., abs=1e-3)` to match README rounding; tighter where the CSV
      supports it):
      - `/api/summary` test: LR/XGB/LSTM Test AUC == 0.5622 / 0.5755 / 0.5263; Test Brier ==
        0.2026 / 0.2013 / 0.2089; `n_games == 558`; `upset_rate ≈ 0.2849`; `baseline_brier ≈ 0.2038`.
      - `/api/summary` cv_with_spread: CV AUC == 0.6497 / 0.6377 / 0.6372; `n_games == 1162`.
      - `/api/summary` cv_without_spread: CV AUC == 0.5707 / 0.5662 / 0.5682 (and Δ vs with-spread
        == −0.0790 / −0.0715 / −0.0690).
      - `/api/disagreement/test`: category counts match the frozen `category` column tallies
        (`all_correct=147, all_wrong=209, lr_lstm=12, lr_xgb=61, only_lr=9, only_lstm=81,
        only_xgb=24, xgb_lstm=15`; shares sum to 100%).
      - `/api/features`: with-spread has 46 coefs, without-spread 42; top-|coef| ordering stable;
        a spot-checked value (e.g. `favorite_turnover_margin_roll3 ≈ -0.0694`) matches the JSON.
      - `/api/seasons/test`: season list + per-season AUC equal a direct recompute from the CSV via
        `safe_roc_auc_score` (guards against a divergent grouping).
      - **Cross-check** a handful of `/api/summary` values against `results/audit_computed.json`
        where overlapping (defense in depth; flag any mismatch rather than papering over it).
- [ ] Mark none of these `slow` — they must run in the default fast suite.

**Verification:**
```bash
.venv/bin/python3 -m pytest tests/dashboard/ -q
# Then confirm they run inside the gate exactly as CI invokes it:
.venv/bin/python3 -m pytest tests/ \
  --ignore=tests/models/test_lstm_model.py \
  --ignore=tests/models/test_lstm_trainer.py -q
```
Both green; the full fast-suite count increases by the number of dashboard tests; no regressions.

---

### Task 4 — Frontend scaffold (Vite + React + TS)

**Files:**
`dashboard/frontend/package.json`, `dashboard/frontend/tsconfig.json`,
`dashboard/frontend/tsconfig.node.json`, `dashboard/frontend/vite.config.ts`,
`dashboard/frontend/index.html`, `dashboard/frontend/src/main.tsx`,
`dashboard/frontend/src/App.tsx`, `dashboard/frontend/src/api/client.ts`,
`dashboard/frontend/src/api/types.ts`, `dashboard/frontend/src/theme/` (tokens + global CSS),
`dashboard/frontend/src/components/{charts,ui}/`, `dashboard/frontend/src/views/`,
`dashboard/frontend/.gitignore` (belt-and-suspenders; root `.gitignore` already ignores
`dashboard/frontend/node_modules/` and `dashboard/frontend/dist/`).

**Steps:**
- [ ] `package.json`: deps `react`, `react-dom`, `recharts`; devDeps `vite`,
      `@vitejs/plugin-react`, `typescript`, `@types/react`, `@types/react-dom`. Scripts: `dev`,
      `build` (`tsc -b && vite build`), `preview`, `typecheck` (`tsc --noEmit`). **Pin versions**
      and commit a `package-lock.json` so `npm ci` (used by `make dashboard`) is reproducible.
- [ ] `vite.config.ts`: `base: "/"`, `build.outDir: "dist"`. Dev `server.proxy` mapping `/api` →
      `http://127.0.0.1:8050` so `make dashboard-dev` works against the backend with no CORS in the
      browser (backend CORS still enabled for direct calls — see Task 7).
- [ ] `tsconfig.json`: `strict: true`, `noEmit`, bundler module resolution; `tsconfig.node.json`
      for the Vite config.
- [ ] `api/types.ts`: hand-written TS interfaces **mirroring the Pydantic schemas** (single source
      of contract truth: `schemas.py`). One interface per response model from Task 2. (Stretch,
      not in this task: `openapi-typescript` codegen from `/openapi.json`.)
- [ ] `api/client.ts`: a tiny typed fetch wrapper — `getSummary()`, `getPredictions(dataset)`,
      `getDisagreement(dataset)`, `getFeatures()`, `getSeasons(dataset)`, `getCurves(dataset)`,
      each typed by `types.ts`, hitting same-origin `/api/*`.
- [ ] `App.tsx` + `main.tsx`: app shell with the 5-tab nav (Overview / Game Results / Disagreement /
      Feature Weights / Experiments) wired to the views (Task 5). Loading + error states (parity
      with the legacy "Reading results…" / error panel).

**Verification:**
```bash
npm --prefix dashboard/frontend ci
npm --prefix dashboard/frontend run typecheck   # tsc --noEmit, clean
npm --prefix dashboard/frontend run build        # vite build succeeds, dist/ created
test -f dashboard/frontend/dist/index.html && echo "build artifact present"
```
All clean; `dist/index.html` exists.

---

### Task 5 — Views (parity-first, then additive)

Mirror the OLD IA so no analysis is lost, then add the new charts (Task 6). Each view maps to
endpoint(s).

**Files:** `dashboard/frontend/src/views/{Overview,Predictions,Disagreement,Features,Experiments}.tsx`,
plus `components/ui/{StatCard,DataTable,Tabs,SectionNote}.tsx`.

**Steps & endpoint mapping:**
- [ ] **Overview** ← `/api/summary` (+ `/api/seasons/{test,cv_with_spread}`, `/api/curves/test` for
      the new ROC). Reading guide; per-dataset stat strip (Games / Upset Rate / AUC Leader); model
      table (AUC, Brier, Log Loss); season breakdown table. **+New:** ROC curves (Task 6).
- [ ] **Game Results / Predictions** ← `/api/predictions/test`. Sortable + text-filterable
      `DataTable` over the 13 display columns; "X of 558" count; `y_true`→Upset/Favorite-held,
      `category`→friendly label, `*_correct`→correct/missed. (Parity: test set only, as legacy.)
- [ ] **Disagreement** ← `/api/disagreement/{test,cv_with_spread}`. Category table (Category, Games,
      Share %, Actual Upset Rate, Interpretation prose — port `CAT_FRIENDLY_NAMES`/`CAT_EXPLAIN`
      strings). **+New:** disagreement category bars (Task 6). **Improvement:** CV disagreement now
      populated (categories derived in backend), fixing the legacy empty section.
- [ ] **Feature Weights** ← `/api/features`. Two ranked tables (with-spread 46, without-spread 42):
      Rank, Feature, Coefficient, Direction. (Optional: a themed horizontal bar per coefficient,
      sign-colored — keep it restrained.)
- [ ] **Experiments** ← `/api/summary` (all three datasets) + `/api/disagreement/{cv_with_spread,
      cv_without_spread}` + `/api/seasons/cv_with_spread` (+ `/api/curves/cv_*` optional). Four
      sections: Spread Ablation (With/Without/Δ), CV-vs-Test (CV/Test/Gap), Disagreement Shift
      after removing spread (now non-empty), Per-Season CV. **+New:** per-season AUC trend (Task 6).
      Link out to `results/ab_experiment/significance_and_analysis_2026-03-10.md` (Track B5 tie-in).

**Verification:** covered by Task 8 (smoke render against a running backend) and Task 4 typecheck.
Interim check: `npm --prefix dashboard/frontend run typecheck` stays clean as views are added.

---

### Task 6 — Charts (Recharts, distinctively themed)

Use the **`frontend-design` skill** during the build to produce a distinctive dark analytics theme
(restrained single accent, real typographic hierarchy, editorial identity carried from the existing
serif masthead into a considered dark UI) — explicitly avoid the generic AI-dashboard look.

**Files:** `dashboard/frontend/src/components/charts/{RocCurve,CalibrationPlot,DisagreementBars,
SeasonTrend}.tsx`, `dashboard/frontend/src/theme/{tokens.ts,global.css,recharts-theme.ts}`.

**Steps:**
- [ ] Invoke the `frontend-design` skill to generate the design tokens (dark palette, accent,
      type scale, spacing) and chart styling; codify in `theme/tokens.ts` + a shared Recharts theme
      wrapper so all charts read consistent.
- [ ] **RocCurve** ← `/api/curves/{dataset}`: 3 model curves + diagonal reference; AUC in legend.
- [ ] **CalibrationPlot** (reliability) ← `/api/curves/{dataset}`: predicted vs observed per bin +
      identity line (uses backend `calculate_calibration_metrics` output).
- [ ] **DisagreementBars** ← `/api/disagreement/{dataset}`: category share bars with actual-upset-
      rate overlay/tooltip.
- [ ] **SeasonTrend** ← `/api/seasons/{dataset}`: per-season AUC line trend for the 3 models.
- [ ] Ensure charts degrade gracefully on NaN AUC seasons (gaps, not zeros).

**Verification:**
```bash
npm --prefix dashboard/frontend run typecheck
npm --prefix dashboard/frontend run build
```
Clean typecheck + build with the charts wired into the views.

---

### Task 7 — Run wiring (`make dashboard` / `make dashboard-dev`)

**Files:** `Makefile` (replace the two placeholder echo targets), `dashboard/README.md`,
`dashboard/backend/app.py` (CORS for dev).

**Steps:**
- [ ] Replace `make dashboard` placeholder with: `npm --prefix dashboard/frontend ci && npm
      --prefix dashboard/frontend run build`, then start the backend serving `frontend/dist` on
      `127.0.0.1:8050` (e.g. `.venv/bin/python3 -m dashboard.backend.app`). One command, one port.
- [ ] Replace `make dashboard-dev` placeholder with: start the backend (uvicorn `--reload`, CORS
      enabled for `http://localhost:5173`) **and** the Vite dev server (`npm --prefix
      dashboard/frontend run dev`, HMR, `/api` proxied to 8050). Document running them in two panes
      or via a `&`-backgrounded helper; keep it simple.
- [ ] In `app.py`, add `CORSMiddleware` allowing the Vite dev origin **only when** a dev flag/env is
      set (production single-port build needs no CORS — same origin).
- [ ] `dashboard/README.md`: prerequisites (Node LTS — verified v22.22.0; Python + venv),
      `make dashboard` vs `make dashboard-dev`, the port (8050), the `/docs` OpenAPI URL, and the
      note that `frontend/dist` + `node_modules` are gitignored (already are).

**Verification:**
```bash
# Build + serve, then probe in the background and tear down:
make dashboard &  MAKE_PID=$!
# poll until up, then:
curl -fsS http://127.0.0.1:8050/api/summary > /dev/null && echo "api up"
curl -fsS http://127.0.0.1:8050/ | grep -qi "<div id=\"root\"" && echo "spa served"
curl -fsS http://127.0.0.1:8050/docs > /dev/null && echo "openapi docs up"
kill $MAKE_PID
```
All three echoes print (API, SPA, and `/docs` reachable on the single port).

---

### Task 8 — Frontend verification (typecheck + build + smoke render)

**Files:** none new (uses Task 7's running server). Optional: a Playwright smoke script under
`dashboard/frontend/` (dev-only, NOT in the Python gate) — keep optional to avoid toolchain bloat.

**Steps:**
- [ ] `tsc --noEmit` clean (`npm run typecheck`).
- [ ] `vite build` clean (`npm run build`).
- [ ] Smoke: with `make dashboard` running, load `/` and click each of the 5 tabs; confirm each
      view renders data (no blank panels, no console errors), charts draw, and the Predictions
      table shows "X of 558". If Playwright is used, assert each view's heading + at least one
      data row/chart node is present; otherwise document a manual checklist in `dashboard/README.md`.

**Verification:**
```bash
npm --prefix dashboard/frontend run typecheck && npm --prefix dashboard/frontend run build
# Smoke (manual or scripted) against the running backend — all 5 views render with data.
```
Typecheck + build clean; smoke checklist passes against a running backend.

---

### Task 9 — Retire the old dashboard

**Files (remove):** `tools/dashboard/` (`serve.py`, `index.html`, `__init__.py`).
**Files (keep — explicit):** `tools/presentation/`, `tools/numerical_audit_compute.py`,
`tools/__init__.py`.
**Files (edit):** `README.md` (Repo Structure + any reproduce/dashboard references), and any doc
that points at `tools/dashboard` or `python -m tools.dashboard.serve`.

**Steps:**
- [ ] Only after Tasks 2–8 are green and parity is demonstrated, remove `tools/dashboard/`.
- [ ] Grep the repo for `tools.dashboard` / `tools/dashboard` / `serve.py` references and update
      them to the new `make dashboard` / `dashboard/` paths.
- [ ] Update README "Repo Structure" (`tools/` line) and add/repoint the Dashboard section to the
      new app + `make dashboard`.
- [ ] Confirm `tools/presentation/` and `tools/numerical_audit_compute.py` are untouched.

**Verification:**
```bash
! test -d tools/dashboard && echo "old dashboard removed"
test -f tools/numerical_audit_compute.py && test -d tools/presentation && echo "kept tools intact"
grep -rn "tools.dashboard\|tools/dashboard" --include="*.md" --include="*.py" . ; echo "(expect: no stale refs)"
.venv/bin/python3 -m pytest tests/ --ignore=tests/models/test_lstm_model.py --ignore=tests/models/test_lstm_trainer.py -q
```
Old dir gone, kept tools present, no stale references, fast suite green.

---

### Task 10 — README dashboard section + screenshot/GIF hook

**Files:** `README.md`, `dashboard/README.md`, `docs/assets/` (new; placeholder hero image path).

**Steps:**
- [ ] Add a **Dashboard** section to `README.md`: one-command run (`make dashboard` → localhost:8050),
      what it shows (the 5 views + new charts), the `/docs` OpenAPI link, prereqs (Node LTS +
      Python venv).
- [ ] Insert a **hero image placeholder** near the top of `README.md` (per Track A3), e.g.
      `![Dashboard](docs/assets/dashboard.png)` with a TODO note that the actual capture is a
      polish step (do not fabricate the image). Reserve `docs/assets/dashboard.png` /
      `dashboard.gif`.
- [ ] Note in the section that numbers shown match `results/`/README exactly (numbers-frozen),
      reinforcing the credibility story.

**Verification:**
```bash
grep -qi "## Dashboard" README.md && echo "dashboard section present"
grep -q "docs/assets/dashboard" README.md && echo "hero placeholder wired"
```
Both present (the real screenshot capture is a separate polish step, not this plan's blocker).

---

## 5. Definition of Done (whole track)

- [ ] `make dashboard` builds the SPA and serves `/api/*` + `/` + `/docs` on a single port (8050),
      no cloud.
- [ ] Backend is typed (mypy, scope-extended), ruff/black clean, and covered by
      `tests/dashboard/` which run in the **fast** gate.
- [ ] **Parity tests pass:** every dashboard number equals the canonical `results/`/README values
      (no metric recomputed differently; `src.evaluation.metrics` reused).
- [ ] All five legacy views reproduced (IA parity) + four new charts added; the legacy empty CV-
      disagreement section is now populated (improvement, not a number change).
- [ ] Frontend `tsc --noEmit` + `vite build` clean; 5-view smoke render passes.
- [ ] `tools/dashboard/` removed; `tools/presentation/` and `tools/numerical_audit_compute.py`
      kept; README repointed; hero placeholder wired.
- [ ] CI green on the fast suite (verification-before-completion: report real output, not claims).
- [ ] Commits are scoped per task; no `node_modules/`, `dist/`, untracked paper files, or `.claude/`
      committed.

## 6. Risks & Mitigations

- **Number drift** → backend reuses `src.evaluation.metrics`; Task 3 parity tests pin to `results/`/
  README; `audit_computed.json` cross-check as defense in depth.
- **CV disagreement was silently empty in the legacy app** → backend derives categories at the
  base-rate threshold (never 0.5), fixing the gap while staying numbers-frozen.
- **`TestClient` import error** → Task 1 adds `httpx` to `dev` and `dashboard` to CI install before
  any test runs.
- **Backend escaping the gate** → Task 2.1 extends mypy `files` + ruff/black/CI paths to
  `dashboard/backend`; backend NOT placed under ruff-excluded `tools/`.
- **Node absent on a future machine** → backend + `/docs` run without a build; SPA at `/` documented
  as needing a one-time `npm ci && npm run build`; parity tests are pure Python.
- **Memory-vs-README LSTM number conflict (0.6407/0.5202 vs 0.6372/0.5263)** → anchor to README/
  `results/` (canonical), surface the stale memory figure to the user; do not edit the README
  without sign-off.
- **Two toolchains reading as over-engineered** → minimal frontend, single `make dashboard`, small
  backend; no state library / CSS framework bloat (per design guardrail).
