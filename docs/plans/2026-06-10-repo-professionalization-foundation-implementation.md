# Repo Professionalization — Foundation Implementation Plan (Tracks A + B)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Design spec:** [`2026-06-10-repo-professionalization-design.md`](2026-06-10-repo-professionalization-design.md) — read it first. This plan covers **Tracks A (repo surface) and B (code rigor)** only. Track C (dashboard rebuild) is a separate plan written after this one lands.

**Goal:** Raise the repository's professional surface and engineering rigor — CI, license, clean root, modern dependency management, typing, reproducibility, model cards, and a claim–evidence audit — without altering the submitted research.

**Architecture:** A sequence of small, independently-committable changes. Low-risk cleanup first (dead-code removal, file relocations, license), then tooling + CI, then rigor-surfacing docs. Every step keeps the fast test suite green.

**Tech Stack:** Python 3.10+, pytest, ruff, black, mypy, pre-commit, GitHub Actions, PEP 621 `pyproject.toml`, GNU Make.

**Conventions for this plan:**
- **Fast test suite** = `python3 -m pytest tests/ --ignore=tests/models/test_lstm_model.py --ignore=tests/models/test_lstm_trainer.py -q`
- Work on branch `feature/repo-professionalization` (already created).
- Author for license/headers: **Wil Fowler**, year **2026**.
- Guardrail: numbers stay anchored to `results/` artifacts; claims may be corrected by evidence (Task 12) with user sign-off; no model re-runs.

---

## Task 0: Establish a green baseline

**Files:** none (verification only).

- [ ] **Step 1: Confirm the venv and run the fast suite**

Run:
```bash
source .venv/bin/activate
python3 -m pytest tests/ --ignore=tests/models/test_lstm_model.py --ignore=tests/models/test_lstm_trainer.py -q
```
Expected: all tests pass. **Record the passing count** (e.g. "221 passed") — later tasks reference this number to prove nothing regressed.

- [ ] **Step 2: Confirm lint is clean**

Run:
```bash
python3 -m ruff check src/ tests/ && black --check src/ tests/
```
Expected: ruff reports no errors; black reports all files would be left unchanged. If not clean, STOP and report — the baseline must be green before proceeding.

---

## Task 1: Remove dead code (`comparison.py`)

**Files:**
- Delete: `src/evaluation/comparison.py`
- Delete: `tests/evaluation/test_comparison.py`
- Modify: `src/evaluation/__init__.py` (remove the `comparison.py` docstring line)
- Modify: `CLAUDE.md` (remove the "comparison.py is unused" invariant + Change-Impact Map line)
- Modify: `AGENTS.md` (no `comparison` line in its map — verify, no change expected)

**Context:** Confirmed unused — no `ModelComparison` import exists in `src/` or `tools/`; `CLAUDE.md` already documents it as dead. This is the one deletion in the plan.

- [ ] **Step 1: Prove it is unused (trace before delete)**

Run:
```bash
grep -rn "ModelComparison\|evaluation.comparison\|from .comparison\|import comparison" src/ tools/ --include='*.py' | grep -v "src/evaluation/comparison.py"
```
Expected: **no output** (empty). If anything prints, STOP — it is not dead; report and revisit.

- [ ] **Step 2: Delete the module and its test**

Run:
```bash
git rm src/evaluation/comparison.py tests/evaluation/test_comparison.py
```

- [ ] **Step 3: Remove the docstring reference in `src/evaluation/__init__.py`**

Delete this line from the module docstring:
```
  comparison.py   — Model comparison utilities.
```

- [ ] **Step 4: Update `CLAUDE.md`**

Remove the invariant line:
> - **comparison.py is unused**: `ModelComparison` is not imported anywhere in production. Use `DisagreementAnalyzer` instead.

And remove this line from the Change-Impact Map:
```
comparison.py                   → (UNUSED in production — dead code)
```

- [ ] **Step 5: Run the fast suite — verify it still passes with a lower count**

Run the fast test suite (see conventions).
Expected: PASS, with the count reduced by exactly the number of tests that were in `test_comparison.py` (the comparison tests are gone, nothing else changed).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: remove unused comparison.py dead code

Confirmed no ModelComparison import in src/ or tools/. Drops the module,
its test, and the now-obsolete dead-code notes in CLAUDE.md.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Relocate internal docs out of the root

**Files:**
- Move: `PROJECT.md` → `docs/architecture-and-analysis.md`
- Create dir + move: `BUGHUNT.md`, `audit_results.md`, `slide_reconciliation.md` → `docs/development/`
- Modify: `tools/numerical_audit_compute.py` (output paths for the two generated files)
- Modify: `CLAUDE.md` (BUGHUNT.md links)
- Modify: `tools/presentation/rewrite_presentation.py:133` (stale `PROJECT.md` docstring reference)

**Context:** `docs/architecture-and-analysis.md` is referenced by `CLAUDE.md` and `AGENTS.md` but does not exist — its content is `PROJECT.md` at root. `audit_results.md` and `slide_reconciliation.md` are **generated** by `numerical_audit_compute.py`, so their output paths must move too or they reappear at root.

- [ ] **Step 1: Move the architecture doc (fixes the broken link)**

```bash
git mv PROJECT.md docs/architecture-and-analysis.md
```
This makes the existing `CLAUDE.md` and `AGENTS.md` references resolve. No edits needed to those two references — they already point at `docs/architecture-and-analysis.md`.

- [ ] **Step 2: Move the internal process docs**

```bash
mkdir -p docs/development
git mv BUGHUNT.md docs/development/BUGHUNT.md
git mv audit_results.md docs/development/audit_results.md
git mv slide_reconciliation.md docs/development/slide_reconciliation.md
```

- [ ] **Step 3: Update the audit generator's output paths**

In `tools/numerical_audit_compute.py`, change the two write targets (around lines 1842–1843):
```python
    (ROOT / "audit_results.md").write_text(render_audit_results(audit))
    (ROOT / "slide_reconciliation.md").write_text(render_slide_reconciliation(audit))
```
to:
```python
    (ROOT / "docs" / "development" / "audit_results.md").write_text(render_audit_results(audit))
    (ROOT / "docs" / "development" / "slide_reconciliation.md").write_text(
        render_slide_reconciliation(audit)
    )
```
Also update the two `print(...)` confirmations just below to the new paths.

- [ ] **Step 4: Fix the BUGHUNT.md links in `CLAUDE.md`**

Update the two references (lines ~109, ~112) to point at the new location:
```
See [`docs/development/BUGHUNT.md`](docs/development/BUGHUNT.md) for the full list of 43 bugs found 2026-03-09.
```
(and the second `BUGHUNT.md` mention similarly).

- [ ] **Step 5: Fix the stale `PROJECT.md` docstring in the presentation tool**

In `tools/presentation/rewrite_presentation.py:133`, change `from PROJECT.md` to `from docs/architecture-and-analysis.md`.

- [ ] **Step 6: Verify no stale references remain**

Run:
```bash
grep -rn "PROJECT.md\|](BUGHUNT.md)\|(audit_results.md)\|(slide_reconciliation.md)" --include='*.md' --include='*.py' . | grep -v '.git/' | grep -v '.venv/' | grep -v '.claude/'
```
Expected: no output (all references updated).

- [ ] **Step 7: Run the fast suite (import-path sanity) and commit**

Run the fast test suite. Expected: PASS (no behavior changed).
```bash
git add -A
git commit -m "docs: relocate internal docs out of repo root

PROJECT.md -> docs/architecture-and-analysis.md (fixes broken CLAUDE.md /
AGENTS.md links). BUGHUNT.md + audit_results.md + slide_reconciliation.md ->
docs/development/. Updates the numerical-audit generator output paths and
stale references.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Add LICENSE and tighten `.gitignore`

**Files:**
- Create: `LICENSE`
- Modify: `.gitignore`

- [ ] **Step 1: Create `LICENSE` (MIT)**

```
MIT License

Copyright (c) 2026 Wil Fowler

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 2: Add ignore rules for the worktree clutter and future dashboard build**

Append to `.gitignore`:
```gitignore
# Stale Claude worktrees
.claude/worktrees/

# Dashboard build artifacts (Track C)
dashboard/frontend/node_modules/
dashboard/frontend/dist/
```

- [ ] **Step 3: Commit**

```bash
git add LICENSE .gitignore
git commit -m "chore: add MIT LICENSE and ignore worktree/dashboard build artifacts

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Consolidate dependencies into `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`
- Delete: `requirements.txt`
- Modify: `CLAUDE.md` (install instructions), `AGENTS.md` (if it references requirements.txt — it does not; verify)

**Context:** Drop the six confirmed-unused packages (`captum`, `optuna`, `matplotlib`, `seaborn`, `plotly`, `jupyter`). Keep `mlflow` + `shap` (used). Split heavy/optional deps into extras.

- [ ] **Step 1: Replace `pyproject.toml` with consolidated metadata + deps**

```toml
[project]
name = "nfl-upset-prediction"
version = "0.1.0"
description = "Multi-architecture model disagreement for NFL upset prediction"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [{ name = "Wil Fowler" }]

dependencies = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "nfl-data-py>=0.3.0",
    "xgboost>=2.0.0",
    "scikit-learn>=1.3.0",
    "shap>=0.42.0",
    "mlflow>=2.8.0",
]

[project.optional-dependencies]
lstm = ["torch>=2.0.0"]
dashboard = ["fastapi>=0.110.0", "uvicorn>=0.29.0"]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.9.0",
    "ruff>=0.1.0",
    "mypy>=1.8.0",
    "pre-commit>=3.5.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
addopts = "-q --tb=short"

[tool.black]
line-length = 88
target-version = ["py310"]

[tool.ruff]
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I"]
```

- [ ] **Step 2: Remove the old requirements file**

```bash
git rm requirements.txt
```

- [ ] **Step 3: Verify a clean install resolves**

Run:
```bash
pip install -e ".[dev,lstm]" --dry-run
```
Expected: pip resolves the dependency set without error. (Use `--dry-run` to avoid mutating the working venv; if unsupported by the local pip, run the real install in a scratch venv.)

- [ ] **Step 4: Update install instructions in `CLAUDE.md`**

Replace the Environment Setup block's `pip install -r requirements.txt` line with:
```bash
pip install -e ".[dev,lstm]"   # editable install incl. PyTorch (lstm) + dev tools
```

- [ ] **Step 5: Run the fast suite and commit**

Run the fast test suite. Expected: PASS (deps unchanged for what the code imports).
```bash
git add -A
git commit -m "build: consolidate dependencies into pyproject.toml (PEP 621)

Drops six unused packages (captum, optuna, matplotlib, seaborn, plotly,
jupyter). Splits torch into a 'lstm' extra and fastapi/uvicorn into a
'dashboard' extra; dev tooling into 'dev'. Removes flat requirements.txt.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Add `mypy` and reach a green type check

**Files:**
- Modify: `pyproject.toml` (append `[tool.mypy]`)

**Context:** Goal is a *passing, useful* type check, not full strictness. Start permissive (catch real errors, don't demand annotations on data scripts), tighten later. This is a discovery task: run mypy, then resolve reported errors either by a real fix or a scoped per-module override.

- [ ] **Step 1: Append a pragmatic mypy config to `pyproject.toml`**

```toml
[tool.mypy]
python_version = "3.10"
ignore_missing_imports = true
warn_unused_ignores = true
warn_redundant_casts = true
no_implicit_optional = true
files = ["src"]
# Pragmatic start: do not require full annotations yet.
disallow_untyped_defs = false
check_untyped_defs = false
```

- [ ] **Step 2: Run mypy and read the report**

Run:
```bash
python3 -m mypy
```
Expected initially: a list of real type errors (if any).

- [ ] **Step 3: Resolve to exit code 0**

For each reported error, prefer a genuine fix (correct an annotation, add a missing return, narrow an Optional). For any error that is a false positive or in a module not worth annotating now, add a scoped override rather than weakening the global config, e.g.:
```toml
[[tool.mypy.overrides]]
module = "src.models.lstm_trainer"
check_untyped_defs = false
```
Re-run `python3 -m mypy` until it reports `Success: no issues found`. **Do not** silence errors with blanket `# type: ignore` on whole files; use targeted, commented ignores only where a fix is genuinely impractical.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml src/
git commit -m "build: add mypy config and reach a green type check

Pragmatic baseline (ignore_missing_imports, no forced annotations) with
per-module overrides where needed. Catches real type errors without
demanding a full annotation pass.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Add a `Makefile`

**Files:**
- Create: `Makefile`

- [ ] **Step 1: Create `Makefile`**

```makefile
.PHONY: help install test test-all lint format typecheck features evaluate ab reproduce dashboard dashboard-dev clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Create venv and install deps (incl. lstm + dev)
	python3 -m venv .venv && . .venv/bin/activate && pip install -e ".[dev,lstm]"

test:  ## Fast test suite (excludes slow LSTM model/trainer tests)
	python3 -m pytest tests/ --ignore=tests/models/test_lstm_model.py --ignore=tests/models/test_lstm_trainer.py -q

test-all:  ## Full test suite including slow LSTM tests
	python3 -m pytest tests/ -q

lint:  ## Ruff + black --check
	python3 -m ruff check src/ tests/ && black --check src/ tests/

format:  ## Auto-format with black and ruff --fix
	black src/ tests/ && python3 -m ruff check --fix src/ tests/

typecheck:  ## Run mypy
	python3 -m mypy

features:  ## Regenerate train.csv + test.csv
	python3 -m src.data.generate_features

evaluate:  ## Evaluate on the held-out test set
	python3 -m src.models.evaluate_test_set

ab:  ## Run the spread A/B experiment (quick: LR + XGB)
	python3 -m src.models.run_ab_experiment --quick

reproduce: features ab evaluate  ## Deterministic end-to-end regeneration (numbers stay anchored to results/)

dashboard:  ## Build the frontend and serve the dashboard on localhost (Track C)
	@echo "Dashboard target is wired up in the Track C plan."

dashboard-dev:  ## Run the dashboard in dev mode with hot reload (Track C)
	@echo "Dashboard-dev target is wired up in the Track C plan."

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__
```

- [ ] **Step 2: Verify a few targets work**

Run:
```bash
make help
make lint
make test
```
Expected: `make help` prints the target list; `make lint` and `make test` pass. (The `dashboard*` targets are placeholders until Track C; they print a note and exit 0.)

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "build: add Makefile for one-command dev ergonomics

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Add `pre-commit`

**Files:**
- Create: `.pre-commit-config.yaml`

- [ ] **Step 1: Create `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
  - repo: https://github.com/psf/black
    rev: 24.4.2
    hooks:
      - id: black
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix]
```

- [ ] **Step 2: Install and run against all files**

Run:
```bash
pre-commit install
pre-commit run --all-files
```
Expected: hooks run; black/ruff make no changes (baseline already formatted). If a hook reports fixes, review them, ensure the fast suite still passes, and stage the changes.

- [ ] **Step 3: Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "build: add pre-commit hooks (black, ruff, hygiene checks)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Add GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create the workflow**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  quality:
    name: Lint, type-check, fast tests
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - name: Install (dev + lstm)
        run: pip install -e ".[dev,lstm]"
      - name: Ruff
        run: python -m ruff check src/ tests/
      - name: Black
        run: black --check src/ tests/
      - name: Mypy
        run: python -m mypy
      - name: Fast tests
        run: python -m pytest tests/ --ignore=tests/models/test_lstm_model.py --ignore=tests/models/test_lstm_trainer.py -q

  lstm-tests:
    name: Slow LSTM tests (non-blocking)
    runs-on: ubuntu-latest
    continue-on-error: true
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - name: Install (dev + lstm)
        run: pip install -e ".[dev,lstm]"
      - name: LSTM tests
        run: python -m pytest tests/models/test_lstm_model.py tests/models/test_lstm_trainer.py -q
```

- [ ] **Step 2: Validate the YAML locally**

Run:
```bash
python3 -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml')); print('YAML OK')"
```
Expected: `YAML OK`.

- [ ] **Step 3: Commit and push the branch to trigger CI**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions (ruff, black, mypy, fast tests on 3.10/3.11)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git push -u origin feature/repo-professionalization
```
Expected: after push, the `quality` job goes green on GitHub. **Confirm the green check before claiming this task done** (per verification-before-completion). If red, read the failing job log and fix forward.

---

## Task 9: README badges + Quickstart refresh

**Files:**
- Modify: `README.md`

**Context:** Keep the existing honest prose. Add a badge row and a `pyproject`-based quickstart. The dashboard screenshot/GIF and any final number reconciliation come later (Track C / Task 12).

- [ ] **Step 1: Add a badge row directly under the title**

Insert after the `# NFL Upset Taxonomy via Multi-Architecture Disagreement` line (replace `OWNER/REPO` with the actual GitHub path):
```markdown
[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
```

- [ ] **Step 2: Replace the Reproduce section with a pyproject-based Quickstart**

```markdown
## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,lstm]"        # editable install incl. PyTorch + dev tools

make features      # regenerate train.csv + test.csv
make ab            # spread A/B experiment (quick: LR + XGB)
make evaluate      # held-out test-set evaluation
make test          # fast test suite
```
```

- [ ] **Step 3: Verify the GitHub owner/repo slug**

Run:
```bash
git remote get-url origin
```
Use the returned `OWNER/REPO` to fill the badge URLs. If there is no remote yet, leave a clearly-marked `OWNER/REPO` placeholder and note it in the commit body.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add CI/license/python badges and a pyproject-based quickstart

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Reproducibility — a seed helper

**Files:**
- Create: `src/utils/__init__.py`
- Create: `src/utils/seed.py`
- Create: `tests/utils/__init__.py`
- Create: `tests/utils/test_seed.py`

**Context:** Provide one documented place that seeds Python, NumPy, and (if present) Torch. This is a *new utility with a clear contract*, so it is test-driven. It does not change any existing result — callers adopt it incrementally; wiring it into trainers is out of scope for this task to avoid touching frozen numbers.

- [ ] **Step 1: Write the failing test**

Create `tests/utils/test_seed.py`:
```python
from __future__ import annotations

import numpy as np

from src.utils.seed import set_global_seed


def test_set_global_seed_makes_numpy_reproducible():
    set_global_seed(123)
    a = np.random.rand(5)
    set_global_seed(123)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_set_global_seed_returns_the_seed():
    assert set_global_seed(7) == 7
```

- [ ] **Step 2: Run it to verify it fails**

Run:
```bash
python3 -m pytest tests/utils/test_seed.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils.seed'`.

- [ ] **Step 3: Implement the helper**

Create `src/utils/__init__.py` (empty) and `src/utils/seed.py`:
```python
"""Global seeding for reproducible runs.

Seeds Python's `random`, NumPy, and (when installed) PyTorch from a single
entry point so experiment scripts can pin determinism in one call.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> int:
    """Seed Python, NumPy, and Torch (if available). Returns the seed used."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed
```

- [ ] **Step 4: Create `tests/utils/__init__.py` (empty) and run the test**

Run:
```bash
python3 -m pytest tests/utils/test_seed.py -q
```
Expected: PASS (2 passed).

- [ ] **Step 5: Run the full fast suite (no regressions) and commit**

Run the fast test suite. Expected: PASS, count up by 2 from Task 4's baseline.
```bash
git add src/utils tests/utils
git commit -m "feat: add set_global_seed reproducibility helper (tested)

Single entry point seeding random/numpy/torch. Does not alter existing
results; available for experiment scripts to adopt.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Model cards

**Files:**
- Create: `docs/model-cards/README.md`
- Create: `docs/model-cards/logistic-regression.md`
- Create: `docs/model-cards/xgboost.md`
- Create: `docs/model-cards/lstm.md`

**Context:** Standard, professional artifact. All content is drawn from existing frozen results (`README.md`, `results/`, `docs/architecture-and-analysis.md`). No new claims. Use the same numbers already published.

- [ ] **Step 1: Create the index `docs/model-cards/README.md`**

```markdown
# Model Cards

Each model in this project is documented as a model card: its representation,
features, training data, intended use, evaluation, and limitations. All numbers
are anchored to `results/` and match the figures reported in the top-level README.

- [Logistic Regression](logistic-regression.md) — the static snapshot baseline
- [XGBoost](xgboost.md) — snapshot + short lag structure
- [Siamese LSTM](lstm.md) — each team as a recent sequence

> These models are diagnostic instruments for studying upset *mechanisms* via
> disagreement, not a production betting system. See the repository README.
```

- [ ] **Step 2: Create `docs/model-cards/logistic-regression.md`**

Fill this template using the published numbers (CV AUC 0.6497, Test AUC 0.5622, Test Brier 0.2026; 46 features / 42 no-spread):
```markdown
# Model Card — Logistic Regression ("The Summary")

**Representation:** A single static statistical snapshot of each matchup.

## Features
- 46 base features (42 in the no-spread variant): rolling averages, differentials,
  market line, and Elo. Canonical list: `src/features/pipeline.py`.

## Training data
- 3,495 labeled games, 2005–2022. Labels apply only to games with `spread >= 3`;
  sub-3 games are retained for rolling-feature continuity and excluded via `upset.notna()`.
- Base upset rate ≈ 30%. Decision threshold is the base rate, **not** 0.5.

## Evaluation (frozen artifacts)
| Split | AUC | Brier |
|-------|-----|-------|
| 6-fold expanding-window CV | 0.6497 | — |
| Held-out test (2023–2025) | 0.5622 | 0.2026 |

## Intended use
- Diagnostic baseline: the most interpretable model in the set; LR coefficients
  give directional influence directly (see the dashboard Feature Weights view).

## Limitations
- Linear snapshot only; cannot represent interactions or temporal dynamics.
- Like all three models, loses substantial signal when the market spread is removed.
```

- [ ] **Step 3: Create `docs/model-cards/xgboost.md`**

Same template, XGBoost specifics (70 features = 46 base + 24 lag, 66 no-spread; CV AUC 0.6377; Test AUC 0.5755; Test Brier 0.2013; best held-out generalizer; top-10 held-out predictions contain 6 real upsets). Note `max_depth` as configured in `src/models/xgboost_model.py` — read the actual value rather than assuming.

- [ ] **Step 4: Create `docs/model-cards/lstm.md`**

Same template, LSTM specifics (14 sequence features × 8 timesteps + 10 matchup, 8 matchup no-spread; CV AUC 0.6372; Test AUC 0.5263; Test Brier 0.2089; largest CV→test drop; most behaviorally distinct on test — probability correlations 0.373 LR-LSTM, 0.309 XGB-LSTM). Frame its value as diagnostic per the README.

- [ ] **Step 5: Verify numbers against the README (consistency check)**

Cross-check every figure in the three cards against the "Results At A Glance" and ablation tables in `README.md`. They must match exactly. If any differ, the card is wrong — fix the card (this task does not change README numbers).

- [ ] **Step 6: Commit**

```bash
git add docs/model-cards
git commit -m "docs: add model cards for LR, XGBoost, and LSTM

All figures anchored to results/ and consistent with the README.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Claim–evidence reconciliation audit

**Files:**
- Create: `docs/development/claim-audit-2026-06-10.md` (findings)
- Modify (only after user sign-off): `README.md`, the canonical paper, `docs/architecture-and-analysis.md`

**Context:** This is the "claims may be corrected by evidence" guardrail in action. It is **interactive**: produce a findings table, get the user's sign-off per change, then apply. Build on the existing audits in `docs/development/`.

- [ ] **Step 1: Extract every quantitative/interpretive claim from README + paper**

Read `README.md`, the canonical paper, and `docs/architecture-and-analysis.md`. For each numeric or comparative claim (e.g., "XGBoost generalizes best", "6 of top-10 are real upsets", every AUC/Brier figure), note the claim and the artifact that should support it (`results/test/predictions.csv`, `results/ab_experiment/*`).

- [ ] **Step 2: Verify each claim against the artifact**

Recompute the supporting figure from the artifact (reuse `src/evaluation/metrics.py`; do **not** retrain anything). Cross-reference the existing `docs/development/audit_results.md` and `verified-numbers.md`. Classify each claim: **Confirmed**, **Imprecise** (needs wording fix), or **Wrong** (number/claim contradicts the artifact).

- [ ] **Step 3: Write the findings doc**

Create `docs/development/claim-audit-2026-06-10.md` as a table: `Claim | Location | Artifact | Recomputed value | Verdict | Proposed change`. Leave proposed changes concrete and minimal.

- [ ] **Step 4: Surface findings to the user and get per-change sign-off**

Present the table. **Do not edit README/paper claims without explicit approval** for each non-trivial change (the guardrail). Trivial typo/rounding fixes may be applied and listed.

- [ ] **Step 5: Apply approved corrections consistently**

For each approved change, update README, the paper, the architecture doc, and any model card so the figure/claim is identical everywhere. Re-run the consistency cross-check from Task 11 Step 5.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "docs: reconcile claims against results/ artifacts

Adds claim-audit findings and applies approved evidence-grounded corrections
across README, paper, architecture doc, and model cards. No models re-run.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Done criteria (Foundation)

- [ ] Fast suite green; full suite green locally; CI `quality` job green on GitHub.
- [ ] `make help/test/lint/typecheck` all work.
- [ ] Root contains only: `README.md`, `LICENSE`, `Makefile`, `pyproject.toml`, `CLAUDE.md`, `AGENTS.md`, `.gitignore`, plus the source/doc/data directories.
- [ ] `comparison.py` gone; no dead references.
- [ ] `docs/architecture-and-analysis.md` exists; no broken doc links.
- [ ] Dependencies in `pyproject.toml`; six unused packages dropped; `requirements.txt` removed.
- [ ] Model cards present and number-consistent with README.
- [ ] Claim audit complete; approved corrections applied.
- [ ] Then proceed to the Track C (dashboard) plan.
```
