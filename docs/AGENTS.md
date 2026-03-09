# docs/AGENTS.md

Use the root [`AGENTS.md`](/Users/wilfowler/Documents/Projects/apr-research/AGENTS.md) naming scheme for all repo documentation. This file adds docs-specific formatting rules.

## Required Doc Labels

For any new design, implementation, or review doc, include near the top:

- `Doc Type`
- `Topic`
- `Topic Slug`
- `Date`
- `Status`
- `Primary Spec` for implementation and review docs

## Canonical Meanings

- `Design Spec` means the main doc for a topic.
- `Implementation Plan` means execution detail for a design spec.
- `Review Report` means critique or audit of a design or result.
- `Results` means measured outcomes from actual runs.
- `Notes` means non-authoritative working material.

## Writing Rules

- If a topic has multiple docs, use the same topic slug across all of them.
- Keep design and implementation docs in `docs/plans/`.
- Keep review reports, results summaries, and paper-facing docs in `docs/`.
- If a review recommendation is accepted, update the design spec instead of letting the review remain the only place where the decision exists.
- Do not call a review or implementation doc the "main doc".

## Current Data Invariants

- The canonical upset label applies only to games with `spread >= 3`; sub-3 games remain in generated datasets with `upset = NaN` for rolling-feature continuity and are excluded from modeling via `upset.notna()`.
- Current clean labeled splits are 3,495 train games and 558 test games, with upset rate roughly 30% in both sets.
- March 2026 label cleanup removed 846 train and 193 test fake negatives from the labeled data.
- `src.data.generate_features` now enforces 8 strict validation checks on each split and a separate train/test season-overlap check.
- When writing results, status, or review docs, treat pre-cleanup metrics based on the old ~21% upset rate as historical only unless they have been recomputed on the clean labeled data.
- If you cite dataset size, distinguish total generated rows from labeled rows.

## Example (Completed)

- `feature-redesign` design spec (implemented 2026-03-03):
  - [`docs/plans/2026-03-02-feature-redesign-design.md`](/Users/wilfowler/Documents/Projects/apr-research/docs/plans/2026-03-02-feature-redesign-design.md)
- `feature-redesign` implementation plan:
  - [`docs/plans/2026-03-02-feature-redesign-implementation.md`](/Users/wilfowler/Documents/Projects/apr-research/docs/plans/2026-03-02-feature-redesign-implementation.md)
- `feature-redesign` review report:
  - [`docs/2026-03-02-feature-redesign-review-report.md`](/Users/wilfowler/Documents/Projects/apr-research/docs/2026-03-02-feature-redesign-review-report.md)
