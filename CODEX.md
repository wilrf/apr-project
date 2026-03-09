# CODEX.md

Canonical repo instructions, change-impact map, and coding conventions live in [`AGENTS.md`](AGENTS.md).

## Codex Delta

- When a user asks for the "main doc" for a topic, resolve to the matching `Design Spec`.
- Full architecture reference: `docs/architecture-and-analysis.md` — read before non-trivial changes.
- The canonical feature pipeline produces 70 features (46 base + 24 XGB per-game lags) in `src/features/pipeline.py`.
- Each model gets a different representation: LR=46 base, XGB=70 expanded, LSTM=14seq×8ts+10matchup.
- The canonical target labels only games with `spread >= 3`; sub-3 games stay with `upset = NaN`, excluded via `upset.notna()`.
- Disagreement analyzer uses base-rate threshold (~0.30), not 0.5.
- Treat metrics from before the March 2026 label cleanup as stale.
- Treat the March 2026 feature-redesign docs as historical background.

## Common Commands

```bash
python3 -m pytest tests/ --ignore=tests/models/test_lstm_model.py -v
python3 -m pytest tests/models/test_lstm_model.py -v
black src/ tests/
python3 -m ruff check src/ tests/
python3 -m src.data.generate_features
python3 -m src.models.evaluate_test_set
python3 -m src.models.run_ab_experiment --quick
```
