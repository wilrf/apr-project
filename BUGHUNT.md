# BUGHUNT.md

**Generated:** 2026-03-09
**Last verified:** 2026-03-09

## Scope

This report replaces the earlier oversized bug list.

- I treated the previous `BUGHUNT.md` as a lead list, not as truth.
- I re-checked claims against the current repo with code inspection, targeted `rg` searches, and small Python repros.
- I retained only issues I can defend from the current tree or from directly inspected upstream package source.
- I merged duplicates and dropped claims that were stale, environment-local, future-version speculation, or too weakly supported.

## Current State

The most important result of this pass is that the previously reported critical code-path bugs around target-label coverage, NaN-safe sequence normalization, and calibration-path `fillna(0)` are already fixed in the current tree.

The repo still has a meaningful set of open issues, but the current risk profile is:

- result integrity and evaluation edge cases,
- dependency/reproducibility weakness,
- silent fallback behavior in the LSTM data path,
- thin tests around the experiment entrypoints.

## Fixed Since The Prior Report

These prior claims are already fixed in the current tree and are not carried as open issues here:

`C1`, `C2`, `C3`, `C4`, `H1`, `H2`, `H3`, `H5`, `H6`, `H7`, `H8`, `H9`, `H10`, `H11`, `H12`, `H13`, `H14`, `H15`

## Current High-Priority Issues

- Incomplete normalization stats still fail open in the LSTM path. In `src/models/sequence_builder.py`, sequence stats only use provided values when `feature_name in stats`; otherwise they recompute from current data, and matchup stats use `.get(feat_name, (0.0, 1.0))`. Effect: a feature-name mismatch can leak evaluation data into normalization or silently disable normalization. Maps old `H23`, `M47`.

- The primary raw-data dependency is both archived and insecure on the schedule path. `src/data/nfl_loader.py` still depends on `nfl_data_py`, and the installed package source still references `http://www.habitatring.com/games.csv`. Effect: fresh data pulls are tied to an unmaintained package and one critical schedule source is fetched over plain HTTP. Maps old `H20`, `H28`.

- Dependency resolution is still not reproducible. `requirements.txt` has only open-ended lower bounds, there is no lock file in the repo, and the runtime requirements list still includes packages with no `src/` imports (`captum`, `optuna`, `matplotlib`, `seaborn`, `plotly`, `jupyter`). `mlflow` and `shap` are actually used, so the old “8 unused” claim was overstated. Effect: fresh installs can drift and the dependency surface is noisier than necessary. Maps old `H18`, `H19`, revised `H21`.

- `src/data/generate_features.py` writes `train.csv` and `test.csv` sequentially without an atomic handoff. A crash between the two writes can leave mismatched splits on disk. Maps old `H24`.

- `src/evaluation/calibration.py` does not validate key alignment in `calibrate_models()`. If `test_probs` has an extra key it is silently dropped; if `cal_probs` has a key missing from `test_probs`, the function raises a raw `KeyError`. Effect: calibration failures are hard to diagnose and one-sided key mismatches can silently remove a model from results. Maps old `M17`.

- `src/models/sequence_builder.py` can silently build ghost or all-zero samples when metadata is bad. I reproduced a case where `upset` is present but `underdog` is `NaN`: the sample keeps a real target, zero masks, and a real `game_id`. The same function also silently zeroes out sequences if required columns are missing or score columns do not match the expected schema. Effect: the LSTM can train on “no signal” samples without raising. Maps old `M24`, `M32`, `M33`.

- `roc_auc_score` is still unguarded in several trainer and experiment paths: `src/models/trainer.py`, `src/models/lstm_trainer.py`, `src/models/unified_trainer.py`, and `src/models/run_ab_experiment.py`. Effect: a single-class fold or bucket can still crash a long run outside the one place that already catches it. Maps old `M2`.

- The experiment/test boundary is still under-protected. `src/models/run_ab_experiment.py` has no tests at all, and `tests/data/test_nfl_loader.py` still makes live schedule and PBP calls. Effect: the main experiment entrypoint can regress silently, and the data-loader test suite is slow, network-dependent, and non-hermetic. Maps old `H22`, `H33`.

## Current Medium-Priority Issues

- `GamePrediction.lr_pred`, `xgb_pred`, and `lstm_pred` still hardcode `0.5`, even though the project-wide disagreement threshold is base-rate driven. The `*_pred_at()` methods are correct; the properties are the trap. Maps old `M1`.

- Several `log_loss` call sites still do not clip probabilities before scoring. The unguarded calls remain in `src/models/trainer.py`, `src/models/unified_trainer.py`, `src/models/evaluate_test_set.py`, and `src/models/run_ab_experiment.py`. Maps old `M3`.

- Matchup normalization still has three silent fallback behaviors in `src/models/sequence_builder.py`: zero-std columns are left uncentered, NaN-filled zeros bias the computed stats, and missing matchup feature columns become all-zero arrays. Effect: the matchup half of LSTM normalization is still much weaker than the sequence half. Maps old `M4`, `M27`, `M28`, `L15`.

- `_identify_underdog()` in `src/features/pipeline.py` still returns `home_team` whenever `team_favorite_id` is present but does not match `home_team`. If the favorite ID matches neither side, the function still returns the home team as the underdog instead of failing closed. Maps old `M5`.

- `src/models/evaluate_test_set.py` still has no top-level failure boundary around the full evaluation pipeline. If a later-stage failure occurs after model training, the script still loses all progress. Maps old `M6`.

- Calibration and reporting edge cases are still brittle. The current code still has fragile ECE bin matching, constant-prediction correlations that become `NaN`, all-`NaN` `qcut` buckets, and `ReportGenerator.generate_summary()` still crashes on empty `y_pred`. Maps old `M7`, `M15`, `M26`, `L16`.

- Calibration input validation is still incomplete. `generate_calibration_predictions()` checks `cal_df` but not `fit_df`, so an earliest-season calibration split still fails with a raw sklearn error. `PlattScaler.fit()` also relies on sklearn’s raw single-class failure instead of raising a project-specific message. Maps old `M9`, `M31`, duplicate `M42`.

- The LSTM still behaves differently across code paths. There is no explicit PyTorch seeding in the training loops, the attention mask is applied after softmax in `src/models/lstm_model.py`, the calibration path trains via `train_final()` without early stopping, ties are still encoded as losses in team history, and `epochs=0` still crashes `_run_lstm_training()` with `UnboundLocalError`. Maps old `M16`, `M18`, `M22`, `M23`, `M34`.

- Type boundaries are still loose in the training/evaluation layer. `SiameseLSTMTrainer` still does not satisfy the shared `Model` protocol, `train_final()` still returns `Dict[str, Any]`, many public functions in `evaluate_test_set.py` and `run_ab_experiment.py` still lack annotations, `generate_calibration_predictions()` still breaks `typing.get_type_hints()` because `pd` is not imported at module scope, and helper signatures like `normalize_team_abbr()` still lie about `NaN`-ish inputs. Maps old `H16`, `H17`, `M10`, `M11`, `M39`.

- MLflow/SHAP hardening is still thin. `MLflowTracker` still accepts arbitrary `tracking_uri`, the enabled MLflow behavior is barely tested, one MLflow test is still tautological, and `compute_shap_values()` still does not guard against an unfitted model. Maps old `M8`, `M12`, `M20`, `M45`.

- Result assembly still has fragile edge cases. `build_game_predictions()` and `_build_predictions()` still stringify `NaN` team names to `"nan"`, the prediction-category export path still keys lookups by `id()`, `_compute_checked_loss()` still does not validate targets before loss computation, and the CV aggregators still use plain `np.mean()`/`np.std()` over fold metrics. Maps old `M25`, `M29`, `M30`, `M41`.

- The pipeline still does redundant work. Calibration retrains models instead of reusing CV outputs, unified CV rebuilds team history repeatedly, `sequence_builder.py` duplicates aggregation logic already present in `pipeline.py`, and the full A/B run still redoes the quick LR/XGB fits before running the full experiment. Maps old `M14`, `H25`, `H29`, `H30`, `H31`.

- Several high-value test blind spots remain. The LSTM trainer fixture still forces `spread_favorite = abs(spread)`, there is still no direct test that `verify_data()` raises on season overlap, and the calibration-metrics test still exercises `n_bins=1` rather than the production-style path. Maps old `M13`, `M44`, `M46`.

## Current Low-Priority Issues

- `tests/models/test_cv_splitter.py` still checks `train_sizes == sorted(train_sizes)`, which allows equality instead of strict growth. Maps old `L2`.

- `src/features/pipeline.py` still builds `upset_tier` with `default=None`, creating an object-dtype column. Maps old `L5`.

- `src/data/betting_loader.py` still uses truthiness checks for `min_season` and `max_season`, so `0` is treated as “not provided”. Maps old `L8`.

- `src/data/merger.py::save_merge_audit()` still writes directly to the target path without ensuring the parent directory exists. Maps old `L9`.

- The five `src/**/__init__.py` files still do not follow the repo convention that says every module should start with `from __future__ import annotations`. I am keeping this as low-priority convention debt rather than a runtime bug. Maps old `L12`.

## Not Carried Forward From The Prior Report

These prior claims were reviewed and are not carried as current repo bugs in this rewritten report:

- `H4` is still a false positive. The betting-profit math is algebraically correct.

- `C5` is not carried as a current repo bug because the repo does not call `torch.load()` anywhere. I still count dependency pinning as a reproducibility problem, but not as an active runtime exploit path in this codebase.

- `C6` and `H27` were not independently reproduced from the current tree. I found upstream discussion around `nfl_data_py` brittleness, but I did not reproduce the exact NumPy 2 / Python 3.13 failure claims from this environment.

- `H26` is a local environment mismatch, not a tracked repo defect. The repo declares `requires-python >= 3.10`; the fact that the current shell is running Python 3.9.6 is real, but it belongs in local setup notes rather than the canonical bug count.

- `M19`, `M21`, `M36`, and `M38` are version-specific future-compatibility claims that I did not retain as current bugs. In the current environment, `_fmt(np.float64(...))` still works, installed `shap` does not currently violate the declared Python floor, and the local scikit-learn install does not emit the claimed `penalty='l1'` deprecation.

- `L1`, `L3`, `L7`, `L14`, and `M42` were duplicate or severity-upgrade placeholders in the previous report and are merged into the retained issues above.

- `L4`, `L6`, `L10`, `L11`, `L13`, `L17` through `L31`, `M40`, and `M43` are either style-only cleanup, convention debt, or too low-value to keep in the primary open-bug list after verification. Several are real but not worth giving equal weight to the current correctness, integrity, and reproducibility problems.

## Priority Order

If work resumes from this file, the next useful order is:

1. close the silent-fallback/data-integrity issues in `sequence_builder.py`,
2. harden calibration and experiment scoring/error boundaries,
3. clean up dependency/reproducibility risk around `requirements.txt` and `nfl_data_py`,
4. add coverage for `run_ab_experiment.py`, `verify_data()`, and the enabled MLflow path.
