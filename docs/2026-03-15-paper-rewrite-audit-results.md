# AP Research Metrics Audit

- Doc Type: Results
- Topic: Paper Rewrite Audit
- Topic Slug: paper-rewrite-audit
- Date: 2026-03-15
- Status: Complete

## Scope

This document extracts counts, metrics, tables, and model settings from repository outputs and code, not from paper prose.

Primary numeric sources:

- `data/features/train.csv`
- `data/features/test.csv`
- `results/ab_experiment/predictions_with_spread.csv`
- `results/ab_experiment/predictions_without_spread.csv`
- `results/test/predictions.csv`
- `results/ab_experiment/lr_coefs_with_spread.json`

Primary code/config sources:

- `src/features/pipeline.py`
- `src/models/sequence_builder.py`
- `src/models/logistic_model.py`
- `src/models/xgboost_model.py`
- `src/models/lstm_model.py`
- `src/models/lstm_config.py`
- `src/models/unified_trainer.py`
- `src/models/cv_splitter.py`

Method notes:

- All CV/test metrics below were recomputed directly from the saved prediction CSVs.
- Bootstrap confidence intervals were recomputed as paired nonparametric bootstrap CIs with 10,000 resamples, seed `42`, percentile interval.
- Pre-calibration test probability ranges were regenerated on 2026-03-15 by rerunning the repository’s final-model prediction path from `src.models.evaluate_test_set`.

## Important Mismatches

- The current raw CV prediction CSV gives LSTM CV AUC `0.6371631809554219`, not `0.6407`.
- The current raw calibrated test prediction CSV gives LSTM test AUC `0.5240144386122539`, not `0.5202`.
- The current raw calibrated test prediction CSV gives simple-average test ensemble AUC `0.5656195395891298`, not `0.5685`.
- The current raw outputs do not reproduce the stale flat LSTM-exclusive count `65 (12 upsets, 53 non-upsets)`.
- The current raw outputs support:
  - `72 (15 upsets, 57 non-upsets)` if you apply the documented single global CV base-rate threshold `0.29690189328743544`.
  - `62 (13 upsets, 49 non-upsets)` if you use the stored `lr_pred`/`xgb_pred`/`lstm_pred` columns in `predictions_with_spread.csv`.
- The stale `65 / 12 / 53` figure appears in markdown summaries, but not in the current raw CSV outputs.

## 1. Dataset Counts

Overall:

| Split | Total rows | Labeled games | Upsets | Upset rate | Excluded by spread < 3 | Other unlabeled |
|---|---:|---:|---:|---|---:|---:|
| Train (2005-2022) | 4351 | 3495 | 1061 | `1061/3495 = 0.303576537911301859799713876967` | 849 | 7 |
| Test (2023-2025) | 768 | 558 | 159 | `159/558 = 0.284946236559139784946236559140` | 209 | 1 |

Training seasons:

| Season | Total rows | Labeled games | Upsets | Upset rate | Excluded by spread < 3 | Other unlabeled |
|---|---:|---:|---:|---|---:|---:|
| 2005 | 239 | 196 | 45 | `45/196 = 0.229591836734693877551020408163` | 43 | 0 |
| 2006 | 240 | 195 | 77 | `77/195 = 0.394871794871794871794871794872` | 45 | 0 |
| 2007 | 240 | 201 | 58 | `58/201 = 0.288557213930348258706467661692` | 39 | 0 |
| 2008 | 240 | 201 | 60 | `60/201 = 0.298507462686567164179104477612` | 38 | 1 |
| 2009 | 240 | 209 | 62 | `62/209 = 0.296650717703349282296650717703` | 31 | 0 |
| 2010 | 240 | 197 | 65 | `65/197 = 0.329949238578680203045685279188` | 43 | 0 |
| 2011 | 240 | 200 | 57 | `57/200 = 0.285000000000000000000000000000` | 40 | 0 |
| 2012 | 240 | 197 | 63 | `63/197 = 0.319796954314720812182741116751` | 42 | 1 |
| 2013 | 240 | 179 | 47 | `47/179 = 0.262569832402234636871508379888` | 60 | 1 |
| 2014 | 240 | 189 | 56 | `56/189 = 0.296296296296296296296296296296` | 50 | 1 |
| 2015 | 240 | 185 | 64 | `64/185 = 0.345945945945945945945945945946` | 55 | 0 |
| 2016 | 240 | 184 | 62 | `62/184 = 0.336956521739130434782608695652` | 55 | 1 |
| 2017 | 241 | 187 | 48 | `48/187 = 0.256684491978609625668449197861` | 54 | 0 |
| 2018 | 240 | 192 | 57 | `57/192 = 0.296875000000000000000000000000` | 48 | 0 |
| 2019 | 240 | 194 | 63 | `63/194 = 0.324742268041237113402061855670` | 46 | 0 |
| 2020 | 240 | 193 | 54 | `54/193 = 0.279792746113989637305699481865` | 46 | 1 |
| 2021 | 256 | 209 | 68 | `68/209 = 0.325358851674641148325358851675` | 46 | 1 |
| 2022 | 255 | 187 | 55 | `55/187 = 0.294117647058823529411764705882` | 68 | 0 |

Test seasons:

| Season | Total rows | Labeled games | Upsets | Upset rate | Excluded by spread < 3 | Other unlabeled |
|---|---:|---:|---:|---|---:|---:|
| 2023 | 256 | 185 | 55 | `55/185 = 0.297297297297297297297297297297` | 71 | 0 |
| 2024 | 256 | 192 | 45 | `45/192 = 0.234375000000000000000000000000` | 64 | 0 |
| 2025 | 256 | 181 | 59 | `59/181 = 0.325966850828729281767955801105` | 74 | 1 |

Spread buckets on evaluation sets:

| Evaluation set | Small 3-6.5 | Medium 7-13.5 | Large 14+ |
|---|---:|---:|---:|
| CV predictions (`results/ab_experiment/predictions_with_spread.csv`) | 700 | 402 | 60 |
| Test predictions (`results/test/predictions.csv`) | 360 | 175 | 23 |

## 2. Feature Counts

Counts:

| Model | Count |
|---|---:|
| LR | 46 |
| XGBoost | 70 |
| LSTM sequence | `14 × 8 = 112` |
| LSTM matchup context | 10 |
| LSTM total if flattened | 122 |

No-spread counts:

| Model | No-spread count |
|---|---:|
| LR | 42 |
| XGBoost | 66 |
| LSTM matchup context | 8 |

LR feature names:

```text
underdog_pass_epa_roll3
underdog_rush_epa_roll3
underdog_success_rate_roll3
underdog_cpoe_roll3
underdog_turnover_margin_roll3
favorite_pass_epa_roll3
favorite_rush_epa_roll3
favorite_success_rate_roll3
favorite_cpoe_roll3
favorite_turnover_margin_roll3
pass_epa_diff
rush_epa_diff
success_rate_diff
cpoe_diff
turnover_margin_diff
underdog_total_epa_std_roll3
favorite_total_epa_std_roll3
total_epa_std_diff
underdog_success_rate_std_roll3
favorite_success_rate_std_roll3
success_rate_std_diff
underdog_total_epa_trend
favorite_total_epa_trend
total_epa_trend_diff
underdog_success_rate_trend
favorite_success_rate_trend
success_rate_trend_diff
underdog_rest_days
favorite_rest_days
rest_days_diff
short_week_game
divisional_game
home_implied_points
away_implied_points
spread_magnitude
total_line
underdog_elo
favorite_elo
elo_diff
temperature
wind_speed
is_dome
temperature_missing
wind_speed_missing
underdog_is_home
week_number
```

XGBoost feature names:

- XGBoost uses all 46 LR features above.
- XGBoost-only lag features:

```text
underdog_last1_total_epa
underdog_last1_success_rate
underdog_last1_turnover_margin
underdog_last1_margin
favorite_last1_total_epa
favorite_last1_success_rate
favorite_last1_turnover_margin
favorite_last1_margin
underdog_last2_total_epa
underdog_last2_success_rate
underdog_last2_turnover_margin
underdog_last2_margin
favorite_last2_total_epa
favorite_last2_success_rate
favorite_last2_turnover_margin
favorite_last2_margin
underdog_last3_total_epa
underdog_last3_success_rate
underdog_last3_turnover_margin
underdog_last3_margin
favorite_last3_total_epa
favorite_last3_success_rate
favorite_last3_turnover_margin
favorite_last3_margin
```

LSTM feature names:

Sequence features per timestep:

```text
total_epa
pass_epa
rush_epa
success_rate
cpoe
turnover_margin
points_scored
points_allowed
point_diff
opponent_elo
win
was_home
days_since_last_game
short_week
```

Matchup context features:

```text
spread_magnitude
total_line
underdog_elo
favorite_elo
underdog_is_home
underdog_rest_days
favorite_rest_days
week_number
divisional_game
is_dome
```

## 3. Cross-Validation Results (6-fold)

CV set summary:

- Total CV predictions: 1162
- CV upset count: 345
- CV upset rate: `345/1162 = 0.296901893287435456110154905336`
- Documented global CV disagreement threshold: `0.29690189328743544`

CV metrics from `results/ab_experiment/predictions_with_spread.csv`:

| Model | AUC-ROC | Brier Score | Log Loss |
|---|---|---|---|
| LR | `0.6496620722686393` | `0.19740239439310345` | `0.5807228951785454` |
| XGB | `0.6376953506111082` | `0.1991429029722572` | `0.58549320314039` |
| LSTM | `0.6371631809554219` | `0.1997408165131562` | `0.5858379064350112` |

Pairwise AUC differences, 10,000 paired bootstrap resamples:

| Comparison | Point estimate | Bootstrap mean | 95% CI |
|---|---|---|---|
| LR - XGB | `0.011966721657531099` | `0.012108636614825849` | [`-0.006679067061154774`, `0.030839809933984476`] |
| LR - LSTM | `0.012498891313217353` | `0.012425701255141926` | [`-0.012347728210737001`, `0.03724002957791745`] |
| XGB - LSTM | `0.0005321696556862543` | `0.0003170646403160775` | [`-0.027766217966179167`, `0.028200941143451327`] |

CV probability correlation matrix:

| Pair | Correlation |
|---|---|
| LR-XGB | `0.8742252247420503` |
| LR-LSTM | `0.7643115000863674` |
| XGB-LSTM | `0.6736875321245558` |

Fold breakdown:

| Fold | Training seasons | Validation season | Train games | Validation games | Validation upsets | Validation upset rate |
|---|---|---:|---:|---:|---:|---|
| 1 | 2005-2016 | 2017 | 2333 | 187 | 48 | `48/187 = 0.256684491978609625668449197861` |
| 2 | 2005-2017 | 2018 | 2520 | 192 | 57 | `57/192 = 0.296875000000000000000000000000` |
| 3 | 2005-2018 | 2019 | 2712 | 194 | 63 | `63/194 = 0.324742268041237113402061855670` |
| 4 | 2005-2019 | 2020 | 2906 | 193 | 54 | `54/193 = 0.279792746113989637305699481865` |
| 5 | 2005-2020 | 2021 | 3099 | 209 | 68 | `68/209 = 0.325358851674641148325358851675` |
| 6 | 2005-2021 | 2022 | 3308 | 187 | 55 | `55/187 = 0.294117647058823529411764705882` |

## 4. Test Set Results

Test set summary:

- Total test predictions: 558
- Test upset count: 159
- Test upset rate: `159/558 = 0.284946236559139784946236559140`
- Baseline Brier: `0.20375187882992257`

Calibrated test metrics from `results/test/predictions.csv`:

| Model | AUC-ROC | Brier Score | Log Loss |
|---|---|---|---|
| LR | `0.5621601172743179` | `0.20262917867520738` | `0.5941795001652119` |
| XGB | `0.5755268674831734` | `0.2012976831227052` | `0.5914684911995604` |
| LSTM | `0.5240144386122539` | `0.20749745070876935` | `0.6054549713292178` |

Generalization gaps (`CV AUC - Test AUC`):

| Model | Gap |
|---|---|
| LR | `0.08750195499432134` |
| XGB | `0.06216848312793477` |
| LSTM | `0.11314874234316796` |

Per-season test AUC:

| Season | Games | Upsets | Upset rate | LR AUC | XGB AUC | LSTM AUC |
|---|---:|---:|---|---|---|---|
| 2023 | 185 | 55 | `55/185 = 0.297297297297297297297297297297` | `0.5118881118881119` | `0.5209790209790209` | `0.4692307692307692` |
| 2024 | 192 | 45 | `45/192 = 0.234375000000000000000000000000` | `0.5522297808012093` | `0.5535903250188964` | `0.5490551776266062` |
| 2025 | 181 | 59 | `59/181 = 0.325966850828729281767955801105` | `0.616699083078633` | `0.6394831897749376` | `0.5558488469019172` |

Test probability correlation matrix:

| Pair | Correlation |
|---|---|
| LR-XGB | `0.8776265391553654` |
| LR-LSTM | `0.4289760444961724` |
| XGB-LSTM | `0.408427278346498` |

Pre-calibration vs post-calibration probability ranges:

| Model | Pre-calibration range | Post-calibration range |
|---|---|---|
| LR | [`0.034478984821464234`, `0.5192289344189776`] | [`0.1942155149676448`, `0.4349105659532186`] |
| XGB | [`0.08432800322771072`, `0.6433864831924438`] | [`0.2161089017573103`, `0.497659295277168`] |
| LSTM | [`0.0012638866901397705`, `0.9259230494499207`] | [`0.2179035435907117`, `0.5475030368391803`] |

## 5. Spread Ablation

CV AUC with and without spread:

| Model | With spread | Without spread | Delta (`no_spread - with_spread`) |
|---|---|---|---|
| LR | `0.6496620722686393` | `0.5706703563762794` | `-0.07899171589235987` |
| XGB | `0.6376953506111082` | `0.566150462100651` | `-0.07154488851045715` |
| LSTM | `0.6371631809554219` | `0.568162063399145` | `-0.06900111755627691` |

Bootstrap CIs on AUC deltas:

| Model | Point estimate | Bootstrap mean | 95% CI |
|---|---|---|---|
| LR | `-0.07899171589235987` | `-0.07880369020121725` | [`-0.10764758628136996`, `-0.050254940651768575`] |
| XGB | `-0.07154488851045715` | `-0.07138969328973724` | [`-0.10149758538554216`, `-0.04141966118994702`] |
| LSTM | `-0.06900111755627691` | `-0.06902577856135786` | [`-0.10316290654607435`, `-0.03579845674934776`] |

Delta-of-deltas:

| Comparison | Point estimate | Bootstrap mean | 95% CI |
|---|---|---|---|
| LSTM delta - LR delta | `0.009990598336082956` | `0.009777911639859395` | [`-0.022558932434816884`, `0.04249719312113592`] |
| LSTM delta - XGB delta | `0.0025437709541802356` | `0.002363914728379394` | [`-0.034481018832379896`, `0.039653661367424764`] |

No-spread probability correlation matrix:

| Pair | Correlation |
|---|---|
| LR-XGB | `0.7415941445772035` |
| LR-LSTM | `0.5155617134036221` |
| XGB-LSTM | `0.37229439203590964` |

Agreement and LSTM exclusives, global threshold `0.29690189328743544`:

| Metric | With spread | Without spread |
|---|---|---|
| LR-XGB agreement | `1011/1162 = 0.8700516351118761` | `902/1162 = 0.7762478485370051` |
| LR-LSTM agreement | `954/1162 = 0.8209982788296041` | `796/1162 = 0.685025817555938` |
| XGB-LSTM agreement | `919/1162 = 0.7908777969018933` | `736/1162 = 0.6333907056798623` |
| All-three agreement | `861/1162 = 0.7409638554216867` | `636/1162 = 0.5473321858864028` |
| `only_lstm` count/rate | `72/1162 = 0.06196213425129088` | `125/1162 = 0.10757314974182444` |

## 6. Disagreement Table (CV, with spread)

This table uses the documented single global CV base-rate threshold `0.29690189328743544`.

| Category | N | Percentage | Upsets | Upset rate | Mean LR prob | Mean XGB prob | Mean LSTM prob |
|---|---:|---|---:|---|---|---|---|
| all_correct | 528 | `528/1162 = 0.454388984509466437177280550775` | 196 | `196/528 = 0.371212121212121212121212121212` | `0.2592271580052272` | `0.2652443358604091` | `0.25808550910071726` |
| all_wrong | 333 | `333/1162 = 0.286574870912220309810671256454` | 68 | `68/333 = 0.204204204204204204204204204204` | `0.34997664531476835` | `0.3506342290646142` | `0.3533384035098123` |
| only_lr | 31 | `31/1162 = 0.026678141135972461273666092943` | 7 | `7/31 = 0.225806451612903225806451612903` | `0.2920936745757443` | `0.3195238204733018` | `0.32028055527517874` |
| only_xgb | 48 | `48/1162 = 0.041308089500860585197934595525` | 11 | `11/48 = 0.229166666666666666666666666667` | `0.31739079565695916` | `0.28626428203036386` | `0.34406098164618015` |
| only_lstm | 72 | `72/1162 = 0.061962134251290877796901893287` | 15 | `15/72 = 0.208333333333333333333333333333` | `0.32605149923922827` | `0.3453789231263929` | `0.2355210818350315` |
| lr_xgb | 78 | `78/1162 = 0.067125645438898450946643717728` | 21 | `21/78 = 0.269230769230769230769230769231` | `0.2853277870739152` | `0.27541445252987057` | `0.30684876814484596` |
| lr_lstm | 45 | `45/1162 = 0.038726333907056798623063683305` | 17 | `17/45 = 0.377777777777777777777777777778` | `0.28975291364818023` | `0.308234836657842` | `0.28053041835212045` |
| xgb_lstm | 27 | `27/1162 = 0.023235800344234079173838209983` | 10 | `10/27 = 0.370370370370370370370370370370` | `0.31053130397782797` | `0.29211298900621907` | `0.257460058528792` |

## 7. Spread-Stratified Disagreement (CV, with spread)

This section also uses the documented single global CV base-rate threshold `0.29690189328743544`.

Bucket totals:

| Bucket | Total games | Upsets | Upset rate |
|---|---:|---:|---|
| Small 3-6.5 | 700 | 263 | `263/700 = 0.375714285714285714285714285714` |
| Medium 7-13.5 | 402 | 76 | `76/402 = 0.189054726368159203980099502488` |
| Large 14+ | 60 | 6 | `6/60 = 0.100000000000000000000000000000` |

Category percentages within each bucket:

| Category | Small 3-6.5 | Medium 7-13.5 | Large 14+ |
|---|---|---|---|
| all_correct | `215/700 = 0.307142857142857142857142857143` | `259/402 = 0.644278606965174129353233830846` | `54/60 = 0.900000000000000000000000000000` |
| all_wrong | `271/700 = 0.387142857142857142857142857143` | `56/402 = 0.139303482587064676616915422886` | `6/60 = 0.100000000000000000000000000000` |
| only_lr | `27/700 = 0.038571428571428571428571428571` | `4/402 = 0.009950248756218905472636815920` | `0/60 = 0.000000000000000000000000000000` |
| only_xgb | `39/700 = 0.055714285714285714285714285714` | `9/402 = 0.022388059701492537313432835821` | `0/60 = 0.000000000000000000000000000000` |
| only_lstm | `60/700 = 0.085714285714285714285714285714` | `12/402 = 0.029850746268656716417910447761` | `0/60 = 0.000000000000000000000000000000` |
| lr_xgb | `38/700 = 0.054285714285714285714285714286` | `40/402 = 0.099502487562189054726368159204` | `0/60 = 0.000000000000000000000000000000` |
| lr_lstm | `29/700 = 0.041428571428571428571428571429` | `16/402 = 0.039800995024875621890547263682` | `0/60 = 0.000000000000000000000000000000` |
| xgb_lstm | `21/700 = 0.030000000000000000000000000000` | `6/402 = 0.014925373134328358208955223881` | `0/60 = 0.000000000000000000000000000000` |

LSTM exclusives:

| Bucket | Total | Upsets caught | Non-upsets rejected |
|---|---:|---:|---:|
| Small 3-6.5 | 60 | 5 | 55 |
| Medium 7-13.5 | 12 | 10 | 2 |
| Large 14+ | 0 | 0 | 0 |
| Overall | 72 | 15 | 57 |

All-wrong decomposition:

| Bucket | Total | False alarms | Missed upsets |
|---|---:|---:|---:|
| Small 3-6.5 | 271 | 263 | 8 |
| Medium 7-13.5 | 56 | 2 | 54 |
| Large 14+ | 6 | 0 | 6 |
| Overall | 333 | 265 | 68 |

### LSTM-Exclusive Discrepancy Diagnosis

Current raw outputs do not support a single `65 (12 upsets, 53 non-upsets)` answer.

What the current raw artifacts support:

| Reconstruction path | Total `only_lstm` | Upsets | Non-upsets | Bucket split |
|---|---:|---:|---:|---|
| Current raw probabilities + documented global threshold `0.29690189328743544` | 72 | 15 | 57 | `60 small + 12 medium + 0 large` |
| Current stored binary columns `lr_pred` / `xgb_pred` / `lstm_pred` in `predictions_with_spread.csv` | 62 | 13 | 49 | `49 small + 13 medium + 0 large` |
| Stale markdown summaries | 65 | 12 | 53 | `not reproducible from current raw CSVs` |

Conclusion:

- If you want the paper to match the stated method `threshold = base upset rate`, the correct flat `only_lstm` count is `72`, not `65`.
- If you instead want to match the stored binary prediction columns in the CSV, the correct count is `62`, not `65`.
- The `65 / 12 / 53` line is stale prose, not a current raw-output value.

## 8. Top-K Analysis (test set)

Base rate used for lift:

- `159/558 = 0.2849462365591398`

Top-K hit rate and lift:

| K | LR | XGB | LSTM | Ensemble |
|---|---|---|---|---|
| 10 | `5/10`, hit rate `0.5`, lift `1.7547169811320755` | `6/10`, hit rate `0.6`, lift `2.1056603773584905` | `4/10`, hit rate `0.4`, lift `1.4037735849056604` | `5/10`, hit rate `0.5`, lift `1.7547169811320755` |
| 20 | `8/20`, hit rate `0.4`, lift `1.4037735849056604` | `9/20`, hit rate `0.45`, lift `1.579245283018868` | `7/20`, hit rate `0.35`, lift `1.2283018867924527` | `9/20`, hit rate `0.45`, lift `1.579245283018868` |
| 50 | `19/50`, hit rate `0.38`, lift `1.3335849056603774` | `22/50`, hit rate `0.44`, lift `1.5443396226415096` | `14/50`, hit rate `0.28`, lift `0.9826415094339623` | `17/50`, hit rate `0.34`, lift `1.1932075471698111` |

## 9. LR Coefficients

Confirmed `spread_magnitude` coefficient:

- `-0.5387633206071474`

Top 5 standardized coefficients by absolute value:

| Rank | Feature | Coefficient | Absolute value |
|---|---|---|---|
| 1 | spread_magnitude | `-0.5387633206071474` | `0.5387633206071474` |
| 2 | temperature | `0.07718168911620618` | `0.07718168911620618` |
| 3 | favorite_turnover_margin_roll3 | `-0.06938256105795056` | `0.06938256105795056` |
| 4 | underdog_rush_epa_roll3 | `0.06470895946679617` | `0.06470895946679617` |
| 5 | short_week_game | `-0.06378453376240202` | `0.06378453376240202` |

## 10. Ensemble Results

With-spread ensemble AUCs:

| Ensemble | CV AUC | Test AUC |
|---|---|---|
| Simple average of all three | `0.6551251515120437` | `0.5656195395891298` |
| LR+XGB average | `0.6486959879940563` | `0.5707459792896732` |

No-spread ensemble AUCs that were also computable from saved CV outputs:

| Ensemble | CV no-spread AUC |
|---|---|
| Simple average of all three | `0.5808667269792277` |
| LR+XGB average | `0.5711954304365565` |

Conclusion claims check:

- “Ensemble AUC of `0.655` in CV” is directionally close, but the current raw CSV gives `0.6551251515120437`.
- “`0.649` for LR+XGB” is directionally close, but the current raw CSV gives `0.6486959879940563`.

## 11. Hyperparameters

LR:

| Setting | Value |
|---|---|
| `C` | `0.1` |
| `solver` | `saga` |
| `penalty` | `l1` |
| `max_iter` | `1000` |
| `random_state` | `42` |
| preprocessing | `StandardScaler` on all LR features before `LogisticRegression` |

XGBoost:

| Setting | Value |
|---|---|
| `max_depth` | `2` |
| `learning_rate` | `0.03` |
| `n_estimators` | `300` |
| `min_child_weight` | `1` |
| `objective` | `binary:logistic` |
| `eval_metric` | `logloss` |
| `random_state` | `42` |

LSTM:

| Setting | Value |
|---|---|
| sequence features | `14` |
| sequence length | `8` |
| matchup features | `10` |
| hidden size | `64` |
| number of LSTM layers | `3` |
| dropout | `0.25` |
| optimizer | `Adam` |
| learning rate | `0.001` |
| batch size | `64` |
| epochs | `25` |
| patience | `6` in CV training with validation; no early stopping in final `train_final()` |
| encoder | shared unidirectional LSTM for underdog and favorite |
| attention | single shared linear attention layer `nn.Linear(hidden_size, 1)` with softmax over timesteps and mask renormalization |
| dense head | `Concat(team_encoding_A, team_encoding_B, matchup_features) -> Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear -> Sigmoid` |

## 12. Binomial Test

There are three different answers depending on which disagreement definition you use:

| Definition | Upsets / total | Base rate | One-sided `p` for `rate < base_rate` |
|---|---|---|---|
| Current raw probabilities + documented global threshold | `15/72` | `345/1162 = 0.29690189328743544` | `0.061331555270780276` |
| Current stored binary columns in `predictions_with_spread.csv` | `13/62` | `345/1162 = 0.29690189328743544` | `0.08315222785843168` |
| Stale markdown figure | `12/65` | `345/1162 = 0.29690189328743544` | `0.028548753617594777` |

If the paper is going to use the documented single global CV threshold, the correct current raw-output p-value is `0.061331555270780276`, not `0.029`.
