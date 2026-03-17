# NFL Upset Taxonomy via Multi-Architecture Disagreement

Most NFL prediction research asks *can we predict upsets better?* This project asks a different question: **why do upsets happen, and what can the structure of prediction failure tell us about them?**

## The Idea

We train three architecturally distinct models — logistic regression, XGBoost, and a siamese LSTM — on the same game data, each in its own representation. Then, instead of picking a winner, we study where they agree and disagree.

When all three models get a game right, the signal was readable from every angle. When only one succeeds, its architectural strength points to the mechanism: spread mispricing, non-linear interactions, or temporal momentum. When none of them get it right, the spread tells us whether it was noise or something outside the data entirely.

The disagreement is the finding.

## Project Structure

```
src/data/           Data loading — NFL schedules, betting lines, play-by-play stats
src/features/       Feature engineering and target definition
src/models/         Model implementations, training, and evaluation
src/evaluation/     Metrics, calibration, disagreement analysis, reporting
docs/               Research paper and presentation materials
```
