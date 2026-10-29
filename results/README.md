# Saved experimental measurements

This directory contains the measurements required to reproduce
and verify the reported scientific results.

- `experiment_results.csv`, `experiment_results.jsonl`, `leaderboard.json`:
  the original ten-model comparison and recorded measurements.
- `holdout.json`, `cross_validation.json`, `validation_summary.json`:
  evaluation and cross-validation results.
- `ablation_results.csv`, `selected_threshold_curve.json`:
  representation and threshold comparisons.
- `data_audit.json`, `feature_audit.csv`, `correlated_features.csv`:
  dataset diagnostics.
- `protocol.json`, `planned_configurations.json`:
  experimental setup and model-selection plan.
- `review_20261029/diagnostics.json`:
  numerical and categorical robustness diagnostics used by the report builder.
- `reproducibility.json`, `review_20261029/reproducibility.json`:
  saved reproduction evidence.

The raw training data, generated competition predictions, private
candidate checkpoints and split indices are not distributed.
