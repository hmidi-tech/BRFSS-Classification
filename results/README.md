# Measurement inventory

The original ten-model measurements, holdout scores, CV aggregates, feature
audit and selection protocol are retained unchanged. They describe the historical
experiment, not an evaluation of newly selected review models.

- `experiment_results.csv`, `leaderboard.json`, `experiment_results.jsonl`:
  original configuration comparisons and fit traces.
- `holdout.json`, `cross_validation.json`, `validation_summary.json`,
  `ablation_results.csv`: original evaluation views, cross-checked by the report builder.
- `selected_threshold_curve.json`: aggregate original tuning curve extracted
  during review; no individual rows, labels or scores are included.
- `review_20261029/`: new regression, reproduction, numerical/fold diagnostics
  and report verification. These are post-selection checks, not new holdout tuning.

Earlier `final_audit.json`, `report_audit.json`, `package_manifest.json`, test logs
and reproduction logs are **historical snapshots**. Their timestamps, old paths,
inventories and null experiment-time Git fields must not be interpreted as
the current HEAD, tracked file list, current test count or published state.
They are retained to preserve provenance and prior failures.

Raw CSVs, candidate checkpoints, saved holdout labels, split indices and generated
competition CSVs are local-only. They are not runtime prerequisites for inference
from the tracked final model. The historical candidate paths in configuration
and logs describe provenance; the final training and inference entry points
do not require them. Current handoff: `docs/final_audit.md`.
