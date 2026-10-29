# Final integrity and reproducibility audit

**State: Partially ready, with specific remaining issues.** Technical artifacts are available for final student review; mandatory course-policy/public-link requirements remain unverified. No upload, push or final submission occurred.

Audit time: 2026-10-29T00:46:49.391441+01:00. Deadline: October 29, 2026, 16:00 Europe/Zurich. Confirmed author order: Mohamed Amine Hmidi, Mohamed Yebari, Omar Marouane Bensaid; affiliation EPFL.

## Verified implementation and tests

All six mandatory interfaces have meaningful docstrings, correct signatures and passing official numerical oracles. Mandatory methods preserve all-coordinate regularization, final-state unregularized losses and 1D weight outputs. Independent gradient, SGD, edge-case, mutation, preprocessing, metric and CSV checks passed.

Final test result: **46 passed, 1 failed** (19 independent and 27 public passes). The only failure is the real GitHub-commit-permalink check because a local path was supplied. Public tests, helpers and the official source clone were not modified. All 13 supplied reference files match current official Git blobs. The legacy grader environment was not executed.

Exact test command:

```bash
PYTHONPATH=.:/tmp/cs433-grader-deps OPENBLAS_NUM_THREADS=4 python -m pytest tests grading_tests --github_link "$PWD" -q
```

The Black CLI stalled; direct Black formatting and the public formatting test passed. Its session-owned stalled processes were stopped. Initial failures from missing GitPython, README and run.py were resolved; logs preserve those failures. Runtime source imports contain only NumPy, standard-library modules and local project modules. PDF/formatting/testing tools are separate.

## Data and selection

Train 328,135 × 321; test 109,379 × 321. Positive 28,975 (8.8302%), negative 299,160. Raw missingness: 44.7880% / 44.7852% train/test. Schemas, IDs, row lengths and label ordering pass. No raw full-feature duplicate rows were found, which does not guarantee person-level independence.

HAREHAB1 is excluded: CDC metadata establishes a heart-attack-dependent question, and all 773 available training values belong to positives. IDs and sequence keys are excluded. No global survey-sentinel replacement is performed. Detailed cardinality, scales, correlation and missingness findings are in `docs/dataset_summary.md`.

Ten genuinely completed comparisons on the fixed tuning partition:

| Model | L2 lambda | Features | Threshold | Tuning accuracy | Tuning F1 |
|---|---:|---:|---:|---:|---:|
| A3_l2_0 | 0 | 1172 | 0.21 | 0.8763 | 0.4327 |
| A3_l2_0.0001 | 0.0001 | 1172 | 0.20 | 0.8722 | 0.4321 |
| A3_categories | 0.001 | 1172 | 0.20 | 0.8719 | 0.4285 |
| A2_missing | 0.001 | 552 | 0.20 | 0.8671 | 0.4139 |
| A1_median | 0.001 | 318 | 0.18 | 0.8555 | 0.4136 |
| A0_mean | 0.001 | 318 | 0.21 | 0.8724 | 0.4133 |
| A3_l2_0.01 | 0.01 | 1172 | 0.19 | 0.8660 | 0.4099 |
| linear_0 | 0 | 552 | 0.22 | 0.8653 | 0.4057 |
| linear_0.001 | 0.001 | 552 | 0.22 | 0.8666 | 0.4049 |
| A3_l2_0.1 | 0.1 | 1172 | 0.14 | 0.8319 | 0.3751 |

Selected model: median fill, missing indicators, small-integer categorical reference coding, L2 logistic lambda 1e-4, threshold 0.20, seed 433. The predeclared 0.002 F1 tolerance chooses regularization over the slightly higher-scoring unregularized model. The final refit has 1,183 columns versus 1,172 during selection, because fitted categories expand on all labeled rows. Optional logistic intercept is unpenalized; mandatory methods keep the official convention.

Untouched holdout: accuracy 0.868530, F1 0.422490, balanced accuracy 0.722255, precision 0.345107, recall 0.544607, specificity 0.899903. Confusion counts: TP 3156, FP 5989, FN 2639, TN 53843. ROC-AUC 0.858494; AP 0.386129.

Three-fold fixed-configuration CV: F1 0.431010 ± 0.000407 sample SD. It is not nested family selection; preprocessing and threshold calibration respect each outer fold. Conditional IID-row holdout F1 interval: [0.4125131342695292, 0.43213105199041973]; survey design and model-selection variability are ignored.

At common threshold 0.20: mean→median adds 0.00023 F1, missing indicators add 0.00075, categorical encoding adds 0.01465, and reducing lambda 1e-3→1e-4 adds 0.00360. These are tuning effects; ablation consistency across folds is unverified. Compared with threshold 0.50, the selected holdout threshold adds 0.17809 F1 and loses 0.04564 accuracy.

Final all-label fit: 11.29 seconds measured, 120 iterations, gradient infinity norm 0.00021080. It exhausted the iteration budget; convergence is not claimed. No class weighting, ensemble or additional model family is implied.

## CSV and reproduction

`submission.csv` has 109,379 ordered predictions, header `Id,Prediction`, labels strictly -1/+1, 14,925 positives and 94,454 negatives. No missing/extra rows or duplicate IDs. Written by the original helper and independently verified.

SHA-256:

```text
e8ef72c62f514931ed2d66f28fcb29201756c66d6d90f4c6fa9a73b4df0b0665
```

Tested command (choose a fresh output name for another run because outputs are preserved):

```bash
OPENBLAS_NUM_THREADS=4 python run.py --data-dir dataset --output submission.csv --seed 433
```

A second fresh-directory/fresh-cache inference produced the exact hash in the same ml26 environment. A separate all-data retraining produced exactly equal weights and another identical CSV. All fitted transformation parameters are saved with weights. No cross-backend or independently provisioned-environment identity is claimed. The CSV has not been uploaded; matching a best accepted AIcrowd entry remains unverified.

## Report

Source `report/report.tex`; compiled draft `report/report.pdf`, SHA-256 `ef4fbad840fe42ac4f79ad167638f5f72568828e99f00176d1215beb3898cc3e`. Two body pages and a references-only third page; no appendix. All pages rendered and visually inspected. Official 10pt IEEEtran class unchanged; no overfull boxes, no font warnings and no Type3 figure fonts. One underfull line warning remains. Author order/EPFL affiliation confirmed. Public commit link, course AI policy and competition metric remain explicitly unresolved.

## Remaining human actions

- Independently verify Fall 2026 staff AI policy and required acknowledgment; team authorization alone is documented.
- Resolve written accuracy versus observed F1 leaderboard contradiction before competition-specific submission.
- Publish only after explicit approval, supply a real final public GitHub commit permalink and execute the grader against it.
- All three students must review and approve the work; no individual contribution allocation is inferred.
- Obtain explicit approval for any external upload, observe the team daily budget, record actual acceptance/score, then verify best-entry identity.

No mandatory criterion is waived by technical success. Treat the work as a reviewable draft until the remaining course and external-submission requirements are resolved. Local independent repository packaging is recorded in `results/package_manifest.json`; experiment-time commit fields correctly remain null.
