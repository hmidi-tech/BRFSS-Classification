# CS-433 Project 1 — Fall 2026

NumPy-only classification of self-reported myocardial infarction / coronary
heart disease in BRFSS survey data. This does not establish prospective clinical
diagnosis. Confirmed author order: Mohamed Amine Hmidi, Mohamed Yebari,
and Omar Marouane Bensaid, all affiliated with EPFL.

**Draft for student review.** The requester authorized unrestricted AI assistance;
the applicable staff policy and acknowledgment requirement remain independently
unverified. See [the policy review](docs/ai_usage_policy_review.md). No upload,
push, publication, competition acceptance or platform score is claimed. Written
competition rules specify accuracy but the observed leaderboard favors F1;
our provisional local objective is F1. A public commit permalink is unresolved.

## Files and data

- `implementations.py`: six mandatory methods, with official mathematical conventions.
- `data_io.py`, `preprocessing.py`, `models.py`, `evaluation.py`: validated loading,
  fold-fitted preprocessing, NumPy optimizers, metrics and stratification.
- `experiments.py`: bounded comparison, frozen selection and three-fold diagnostic.
- `train_final.py`, `run.py`: all-label refit and stored-weight inference.
- `config/selected_model.json`: frozen choices and dataset fingerprints.
- `artifacts/final_model.npz` and `.json`: essential weights and full fitted
  preprocessing state; both accompany inference.
- `results/`: actual measurements, audits, ablations and reproduction evidence.
- `docs/`: requirements, methods, protocol and policy review.
- `report/`: LaTeX source, measurement-bound tables, figures and compiled draft.
- `tests/`: independent analytical/integration checks.
- `grading_tests/`, `helpers.py`, `latex-example-paper/`: unchanged supplied files.

Place `x_train.csv`, `y_train.csv`, `x_test.csv` and `sample_submission.csv` in
`dataset/`. The archive used `data/dataset/`; pass that path to `--data-dir` if
appropriate. Actual sizes are 328,135 training and 109,379 test rows, 321
features, and 28,975 positives (8.8302%). Raw CSVs/ZIPs and disposable caches
are ignored by Git. Caches are regenerated from CSVs and keyed by source SHA-256.
There is no external-data download.

## Runtime and prediction reproduction

Training/inference require Python's standard library and NumPy only. Executed:
Python 3.12.14, NumPy 2.5.3, Linux/WSL2, four OpenBLAS threads. Matplotlib is used
only for figures. The legacy grader's Python 3.9 / NumPy 1.23.1 was not executed.

Tested from the project root:

```bash
OPENBLAS_NUM_THREADS=4 python run.py --data-dir dataset --output submission.csv --seed 433
```

The CSV already exists: choose a new output filename for another run. Existing
outputs are preserved. A second execution from a fresh source directory and CSV
cache, in the same ml26 environment, produced the identical SHA-256:

```text
e8ef72c62f514931ed2d66f28fcb29201756c66d6d90f4c6fa9a73b4df0b0665
```

The CSV has 109,379 rows: 14,925 positive and 94,454 negative predictions,
original test IDs/order and exact `Id,Prediction` header. The official helper
writes it; an independent readback validates the result. Sidecars `.json` and
`.sha256` record checks. See `results/reproducibility.json`. This is fresh-directory
reproduction in the same environment, not an independently provisioned environment.
Different BLAS backends' bitwise identity is unverified.

Executed all-label training commands:

```bash
OPENBLAS_NUM_THREADS=4 python train_final.py --data-dir dataset
OPENBLAS_NUM_THREADS=4 python train_final.py --data-dir dataset --model artifacts/retrained_model.npz
```

Training refuses to replace existing weights. The second path preserves the
verified model while checking a separate refit. Inference uses the saved model,
not notebook state. It needs test inputs and both model files, not cached training
data. Final preprocessing has 1,183 design columns versus 1,172 during selection,
because all-label fitting discovers more categories.

## Model and evaluation

Seed 433 gives stratified development/tuning/untouched holdout sizes
196,881 / 65,627 / 65,627, saved in `artifacts/split_indices.npz`. Ten planned
configurations compare mean/median fill, missing indicators, categorical coding,
logistic L2 strengths, linear regression and ridge. Fit values, scales and
categories use only each training partition.

Exclude `Id`, `SEQNO`, `_PSU`, raw `IDATE`, and `HAREHAB1`, a documented
target-dependent rehabilitation question. No universal recoding of survey codes
such as 88/99 is applied. The categorical heuristic reference-encodes integer
columns with at most 12 observed fitting levels; unseen levels use the reference.
Higher-cardinality survey coding remains an explicit limitation.

Selection is recorded before the holdout is opened. Within 0.002 of maximum
tuning F1, prefer fewer features, then stronger L2. Selected: median fill,
missing indicators, categorical coding, logistic `lambda=0.0001`, threshold 0.20.
The optional optimizer is NumPy L-BFGS, history ten, Armijo search, 120-iteration
cap and `1e-5` gradient target; its intercept is unpenalized. Mandatory functions
retain official all-coordinate penalties. Selected and final fits stop at the
iteration cap; convergence is not claimed.

| Evaluation | Accuracy | F1 | Recall | Balanced accuracy |
|---|---:|---:|---:|---:|
| Tuning, threshold 0.20 | 0.8722 | 0.4321 | 0.5506 | 0.7270 |
| Holdout, threshold 0.20 | 0.8685 | 0.4225 | 0.5446 | 0.7223 |
| Same holdout model, threshold 0.50 | 0.9142 | 0.2444 | 0.1572 | 0.5723 |
| Majority holdout baseline | 0.9117 | 0.0000 | 0.0000 | 0.5000 |

At common tuning threshold 0.20, categorical coding adds 0.01465 F1; reducing
L2 from `1e-3` to `1e-4` adds 0.00360. These are selection-data comparisons.
Three-fold fixed-configuration CV gives F1 0.43101 ± 0.00041 (sample SD), with
processing refit and thresholds calibrated within each outer training fold.
Family selection is not nested: CV is a consistency check. Ablation improvements
were not separately established across folds. See [the protocol](docs/experimental_protocol.md).

## Experiments, figures and tests

Executed search and analysis commands:

```bash
OPENBLAS_NUM_THREADS=4 python experiments.py --data-dir dataset --seed 433 --cross-validate
OPENBLAS_NUM_THREADS=4 python audit_data.py
python summarize_results.py
python build_report.py
```

The search preserves existing selected configurations. Reproduction in an
already populated checkout needs a separate working copy, or the documented
`--output-dir`, `--artifact-dir`, and `--config-dir` options. Do not discard the
verified model. `--help` describes those options.

`results/experiment_results.jsonl` preserves start/success/failure events, source
and data hashes, timestamps, parameters, metrics, termination and memory/runtime.
`experiment_results.csv` is the readable leaderboard; `ablation_results.csv`
uses a common threshold. Git commit fields are null because no valid repository
existed at experiment time.

The public grader requires pytest, GitPython and Black as development-only
dependencies. Its additional ML packages are not permitted prediction dependencies.
GitPython 3.1.18 was installed only under `/tmp`, leaving ml26 unchanged. Local
test command (the temporary dependency directory exists only in this session):

```bash
PYTHONPATH=.:/tmp/cs433-grader-deps OPENBLAS_NUM_THREADS=4 python -m pytest tests grading_tests --github_link "$PWD" -q
```

With normally installed grader dependencies, omit the temporary path. A local
path necessarily fails the GitHub permalink-format check; test the actual public
commit link before submission. Independent checks cover gradients, penalties,
zero iterations, input mutation, SGD sampling, extreme logits, singular data,
metrics, leakage boundaries, serialization and submission readback.

## Report and human review

`report/report.tex` uses an unchanged official IEEEtran class. Tables bind to
saved results through `build_report.py`. Author order and EPFL affiliation were
confirmed; see `results/report_compilation.log` and the final audit for layout checks.
The original example PDF is not our report. Dependencies used only for PDF
compilation/inspection live under `/tmp` and are not inference dependencies.

Before upload, the team must independently verify the
staff AI-use policy, resolve the metric conflict, review all generated work,
provide a public commit permalink and approve the precise CSV. No contributions
are assigned from the supplied biographies. The five daily competition
submissions are shared by the team. Matching a best accepted competition entry
remains unverified until an actual approved upload occurs.
