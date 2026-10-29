# BRFSS Classification — EPFL CS-433 Project 1, Fall 2026

Classification of **self-reported historical myocardial infarction or coronary
heart disease** using linear models implemented in NumPy. This is a formative
course project, not a clinically validated diagnosis or forecast of future disease.

**Authors, in confirmed order:** Mohamed Amine Hmidi, Mohamed Yebari,
Omar Marouane Bensaid. **Affiliation:** École Polytechnique Fédérale de Lausanne
(EPFL). Individual contributions are not inferred from the author order.

The review preserves the original model and F1-oriented predictions. New commits
remain local for student review. No AIcrowd acceptance, platform score or grading
against a published review commit is claimed.

## Data and availability

The official course data contain 328,135 labeled rows, 109,379 test rows and
321 features. There are 28,975 positive labels (8.8302%); training missingness is
44.7880%. Labels are −1/+1. Data and original course materials are **not tracked**.
Obtain them through the [course project](https://github.com/epfml/ML_course/tree/main/projects/project1).

In this workspace the data are in `../dataset/`, relative to this repository:

```text
project1ML/
├── dataset/                 # official CSVs, outside this repository
└── team_repository/        # run commands here
```

Use `--data-dir ../dataset` for this layout. If you instead place official CSVs
in an ignored `dataset/` inside the repository, use `--data-dir dataset`.
Inference needs only `x_test.csv`; training needs `x_train.csv` and `y_train.csv`.
EDA additionally uses `sample_submission.csv`. There is no external-data download.

## Requirements

Training and inference use the Python standard library and NumPy only.
The verified environment is **Python 3.12.14, NumPy 2.5.3, Linux/WSL2**, with
`OPENBLAS_NUM_THREADS=4`. Activate your intended environment before execution;
`requirements.txt` records the tested NumPy version.

Testing and visualization dependencies are separate in `requirements-dev.txt`.
Matplotlib is used only for figures. PDF compilation/inspection tools are not
runtime dependencies. The official legacy Python 3.9 / NumPy 1.23.1 environment
has not been independently tested; the current public tests were run in ml26.

## Reproduce the preserved predictions

From this repository root, the tested invocation is:

```bash
OPENBLAS_NUM_THREADS=4 python run.py --data-dir ../dataset --output submission.csv --seed 433
```

The local `submission.csv` already exists. Choose a **fresh** output path when
repeating the command; the CSV and its `.json`/`.sha256` sidecars are protected
against replacement. For example, use a new directory under ignored `artifacts/tmp/`.

Expected output: exact header `Id,Prediction`, 109,379 rows in original test order,
14,925 positive and 94,454 negative predictions, with labels strictly −1/+1.
Verified CSV SHA-256:

```text
e8ef72c62f514931ed2d66f28fcb29201756c66d6d90f4c6fa9a73b4df0b0665
```

`artifacts/final_model.npz` and `artifacts/final_model.json` are the required
inference inputs: training produced them, and the JSON contains the complete
fitted transformation state. Inference does not need training CSVs, notebooks,
candidate checkpoints, holdout labels or split-index files.
The `candidate_artifact` field in `config/selected_model.json` is a historical
provenance reference; neither inference nor the final refit dereferences it.

Parsed caches are disposable generated outputs, not required model artifacts.
They are keyed by source fingerprint and cache version, with a checked array
checksum. `--cache-dir` can select a new cache without deleting an existing one.
Inference transforms bounded row blocks to limit peak memory.

Fresh-directory inference with test data alone, and a separate full-data refit,
reproduced the exact CSV in the same environment. Cross-machine, cross-NumPy and
cross-BLAS bitwise identity are unverified. See
[reproduction evidence](results/review_20261029/reproducibility.json).

## Refit the frozen model

The following command was executed, using a fresh path to preserve original weights:

```bash
OPENBLAS_NUM_THREADS=4 python train_final.py --data-dir ../dataset --model artifacts/tmp/review_20261029T0242/refitted_model.npz
```

That path now exists; use another fresh path for a new refit. The frozen
configuration intentionally retains its historical float32 optimization matvecs.
New direct optimizer calls default to blocked float64 evaluation; this numerical
audit path does not silently replace the frozen model or its configuration.

## Repository architecture

| Component | Purpose |
|---|---|
| `implementations.py` | Six mandatory methods and official mathematical contracts |
| `data_io.py` | Validated CSV loading, checked caches and local provenance |
| `preprocessing.py` | Training-only fill, scaling, indicators and reference coding |
| `models.py` | NumPy optimization, model dispatch and scores |
| `evaluation.py` | Validated binary metrics, stratification and threshold selection |
| `experiments.py` | Bounded comparison, frozen selection and CV orchestration |
| `train_final.py`, `run.py` | Final all-label training and test-only inference |
| `config/`, `artifacts/final_model.*` | Frozen selection and complete fitted model |
| `results/` | Aggregate measurements and clearly identified historical evidence |
| `report/` | Official class, source-bound tables, figures and compiled PDF |
| `tests/` | Independent analytical, numerical and integration checks |

Official grading tests, original course examples/PDFs, raw CSVs, candidate models,
saved holdout labels, split indices and generated competition CSVs remain local
and ignored. They are not promised as files available in a fresh Git checkout.

## Methods and validation

Mandatory MSE is `||Xw-y||²/(2N)`; SGD uses exactly one observation per update.
Ridge uses `X.T@X + 2*N*lambda*I`. Mandatory logistic methods take 0/1 labels;
all required regularized methods penalize every supplied coefficient and return
the final **unregularized** data loss. See [mathematical details](docs/methods.md).

Seed 433 defines development/tuning/holdout sizes 196,881 / 65,627 / 65,627.
All fitted processing and threshold selection respect their training boundaries.
Exclude IDs, `SEQNO`, `_PSU`, raw `IDATE` and target proxy `HAREHAB1`.
No universal recoding of values such as 88/99 is applied. Integer columns with
at most twelve observed training levels receive optional reference coding;
unseen levels use the fitted reference. Survey semantics remain incomplete.

The retained model uses median fill, indicators, categorical encoding,
L2 logistic regression with `lambda=0.0001` and threshold `0.20`.
NumPy L-BFGS uses history ten and Armijo search. The optional intercept is
unpenalized, separately from the mandatory contracts. Its 120-update final fit
stops above the requested `1e-5` gradient tolerance; convergence is not claimed.
The development design has 1,172 columns; all-label fitting discovers more
categories and produces 1,183, with its own complete saved mapping.

Selection prefers fewer columns, then stronger L2, within 0.002 of maximum
tuning F1. It was frozen before the historical holdout evaluation. This review
does not reuse that opened holdout for selecting a new model.

| Fixed model evaluation | Accuracy | F1 | Recall | Balanced accuracy |
|---|---:|---:|---:|---:|
| Original tuning, threshold 0.20 | 0.8722 | 0.4321 | 0.5506 | 0.7270 |
| Historical holdout, threshold 0.20 | 0.8685 | 0.4225 | 0.5446 | 0.7223 |
| Same holdout scores, threshold 0.50 | 0.9142 | 0.2444 | 0.1572 | 0.5723 |
| Majority holdout baseline | 0.9117 | 0.0000 | 0.0000 | 0.5000 |

Fixed-threshold tuning ablations add 0.01465 F1 for categorical encoding and
0.00360 when reducing L2 from `1e-3` to `1e-4`. Three shared development-only
folds give categorical gains 0.00840, 0.01675 and 0.01183. These are conditional
robustness diagnostics, not causal effects or independent post-selection tests.
The original fixed-configuration CV gives F1 0.43101 ± 0.00041 sample SD;
family selection was not nested. Survey observation weights are not applied.

Written AIcrowd rules specify accuracy. Observed leaderboard ordering alone
does not resolve the Fall 2026 objective. The current CSV remains F1-oriented;
an accuracy-oriented alternative must be selected from development/tuning data,
kept separate and approved by the students. Nothing is uploaded by this code.

## Tests and experiments

Independent checks:

```bash
OPENBLAS_NUM_THREADS=4 python -m pytest tests -q
```

Run the unmodified public tests from the separate course clone, not a copied
grader in the student repository. The exact local verification command was:

```bash
PYTHONPATH=.:/tmp/cs433-grader-deps OPENBLAS_NUM_THREADS=4 python -m pytest --rootdir=. -o cache_dir=artifacts/tmp/review_20261029T0242/pytest_cache ../../ML_course_official/projects/project1/grading_tests --github_link "$PWD" -q
```

The `/tmp` prefix supplied session-local GitPython without altering ml26; omit
it when development dependencies are installed normally. The public GitHub URL
test intentionally fails for a local path. It remains pending until students
publish the reviewed commit and test its actual permalink.

To repeat the original search, use **fresh** output, artifact and config
directories through the CLI; populated directories are rejected before writes.
`review_experiments.py` provides the bounded diagnostic protocol, with raw
weights and indices confined to its local output directory. Exact executed
commands and status are in [the final audit](docs/final_audit.md).

## Report and scientific limitations

[The compiled report](report/report.pdf) has two main-content pages and a
references-only third page, using the unchanged official IEEEtran class.
Rebuild numeric macros and figures from tracked aggregate evidence:

```bash
python build_report.py
OPENBLAS_NUM_THREADS=4 python summarize_results.py
```

No private candidate files, raw data or original course-example directory are
needed for these two commands. For independent aggregate recomputation, use
`summarize_results.py --recompute` with the data, local candidate artifacts and
a fresh `--output-dir`. Compilation was checked with Tectonic 0.17.0 using
only its local cache; the exact compiler command is in the audit.

Limitations include high missingness, heuristic feature typing, unweighted survey
evaluation, possible dependence between respondents, non-nested selection,
budget-limited optimization and unverified external clinical generalization.
The calibrated-probability F1 threshold theorem is not assumed to apply to this
classifier. [The source audit](docs/source_audit.md) records verified references
and their scope.

## Academic integrity and student handoff

AI assisted implementation, debugging, experimentation and editing. The requester
authorized assistance; independent CS-433 staff-policy confirmation remains
pending. EPFL guidance calls for course-specific clarification, verification and
disclosure. Preserve attribution and finalize the declaration with the instructors.
See [policy review](docs/ai_usage_policy_review.md) and
[student review checklist](docs/student_review_checklist.md).

All three students must understand and approve the work. Students alone handle
the GitHub push, public-permalink grading check, any optional AIcrowd submission
within the shared daily budget, and the EPFL final submission.
