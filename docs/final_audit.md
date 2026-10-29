# Final local quality-review handoff

**Status: technically ready for final student review and student-controlled publication.** Local checks pass apart from the expected public-permalink-format failure. Final group approval, an actual published-permalink check and external submission remain student actions. The requester reports staff permission for assistance with no separate manuscript mention; this is recorded as a student account rather than independently obtained written policy.

The initial quality review was recorded at 2026-10-29T03:30:32.098032+01:00. Its starting HEAD: `d078c71960d87d59a4b1cd33b4e1fd40ec4d6f31`. Branch `main`; origin remains `https://github.com/hmidi-tech/BRFSS-Classification.git`. The original root commit is an ancestor of the review commits. Git identity and history were not rewritten.

## Final controlled polish — 29 October 2026

This pass started from clean `829f08bb4c72f98a4b7d41c233f3463f7acd012f`
on `main`, with 93 tracked files and the six previous review commits intact.
It makes editorial and documentation changes only: no new training, model
selection, threshold tuning or numerical-code changes. Verification evidence is
in `results/review_20261029/final_polish_verification.json`; detailed local logs
and previews are ignored under `artifacts/tmp/polish_20261029T041214/`.

- The report states the research question directly and identifies the 0.58
  accuracy-selected result as **tuning** accuracy 0.9153 and F1 0.1796.
  These values were checked against the selected candidate's saved aggregate.
- Temporary publication/AI-policy wording was removed from the scientific
  paper, consistent with the newly reported staff guidance. Internal disclosure
  drafts and historical assistance records remain; no written staff ruling or
  all-student approval is fabricated.
- LaTeX now preserves literal double hyphens in CLI flags. Rephrasing the
  penalty list also removes its badly stretched lines. The final PDF was
  rebuilt offline, rendered and visually checked on all three pages.
- The README describes the scientific software and durable reproduction steps,
  without session-specific handoff paragraphs or a dedicated AI section.
- Both current-checkout inference and a fresh source directory with only the
  test CSV and complete saved model reproduce the reference CSV exactly.
  Python compilation and the 50 independent/27 public passes were rerun;
  the sole public failure remains the real-permalink-format requirement.
- All original result files, weights, metadata, selection and competition CSV
  retain their pre-polish hashes. Actual data CSV fingerprints were rechecked.
  No fresh refit or expensive experiment was needed.

Exact successful current-layout reproduction:

```bash
OPENBLAS_NUM_THREADS=4 python run.py --data-dir ../dataset --output artifacts/tmp/polish_20261029T041214/submission.csv --seed 433
```

The fresh-directory check used `--data-dir dataset --output submission.csv`
with only `x_test.csv` in its dataset directory, no Git checkout, no training
labels, no candidate checkpoints and a new cache. Its CSV hash is identical.

Final check commands (executed from the repository root):

```bash
OPENBLAS_NUM_THREADS=4 python -m pytest tests -q
PYTHONPATH=.:/tmp/cs433-grader-deps OPENBLAS_NUM_THREADS=4 python -m pytest --rootdir=. ../../ML_course_official/projects/project1/grading_tests --github_link "$PWD" -q --junitxml=artifacts/tmp/polish_20261029T041214/public_tests.xml
python -m compileall -q .
git diff --check
python build_report.py
```

The test suite was rerun with JUnit output in the ignored review directory;
50 independent passes, 27 public passes, one public format failure, no skipped
tests. The temporary Python path supplies GitPython only to the public grader,
without altering ml26. The official course clone remains clean.
An initial macro-builder invocation used the report directory and failed to
locate the root script; rerunning it from the repository root passed. No
failed invocation is counted as successful.

The scoped tracked-content scan finds no prohibited datasets, labels, split
files, candidates, original course grader/PDFs or matching credential patterns.
Existing historical logs and the original CSV sidecar contain local machine
paths as reproduction provenance; these were retained and clearly identified,
not silently sanitized. This is not an exhaustive secret audit.

## Findings and disposition

| Status | Finding | Disposition |
|---|---|---|
| Fixed | Finite extreme Ridge penalties overflowed `2N*lambda` | Scale the same normal system before solving; independent regression passes |
| Fixed | Metric/threshold functions accepted malformed shapes, labels and nonfinite scores | Shared validation prevents silent broadcasting and misleading metrics |
| Fixed | Optional optimizer misreported final-step/zero-step status and accepted invalid penalties | Count accepted updates and assess the final gradient |
| Fixed | Optional sigmoid truncated integer probabilities | Preserve floating probabilities while retaining float32 legacy arithmetic |
| Fixed | Same-shaped corrupted caches could be reused | Versioned cache content checksums and temporary parsing; original caches retained |
| Fixed | Direct preprocessing accepted incomplete schemas and boolean indices | Validate fitting state, row indices and serialized mappings; explicitly guard IDs |
| Fixed | Inference allocated the full expanded matrix and could replace sidecars | Bounded transformation/scoring and exclusive verified output publication |
| Fixed | Training imported experiment orchestration; scripts could overwrite historical outputs | Shared model dispatch and fresh-directory checks; explicit audit/aggregate output paths |
| Fixed | Report builds assumed untracked course examples/candidates | Use tracked class and aggregate threshold curve; validate saved numeric views |
| Verified | Six mandatory methods | All public numerical oracles and independent tests pass; contracts unchanged |
| Verified | Preserved baseline | Original weights, metadata, configuration, CSV and historical result files match pre-review hashes |
| Verified | Scientific selection history | Ten tuning comparisons and the recorded 0.002-tolerance rule independently recomputed; original holdout scores checked without reselection |
| Not applicable | ID leakage in the original inference | The loader already separated IDs; the new direct-preprocessor guard is defensive, not evidence of prior ID-based predictions |
| Open | Competition objective | Written rules say accuracy; leaderboard ordering does not establish a current staff clarification. Retain original F1-oriented CSV |
| Reported | Staff guidance on assistance | Requester reports permission and no separate report mention; student account recorded internally, without inventing written confirmation |
| Open | Final student approval | All three students must review and approve the work; no approval is inferred |
| Not tested | Legacy environment and external generalization | No Python 3.9/NumPy 1.23.1, cross-backend bitwise, prospective clinical or population-fairness validation claimed |

The holdout was already opened before this review. No new model or threshold is selected against it. New development diagnostics are explicitly conditional post-selection checks.

## Baseline and measured diagnostics

Actual data: 328,135 training rows, 109,379 test rows, 321 features, 28,975 positives and 299,160 negatives. All three source CSV fingerprints match the historical measurements. Target interpretation, missingness and HAREHAB1 exclusion were checked against original data/evidence and CDC metadata.

Frozen selection: median fill, indicators, small-cardinality reference encoding, logistic L2 0.0001, threshold 0.20, seed 433. Holdout F1 0.422490, accuracy 0.868530, precision 0.345107, recall 0.544607 and balanced accuracy 0.722255; TP 3156, FP 5989, FN 2639, TN 53843. Original fixed-configuration CV is 0.431010 ± 0.000407 sample SD; model-family selection was not nested.

| Development-only diagnostic | Updates | Exact float64 objective | Gradient infinity norm | Tuning F1 at 0.20 |
|---|---:|---:|---:|---:|
| legacy_float32 | 120 | 0.218980560 | 0.00013143575 | 0.432121 |
| float64 | 120 | 0.218974197 | 0.00011329874 | 0.432440 |
| float64 | 240 | 0.218957628 | 2.141539e-05 | 0.432176 |

The bounded extension was triggered by recorded objective progress. It improves optimization diagnostics without a consistent F1 gain; it does not replace the frozen model. All full-data legacy fits still stop at 120 updates with final gradient 0.0002107956, above 1e-5.

Paired categorical-encoding F1 gains on three shared development folds: +0.008399, +0.016755, +0.011831. Same lambda 0.001, threshold 0.20 and 120-update legacy budget. This supports repeatability within reused development data; it is not a causal effect, significance test or independent estimate of the full search.

Numeric diagnostic JSON records fit durations, optimizer state, data/source fingerprints and split protocol. Per-fit start wall timestamps were not recorded in that diagnostic run; do not infer them from checkpoint filesystem times. The original experiment logs retain their actual timestamps.

Inference single-run peak RSS fell from 1,833,388 KiB to 461,760 KiB (warm versioned cache). Corresponding wall times were 4.65 s and 2.58 s; the first new-cache parse took 10.32 s. These are same-machine observations, not a general speed guarantee.

## Tests and static review

**77 passed, 1 failed, 0 skipped:** 50 independent checks and 27 public passes. The only failure is `test_github_link_format`, because the supplied target was local. The unmodified tests came from the separate official course clone and ran under ml26; its working tree remains unchanged. The upstream legacy loader emits one deprecation warning.

During the previous interrupted review, its state was independently
rechecked: the same 77/1/0 combined outcome, `python -m compileall -q .` with
exit code zero, another identical CSV from the current saved-model pipeline,
and current-source/PDF timestamps and hashes. All three then-current PDF pages were
rendered again and inspected. The later polish verification is recorded above;
no earlier work or experiments were restarted.

Executed combined command from this repository root:

```bash
PYTHONPATH=.:/tmp/cs433-grader-deps OPENBLAS_NUM_THREADS=4 python -m pytest --rootdir=. -o cache_dir=artifacts/tmp/review_20261029T0242/pytest_cache tests ../../ML_course_official/projects/project1/grading_tests --github_link "$PWD" -q --junitxml=artifacts/tmp/review_20261029T0242/final_tests.xml
```

Fresh source copies with test CSV only and a new cache reproduced the exact CSV. A separate refit using final numerical code produced exactly equal weights and another identical CSV. Runtime import checks find only NumPy, standard-library and local modules; no hard-coded user-home path is present in runtime sources. Scoped scans found no credential assignments or prohibited tracked datasets, labels, checkpoints, predictions or original course tests/PDFs. Historical logs retain machine paths as provenance and are labeled as historical. This is not an exhaustive secret-detection guarantee.

## Reproduction and protected fingerprints

Executed current-layout command (the original output already exists, so choose a fresh output path for another run):

```bash
OPENBLAS_NUM_THREADS=4 python run.py --data-dir ../dataset --output submission.csv --seed 433
```

Original and reproduced CSV SHA-256:

```text
e8ef72c62f514931ed2d66f28fcb29201756c66d6d90f4c6fa9a73b4df0b0665
```

The CSV has 109,379 ordered rows, header `Id,Prediction`, labels -1/+1, 14,925 positives and 94,454 negatives. Inference needs only x_test.csv and both final model files; no training data, cached design, private candidates or saved labels.

Protected weights SHA-256: `443e021cf4543d8fcea015a23d3683c3425fa9083f1c2d11eec54ce3a7a2b393`.
Protected metadata SHA-256: `8f5916747e15ac6b2e12f8dabddd10af54602a9ba2b4013e5f4433ee26fdde11`.

The `--data-dir dataset` variant was executed successfully in a fresh source directory whose data directory contained only x_test.csv. In the existing checkout a pre-existing submission.csv deliberately causes an explicit preservation error. The documented ../dataset layout was separately executed with fresh outputs. The final refit command was:

```bash
OPENBLAS_NUM_THREADS=4 python train_final.py --data-dir ../dataset --model artifacts/tmp/review_20261029T0242/refitted_final_code.npz
```

Supported environment: Python 3.12.14, NumPy 2.5.3, Linux/WSL2, four OpenBLAS threads. New caches were tested without deleting old caches. No cross-environment bitwise guarantee is made.

## Report and provenance

Compiled PDF: `report/report.pdf`; SHA-256 `61eaf4978abb5c3c949a311ebef67c9e3dc77f9a8beb38727f51a43e9ffe2735`. Two content pages plus a references-only third page, no appendix; all pages rendered and inspected. Official 10pt class preserved, no overfull boxes, undefined references or font warnings. Eight underfull line warnings remain; no clipping was observed. Confirmed authors: Mohamed Amine Hmidi, Mohamed Yebari, Omar Marouane Bensaid; EPFL.

The report now explains sample weighting, calibrated-F1 theory, conditional CV, deliberate regularization preference, finite-precision stopping and limitations. Numbers are source-bound and cross-checked against CSV/JSON; aggregate-only builds work without untracked candidate/course resources. Original report/source copies are preserved in the ignored baseline directory and starting Git revision. Verified citations and limits are recorded in `docs/source_audit.md`.

Executed report commands:

```bash
python build_report.py
OPENBLAS_NUM_THREADS=4 python summarize_results.py
cd report
XDG_CACHE_HOME=/tmp/cs433-tex-cache /tmp/cs433-tectonic/tectonic --only-cached --keep-logs report.tex
```

A permission review for an unrestricted compiler invocation timed out; the complete PDF build succeeded through the safer cached-only local invocation. No upload was needed. Earlier initial-build status remains in `docs/history/initial_build_audit.md`; original result manifests and failed-check logs are preserved, not rewritten to describe current success.

Local review commits preceding this handoff commit:

```text
0660811 fix: strengthen numerical contracts and optimizer diagnostics
ff0cce3 refactor: validate caches and schemas and bound inference memory
90fe1f5 test: measure optimizer precision and categorical robustness
518bbf9 fix: preserve probabilities for integer sigmoid inputs
a10e372 docs: strengthen scientific reporting and source-bound report builds
829f08b docs: finalize reproducibility and review guidance
faf7837 docs: polish final scientific report and clarify evaluation metrics
```

This updated audit is included in the second controlled-polish commit; retrieve its complete final identity with `git rev-parse HEAD`. No fictional dates, coauthor contributions or rewritten historical identities were introduced.

## Student actions

- Follow the actual course instructions for any required attribution or declaration. The reported staff clarification is recorded in `docs/ai_usage_policy_review.md`; an optional internal draft remains in `docs/ai_usage_declaration.md`. No separate paper mention is inserted contrary to that account.
- All three students review, understand and approve the code, measured claims and report.
- Resolve the competition objective before choosing an accuracy-oriented alternative; do not tune against the opened holdout.
- Students alone push the reviewed commits and verify the actual public commit permalink with the unmodified grader. A configured remote does not prove these commits are published.
- Any optional competition entry and EPFL final submission remain separate student actions. No acceptance or external score is claimed.

**No push, publication, AIcrowd submission or EPFL upload was performed in this review.** Read-only public-source research and local Git commits were used.
