# Historical initial-build execution record

This record describes the original build before repository cleanup and the final
quality review. It does not describe the current tracked file inventory or HEAD.
Current status is in `docs/final_audit.md` and `results/review_20261029/`.
Commands were run from the original project root unless a working directory is stated.
Full outputs and machine-readable measurements are in `results/`.

| Phase | Actual command / action | Outcome |
|---|---|---|
| Discovery | `pwd`, `git status --short`, `git branch --all`, scoped `rg --files`, Python/environment inspection | Starter/data only; root `.git` not a repository; ml26 Python 3.12.14 / NumPy 2.5.3 |
| Current requirements | GitHub connector reads of course project/tree and web reads of course/competition pages | Current PDF/helper/test Git blobs checked; accuracy/F1 conflict found |
| Course information | Official PDF downloaded read-only into `/tmp`; local PDF text-show operators decoded with standard library | No explicit AI-use clause; internal date contradiction documented |
| Initial grader | `python -m pytest --github_link "$PWD" grading_tests -q` | Collection error: missing GitPython |
| Grader dependency | `python -m pip install --target /tmp/cs433-audit-deps GitPython==3.1.18` | Development-only temporary install; later isolated under `/tmp/cs433-grader-deps` |
| Formatting | Black CLI with default workers, then one worker | Both stalled; exact cause unconfirmed; session-owned processes stopped explicitly |
| Formatting fallback | Black `format_str` API over authored root/test Python files | Completed; public formatting check passed; original helpers/tests/template unchanged |
| Initial method tests | Public grader plus independent analytical tests | 37 passed; missing README/run and real URL initially failed |
| Pipeline unit tests | `python -m pytest tests -q` with temporary grader path | Initial 17 passed; later two substantive checks added |
| CSV parsing | `load_data('dataset')` | 27.98 s; exact schemas/row counts/ID alignment checked; SHA-256 caches created |
| EDA | `OPENBLAS_NUM_THREADS=4 python -u audit_data.py` | Completed; actual feature/duplicate/missingness results saved |
| Main comparison | `OPENBLAS_NUM_THREADS=4 python -u experiments.py --data-dir dataset --seed 433 --cross-validate` | Ten successful comparisons, frozen selection, untouched holdout, three CV folds; no failed model fits |
| Final refit | `OPENBLAS_NUM_THREADS=4 python -u train_final.py --data-dir dataset` | 328,135 rows, 1,183 columns; 120 steps, iteration cap, 11.29 s measured fit |
| Prediction | `OPENBLAS_NUM_THREADS=4 python -u run.py --data-dir dataset --output submission.csv --seed 433` | Official-format CSV verified; output sidecars saved |
| Fresh-directory prediction | Copied essential runtime/model files into fresh `/tmp/cs433-clean-reproduction-*`; same `run.py`, fresh CSV cache | Exact original CSV hash; same ml26 environment, not a new dependency environment |
| Separate retraining | `OPENBLAS_NUM_THREADS=4 python train_final.py --data-dir dataset --model artifacts/retrained_model.npz` | Exact weight equality, maximum absolute difference zero; 10.37 s fit |
| Retraining prediction | `OPENBLAS_NUM_THREADS=4 python run.py --data-dir dataset --output artifacts/retraining_submission.csv --model artifacts/retrained_model.npz --seed 433` | Exact original CSV hash |
| Result binding | `python summarize_results.py`, `python build_report.py` | Ablations, bootstrap diagnostics, figures and TeX tables generated from saved measurements |
| LaTeX tooling | Official portable Tectonic 0.17.0 downloaded to `/tmp`, SHA-256 verified against official release metadata | No system-wide install; TeX cache in `/tmp` |
| PDF inspection | pypdf/PyMuPDF installed under `/tmp` only | PDF tools brought additional dependencies into that temporary directory; ML commands still used original ml26 NumPy 2.5.3; grader path isolated to avoid shadowing NumPy |
| Report compilation | In `report/`: `XDG_CACHE_HOME=/tmp/cs433-tex-cache /tmp/cs433-tectonic/tectonic --keep-logs report.tex` | First draft spilled body to page three; font encoding/table width corrected; final body pages 1–2, references only on page 3 |
| Combined tests | `PYTHONPATH=.:/tmp/cs433-grader-deps OPENBLAS_NUM_THREADS=4 python -m pytest tests grading_tests --github_link "$PWD" -q` | 46 passed, one expected real-GitHub-permalink failure; legacy loader deprecation warning |
| Original preservation | SHA-256/byte comparisons for 13 supplied files; read-only official clone Git status | All 13 identical; official working tree clean |

No remote Git mutation, message to another person, competition upload or final
portal submission was performed. A local independent repository snapshot, if
created during final packaging, is recorded in `results/package_manifest.json`.
