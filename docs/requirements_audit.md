# Fall 2026 requirements audit

Internal checklist; primary specification: supplied `project1_description.pdf`.
Original starter files and `/home/amine/Projects_WSL/ML_course_official` are preserved.
The workspace's protected `.git` directory is not a functioning repository.

| Requirement | Evidence / verification |
|---|---|
| Deadline October 29, 2026, 16:00 Zurich | Project PDF p.4; current course website confirms Oct 29 |
| Team of three | PDF p.1; requester confirmed Hmidi, Yebari, Bensaid author order and EPFL affiliation |
| Six methods in root `implementations.py` | PDF p.2 and unmodified public tests |
| Fixed signatures, `(w, loss)`, 1D vectors | Public tests and PDF p.2 |
| MSE factor 1/2; final-state data loss | PDF p.2 and numerical public tests |
| SGD mini-batch size one | PDF p.2 |
| Normal-equation least squares, no `lstsq` | PDF p.2 |
| Ridge factor `2*N*lambda`; all weights penalized | Course convention and public lambda=1 numerical oracle |
| Logistic 0/1 labels; L2 gradient `2*lambda*w` | PDF p.2 and public numerical oracle |
| NumPy and Python standard library only for ML | PDF p.1; plotting libraries only for figures |
| No external code, libraries, datasets | PDF p.3–4; metadata reference explicitly points to CDC 2015 |
| Local validation / CV | PDF p.3, p.7 |
| Reproduction `run.py` / `run.ipynb` and root README | PDF p.4; public tests |
| Exact reproduction of best competition CSV | PDF p.4; unverified until a real approved upload exists |
| Two-page LaTeX report, third page references only, no appendix | PDF p.1, p.7 |
| Specific public GitHub commit permalink | Public grader requires `https://github.com/.../tree/<commit>` |
| Five submissions per day shared by whole team | PDF p.3 and live AIcrowd challenge description |
| Explicit approval before publication/upload | User instruction; none performed |
| AI policy and acknowledgment | Team confirmation received; staff policy still unverified |

Unresolved contradictions:

1. AIcrowd [rules](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1/challenge_rules)
   specify accuracy. The [live leaderboard](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1/leaderboards)
   displays F1 then accuracy; observed ordering favors F1 (higher accuracy does
   not imply a higher rank). Use F1 as a provisional local objective, preserve
   accuracy results separately, and obtain staff clarification before upload.
2. The course information sheet says Oct 29 in its assessment overview but
   contains a stale Nov 1 sentence in the Project 1 subsection. The current
   project-specific PDF and course website agree on Oct 29 and take precedence.
3. The supplied test environment targets Python 3.9 / NumPy 1.23.1, while ml26
   runs Python 3.12.14 / NumPy 2.5.3. Actual testing uses ml26; legacy-environment
   compatibility must not be claimed without a separate run.
4. The archive uses `sample_submission.csv`, whereas PDF prose says
   `sample-submission.csv`. Inspecting the actual header/IDs establishes format.

Materials found: the project PDF, supplied helpers, public grader, official LaTeX
template, four actual dataset CSVs, dataset ZIP, and parent `../project1ML.zip`.
The latter contains starter files only, with data under `data/dataset/`.
The current extracted dataset is instead `dataset/`.
No local student implementation, notebook, report draft, model or experiment
history existed at discovery. The four educational guides and environment PDF
named in the request were not found in the project or relevant course clone.

Known grader limitations: the SGD oracle has only one sample and does not test
multi-sample randomness; mandatory-function gradients and mutation behavior
need independent tests. `create_csv_submission` uses `zip`, which can silently
truncate mismatched vectors; our caller checks lengths and verifies the file.
The official loader removes `Id` but does not validate label IDs, column order
or row counts; our loader explicitly validates these without modifying helpers.
