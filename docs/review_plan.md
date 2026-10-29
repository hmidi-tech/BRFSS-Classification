# Initial local quality-review plan — 29 October 2026

This plan records the review that started at `d078c71` and was completed in
the six commits through `829f08b`. The later controlled editorial polish is
recorded in `docs/final_audit.md`; the original plan is retained as provenance.

Starting commit: `d078c71960d87d59a4b1cd33b4e1fd40ec4d6f31`, branch `main`,
clean working tree. The configured origin is retained. Local commits only;
no publication or submission is authorized. Environment verified: ml26,
Python 3.12.14, NumPy 2.5.3. Deadline: 16:00 Europe/Zurich.

Baseline manifests and copies are under ignored
`artifacts/tmp/review_20261029T0242/baseline/`. Existing final model files,
configuration, CSV and historical experiment outputs are protected.

Review boundaries:

1. **Numerical contracts:** reproduce failures before corrections; validate
   scalar parameters, extreme regularization, metric shapes and optional
   objective/gradient consistency. Keep legacy training arithmetic available
   so the frozen model remains exactly reproducible. Test L-BFGS termination
   against its actual final state, including zero updates and line-search stops.
2. **Pipeline and runtime:** validate fitted schemas/indices, exclude IDs even
   when calling preprocessing directly, verify cached content, prevent partial
   cache publication and output collisions. Bound inference memory and decouple
   final training from experimental orchestration. Verify fresh-directory
   inference from test CSV plus model files alone.
3. **Measured checks and report:** audit historical selection and metrics;
   perform only bounded development/tuning numerical diagnostics, never new
   holdout-driven selection. Verify primary references, improve report prose
   and source-bind its numbers. Generate figures without private candidate
   checkpoints, compile and render all pages.
4. **Handoff:** replace stale README assumptions about untracked course/data
   artifacts; clearly label historical evidence and current status. Run local
   official tests from the separate course clone, check privacy/tracking and
   stage only reviewed paths. Preserve prior Git provenance and configured
   identity. End with a clean working tree and no remote mutation.

Proposed commit boundaries follow these real changes, rather than creating
empty commits. Each includes focused checks and `git diff --cached --check`.
The GitHub permalink check remains pending; a configured URL alone does not
prove the reviewed commits have been published.
