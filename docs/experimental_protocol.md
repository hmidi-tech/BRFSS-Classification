# Experimental protocol and limitations

Seed 433; labeled rows split deterministically and by class into 196,881
development, 65,627 tuning and 65,627 final holdout observations. Stored indices:
`artifacts/split_indices.npz` locally (not tracked). They are regenerated from
the recorded seed by `stratified_split`. Each class is permuted independently
before the 60/20/20 partition. During model selection the holdout is not passed
to model/preprocessing fitting or threshold selection. The later all-label
refit legitimately uses all labels after freezing and evaluating the choice.

Before fitting, `experiments.py` writes `results/protocol.json` and ten planned
configurations. This is a bounded comparison, not an exhaustive search. The
operational objective is provisionally F1 because the observed leaderboard
orders by F1 despite the written accuracy rule; both metrics are saved. A
different objective would justify a different threshold and possibly a different
model, and requires staff clarification before competition upload.

Compare mean vs median imputation, missing indicators and categorical expansion,
then five L2 strengths on the categorical design (0, 1e-4, 1e-3, 1e-2, 1e-1).
Also fit unregularized linear and ridge baselines. Logistic labels are converted
from -1/+1 to 0/1. All learning code uses NumPy and the standard library.

Exclude `Id` while loading. Exclude `SEQNO`, `_PSU`, and raw `IDATE` in every
design. Also exclude `HAREHAB1`, a documented target-dependent rehabilitation
question. It has 773 observed training entries, all positive; its missingness
alone reveals a subset of positive outcomes. The original target constituent
columns `CVDINFR4` and `CVDCRHD4` and target `_MICHD` are guarded against if
present. Other medically relevant predictors are retained.

Numeric columns receive fitted mean/median fill and fitted mean/std scaling.
Completely missing columns and fully observed constants are removed. Partially
missing columns with one observed value may remain with a missing indicator.
For the categorical variant, integer columns with at most 12 observed training
levels receive reference coding; the most frequent level is the reference.
Missingness gets a separate indicator if observed during fitting. Unseen levels
map to the reference representation. No universal recoding of 7/9/77/88/99 is
performed: the CDC codebook confirms, for example, 88 means zero days for
`PHYSHLTH`, rather than an unknown response. High-cardinality coded quantities
remain numeric, which is a conservative but imperfect representation choice.
This limitation is explicit; no comprehensive semantic cleaning is claimed.

For each candidate, scan thresholds 0.05–0.65 inclusive in steps of 0.01 on
tuning predictions. Equal objective values prefer the larger threshold. Preserve
metrics at 0.5 and at an accuracy-selected threshold too. A common threshold
of 0.20 is used for preprocessing ablation deltas, avoiding conflating threshold
selection with a representation change. With equal iteration budgets, some
models remain incompletely optimized, so those comparisons include the practical
effect of conditioning within that budget.

Select highest tuning F1; within 0.002 prefer fewer features, then stronger L2,
then higher F1. This rule was saved before any comparison. Freeze selection in
`config/selected_model.json` before opening the holdout. Holdout results are an
evaluation of that fixed choice, not a new basis for tuning. No model was
changed using holdout results.

Three-fold CV uses the 80% non-holdout pool and the fixed selected family and
hyperparameters. Each outer fold fits preprocessing only on its training rows.
Thresholds are calibrated using an inner 80/20 partition of that outer training
fold, then the model and preprocessing are refit on the full outer training fold.
Because family/hyperparameter search was not nested, CV is a consistency check,
not an unbiased estimate of the full search procedure. Different inner seeds
and partitions also probe split sensitivity; broad random-seed optimization was
not performed. Coefficient norms are logged; coefficient-wise stability across
varying encodings is not established.

After evaluation, refit the frozen configuration and preprocessing on all
328,135 labeled rows. Category discovery on more training rows can increase the
design width: 1,172 in the selection fit, 1,183 in the final fit. The selected
threshold remains 0.20. No test labels or public scores are used. Stored weights
and all fitted feature specifications support inference-only reproduction.

The code logs started/successful/failed/interrupted states, data hashes, source
hashes, timestamps, parameters, runtime, fit termination, metrics and peak RSS.
Historical experiment-time Git fields are null because those experiments
preceded repository initialization. The student repository now has real local
commits; review evidence records source hashes and the observed source commit.
The final CSV has not been uploaded by this workflow. Its match to a best actual
competition entry cannot yet be verified.

The final quality review performs bounded precision diagnostics and three paired
categorical-encoding checks on development data. These happen after the holdout
has been opened; they are not a fresh independent final evaluation. The frozen
configuration, weights, metadata and original CSV remain unchanged. New direct
optimizer calls use float64 evaluation; the frozen configuration dispatches
explicit legacy float32 arithmetic for exact historical refitting.
