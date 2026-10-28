# Dataset diagnostics

Measured from every supplied CSV row; no data was uploaded.

Training: 328,135 × 321; test: 109,379 × 321. Positive labels: 28,975 (8.8302%). Negative labels: 299,160.

Missing cells: 47,175,779 (44.7880%) in train and 15,724,379 (44.7852%) in test. No infinities, inconsistent row lengths, duplicate IDs, ID overlap, column mismatch or label-ID mismatch were found.

Six features have one observed value but also missing entries: CTELENUM, COLGHOUS, STATERES, CTELNUM1, CELLFON2, CCLGHOUS. A constant observed value is not the same as a fully observed constant; its availability may still carry information. There are no completely missing training columns. Nine observed-value distributions have a modal proportion at least 99.9%, conditional on an observed entry.

225 columns have at most 12 distinct observed integer values. This is a cardinality screen, not an established semantic feature typing. Full minima, maxima, means, medians, standard deviations and cardinalities are in `results/feature_audit.csv`.

## Extreme raw scales

| Feature | Training minimum | Training maximum | Training median |
|---|---:|---:|---:|
| SEQNO | 2.015e+09 | 2.015e+09 | 2.015e+09 |
| _PSU | 2.015e+09 | 2.015e+09 | 2.015e+09 |
| IDATE | 1.012e+06 | 1.2312e+07 | 6.242e+06 |
| FLSHTMY2 | 12014 | 1e+06 | 1.0201e+05 |
| HIVTSTD3 | 11985 | 1e+06 | 1.1201e+05 |
| _STSTR | 11011 | 7.2202e+05 | 2.9204e+05 |
| _DRNKWEK | 0 | 99900 | 23 |
| PA1MIN_ | 0 | 54000 | 270 |

Sequence numbers/date encodings are not physical measurements; raw large values are not automatically outliers. Survey-specific special codes remain unchanged unless their meaning is established. No full semantic recoding, quantitative outlier deletion or global winsorization is claimed.

## Largest observed missingness shifts

| Feature | Train missing | Test missing | Difference (percentage points) |
|---|---:|---:|---:|
| HHADULT | 58.678% | 59.081% | +0.403 |
| NUMWOMEN | 42.521% | 42.162% | -0.359 |
| NUMMEN | 42.520% | 42.162% | -0.359 |
| CELLFON3 | 42.487% | 42.131% | -0.357 |
| CSTATE | 57.513% | 57.869% | +0.356 |

Similar aggregate missingness does not rule out covariate shift; no comprehensive distribution-equality test is claimed.

## Redundancy and leakage

`results/correlated_features.csv` screens correlations on 20,000 deterministic training rows, mean-filling missing values only for that diagnostic. It is not a fitted modeling transformation. SEQNO/_PSU are duplicates; _CRACE1/_CPRACE have correlation approximately one in the sample. No feature selection based on test labels is possible or performed.

No exact duplicate rows were found when retaining all 321 features, including survey sequence keys. This does not prove person-level independence after ignoring those keys.

All 773 nonmissing HAREHAB1 training entries belong to positive labels. CDC codebook p.73 establishes a heart-attack-dependent skip condition, so the feature and its missing indicator are excluded from every comparison. The raw target and original target-constituent columns are absent from inputs and guarded against in preprocessing. Other medical variables are not removed merely because they correlate with the target. An exhaustive leakage guarantee for every optional survey module is not established.

[CDC metadata](https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf) confirms that PHYSHLTH=88 means zero days; 77/99 have different meanings. No additional observations, external labels or pretrained model were imported.

The majority-class baseline has accuracy 0.911698, F1 0, recall 0, specificity 1 and balanced accuracy 0.5. Its precision is undefined and recorded as null.

Audit peak RSS: 1329.5 MiB. Raw CSV parsing initially took 28.0 seconds; cached full-data diagnostics took 7.63 seconds. See saved logs for measured training costs.
