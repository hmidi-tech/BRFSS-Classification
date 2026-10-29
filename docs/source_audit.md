# Verified source audit — 29 October 2026

Primary materials were read for the arguments below. No source code, external
observations, pretrained models or additional labels were imported. Community
opinions are not used as scientific evidence.

| Source | Verified passage and relevance | Limits |
|---|---|---|
| CDC, *2015 BRFSS Codebook Report*, August 23, 2016. [PDF](https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf) | Printed p.73: HAREHAB1 concerns rehabilitation following heart attack; blank responses have a CVDINFR4-dependent skip condition. Printed p.13: PHYSHLTH=88 means zero unhealthy days, while 77/99 have other meanings. | Supports specific exclusions/coding cautions, not a comprehensive semantic or leakage audit of all columns. |
| CDC, *Overview: BRFSS 2015*, July 2016. [PDF](https://www.cdc.gov/brfss/annual_data/2015/pdf/overview_2015.pdf) | Weighting section explains design weights, overlapping telephone frames and raking. | Our unweighted classification and IID-row interval do not estimate design-adjusted population quantities. Including weight-related columns as predictors is not survey-weighted estimation. |
| Zachary C. Lipton, Charles Elkan, Balakrishnan Narayanaswamy, *Optimal Thresholding of Classifiers to Maximize F1 Measure*, ECML-PKDD 2014, pp.225–239. [DOI](https://doi.org/10.1007/978-3-662-44851-9_15), [author preprint](https://arxiv.org/html/1402.1892v2) | Publisher metadata and the preprint's Corollary 1 were verified. If scores are calibrated conditional probabilities, the expected-F1-optimal threshold is half the optimal F1. Section 7 discusses finite-sample threshold-selection effects. | Calibration was not established here. Neither the fitted threshold nor half the observed tuning/holdout F1 verifies the theorem's assumptions. Empirical tuning remains separate from evaluation. The preprint title is *Thresholding Classifiers to Maximize F1 Score*. |
| L. Sasse, E. Nicolaisen-Sobesky, J. Dukart et al., *Overview of leakage scenarios in supervised machine learning*, Journal of Big Data 12, 135 (2025). [DOI](https://doi.org/10.1186/s40537-025-01193-8) | Model assessment/selection and preprocessing sections require fitting imputation, normalization and other data-dependent choices within training folds. | Supports our procedural audit, not a proof that every survey predictor is free of target-dependent information. |
| Gavin C. Cawley and Nicola L. C. Talbot, *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*, JMLR 11, pp.2079–2107 (2010). [Publisher](https://www.jmlr.org/papers/v11/cawley10a.html) | Abstract and performance-estimation discussion distinguish optimization of a finite validation criterion from independent evaluation. | Three fold scores do not give an unbiased estimate of a search whose family selection was not nested. Their sample SD is not a standard error or a confidence interval. |
| AIcrowd, *EPFL Machine Learning Project 1: Challenge Rules*, accessed October 29, 2026. [Rules](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1/challenge_rules) | Written metric is classification accuracy; five entries per day. | The separately observed [leaderboard](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1/leaderboards) displays F1 first. Its ordering is insufficient evidence to override a current written rule or establish a Fall 2026 staff clarification. Both metrics are reported; the frozen CSV remains F1-oriented. |
| EPFL, *Guidelines for using (Gen)AI in studying*, accessed October 29, 2026. [Guidance](https://www.epfl.ch/education/teaching/index-html/ai-teaching/guidelines-for-using-ai-in-studying/) | Coursework guidance calls for course-specific clarification, checking generated material and disclosing assistance under the cited EPFL originality rules. | General institutional guidance does not establish CS-433-specific authorization. The requester authorized assistance; staff policy and final disclosure placement still require student confirmation. |

The current Fall 2026 project PDF was also read locally, including mandatory
interfaces, allowed dependencies, final-loss conventions, the prohibition on
`numpy.linalg.lstsq`, and two body pages plus a possible references-only third
page. Public grading instructions were read from the separate official clone.
No Moodle/Ed staff clarification was available. No inference is made from a
historical course year's rules. The report cites only references that support
its actual methodological statements.
