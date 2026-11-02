# Result files as released in v18.1 (provenance only)

These are the result files from `retrain/results_v2/` exactly as they stood at
commit ab5c05a (tag `v18.1-jlppi-reviewed-2026-10-20`), copied here before the
November 2026 rerun overwrote them. Nothing in the revised manuscript is read
from this folder.

They differ from the rerun in three ways that were corrected in the code:

1. **Gradient boosting** was the clean-air-weighted configuration (sample
   weights NoGas 3.0, Perfume 1.0, Smoke 1.2, Mixture 1.2), described at the time
   as cost-sensitive. The rerun uses the hazard-weighted configuration (NoGas 1.0,
   Perfume 1.0, Smoke 2.5, Mixture 2.5) as the primary comparator.
   `v3_*_hazard_gbm*` were the supplementary runs of the hazard-weighted model.
2. **Conservative Q-learning** bootstrapped across class boundaries and across
   the removed test block, because only the last training row was treated as
   terminal.
3. **The recurrent model** grouped training histories by gas, so histories near
   the removed test block spliced the leading and trailing training segments.

They were produced in the environment recorded in `requirements.lock` at that
commit (Python 3.11.17, torch 2.14.0+cu130). The CPU rerun of 2 November 2026
that found 96 verification mismatches against them is described in the
assessment of the co-author review and is not stored here.
