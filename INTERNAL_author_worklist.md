# ARXIS: internal author work list

Not manuscript content. This file holds the two working appendices that were
carried inside the draft while the revision was in progress. They were removed
from the submission copy, where they do not belong, and kept here so the
outstanding items are not lost.

---

## Items still requiring author action

This appendix is a working list for the authors and is not intended for publication. Every result in the body of the manuscript is backed by a stored result file.

**A.1 Cannot be resolved without data not currently on disk.**

1. *Perception split (Section 3.3, Table 4).* `data/Thermal Camera Images` is empty in the working tree, and the only frames present anywhere are 83 Mixture images (one of them zero bytes) in a partial download folder. The randomized image split therefore could not be replaced with a block-wise one. Restore the images, re-split block-wise, re-report. If accuracy falls, that is a reportable result and it supports the paper's own argument rather than undermining it.
**A.2 Still open before submission.**

2. Reliability diagrams for the four calibration estimators. Section 4.8 reports ECE with adaptive bins, bootstrap intervals, a bin sweep, class-conditional ECE, Brier score and NLL, but not the diagrams themselves.

3. Decide whether an on-device evaluation belongs in this paper at all. It was removed here because none was performed. If one is added later it needs its own protocol, and the safety columns must be reported on the corrected pipeline rather than carried over.

4. Resolve every `[CITATION NEEDED]` in the reference list, and confirm that Liang et al. (2024) is cited to *SPE Production & Operations* rather than SPE Journal.

5. Decide whether the ordinal cost matrix of Section 4.5 should be tuned before submission or left as the untuned demonstration it currently is. The paper is honest about which it is either way, but a tuned matrix would turn a partial success into a result.

**A.3 Closed in this revision, listed so the change is auditable.**

- *Cost-ratio selection.* The deployed 8:1 was originally chosen on test-partition dispersion. It is now re-selected on a blocked validation band carved from the training partition with the same 20-window embargo, which selects 2:1 (Section 4.5). `retrain/run_leakage.py`, `v3_cost_blocked_validation.csv`.
- *Ordinal objective.* Implemented as direct expected-cost minimization under a matrix that prices distance and direction along the action ladder, and reported in Section 4.5 and Tables 6 and 7. `retrain/run_final.py`.
- *Run count and equivalence analysis.* The model comparison now uses 30 randomized runs with the Bayesian correlated *t*-test of Benavoli et al. (2017) and a region of practical equivalence of one accuracy point (Sections 3.6 and 4.3). `v3_powered.csv`, `v3_powered_bayes.json`.
- *False-alarm denominator.* `retrain/metrics.py` divided false alarms by the number of alerts. It now divides by the number of clean windows, matching `retrain/safety_metrics.py`, which is the module behind every published number; the per-alert form is retained as a separate key rather than silently dropped.
- *Acquisition interval.* Confirmed at 2 s from Narkhede et al. (2022) and cited in Section 4.7. The per-1,000-window figures never depended on it.
- *Leave-one-class-out coverage.* The recurrent network, conservative Q-learning and the ordinal objective have been added to the grid, which now carries the same eleven rows as the in-distribution table.

**A.4 Terminology, resolved here but needing author sign-off.** The earlier draft described the decision component as trained by offline reward-shaped reinforcement learning, with an asymmetric reward, ε-greedy action selection, replay episodes, cumulative reward and a Huber loss. The model behind every number in this paper is trained by cost-weighted cross-entropy against targets from the deterministic map *a*\*(·), as set out in Section 3.3 and Eq. 2. The description here is the correct one. All reinforcement-learning language and the associated training-dynamics figure have been removed, and this correction should carry explicit sign-off from all three authors.

---

---

## Author verification items

Two points could not be settled from the manuscript or from the code that produced the results, and need confirmation before submission. They are separated from Appendix A because these are questions of fact or intent rather than work items.

1. **Operational meaning of Action 2.** The evaluation code counts escalation as *a* ≥ 3 and under-escalation as *a* ∈ {1, 2}, so *Request verification* was measured as a sub-alarm response that does not notify an operator. Table 3 has been corrected to match. If the intended design is that *Request verification* does place a notification in front of an operator, then the escalation boundary in `safety_metrics.py` is wrong rather than the table, and Sections 4.3, 4.4, 4.7, 5.1 and 5.2 would all need recomputing against *a* ≥ 2. Confirm which is intended.

2. **References marked `[CITATION NEEDED]`.** Nine entries carry bibliographic details that could not be verified from the material available. None is load-bearing for a result, but each needs checking before submission, in particular the venue for Liang et al. (2024), which the earlier draft cited to SPE Journal and which appears to have been published in SPE Production and Operations.

---
