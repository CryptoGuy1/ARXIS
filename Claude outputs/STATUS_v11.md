# Status of the revision: what is regenerated and what is not

Read this before the manuscript. Both P0 corrections are implemented and the
pipeline is re-running, but the longest job, the 30-run comparison, had not
finished when this package was assembled. This file says exactly which numbers
are post-correction and which are still carried over, so that nothing is taken
on trust.

## Both P0 corrections are implemented and verified

| Correction | Status | Evidence |
|---|---|---|
| 4.1 Two-sided embargo | **Implemented** | `retrain/raw_pipeline.py:block_wise_holdout`. Partition is now 4,900 / 1,264. Minimum train-to-test separation measured at 21 window starts on every seed, against the 20 needed for zero shared readings |
| 4.2 Partition-local anomaly path | **Implemented** | `retrain/raw_pipeline.py:partition_anomaly`, called from `prepare()` and explicitly in both class-disjoint drivers. Autoencoder and its input scaler are fitted on training-partition nominal windows only |
| CUSUM reset semantics | **Implemented** | `retrain/new_baselines.py`. Statistic resets at every class-block change |

## Tables regenerated under the corrected pipeline

| Object | Source | Headline change |
|---|---|---|
| Table 8, protocol sensitivity | `v3_leakage.csv` | Random-split inflation rises from 1.66–4.08 to 3.23–5.57 points on temporal models |
| Table 14, decision-state ablation | `v2_ablation.csv` | Anomaly-only state falls from 75.1% to 65.8%; removing σ now *improves* accuracy by 0.40 points |
| Table 15, perturbation | `v2_perturb.csv` | Gradient boosting's miss at total sensor loss falls from 60% to 19.5%; a fourth failure pattern appears in the perceptron |
| Table 18, alarm episodes | `v3_alarm_episodes.csv` | New. Compression from 5× to 316×; the eventized ordering reverses the indication ordering |
| Table S2, cost sweep | `v2_costsweep.csv` | Accuracy peak moves from 6:1 to 1:1 |
| Table S3, calibration | `v2_calibration.csv` | Lowest ECE moves from the deep ensemble to MC dropout, intervals still overlapping |
| Anomaly component | `v2_anomaly.json` | **The largest single change.** Held-out ROC-AUC falls from 0.9928 ± 0.0053 to 0.9251 ± 0.0618, and FPR rises from 0.0000 to 0.1101 ± 0.0992. The zero-false-positive claim was an artifact of the leakage |
| Anomaly-input probe | `v2_anomaly_influence.csv` | Gradient boosting's miss under a forced-zero anomaly input rises from 0.3405 to 0.4203 |
| CUSUM | `v2_zoo.csv` | Accuracy 0.4994 → 0.4535; false-alarm rate 0 → 0.1861, the cost of the corrected reset |

## Tables NOT yet regenerated

These still carry pre-correction numbers. The job producing them
(`retrain/run_final.py`, the 30-run protocol) was at run 25 of 30 when this was
packaged, and the drivers that consume its output had not run.

| Object | Waiting on | What to do |
|---|---|---|
| Table 9, 30-run model comparison | `run_final.py` | `make final` |
| Table 10, ρ sensitivity | `run_rho_sensitivity.py` | `make rho` |
| Table 11, ROPE sensitivity (new) | `run_rope_sensitivity.py` | `make rope` |
| Table 12, class-disjoint evaluation | `run_final.py` LOCO grid | `make final` |
| Partition-level bounds quoted in text | `run_partition_bounds.py` | `make bounds` |
| Paired bootstrap intervals | `run_paired_bootstrap.py` | `make bootstrap` |
| All 20 figures | every driver above | `make figures` |

**Do not submit until those are regenerated and `make verify` is green.** The
chain is scripted end to end:

```bash
cd /home/claude/arxis
./rerun_v11.sh          # picks up where it left off if re-run from the top
make figures manifest verify
```

`verify_v11.py` is built from `verify_v10.py` and needs its table bindings
re-pointed at the new files once those tables exist; until then, `verify_v10.py`
will fail on exactly the tables listed above, which is the correct behavior and
is how you can tell the job is not finished.

## Three things that are author-side and cannot be done from here

1. **Tag and archive the release.** The repository is not a git checkout in this
   workspace. `RELEASE_CHECKLIST.md` has the sequence, including reserving the
   Zenodo DOI before the manuscript's final pass.
2. **The Ultralytics version** used to train the thermal classifier. The manifest
   cannot observe it because that training happened outside this pipeline.
3. **The EEMUA 191 fourth-edition numeric envelope.** The edition and its contents
   structure are verified; the specific alarm-rate figures in the 2024 text are
   paywalled and were not re-checked against it.
