# One command per table and figure in the manuscript.
#
# Reviewer comment Section 6: a reader should be able to regenerate any reported
# object without reverse-engineering which script produced it. Every target below
# names the manuscript objects it produces.
#
#   make all          everything, in dependency order (several hours on 2 cores)
#   make verify       check every value in the manuscript against the result files
#   make docx         build the submission documents from the Markdown sources
#   make proof        read the built documents the way a production editor would
#   make manifest     record commit, host, package versions and file hashes
#   make clean-cache  drop the feature cache so the pipeline rebuilds from raw data

PY := python3
RESULTS := retrain/results_v2
FIGS := figures_v2

.PHONY: all verify selfaudit test check figtext proof docx manifest figures clean-cache release-check perseed \
        comparison classdisjoint leakage calibration anomaly episodes final \
        hazardepisodes blockboot rho rope bounds bootstrap \
        anomalyclass metriccost persistence splitcounts bayes reject rawfaults overlap weighting hazardgbm nonhazard rungcheck thermal

## ---------------------------------------------------------------- experiments

# the five-seed comparator grid: the in-distribution set, Table S2, Table S6, Fig. 6
comparison:
	$(PY) -m retrain.run_all_v2

# Table 7 and Experiment 6: per-partition counts behind the one-partition claims
perseed:
	$(PY) -m retrain.run_perturb_perseed

# Table 6 and Fig. 5
classdisjoint: comparison
	$(PY) -m retrain.run_loco_v3

# Table 4 and Fig. 3
leakage: comparison
	$(PY) -m retrain.run_leakage

# Table S3, Fig. S3, the action matrix of Table S20, and the ROC curves of Fig. S6
calibration: comparison
	$(PY) -m retrain.run_calib_actions_v2

# Fig. S5, the anomaly-influence panel
anomaly: comparison
	$(PY) -m retrain.run_anomaly_influence

# Tables S11 and S15 (eventized alarm burden)
episodes: comparison
	$(PY) -m retrain.run_alarm_episodes

# Tables S12 and S16 (episode-level hazard metrics and detection latency)
hazardepisodes: comparison
	$(PY) -m retrain.run_episode_metrics

# the primary uncertainty treatment: moving-block bootstrap and the
# disjoint-support correction quoted throughout the manuscript
blockboot: comparison
	$(PY) -m retrain.run_block_bootstrap

# Table 5 and Fig. 4; the 30-run protocol and the equivalence test
final: comparison
	$(PY) -m retrain.run_final

# Table S4 (correlation-term sweep)
rho: final
	$(PY) -m retrain.run_rho_sensitivity

# Table S5 (ROPE-width sweep)
rope: final
	$(PY) -m retrain.run_rope_sensitivity

# partition-level binomial bounds quoted throughout
bounds: comparison classdisjoint final
	$(PY) retrain/run_partition_bounds.py

# paired bootstrap over the 30 runs
bootstrap: final
	$(PY) retrain/run_paired_bootstrap.py

# Fig. S10: per-class reconstruction error under the partition-local anomaly path
anomalyclass: comparison
	$(PY) -m retrain.run_anomaly_by_class

# Supplementary Section S8: what the metric set costs to compute
metriccost: comparison
	$(PY) -m retrain.run_metric_cost

# the correlation term of Eq. S2, read from the partition rather than hard-coded
splitcounts:
	$(PY) -m retrain.run_split_counts

# Table 5 supplementary row and Table 6 variant values: gas-keyed gradient boosting, 30 runs and five seeds
hazardgbm: bayes classdisjoint
	$(PY) -m retrain.run_hazard_gbm

# recompute the Bayesian posteriors at the corrected rho, without retraining
bayes: final splitcounts
	$(PY) -m retrain.run_bayes_recompute

# Tables S9 and S13: what a confidence-threshold reject option buys and costs
reject: comparison
	$(PY) -m retrain.run_reject_option

# Tables S10 and S14: faults injected in raw sensor space, anomaly recomputed
rawfaults: comparison
	$(PY) -m retrain.run_raw_faults

# Table S8: overlap contamination against temporal position at matched n
overlap: comparison
	$(PY) -m retrain.run_overlap_isolation

# The weighting sign of the cost-sensitive GBM, and the ordinal objective's
# action vocabulary: both raised in review, both answered by counting
weighting: comparison
	$(PY) -m retrain.run_weighting_probe
# Supplementary Section S27: the nonhazardous alarm rate over NoGas and Perfume
nonhazard: calibration
	$(PY) -m retrain.run_nonhazard_alarm

# Table S7: does the eventized result depend on the persistence rule?
persistence: comparison
	$(PY) -m retrain.run_persistence_sweep

# Escalation at a >= 3, 2 and 1 on the held-out Mixture class (Section S27)
rungcheck: final
	$(PY) -m retrain.run_rung_check

## ------------------------------------------------------------------- products

# Figs. 1 to 6 and Figs. S1 to S6, S11 to S16; Figs. S7 to S9 come from make_figs_arch
figures: comparison classdisjoint leakage calibration anomaly final
	$(PY) -m retrain.make_figs_v2
	$(PY) -m retrain.make_figs_arch

verify:
	$(PY) verify_v15.py

# the thermal classifier under both protocols: per-run checks and the summary the perception numbers come from
thermal:
	$(PY) retrain/summarize_thermal.py

# the document against itself: superseded values, traceability, cross-references,
# notation, vocabulary, prose risk
selfaudit:
	$(PY) self_audit.py

# the metric module the tables run through
test:
	$(PY) test_safety_metrics.py

# everything a reviewer would want green before reading the paper
check: verify selfaudit test figtext proof thermal
	$(PY) audit_release.py

## every pair of text elements in every figure, tested for overlap
figtext:
	$(PY) check_figure_text.py

## the submission documents, built and post-processed reproducibly
docx:
	$(PY) build_docx.py

## the built document as a production editor reads it: broken words inside table
## columns, figure placement width, rows breaking across pages, caption order
proof:
	$(PY) check_page_proof.py

manifest:
	$(PY) make_manifest.py

all: comparison classdisjoint leakage calibration anomaly episodes hazardepisodes \
     blockboot anomalyclass metriccost persistence reject rawfaults overlap \
     final splitcounts bayes rho rope bounds bootstrap weighting nonhazard \
     perseed rungcheck hazardgbm figures manifest check

# Preserve experiment/product ordering even when the caller uses make -j.
.NOTPARALLEL: all

## ----------------------------------------------------------------- housekeeping

clean-cache:
	rm -f retrain/results/real_features.csv retrain/results/real_features.csv.meta.json \
	      retrain/results/real_features.csv.windows.npy

# Fails loudly if the tree is not in a releasable state.
release-check: verify
	$(PY) audit_release.py
	$(PY) -c "import subprocess; p=subprocess.run(['git','status','--porcelain'],capture_output=True,text=True,check=True); assert not p.stdout.strip(), 'working tree is dirty; commit before tagging'; print('releasable: tree clean and released artifacts verified')"
