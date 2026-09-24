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

.PHONY: all verify selfaudit test check figtext proof docx manifest figures clean-cache release-check \
        comparison classdisjoint leakage calibration anomaly episodes final \
        hazardepisodes blockboot rho rope bounds bootstrap \
        anomalyclass metriccost persistence splitcounts bayes reject rawfaults overlap weighting

## ---------------------------------------------------------------- experiments

# Table 11; Figs. 14 and 16; the in-distribution comparator set; Tables S2 and S6
comparison:
	$(PY) -m retrain.run_all_v2

# Table 10; Figs. 11 to 13
classdisjoint: comparison
	$(PY) -m retrain.run_loco_v3

# Table 7; Fig. 8
leakage: comparison
	$(PY) -m retrain.run_leakage

# Table 9; Fig. 10; Table S3 and Fig. S3; the ROC curves
calibration: comparison
	$(PY) -m retrain.run_calib_actions_v2

# Fig. S5, the anomaly-influence panel
anomaly: comparison
	$(PY) -m retrain.run_anomaly_influence

# Table 14 (eventized alarm burden)
episodes: comparison
	$(PY) -m retrain.run_alarm_episodes

# Table 15 (episode-level hazard metrics and detection latency)
hazardepisodes: comparison
	$(PY) -m retrain.run_episode_metrics

# the primary uncertainty treatment: moving-block bootstrap and the
# disjoint-support correction quoted throughout the manuscript
blockboot: comparison
	$(PY) -m retrain.run_block_bootstrap

# Table 8 and Fig. 9; the 30-run protocol and the equivalence test
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

# Fig. 7: per-class reconstruction error under the partition-local anomaly path
anomalyclass: comparison
	$(PY) -m retrain.run_anomaly_by_class

# Supplementary Section S8: what the metric set costs to compute
metriccost: comparison
	$(PY) -m retrain.run_metric_cost

# the correlation term of Eq. 11, read from the partition rather than hard-coded
splitcounts:
	$(PY) -m retrain.run_split_counts

# recompute the Bayesian posteriors at the corrected rho, without retraining
bayes: splitcounts
	$(PY) -m retrain.run_bayes_recompute

# Table 12 and Table S9: what a confidence-threshold reject option buys and costs
reject: comparison
	$(PY) -m retrain.run_reject_option

# Table 13 and Table S10: faults injected in raw sensor space, anomaly recomputed
rawfaults: comparison
	$(PY) -m retrain.run_raw_faults

# Table S8: overlap contamination against temporal position at matched n
overlap: comparison
	$(PY) -m retrain.run_overlap_isolation

# The weighting sign of the cost-sensitive GBM, and the ordinal objective's
# action vocabulary: both raised in review, both answered by counting
weighting: comparison
	$(PY) -m retrain.run_weighting_probe

# Table S7: does the eventized result depend on the persistence rule?
persistence: comparison
	$(PY) -m retrain.run_persistence_sweep

## ------------------------------------------------------------------- products

# Figs. 1 to 16, and Figs. S1 to S6 in the supporting information
figures: comparison classdisjoint leakage calibration anomaly final
	$(PY) -m retrain.make_figs_v2
	$(PY) -m retrain.make_figs_arch

verify:
	$(PY) verify_v15.py

# the document against itself: superseded values, traceability, cross-references,
# notation, vocabulary, prose risk
selfaudit:
	$(PY) self_audit.py

# the metric module the tables run through
test:
	$(PY) test_safety_metrics.py

# everything a reviewer would want green before reading the paper
check: verify selfaudit test figtext proof
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
     final splitcounts bayes rho rope bounds bootstrap weighting \
     figures manifest check

## ----------------------------------------------------------------- housekeeping

clean-cache:
	rm -f retrain/results/real_features.csv retrain/results/real_features.csv.meta.json \
	      retrain/results/real_features.csv.windows.npy

# Fails loudly if the tree is not in a releasable state.
release-check: manifest verify
	@$(PY) - <<'EOF'
	import json, subprocess, sys
	m = json.load(open("RUN_MANIFEST.json"))
	problems = []
	if m["git"]["commit"] is None:
	    problems.append("not a git repository; cannot tag a release")
	elif m["git"]["dirty"]:
	    problems.append("working tree is dirty; commit before tagging")
	if not m["results"]:
	    problems.append("no result files found in retrain/results_v2")
	if len(m["figures"]) < 20:
	    problems.append(f"only {len(m['figures'])} figures present, expected 20")
	for p in m["drivers"]:
	    if p["sha256_16"] is None:
	        problems.append(f"driver missing: {p['path']}")
	if problems:
	    print("NOT RELEASABLE:"); [print("  -", p) for p in problems]; sys.exit(1)
	print("releasable: tree clean, manifest written, all drivers and figures present")
	EOF
