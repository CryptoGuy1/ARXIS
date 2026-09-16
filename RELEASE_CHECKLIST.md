# Release checklist: turning the development repository into a frozen artifact

Reviewer comment 4.3 and Section 6. The reviewer's judgment was that public reproducibility is the manuscript's weakest area, scored 4/10, and that the repository and the paper currently describe different experimental generations. This is the sequence that closes that gap. Every step is author-side: it needs the real repository, the real remote, and in two places a real account.

Work top to bottom. Do not tag until `make release-check` passes.

---

## 1. Bring the working tree to the state the manuscript describes

- [ ] Copy the corrected pipeline over the repository copy: `retrain/raw_pipeline.py` (two-sided embargo, partition-local anomaly), `retrain/new_baselines.py` (CUSUM per-block reset), the patched drivers (`run_all_v2.py`, `run_calib_actions_v2.py`, `run_final.py`, `run_loco_v3.py`), and the two new drivers `run_alarm_episodes.py` and `run_rope_sensitivity.py`.
- [ ] Copy `Makefile`, `make_manifest.py`, `verify_v11.py`, `README.md`, `textcheck.py`, `stylecheck.py`.
- [ ] Copy the regenerated `retrain/results_v2/` and `figures_v2/` in full. Delete any result file the manifest does not list: a stale CSV in that directory is exactly how a reviewer concludes the paper and the code disagree.
- [ ] Delete or clearly quarantine the superseded scripts (`run_exp1_real.py`, `run_exp3_loco.py`, `run_exp4_calibration.py`, `run_expzoo.py`). If they are kept for history, move them to `retrain/legacy/` with a one-line note saying they do not produce any reported number.

## 2. Make the environment claim true

- [ ] `make manifest` to regenerate `RUN_MANIFEST.json` and `requirements.lock` from the interpreter that actually produced the results.
- [ ] Open Appendix A of the manuscript and make the version list match `requirements.lock` **exactly**. The reviewer noticed that the manuscript and the requirements file disagreed; that is a five-minute fix and an expensive thing to be caught on twice.
- [ ] Record the Ultralytics version used to train the thermal classifier. It is the one version the manifest cannot observe, because that training happened outside this pipeline. If it cannot be recovered, say so in Appendix A in one sentence rather than leaving the placeholder.
- [ ] Replace `requirements.txt` with `requirements.lock`, or make the former a superset that does not contradict the latter.

## 3. Make the repository and the paper describe the same object

- [ ] `make verify`. It must report zero failures. If it does not, the manuscript is wrong, not the checker.
- [ ] Confirm `src/` carries a note at the top of its entry point saying the runtime adds guardrails not evaluated in the manuscript, matching the README and the paper's "What Is Evaluated, and What Is Not".
- [ ] Search the repository for `8:1`, `DQN`, `reinforcement learning`, `LOCO` and any headline accuracy figure, and correct or delete every stale occurrence. The README is rewritten; configuration files, notebooks and docstrings may not be.
- [ ] Search for the old LOCO numbers and the old `0.9928` anomaly ROC-AUC. Both changed with the leakage fixes.

## 4. Commit, tag, archive

- [ ] Commit on a release branch with a message that names the two P0 corrections explicitly, so the history shows what changed and why.
- [ ] `make release-check`. It fails on a dirty tree, a missing driver, fewer than 20 figures, or a failing verification.
- [ ] Tag: `git tag -a v1.0-spe-submission -m "Artifact for the SPE Journal submission"` and push the tag.
- [ ] Archive the tagged release. On Zenodo, link the GitHub repository, then publish the release; Zenodo mints a DOI against that exact tag. **Reserve the DOI before the manuscript's final pass**, because it has to appear in Data and Code Availability.
- [ ] Put the DOI in the manuscript's Data and Code Availability section and in the README's citation block.

## 5. What a reviewer will do, so do it first

- [ ] Clone the tagged release into an empty directory on a machine that has never seen this project.
- [ ] `pip install -r requirements.lock`, drop the corpus into `data/`, run `make comparison` alone. If that one target does not run clean from a cold clone, nothing else in this checklist matters.
- [ ] `make verify` on the cold clone.
- [ ] Open the repository landing page and read it as a stranger. The first screen should say what is evaluated, what is not, and that the analytes are surrogates.

---

## Two things this checklist cannot do for you

**The thermal split.** The perception component is trained under a randomized image split, not the block-wise protocol used everywhere on the sensor side. The images are not in this workspace, so the block-wise re-split remains outstanding. The manuscript states the asymmetry and rests no result on the perception number, which is the honest interim position, but a reviewer may still ask for it.

**The runtime experiment.** Reviewer comment 4.4 offers two remedies, and the manuscript takes the first: name the evaluated object precisely and state that the runtime differs. The second, evaluating the full runtime decision logic against the same metric set, is a real experiment that this submission does not contain. If the editor presses on it, it is roughly one driver: run the guardrails over the same held-out partitions and report the same five quantities beside the bare-policy numbers.
