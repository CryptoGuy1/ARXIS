# Supporting Information

**A Safety-Oriented Evaluation Protocol for Learned Gas-Monitoring Models: Leakage-Controlled Class Exclusion and Synthetic Sensor Perturbation on a Laboratory MOX Benchmark**

Benjamin C. Nweke, Gholamreza Ramezan, Soheil Saraji

This file carries the material referenced from the manuscript body. Nothing here
is a separate result: every number is produced by the same drivers, from the same
result files, under the same protocol, and is checked by the same verification
script. It is here because the components and experiments below do not determine
the paper's safety findings and their full treatment would crowd out the ones
that do.

Objects are numbered S1, S2, S3 and are cited from the body by those labels.

---

## S1 Perception Component

The thermal path is described here in full. It is worth being explicit about why
it sits in supporting information: the decision state of Eq. 1 carries no visual
term, so no safety result in the manuscript depends on any number in this
section, and the component's evaluation is held to a weaker standard than the
sensor path's.

**Training.** Thermal images are classified with YOLOv8n-cls (Jocher et al.
2023), the smallest classification variant in the family. Images are resized to
224 by 224 and the network starts from pretrained weights. Training uses AdamW at
10⁻³ with cosine annealing, weight decay 10⁻⁴, batch size 32 and 50 epochs, with
random flip, rotation and brightness augmentation. The split is 80/20 stratified
over the 6,400 frames. The full thermal set was used, the classifier was trained
and validated on it, and the trained weights are among the released artifacts, so
the perception component reported here is the one the assembled pipeline runs.

**The split, and what it costs.** The sensor-side partitioning discipline was not
carried across to the images. That split is randomized over frames captured
consecutively within a single session, so it is exposed to the same leakage
mechanism the block-wise holdout was built to avoid, and adjacent thermal frames
are close to duplicates. The figure below is an optimistic upper bound rather
than an independent estimate. This leaves the two halves of the pipeline held to
different evidential standards, which is stated in the manuscript's limitations
rather than left to this file. A block-wise re-split of the image set is
outstanding work.

**Results.**

The thermal classifier reached 98.8% top-1 accuracy on 1,280 held-out validation images (**Table S1**). All 15 misclassifications fell on the NoGas and Perfume boundary, so neither hazardous class was confused with a non-hazardous one, which is what matters once the class label is mapped to an action (**Fig. S1**).

| Gas class | Precision | Recall | F1 | Accuracy (%) |
|---|---|---|---|---|
| Mixture | 1.000 | 1.000 | 1.000 | 100.0 |
| NoGas | 0.990 | 0.963 | 0.976 | 96.3 |
| Perfume | 0.972 | 0.991 | 0.981 | 99.1 |
| Smoke | 0.991 | 1.000 | 0.995 | 100.0 |
| Overall | 0.988 | 0.988 | 0.988 | 98.8 |

**Table S1—Perception component, 1,280 held-out validation images.** Obtained under a randomized image split rather than the block-wise protocol used on the sensor side, so read as an optimistic upper bound rather than an independent estimate. These figures are carried over unchanged from the previous version: the thermal classifier is trained outside the sensor pipeline and was not affected by either of the partitioning corrections, and the block-wise re-split of the image set remains outstanding.

![](figures_v2/fig_perception.png)

**Fig. S1—Thermal classifier behavior on 1,280 held-out validation images.** (a) Counts, (b) row-normalized, (c) per-class scores.


---

## S2 Explanation Component

Explanations are generated locally with Gemma 3 1B served through Ollama (Gemma
Team 2025), prompted with the gas class, classification confidence, normalized
anomaly score and threshold, and the selected action, capped at 150 tokens.
Nothing leaves the device.

The component sits off the safety-critical path by construction. The action is
fixed by the decision component before any text is generated, and the generated
text cannot alter it. That separation was deliberate: a fluent explanation
inconsistent with the decision it claims to explain is a hazard in a control room
rather than an aid, and the only structural defense against it is to deny the
generator any influence over the action.

The component is described as part of the system and is evaluated nowhere in this
work. It is not timed, not inspected and not assessed for faithfulness. No result
in the manuscript depends on it, and it should be read as apparatus rather than as
a contribution.

---

## S3 Cost Asymmetry and the Ordinal Objective

The manuscript reports this experiment's conclusions. The sweep, the objective,
the cost matrix and the action-distribution analysis are here.

The loss-weight ratio *C* from Eq. 3 was swept across nine values with everything else fixed (**Table S2**, **Fig. S2**).

| Cost ratio *C* | Decision accuracy | SD | Missed-hazard | Escalation | High-severity FA |
|---|---|---|---|---|---|
| 1:1 (symmetric) | **0.9646** | 0.0252 | 0.0000 | 1.000 | 0.0000 |
| 2:1 | 0.9584 | 0.0317 | 0.0000 | 1.000 | 0.0000 |
| 4:1 | 0.9620 | 0.0292 | 0.0000 | 1.000 | 0.0000 |
| 6:1 | 0.9593 | 0.0310 | 0.0000 | 1.000 | 0.0000 |
| 8:1 (reference) | 0.9630 | 0.0270 | 0.0000 | 1.000 | 0.0000 |
| 10:1 | 0.9631 | 0.0251 | 0.0000 | 1.000 | 0.0000 |
| 12:1 | 0.9573 | 0.0338 | 0.0000 | 1.000 | 0.0000 |
| 16:1 | 0.9241 | 0.0979 | 0.0000 | 1.000 | 0.0000 |
| 20:1 | 0.9090 | 0.0962 | 0.0000 | 1.000 | 0.0000 |
| *Label function* | *1.0000* | *0.0000* | *0.0000* | *1.000* | *0.0000* |

**Table S2—Cost-asymmetry sweep,** mean ± SD across five runs per configuration varying partition and initialization. The label-function row applies *y*(·) of Eq. 2a to the true class label. It reaches the ceiling by construction, consumes the label it is meant to infer, and is not a deployable baseline.

Missed-hazard rate is zero and escalation adequacy is 1.000 at every ratio, the symmetric 1:1 configuration included. Accuracy is highest at 1:1 (0.9646), sits at 0.9630 at the reference 8:1, and falls away past 12:1 to 0.9090 at 20:1, with variance up roughly fourfold. The asymmetry produced no measurable safety benefit at any setting and cost accuracy at the extremes.

The reason looks structural. The target action is a deterministic function of a well-separated class label, and a miss under this definition requires assigning a hazardous window to the class furthest from it in feature space. The miss rate is therefore already on the floor before any weighting is applied. There is nothing for the asymmetry to reshape. That is not evidence against cost-sensitive learning in general, since the literature is clear it works where the error surface is non-trivial (Elkan 2001). What this benchmark shows is narrower: it does not present a sufficiently difficult error surface for a measurable safety benefit from scalar class-level cost weighting to appear.

The reference configuration deserves one more note. *C* = 8:1 is not the accuracy optimum and was originally chosen on test-partition dispersion, which is not a defensible criterion. Re-selecting it needs a validation partition, and the obvious construction fails instructively: a uniformly random validation subset carved from training cannot discriminate between ratios at all, because it inherits exactly the window overlap Experiment 1 measures. Carving the validation band contiguously from the end of each class block instead, with the same embargo, restores discrimination and selects *C* = 6:1 (validation 0.9584, test 0.9397), with 1:1 second at 0.9580 and the reference 8:1 fifth at 0.9551. The reference is not selected under either criterion.

*An ordinal objective, since the evidence points at one.* Eq. 3 prices all wrong actions alike, so under-escalating a hazardous window and over-escalating it cost the same. That is a testable claim, so we tested it. We replaced the sample-weighted cross-entropy with direct minimization of expected cost,

&nbsp;&nbsp;&nbsp;&nbsp;*L*~ord~(θ) = (1/*N*) Σₜ Σ_a *p*~θ~(*a* | φₜ) · *M*[*g*ₜ, *a*],  ............ (S1)

under a matrix *M* that prices distance and direction along the action ladder. For a hazardous class, silence costs 10, a sub-alarm response 5, and notifying at the wrong tier 1. For clean air, a high-severity alarm costs 4 against 1 for a nuisance action. For the volatile-organic-compound class, silence costs 2 and a high-severity alarm 3. The structure, not the exact values, is what the experiment tests. This is a single untuned matrix, chosen by hand for its shape and run once, offered as a test of a mechanism rather than as a model we are proposing.

It works on the quantity it targets. On the held-out Mixture class the ordinal objective escalates 0.393 ± 0.235 of hazardous windows against 0.245 ± 0.199 for the cost-weighted policy, a relative improvement of about 60% while still recording no observed miss (Table 12 of the manuscript). It is the only change to the objective anywhere in this study that moved escalation adequacy at all. It also costs more than it returns as specified: in-distribution accuracy falls to 0.7472 ± 0.0054, some 21 points below the cost-weighted policy, and expected cost under the matrix the objective itself minimizes rises from 0.0631 to 0.2585. The mechanism is visible in the action distribution, where the model stops emitting *Monitor* for clean air and sends it to *Increase sampling* instead. The matrix prices that error at 1 against 10 for a missed hazard, and the NoGas and Perfume classes overlap precisely where that trade is decided. The result is a system in a permanent state of mildly elevated sampling.

What has been demonstrated is narrow. The mechanism is confirmed: pricing direction along the action ladder moves escalation, where scalar reweighting did not. What has not been demonstrated is a cost matrix worth deploying. A matrix that buys 15 points of escalation on a held-out hazardous class by paying 21 points of accuracy on clean air is not one we would put in front of an operator. A single untuned draw cannot locate the optimum over the space of such matrices. The experiment demonstrates that a directional cost structure can alter escalation behavior. It does not establish an optimal cost matrix. It is evidence about this particular matrix under this training procedure, not about ordinal objectives in general, and the row the objective occupies in the manuscript's Tables 8 and 12 is a mechanism result rather than a deployable policy.

![](figures_v2/fig_costsweep.png)

**Fig. S2—Cost-asymmetry sweep.** Shading is ± 1 SD across five seed-varying partitions.


---

## S4 Confidence Calibration

Confidence matters here because any escalation threshold, and any deferral rule, would be set against it. Four estimators were compared: the raw softmax, MC dropout with 20 passes, temperature scaling, and a five-member deep ensemble built from the same model class and objective as the single model. Temperature is fitted on a calibration split carved out of training, never on the training logits themselves. Expected calibration error uses equal-mass bins because confidence on this corpus piles up near 1 and equal-width bins leave most of the range empty (**Table S3**, **Fig. S3**).

| Estimator | ECE | SD | Block-bootstrap 95% CI | ECE on hazardous windows | Brier | NLL | Accuracy |
|---|---|---|---|---|---|---|---|
| MC dropout (20) | **0.0275** | 0.0109 | 0.0102 to 0.0569 | 0.0003 | 0.0903 | 0.1644 | 0.9386 |
| Temperature scaling | 0.0358 | 0.0110 | 0.0182 to 0.0662 | 0.0043 | 0.0935 | 0.1786 | 0.9392 |
| Deep ensemble (5) | 0.0363 | 0.0116 | 0.0134 to 0.0665 | 0.0002 | 0.0928 | 0.1864 | 0.9413 |
| Raw softmax | 0.0431 | 0.0124 | 0.0171 to 0.0755 | 0.0002 | 0.1000 | 0.2089 | 0.9392 |

**Table S3—Calibration,** mean ± SD across five runs varying partition and initialization. ECE with 15 equal-mass bins; intervals from 400 moving-block bootstrap resamples per run, blocks of 20 windows, so the resampling does not assume independent windows.

MC dropout has the lowest point estimate of ECE, but every bootstrap interval overlaps every other, so the four estimators are not separated in this experiment. The lowest point estimate has changed hands twice across the corrections applied to this study, which is a further reason not to read an ordering into it. Two observations are more useful than the ordering. First, overall ECE is driven entirely by the non-hazardous classes: restricted to Smoke and Mixture windows every estimator is calibrated to within 0.02%, and temperature scaling to within rounding. The miscalibration lives on the NoGas and Perfume boundary, the same place the perception errors and the anomaly-score overlap live. For a system whose escalation threshold would be set on hazardous-class confidence that is reassuring, and it is invisible in a single pooled ECE number.

Second, the temperature itself carries a methodological lesson about this study rather than about temperature scaling. Under an earlier implementation the calibration split was a uniformly random subset of the training partition, over windows overlapping by 19 of 20 raw rows, so the temperature was effectively fitted against near-copies of the data it was fitted on. It came out consistently below 1, and the previous version of this work concluded that the network was under-confident. Carving the calibration band contiguously from the end of each class block, with the same 20-window embargo used for the outer holdout, reverses that: the fitted temperature is 1.423 ± 0.202 across runs and never below 1.20, so the network is over-confident, which is the ordinary finding. The earlier conclusion was an artifact of a leaky calibration split, and it is reported here because it is a clean illustration of the paper's own argument turning up inside the paper's own analysis. The bin sweep in Fig. S3c shows the ordering across estimators is stable from 5 to 30 bins.

![](figures_v2/fig_calibration.png)

**Fig. S3—Calibration.** (a) ECE with bootstrap intervals. (b) The same, split into all windows against hazardous windows only. (c) Sensitivity to the number of bins.


---

## S5 Comparator Specifications

Two comparators are not plain supervised classifiers, and the manuscript's
one-line descriptions of them are not enough to reimplement or to judge. Their
full construction is here.

*Conservative Q-learning.* The setting is offline and single-step. The behavior policy is the deterministic target map *y*(·) of Eq. 2a, so the behavior action for a window is *y*(*g*ₜ) and the offline dataset is exactly the supervised training partition re-read as (*s*, *a*_data, *r*, *s*′) tuples. Coverage is therefore complete over the state space and empty over every non-behavior action, which is the regime the conservative penalty exists for. The reward is the asymmetric reward of Eq. 3's cost structure, with a missed hazard priced at 10 against 1 for a false alarm, and it is a function of (class, action) alone. The discount is γ = 0.99 and the target is a TD(0) bootstrap *r* + γ max\_*a*′ *Q*(*s*′, *a*′), fitted with a smooth L1 loss. The conservative term is the standard CQL(H) penalty, α (logsumexp\_*a* *Q*(*s*, *a*) − *Q*(*s*, *a*_data)) with α = 1.0, which pushes down the value of actions the behavior policy never took. Successor states are the next row in acquisition order within the training partition, and the final row of the partition is masked out because it has no successor. There are no episodes and no terminal states in any meaningful sense: monitoring is a continuing task, each window is one step, and the 0.99 discount over a one-step bootstrap makes the agent close to a myopic cost minimizer. That is what the comparator is, and it is why it tests whether an offline value-based objective beats a cost-weighted supervised one rather than whether reinforcement learning solves monitoring.

*CUSUM.* The statistic is one-sided on the anomaly score, *S*ₜ = max(0, *S*ₜ₋₁ + *x*ₜ − μ₀ − *k*), with μ₀ and σ estimated from the training partition's NoGas windows, a slack of *k* = 0.5σ, and the alarm threshold *h* set at the 99th percentile of the statistic's own path over those training windows. Reset semantics matter for a sequential detector and are stated here rather than left to the code: *S*ₜ is zero at the first window of every evaluation set and is reset to zero at every change of class block, so it is never carried across a block boundary, across the embargo band, or between partitions. Within a block it accumulates normally. An earlier implementation reset only at the start of an evaluation set, which let the block following a hazardous one inherit an already-elevated statistic; the results reported here use the per-block reset.


---

## S6 Sensitivity of the Equivalence Conclusions

The manuscript reports the conclusions of these two sweeps. The tables and the
full reading are here.

| Comparison | ρ = 0 | ρ = 0.05 | ρ = 0.1 | ρ = 0.2051 | ρ = 0.3 | ρ = 0.4 | ρ = 0.5 |
|---|---|---|---|---|---|---|---|
| Unweighted network | 1.000 | 1.000 | 1.000 | 0.992 | 0.969 | 0.925 | 0.861 |
| Recurrent (LSTM) | 0.991 | 0.934 | 0.878 | 0.779 | 0.700 | 0.622 | 0.545 |
| Conservative Q-learning | 0.981 | 0.907 | 0.847 | 0.756 | 0.688 | 0.619 | 0.550 |
| Multilayer perceptron | 0.971 | 0.884 | 0.812 | 0.691 | 0.600 | 0.517 | 0.443 |
| Gradient boosting | 0.924 | 0.805 | 0.719 | 0.586 | 0.496 | 0.419 | 0.354 |
| Random forest | 0.871 | 0.747 | 0.665 | 0.539 | 0.456 | 0.385 | 0.325 |
| Support-vector classifier | 0.493 | 0.493 | 0.484 | 0.445 | 0.401 | 0.353 | 0.307 |
| k-nearest neighbors | 0.240 | 0.329 | 0.360 | 0.367 | 0.347 | 0.316 | 0.281 |
| Shallow decision tree | 0.000 | 0.000 | 0.004 | 0.027 | 0.056 | 0.085 | 0.107 |
| Ordinal cost objective | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

**Table S4—Sensitivity of the Bayesian equivalence probability to the assumed run correlation ρ.** The deployed value is 1,264/(4,900 + 1,264) = 0.2051, read from the partition rather than fixed by hand, and that column is identical to the final column of Table 8 of the manuscript. Raising ρ widens the posterior and moves mass out of the region of practical equivalence, so the probabilities fall monotonically. Two comparisons change their most probable outcome across the range, and both are named below.

| Comparison | ±0.5 pt | ±1 pt | ±2 pt | Most probable outcome, ±0.5 → ±1 → ±2 |
|---|---|---|---|---|
| Unweighted network | 0.835 | 0.992 | 1.000 | equivalent → equivalent → equivalent |
| Recurrent (LSTM) | 0.452 | 0.779 | 0.987 | **other better → equivalent → equivalent** |
| Conservative Q-learning | 0.421 | 0.756 | 0.987 | **reference better → equivalent → equivalent** |
| Multilayer perceptron | 0.389 | 0.691 | 0.957 | **other better → equivalent → equivalent** |
| Gradient boosting | 0.317 | 0.586 | 0.894 | **reference better → equivalent → equivalent** |
| Random forest | 0.288 | 0.539 | 0.858 | **other better → equivalent → equivalent** |
| Support-vector classifier | 0.228 | 0.445 | 0.783 | **reference better → reference better → equivalent** |
| k-nearest neighbors | 0.183 | 0.367 | 0.702 | **reference better → reference better → equivalent** |
| Shallow decision tree | 0.011 | 0.027 | 0.101 | reference better → reference better → reference better |
| Ordinal cost objective | 0.000 | 0.000 | 0.000 | reference better → reference better → reference better |

**Table S5—Sensitivity of the equivalence conclusion to the width of the region of practical equivalence.** Columns give the posterior probability of practical equivalence to the cost-weighted policy under Eq. 11 at the deployed ρ. The final column gives the most probable of the three outcomes at each width; rows in bold change their answer across the sweep, and 7 of 10 do. The paired bootstrap reaches the same most probable outcome at every width.


Eight of the ten comparisons keep their most probable outcome across the whole range. The decision tree and the ordinal objective remain the more probable losers at every ρ, and the unweighted network stays practically equivalent to the reference even at ρ = 0.5, where the posterior is far wider than this design warrants, though its strength falls from 1.000 to 0.861. Two comparisons change. The random forest is the most probable equivalent up to ρ = 0.3 and the more probable winner at 0.4 and 0.5, and gradient boosting is equivalent up to ρ = 0.4 and the more probable loser at 0.5. Both flips occur only at correlation values above anything this design gives reason to assume, but they are the honest measure of how much the assumption is carrying.

The ROPE width carries considerably more than ρ does, and Table S5 is the more informative of the two sweeps. Seven of the ten comparisons change their most probable outcome between ±0.5 and ±2 accuracy points. At the tightest width the reference is most probably better than four comparators and worse than three, and only the unweighted network is equivalent; at the widest, eight of ten are equivalent and only the decision tree and the ordinal objective are separated. The equivalence claims made in this paper are therefore claims made at a stated threshold, not properties of the models. Two survive the full sweep: the unweighted network is equivalent at every width, and the decision tree and the ordinal objective are worse at every width. Everything between those is width-dependent and should be read as such.

*A comparison that assumes no correlation at all.* Because ρ has to be assumed rather than estimated, we also ran a paired bootstrap in which the run is the resampling unit and nothing is assumed about structure inside a run. The 30 paired differences are resampled with replacement 20,000 times, and the same one-point region of practical equivalence is applied to the replicate means. This asks the same question under a weaker assumption. It agrees with the correlated *t*-test on the most probable outcome in all ten comparisons, and at all three ROPE widths. Under the bootstrap the unweighted network is practically equivalent to the reference in every replicate, interval [−0.003, +0.002] on the paired difference. The recurrent network is equivalent in 99.3% of replicates and the perceptron in 96.5%. The random forest is equivalent in 88.4% and better in the remaining 11.6%, interval [−0.013, +0.003]. The decision tree and the ordinal objective are worse in every replicate, by 3.9 and 21.0 accuracy points respectively. We therefore treat the ρ sweep as a sensitivity analysis rather than as the primary evidence, and rest the ranking claim on the agreement between the two.

The two analyses share every observation. The 30 runs differ in seed, block position, initialization and hyperparameter draw, but they are 30 passes over one recording session on one device, so they are not 30 independent physical experiments and the bootstrap does not replicate anything. What the agreement establishes is that the conclusion is not an artifact of the correlated *t*-test's assumed ρ, which is a real and useful robustness property and nothing more. Independent replication would require a second acquisition, and this study does not have one.


---

## S7 Decision-State Ablation and Upstream-Input Sensitivity

The manuscript reports the two conclusions of this experiment. The tables, the
figures and the full reading are here.

The decision state was ablated by feature group with the architecture and protocol held fixed (**Table S6**, **Fig. S4**).

| Decision state | *d* | Decision accuracy | SD | Δ vs. full | Missed-hazard | Escalation |
|---|---|---|---|---|---|---|
| Without per-sensor σ | 15 | 0.9653 | 0.0147 | +0.40 pp | 0.0000 | 1.000 |
| Full state | 22 | 0.9614 | 0.0139 | reference | 0.0000 | 1.000 |
| Without anomaly score | 21 | 0.9571 | 0.0215 | −0.43 pp | 0.0000 | 1.000 |
| Without per-sensor δ | 15 | 0.9465 | 0.0260 | −1.49 pp | 0.0000 | 1.000 |
| Without δ and σ | 8 | 0.8854 | 0.0798 | −7.59 pp | 0.0000 | 1.000 |
| Current readings only | 7 | 0.8804 | 0.0813 | −8.10 pp | 0.0000 | 1.000 |
| Anomaly score only | 1 | 0.6581 | 0.1009 | −30.33 pp | 0.0000 | 1.000 |

**Table S6—Decision-state ablation,** mean ± SD across five runs varying partition and initialization, with the architecture held fixed.

Temporal features carry the accuracy: dropping both δ and σ costs 7.59 points, and a single anomaly score on its own manages 65.8%. Two feature groups are not carrying their weight at all. Removing the anomaly score costs 0.43 points, and removing the per-sensor standard deviations *improves* accuracy by 0.40 points, both well inside the seed-to-seed spread, so the 22-dimensional state is not minimal. But missed-hazard rate stays at zero and escalation adequacy at 1.000 in every condition, including the one-dimensional state. No feature group changes the in-distribution safety metrics at all, which is another instance of the same trap: an ablation reported on accuracy alone would appear to identify which features drive safety behavior, when it is measuring which features drive the in-distribution decision metrics.

![](figures_v2/fig_ablation.png)

**Fig. S4—Decision-state ablation.** Bars are mean ± SD decision accuracy. Missed-hazard rate and escalation adequacy are constant across every condition.

Retraining without a feature answers whether the feature is necessary. The sharper question, if an upstream component can fail in service, is what a trained model does when that input goes wrong while everything else stays correct. The anomaly score was swept across its full range on every test window with the other twenty-one features held fixed, and the share of windows whose action moved was recorded (**Fig. S4**). This is a synthetic sensitivity probe, not a physical failure model.

The models divide sharply. For the decision tree the answer is zero by construction, since it reads only the raw channels. For k-nearest neighbors, the support-vector classifier and the perceptron the action moves on under 3% of windows. For the cost-weighted policy it moves on 6.65 ± 3.83% of windows and for the random forest on 23.59 ± 8.81%. Gradient boosting is the outlier at 64.22 ± 7.65%, so for that model the anomaly channel is load-bearing where for the others it is close to inert at inference time. The consequence shows up in the safety metrics, and only for that model: forcing the anomaly score to zero, which is what a failed or unconnected autoencoder would supply, leaves every other comparator's missed-hazard rate at zero and moves gradient boosting from 0.0092 to 0.4203. Escalation adequacy for that model falls from 0.992 to 0.497, so a model that answers essentially every hazardous window with an alarm-grade action answers half of them with silence once one upstream input fails. The measured safety behavior of this model on this benchmark is therefore highly sensitive to the anomaly-score input, and nothing in the accuracy column or the ablation table shows it.

A single conditional probe supports a narrow reading: the override is synthetic and the other features are held at values that in reality would co-vary with the anomaly score. It is consistent with two results already in hand, since gradient boosting is both the model that fails silently under sensor loss and the one that escalates the held-out Mixture class perfectly. Dependence on upstream components deserves to be measured rather than assumed.

![](figures_v2/fig_anominfluence.png)

**Fig. S5—Dependence of the selected action on the anomaly input,** five seed-varying partitions. (a) Share of test windows whose action changes at any point as the anomaly score is swept across [0, 1] with all other features held fixed, mean ± 1 SD. (b) Missed-hazard rate against the forced anomaly value. Six comparators are flat; gradient boosting moves from 0.0092 to 0.4203 when the channel reads zero.


---

## S8 What the Protocol Costs to Adopt

The metric set is inexpensive to adopt, and that cost is measured rather than asserted. Computing the five quantities of Table 5 over the real 1,264-window test partition takes 0.455 ms against 0.290 ms for accuracy alone, a ratio of 1.57×, which is arithmetic over predicted actions the evaluation already has; the episode-level quantities are the same kind of pass over the same predictions (`retrain/run_metric_cost.py`, `v3_metric_cost.json`). Both figures are means over 200 timed repetitions after warm-up and are machine-dependent, so the ratio rather than either absolute figure is the transferable quantity. The cost that is not negligible is retraining, and it falls unevenly. Reporting the metric set on an existing evaluation needs none, and neither do the perturbation suite or the upstream-input probe, since both re-evaluate an already-trained model under modified inputs. Leave-one-class-out needs one retrain per withheld hazardous class, so two here. The 30-run comparison with the equivalence test is the only element that costs a multiple of the original training budget, and it is needed only when the question is model selection rather than model characterization. A facility qualifying a single candidate model therefore pays almost nothing for the screen in Table 12 of the manuscript. A study ranking candidates pays for the run count, and Table 8 of the manuscript shows what that buys.


---

## S9 Sensitivity of the Eventized Alarm Result to the Persistence Rule

Table 13 of the manuscript collapses alarm-grade windows into annunciated episodes under one rule: an on-delay of three windows and an off-delay of fifteen. That rule is a stated convention, not an industrially validated persistence specification, and the manuscript uses the eventized view to reorder the models relative to the raw indication view. If the ordering is a property of the convention rather than of the models, the reordering claim has to be weakened. The question is therefore checked directly.

Models are fitted once per seed, the alarm-grade flag sequence is computed once per model, seed and condition, and the same flag sequences are then eventized at every point of a grid of on-delays {1, 2, 3, 5, 10} against off-delays {5, 10, 15, 30, 60}, twenty-five settings in all. Nothing is refitted between grid points, so every difference below is caused by the persistence rule alone. Rank agreement with the deployed setting is measured per condition by Kendall's τ over the six comparators, and every pairwise reversal is recorded; with fifteen model pairs and twenty-four comparison settings there are 360 opportunities for a reversal in each condition (`v3_persistence_sweep.csv`, `v3_persistence_rank.json`).

| Condition | Models annunciating at 3/15 | Highest episode rate at 3/15 | Highest over the grid | Kendall τ, min | Reversals |
|---|---|---|---|---|---|
| Clean | 0 of 6 | 0.00 | 0.00 | — | n/a |
| One channel lost | 4 of 6 | 5.70 | 10.25 | 0.56 | 14 |
| Four channels lost | 5 of 6 | 10.25 | 19.37 | 0.74 | 2 |
| Total sensor loss | 5 of 6 | 7.97 | 10.25 | 0.50 | 9 |
| Drift ±30% | 0 of 6 | 0.00 | 0.00 | — | n/a |
| Drift ±50% | 0 of 6 | 0.00 | 2.28 | — | n/a |
| Noise σ = 0.3 | 0 of 6 | 0.00 | 50.13 | — | n/a |
| Noise σ = 0.5 | 3 of 6 | 5.70 | 109.37 | −0.78 | 22 |

**Table S7—Sensitivity of the eventized alarm result to the persistence rule,** over twenty-five on-delay and off-delay settings applied to the same alarm-grade flag sequences. Episode rates are per hour under the illustrative duty cycle of Eq. 12. Kendall's τ compares the ordering of the six comparators at each setting against the ordering at the deployed 3/15 rule; "reversals" counts pairwise order changes out of 360 model-pair by setting comparisons. A dash marks a condition where no ordering exists at the deployed rule because no model annunciates there.

Three findings follow, and two of them weaken claims made in the manuscript.

The channel-loss conditions are reasonably stable. Mean τ against the deployed rule is 0.84, 0.93 and 0.93 at one, four and seven lost channels, and reversals occur in 14, 2 and 9 of 360 comparisons. The specific reordering the manuscript relies on survives the entire grid: at single-channel loss the shallow tree annunciates no more often than the cost-weighted policy at every one of the twenty-five settings, strictly less at twenty of them and equal at the remaining five, with no setting reversing it. The secondary observation, that the cost-weighted policy annunciates more at four lost channels than at seven, holds at twenty-one settings, ties at two and reverses at two, both of which have an off-delay of sixty windows, two minutes of required quiet. It is reported in the manuscript with that qualification.

The noise conditions are not stable, and this is the finding that matters. At σ = 0.5 the minimum τ is −0.78, an ordering close to the reverse of the deployed one, with 22 reversals. The magnitudes move further than the ranks: the highest episode rate at any comparator runs from 0 to 109.37 per hour across the grid, and which models annunciate at all changes with the rule. At σ = 0.3 the effect is starker still, because the deployed rule reports zero annunciated episodes from every comparator while a shorter on-delay reports up to 50.13 per hour. A conclusion drawn at 3/15 alone would have recorded that condition as producing no operator-facing alarms whatsoever.

The consequence for the manuscript is specific. The compression result and the channel-loss reordering stand. The statement that every condition tested falls within or near the EEMUA reference once eventized does not: it holds at the deployed rule and fails under shorter on-delays, so it is a property of the persistence convention and has been corrected in the manuscript to say so. The general lesson is the one the persistence rule was introduced to make, turned on the rule itself. Eventization is the right level at which to discuss operator load, and the level at which a facility's own persistence specification, rather than a convention chosen by an evaluator, has to supply the numbers.

---

## S10 Reproducing These Results

`make comparison` produces Table S2 and Fig. S2. `make calibration` produces
Table S3 and Fig. S3 together with the perception figures. The verification
script checks every value in this file against the same result files it checks
the manuscript against, so a number here cannot drift from a number there
without the check failing.
