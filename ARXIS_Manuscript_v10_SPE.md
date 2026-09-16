# ARXIS: Safety-Relevant Evaluation of Gas-Hazard Monitoring Models Under Distribution Shift and Sensor Degradation

**Benjamin C. Nweke**^1,2,\*, **Gholamreza Ramezan**^2, **Soheil Saraji**^1

^1 Subsurface Energy and Digital Innovation Center, University of Wyoming
^2 Fides Innova Labs

\* Corresponding author; email: bnweke@uwyo.edu

*Prepared for submission to SPE Journal, Data Science and Engineering Analytics.*

**Keywords:** safety-relevant evaluation; gas-hazard monitoring; distribution shift; alarm management; cost-sensitive learning

---

## Summary

Fixed gas detectors at wellpads, tank batteries and compressor stations generate far more abnormal indications in a week than any operator can act on, and the machine-learning literature that interprets those indications is ranked almost entirely by classification accuracy. Accuracy is silent on the two quantities that decide whether a monitoring system is safe to install: how often a hazard draws no response at all, and how often clean air draws a high-severity one. We evaluated a four-stage monitoring pipeline against a five-quantity safety metric set on the public MultimodalGasData corpus, under a leakage-controlled block-wise holdout, with an exact confidence bound on every observed zero. The corpus contains incense smoke, alcohol-based vapor and a mixture of the two, not hydrocarbons, so what follows characterizes evaluation methodology on a representative metal-oxide array rather than field performance. Seven experiments over twelve models produced four results. Among models that read temporal features, the partitioning protocol moves reported accuracy by more than the choice of model does, a random split over overlapping sliding windows adding between 1.66 and 4.08 accuracy points. In distribution the safety metrics are weakly discriminative: ten comparators lie within 5.66 accuracy points over 30 randomized runs, and all but one record no observed missed hazards, escalation adequacy of 1.000 and no observed high-severity false alarms, with removing the cost weighting practically equivalent to keeping it under both resampling analyses we ran. Withholding one hazardous class then separates those same models by more than two orders of magnitude on missed-hazard rate, a model mid-table on accuracy proving the worst in the field. And missed-hazard rate is necessary but not sufficient: ten of eleven comparators have 95% upper confidence bounds at or below 3.3% on missed hazards while escalation adequacy across them runs from 0.005 to 1.000, so a review conducted on miss rate alone would clear a model that alerts an operator to one hazardous window in two hundred. Under graded sensor loss three distinct failure modes appear, and each single metric certifies a different model as safe. We therefore recommend reporting missed-hazard rate with a confidence bound, escalation adequacy, false-alarm burden in alarm-management units, and the degradation of all three under sensor loss and drift, together rather than singly.

---

## Introduction

Serious process-safety incidents are rarely preceded by silence. The warning is usually present, spread across several indications, and simply never converted into action in time. In the eleven minutes before the Milford Haven refinery explosion the control-room operator received 275 alarms (Goel et al. 2017). Sensing was not the failure. Turning what was sensed into a decision was.

The same shape of problem appears at upstream and midstream facilities. Wellpads, tank batteries, compressor stations and gas plants carry fixed point-gas detectors, often low-cost metal-oxide-semiconductor (MOX) elements, feeding alarm systems that in many installations already run at or above the long-term average rates industry guidance treats as manageable for a single operator (EEMUA 2013; ISA 2016). Added to that is the growing obligation to find and act on fugitive hydrocarbon release, where the response is at once a safety action and an emissions action. In both settings the operational question is what to do in the next minute, and with what confidence.

Machine learning has been applied heavily to the first question and hardly at all to the second. Gas-detection models are ranked by classification accuracy, and published results in this area are high and closely spaced. They are also not directly comparable, because studies differ in dataset, modality, task definition and partitioning protocol, so a single quoted range across them would mislead more than it informs. Table 1 sets the representative results out with those columns attached. What the comparison shows is that reported accuracy on this benchmark lineage is uniformly high, and that it is sensitive to evaluation design in ways the headline numbers do not expose. Once that is true, small differences between leading methods stop carrying information about deployment, and what remains open is what accuracy does not measure.

**A Symmetric Metric on an Asymmetric Problem.** Monitoring errors are not interchangeable, and no single scalar orders them sensibly. A hazardous condition that draws no response gives up the whole intervention window. A hazardous condition that draws a sub-alarm response, such as increased sampling or a request for verification, keeps part of that window but notifies nobody, and a miss metric treats it as a success. A clean condition that draws a high-severity response spends operator attention, and at sufficient rate degrades the alarm system that later hazards depend on (Laberge et al. 2014). A hazardous condition that draws the wrong high-severity response is, in most operating regimes, almost as good as the right one. A model ranked first on accuracy can be ranked last on any of these, and the accuracy table gives no hint of which.

**What Is Settled and What Is Not.** Most of the component technology is mature. Reconstruction-based sequence models trained only on normal operation have been a widely used unsupervised approach to multivariate sensor anomaly detection since Malhotra et al. (2016), with field-validated applications in natural-gas pipeline monitoring (Liang et al. 2023). Vision models classify thermal and optical gas signatures fast enough for embedded hardware (Korjani et al. 2024; Jocher et al. 2023). Cost-sensitive learning has a clean formal account of unequal error costs (Elkan 2001). Fusing thermal imagery with MOX arrays improves classification relative to either alone (Narkhede et al. 2021, 2022). Compact language models make local explanation feasible without sending monitoring data off site (Dehnaw et al. 2024). What is not settled is how any of this behaves when ranked by something other than accuracy. Five questions follow.

**RQ0.** Does the choice of partitioning protocol change what a comparison reports?

**RQ1.** Does accuracy-ranked model selection survive a missed-hazard criterion?

**RQ2.** Do models that agree in distribution still agree when the hazard is one they were never trained on? A detector in service will eventually meet a hazardous condition outside its training classes.

**RQ3.** Is missed-hazard rate itself sufficient, or can it be satisfied by under-response that a miss count never registers?

**RQ4.** Do any of these behaviors survive the sensor faults a fixed detector actually meets in service, or do they hold only on clean data?

This study contributes in the following ways.

- **An evaluation protocol for safety-relevant comparison.** A leakage-controlled block-wise holdout with an explicit embargo and a seed-varying block position; a five-level action target that keeps missed hazard, under-escalation and high-severity false alarm apart; leave-one-class-out evaluation against a held-out hazardous class; a graded perturbation suite covering additive noise, calibration drift and channel loss at five to seven severities; and exact Clopper-Pearson bounds on every zero count. We did not find a prior study combining these elements on a MOX gas-detection benchmark, though we make no claim to have searched exhaustively.
- **Quantification of the leakage the protocol exists to prevent.** The argument that a random split over overlapping sliding windows inflates accuracy is standard but, as far as we can tell, unmeasured on this corpus. We measure it: among models that read temporal features, the protocol effect exceeds the spread between the models themselves.
- **Evidence that accuracy ranking carries little information about safety behavior.** Ten models spanning very different inductive biases span 5.66 accuracy points in distribution, are indistinguishable there on every safety metric we report, and then separate by more than two orders of magnitude on missed-hazard rate once the hazardous class is held out of training.
- **Evidence that missed-hazard rate is necessary and not sufficient.** On a held-out hazardous class, models whose missed-hazard upper bounds all sit at or below 3.3% span the full range of escalation adequacy, from 0.005 to 1.000, and the group with the lowest miss rates contains both the best and among the worst escalators.
- **Two negative results.** On this benchmark the tested scalar class-level cost ratios produced no measurable change in missed-hazard rate or escalation adequacy at any value from 1:1 to 20:1, and degraded accuracy past 6:1. Separately, a deterministic label-function reference reaches the accuracy ceiling by construction, which means any benefit from a learned policy on this corpus must come from degraded conditions rather than nominal ones.
- **Comparators a facility would recognize.** A depth-3 decision tree on seven raw sensor channels and a CUSUM sequential detector run under the identical protocol, so the learned models are measured against interpretable alternatives rather than only against other neural networks.

This study does not aim to advance classification accuracy on this benchmark, which has saturated. What remains open is what those numbers miss.

---

## Prior Research Review

**Detection, and the Gap Between an Indication and a Decision.** Gas-hazard detection rests on a well-developed base of hardware and software methods (Murvay and Silea 2012; Adegboye et al. 2019), including deep-learning electronic-nose approaches to pollutant classification (Faleh and Kachouri 2023). On the hardware side sit distributed acoustic and fiber-optic sensing, vapor sensing and negative-pressure-wave detection. On the software side sit real-time transient modeling, mass balance, statistical analysis and model-based diagnostics (Zhang et al. 2013; Bustnes et al. 2011). These have improved identification and localization considerably, and their weak spot is usually uneven performance under real operating conditions rather than any absence of capability (Han and Kim 2014).

For the present argument a different limitation matters more. All of them stop at a leak indication, a location estimate or a concentration estimate. Those outputs are necessary for loss prevention, but they are not response decisions, and the literature evaluating them does not ask whether the downstream response would have been adequate. Detection quality is being reported as though it settled the response question, which it does not reach.

**Anomaly Scores as Evidence Rather Than Verdicts.** Since Malhotra et al. (2016), reconstruction-based sequence models trained only on normal operation have been a widely used unsupervised route to multivariate sensor anomaly detection, and they have been used in pipeline monitoring with field validation (Liang et al. 2023). Their appeal for safety work is obvious, since no labeled examples of every hazardous condition are needed.

An anomaly score is nevertheless an intermediate quantity. It says behavior has departed from baseline. It does not say what to do, and nothing in the training procedure calibrates it to hazard severity. Any use of such a score inside a safety path carries an obligation to show that it ranks hazard monotonically, and the literature rarely discharges that obligation. We test it directly below.

**A Saturated Benchmark.** MultimodalGasData pairs thermal frames with a seven-channel MOX array, and the original work showed that fusing the two beats either alone: 96% for the fused model against 82% for the MOX channels alone and 93% for thermal images alone (Narkhede et al. 2021, 2022). It has since become a reference corpus, and later results are high but not mutually comparable. Sharma et al. (2024) report 99.7% validation accuracy for a multimodal federated model on the same four-class corpus. El Barkani et al. (2024) report 91.7% test accuracy, but on the thermal images alone and on embedded hardware. Zhang and Zhang (2025) report 98.9% accuracy, and that figure belongs to a simulated pipeline test set of their own collection rather than to this corpus, which they use only for initial training of a binary leak classifier; on real field data from a methane emissions test facility the same system reports 95.4%. Quoting those four numbers as a single range would compare a four-class multimodal task, a single-modality embedded task and a binary leak task across three different test sets.

Two features of that literature bear on what follows. First, those figures appear to be computed on random splits over overlapping sliding windows drawn from a single recording session. Dennler et al. (2022) demonstrated analogous drift-related evaluation problems in a widely used MOX benchmark, where recording time is confounded with class identity, and flagged a sizeable body of published work for re-evaluation; Vergara et al. (2012) had already documented how far MOX response drifts over long horizons. Second, and more to the point, the cited studies emphasize classification or detection performance rather than the action-level safety quantities evaluated here: missed-hazard rate, escalation adequacy, false-alarm burden and behavior under sensor loss. The benchmark has saturated on a metric that does not discriminate deployment fitness.

**Cost-Sensitive Learning.** That error costs are unequal, and that classifiers ought to be trained accordingly, has had a clean formal treatment since Elkan (2001), and cost-sensitive reweighting is now routine practice. What gets reported far less often is whether a given asymmetry actually moved the specific error it was introduced to suppress. We run that test and return a negative answer, one we suspect generalizes to any saturated, well-separated benchmark. The follow-up question, whether an objective that prices direction along a graded response ladder does better than a scalar weight, is one the evidence here points toward, and we test that too.

**Coordination Layers and Operator-Facing Explanation.** Systems that coordinate several specialized models toward a larger task have been applied in the energy sector to well-construction assistance, completions optimization and enterprise modeling (Sabbagh et al. 2024; Santiago et al. 2025; Jessen and Roshchin 2025). Those are analytical and advisory settings rather than real-time monitoring, and their evaluation criteria differ accordingly.

Explanation matters for operator-facing safety systems because trust, review and post-incident accountability all depend on it. It also brings a hazard of its own. A language model can produce an explanation that reads fluently and does not match the decision it claims to explain, which in a control room is worse than offering nothing. Our response to that is architectural rather than empirical, and we describe it below.

**Where This Study Sits.** **Table 1** lists representative prior work with the evaluation gap each leaves open, and states our own limitation in the same column so the comparison is symmetric.

| Study | Dataset | Modality and task | Validation protocol | Reported result | Evaluation gap |
|---|---|---|---|---|---|
| Narkhede et al. (2021, 2022) | MultimodalGasData | MOX + thermal, 4-class | reported test split over a single session | 96% fused, 82% MOX only, 93% thermal only | Classification accuracy only; random split over overlapping windows |
| Sharma et al. (2024) | MultimodalGasData | MOX + thermal, 4-class, federated | reported validation split | 99.7% validation | No missed-hazard, escalation, false-alarm or degradation reporting |
| El Barkani et al. (2024) | MultimodalGasData, thermal subset | thermal only, 4-class, embedded | reported test split | 91.7% | Single modality; no action-level or degraded-sensing metric |
| Zhang and Zhang (2025) | own simulated set; this corpus for pretraining | MOX + thermal, binary leak | simulated test set; separate field set | 98.9% simulated, 95.4% field | Different task and test set; not comparable to four-class results here |
| Dennler et al. (2022) | a different MOX benchmark | drift analysis | re-partitioned | not applicable | Demonstrates the leakage mechanism; supplies no safety metric |
| Liang et al. (2023) | pipeline flow data | normal-only anomaly detection | field validated | not applicable | Stops at detection; downstream response unevaluated |
| Korjani et al. (2024) | optical gas imaging | vision detection | field imagery | not applicable | Degraded-sensing behavior not treated as a safety metric |
| Elkan (2001) | not applicable | cost-sensitive learning theory | not applicable | not applicable | Effectiveness on a saturated benchmark untested |
| **This study** | **MultimodalGasData** | **decision from MOX and anomaly state; thermal in the perception branch only** | **block-wise holdout, 20-window embargo, seed-varying block** | **see Tables 9 and 12** | **Laboratory surrogate analytes; single session and device; perception path not held to the same partitioning standard** |

**Table 1—Representative prior work with dataset, task and validation protocol.** The columns are present because the reported results are not comparable without them.

Our contribution lives in that right-hand column. It is not a new architecture. It is evidence about what the usual evaluation of these architectures fails to reveal.

---

## Proposed Method

The method has two halves, and only the second is novel. The first is a monitoring pipeline assembled from mature components. The second is the evaluation protocol and metric set that this study proposes, and which the pipeline exists to be measured by. **Fig. 1** gives the whole workflow: what is guarded against at each stage, what the protocol does, and which evaluation regimes the trained models are then put through.

![](figures_v2/fig_workflow.png)

**Fig. 1—Workflow of the evaluation protocol, from raw corpus to the reported metric set.** Left column, the failure each stage prevents; right column, the four evaluation regimes.

**Corpus and Preprocessing.** The public MultimodalGasData corpus (Narkhede et al. 2022; Mendeley Data, CC BY 4.0) pairs a seven-channel MQ-series MOX array with synchronized 206 by 156 thermal frames from a Seek Compact camera, logged at 2 s intervals. MOX elements respond broadly and cross-sensitively, so a reading on MQ-2 is evidence of a reducing-gas response rather than of methane specifically; the array is informative because the seven responses differ in relative magnitude, not because any one channel is selective. There are 6,400 labeled samples in four classes of 1,600: clean air (NoGas), incense smoke (Smoke), alcohol-based deodorant vapor (Perfume), and a smoke and vapor mixture (Mixture). Smoke and Mixture are treated throughout as hazardous surrogate conditions rather than as hazardous petroleum-gas releases, and a window of either is a hazardous window in the metric definitions, not a facility hazard. **Table 2** lists the array.

| Sensor | Commonly associated target gases (nominal sensitivity) |
|---|---|
| MQ-2 | LPG, butane, methane, smoke |
| MQ-3 | Alcohol, ethanol, organic vapors |
| MQ-5 | LPG, natural gas |
| MQ-6 | LPG, butane, iso-butane |
| MQ-7 | Carbon monoxide |
| MQ-8 | Hydrogen |
| MQ-135 | Air-quality indicators including NH₃, benzene, NOₓ and CO₂ |

**Table 2—Composition of the MOX array,** with the gases each element is nominally marketed as sensitive to.

Two properties of the corpus shaped everything downstream. The analytes are laboratory surrogates. Incense smoke and alcohol-based vapor exercise the MOX transduction mechanism, the cross-sensitivity structure of the array and its baseline drift, all of which a facility detector meets in service, but they do not establish detection performance for methane or anything heavier. And the four classes were recorded as four contiguous blocks inside a single session of roughly 90 minutes, so within-session baseline drift is confounded with class identity. That is the structure Dennler et al. (2022) showed inflates reported accuracy on a MOX benchmark under a randomized split, which motivates leakage-controlled evaluation here rather than establishing the same magnitude on this corpus.

**Leakage-Controlled Partitioning.** Seven-channel readings were formed into sliding windows of length 20, each labeled by its final reading, with windows spanning a class boundary discarded so that no window mixes two classes. That leaves 6,324 windows, 1,581 per class.

Consecutive windows share 19 of 20 raw rows, so a random split drops near-duplicates into both partitions. A plain sequential split is no better, because the corpus is block-ordered by class. We use a block-wise holdout instead: inside each class block a contiguous 20% forms the test partition, separated from the training segments by an embargo of *G* = 20 windows, so no training window shares a raw row with any test window (**Fig. 2**). The embargo removes overlap across the partition boundary; it does not remove the temporal dependence that remains among windows inside the training partition and inside the test partition, and the uncertainty treatment below accounts for that rather than assuming it away. The position of the held-out block moves with the run seed, which means the dispersion we report includes data-partition variance and not only model-initialization variance (Bouthillier et al. 2021). The result is 4,980 training and 1,264 test windows, 316 per class. Feature scaling and anomaly normalization are fitted on the training partition only, then applied without refitting.

![](figures_v2/fig_protocol.png)

**Fig. 2—Corpus layout and the leakage-controlled split.** Each class occupies a contiguous block of 1,581 windows. A contiguous 20% is held out, with a 20-window embargo on each side, and the position of that block is drawn from the run seed.

**Monitoring Pipeline.** The pipeline, which we call ARXIS, has five parts. At each inference cycle it takes a thermal image *I*ₜ and a sensor window **X**ₜ ∈ ℝ^(20×7) and returns a safety action *a*ₜ ∈ {0, …, 4} plus an explanation *E*ₜ. A perception component classifies the thermal image. An anomaly component scores the sensor window by reconstruction error against a normal-only model. A coordination component assembles the decision state. A decision component maps that state to an action. A reasoning component writes the operator-facing explanation. **Fig. 3** shows how the five connect, and in particular which paths the thermal evidence does and does not reach.

The pipeline is advisory. It sits in the alarm-management and operator-support layer and actuates nothing: the action labeled *Emergency shutdown* recommends that an operator initiate the facility procedure, and it does not command a trip. IEC 61511 (IEC 2016) sets out the safety lifecycle, specification and independence expectations for safety instrumented systems in the process sector, and the component described here is positioned as an operator-support layer under that framework rather than as any part of a safety instrumented function. We make no claim that the standard mandates this particular arrangement. The narrower point is that the safety instrumented function remains separately specified, deterministic and independently verified, and that a single learned component selecting across both the alarm and the shutdown layer would collapse two protection layers into one common-cause element.

![](figures_v2/fig_architecture.png)

**Fig. 3—The monitoring pipeline.** Thermal evidence reaches the perception and explanation paths but never the decision state, and the language model sits downstream of an action it cannot alter.

**Component Models.** *Perception.* Thermal images are classified with YOLOv8n-cls (Jocher et al. 2023), the smallest classification variant in the family: images resized to 224 by 224, pretrained initialization, AdamW at 10⁻³ with cosine annealing, weight decay 10⁻⁴, batch size 32, 50 epochs, with random flip, rotation and brightness augmentation, on an 80/20 stratified split of the 6,400 frames. The full thermal set was used, the classifier was trained and validated on it, and the trained weights are among the artifacts listed in Appendix A, so the perception component reported here is the one the assembled pipeline runs.

What we did not do is carry the sensor-side partitioning discipline across to the images. That split is randomized over frames captured consecutively within a single session, so it is exposed to the same leakage mechanism the block-wise holdout was built to avoid, and adjacent thermal frames are close to duplicates. The perception figure below should be read as an optimistic upper bound rather than an independent estimate. This leaves the two halves of the pipeline held to different evidential standards. No claim in this paper rests on the perception number, because the decision state carries no visual term and every safety result is computed from actions the sensor path selected, but the multimodal system as a whole has not been validated to the standard applied to the sensor path.

*Anomaly Detection.* An LSTM autoencoder with a single-layer encoder and decoder, hidden dimension 32 and a linear output, trained on NoGas windows only, following Malhotra et al. (2016). The mean reconstruction error over the full 20-step window is the anomaly score. The threshold τ is the 95th percentile of reconstruction error on NoGas training windows only, and evaluation is on held-out windows, a distinction Experiment 1's component results quantify.

*Coordination.* The 22-dimensional decision state is

&nbsp;&nbsp;&nbsp;&nbsp;φₜ = [ ρ̃ₜ , **X**ₜ[−1,:] , **δ**ₜ , **σ**ₜ ] ∈ ℝ²²  ............ (1)

with ρ̃ₜ the normalized anomaly score, **X**ₜ[−1,:] the current seven-channel reading, **δ**ₜ the per-sensor change across the window, and **σ**ₜ the per-sensor standard deviation. There is no visual term in the state.

*Decision.* A feedforward network with a dueling value and advantage head (Wang et al. 2016), mapping φₜ to five action scores, trained by cost-weighted cross-entropy. Not by reinforcement learning, despite the architecture's provenance. Target actions come from a deterministic map *a*\*(·) from class to acceptable action set,

&nbsp;&nbsp;&nbsp;&nbsp;*a*\*(NoGas) = {0}, *a*\*(Smoke) = {3}, *a*\*(Mixture) = {4}, *a*\*(Perfume) = {1, 2}.  ............ (2) Each sample carries a loss weight *w*(*g*ₜ) equal to *c*_miss for hazardous classes and *c*_false otherwise, with *C* = *c*_miss / *c*_false. The objective is

&nbsp;&nbsp;&nbsp;&nbsp;*L*(θ) = − (1/*N*) Σₜ *w*(*g*ₜ) · log *p*_θ( *a*\*(*g*ₜ) | φₜ )  ............ (3)

where *p*_θ is the softmax over action scores. We retain *C* = 8:1 as the reference configuration from the original experiment design, and sweep the rest below. It is called a reference rather than a deployed setting because, as Experiment 4 shows, it was not selected by a defensible validation procedure.

Two consequences of Eq. 3 bound what can be claimed from any of this. The target action is a deterministic function of the class label under Eq. 2, so decision accuracy is four-class classification accuracy composed with a fixed map, not a separate quantity measured on a separate task. What differs between the decision framing and the classification framing is the error metric, not the task, and the error metric is what this paper is about. Second, a rule that applies the map of Eq. 2 to the true label reaches 100% decision accuracy by construction. Such a rule consumes the very label it is supposed to infer, so it is not deployable. Where it appears below we call it a label-function reference, never a baseline.

*Reasoning.* Explanations are generated locally with Gemma 3 1B served through Ollama (Gemma Team 2025), prompted with the gas class, classification confidence, normalized anomaly score and threshold, and the selected action, capped at 150 tokens. Nothing leaves the device. The explanation is advisory and sits off the safety-critical path: the action is fixed by the decision component and the generated text cannot change it. That separation was deliberate, on the reasoning that a fluent explanation inconsistent with the decision is a hazard in a control room rather than an aid. The explanation component is described here as part of the system and is evaluated nowhere in this paper. No result below depends on it.

**Comparators.** Eleven comparators, one protocol: the cost-weighted policy of Eq. 3; an unweighted network of identical topology; a multilayer perceptron (256-256-128); cost-sensitive gradient boosting (300 trees); an RBF support-vector classifier; a random forest (300 trees); a recurrent network over *K* = 10 consecutive states; conservative Q-learning with a TD(0) bootstrap and a conservative penalty on non-behavior actions; k-nearest neighbors with *k* = 5; a shallow decision tree of depth 3 restricted to the seven raw current-sensor channels, which is interpretable, deployable and auditable in a way the learned networks are not; and CUSUM (Page 1954), a one-sided cumulative-sum detector on the anomaly score, with μ₀ and σ estimated from NoGas training windows and the decision threshold set on the training partition.

CUSUM is two-class by construction. It emits only *Monitor* and *Raise alarm*, so it cannot separate Smoke from Mixture. Over five seeds it scores 0.4994 ± 0.0004 on decision accuracy, and a reader should not take that as evidence that the detector failed; it is an artifact of scoring a two-state output against a four-class target map. Its missed-hazard rate of 0.0013 and escalation adequacy of 0.9987 are directly comparable, because those definitions do not depend on how many classes a model can express.

Twelve models appear in total: the eleven comparators above plus the ordinal cost objective introduced in Experiment 4, which is a variant of the decision component rather than an independent method. Subsets differ between experiments, and **Table 3** states which model appears where so no reader has to infer it. Calibration uses only the cost-weighted policy, because the four estimators compared there are properties of a single model's confidence rather than of the model inventory, and the ablation likewise holds the architecture fixed so the feature set is the only thing varying.

| Model | In distribution (Exp. 2) | Leave-one-class-out (Exp. 3) | Degradation (Exp. 6) | Action matrix and input probe (Exps. 2, 5) |
|---|---|---|---|---|
| Cost-weighted policy | ✓ | ✓ | ✓ | ✓ |
| Unweighted network | ✓ | ✓ | ✓ | |
| Multilayer perceptron | ✓ | ✓ | ✓ | ✓ |
| Gradient boosting | ✓ | ✓ | ✓ | ✓ |
| Random forest | ✓ | ✓ | ✓ | ✓ |
| Support-vector classifier | ✓ | ✓ | | ✓ |
| Recurrent (LSTM) | ✓ | ✓ | | |
| Conservative Q-learning | ✓ | ✓ | | |
| k-nearest neighbors | ✓ | ✓ | | ✓ |
| Shallow decision tree | ✓ | ✓ | ✓ | ✓ |
| CUSUM | | | | |
| Ordinal cost objective | ✓ | ✓ | | |
| **Total** | **11** | **11** | **6** | **7** |

**Table 3—Which model appears in which experiment.** CUSUM is reported in the text above rather than in any table, for the reason given there.

**The Safety Metric Set.** Five quantities rather than one. This set is the methodological proposal. **Table 4** gives the action space it is defined over, and **Fig. 4** shows where each quantity reads that ladder.

| Action | Label | Operational response |
|---|---|---|
| 0 | Monitor | Routine monitoring, no operator notification |
| 1 | Increase sampling | Elevated scan frequency and event logging |
| 2 | Request verification | Secondary sensor check and event logging; no alarm is raised |
| 3 | Raise alarm | Automated alarm, incident logging, field dispatch |
| 4 | Emergency shutdown | Recommend initiating the facility shutdown procedure |

**Table 4—Safety-action space and its alarm boundary.** Actions 3 and 4 are alarm-grade; actions 1 and 2 place nothing in front of an operator.

Every metric below follows that boundary exactly, counting escalation as *a* ≥ 3 and under-escalation as *a* ∈ {1, 2}. The boundary is a modeling choice about this implementation rather than a claim about how a verification request must be handled in general, and Experiment 3 measures what moving it would do.

![](figures_v2/fig_decisionlogic.png)

**Fig. 4—One window, one action, and the boundary each metric reads.** Three quantities partition the hazardous windows; the fourth counts clean windows reaching the alarm-grade rungs.

Let the test partition be windows *t* = 1 … *n* with true classes *g*ₜ and selected actions *a*ₜ. Write *H* = {*t* : *g*ₜ ∈ {Smoke, Mixture}} for the hazardous-surrogate windows and *Z* = {*t* : *g*ₜ = NoGas} for the clean ones, and let 1[·] be the indicator function.

*Decision accuracy* is the share of test windows assigned an action in the acceptable set,

&nbsp;&nbsp;&nbsp;&nbsp;Acc = (1/*n*) Σₜ 1[ *a*ₜ ∈ *a*\*(*g*ₜ) ].  ............ (4)

It is reported for continuity with prior work. It is not a safety metric and we do not treat it as one.

*Missed-hazard rate* is the share of hazardous windows assigned the passive *Monitor* action,

&nbsp;&nbsp;&nbsp;&nbsp;*R*_miss = (1/|*H*|) Σ_{t ∈ H} 1[ *a*ₜ = 0 ].  ............ (5)

Only *a* = 0 counts. Assigning a hazardous window to *Raise alarm* when *Emergency shutdown* was the target is not a miss, because an operator is still notified. The limitation is that assigning a hazardous window to *Increase sampling* or *Request verification* is also not a miss, even though neither action places anything in front of an operator, which is why the next two quantities exist.

*Escalation adequacy* is the share of hazardous windows assigned an alarm-grade action, and *under-escalation* the share receiving a sub-alarm one,

&nbsp;&nbsp;&nbsp;&nbsp;*R*_esc = (1/|*H*|) Σ_{t ∈ H} 1[ *a*ₜ ≥ 3 ],  ............ (6)

&nbsp;&nbsp;&nbsp;&nbsp;*R*_under = (1/|*H*|) Σ_{t ∈ H} 1[ *a*ₜ ∈ {1, 2} ].  ............ (7)

By construction *R*_miss + *R*_under + *R*_esc = 1, so the three partition the hazardous windows and no response is counted twice or left out. It is precisely this partition that a miss metric alone collapses.

*High-severity false-alarm rate* is the share of clean windows assigned an alarm-grade action, and the corresponding alarm burden is that rate expressed per 1,000 clean windows,

&nbsp;&nbsp;&nbsp;&nbsp;*R*_fa = (1/|*Z*|) Σ_{t ∈ Z} 1[ *a*ₜ ≥ 3 ],  ............ (8)

&nbsp;&nbsp;&nbsp;&nbsp;*B* = 1000 · *R*_fa.  ............ (9)

Eq. 9 is what makes the quantity comparable with an alarm budget; dividing instead by the number of alerts raised gives a number that cannot be placed against one.

*Bounds on every zero.* An observed zero is not a demonstrated zero. All zero-count claims carry a one-sided 95% Clopper-Pearson upper bound (Clopper and Pearson 1934). For *k* observed failures in *n* opportunities at confidence *c*,

&nbsp;&nbsp;&nbsp;&nbsp;*U*(*k*, *n*) = BetaInv( *c*; *k* + 1, *n* − *k* ),  ............ (10)

which for *k* = 0 reduces to *U* = 1 − (1 − *c*)^(1/*n*).

The unit that goes into *n* matters more than the formula. It is tempting to pool every test window across seeds, which would give 3,160 hazardous and 1,580 clean windows in the five-seed protocol and a bound of 0.095% on a zero. We do not do that, because this design does not deliver that many independent trials. Consecutive windows share 19 of 20 raw rows, and because the held-out block position moves with the seed, the same underlying window is evaluated in more than one partition. Pooling would claim more independent information than the experiment produced.

We therefore evaluate Eq. 10 with the partition as the unit and report the worst case across partitions. A single test partition carries 632 hazardous and 316 clean windows, so an observed zero supports at most 0.47% and at most 0.94% respectively, roughly five times the pooled figure. Reaching below 10⁻³ on a zero count would need about 2,995 independent observations, which puts the usual 0.0000 in a table into perspective. One caveat remains and we would rather state it than bury it: windows overlap inside a partition too, so even the partition-level bound assumes more independence than the data strictly supply. It is an optimistic bound, and the true effective sample size is smaller than 632.

**Table 5** states the proposed metric set in the form we would ask another study to adopt, including what each quantity does not catch, since a reporting standard that omits its own blind spots is the problem this paper is about.

| Quantity | Symbol | Reads | Catches | Does not catch | Report with |
|---|---|---|---|---|---|
| Decision accuracy | Acc, Eq. 4 | all windows | overall correctness | any asymmetry between error types | SD across seed-varying partitions |
| Missed-hazard rate | *R*_miss, Eq. 5 | hazardous windows at *a* = 0 | the fully passive response to a hazard | sub-alarm responses that notify nobody | exact upper bound, Eq. 10 |
| Escalation adequacy | *R*_esc, Eq. 6 | hazardous windows at *a* ≥ 3 | whether an operator is notified | which alarm-grade action was chosen | SD across partitions, and the boundary assumed |
| Under-escalation | *R*_under, Eq. 7 | hazardous windows at *a* ∈ {1, 2} | the gap a miss metric hides | how severe the under-served hazard was | beside *R*_esc, since the two sum with *R*_miss to 1 |
| High-severity false-alarm rate | *R*_fa, Eq. 8 | clean windows at *a* ≥ 3 | operator attention spent on clean air | nuisance actions below the alarm boundary | burden per 1,000 windows, Eq. 9, and a bound on a zero |

**Table 5—The proposed metric set,** with the blind spot of each quantity stated alongside it.

**Protocol and Statistics.** Two protocols are used, and the distinction matters for how the results should be read. The five-seed protocol (42, 1337, 7, 2024, 99), each seed carrying its own block position, is used for experiments that vary something other than the model: the leave-one-class-out grid, the perturbation suite, the ablation, the calibration study and the anomaly-input probe. It is adequate there because the comparison of interest is between conditions, not between models.

The model comparison itself is run 30 times, with seed, held-out block position and a hyperparameter draw all varying per run (Bouthillier et al. 2021). Five paired observations cannot support a model ranking: the two-sided Wilcoxon signed-rank test has a minimum attainable *p*-value of 2^(1−*n*) = 0.0625 at *n* = 5, so no comparison can reach α = 0.05 at any effect size, and reporting one would be reporting the design rather than the data. With 30 runs we use the Bayesian correlated *t*-test of Benavoli et al. (2017). For a vector of paired differences *d* with mean *d̄* and sample variance *s*², the posterior is a Student *t* with *n* − 1 degrees of freedom, mean *d̄*, and scale

&nbsp;&nbsp;&nbsp;&nbsp;*s*_post = sqrt( *s*² · ( 1/*n* + ρ/(1 − ρ) ) ),  ............ (11)

with the correlation ρ = *n*_test/(*n*_train + *n*_test) = 0.202. Integrating that posterior over a region of practical equivalence of ± 1 accuracy point, chosen because a smaller difference would not change a procurement decision on this corpus, returns three probabilities per comparison: that one model is practically better, that the other is, and that the two are practically equivalent. The last is a positive statement, which a null hypothesis test cannot give.

One assumption there deserves stating rather than inheriting quietly. Benavoli et al. (2017) derive the correlation term for correlated resampling, canonically *k*-fold cross-validation, where runs are correlated because their training sets overlap, and where ρ = *n*_test/*n*_total is itself a working approximation rather than an identity. Our design is not *k*-fold: seed, block position and a hyperparameter draw all vary per run, which is the fuller variance accounting Bouthillier et al. (2021) argue for. The training partitions still overlap heavily, roughly 80% of the same windows on every run, so a positive correlation is certainly present and treating the runs as independent would be anti-conservative. But the particular value 0.202 is carried across from a design that is not quite ours, so we report the comparison over a range of ρ rather than at a single value. Determinism guards are enabled and execution is single-threaded throughout.

---

## Experimental Results and Discussion

This section reports seven experiments under the protocol above. We describe the component behavior first, then take the evaluation protocol itself as the subject of the first experiment, because every number after it depends on that protocol being defensible. **Table 6** maps the research questions onto the experiments that answer them; the three experiments carrying no research question are supporting evidence rather than answers.

| Question | Answered by | Answer |
|---|---|---|
| RQ0. Does the partitioning protocol matter? | Experiment 1 | Yes, and among temporal models by more than the model choice does |
| RQ1. Does accuracy-ranked selection survive a missed-hazard criterion? | Experiments 2 and 3 | In distribution the question is moot, since no metric discriminates; under the class-disjoint shift, no |
| RQ2. Do models that agree in distribution agree on a held-out hazardous class? | Experiment 3 | No, by more than two orders of magnitude |
| RQ3. Is missed-hazard rate sufficient on its own? | Experiment 3 | No, under-escalation satisfies it while notifying nobody |
| RQ4. Do these behaviors survive realistic sensor faults? | Experiment 6 | No, and three distinct failure modes appear |
| Supporting evidence | Experiments 4, 5 and 7 | Cost asymmetry, decision-state contribution and confidence calibration |

**Table 6—Research questions and the experiments that address them.** RQ0 is added here because the protocol result conditions every number that follows it.

**Component Behavior.** *Perception.* The thermal classifier reached 98.8% top-1 accuracy on 1,280 held-out validation images (**Table 7**). All 15 misclassifications fell on the NoGas and Perfume boundary, so neither hazardous class was confused with a non-hazardous one, which is what matters once the class label is mapped to an action (**Fig. 5**).

| Gas class | Precision | Recall | F1 | Accuracy (%) |
|---|---|---|---|---|
| Mixture | 1.000 | 1.000 | 1.000 | 100.0 |
| NoGas | 0.990 | 0.963 | 0.976 | 96.3 |
| Perfume | 0.972 | 0.991 | 0.981 | 99.1 |
| Smoke | 0.991 | 1.000 | 0.995 | 100.0 |
| Overall | 0.988 | 0.988 | 0.988 | 98.8 |

**Table 7—Perception component, 1,280 held-out validation images.** Obtained under a randomized image split rather than the block-wise protocol used on the sensor side, so read as an optimistic upper bound rather than an independent estimate.

![](figures_v2/fig_perception.png)

**Fig. 5—Thermal classifier behavior on 1,280 held-out validation images.** (a) Counts, (b) row-normalized, (c) per-class scores.

*Anomaly Detection.* Two questions decide how much weight this component can carry. Does the score track hazard? Mean reconstruction error by class comes out at NoGas 0.736, Perfume 1.543, Mixture 117.72 and Smoke 229.25 (**Fig. 6a**), so hazardous conditions separate from benign ones by two orders of magnitude and the score is usable as a hazard indicator. It is not monotone in target severity, though: Mixture carries the higher target action yet scores well below Smoke, so it is evidence of abnormality rather than a severity estimate, and no result here leans on it being the latter.

Does the partition used to fit the threshold matter? Considerably. Setting τ at the 95th percentile of all NoGas windows and evaluating on those same windows returns AUC 0.962 at a false-positive rate of exactly 5.000%, which is the nominal value a 95th-percentile threshold produces on the sample that defined it. That number measures the definition, not the detector. Fitting τ on training windows only and evaluating on held-out windows gives ROC-AUC 0.9928 ± 0.0053, TPR 0.7363 ± 0.0949 and FPR 0.0000 ± 0.0000 across five seeds (**Fig. 6b**, **Fig. 7**): discrimination improves and no false positive is observed, while sensitivity falls by roughly seven points. That zero rests on 316 held-out NoGas windows per partition, which by Eq. 10 bounds the underlying rate at 0.94% on the worst partition and no lower.

![](figures_v2/fig_anomaly.png)

**Fig. 6—Anomaly component.** (a) Mean reconstruction error by class, log scale. (b) Operating point with the threshold fitted on its own sample, against the threshold fitted on training windows and evaluated on held-out windows.

![](figures_v2/fig_roc.png)

**Fig. 7—Anomaly detector in detail.** (a) ROC curves, one per seed held out, against the in-sample curve, with both operating points marked. (b) Score distributions by class, log axis.

**Experiment 1: Sensitivity of Reported Accuracy to the Partitioning Protocol.** The block-wise split is justified above on the grounds that a random split over overlapping sliding windows places near-duplicates in both partitions. That argument is standard (Dennler et al. 2022) but it has been demonstrated on a different MOX benchmark, not this one, and an argument of that kind is cheap to make and worth measuring. Six models were trained under four partitioning protocols: a uniformly random 80/20 split over all windows; a contiguous per-class block with no separation band; the same block with the 20-window embargo used everywhere else here; and a stricter temporal protocol that trains on the first 60% of each class block, tests on the last 20% and discards the middle 20% entirely (**Table 8**, **Fig. 8**).

| Model | Random split | Blocked, no embargo | Blocked + embargo | Train early, test late |
|---|---|---|---|---|
| Cost-weighted policy | 0.9959 | 0.9671 | 0.9623 | 0.9478 |
| Multilayer perceptron | 0.9940 | 0.9616 | 0.9612 | 0.9726 |
| Gradient boosting | 0.9940 | 0.9636 | 0.9634 | 0.6858 |
| Random forest | 0.9935 | 0.9772 | 0.9769 | 0.7080 |
| k-nearest neighbors | 0.9929 | 0.9524 | 0.9521 | 0.9566 |
| Shallow decision tree | 0.9223 | 0.9294 | 0.9297 | 0.6349 |
| **Mean** | **0.9821** | **0.9585** | **0.9576** | **0.8176** |

**Table 8—Decision accuracy under four partitioning protocols,** five seed-varying runs each.

On this corpus, randomized splitting increased reported decision accuracy by between 1.66 and 4.08 percentage points for every model that reads temporal features, by 3.09 points on average across the five of them, and by 2.45 points averaged over all six including the shallow decision tree. Reported at four significant figures on a saturated benchmark, that is the difference between 0.99 and 0.96, which is roughly the spread that separates published results on this corpus from one another. We would not claim this explains any particular published number, and we have not re-run anyone else's method. What it does establish needs its comparison stated, because the claim is otherwise unfalsifiable. Among the five models that read temporal features, switching protocol moves accuracy by 3.09 points on average, while those same five span 2.48 points between them under the blocked protocol. On that comparison the choice of partitioning moves the reported number by more than the choice of model does. The comparison does not survive adding the shallow decision tree, which the protocol barely touches and which sits 4.72 points below the best model, so the honest form of the claim is restricted to models capable of exploiting the overlap. Either way, comparisons across studies using different protocols carry little information about relative merit.

The embargo contributes almost nothing, 0.9585 without it against 0.9576 with it: the block structure does the work, and the 20-window gap is close to a rounding error. And the shallow decision tree is the one model a random split does not help, losing 0.74 points instead of gaining, because it reads only the seven current sensor values and so has no capacity to exploit the near-duplication the learned models exploit. The last column is a harsher test: training early and testing late costs gradient boosting 27.8 points, the random forest 26.9 and the shallow decision tree 29.5, while the perceptron and k-nearest neighbors barely move. Within-session drift is confounded with position in each class block, so this protocol asks a question closer to deployment than anything else here, and the models disagree about it sharply.

![](figures_v2/fig_leakage.png)

**Fig. 8—Protocol sensitivity.** (a) Decision accuracy for six models under four partitioning protocols, mean ± 1 SD over five runs. (b) Accuracy points added by a random split relative to the blocked, embargoed split used throughout. Only the decision tree, which reads no temporal features, fails to benefit.

**Experiment 2: In-Distribution Model Comparison.** Every comparator was trained 30 times under the protocol above and evaluated on the corresponding held-out partition (**Table 9**, **Fig. 9**). This addresses RQ1.

| Model | Decision accuracy | SD | Missed-hazard | Escalation | High-severity FA | Expected cost | P(equivalent to cost-weighted) |
|---|---|---|---|---|---|---|---|
| Random forest | **0.9746** | 0.0222 | 0.0000 | 1.000 | 0.0000 | **0.0404** | 0.349 |
| Recurrent (LSTM) | 0.9648 | 0.0245 | 0.0000 | 0.999 | 0.0000 | 0.0656 | 0.732 |
| Gradient boosting | 0.9627 | 0.0252 | 0.0063 | 0.992 | 0.0000 | 0.0911 | 0.591 |
| Unweighted network | 0.9618 | 0.0261 | 0.0000 | 1.000 | 0.0000 | 0.0608 | **0.960** |
| Multilayer perceptron | 0.9618 | 0.0276 | 0.0003 | 1.000 | 0.0000 | 0.0625 | **0.938** |
| **Cost-weighted policy** | 0.9608 | 0.0317 | 0.0000 | 1.000 | 0.0000 | 0.0631 | reference |
| Conservative Q-learning | 0.9562 | 0.0304 | 0.0000 | 1.000 | 0.0000 | 0.0792 | 0.708 |
| Support-vector classifier | 0.9506 | 0.0268 | 0.0000 | 1.000 | 0.0000 | 0.0884 | 0.448 |
| k-nearest neighbors | 0.9497 | 0.0190 | 0.0000 | 1.000 | 0.0000 | 0.0855 | 0.416 |
| Shallow decision tree | 0.9179 | 0.0435 | 0.0000 | 1.000 | 0.0000 | 0.1226 | 0.012 |
| Ordinal cost objective | 0.7472 | 0.0054 | 0.0000 | 1.000 | 0.0000 | 0.2585 | 0.000 |

**Table 9—Model comparison over 30 randomized runs.** Expected cost is the mean of the cost matrix defined under Experiment 4 and is reported so the comparison is not carried entirely by accuracy. The final column is the posterior probability of practical equivalence to the cost-weighted policy under Eq. 11 with a region of practical equivalence of ± 1 accuracy point.

Measured by Eq. 4, the ten comparators other than the ordinal objective span 5.66 accuracy points, from 0.9179 to 0.9746; highest accuracy and lowest expected cost belong to the random forest, and the cost-weighted policy places sixth. The shallow decision tree on seven raw sensor values sits 5.7 points behind the best learned model, a wider gap than the five-seed protocol suggested.

The safety metrics are weakly discriminative here, because almost every model produces the same safety outcome under the nominal distribution. Every comparator except gradient boosting records no observed missed hazard and escalation adequacy of 1.000, and no comparator produces a single high-severity false alarm across 30 runs. Each run contributes 632 hazardous and 316 clean windows, so those observed zeros bound the underlying rates at 0.47% and 0.94% on the worst partition, not at the far smaller figure a pooled count would suggest.

With 30 runs we can state something positive rather than merely decline to rank. The cost-weighted policy is practically equivalent to the unweighted network with posterior probability 0.960 and to the multilayer perceptron with 0.938, so removing the cost weighting is very likely to make no difference that matters on this corpus. Equivalence to the recurrent network (0.732), conservative Q-learning (0.708) and gradient boosting (0.591) is weaker but still the most probable outcome in each case, and two comparisons resolve. Against the random forest the posterior splits 0.631 that the forest is practically better and 0.349 equivalent, the value in Table 9; against the shallow decision tree it splits 0.988 that the reference is practically better and 0.012 equivalent. Table 9 prints the equivalence column alone, so the complementary mass is stated here. That is a more useful statement than "no significant difference was detected", and it is available only because the run count was raised.

![](figures_v2/fig_powered.png)

**Fig. 9—Model comparison over 30 randomized runs.** (a) Decision accuracy, mean ± 1 SD, with the cost-weighted policy highlighted and the ordinal cost objective of Experiment 4 in green. (b) Posterior probabilities from the Bayesian correlated *t*-test against the cost-weighted policy.

*How much of that depends on the correlation term.* Because ρ = 0.202 is borrowed from a cross-validation setting rather than derived for this one, we swept it from 0, which treats the runs as independent, to 0.5, which assumes far more correlation than 30 randomized partitions plausibly carry (**Table 10**).

| Comparison | ρ = 0 | ρ = 0.10 | ρ = 0.202 | ρ = 0.30 | ρ = 0.50 |
|---|---|---|---|---|---|
| Unweighted network | 1.000 | 0.994 | **0.960** | 0.901 | 0.740 |
| Multilayer perceptron | 1.000 | 0.989 | **0.938** | 0.866 | 0.691 |
| Recurrent (LSTM) | 0.984 | 0.847 | 0.732 | 0.638 | 0.476 |
| Conservative Q-learning | 0.971 | 0.817 | 0.708 | 0.620 | 0.466 |
| Gradient boosting | 0.971 | 0.748 | 0.591 | 0.488 | 0.341 |
| Support-vector classifier | 0.471 | 0.477 | 0.448 | 0.408 | 0.318 |
| k-nearest neighbors | 0.388 | 0.436 | 0.417 | 0.381 | 0.298 |
| Random forest | 0.164 | 0.316 | 0.349 | 0.346 | 0.298 |
| Shallow decision tree | 0.000 | 0.001 | 0.012 | 0.032 | 0.083 |
| Ordinal cost objective | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

**Table 10—Sensitivity of the Bayesian equivalence probability to the assumed run correlation ρ.** The deployed value is 0.202, and the ρ = 0.202 column is identical to the final column of Table 9. Raising ρ widens the posterior and moves mass out of the region of practical equivalence, so the probabilities fall monotonically; what does not change is which of the three outcomes is most probable in each row.

The direction of every comparison is stable across the whole range. The random forest remains the more probable winner at every ρ, the decision tree and the ordinal objective remain the more probable losers at every ρ, and the unweighted network and the perceptron remain practically equivalent to the reference as the single most probable outcome even at ρ = 0.5, where the posterior is far wider than this design warrants. What does move is strength: 0.960 falls to 0.740 under the most pessimistic assumption we tried.

*A comparison that assumes no correlation at all.* Because ρ has to be assumed rather than estimated, we also ran a paired bootstrap in which the run is the resampling unit and nothing is assumed about structure inside a run: the 30 paired differences are resampled with replacement 20,000 times, and the same one-point region of practical equivalence is applied to the replicate means. This asks the same question under a weaker assumption. It agrees with the correlated *t*-test on the most probable outcome in all ten comparisons. Under the bootstrap the unweighted network and the perceptron are practically equivalent to the reference in essentially every replicate, with 95% intervals on the paired difference of [−0.004, +0.002] and [−0.004, +0.002]; the random forest is better in 85% of replicates, interval [−0.021, −0.007]; and the decision tree and the ordinal objective are worse in every replicate. We therefore treat the ρ sweep as a sensitivity analysis rather than as the primary evidence, and rest the ranking claim on the agreement between the two.

Where the actions land makes the saturation concrete (**Fig. 10**): every comparator sends essentially all Smoke windows to *Raise alarm* and all Mixture windows to *Emergency shutdown*, and the only visible spread is on the NoGas and Perfume boundary. One detail there belongs to the action space rather than to any model. Action 2, *Request verification*, is never selected by any of the seven comparators in Fig. 10, on any run, and the reason is structural: the target map sends Perfume to action 1, so action 2 is never a positive training target and cannot be scored correct under any model trained against that map. Although the action space contains five levels, the deterministic target map never assigns action 2 as a positive target, so the empirical decision space behaves as a four-action system and nothing here measures how a model would use an intermediate verification step. That is an unresolved question in the action design rather than a failure of the models, and a verification action triggered by predictive uncertainty rather than class identity is the obvious candidate.

Decision accuracy pools the four classes, which makes the spread in Table 9's first column hard to place. Resolving it by class shows where the disagreement lives (**Table 11**). The inventory is the seven models of Fig. 10 over five seed-varying partitions rather than the eleven of Table 9 over 30 runs, so the two are not directly comparable row by row; the pattern, however, is not subtle.

| Model | NoGas | Smoke | Mixture | Perfume |
|---|---|---|---|---|
| Gradient boosting | **0.9703** | 0.9804 | 1.0000 | 0.9032 |
| Random forest | 0.9627 | 1.0000 | 1.0000 | **0.9449** |
| Support-vector classifier | 0.9532 | 1.0000 | 1.0000 | 0.8513 |
| Multilayer perceptron | 0.9500 | 1.0000 | 1.0000 | 0.8949 |
| k-nearest neighbors | 0.9386 | 1.0000 | 1.0000 | 0.8696 |
| Cost-weighted policy | 0.9342 | 1.0000 | 1.0000 | 0.9152 |
| Shallow decision tree | 0.8728 | 0.9994 | 1.0000 | 0.8468 |
| **Range** | **0.873 to 0.970** | **0.980 to 1.000** | **1.000** | **0.847 to 0.945** |

**Table 11—Per-class share of windows assigned an acceptable action, in distribution,** averaged over five seed-varying partitions.

Every model assigns the correct action to every Mixture window and to at least 98.0% of Smoke windows. The whole of the between-model spread sits on NoGas and Perfume, which each span 9.8 points across the seven models. Accuracy on this corpus is therefore very largely a measure of how a model resolves the boundary between clean air and a volatile organic compound, and not a measure of hazardous-class performance at all. That is worth knowing before reading a single accuracy figure on this benchmark as evidence about hazard handling.

![](figures_v2/fig_actionmatrix.png)

**Fig. 10—Action-selection matrices in distribution,** averaged over five seed-varying partitions. Rows are gas classes, columns are actions.

**Experiment 3: Class-Disjoint Hazard Evaluation.** One hazardous surrogate class is withheld from training and the model is evaluated on that class after being trained on the remaining three, which is a distribution shift induced by class exclusion rather than by a genuinely novel chemistry. Each hazardous class was withheld in turn, models retrained on the remaining three, then evaluated on every window of the withheld class. Scaling and anomaly normalization were fitted on the three training classes only, so no statistic of the withheld class leaks into training. Held-out decision accuracy is zero by construction, since withholding a class removes its target action from the training label set, so what the model does instead is the interesting part. This addresses RQ2 and RQ3 (**Table 12**, **Figs. 11** to **13**).

| Held-out | Model | Missed-hazard | 95% CP bound | **Escalation (a ≥ 3)** | **Under-escalation (a ∈ {1,2})** |
|---|---|---|---|---|---|
| Smoke | Cost-weighted policy | 0.0000 | ≤0.19% | 1.000 ± 0.000 | 0.000 |
| Smoke | Unweighted network | 0.0000 | ≤0.19% | 1.000 ± 0.000 | 0.000 |
| Smoke | Gradient boosting | 0.0000 | ≤0.19% | 1.000 ± 0.000 | 0.000 |
| Smoke | Random forest | 0.0000 | ≤0.19% | 1.000 ± 0.000 | 0.000 |
| Smoke | Conservative Q-learning | 0.0000 | ≤0.19% | 1.000 ± 0.000 | 0.000 |
| Smoke | Shallow decision tree | 0.0000 | ≤0.19% | 1.000 ± 0.000 | 0.000 |
| Smoke | Ordinal cost objective | 0.0000 | ≤0.19% | 0.982 ± 0.007 | 0.018 |
| Smoke | Support-vector classifier | 0.0042 | ≤0.83% | 0.928 ± 0.023 | 0.068 |
| Smoke | Recurrent (LSTM) | 0.0083 | ≤1.30% | 0.991 ± 0.009 | 0.000 |
| Smoke | k-nearest neighbors | 0.0116 | ≤1.68% | 0.943 ± 0.005 | 0.045 |
| Smoke | **Multilayer perceptron** | **0.0949** | **≤10.79%** | 0.902 ± 0.130 | 0.003 |
| Mixture | Gradient boosting | 0.0000 | ≤0.19% | **1.000 ± 0.000** | 0.000 |
| Mixture | Random forest | 0.0000 | ≤0.19% | **0.990 ± 0.010** | 0.010 |
| Mixture | Conservative Q-learning | 0.0000 | ≤0.19% | **0.506 ± 0.206** | 0.494 |
| Mixture | Ordinal cost objective | 0.0000 | ≤0.19% | **0.393 ± 0.235** | 0.607 |
| Mixture | Cost-weighted policy | 0.0000 | ≤0.19% | **0.245 ± 0.199** | 0.755 |
| Mixture | Unweighted network | 0.0003 | ≤0.19% | **0.217 ± 0.211** | 0.783 |
| Mixture | Support-vector classifier | 0.0003 | ≤0.19% | **0.005 ± 0.008** | **0.995** |
| Mixture | Recurrent (LSTM) | 0.0053 | ≤0.91% | **0.678 ± 0.112** | 0.316 |
| Mixture | Multilayer perceptron | 0.0070 | ≤1.15% | **0.242 ± 0.121** | 0.751 |
| Mixture | k-nearest neighbors | 0.0254 | ≤3.28% | **0.205 ± 0.072** | 0.770 |
| Mixture | Shallow decision tree | 0.2487 | ≤26.71% | 0.564 ± 0.399 | 0.188 |

**Table 12—Class-disjoint hazard evaluation.** *n* = 1,581 windows of the held-out class per partition; the bound column is the 95% Clopper-Pearson upper bound a single partition supports, not a pooled figure. Scaling fitted on the three training classes only.

*In-distribution accuracy does not order held-out-class missed-hazard rate.* On the held-out Smoke class the multilayer perceptron, which sat fifth of ten by point accuracy in Table 9 and was practically equivalent to the reference there with posterior probability 0.938, records a missed-hazard rate under Eq. 5 of 9.49% on the held-out class, with a 95% upper bound of 10.79% on a single partition. Seven other models have upper bounds of 0.19%. That is a separation of more than two orders of magnitude and it is entirely invisible in Table 9, where the same model is practically equivalent to the reference. The rank reversal is the evidence: a model in the middle of the accuracy table is the worst in the field on the quantity that matters, so no accuracy-based selection procedure would have steered a team away from it. That answers RQ2 in the negative.

We are deliberately not converting that into a correlation coefficient. Over the ten comparators the Pearson correlation between in-distribution accuracy and held-out-class missed-hazard rate is *r* = +0.13, with a 95% confidence interval of [−0.55, +0.70]; for escalation on the held-out Mixture class it is *r* = +0.25, interval [−0.45, +0.76] (**Fig. 11**). Both intervals comfortably contain a strong positive association, so at *n* = 10 the correlation is unresolved rather than absent, and reporting the point estimate as evidence of absence would be the same error this paper spends Experiment 2 warning against. What the data do support is the ordering claim above, which needs no coefficient.

![](figures_v2/fig_dissociation.png)

**Fig. 11—In-distribution accuracy against held-out-class safety behavior,** for the ten comparators of Table 9 at their 30-run decision accuracy. The ordinal cost objective is excluded.

*Missed-hazard rate is not enough on its own.* On the held-out Mixture class, ten of eleven comparators have 95% upper confidence bounds at or below 3.3% on missed hazards and would pass any review conducted on that number, while their escalation adequacy runs from 0.5% to 100%, a factor of about 198. The support-vector classifier misses 0.03% of held-out-class hazardous windows while escalating 0.5% of them, answering roughly 99.5% of the class carrying the highest target action with *Increase sampling*, the only sub-alarm action any model here emits. The miss metric records that as a clean sheet. The cost-weighted policy escalates 24.5% and, by Eq. 7, under-escalates 75.5%; gradient boosting, carrying none of the pipeline's safety machinery, escalates all of them. Which metric is chosen, rather than which model, is what decides whether a model looks safe: a metric set reporting missed-hazard rate alone would rank the support-vector classifier above gradient boosting, while on escalation adequacy the ordering reverses completely. That answers RQ3.

A sharper objection than the severity one deserves an answer here. Withholding Mixture removes action 4 from the training label set, so escalation adequacy on the held-out Mixture class is arithmetically a question about which of the three remaining classes a model assigns those windows to: route them to Smoke and the model escalates, route them to Perfume and it under-escalates. The metric is therefore a re-expression of a classification decision, not an independent measurement. We agree, and we think that is the point. Held-out decision accuracy is zero for every model by construction, so the classification decision is invisible to accuracy: the accuracy column cannot distinguish a model that routes a held-out hazardous class to *Raise alarm* from one that routes it to *Increase sampling*, and both score zero. The escalation column is what makes the distinction visible, and the distinction is the one an operator experiences. That a safety metric can be rewritten as a statement about class assignment does not make it redundant; it makes it the statement about class assignment that matters operationally. Cross-sensitivity to interferents outside the training set is routine for MOX arrays in service (Vergara et al. 2012), so the situation is not contrived.

*How much of that depends on where the alarm boundary sits.* Table 4 places the boundary at *a* ≥ 3 on the grounds that actions 1 and 2 place nothing in front of an operator, and that is a modeling choice about this implementation. It is also the choice the 198-fold spread depends on, so we tested it rather than leaving it as a caveat. Because *R*_miss, *R*_under and *R*_esc partition the hazardous windows, escalation under a boundary at *a* ≥ 1 is exactly 1 − *R*_miss, which the stored results already contain. Moving the boundary down one rung collapses the spread: on the held-out Mixture class, escalation adequacy runs from 0.005 to 1.000 at *a* ≥ 3 and from 0.751 to 1.000 at *a* ≥ 1, a factor of 198 against a factor of 1.3. On the held-out Smoke class the two boundaries agree, 0.902 to 1.000 against 0.905 to 1.000, because almost nothing lands in the sub-alarm band there.

That is worth stating precisely, because it cuts both ways. The entire disagreement between models on the held-out Mixture class lives in the sub-alarm band, so the result is not robust to the boundary in the sense that a different implementation of action 1 would report a different number. What the result does establish, and what does not depend on the boundary, is that ten of eleven models route most of a held-out hazardous class to actions 1 or 2 rather than to 3 or 4, and that a miss metric scores every one of those routings as a success. Whether that is acceptable is then a question about the facility's implementation of *Increase sampling*, which is exactly where we think the question belongs. It is not a question a miss metric can be asked at all.

Two further patterns are worth noting. The models that escalate fully on both held-out classes are the two tree ensembles, but the ordering in between is not a simple split between trees and networks: the recurrent network reaches 0.678 on the held-out Mixture class and conservative Q-learning 0.506, both well above the cost-weighted policy at 0.245 and the perceptron at 0.242. And the deployable decision tree fails loud on the held-out Smoke class while failing badly on the held-out Mixture class, at 24.87% missed, with an upper bound of 26.71%. No comparator is uniformly best, which is itself an argument for reporting the full set. **Fig. 12** puts the two panels side by side on a logarithmic axis, which the three-decade spread requires: on a linear axis a 9.49% failure and a 0.03% one sit at indistinguishable heights, and **Fig. 13** decomposes the same responses into alarm-grade, sub-alarm and passive shares, which is where the support-vector column becomes hard to defend: almost the whole bar is sub-alarm.

![](figures_v2/fig_loco.png)

**Fig. 12—Behavior on a hazardous class withheld from training,** eleven comparators, mean over five seed-varying partitions. (a) Missed-hazard rate, logarithmic axis, bars at the floor line being observed zeros bounded at 0.19% per partition. (b) Escalation adequacy, mean ± 1 SD.

![](figures_v2/fig_disposition.png)

**Fig. 13—Disposition of hazardous windows from a held-out class,** decomposed into alarm-grade, sub-alarm and passive shares.

**Experiment 4: Cost Asymmetry and an Ordinal Alternative.** The loss-weight ratio *C* from Eq. 3 was swept across nine values with everything else fixed (**Table 13**, **Fig. 14**).

| Cost ratio *C* | Decision accuracy | SD | Missed-hazard | Escalation | High-severity FA |
|---|---|---|---|---|---|
| 1:1 (symmetric) | 0.9562 | 0.0386 | 0.0000 | 1.000 | 0.0000 |
| 2:1 | 0.9532 | 0.0418 | 0.0000 | 1.000 | 0.0000 |
| 4:1 | 0.9570 | 0.0390 | 0.0000 | 1.000 | 0.0000 |
| 6:1 | **0.9655** | 0.0272 | 0.0000 | 1.000 | 0.0000 |
| 8:1 (reference) | 0.9623 | 0.0214 | 0.0000 | 1.000 | 0.0000 |
| 10:1 | 0.9589 | 0.0338 | 0.0000 | 1.000 | 0.0000 |
| 12:1 | 0.9533 | 0.0346 | 0.0000 | 1.000 | 0.0000 |
| 16:1 | 0.9427 | 0.0463 | 0.0000 | 1.000 | 0.0000 |
| 20:1 | 0.8389 | 0.1189 | 0.0000 | 1.000 | 0.0000 |
| *Label function* | *1.0000* | *0.0000* | *0.0000* | *1.000* | *0.0000* |

**Table 13—Cost-asymmetry sweep, five seeds per configuration.** The label-function row applies the map of Eq. 2 to the true class label. It reaches the ceiling by construction, consumes the label it is meant to infer, and is not a deployable baseline.

Missed-hazard rate is zero and escalation adequacy is 1.000 at every ratio, the symmetric 1:1 configuration included. Accuracy climbs to a maximum of 0.9655 at 6:1, sits at 0.9623 at the reference 8:1, then falls away to 0.8389 at 20:1, twelve points below the peak with variance up roughly fourfold. The asymmetry produced no measurable safety benefit and, past 6:1, cost accuracy.

The reason looks structural. Because the target action is a deterministic function of a well-separated class label, and a miss under this definition requires assigning a hazardous window to the class furthest from it in feature space, the miss rate is already on the floor before any weighting is applied. There is nothing for the asymmetry to reshape. That is not evidence against cost-sensitive learning in general, since the literature is clear it works where the error surface is non-trivial (Elkan 2001). What this benchmark shows is narrower: it does not present a sufficiently difficult error surface for a measurable safety benefit from scalar class-level cost weighting to appear.

The reference configuration deserves one more note. *C* = 8:1 is not the accuracy optimum and was originally chosen on test-partition dispersion, which is not a defensible criterion. Re-selecting it needs a validation partition, and the obvious construction fails instructively: a uniformly random validation subset carved from training scores between 0.9948 and 0.9994 for every ratio from 1:1 to 8:1, because it inherits exactly the window overlap Experiment 1 measures. Those five configurations differ by up to 0.6 accuracy points on the test set and by 0.005 on this validation split. Carving the validation band contiguously from the end of each class block instead, with the same embargo, restores discrimination and selects *C* = 2:1 (validation 0.9606, test 0.9430), with 6:1 close behind at 0.9604. The reference 8:1 is not selected under either criterion.

*An ordinal objective, since the evidence points at one.* Eq. 3 prices all wrong actions alike, so under-escalating a hazardous window and over-escalating it cost the same. That is a testable claim, so we tested it. We replaced the sample-weighted cross-entropy with direct minimization of expected cost,

&nbsp;&nbsp;&nbsp;&nbsp;*L*_ord(θ) = (1/*N*) Σₜ Σ_a *p*_θ(*a* | φₜ) · *M*[*g*ₜ, *a*],  ............ (12)

under a matrix *M* that prices distance and direction along the action ladder: for a hazardous class, silence costs 10, a sub-alarm response 5, and notifying at the wrong tier 1; for clean air, a high-severity alarm costs 4 against 1 for a nuisance action; for the volatile-organic-compound class, silence costs 2 and a high-severity alarm 3. The structure, not the exact values, is what the experiment tests. This is a single untuned matrix, chosen by hand for its shape and run once, offered as a test of a mechanism rather than as a model we are proposing.

It works on the quantity it targets. On the held-out Mixture class the ordinal objective escalates 0.393 ± 0.235 of hazardous windows against 0.245 ± 0.199 for the cost-weighted policy, a relative improvement of about 60% while still recording no observed miss (Table 12), and it is the only change to the objective anywhere in this study that moved escalation adequacy at all. It also costs more than it returns as specified: in-distribution accuracy falls to 0.7472 ± 0.0054, some 21 points below the cost-weighted policy, and expected cost under the matrix the objective itself minimizes rises from 0.0631 to 0.2585. The mechanism is visible in the action distribution, where the model stops emitting *Monitor* for clean air and sends it to *Increase sampling* instead, because the matrix prices that error at 1 against 10 for a missed hazard and the NoGas and Perfume classes overlap precisely where that trade is decided. The result is a system in a permanent state of mildly elevated sampling.

So what has been demonstrated is narrow, and we would rather bound it ourselves. The mechanism is confirmed: pricing direction along the action ladder moves escalation, where scalar reweighting did not. What has not been demonstrated is a cost matrix worth deploying. A matrix that buys 15 points of escalation on a held-out hazardous class by paying 21 points of accuracy on clean air is not one we would put in front of an operator, and a single untuned draw cannot locate the optimum over the space of such matrices. The experiment demonstrates that a directional cost structure can alter escalation behavior; it does not establish an optimal cost matrix, and the row the objective occupies in Tables 9 and 12 should be read as a mechanism result rather than as a deployable policy.

![](figures_v2/fig_costsweep.png)

**Fig. 14—Cost-asymmetry sweep.** Shading is ± 1 SD across five seed-varying partitions.

**Experiment 5: Decision-State Ablation and Upstream-Input Sensitivity.** The decision state was ablated by feature group with the architecture and protocol held fixed (**Table 14**, **Fig. 15**).

| Decision state | *d* | Decision accuracy | SD | Δ vs. full | Missed-hazard | Escalation |
|---|---|---|---|---|---|---|
| Full state | 22 | 0.9612 | 0.0110 | reference | 0.0000 | 1.000 |
| Without per-sensor σ | 15 | 0.9601 | 0.0357 | −0.11 pp | 0.0000 | 1.000 |
| Without anomaly score | 21 | 0.9574 | 0.0206 | −0.38 pp | 0.0000 | 1.000 |
| Without per-sensor δ | 15 | 0.9399 | 0.0413 | −2.14 pp | 0.0000 | 1.000 |
| Without δ and σ | 8 | 0.8875 | 0.0730 | −7.37 pp | 0.0000 | 1.000 |
| Current readings only | 7 | 0.8845 | 0.0764 | −7.67 pp | 0.0000 | 1.000 |
| Anomaly score only | 1 | 0.7513 | 0.1349 | −21.00 pp | 0.0000 | 1.000 |

**Table 14—Decision-state ablation, five seeds.**

Temporal features carry the accuracy: dropping both δ and σ costs 7.37 points, and a single anomaly score on its own still manages 75.1%. But missed-hazard rate stays at zero and escalation adequacy at 1.000 in every condition, including the one-dimensional state. No feature group changes the in-distribution safety metrics at all, which is another instance of the same trap: an ablation reported on accuracy alone would appear to identify which features drive safety behavior, when it is measuring which features drive the in-distribution decision metrics.

![](figures_v2/fig_ablation.png)

**Fig. 15—Decision-state ablation.** Bars are mean ± SD decision accuracy. Missed-hazard rate and escalation adequacy are constant across every condition.

Retraining without a feature answers whether the feature is necessary. A sharper question, and the one that matters if an upstream component can fail in service, is what a trained model does when that input goes wrong while everything else stays correct. What follows is a synthetic sensitivity probe rather than a physical failure model. We swept the anomaly score across its full range on every test window, holding the other twenty-one features fixed, and recorded how often the selected action moved (**Fig. 16**).

The models divide sharply. For the decision tree the answer is zero by construction, since it reads only the raw channels. For k-nearest neighbors, the support-vector classifier and the perceptron the action moves on under 4% of windows; for the cost-weighted policy on 8.0 ± 5.8% and the random forest on 16.1 ± 7.6%. Gradient boosting is the outlier at 60.6 ± 7.9%, so for that model the anomaly channel is load-bearing where for the others it is close to inert at inference time. The consequence shows up in the safety metrics, and only for that model: forcing the anomaly score to zero, which is what a failed or unconnected autoencoder would supply, leaves every other comparator's missed-hazard rate at zero and moves gradient boosting from 0.0098 to 0.3405, with escalation adequacy falling from 0.990 to 0.495. The measured safety behavior of this model on this benchmark is therefore highly sensitive to the anomaly-score input, and nothing in the accuracy column or the ablation table shows it.

A single conditional probe supports a narrow reading. The override is synthetic, the other features are held at values that in reality would co-vary with the anomaly score, and one corpus is a narrow basis for a claim about any model family. Still, it is consistent with two results we already had: gradient boosting is the model that fails silent under sensor loss and the one that escalates the held-out Mixture class perfectly. A heavier dependence on the anomaly channel is a plausible common mechanism, and it suggests that dependence on upstream components deserves to be measured rather than assumed.

![](figures_v2/fig_anominfluence.png)

**Fig. 16—Dependence of the selected action on the anomaly input,** five seed-varying partitions. (a) Share of test windows whose action changes at any point as the anomaly score is swept across [0, 1] with all other features held fixed, mean ± 1 SD. (b) Missed-hazard rate against the forced anomaly value. Six comparators are flat; gradient boosting moves from 0.0098 to 0.3405 when the channel reads zero.

**Experiment 6: Graded Sensor Degradation.** Six models were put through synthetically imposed degradation designed to probe robustness: additive noise (σₙ = 0.1 to 0.5), calibration drift (gain and baseline shift of ±10% to ±50%) and channel dropout (*k* = 1 to 7 of 7 sensors) at graded severity, with all three safety metrics reported. Selected perturbation results appear in **Table 15** and the full sweep in **Figs. 17** and **18**. This addresses RQ4.

| Condition | Model | Decision acc. | Missed-hazard | Escalation | High-severity FA | Indications per 1,000 windows |
|---|---|---|---|---|---|---|
| Clean | Cost-weighted | 0.962 | 0.0000 | 1.000 | 0.0000 | ≤ 9.4\* |
| Clean | Random forest | 0.977 | 0.0000 | 1.000 | 0.0000 | ≤ 9.4\* |
| Clean | Shallow decision tree | 0.930 | 0.0000 | 1.000 | 0.0000 | ≤ 9.4\* |
| Drift ±50% | Cost-weighted | 0.916 | 0.0000 | 1.000 | 0.0006 | 0.6 |
| Drift ±50% | Multilayer perceptron | 0.927 | 0.0003 | 0.999 | 0.0000 | 0 |
| Drift ±50% | Gradient boosting | 0.876 | **0.0775** | 0.890 | 0.0000 | 0 |
| Noise σₙ = 0.5 | Cost-weighted | 0.843 | 0.0000 | 1.000 | 0.1342 | 134 |
| Noise σₙ = 0.5 | Random forest | 0.889 | 0.0000 | 1.000 | 0.0063 | 6.3 |
| Dropout *k* = 1 | Cost-weighted | 0.883 | 0.0000 | 1.000 | 0.1171 | **117** |
| Dropout *k* = 1 | Shallow decision tree | 0.723 | 0.0000 | 1.000 | 0.2000 | **200** |
| Dropout *k* = 1 | Random forest | 0.974 | 0.0000 | 1.000 | 0.0000 | 0 |
| Dropout *k* = 3 | Cost-weighted | 0.793 | 0.0000 | 0.992 | 0.1532 | 153 |
| Dropout *k* = 3 | Multilayer perceptron | 0.851 | 0.0218 | 0.975 | 0.0234 | 23 |
| Dropout *k* = 7 | Cost-weighted | 0.318 | 0.0000 | 1.000 | **1.0000** | **1,000** |
| Dropout *k* = 7 | Shallow decision tree | 0.250 | 0.0000 | 1.000 | **1.0000** | **1,000** |
| Dropout *k* = 7 | Random forest | 0.355 | 0.0000 | 1.000 | 0.6000 | 600 |
| Dropout *k* = 7 | Multilayer perceptron | 0.332 | **0.5266** | 0.473 | 0.2000 | 200 |
| Dropout *k* = 7 | Gradient boosting | 0.381 | **0.6000** | 0.400 | 0.0000 | 0 |

**Table 15—Selected perturbation results**, with the full sweep in Fig. 17. Alarm burden is the high-severity false-alarm rate expressed per 1,000 clean windows, per detector, which keeps it independent of any particular duty cycle. \*The clean-condition burden is the Clopper-Pearson upper bound a single partition supports on an observed zero, not a point estimate.

Three failure modes separate cleanly, and no single metric tells them apart. *Fail-loud:* the cost-weighted policy and the decision tree record no observed missed hazards and full escalation right through complete sensor loss, and they do it by escalating everything. False-alarm rate reaches 1.000, meaning every clean window raises a high-severity alarm. Under EEMUA (2013) that is a hazard, not a safe failure. On the missed-hazard metric alone it is a perfect score. *Fail-silent:* gradient boosting records no false alarm under Eq. 8 at any severity and misses 60% of hazards at *k* = 7, escalation collapsing to 0.400. On the false-alarm metric alone, also a perfect score. *Graceful, then brittle:* the random forest holds zero misses, full escalation and no false alarms out to *k* = 3, then jumps to a 60% false-alarm rate by *k* = 7.

Drift is the mildest of the three families on the false-alarm side: across every model and every severity the peak high-severity false-alarm rate under drift is 0.06%, reached by the cost-weighted policy at ±30%, so the damage drift does shows up as missed hazards rather than as alarm load. Noise and channel loss behave in the opposite direction.

The number that matters operationally is the first onset rather than the endpoint. Losing one sensor of seven takes the cost-weighted policy from zero to 117 alarm-grade indications per 1,000 clean windows, and the decision tree to 200, while the random forest is unaffected. Against the EEMUA envelope drawn in **Fig. 18**, two of the five models cross it at a single lost channel, two more cross it by three, and gradient boosting never crosses it at any severity because it fails in the other direction.

Converting a rate into a burden is worth spelling out, because a bare rate is hard to place against the envelopes an operator works inside whereas an alarm count is not. Any count per unit time needs a duty cycle. The corpus carries no timestamp column, but the acquisition protocol is documented at 2 s intervals (Narkhede et al. 2022), which puts a detector at *f* = 1,800 windows per hour, so the hourly indication rate is

&nbsp;&nbsp;&nbsp;&nbsp;*A* = *B* · *f* / 1000.  ............ (13)

We report *B* from Eq. 9 as the primary quantity, because that is a property of the model while the hourly figure is a property of the deployment. On that basis Eq. 13 puts the single-sensor-loss figure at about 211 alarm-grade indications per hour from one instrument, against the EEMUA (2013) treatment of roughly six alarms per hour as the limit of what an entire operator position can absorb. The comparison is not an estimate of plant alarm-system loading. It is an order-of-magnitude indication-rate comparison, intended to show why degraded model behavior belongs in an alarm-management review rather than in a model-selection spreadsheet. What survives without those assumptions is the ordering and the magnitude: neither an accuracy table nor a missed-hazard table would have predicted that one lost channel separates these models by two orders of magnitude in alarm production.

The arithmetic cuts the other way on the clean-condition rows. A zero across the 316 clean windows of one partition bounds the burden at 9.4 alarm-grade indications per 1,000 clean windows, roughly 17 per hour under the same illustrative duty cycle, which is several times the EEMUA figure for a whole operator position rather than a fraction of it. An observed zero false-alarm rate is therefore not yet evidence of an acceptable indication load, and the earlier pooled arithmetic understated how far short of that evidence the experiment falls. And below roughly 60% accuracy none of these models is fit for service: zero missed hazards at 31.8% accuracy describes a failure mode, not fitness.

![](figures_v2/fig_perturb.png)

**Fig. 17—Graded perturbation suite,** rows sharing a scale so the panels can be compared.

![](figures_v2/fig_alarmburden.png)

**Fig. 18—False-alarm rate expressed as alarm burden,** per 1,000 clean windows. The dashed line marks the EEMUA 191 envelope for a whole operator position at one window every 2 s.

**Experiment 7: Confidence Calibration.** Confidence matters here because any escalation threshold, and any deferral rule, would be set against it. Four estimators were compared: the raw softmax, MC dropout with 20 passes, temperature scaling, and a five-member deep ensemble built from the same model class and objective as the single model. Temperature is fitted on a calibration split carved out of training, never on the training logits themselves, and expected calibration error uses equal-mass bins because confidence on this corpus piles up near 1 and equal-width bins leave most of the range empty (**Table 16**, **Fig. 19**).

| Estimator | ECE | SD | Bootstrap 95% CI | ECE on hazardous windows | Brier | NLL | Accuracy |
|---|---|---|---|---|---|---|---|
| Deep ensemble (5) | **0.0162** | 0.0129 | 0.0107 to 0.0253 | 0.0002 | 0.0552 | 0.1082 | 0.9663 |
| MC dropout (20) | 0.0222 | 0.0184 | 0.0150 to 0.0320 | 0.0002 | 0.0613 | 0.1145 | 0.9619 |
| Raw softmax | 0.0244 | 0.0187 | 0.0166 to 0.0342 | 0.0002 | 0.0628 | 0.1397 | 0.9628 |
| Temperature scaling | 0.0263 | 0.0194 | 0.0178 to 0.0358 | 0.0000 | 0.0640 | 0.1560 | 0.9628 |

**Table 16—Calibration, five seeds.** ECE with 15 equal-mass bins; intervals from 400 bootstrap resamples per seed.

The deep ensemble has the lowest point estimate of ECE, but every bootstrap interval overlaps every other, so the four estimators are not separated in this experiment. Two observations are more useful than the ordering. First, overall ECE is driven entirely by the non-hazardous classes: restricted to Smoke and Mixture windows every estimator is calibrated to within 0.02%, and temperature scaling to within rounding. The miscalibration lives on the NoGas and Perfume boundary, the same place the perception errors and the anomaly-score overlap live. For a system whose escalation threshold would be set on hazardous-class confidence that is reassuring, and it is invisible in a single pooled ECE number.

Second, temperature scaling made things slightly worse, which is not what the usual account predicts. The fitted temperature is consistently below 1, at 0.839 ± 0.044 across seeds and never above 0.89, meaning the network is under-confident rather than over-confident here, and sharpening an already under-confident distribution moves it away from calibration. We suspect the cost weighting rather than the architecture, but that is a hypothesis not tested in the present study. One methodological point follows with more confidence: temperature must be fitted on a partition the network has not memorized, since fitting it on training logits produces a temperature above 1 and an apparent improvement that does not transfer. The bin sweep in Fig. 19c shows the ordering is stable from 5 to 30 bins.

![](figures_v2/fig_calibration.png)

**Fig. 19—Calibration.** (a) ECE with bootstrap intervals. (b) The same, split into all windows against hazardous windows only. (c) Sensitivity to the number of bins.

**Discussion.** The central result is a dissociation. Every model in Table 9 is a defensible choice on in-distribution evidence, and the orderings they receive under the class-disjoint shift do not carry across: the model that fails safe on one unseen hazard under-escalates on the other, the two that escalate fully on both are tree ensembles carrying none of the pipeline's safety machinery, and the deployable decision tree is perfect on one unseen hazard and among the worst on the other.

One counter-reading deserves acknowledgment. Leave-one-class-out could be called an unfairly severe test, since no fielded system is asked to classify a gas absent from its training set. We are not persuaded, because a MOX array in service meets cross-sensitivities and interferent mixtures that were in nobody's training set (Vergara et al. 2012), but the objection is reasonable and a reader who takes it seriously would draw a narrower conclusion than we do.

Why missed-hazard rate is not enough is the most transferable point here, and it is not specific to gas monitoring. Wherever a graded response hierarchy is learned, a metric defined on the most extreme failure can be satisfied by systematic under-response one rung above it, which is exactly what Eqs. 6 and 7 make visible and a miss metric alone collapses. The fix is to define the metric at the operationally meaningful threshold, which here is whether an operator is notified, rather than at the worst conceivable outcome. Experiment 6 shows the mirror-image trap: under complete sensor loss the cost-weighted policy and the decision tree score perfectly on missed hazard and escalation by escalating everything, while gradient boosting scores perfectly on false alarms by missing 60% of hazards. Each of the three metrics, taken alone, certifies a different model as safe.

What the learned policy contributes is narrow. In nominal conditions it matches the safety behavior of a shallow decision tree that scores 4.3 accuracy points lower, and its advantages are specific: full escalation on the held-out Smoke class where the perceptron misses 9.49%, and no observed missed hazard under 50% calibration drift where gradient boosting misses 7.75%. Each is bought with a false-alarm cost under sensor loss that the tree ensembles do not pay.

Finally, on standing relative to prior work on this corpus, published classification accuracies run from 91.7% to 99.7% and the decision accuracies here sit below most of them. Our evaluation uses a block-wise holdout with an explicit embargo, discards boundary-straddling windows, moves the partition with the seed and fits all scaling on the training partition, whereas the published comparisons appear to use randomized splits over overlapping windows from a single session; Experiment 1 quantifies what that difference is worth. In our favor, the quantities differ, since a classification error and an action error do not carry the same operational weight. For a like-for-like recognition comparison the relevant number is the thermal classifier's 98.8%, which was obtained under the same kind of randomized split those published figures use. That makes it comparable to them, and it makes all of them comparably optimistic. We are not claiming an accuracy advantage and nothing in the contribution rests on one.

---

## Model Application

The results above are measurements. This section sets out their operational implications. It is written as a proposal rather than as a deployment recommendation, because no deployment was carried out and nothing here has been through an acceptance process on a real installation. Three uses follow from the evidence, and each of them is an activity that the accuracy-only evaluation in the current literature cannot support.

**Selecting a Model When the Metrics Disagree.** The first use is negative and is the most immediately actionable. **Fig. 20** collapses every experiment above into one view: for each evaluation condition, which model does each metric crown, and can that metric choose at all.

![](figures_v2/fig_metricdisagreement.png)

**Fig. 20—Which model each metric selects, condition by condition.** Cells are shaded by whether the metric discriminates at all, agrees with the accuracy winner, or names a different model.

Two patterns matter for procurement. In distribution, accuracy names a winner in every row while the three safety metrics tie between seven and ten models, so a selection made on in-distribution evidence is a selection made on accuracy whether or not the other columns are printed. Under the conditions that do separate models the columns stop agreeing: escalation adequacy on the held-out Mixture class crowns gradient boosting, which accuracy ranks third, and at total sensor loss the false-alarm column crowns that same model for the opposite reason, because it has stopped alarming at all. The practical rule is that a selection defended on a single column is not defensible, and that the burden falls on the evaluator to show the columns agree rather than to assume it.

**A Proposed Evaluation Screen for Sensor Loss.** The second use is an illustrative qualification screen derived from the failure modes observed here, not a validated industry acceptance standard. Single-channel loss is a plausible and operationally relevant fault mode for a fixed sensor array, and Experiment 6 shows it separates models that are identical on every clean-condition metric. An acceptance test built only on clean data would pass all of them. **Table 17** sets out the screen we would propose, with each criterion stated alongside the measurement that motivates it.

| Screen | Criterion | Motivating measurement |
|---|---|---|
| Clean-condition indication burden | Report the upper bound a single evaluation partition supports, not the observed zero, and refer the bound to site-specific alarm-management review rather than to any universal threshold | A zero over one partition's 316 clean windows bounds the burden only at 9.4 per 1,000 windows |
| Single-channel loss | Re-measure all three safety metrics with *k* = 1 of 7 channels zeroed | One lost channel moved alarm burden from 0 to 117 and 200 per 1,000 windows for two of six models, and left a third untouched |
| Failure-mode declaration | Require the vendor to state whether the model fails loud or fails silent under total sensor loss, and to show it | At *k* = 7 two models alarm on every clean window while one raises no alarm and misses 60% of hazards |
| Held-out-class response | Evaluate on a hazardous class withheld from training, reporting escalation adequacy beside missed-hazard rate | Ten of eleven models pass a miss-rate screen on the held-out Mixture class while escalation adequacy across them runs from 0.005 to 1.000 |
| Upstream-input failure | Force each upstream input to a failure value and re-measure | One model's missed-hazard rate moved by a factor of thirty-five when a single input read zero |

**Table 17—A proposed evaluation screen for a learned gas-monitoring component,** with the measurement that motivates each criterion. The screen is illustrative, derived from the failure modes observed here, and is not proposed as a substitute for site-specific functional-safety or alarm-management requirements.

The screen is deliberately cheap. Every row but the fourth is a re-evaluation of an already-trained model on already-collected data; the held-out-class row costs one retrain per withheld class, which is the only retraining the screen requires. None of it needs a new data campaign, and each row corresponds to a failure this study observed rather than to one we imagined.

**Expressing False Alarms in Alarm-Management Units.** The third use concerns how a result is reported rather than what is measured. A false-alarm rate expressed as a fraction is not commensurable with the envelopes an operator works inside, and the conversion to a burden is what makes the number reviewable by the people who own the alarm system. The conversion has two steps and both should be stated: a rate per clean window becomes a count per 1,000 clean windows, which is a property of the model, and then a count per hour under an assumed duty cycle, which is a property of the deployment. Reporting only the second hides an assumption; reporting only the first leaves the reader unable to compare against EEMUA (2013) or ISA (2016). We report both, and we would ask the same of anyone proposing a monitoring model for a facility.

What this does not license is a claim about a complete alarm system. A learned component contributes indications to an alarm system that already has a budget, and the comparison in Experiment 6 is between one instrument's output and the envelope for an entire operator position. The right use of that comparison is as an order-of-magnitude screen, which is how a single lost channel producing roughly 211 indications per hour should be read: not as a prediction of operator load, but as a signal that the model's degraded behavior belongs in the alarm-management review rather than in the model-selection spreadsheet.

**What the Protocol Costs.** The argument above is that the metric set is cheap to adopt, and cheapness should be measured rather than asserted. Computing the five quantities of Table 5 is free in any sense that matters, since they are arithmetic over predicted actions the evaluation already has. The cost that is not negligible is retraining, and it falls unevenly across the protocol. Reporting the metric set on an existing evaluation needs no retraining at all. The graded perturbation suite and the upstream-input probe also need none, since both re-evaluate an already-trained model under modified inputs. Leave-one-class-out needs one retrain per withheld hazardous class, so two here. The 30-run comparison with the equivalence test is the only element that costs a multiple of the original training budget, and it is needed only when the question is model selection rather than model characterization. A facility qualifying a single candidate model therefore pays almost nothing for the screen in Table 17; a study ranking candidates pays for the run count, and Table 10 shows what that buys.

**Future Work.** Four directions follow, roughly in order of what we would do next. Cross-session and cross-device validation is the most important experiment still outstanding, since everything here comes from one acquisition session on one device and sensor ageing, device-to-device variability and long-horizon drift are all absent. Evaluation on a corpus containing a controlled hydrocarbon release would replace the surrogate analytes and is the only route to an external-validity claim. A tuned version of the ordinal cost matrix is the obvious follow-up to Experiment 4, since the untuned matrix establishes the mechanism and buys escalation at a price we would not pay. And class-conditional conformal prediction (Angelopoulos and Bates 2023) would turn the observed zero-miss counts into distribution-free coverage guarantees on the hazardous class, which is what a zero in a safety table ought to carry. Beyond those, a systematic treatment of upstream-component failure is warranted: Experiment 5 found one comparator whose missed-hazard rate moves by a factor of thirty-five when a single input is wrong, and we tested only one such input.

---

## Limitations and Threats to Validity

**Analyte and Acquisition Validity.** Incense smoke, alcohol-based vapor and a mixture of the two are surrogates, so nothing here establishes detection performance for methane or heavier hydrocarbons; external validity to petroleum facility monitoring is argued from the shared MOX transduction mechanism rather than shown. All 6,400 samples come from one acquisition session on one device, so sensor ageing, device-to-device variability, seasonal ambient swings and long-term drift are absent.

**Dependence Between Repeated Partitions.** Repeated blocked partitions share underlying observations: consecutive windows overlap by 19 of 20 raw rows, and because the held-out block position moves with the seed, the same window is evaluated in more than one partition. Uncertainty estimates must therefore treat the partition, not the window, as the experimental unit, which is what the bounds reported here do. Even so, overlap inside a partition means the effective number of independent observations is smaller than the nominal count, so every bound reported here remains optimistic by an amount this design cannot quantify.

**Statistical Power and the Correlation Assumption.** The model comparison uses 30 randomized runs and a Bayesian correlated *t*-test, which resolves two of the ten comparisons. Every other experiment runs on five seeds, where a Wilcoxon signed-rank test cannot reach α = 0.05 at any effect size, so no ranking should be read into the five-seed tables. The correlation term in Eq. 11 is carried across from a cross-validation setting; Table 10 shows the direction of every comparison is stable from ρ = 0 to ρ = 0.5, but the strength of the equivalence claims is conditional on that value.

**Metric Scope and the Alarm Boundary.** Missed-hazard rate counts only the fully passive action, which is why escalation adequacy is reported beside it. Escalation adequacy in turn depends on placing the alarm boundary at *a* ≥ 3, and Experiment 3 shows the spread on the held-out Mixture class collapses from a factor of 198 to a factor of 1.3 if the boundary moves to *a* ≥ 1. The boundary is defensible for this implementation, in which actions 1 and 2 place nothing in front of an operator, and a facility that implements them differently would report different numbers. Separately, action 2 is never a positive training target and is never selected, so the nominal five-level ladder is in effect a four-level one.

**The Ordinal Objective Is Untuned.** The expected-cost formulation of Eq. 12 is a first attempt and not a finished one. The matrix entries were chosen by hand for their structure and never optimized, so the accuracy they cost is a consequence of that choice rather than of the formulation.

**Perception Split, and the Asymmetry It Creates.** The thermal images were collected and used, and the classifier trained on them is part of the pipeline that produced every result here, but its split is randomized over consecutive frames from one session. The two paths through the pipeline are therefore held to different evidential standards. No conclusion rests on the perception number, since the decision state carries no visual term, but the multimodal system as a whole should not be read as validated to the standard applied to the sensor path.

**Perturbation and Probe Realism.** Drift is modeled as a per-channel gain and baseline shift applied at test time, and dropout as channel zeroing, neither of which reproduces the temporal character of real MOX ageing. The upstream-input probe forces the anomaly feature to fixed values while the other twenty-one are held at their observed values, so it isolates a dependence rather than reproducing a realistic failure.

**No Hardware Evaluation, and an Unevaluated Component.** Nothing here establishes that the pipeline runs inside any particular latency, memory or power budget, because no deployment was carried out. The explanation component is described but never timed, never inspected and never evaluated; it supports no result in this paper, and a reader should treat it as apparatus rather than as a contribution.

---

## Conclusion

This study evaluated a gas-monitoring pipeline against a broader metric set than accuracy alone, on a public MOX benchmark of laboratory surrogate analytes, under a leakage-controlled block-wise holdout with seed-varying partitions. Four findings follow.

First, the partitioning protocol materially changes reported accuracy. On this corpus a random split over overlapping sliding windows raised decision accuracy by 1.66 to 4.08 percentage points for every model that reads temporal features, and among those five models that effect exceeded the 2.48-point spread separating the models themselves. A stricter protocol that trains on early windows and tests on late ones cost three of six models between 27 and 29 points. Comparisons across studies that use different protocols therefore carry little information about relative merit.

Second, in-distribution accuracy does not reliably predict safety behavior under a class-disjoint shift. Across 30 randomized runs ten comparators lie within 5.66 accuracy points, and all but one record no observed missed hazards, escalation adequacy of 1.000 and no observed high-severity false alarms, so the safety metrics are weakly discriminative under the nominal distribution. Removing the cost weighting is practically equivalent to keeping it, a conclusion that holds under a Bayesian correlated *t*-test across the full range of the assumed run correlation and under a paired bootstrap that assumes none. Withholding one hazardous surrogate class then separates those same models by more than two orders of magnitude on missed-hazard rate, with a model ranked fifth of ten by accuracy recording the highest missed-hazard rate in the field.

Third, missed-hazard rate alone can reward under-escalation. On a second held-out hazardous class, ten of eleven comparators have 95% upper confidence bounds at or below 3.3% on missed hazards while escalation adequacy across them runs from 0.005 to 1.000. The spread lives entirely in the sub-alarm band, which we established by moving the alarm boundary down one rung: escalation then runs from 0.751 to 1.000 and the models become hard to tell apart. The disagreement is therefore a disagreement about how a facility implements a sub-alarm response, and it is a question a miss metric cannot be asked at all.

Fourth, synthetically imposed sensor degradation produces distinct failure modes that are invisible in nominal accuracy. Fail-loud models raise an alarm-grade indication on every clean window while recording no observed missed hazard. Fail-silent models record no false alarm while missing 60% of hazardous windows. A third pattern degrades gracefully to a threshold and then abruptly. Losing one channel of seven moved the cost-weighted policy from zero to 117 alarm-grade indications per 1,000 clean windows and left the random forest untouched.

Safety-relevant evaluation should therefore report accuracy together with missed-hazard rate and its confidence bound, escalation adequacy, false-alarm burden in alarm-management units, and the degradation of all three under sensor loss and drift. The cost is a few extra columns per experiment, most of them a re-evaluation of an already-trained model on data already collected. The benefit, on the evidence here, is that model-selection decisions reverse: an evaluation reporting accuracy alone would have said nothing useful about any of these behaviors, and each safety metric taken on its own would have selected a different model.

---

## Nomenclature

| Symbol | Definition |
|---|---|
| *A* | indication rate per hour under an assumed duty cycle, Eq. 13 |
| Acc | decision accuracy, Eq. 4 |
| *a* | safety action, *a* ∈ {0, 1, 2, 3, 4} |
| *a*\*(·) | target-action map from gas class to acceptable action set, Eq. 2 |
| *B* | alarm burden, high-severity false alarm-grade indications per 1,000 clean windows, Eq. 9 |
| *C* | cost ratio, *C* = *c*_miss / *c*_false |
| *c* | confidence level in the Clopper-Pearson bound, Eq. 10 |
| *c*_miss, *c*_false | loss weights for hazardous and non-hazardous training samples |
| *d* | dimension of the decision state in the ablation |
| *E*ₜ | operator-facing explanation at cycle *t* |
| *f* | windows per hour under the assumed duty cycle, Eq. 13 |
| *G* | embargo length between training and test partitions, in windows |
| *g*ₜ | ground-truth gas class for window *t* |
| *H* | index set of hazardous test windows, Smoke and Mixture |
| *I*ₜ | thermal image at cycle *t* |
| *k* | sensor channels removed in the dropout sweep, or observed failures in Eq. 10 |
| *L*(θ) | cost-weighted cross-entropy objective, Eq. 3 |
| *L*_ord(θ) | expected-cost objective under the ordinal matrix, Eq. 12 |
| *M* | ordinal cost matrix over (class, action) pairs |
| *N* | number of training samples |
| *n* | number of test windows in Eqs. 4 to 9; opportunities in Eq. 10; paired runs in Eq. 11 |
| *p*_θ | action probability distribution under parameters θ |
| *R*_esc | escalation adequacy, hazardous windows at *a* ≥ 3, Eq. 6 |
| *R*_fa | high-severity false-alarm rate, clean windows at *a* ≥ 3, Eq. 8 |
| *R*_miss | missed-hazard rate, hazardous windows at *a* = 0, Eq. 5 |
| *R*_under | under-escalation, hazardous windows at *a* ∈ {1, 2}, Eq. 7 |
| *s*_post | posterior scale in the correlated *t*-test, Eq. 11 |
| *U* | one-sided upper confidence bound on a rate, Eq. 10 |
| **X**ₜ | seven-channel sensor window, **X**ₜ ∈ ℝ^(20×7) |
| *Z* | index set of clean test windows, NoGas |
| **δ**ₜ | per-sensor change across the window |
| θ | model parameters |
| ρ | assumed correlation between runs in the Bayesian *t*-test |
| ρ̃ₜ | normalized reconstruction-error anomaly score |
| **σ**ₜ | per-sensor standard deviation across the window |
| σₙ | additive-noise standard deviation in the perturbation suite |
| τ | anomaly detection threshold |
| φₜ | 22-dimensional decision state, Eq. 1 |

**Subscript.** *t*, inference cycle or window index.

**Abbreviations.** CP, Clopper-Pearson; ECE, expected calibration error; FA, false alarm; MOX, metal-oxide semiconductor; ROC-AUC, area under the receiver operating characteristic curve; SD, standard deviation.

## Acknowledgments

This work was supported by the Subsurface Energy and Digital Innovation Center at the University of Wyoming. The MultimodalGasData corpus is used under CC BY 4.0, and we thank Narkhede et al. for placing it in the public domain.

## Author Contributions

**B. C. Nweke:** conceptualization, methodology, software, investigation, formal analysis, data curation, visualization, writing of the original draft. **G. Ramezan:** conceptualization, methodology, writing, review and editing. **S. Saraji:** conceptualization, supervision, project administration, resources, writing, review and editing.

## Declaration of Competing Interest

The authors declare no known competing financial interests or personal relationships that could have appeared to influence the work reported here.

## Data and Code Availability

The MultimodalGasData corpus is publicly available (Narkhede et al. 2022) under CC BY 4.0. Experiment drivers, the safety-metric module, the additional baselines, model checkpoints and the result files behind Tables 7 through 16 are available from the corresponding author. See Appendix A.

---

## References

Adegboye, M. A., Fung, W. K., and Karnik, A. 2019. Recent Advances in Pipeline Monitoring and Oil Leakage Detection Technologies: Principles and Approaches. *Sensors* 19 (11): 2548. https://doi.org/10.3390/s19112548.

Angelopoulos, A. N. and Bates, S. 2023. Conformal Prediction: A Gentle Introduction. *Foundations and Trends in Machine Learning* 16 (4): 494–591. https://doi.org/10.1561/2200000101.

Benavoli, A., Corani, G., Demšar, J. et al. 2017. Time for a Change: A Tutorial for Comparing Multiple Classifiers Through Bayesian Analysis. *Journal of Machine Learning Research* 18 (77): 1–36.

Bouthillier, X., Delaunay, P., Bronzi, M. et al. 2021. Accounting for Variance in Machine Learning Benchmarks. *Proc., Machine Learning and Systems (MLSys)* 3: 747–769.

Bustnes, T. E., Rousselet, M., and Berland, S. 2011. Leak Detection Performance of a Commercial Real-Time Transient Model. Paper presented at the PSIG Annual Meeting, Napa Valley, California, 24–27 May. PSIG-1114.

Clopper, C. J. and Pearson, E. S. 1934. The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial. *Biometrika* 26 (4): 404–413. https://doi.org/10.1093/biomet/26.4.404.

Dehnaw, A. M., Lu, Y.-J., Shih, J.-H. et al. 2024. Deep Neural Network Optimization for Efficient Gas Detection Systems in Edge Intelligence Environments. *Processes* 12 (12): 2638. https://doi.org/10.3390/pr12122638.

Dennler, N., Rastogi, S., Fonollosa, J. et al. 2022. Drift in a Popular Metal Oxide Sensor Dataset Reveals Limitations for Gas Classification Benchmarks. *Sensors and Actuators B: Chemical* 361: 131668. https://doi.org/10.1016/j.snb.2022.131668.

EEMUA. 2013. *Alarm Systems: A Guide to Design, Management and Procurement*, third edition. EEMUA Publication 191. London: Engineering Equipment and Materials Users Association.

El Barkani, M., Benamar, N., Talei, H. et al. 2024. Gas Leakage Detection Using Tiny Machine Learning. *Electronics* 13 (23): 4768. https://doi.org/10.3390/electronics13234768.

Elkan, C. 2001. The Foundations of Cost-Sensitive Learning. *Proc., 17th International Joint Conference on Artificial Intelligence (IJCAI)*, Seattle, Washington, 4–10 August, 973–978.

Faleh, R. and Kachouri, A. 2023. A Hybrid Deep Convolutional Neural Network-Based Electronic Nose for Pollution Detection. *Chemometrics and Intelligent Laboratory Systems* 237: 104825. https://doi.org/10.1016/j.chemolab.2023.104825.

Gemma Team, Google DeepMind. 2025. Gemma 3 Technical Report. arXiv:2503.19786 (preprint, submitted 25 March 2025).

Goel, P., Datta, A., and Mannan, M. S. 2017. Industrial Alarm Systems: Challenges and Opportunities. *Journal of Loss Prevention in the Process Industries* 50: 23–36. https://doi.org/10.1016/j.jlp.2017.09.001.

Han, P. and Kim, M. 2014. Optimizing Leak Detection Performance. Paper presented at the PSIG Annual Meeting, Baltimore, Maryland, 6–9 May. PSIG-1407.

IEC. 2016. *Functional Safety: Safety Instrumented Systems for the Process Industry Sector, Part 1: Framework, Definitions, System, Hardware and Application Programming Requirements*, IEC 61511-1:2016 with Amendment 1:2017. Geneva: International Electrotechnical Commission.

ISA. 2016. *Management of Alarm Systems for the Process Industries*, ANSI/ISA-18.2-2016. Research Triangle Park, North Carolina: International Society of Automation.

Jessen, H. and Roshchin, M. 2025. Agentic AI Revolution Across O&G Value Streams: New Strategy Through ENERGYai Examples. Paper SPE-229240-MS presented at the Abu Dhabi International Petroleum Exhibition and Conference, Abu Dhabi, UAE, 3–6 November.

Jocher, G., Chaurasia, A., and Qiu, J. 2023. YOLOv8 by Ultralytics. https://github.com/ultralytics/ultralytics.

Korjani, M., Conley, D., and Smith, M. 2024. Temporal Deep Learning Image Processing Model for Natural Gas Leak Detection Using OGI Camera. Paper OTC-34756-MS presented at the Offshore Technology Conference Asia, Kuala Lumpur, Malaysia, 27 February–1 March.

Laberge, J. C., Bullemer, P., Tolsma, M. et al. 2014. Addressing Alarm Flood Situations in the Process Industries Through Alarm Summary Display Design and Alarm Response Strategy. *International Journal of Industrial Ergonomics* 44 (3): 395–406. https://doi.org/10.1016/j.ergon.2013.11.008.

Liang, J., Liang, S., Zhang, H. et al. 2023. Leak Detection in Natural Gas Pipelines Based on Unsupervised Reconstruction of Healthy Flow Data. *SPE Prod & Oper* 38 (3): 513–526. SPE-214686-PA. https://doi.org/10.2118/214686-PA.

Malhotra, P., Ramakrishnan, A., Anand, G. et al. 2016. LSTM-Based Encoder-Decoder for Multi-Sensor Anomaly Detection. arXiv:1607.00148 (preprint, submitted 1 July 2016).

Murvay, P.-S. and Silea, I. 2012. A Survey on Gas Leak Detection and Localization Techniques. *Journal of Loss Prevention in the Process Industries* 25 (6): 966–973. https://doi.org/10.1016/j.jlp.2012.05.010.

Narkhede, P., Walambe, R., Mandaokar, S. et al. 2021. Gas Detection and Identification Using Multimodal Artificial Intelligence Based Sensor Fusion. *Applied System Innovation* 4 (1): 3. https://doi.org/10.3390/asi4010003.

Narkhede, P., Walambe, R., Chandel, P. et al. 2022. MultimodalGasData: Multimodal Dataset for Gas Detection and Classification. *Data* 7 (8): 112. https://doi.org/10.3390/data7080112.

Page, E. S. 1954. Continuous Inspection Schemes. *Biometrika* 41 (1/2): 100–115. https://doi.org/10.1093/biomet/41.1-2.100.

Sabbagh, V. B., Lima, C. B. C., and Xexéo, G. 2024. Comparative Analysis of Single and Multiagent Large Language Model Architectures for Domain-Specific Tasks in Well Construction. *SPE J.* 29 (12): 6869–6882. SPE-223612-PA. https://doi.org/10.2118/223612-PA.

Santiago, C. J. S., Shumaker, N., and Weir, A. 2025. Integrating Data-Driven Insights with Domain Expertise Using Agentic Conversational Analytics for Well Completions Optimization. Paper SPE-228143-MS presented at the SPE Annual Technical Conference and Exhibition, Houston, Texas, 20–22 October.

Sharma, A., Khullar, V., Kansal, I. et al. 2024. Gas Detection and Classification Using Multimodal Data Based on Federated Learning. *Sensors* 24 (18): 5904. https://doi.org/10.3390/s24185904.

Vergara, A., Vembu, S., Ayhan, T. et al. 2012. Chemical Gas Sensor Drift Compensation Using Classifier Ensembles. *Sensors and Actuators B: Chemical* 166–167: 320–329. https://doi.org/10.1016/j.snb.2012.01.074.

Wang, Z., Schaul, T., Hessel, M. et al. 2016. Dueling Network Architectures for Deep Reinforcement Learning. *Proc., 33rd International Conference on Machine Learning (ICML)*, New York City, 19–24 June, 1995–2003.

Zhang, Eddie and Zhang, Evan. 2025. Gas Pipeline Leakage Detection Based on Multiple Multimodal Deep Feature Selections and Optimized Deep Forest Classifier. *Frontiers in Environmental Science* 13: 1569621. https://doi.org/10.3389/fenvs.2025.1569621.

Zhang, J., Hoffman, A., Murphy, K. et al. 2013. Review of Pipeline Leak Detection Technologies. Paper presented at the PSIG Annual Meeting, Prague, Czech Republic, 16–19 April. PSIG-1303.

---

## Appendix A—Reproducibility

**Pipeline.** Windows of length 20 formed over the raw corpus, with windows spanning a class boundary discarded, giving 6,324 windows at 1,581 per class. The anomaly feature is the mean reconstruction error of the full 20-step window under the pretrained NoGas autoencoder. The block-wise holdout reserves a contiguous 20% of each class block for test, separated from both training segments by a 20-window embargo, with the held-out block's position drawn from the run seed. Feature standardization and anomaly-percentile normalization are fitted on the training partition only. In the leave-one-class-out experiment they are fitted on the three training classes only, so no statistic of the withheld class enters training.

**Protocol.** Two protocols, as set out under Proposed Method. The model comparison uses 30 randomized runs (seeds 1000 to 1029) varying block position, initialization and a hyperparameter draw. Everything else uses five seeds (42, 1337, 7, 2024, 99), each with its own partition. PyTorch determinism guards enabled, single-threaded execution. The reported runs used Python 3.11.15, PyTorch 2.14.0, scikit-learn 1.8.0, NumPy 2.4.4, SciPy 1.17.1 and pandas 3.0.2 on CPU. The thermal classifier was trained separately with Ultralytics YOLOv8n-cls; the exact Ultralytics version used for that training is recorded in the project environment rather than reproduced here, and should be stated before submission. Each of the seven experiments was executed in a single uninterrupted run.

**Artifacts.** `retrain/safety_metrics.py` (metric set and Clopper-Pearson bounds); `retrain/new_baselines.py` (k-NN, decision tree, CUSUM); `retrain/run_all_v2.py` (model comparison, cost sweep, perturbation, ablation, anomaly evaluation); `retrain/run_loco_v3.py` (leave-one-class-out with three-class scaling); `retrain/run_calib_actions_v2.py` (calibration, action-selection matrices, ROC curves); `retrain/run_anomaly_influence.py` (anomaly-input sensitivity); `retrain/run_final.py` (30-run comparison with the Bayesian equivalence test, extended leave-one-class-out, ordinal cost objective, cost-ratio selection); `retrain/run_leakage.py` (partitioning-protocol comparison and blocked validation); `retrain/run_rho_sensitivity.py` (sweep of the correlation term in the Bayesian test); `retrain/make_figs_v2.py` (Figs. 1 to 20); `verify_v10.py`, which parses the tables in this manuscript and checks every quoted value against the result files below. Results in `retrain/results_v2/`: `v2_zoo.csv`, `v2_zoo_sig.json`, `v2_loco.csv`, `v2_costsweep.csv`, `v2_perturb.csv`, `v2_ablation.csv`, `v2_anomaly.json`, `v2_calibration.csv`, `v2_calibration_perseed.csv`, `v2_calibration_binsweep.csv`, `v2_action_matrix.csv`, `v2_roc.json`, `v2_anomaly_influence.csv`, `v3_leakage.csv`, `v3_powered.csv`, `v3_powered_bayes.json`, `v3_loco_all.csv`, `v3_cost_validation.csv`, `v3_cost_blocked_validation.csv`, `v3_rho_sensitivity.json`.

**Corrections made during verification.** Two defects were found while the results were being audited against the stored files and were corrected before the runs reported here: a false-alarm denominator that divided by the number of alerts rather than by the number of clean windows, and a leave-one-class-out scaler fitted on all four classes rather than on the three training classes. Both are fixed in the released code, and every number in this manuscript comes from the corrected runs.

**Figure palette.** Categorical colors were checked against colorblind-separation, chroma, lightness and contrast criteria. The green and orange pair sits in the marginal separation band, so every categorical mark also carries a direct value label as a second channel of encoding.
