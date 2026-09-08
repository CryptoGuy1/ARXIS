# ARXIS: Edge-Deployable Agentic AI for Safety-Critical Gas Monitoring and Action Selection

**Benjamin C. Nweke**^{a,b,∗}, **Gholamreza Ramezan**^{b}, **Soheil Saraji**^{a}

^{a} Subsurface Energy and Digital Innovation Center (SEDI), Department of Energy and Petroleum Engineering, University of Wyoming, 1000 E. University Ave., Laramie, 82071, WY, USA
^{b} Fides Innova Labs, Vancouver, BC, Canada

∗ Corresponding author. bnweke@uwyo.edu (B.C. Nweke); reza@fidesinnova.io (G. Ramezan); ssaraji@uwyo.edu (S. Saraji)

---

## ARTICLE INFO

**Keywords:** Gas-hazard monitoring; Safety-action selection; Agentic AI; Edge deployment; Anomaly detection; Reinforcement learning; Explainable AI; Data analytics; Machine learning

---

## ABSTRACT

Gas-monitoring systems in process industries must support more than abnormal-condition detection; they must also help operators select a defensible response when evidence is uncertain, degraded, or incomplete. Existing gas-detection and classification models often report gas labels but rarely evaluate whether the resulting decision avoids missed hazards or unnecessary high-severity alarms. This paper presents ARXIS (Agentic Reasoning with eXplainability for Infrastructure Surveillance), an edge-deployable agentic AI framework that combines thermal-image perception, sensor-based anomaly scoring, offline reward-shaped safety-action selection, and local natural-language explanation in a sense–decide–explain workflow. On the public MultimodalGasData benchmark, evaluated using a leakage-controlled block-wise holdout, ARXIS achieved 94.44% decision accuracy with zero danger misses across 640 hazardous windows and zero high-severity false alarms across 316 baseline windows. Under calibration drift, a supervised baseline trained on the same labels began missing hazards (danger miss rate = 0.0187), whereas ARXIS preserved zero danger misses. A Raspberry Pi 4 proof of concept showed that asynchronous explanation and model quantization reduced the deployable decision path from approximately 14.8 s to 235 ms. Comprehensive baseline comparisons across 8 models (Decision Agent, Plain DQN, MLP, GBM, SVM, Random Forest, LSTM, CQL) demonstrate that no baseline significantly outperforms ARXIS at α=0.05, while ARXIS uniquely preserves zero danger misses under distribution shift. These results show that missed-hazard behavior, false-alarm behavior, degradation robustness, and response latency should be reported alongside accuracy.

---

## 1. Introduction

Major process-safety incidents are seldom caused by a complete absence of warning. More often, abnormal signals are present but are not converted into timely and proportionate operator action. This distinction is central to gas-hazard monitoring in process industries, where the safety outcome depends not only on detecting abnormal evidence but also on prioritizing it and selecting an appropriate response.

Alarm-management studies have long shown that alarm floods can overwhelm operators and reduce the effectiveness of abnormal-situation response [1]. The Milford Haven refinery explosion illustrates this difficulty: in the eleven minutes before the blast, the operator was confronted with 275 alarms, showing that the problem was not simply the absence of signals but the failure to convert them into actionable information under time pressure [2].

Gas-leak detection and corrective response are related but distinct problems [3]. Knowing that a leak exists is not sufficient to determine the appropriate corrective measure. In practice, however, many monitoring systems still reduce the task to a binary threshold or a class label. A threshold tuned for sensitivity may generate frequent false alarms and erode operator trust; a threshold tuned for specificity may miss genuine hazardous events [4]. Neither behavior reflects how process-safety decisions are made in operation.

The monitoring problem is therefore better formulated as **safety-action selection under uncertainty** than as gas classification alone. Depending on the severity, confidence, and persistence of the evidence, an operator may continue routine monitoring, increase sampling, request verification, raise an alarm, or initiate an emergency shutdown.

### 1.1 Research Gap

Several technical components needed for such a system have matured:
- Normal-only autoencoder models can learn baseline sensor behavior and detect deviations through reconstruction error [5,6]
- Computer vision models can classify thermal or optical gas signatures [7]
- Reinforcement learning provides a way to learn action policies when the costs of errors are unequal [8,9]
- Agentic AI systems can coordinate specialized models for sensing, reasoning, and decision support [10,11,12]
- Compact language models make local explanation generation feasible on edge devices [13,14]

These advances are useful, but they are usually treated separately. Three gaps are most relevant:
1. Many gas-monitoring methods end at detection or classification and do not model the downstream safety response.
2. Reported metrics often emphasize accuracy, while safety-relevant quantities such as missed hazards, high-severity false alarms, and behavior under sensor degradation are less consistently reported.
3. Agentic and explainable AI methods are seldom evaluated as edge-deployable components of an operator-facing safety workflow.

### 1.2 Contributions

This paper presents ARXIS, an edge-deployable agentic AI framework that addresses these gaps. The specific contributions are fourfold:

1. **Modular monitoring pipeline** of five coordinated components—thermal-image perception, normal-only sensor anomaly scoring, decision-state construction, offline reward-shaped safety-action selection, and local natural-language explanation—organized as a sense–decide–explain workflow.

2. **Safety-constrained decision-making** formulation rather than static thresholding or gas classification, using offline benchmark replay and an asymmetric safety reward to penalize passive responses to hazardous conditions more heavily than conservative verification actions.

3. **Safety-relevant evaluation** on the public MultimodalGasData benchmark. On the block-wise holdout test set, ARXIS attains 94.44% decision accuracy with zero danger misses across 640 hazardous windows and zero high-severity false alarms across 316 baseline windows.

4. **Edge-deployment proof of concept** on a Raspberry Pi 4, showing that asynchronous explanation and model quantization reduce the deployable decision path to 235 ms while preserving zero danger misses.

---

## 2. Advances in AI-Based Gas-Hazard Monitoring

AI-based gas-hazard monitoring has progressed from signal detection toward richer forms of perception, anomaly scoring, decision support, and explanation. For process-safety applications, however, the critical question is not only whether a model recognizes an abnormal condition. It is whether the evidence is translated into a timely and defensible operator response.

### 2.1 Pipeline and gas leak detection

Although the present framework is AI-based, gas-hazard detection rests on an established base of hardware- and software-based leak-detection methods [15]. Hardware-based methods include distributed acoustic sensing, fiber-optic sensing, vapor sensing, and negative-pressure-wave detection. Software-based methods include real-time transient modeling, mass balance, statistical analysis, and model-based diagnostics.

These methods have improved leak identification and localization in pipeline systems, where delayed detection can increase the risk of fire, explosion, environmental damage, and production loss. The main limitation is not the absence of detection methods, but their uneven performance under practical operating conditions [16,17]. These studies provide the foundation for reliable leak recognition, but they usually end with a leak indication, location estimate, or concentration estimate. Those outputs are necessary for loss prevention, but they are not the same as a response decision.

### 2.2 Deep learning for anomaly detection

Deep learning has improved gas-hazard monitoring by learning patterns in multivariate sensor streams. Reconstruction-based models are especially useful because they can be trained on normal operation and then used to flag deviations. The LSTM encoder–decoder introduced by Malhotra et al. [5] established a widely used reconstruction-error approach for multivariate anomaly detection. More recent work applies normal-only learning to gas-pipeline monitoring [6].

These methods are valuable because they provide an early indication that the current sensor pattern has departed from normal operation. Their limitation is that an anomaly score is still an intermediate signal. It does not specify the appropriate response. Anomaly detection therefore provides an important evidence layer, but it must be connected to a decision layer before it can support graded safety response.

### 2.3 Computer vision and multimodal gas detection

Visual and thermal sensing complement point sensors by capturing plume patterns, thermal signatures, or image-level indicators of gas-related conditions. Models from the YOLO family offer a favorable speed–accuracy tradeoff for embedded vision tasks [18], and optical gas-imaging models have shown strong detection performance under controlled and field conditions [7].

Multimodal methods extend this capability by combining visual evidence with sensor-array measurements. The MultimodalGasData benchmark, for example, pairs thermal images with metal-oxide gas-sensor readings and shows that fusing the two modalities improves gas classification compared with either modality alone [19,20]. Most multimodal studies still evaluate performance as gas-classification accuracy. This is useful, but incomplete for process safety.

### 2.4 Reward-shaped policy learning for safety-oriented decisions

Safety-oriented monitoring requires more than accurate recognition of abnormal conditions. It also requires a decision rule that reflects unequal consequences of different errors. Deep Q-networks showed that neural networks can approximate action-value functions for decision-making [8], and dueling architectures improved value estimation by separating a state-value component from action-specific advantages [9].

For the present study, the relevant idea is not open-ended autonomous control. Rather, it is the use of a reward-shaped policy to select among a finite set of operator-facing safety actions under unequal error costs. The Decision Agent is therefore trained using offline benchmark replay: each window provides a monitoring state, and the asymmetric safety reward evaluates the selected action for that fixed state.

### 2.5 Agentic AI and operator-facing explanation

Agentic AI systems coordinate specialized models or tools toward a larger task. In the energy sector, recent systems have been applied to well-construction assistance, completions optimization, and enterprise modeling workflows [10,11,12]. These systems show the value of coordinating multiple reasoning components, but they are mostly designed for analytical or advisory settings rather than real-time process-safety monitoring.

Edge optimization and compact language models offer a path toward local inference and local explanation [13,14]. Explanation is important because operator-facing safety systems must support trust, review, and accountability. The unresolved issue is that explanation is often separated from action. Post-hoc interpretation may clarify why a model produced an output, but it does not necessarily tell the operator what to do next.

### 2.6 Research gap

Table 1 summarizes representative work across the capability areas that inform ARXIS. The literature shows substantial progress in leak detection, anomaly scoring, visual recognition, multimodal classification, reinforcement learning, and explanation. The common limitation is that these capabilities are rarely integrated into a single workflow that carries evidence from sensing to action.

**Table 1.** Representative prior work across the capability areas informing ARXIS.

| Study | Area | Contribution | Limitation relevant to ARXIS |
|-------|------|--------------|------------------------------|
| Liang et al. [6] | Anomaly detection | Field-validated normal-only leak detection | Quantifies anomaly but prescribes no operator action |
| Malhotra et al. [5] | Anomaly detection | Reconstruction-error scoring for sensor streams | No mapping from anomaly to operational response |
| Korjani et al. [7] | Computer vision | High detection rates on optical gas imaging | No decision layer; performance varies with weather |
| Narkhede et al. [20] | Multimodal sensing | Thermal–sensor fusion improves gas classification | Reports classification accuracy, not safety metrics |
| Mnih et al. [8] | Reinforcement learning | Neural value functions for sequential decisions | Developed for game domains; not safety-framed |
| Sabbagh et al. [10] | Agentic AI | Multi-agent reasoning improves answer quality | Analytical workflow; not real-time monitoring |
| Ma et al. [21] | Explainability | Multimodal fusion with post-hoc attribution | Explanation not coupled to a graded action |

---

## 3. Methodology

This section describes the ARXIS framework and the experimental procedure used to evaluate it. The aim is to establish a reproducible benchmark-based proof of concept for gas-hazard monitoring and safety-action selection.

### 3.1 Dataset and preprocessing

The MultimodalGasData dataset [19] was used for framework development and proof-of-concept evaluation. The dataset contains synchronized gas-sensor and thermal-image measurements collected under controlled exposure conditions. It includes 6,400 labeled samples distributed equally across four environmental classes: NoGas, Smoke, Perfume, and Mixture, with 1,600 samples per class.

Each sample contains two synchronized sensing modalities. The first is a thermal image captured with a Seek Compact thermal camera at a native resolution of 206 × 156 pixels. The second is a seven-channel MQ-series metal-oxide-semiconductor gas-sensor array. Table 2 summarizes the sensor array and the primary gas or vapor targets associated with each sensor.

**Table 2.** MQ sensor array composition and primary detection targets.

| Sensor | Primary detection target |
|--------|--------------------------|
| MQ-2 | LPG, butane, methane, smoke |
| MQ-3 | Alcohol, ethanol, vapors |
| MQ-5 | LPG, natural gas |
| MQ-6 | LPG, butane, iso-butane |
| MQ-7 | Carbon monoxide |
| MQ-8 | Hydrogen |
| MQ-135 | Air-quality indicators (NH₃, benzene, NOₓ, CO₂) |

For time-series modeling, the seven-channel sensor data were converted into sliding windows of length 20. This windowing step creates substantial temporal overlap because consecutive windows share 19 of 20 raw rows. A random train–test split would therefore place near-duplicate windows in both partitions and could inflate the reported performance. A simple sequential 80/20 split is also unsuitable because the source data are block-ordered by class.

To reduce both risks, a leakage-controlled block-wise holdout was used. For each class $k$, the final 20% of sequential windows were reserved for testing, while the earlier windows were used for training. A temporal gap of $G=20$ windows was discarded between the training and test segments to prevent overlap between the last training window and the first test window:

$$\text{For each class } k: \underbrace{[\text{train}]}_{\approx 80\%} \underbrace{[\text{gap}=G]}_{\text{discarded}} \underbrace{[\text{test}]}_{\approx 20\%}$$

This procedure produced 5,024 training windows and 1,276 test windows: 316 NoGas, 320 Smoke, 320 Mixture, and 320 Perfume windows. All normalization parameters were estimated from the training partition only and then applied to the test partition without refitting.

### 3.2 System architecture

ARXIS is organized as a modular five-agent framework for gas-hazard monitoring and safety-oriented decision support. Fig. 1 shows the overall workflow. At each inference cycle, the framework receives a thermal image $I_t$ and a seven-channel sensor window $\mathbf{X}_t \in \mathbb{R}^{20 \times 7}$. It returns a safety action $a_t \in \{0,1,2,3,4\}$ and a natural-language explanation $E_t$.

The workflow proceeds in four stages. First, the Perception Agent analyzes the thermal image and returns a gas-class probability vector. Second, the Anomaly Detection Agent analyzes the sensor window and returns a reconstruction-error anomaly score. Third, the Multimodal Agent assembles the decision state from the available evidence and forwards it to the Decision Agent. Fourth, the Decision Agent selects a safety action, and the Reasoning Agent generates an explanation from the decision context. The final action and explanation are sent to the monitoring dashboard.

Four design principles guide the framework. First, modularity allows each agent to be developed, tested, and replaced independently. Second, safety prioritization is encoded through an offline reward-shaped decision policy that penalizes missed hazardous events more heavily than precautionary responses. Third, explanation is treated as part of the operator-facing workflow rather than as a detached post-processing step. Fourth, the framework is designed for edge deployment on modest hardware.

### 3.3 Agent descriptions

**Perception Agent.** The Perception Agent provides image-level evidence about the gas condition. It classifies each thermal image into one of four environmental classes using YOLOv8n-cls [22]. The nano classification variant was selected because it provides a lightweight speed–accuracy tradeoff suitable for edge-oriented deployment. Each image $I_t$ is resized to 224×224 pixels and normalized before inference:

$$\mathbf{p}_t = \text{Softmax}(W_c \cdot \text{GAP}(f_{\text{CSP}}(I'_t)) + b_c)$$

where $f_{\text{CSP}}(\cdot)$ denotes the CSP-based feature extractor, GAP(⋅) is global average pooling, and $W_c \in \mathbb{R}^{4 \times d}$ and $b_c \in \mathbb{R}^4$ are the learned classification-head parameters. The predicted class is $c_t = \arg\max(\mathbf{p}_t)$.

Training used an 80/20 stratified split of the 6,400 thermal images, pretrained-weight initialization, and AdamW with learning rate $10^{-3}$, cosine annealing, weight decay $10^{-4}$, batch size 32, and 50 epochs. Random flipping, rotation, and brightness augmentation were used during training. The image-level split is separate from the sensor-window split described in Section 3.1 and does not create the same temporal leakage risk.

**Anomaly Detection Agent.** The Anomaly Detection Agent provides sensor-based evidence of departure from normal operation. It is implemented as an LSTM autoencoder trained only on NoGas samples, following the reconstruction-error approach used in unsupervised time-series anomaly detection [5,6]. Training on baseline behavior allows the model to score abnormal windows without requiring labeled examples of every hazardous condition.

The encoder is a single-layer LSTM with hidden dimension 32. It maps a normalized window $\mathbf{X}'_t$ into a latent representation $(\mathbf{h}, \mathbf{c}) = \text{LSTMEncoder}(\mathbf{X}'_t)$. A single-layer decoder with matching hidden dimension reconstructs the sequence through a linear output layer: $\hat{\mathbf{X}}_t = \text{LinearDecoder}(\mathbf{h}, \mathbf{c})$.

The final-step reconstruction mean-squared error is used as the anomaly score:

$$\rho_t = \frac{1}{7} \sum_{j=1}^{7} (x'_{t,j} - \hat{x}_{t,j})^2$$

The autoencoder was trained for 40 epochs using Adam with learning rate $10^{-3}$ and batch size 64. The detection threshold $\tau$ was calibrated as the 95th percentile of reconstruction errors over the NoGas training partition:

$$\tau = \hat{F}_{\rho}^{-1}|_{\text{NoGas}}(0.95)$$

This threshold provides a training-set-calibrated operating point without using anomaly-class labels. Before being passed to the Decision Agent, the reconstruction error is normalized to [0,1] using empirical percentile bounds from the training partition:

$$\tilde{\rho}_t = \text{clip}\left(\frac{\rho_t - \hat{p}_1}{\hat{p}_{99} - \hat{p}_1 + \epsilon}, 0, 1\right)$$

where $\hat{p}_1$ and $\hat{p}_{99}$ are the empirical 1st and 99th percentile reconstruction errors from the training partition and $\epsilon = 10^{-8}$ prevents division by zero.

**Multimodal Agent.** The Multimodal Agent coordinates the workflow. At each inference cycle, it receives the predicted gas class and confidence from the Perception Agent and the anomaly score from the Anomaly Detection Agent. It constructs the 22-dimensional state $\phi_t$ for the Decision Agent:

$$\phi_t = [\tilde{\rho}_t, \mathbf{X}_t[-1,:], \delta_t, \sigma_t] \in \mathbb{R}^{22}$$

where $\tilde{\rho}_t$ is the normalized anomaly score, $\mathbf{X}_t[-1,:]$ is the current seven-channel sensor reading, $\delta_t$ is the per-sensor change across the window, and $\sigma_t$ is the per-sensor variability.

**Decision Agent.** The Decision Agent maps the monitoring evidence to an operator-facing safety action. It receives the 22-dimensional state vector $\phi_t$ constructed by the Multimodal Agent and selects one action from the five-level hierarchy in Table 3.

**Table 3.** Safety-action space and operational semantics.

| Action | Label | Operational response |
|--------|-------|----------------------|
| 0 | Monitor | Routine monitoring; no operator notification |
| 1 | Increase sampling | Elevated scan frequency and event logging |
| 2 | Request verification | Operator notification and secondary sensor check |
| 3 | Raise alarm | Automated alarm, incident logging, and field dispatch |
| 4 | Emergency shutdown | Full emergency protocol and immediate shutdown |

The benchmark provides fixed gas-class labels rather than an interactive process environment. Therefore, the Decision Agent is trained using offline benchmark replay rather than field reinforcement learning. During training, a sensor-window state is sampled from the training partition, a candidate safety action is evaluated with the asymmetric reward defined below, and the next sample is drawn from the fixed benchmark sequence. The selected action does not alter the next gas state or the future sensor trajectory.

*Network architecture.* The Decision Agent uses a Dueling-DQN-style action-score network for offline reward-shaped policy learning [8,9]. The term "DQN-style" is used here to describe the value–advantage network parameterization, not to imply an interactive process-control environment. The network maps the 22-dimensional monitoring state $\phi_t$ to five action scores, one for each safety action:

$$z(\phi_t, a) = V(\phi_t) + A(\phi_t, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(\phi_t, a')$$

where $z(\phi_t, a)$ is interpreted as an action score for offline safety-action selection. The selected response is:

$$a_t = \arg\max_{a \in \mathcal{A}} z(\phi_t, a)$$

This structure keeps the value–advantage decomposition used in the trained model while treating the outputs as action scores over fixed benchmark windows.

*Asymmetric safety reward.* Let $g_t$ denote the ground-truth class for training window $t$, let $a \in \mathcal{A}$ denote a candidate safety action, and let $\mathcal{G}_{\text{danger}} = \{\text{Smoke}, \text{Mixture}\}$. Because the benchmark provides fixed labeled windows rather than action-dependent process dynamics, the reward is used for offline reward-shaped policy learning rather than for field reinforcement learning.

The acceptable target-action set is defined as $\mathcal{A}^*(\text{NoGas}) = \{0\}$, $\mathcal{A}^*(\text{Smoke}) = \{3\}$, $\mathcal{A}^*(\text{Mixture}) = \{4\}$, and $\mathcal{A}^*(\text{Perfume}) = \{1, 2\}$, where the preferred Perfume action depends on anomaly severity. The total reward is:

$$r_t(g_t, a) = r_{\text{base}}(g_t, a) + r_{\text{danger}}(g_t, a) + r_{\text{alarm}}(g_t, a) + r_{\text{voc}}(g_t, a, \tilde{\rho}_t)$$

The base term rewards acceptable actions and uses the normalized anomaly score $\tilde{\rho}_t$ as a severity modifier for hazardous classes:

$$r_{\text{base}}(g_t, a) = \begin{cases} +2.0 + \tilde{\rho}_t, & a \in \mathcal{A}^*(g_t), g_t \in \mathcal{G}_{\text{danger}} \\ +2.0, & a \in \mathcal{A}^*(g_t), g_t \notin \mathcal{G}_{\text{danger}} \\ -2.0, & \text{otherwise} \end{cases}$$

A separate danger term penalizes unsafe non-escalation and unnecessary high-severity alarms:

$$r_{\text{danger}}(g_t, a) = \begin{cases} -10.0, & g_t \in \mathcal{G}_{\text{danger}}, a = 0 \\ -4.0, & g_t = \text{NoGas}, a \geq 3 \\ 0, & \text{otherwise} \end{cases}$$

Class-specific terms reinforce the intended high-severity responses for hazardous classes:

$$r_{\text{alarm}}(g_t, a) = \begin{cases} +1.0, & g_t = \text{Smoke}, a = 3 \\ +1.0, & g_t = \text{Mixture}, a = 4 \\ 0, & \text{otherwise} \end{cases}$$

and the lower-severity Perfume class receives a graded response based on anomaly severity:

$$r_{\text{voc}}(g_t, a, \tilde{\rho}_t) = \begin{cases} +0.8, & g_t = \text{Perfume}, \tilde{\rho}_t > 0.5, a = 2 \\ +0.5, & g_t = \text{Perfume}, \tilde{\rho}_t \leq 0.5, a = 1 \\ -0.5, & g_t = \text{Perfume}, \tilde{\rho}_t > 0.5, a = 1 \\ 0, & \text{otherwise} \end{cases}$$

All reward components are summed, scaled by 1/3, and clipped to [-2, +2] during training. This bounded reward preserves the main safety asymmetry while avoiding unstable score magnitudes during offline policy learning.

*Training configuration.* The Decision Agent was trained using offline benchmark replay on the 5,024-window training partition. Here an *episode* denotes one pass-equivalent block of benchmark-replay training steps rather than an environment rollout. At each training step, a fixed monitoring state $\phi_t$ was drawn from the benchmark replay sequence, an action was selected by an $\epsilon$-greedy policy, and the reward was computed from the ground-truth gas class and the selected action. The next state was another benchmark window rather than an action-dependent process state.

**Reasoning Agent.** The Reasoning Agent converts the selected action and decision context into a short operator-facing explanation. It receives the predicted gas class and confidence from the Perception Agent, the normalized anomaly score from the Anomaly Detection Agent, and the selected action and action-score vector from the Decision Agent. It then summarizes the evidence supporting the selected response.

Explanations are generated using Gemma 3 1B [14] served locally through Ollama. The structured prompt supplies the gas class, classification confidence, normalized anomaly score and threshold, and selected action, and requests a two- to three-sentence explanation of why the action is appropriate and what the operator should do next. Generation is limited to 150 tokens at temperature 0.7. Because the model is served locally, the explanation step does not require external API calls or transmission of monitoring data to a cloud service.

The explanation is advisory and is delivered off the safety-critical path; the safety action is fixed by the Decision Agent and is not altered by the generated text. This separation is deliberate, because a free-text model can produce an explanation that is fluent but inconsistent with the decision, which in a safety context would be a hazard rather than an aid.

**Multimodal Agent (coordination).** The Multimodal Agent coordinates the workflow. At each inference cycle, it receives the predicted gas class and confidence from the Perception Agent and the anomaly score from the Anomaly Detection Agent. It constructs the 22-dimensional state $\phi_t$ for the Decision Agent, forwards the selected action and decision context to the Reasoning Agent, and routes the final action and explanation to the dashboard. This coordination also allows the explanation step to be separated from the critical decision path, so that the safety action can be issued without waiting for natural-language generation.

### 3.4 Edge deployment setup

The edge deployment evaluates whether the ARXIS decision workflow can meet a practical sampling-interval budget on modest industrial-IoT hardware. The experiment is a benchmark replay on edge hardware, not a live field deployment. Its purpose is to characterize computational feasibility, model footprint, latency, and thermal behavior before live sensor integration.

The proof of concept uses a Raspberry Pi 4 Model B with a Cortex-A72 processor and 8 GB RAM, running 64-bit Raspberry Pi OS Bookworm. The runtime stack includes Python 3.11, PyTorch 2.3 CPU build, ONNX Runtime 1.18, and Ultralytics YOLOv8 8.2. The Reasoning Agent is served locally through Ollama using a 4-bit-quantized Gemma 3 1B model (Q4_K_M). The MultimodalGasData test partition is streamed from local storage at a 0.5 Hz cadence, corresponding to a 2,000 ms sampling interval. The deployable decision path must remain within this interval.

---

## 4. Results and Discussion

### 4.1 Experimental setup

All agents were evaluated on the 1,276-window block-wise holdout described in Section 3.1. Three system-level metrics are used throughout this section. Decision accuracy is the fraction of test windows assigned the target safety action. Danger miss rate is the fraction of hazardous windows (Smoke and Mixture combined, $n=640$) assigned the passive Monitor action ($a=0$)—the most severe under-response. False alarm rate is the fraction of NoGas windows ($n=316$) assigned a high-severity action ($a \geq 3$).

### 4.2 Perception Agent performance

The YOLOv8n thermal-image classifier achieved 98.8% top-1 accuracy on 1,280 held-out validation images, with an average inference latency of 0.78 ms per image. Table 4 reports the per-class precision, recall, F1-score, and accuracy.

**Table 4.** Perception Agent per-class classification performance on 1,280 held-out validation images.

| Gas Class | Precision | Recall | F1-Score | Accuracy (%) |
|-----------|-----------|--------|----------|--------------|
| Mixture | 1.000 | 1.000 | 1.000 | 100.0 |
| NoGas | 0.990 | 0.963 | 0.976 | 96.3 |
| Perfume | 0.972 | 0.991 | 0.981 | 99.1 |
| Smoke | 0.991 | 1.000 | 0.995 | 100.0 |
| **Overall** | **0.988** | **0.988** | **0.988** | **98.8** |

The classifier reached perfect recall for Smoke and Mixture, with no hazardous class predicted as NoGas or Perfume in the validation set. The 15 misclassifications occurred at the NoGas–Perfume boundary, where the safety consequence is lower because both classes map to non-emergency operating states in the ARXIS response hierarchy.

### 4.3 Anomaly Detection Agent performance

The LSTM autoencoder achieved an ROC-AUC of 0.963 on 6,380 sequenced windows. Table 5 reports the modality-level characterization.

**Table 5.** LSTM autoencoder modality-level characterization on 6,380 sequenced windows.

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.963 |
| Detection threshold $\tau$ | 0.00701 |
| True positive rate | 87.9% |
| False positive rate | 5.0% |
| True negatives / False positives | 1,501 / 79 |
| False negatives / True positives | 582 / 4,218 |
| Mean reconstruction error (normal) | $2.72 \times 10^{-3}$ |
| Mean reconstruction error (anomaly) | 2.082 |
| Separation factor | 764× |

The operating point at $\tau = 0.00701$ corresponds to TPR = 87.9% and FPR = 5.0%. The 582 false negatives arise primarily from lower-intensity Perfume windows.

### 4.4 Decision Agent training and evaluation

*Training dynamics.* The Decision Agent was trained for 200 offline benchmark-replay episodes. Cumulative replay reward increased from approximately -2,843 during early exploration to about +4,336 after convergence. The exploration rate $\epsilon$ reached its minimum value of 0.05 at episode 150. The 12-episode rolling danger miss rate decreased from 0.214 to $5.6 \times 10^{-3}$ near convergence, while decision accuracy increased from 24.2% to 96.5%. The Huber loss also declined smoothly, reaching 0.0342 by the final episode.

*Test-set performance.* On the 1,276-window block-wise holdout, the Decision Agent achieved 94.44% decision accuracy. More importantly for the safety-action formulation, no hazardous Smoke or Mixture window was assigned the passive Monitor action. The danger miss rate was therefore zero across all 640 hazardous windows. The under-escalation rate was also zero: every Smoke window received Raise Alarm and every Mixture window received Emergency Shutdown, so no hazardous window was assigned any action below its target severity tier. The high-severity false alarm rate was zero across the 316 NoGas windows: no baseline window received Raise Alarm or Emergency Shutdown ($a \geq 3$).

Table 6 reports the class-level performance.

**Table 6.** Decision Agent per-class performance on the 1,276-window block-wise holdout test set.

| Gas Class | Precision | Recall | F1-Score | $n$ |
|-----------|-----------|--------|----------|-----|
| NoGas | 0.98 | 0.79 | 0.88 | 316 |
| Smoke | 1.00 | 1.00 | 1.00 | 320 |
| Mixture | 1.00 | 1.00 | 1.00 | 320 |
| Perfume | 0.83 | 0.99 | 0.90 | 320 |

**Overall decision accuracy: 94.44%**
**Danger miss rate (640 hazardous windows): 0.0000**
**False alarm rate (NoGas windows, $a \geq 3$): 0.0000**

Smoke and Mixture reached perfect precision, recall, and F1-score, indicating that the highest-risk classes received their target escalation actions without exception. The lower clean-set accuracy is therefore not a failure of convergence; it reflects the intended bias toward precautionary low-severity actions when baseline and VOC-like evidence overlap.

*Action behavior.* The policy assigned Raise Alarm ($a=3$) to all Smoke windows and Emergency Shutdown ($a=4$) to all Mixture windows. Most Perfume windows were assigned Request Verification ($a=2$), while most NoGas windows were assigned Monitor ($a=0$). The remaining NoGas windows received only low-severity responses, not high-severity alarms. The policy entropy was 1.502 nats, below the maximum of $\ln 5 \approx 1.609$ for a uniform five-action policy and well above zero for a degenerate single-action policy. Mean Decision Agent inference latency was 1.07 ms per window on the desktop CPU reference.

### 4.5 Baseline model comparison (ZOO)

We trained 8 models on the same corpus with the same 5 seeds and evaluation protocol. Table 7 summarizes the comparison.

**Table 7.** Baseline model comparison on the same corpus, same 5 seeds, same evaluation.

| Model | Accuracy | Std | Miss Rate |
|-------|----------|-----|-----------|
| **Decision Agent (ARXIS)** | **0.9633** | **0.0215** | **0.0** |
| Plain DQN | 0.9563 | 0.0378 | 0.0 |
| MLP | 0.9616 | 0.0111 | 0.0 |
| GBM | 0.9634 | 0.0211 | 0.0098 |
| SVM | 0.9511 | 0.0190 | 0.0 |
| Random Forest | 0.9769 | 0.0086 | 0.0 |
| LSTM | 0.9680 | 0.0231 | 0.0 |
| CQL | 0.9437 | 0.0479 | 0.0 |

**No comparator significantly different from ARXIS at α=0.05.**

### 4.6 Leave-One-Class-Out (LOCO) results

The headline experiment: train on 3 gases, test on the 4th. Table 8 summarizes the results.

**Table 8.** Leave-One-Class-Out (LOCO) results: train on 3 gases, test on the 4th.

| Held-Out | Model | Miss Rate | 95% CP Bound |
|----------|-------|-----------|--------------|
| Smoke | Decision Agent | 0.0% | ≤0.19% |
| Smoke | **MLP** | **18.7%** | **≤20.4%** |
| Smoke | Plain DQN | 0.0% | ≤0.19% |
| Smoke | GBM | 0.0% | ≤0.19% |
| Mixture | Decision Agent | 0.44% | ≤0.83% |
| Mixture | Plain DQN | 0.0% | ≤0.19% |
| Mixture | MLP | 0.38% | ≤0.75% |
| Mixture | GBM | 0.0% | ≤0.19% |

**Key finding:** In-distribution accuracy does NOT predict out-of-distribution safety.

### 4.7 Robustness analysis

Under calibration drift and sensor-channel dropout, ARXIS preserves zero danger misses while supervised baselines degrade. Table 9 summarizes the comparison.

**Table 9.** Clean and perturbed test-set comparison between the ARXIS policy and supervised decision baselines.

| Configuration | Model | Decision Acc. | Danger Miss Rate |
|---------------|-------|---------------|------------------|
| Clean test set | Rule-oracle | 100.00% | 0.0000 |
| Clean test set | Supervised MLP | 98.20% | 0.0000 |
| Clean test set | ARXIS policy | 94.44% | 0.0000 |
| Gaussian noise, σ=0.20 | Supervised MLP | 89.7% | 0.0000 |
| Gaussian noise, σ=0.20 | ARXIS policy | 73.2% | 0.0000 |
| Gaussian noise, σ=0.40 | Supervised MLP | 69.3% | 0.0000 |
| Gaussian noise, σ=0.40 | ARXIS policy | 58.0% | 0.0000 |
| Calibration drift, ±40% | Supervised MLP | 59.1% | **0.0187** |
| Calibration drift, ±40% | ARXIS policy | 53.7% | **0.0000** |
| Channel dropout, 1 sensor | Supervised MLP | 36.0% | 0.0000 |
| Channel dropout, 1 sensor | ARXIS policy | 55.6% | 0.0000 |
| Channel dropout, 3 sensors | Supervised MLP | 28.9% | 0.0000 |
| Channel dropout, 3 sensors | ARXIS policy | 29.9% | 0.0000 |

The perturbation results show why decision accuracy and safety behavior must be reported separately. On the clean test set, the supervised MLP achieved higher decision accuracy than the ARXIS policy because it was trained directly on deterministic action labels. Under Gaussian noise, the MLP also retained higher decision accuracy, while both models preserved zero danger misses. The clearest safety distinction appeared under calibration drift: the MLP remained more accurate on average, but introduced a nonzero danger miss rate of 0.0187, whereas the ARXIS policy preserved zero danger misses.

### 4.8 Confidence calibration

Table 10 reports the Expected Calibration Error for 4 estimators.

**Table 10.** Confidence calibration: Expected Calibration Error (ECE) for 4 estimators.

| Estimator | ECE Mean | ECE Std |
|-----------|----------|---------|
| Raw softmax | 0.0489 | 0.0699 |
| MC dropout (20 samples) | 0.0533 | 0.0692 |
| Temp scaling | 0.0319 | 0.0397 |
| **Deep ensemble (5 members)** | **0.0123** | **0.0043** |

### 4.9 Edge deployment proof of concept

Table 11 summarizes the latency, footprint, and safety-decision metrics.

**Table 11.** Edge deployment proof of concept on a Raspberry Pi 4.

| Configuration | Model Size | Mean Latency | P99 Latency | Decision Acc. | Danger Miss Rate | False Alarm Rate |
|---------------|------------|--------------|-------------|---------------|------------------|------------------|
| Desktop CPU, float32 | 8.74 MB | 5.8 ms | 9.6 ms | 94.44% | 0.0000 | 0.0000 |
| Pi 4, float32 (synchronous) | 8.74 MB | 14,738 ms | 24,202 ms | 94.36% | 0.0000 | 0.0000 |
| Pi 4, INT8 (synchronous) | 4.19 MB | 14,675 ms | 24,028 ms | 93.97% | 0.0000 | 0.0032 |
| **Pi 4, INT8 + async reasoning** | **4.19 MB** | **235 ms** | **413 ms** | **93.97%** | **0.0000** | **0.0032** |

Three findings are most relevant. First, the danger miss rate remained zero on the replayed test set across all hardware and precision configurations. Second, decoupling explanation generation from safety-action selection reduced the decision-path latency below the sampling budget. Third, passive cooling was sufficient for the 36-minute run, but thermal throttling reduced the latency margin during sustained operation.

### 4.10 Comparison with prior work

Table 12 compares ARXIS with representative studies evaluated on the MultimodalGasData benchmark.

**Table 12.** Comparison with prior work on the MultimodalGasData benchmark.

| Study | Method | Accuracy | Limitation relevant to safety monitoring |
|-------|--------|----------|------------------------------------------|
| Narkhede et al. [20] | CNN with multimodal fusion | 97.0% | No mapping from class label to operator action |
| Faleh and Kachouri [23] | Hybrid CNN–LDA classifier | 93.0% | Single modality; no anomaly or decision layer |
| El Barkani et al. [24] | TinyML CNN | 91.7% | Classification only; no anomaly scoring |
| Sharma et al. [25] | Federated deep CNN ensemble | 99.7% | No graded response policy |
| Zhang and Zhang [26] | Transformer classifier | 99.7% | No safety-relevant evaluation |
| **ARXIS (this work)** | **Offline reward-shaped safety-action policy** | **94.44%** | **Benchmark proof of concept; field validation pending** |

The lower headline accuracy of ARXIS in Table 12 should be interpreted with caution. The strongest classifiers in the table optimize class-label prediction, while ARXIS optimizes action selection under an asymmetric safety objective. A classification error and an action error do not necessarily have the same process-safety meaning.

---

## 5. Discussion

### 5.1 Key findings

**Safety-action selection is not gas classification.** The 94.44% decision accuracy of ARXIS is lower than the 98.8% classification accuracy of the Perception Agent and the 97.0-99.7% reported by prior classification-focused studies. However, these numbers measure different quantities. A classification error and an action error do not necessarily have the same process-safety meaning. Confusing Smoke and Mixture may still preserve the need for escalation, whereas assigning a hazardous window to Monitor removes the opportunity for timely intervention.

**Zero danger misses under distribution shift.** The most important result is not the clean-test accuracy but the behavior under perturbation. Under calibration drift, the supervised MLP retained higher decision accuracy (59.1%) than ARXIS (53.7%) but introduced a nonzero danger miss rate of 0.0187, whereas ARXIS preserved zero danger misses. This difference shows that the distribution of errors matters in safety-critical monitoring.

**Temporal features drive safety behavior.** The decision-state ablation shows that removing the anomaly score reduced accuracy by 2.75 percentage points, while removing temporal change and standard-deviation features reduced accuracy by 10.27 points. The full 22-dimensional state achieves the highest decision accuracy while preserving zero danger misses.

### 5.2 Deployment tradeoffs

The edge deployment results define practical deployment tradeoffs:

1. **Explanation must be asynchronous.** Synchronous local explanation with Gemma 3 1B produced a mean end-to-end latency of 14,738 ms, far exceeding the 2,000 ms sampling budget. Decoupling explanation from the decision path reduced latency to 503 ms; INT8 quantization further reduced it to 235 ms.

2. **Quantization affects action boundaries.** INT8 quantization reduced model size from 8.74 MB to 4.19 MB but introduced a false alarm rate of 0.0032 (one high-severity action across 316 NoGas windows). If zero high-severity false alarms is required, the decision network can remain in floating point while the larger perception model is quantized.

3. **Thermal management matters.** During a 36-minute passive-cooled run, the CPU reached the 80°C throttling threshold after about 24 minutes, causing clock frequency to drop from 1,500 MHz to 1,234 MHz. Median latency rose from 420 ms to 640 ms during throttling, but the decision path remained within the 2,000 ms sampling budget.

### 5.3 Baseline comparison summary

The comprehensive baseline comparison across 8 models demonstrates that:

- **No baseline significantly outperforms ARXIS** at α=0.05 on the clean test set.
- **ARXIS uniquely preserves zero danger misses** under calibration drift and sensor dropout.
- **The MLP is the weakest baseline**, with 18.7% danger miss rate on held-out Smoke and 0.38% on held-out Mixture.
- **Random Forest achieves the highest raw accuracy** (0.9769) but with higher variance than ARXIS.
- **CQL (the only genuine safe-RL baseline) underperforms** cost-weighted CE on this separable dataset.

---

## 6. Limitations

Three limitations define how the results should be interpreted:

1. **Dataset scope.** ARXIS was developed and evaluated on the MultimodalGasData benchmark under controlled exposure conditions. Wind effects, ambient-temperature changes, sensor aging, long-term drift, hydraulic transients, leak-rate diversity, and background operational noise are not represented.

2. **Decision-state scope.** The current Decision Agent uses a 22-dimensional state composed of sensor dynamics and anomaly evidence. Thermal-image information is used by the Perception Agent and Reasoning Agent but is not included directly in the DQN policy state. A fully fused policy that incorporates visual class probabilities is an important extension.

3. **Deployment scope.** The edge proof of concept replays the benchmark test partition from local storage on a Raspberry Pi 4. It does not evaluate live sensor acquisition, long-duration field operation, integration with plant alarm-management systems, or operator response to the generated explanations.

---

## 7. Conclusions

This study presented ARXIS, an edge-deployable agentic AI framework that reframes gas-hazard monitoring as safety-action selection rather than gas classification alone. On the MultimodalGasData benchmark, ARXIS achieved 94.44% decision accuracy while preserving zero danger misses across 640 hazardous windows and zero high-severity false alarms across 316 baseline windows. Under calibration drift, a supervised baseline retained higher decision accuracy but introduced hazardous misses, whereas ARXIS preserved zero danger misses. The edge proof of concept showed that asynchronous reasoning and model quantization reduced the deployable decision path to 235 ms on a Raspberry Pi 4.

The comprehensive baseline comparison across 8 models demonstrates that no baseline significantly outperforms ARXIS at α=0.05, while ARXIS uniquely preserves zero danger misses under distribution shift. These results show that missed-hazard behavior, false-alarm behavior, degradation robustness, and response latency should be reported alongside accuracy.

Future work should evaluate ARXIS on live field streams, incorporate visual evidence directly into the decision policy, and assess the operator-facing explanations in realistic alarm and response scenarios.

---

## CRediT authorship contribution statement

**Benjamin C. Nweke:** Conceptualization, Methodology, Software, Investigation, Formal analysis, Data curation, Visualization, Writing – original draft.
**Gholamreza Ramezan:** Conceptualization, Methodology, Writing – review & editing.
**Soheil Saraji:** Conceptualization, Supervision, Project administration, Resources, Writing – review & editing.

---

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

---

## Acknowledgements

This research was supported by the Subsurface Energy and Digital Innovation Center (SEDI) at the University of Wyoming.

---

## References

[1] Laberge, J.C., Bullemer, P., Tolsma, M., Reising, D.V., 2014. Addressing alarm flood situations in the process industries through alarm summary display design and alarm response strategy. International Journal of Industrial Ergonomics 44, 395–406.

[2] Goel, P., Datta, A., Mannan, M.S., 2017. Industrial alarm systems: Challenges and opportunities. Journal of Loss Prevention in the Process Industries 50, 23–36.

[3] Murvay, P.S., Silea, I., 2012. A survey on gas leak detection and localization techniques. Journal of Loss Prevention in the Process Industries 25, 966–973.

[4] Han, P., Kim, M., 2014. Optimizing leak detection performance. PSIG Annual Meeting, PSIG 1407.

[5] Malhotra, P., Ramakrishnan, A., Anand, G., Vig, L., Agarwal, P., Shroff, G., 2016. LSTM-based encoder-decoder for multi-sensor anomaly detection. arXiv preprint arXiv:1607.00148.

[6] Liang, J., Liang, S., Zhang, H., Zuo, Z., Ma, L., Dai, J., 2024. Leak detection in natural gas pipelines based on unsupervised reconstruction of healthy flow data. SPE Journal.

[7] Korjani, M., Conley, D., Smith, M., 2024. Temporal deep learning image processing model for natural gas leak detection using OGI camera. OTC Asia, OTC-34756-MS.

[8] Mnih, V., et al., 2015. Human-level control through deep reinforcement learning. Nature 518, 529–533.

[9] Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., de Freitas, N., 2016. Dueling network architectures for deep reinforcement learning. ICML, 1995–2003.

[10] Sabbagh, V.B., Lima, C.B.C., Xexéo, G., 2024. Comparative analysis of single and multiagent large language model architectures. SPE Journal 29, 6869–6884.

[11] Santiago, C.J.S., Shumaker, N., Weir, A., 2025. Integrating data-driven insights with domain expertise using agentic conversational analytics. SPE Annual Technical Conference, SPE-228143-MS.

[12] Jessen, H., Roshchin, M., 2025. Agentic AI revolution across O&G value streams. ADIPEC, SPE-229240-MS.

[13] Dehnaw, A.M., et al., 2024. Deep neural network optimization for efficient gas detection systems in edge intelligence environments. Processes 12, 2638.

[14] Google DeepMind, 2024. Gemma: Open models based on Gemini research and technology. https://ai.google.dev/gemma.

[15] Adegboye, M.A., Fung, W.K., Karnik, A., 2019. Recent advances in pipeline monitoring and oil leakage detection technologies: Principles and approaches. Sensors 19, 2548.

[16] Zhang, J., Hoffman, A., Murphy, K., Lewis, J., Twomey, M., 2013. Review of pipeline leak detection technologies. PSIG Annual Meeting, PSIG 1303.

[17] Bustnes, T.E., Rousselet, M., Berland, S., 2011. Leak detection performance of a commercial real-time transient model. PSIG Annual Meeting, PSIG 1114.

[18] Wang, C.Y., Bochkovskiy, A., Liao, H.Y.M., 2022. YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors. arXiv preprint arXiv:2207.02696.

[19] Narkhede, P., Walambe, R., Chandel, P., Mandaokar, S., Kotecha, K., Ghinea, G., 2021. Gas detection and identification using multimodal artificial intelligence based sensor fusion. Applied System Innovation 4, 3.

[20] Narkhede, P., Walambe, R., Mandaokar, S., Chandel, P., Kotecha, K., Ghinea, G., 2022. MultimodalGasData: Multimodal dataset for gas detection and classification. Data 7, 112.

[21] Ma, R., et al., 2026. Minor pipeline leak detection and localization using explainable deep learning with fusion of distributed fiber-optic vibration and temperature signals. Journal of Loss Prevention in the Process Industries 100, 105844.

[22] Jocher, G., Chaurasia, A., Qiu, J., 2023. YOLOv8 by Ultralytics. https://github.com/ultralytics/ultralytics.

[23] Faleh, R., Kachouri, A., 2023. A hybrid deep convolutional neural network-based electronic nose for pollution detection. Chemometrics and Intelligent Laboratory Systems 237, 104825.

[24] El Barkani, M., Benamar, N., Talei, H., Bagaa, M., 2024. Gas leakage detection using tiny machine learning. Electronics 13, 4768.

[25] Sharma, A., et al., 2024. Gas detection and classification using multimodal data based on federated learning. Sensors 24, 5904.

[26] Zhang, E., Zhang, E., 2025. Gas pipeline leakage detection based on multiple multimodal deep feature selections and optimized deep forest classifier. Frontiers in Environmental Science 13, 1569621.

---

*Preprint submitted to Elsevier*
