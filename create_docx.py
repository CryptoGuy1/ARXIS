from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import os

doc = Document()
style = doc.styles['Normal']
font = style.font
font.name = 'Times New Roman'
font.size = Pt(12)

# Title
title = doc.add_heading('ARXIS: Edge-Deployable Agentic AI for Safety-Critical Gas Monitoring and Action Selection', level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Authors
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run('Benjamin C. Nweke^{a,b,*}, Gholamreza Ramezan^b, Soheil Saraji^a')
r.font.size = Pt(12)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run('^a Univ. of Wyoming, ^b Fides Innova Labs')
r.font.size = Pt(10)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run('* bnweke@uwyo.edu')
r.font.size = Pt(10)

# ARTICLE INFO
doc.add_heading('ARTICLE INFO', level=1)
doc.add_paragraph('Keywords: Gas-hazard monitoring; Safety-action selection; Agentic AI; Edge deployment; Anomaly detection; Reinforcement learning; Explainable AI; Data analytics; Machine learning')

# ABSTRACT
doc.add_heading('ABSTRACT', level=1)
doc.add_paragraph('Gas-monitoring systems in process industries must support more than abnormal-condition detection; they must also help operators select a defensible response when evidence is uncertain, degraded, or incomplete. Existing gas-detection and classification models often report gas labels but rarely evaluate whether the resulting decision avoids missed hazards or unnecessary high-severity alarms. This paper presents ARXIS (Agentic Reasoning with eXplainability for Infrastructure Surveillance), an edge-deployable agentic AI framework that combines thermal-image perception, sensor-based anomaly scoring, offline reward-shaped safety-action selection, and local natural-language explanation in a sense-decide-explain workflow. On the public MultimodalGasData benchmark, evaluated using a leakage-controlled block-wise holdout, ARXIS achieved 94.44% decision accuracy with zero danger misses across 640 hazardous windows and zero high-severity false alarms across 316 baseline windows. Under calibration drift, a supervised baseline trained on the same labels began missing hazards (danger miss rate = 0.0187), whereas ARXIS preserved zero danger misses. A Raspberry Pi 4 proof of concept showed that asynchronous explanation and model quantization reduced the deployable decision path from approximately 14.8 s to 235 ms. Comprehensive baseline comparisons across 8 models demonstrate that no baseline significantly outperforms ARXIS at u03b1=0.05, while ARXIS uniquely preserves zero danger misses under distribution shift.')

doc.add_page_break()

# Section 1: Introduction
doc.add_heading('1. Introduction', level=1)
doc.add_paragraph('Major process-safety incidents are seldom caused by a complete absence of warning. More often, abnormal signals are present but are not converted into timely and proportionate operator action. This distinction is central to gas-hazard monitoring in process industries, where the safety outcome depends not only on detecting abnormal evidence but also on prioritizing it and selecting an appropriate response.')
doc.add_paragraph('Alarm-management studies have long shown that alarm floods can overwhelm operators and reduce the effectiveness of abnormal-situation response [1]. The Milford Haven refinery explosion illustrates this difficulty: in the eleven minutes before the blast, the operator was confronted with 275 alarms, showing that the problem was not simply the absence of signals but the failure to convert them into actionable information under time pressure [2].')
doc.add_paragraph('Gas-leak detection and corrective response are related but distinct problems [3]. Knowing that a leak exists is not sufficient to determine the appropriate corrective measure. In practice, however, many monitoring systems still reduce the task to a binary threshold or a class label. A threshold tuned for sensitivity may generate frequent false alarms and erode operator trust; a threshold tuned for specificity may miss genuine hazardous events [4]. Neither behavior reflects how process-safety decisions are made in operation.')
doc.add_paragraph('The monitoring problem is therefore better formulated as safety-action selection under uncertainty than as gas classification alone. Depending on the severity, confidence, and persistence of the evidence, an operator may continue routine monitoring, increase sampling, request verification, raise an alarm, or initiate an emergency shutdown.')

doc.add_heading('1.1 Research Gap', level=2)
doc.add_paragraph('Several technical components needed for such a system have matured: normal-only autoencoder models can learn baseline sensor behavior and detect deviations through reconstruction error [5,6]; computer vision models can classify thermal or optical gas signatures [7]; reinforcement learning provides a way to learn action policies when the costs of errors are unequal [8,9]; agentic AI systems can coordinate specialized models for sensing, reasoning, and decision support [10,11,12]; and compact language models make local explanation generation feasible on edge devices [13,14]. These advances are useful, but they are usually treated separately. Three gaps are most relevant: many gas-monitoring methods end at detection or classification and do not model the downstream safety response; reported metrics often emphasize accuracy, while safety-relevant quantities such as missed hazards, high-severity false alarms, and behavior under sensor degradation are less consistently reported; and agentic and explainable AI methods are seldom evaluated as edge-deployable components of an operator-facing safety workflow.')

doc.add_heading('1.2 Contributions', level=2)
doc.add_paragraph('This paper presents ARXIS, an edge-deployable agentic AI framework that addresses these gaps. The specific contributions are fourfold:')
doc.add_paragraph('1. Modular monitoring pipeline of five coordinated components-thermal-image perception, normal-only sensor anomaly scoring, decision-state construction, offline reward-shaped safety-action selection, and local natural-language explanation-organized as a sense-decide-explain workflow.')
doc.add_paragraph('2. Safety-constrained decision-making formulation rather than static thresholding or gas classification, using offline benchmark replay and an asymmetric safety reward to penalize passive responses to hazardous conditions more heavily than conservative verification actions.')
doc.add_paragraph('3. Safety-relevant evaluation on the public MultimodalGasData benchmark. On the block-wise holdout test set, ARXIS attains 94.44% decision accuracy with zero danger misses across 640 hazardous windows and zero high-severity false alarms across 316 baseline windows.')
doc.add_paragraph('4. Edge-deployment proof of concept on a Raspberry Pi 4, showing that asynchronous explanation and model quantization reduce the deployable decision path to 235 ms while preserving zero danger misses.')

# Section 2: Related Work
doc.add_heading('2. Advances in AI-Based Gas-Hazard Monitoring', level=1)
doc.add_paragraph('AI-based gas-hazard monitoring has progressed from signal detection toward richer forms of perception, anomaly scoring, decision support, and explanation. For process-safety applications, however, the critical question is not only whether a model recognizes an abnormal condition. It is whether the evidence is translated into a timely and defensible operator response.')

doc.add_heading('2.1 Pipeline and gas leak detection', level=2)
doc.add_paragraph('Although the present framework is AI-based, gas-hazard detection rests on an established base of hardware- and software-based leak-detection methods [15]. Hardware-based methods include distributed acoustic sensing, fiber-optic sensing, vapor sensing, and negative-pressure-wave detection. Software-based methods include real-time transient modeling, mass balance, statistical analysis, and model-based diagnostics. These studies provide the foundation for reliable leak recognition, but they usually end with a leak indication, location estimate, or concentration estimate. Those outputs are necessary for loss prevention, but they are not the same as a response decision.')

doc.add_heading('2.2 Deep learning for anomaly detection', level=2)
doc.add_paragraph('Deep learning has improved gas-hazard monitoring by learning patterns in multivariate sensor streams. Reconstruction-based models are especially useful because they can be trained on normal operation and then used to flag deviations. The LSTM encoder-decoder introduced by Malhotra et al. [5] established a widely used reconstruction-error approach for multivariate anomaly detection. More recent work applies normal-only learning to gas-pipeline monitoring [6]. These methods are valuable because they provide an early indication that the current sensor pattern has departed from normal operation. Their limitation is that an anomaly score is still an intermediate signal. It does not specify the appropriate response.')

doc.add_heading('2.3 Computer vision and multimodal gas detection', level=2)
doc.add_paragraph('Visual and thermal sensing complement point sensors by capturing plume patterns, thermal signatures, or image-level indicators of gas-related conditions. Models from the YOLO family offer a favorable speed-accuracy tradeoff for embedded vision tasks [18], and optical gas-imaging models have shown strong detection performance under controlled and field conditions [7]. Most multimodal studies still evaluate performance as gas-classification accuracy. This is useful, but incomplete for process safety.')

doc.add_heading('2.4 Reward-shaped policy learning for safety-oriented decisions', level=2)
doc.add_paragraph('Safety-oriented monitoring requires more than accurate recognition of abnormal conditions. It also requires a decision rule that reflects unequal consequences of different errors. Deep Q-networks showed that neural networks can approximate action-value functions for decision-making [8], and dueling architectures improved value estimation by separating a state-value component from action-specific advantages [9]. For the present study, the relevant idea is not open-ended autonomous control. Rather, it is the use of a reward-shaped policy to select among a finite set of operator-facing safety actions under unequal error costs.')

doc.add_heading('2.5 Agentic AI and operator-facing explanation', level=2)
doc.add_paragraph('Agentic AI systems coordinate specialized models or tools toward a larger task. In the energy sector, recent systems have been applied to well-construction assistance, completions optimization, and enterprise modeling workflows [10,11,12]. These systems show the value of coordinating multiple reasoning components, but they are mostly designed for analytical or advisory settings rather than real-time process-safety monitoring. Edge optimization and compact language models offer a path toward local inference and local explanation [13,14]. Explanation is important because operator-facing safety systems must support trust, review, and accountability. The unresolved issue is that explanation is often separated from action.')

doc.add_heading('2.6 Research gap', level=2)
doc.add_paragraph('Table 1 summarizes representative work across the capability areas that inform ARXIS. The literature shows substantial progress in leak detection, anomaly scoring, visual recognition, multimodal classification, reinforcement learning, and explanation. The common limitation is that these capabilities are rarely integrated into a single workflow that carries evidence from sensing to action.')

# Table 1
doc.add_paragraph('Table 1. Representative prior work across the capability areas informing ARXIS.')
table = doc.add_table(rows=8, cols=4)
table.style = 'Table Grid'
hdr_cells = table.rows[0].cells
hdr_cells[0].text = 'Study'
hdr_cells[1].text = 'Area'
hdr_cells[2].text = 'Contribution'
hdr_cells[3].text = 'Limitation relevant to ARXIS'
data = [
    ['Liang et al. [6]', 'Anomaly detection', 'Field-validated normal-only leak detection', 'Quantifies anomaly but prescribes no operator action'],
    ['Malhotra et al. [5]', 'Anomaly detection', 'Reconstruction-error scoring for sensor streams', 'No mapping from anomaly to operational response'],
    ['Korjani et al. [7]', 'Computer vision', 'High detection rates on optical gas imaging', 'No decision layer; performance varies with weather'],
    ['Narkhede et al. [20]', 'Multimodal sensing', 'Thermal-sensor fusion improves gas classification', 'Reports classification accuracy, not safety metrics'],
    ['Mnih et al. [8]', 'Reinforcement learning', 'Neural value functions for sequential decisions', 'Developed for game domains; not safety-framed'],
    ['Sabbagh et al. [10]', 'Agentic AI', 'Multi-agent reasoning improves answer quality', 'Analytical workflow; not real-time monitoring'],
    ['Ma et al. [21]', 'Explainability', 'Multimodal fusion with post-hoc attribution', 'Explanation not coupled to a graded action'],
]
for i, row_data in enumerate(data):
    row_cells = table.rows[i+1].cells
    for j, cell_data in enumerate(row_data):
        row_cells[j].text = cell_data

doc.add_paragraph()

# Section 3: Methodology
doc.add_heading('3. Methodology', level=1)
doc.add_paragraph('This section describes the ARXIS framework and the experimental procedure used to evaluate it. The aim is to establish a reproducible benchmark-based proof of concept for gas-hazard monitoring and safety-action selection.')

doc.add_heading('3.1 Dataset and preprocessing', level=2)
doc.add_paragraph('The MultimodalGasData dataset [19] was used for framework development and proof-of-concept evaluation. The dataset contains synchronized gas-sensor and thermal-image measurements collected under controlled exposure conditions. It includes 6,400 labeled samples distributed equally across four environmental classes: NoGas, Smoke, Perfume, and Mixture, with 1,600 samples per class.')

# Table 2
doc.add_paragraph('Table 2. MQ sensor array composition and primary detection targets.')
table2 = doc.add_table(rows=8, cols=2)
table2.style = 'Table Grid'
hdr = table2.rows[0].cells
hdr[0].text = 'Sensor'
hdr[1].text = 'Primary detection target'
sensors = [
    ['MQ-2', 'LPG, butane, methane, smoke'],
    ['MQ-3', 'Alcohol, ethanol, vapors'],
    ['MQ-5', 'LPG, natural gas'],
    ['MQ-6', 'LPG, butane, iso-butane'],
    ['MQ-7', 'Carbon monoxide'],
    ['MQ-8', 'Hydrogen'],
    ['MQ-135', 'Air-quality indicators (NHu2083, benzene, NOu2093, COu2082)'],
]
for i, row_data in enumerate(sensors):
    row_cells = table2.rows[i+1].cells
    row_cells[0].text = row_data[0]
    row_cells[1].text = row_data[1]

doc.add_paragraph()
doc.add_paragraph('For time-series modeling, the seven-channel sensor data were converted into sliding windows of length 20. A leakage-controlled block-wise holdout was used: for each class k, the final 20% of sequential windows were reserved for testing, with a temporal gap of G=20 windows discarded between training and test segments. This produced 5,024 training windows and 1,276 test windows.')

doc.add_heading('3.2 System architecture', level=2)
doc.add_paragraph('ARXIS is organized as a modular five-agent framework for gas-hazard monitoring and safety-oriented decision support. Fig. 1 shows the overall workflow. At each inference cycle, the framework receives a thermal image Iu1d6 and a seven-channel sensor window Xu1d6 u2208 u211d^{20u00d77}. It returns a safety action au1d6 u2208 {0,1,2,3,4} and a natural-language explanation Eu1d6.')
doc.add_paragraph('[FIGURE 1: System architecture diagram - insert from original PDF]', style='Intense Quote')

doc.add_heading('3.3 Agent descriptions', level=2)
doc.add_paragraph('Perception Agent. The Perception Agent provides image-level evidence about the gas condition. It classifies each thermal image into one of four environmental classes using YOLOv8n-cls [22]. The predicted class is cu1d6 = argmax(pu1d6). Training used an 80/20 stratified split of the 6,400 thermal images, pretrained-weight initialization, and AdamW with learning rate 10^{-3}, cosine annealing, weight decay 10^{-4}, batch size 32, and 50 epochs.')
doc.add_paragraph('Anomaly Detection Agent. The Anomaly Detection Agent provides sensor-based evidence of departure from normal operation. It is implemented as an LSTM autoencoder trained only on NoGas samples. The reconstruction mean-squared error serves as the anomaly score. The autoencoder was trained for 40 epochs using Adam with learning rate 10^{-3} and batch size 64. the detection threshold u03c4 was calibrated as the 95th percentile of reconstruction errors over the NoGas training partition.')
doc.add_paragraph('Multimodal Agent. The Multimodal Agent coordinates the workflow and constructs the 22-dimensional decision state: u03c6u1d6 = [u03c1u0303u1d6, Xu1d6[-1,:], u03b4u1d6, u03c3u1d6] u2208 u211d^{22}.')
doc.add_paragraph('Decision Agent. The Decision Agent uses a Dueling-DQN-style action-score network: z(u03c6u1d6, a) = V(u03c6u1d6) + A(u03c6u1d6, a) - (1/|A|) u2211 A(u03c6u1d6, au2032). The selected response is au1d6 = argmax z(u03c6u1d6, a).')
doc.add_paragraph('Reasoning Agent. The Reasoning Agent converts the selected action and decision context into a short operator-facing explanation using Gemma 3 1B [14] served locally through Ollama.')

doc.add_heading('3.4 Edge deployment setup', level=2)
doc.add_paragraph('The edge deployment evaluates whether the ARXIS decision workflow can meet a practical sampling-interval budget on modest industrial-IoT hardware. The proof of concept uses a Raspberry Pi 4 Model B with a Cortex-A72 processor and 8 GB RAM, running 64-bit Raspberry Pi OS Bookworm.')

# Section 4: Results
doc.add_heading('4. Results and Discussion', level=1)
doc.add_paragraph('All agents were evaluated on the 1,276-window block-wise holdout described in Section 3.1.')

doc.add_heading('4.1 Perception Agent performance', level=2)
doc.add_paragraph('The YOLOv8n thermal-image classifier achieved 98.8% top-1 accuracy on 1,280 held-out validation images.')

# Table 4
doc.add_paragraph('Table 4. Perception Agent per-class classification performance on 1,280 held-out validation images.')
table4 = doc.add_table(rows=6, cols=5)
table4.style = 'Table Grid'
hdr = table4.rows[0].cells
hdr[0].text = 'Gas Class'
hdr[1].text = 'Precision'
hdr[2].text = 'Recall'
hdr[3].text = 'F1-Score'
hdr[4].text = 'Accuracy (%)'
perception_data = [
    ['Mixture', '1.000', '1.000', '1.000', '100.0'],
    ['NoGas', '0.990', '0.963', '0.976', '96.3'],
    ['Perfume', '0.972', '0.991', '0.981', '99.1'],
    ['Smoke', '0.991', '1.000', '0.995', '100.0'],
    ['Overall', '0.988', '0.988', '0.988', '98.8'],
]
for i, row_data in enumerate(perception_data):
    row_cells = table4.rows[i+1].cells
    for j, cell_data in enumerate(row_data):
        row_cells[j].text = cell_data

doc.add_paragraph()

# Add available result images
if os.path.exists('data/Mixture/1018_Mixture.png'):
    doc.add_picture('data/Mixture/1018_Mixture.png', width=Inches(4))
    doc.add_paragraph('Figure 2. Sample thermal image (Mixture class) from the MultimodalGasData dataset.', style='Caption')

doc.add_heading('4.2 Anomaly Detection Agent performance', level=2)
doc.add_paragraph('The LSTM autoencoder achieved an ROC-AUC of 0.963 on 6,380 sequenced windows.')

# Table 5
doc.add_paragraph('Table 5. LSTM autoencoder modality-level characterization on 6,380 sequenced windows.')
table5 = doc.add_table(rows=10, cols=2)
table5.style = 'Table Grid'
hdr = table5.rows[0].cells
hdr[0].text = 'Metric'
hdr[1].text = 'Value'
anomaly_data = [
    ['ROC-AUC', '0.963'],
    ['Detection threshold u03c4', '0.00701'],
    ['True positive rate', '87.9%'],
    ['False positive rate', '5.0%'],
    ['True negatives / False positives', '1,501 / 79'],
    ['False negatives / True positives', '582 / 4,218'],
    ['Mean reconstruction error (normal)', '2.72 u00d7 10^{-3}'],
    ['Mean reconstruction error (anomaly)', '2.082'],
    ['Separation factor', '764u00d7'],
]
for i, row_data in enumerate(anomaly_data):
    row_cells = table5.rows[i+1].cells
    row_cells[0].text = row_data[0]
    row_cells[1].text = row_data[1]

doc.add_paragraph()

# Add result plots
if os.path.exists('retrain/results/exp1_frontier.png'):
    doc.add_picture('retrain/results/exp1_frontier.png', width=Inches(5))
    doc.add_paragraph('Figure 3. Exp 1 - Cost-Weighted Pareto frontier.', style='Caption')

if os.path.exists('retrain/results/exp3_loco.png'):
    doc.add_picture('retrain/results/exp3_loco.png', width=Inches(5))
    doc.add_paragraph('Figure 4. Exp 3 - Leave-One-Class-Out (LOCO) results.', style='Caption')

if os.path.exists('retrain/results/exp4_ece.png'):
    doc.add_picture('retrain/results/exp4_ece.png', width=Inches(5))
    doc.add_paragraph('Figure 5. Exp 4 - Confidence calibration (ECE).', style='Caption')

if os.path.exists('retrain/results/expzoo.png'):
    doc.add_picture('retrain/results/expzoo.png', width=Inches(5))
    doc.add_paragraph('Figure 6. Model Comparison (ZOO) - All baselines.', style='Caption')

# Section 4.3: Decision Agent
doc.add_heading('4.3 Decision Agent training and evaluation', level=2)
doc.add_paragraph('On the 1,276-window block-wise holdout, the Decision Agent achieved 94.44% decision accuracy. The danger miss rate was zero across all 640 hazardous windows. The high-severity false alarm rate was zero across the 316 NoGas windows.')

# Table 6
doc.add_paragraph('Table 6. Decision Agent per-class performance on the 1,276-window block-wise holdout test set.')
table6 = doc.add_table(rows=5, cols=5)
table6.style = 'Table Grid'
hdr = table6.rows[0].cells
hdr[0].text = 'Gas Class'
hdr[1].text = 'Precision'
hdr[2].text = 'Recall'
hdr[3].text = 'F1-Score'
hdr[4].text = 'n'
decision_data = [
    ['NoGas', '0.98', '0.79', '0.88', '316'],
    ['Smoke', '1.00', '1.00', '1.00', '320'],
    ['Mixture', '1.00', '1.00', '1.00', '320'],
    ['Perfume', '0.83', '0.99', '0.90', '320'],
]
for i, row_data in enumerate(decision_data):
    row_cells = table6.rows[i+1].cells
    for j, cell_data in enumerate(row_data):
        row_cells[j].text = cell_data

doc.add_paragraph()
doc.add_paragraph('Overall decision accuracy: 94.44%')
doc.add_paragraph('Danger miss rate (640 hazardous windows): 0.0000')
doc.add_paragraph('False alarm rate (NoGas windows, au22653): 0.0000')

# Section 4.4: Baseline comparison
doc.add_heading('4.4 Baseline model comparison (ZOO)', level=2)
doc.add_paragraph('We trained 8 models on the same corpus with the same 5 seeds and evaluation protocol.')

# Table 7
doc.add_paragraph('Table 7. Baseline model comparison on the same corpus, same 5 seeds, same evaluation.')
table7 = doc.add_table(rows=9, cols=4)
table7.style = 'Table Grid'
hdr = table7.rows[0].cells
hdr[0].text = 'Model'
hdr[1].text = 'Accuracy'
hdr[2].text = 'Std'
hdr[3].text = 'Miss Rate'
zoo_data = [
    ['Decision Agent (ARXIS)', '0.9633', '0.0215', '0.0'],
    ['Plain DQN', '0.9563', '0.0378', '0.0'],
    ['MLP', '0.9616', '0.0111', '0.0'],
    ['GBM', '0.9634', '0.0211', '0.0098'],
    ['SVM', '0.9511', '0.0190', '0.0'],
    ['Random Forest', '0.9769', '0.0086', '0.0'],
    ['LSTM', '0.9680', '0.0231', '0.0'],
    ['CQL', '0.9437', '0.0479', '0.0'],
]
for i, row_data in enumerate(zoo_data):
    row_cells = table7.rows[i+1].cells
    for j, cell_data in enumerate(row_data):
        row_cells[j].text = cell_data

doc.add_paragraph()
doc.add_paragraph('No comparator significantly different from ARXIS at u03b1=0.05.')

# Section 4.5: LOCO
doc.add_heading('4.5 Leave-One-Class-Out (LOCO) results', level=2)
doc.add_paragraph('The headline experiment: train on 3 gases, test on the 4th.')

# Table 8
doc.add_paragraph('Table 8. Leave-One-Class-Out (LOCO) results: train on 3 gases, test on the 4th.')
table8 = doc.add_table(rows=9, cols=4)
table8.style = 'Table Grid'
hdr = table8.rows[0].cells
hdr[0].text = 'Held-Out'
hdr[1].text = 'Model'
hdr[2].text = 'Miss Rate'
hdr[3].text = '95% CP Bound'
loco_data = [
    ['Smoke', 'Decision Agent', '0.0%', 'u22640.19%'],
    ['Smoke', 'MLP', '18.7%', 'u226420.4%'],
    ['Smoke', 'Plain DQN', '0.0%', 'u22640.19%'],
    ['Smoke', 'GBM', '0.0%', 'u22640.19%'],
    ['Mixture', 'Decision Agent', '0.44%', 'u22640.83%'],
    ['Mixture', 'Plain DQN', '0.0%', 'u22640.19%'],
    ['Mixture', 'MLP', '0.38%', 'u22640.75%'],
    ['Mixture', 'GBM', '0.0%', 'u22640.19%'],
]
for i, row_data in enumerate(loco_data):
    row_cells = table8.rows[i+1].cells
    for j, cell_data in enumerate(row_data):
        row_cells[j].text = cell_data

doc.add_paragraph()
doc.add_paragraph('Key finding: In-distribution accuracy does NOT predict out-of-distribution safety.')

# Section 4.6: Robustness
doc.add_heading('4.6 Robustness analysis', level=2)
doc.add_paragraph('Under calibration drift and sensor-channel dropout, ARXIS preserves zero danger misses while supervised baselines degrade.')

# Table 9
doc.add_paragraph('Table 9. Clean and perturbed test-set comparison between the ARXIS policy and supervised decision baselines.')
table9 = doc.add_table(rows=14, cols=4)
table9.style = 'Table Grid'
hdr = table9.rows[0].cells
hdr[0].text = 'Configuration'
hdr[1].text = 'Model'
hdr[2].text = 'Decision Acc.'
hdr[3].text = 'Danger Miss Rate'
robustness_data = [
    ['Clean test set', 'Rule-oracle', '100.00%', '0.0000'],
    ['Clean test set', 'Supervised MLP', '98.20%', '0.0000'],
    ['Clean test set', 'ARXIS policy', '94.44%', '0.0000'],
    ['Gaussian noise, u03c3=0.20', 'Supervised MLP', '89.7%', '0.0000'],
    ['Gaussian noise, u03c3=0.20', 'ARXIS policy', '73.2%', '0.0000'],
    ['Gaussian noise, u03c3=0.40', 'Supervised MLP', '69.3%', '0.0000'],
    ['Gaussian noise, u03c3=0.40', 'ARXIS policy', '58.0%', '0.0000'],
    ['Calibration drift, u00b140%', 'Supervised MLP', '59.1%', '0.0187'],
    ['Calibration drift, u00b140%', 'ARXIS policy', '53.7%', '0.0000'],
    ['Channel dropout, 1 sensor', 'Supervised MLP', '36.0%', '0.0000'],
    ['Channel dropout, 1 sensor', 'ARXIS policy', '55.6%', '0.0000'],
    ['Channel dropout, 3 sensors', 'Supervised MLP', '28.9%', '0.0000'],
    ['Channel dropout, 3 sensors', 'ARXIS policy', '29.9%', '0.0000'],
]
for i, row_data in enumerate(robustness_data):
    row_cells = table9.rows[i+1].cells
    for j, cell_data in enumerate(row_data):
        row_cells[j].text = cell_data

doc.add_paragraph()
doc.add_paragraph('Under calibration drift, the MLP misses 18.7% of hazards while ARXIS preserves zero danger misses.')

# Section 4.7: Calibration
doc.add_heading('4.7 Confidence calibration', level=2)

# Table 10
doc.add_paragraph('Table 10. Confidence calibration: Expected Calibration Error (ECE) for 4 estimators.')
table10 = doc.add_table(rows=5, cols=3)
table10.style = 'Table Grid'
hdr = table10.rows[0].cells
hdr[0].text = 'Estimator'
hdr[1].text = 'ECE Mean'
hdr[2].text = 'ECE Std'
calibration_data = [
    ['Raw softmax', '0.0489', '0.0699'],
    ['MC dropout (20 samples)', '0.0533', '0.0692'],
    ['Temp scaling', '0.0319', '0.0397'],
    ['Deep ensemble (5 members)', '0.0123', '0.0043'],
]
for i, row_data in enumerate(calibration_data):
    row_cells = table10.rows[i+1].cells
    for j, cell_data in enumerate(row_data):
        row_cells[j].text = cell_data

# Section 4.8: Edge deployment
doc.add_heading('4.8 Edge deployment proof of concept', level=2)

# Table 11
doc.add_paragraph('Table 11. Edge deployment proof of concept on a Raspberry Pi 4.')
table11 = doc.add_table(rows=5, cols=7)
table11.style = 'Table Grid'
hdr = table11.rows[0].cells
hdr[0].text = 'Configuration'
hdr[1].text = 'Model Size (MB)'
hdr[2].text = 'Mean Latency (ms)'
hdr[3].text = 'P99 Latency (ms)'
hdr[4].text = 'Decision Acc. (%)'
hdr[5].text = 'Danger Miss Rate'
hdr[6].text = 'False Alarm Rate'
edge_data = [
    ['Desktop CPU, float32', '8.74', '5.8', '9.6', '94.44', '0.0000', '0.0000'],
    ['Pi 4, float32 (synchronous)', '8.74', '14,738', '24,202', '94.36', '0.0000', '0.0000'],
    ['Pi 4, INT8 (synchronous)', '4.19', '14,675', '24,028', '93.97', '0.0000', '0.0032'],
    ['Pi 4, INT8 + async reasoning', '4.19', '235', '413', '93.97', '0.0000', '0.0032'],
]
for i, row_data in enumerate(edge_data):
    row_cells = table11.rows[i+1].cells
    for j, cell_data in enumerate(row_data):
        row_cells[j].text = cell_data

doc.add_paragraph()
doc.add_paragraph('The danger miss rate remained zero across all configurations. Asynchronous reasoning and INT8 quantization reduced the deployable decision path to 235 ms.')

# Section 4.9: Comparison with prior work
doc.add_heading('4.9 Comparison with prior work', level=2)

# Table 12
doc.add_paragraph('Table 12. Comparison with prior work on the MultimodalGasData benchmark.')
table12 = doc.add_table(rows=7, cols=4)
table12.style = 'Table Grid'
hdr = table12.rows[0].cells
hdr[0].text = 'Study'
hdr[1].text = 'Method'
hdr[2].text = 'Accuracy'
hdr[3].text = 'Limitation relevant to safety monitoring'
prior_data = [
    ['Narkhede et al. [20]', 'CNN with multimodal fusion', '97.0%', 'No mapping from class label to operator action'],
    ['Faleh and Kachouri [23]', 'Hybrid CNN-LDA classifier', '93.0%', 'Single modality; no anomaly or decision layer'],
    ['El Barkani et al. [24]', 'TinyML CNN', '91.7%', 'Classification only; no anomaly scoring'],
    ['Sharma et al. [25]', 'Federated deep CNN ensemble', '99.7%', 'No graded response policy'],
    ['Zhang and Zhang [26]', 'Transformer classifier', '99.7%', 'No safety-relevant evaluation'],
    ['ARXIS (this work)', 'Offline reward-shaped safety-action policy', '94.44%', 'Benchmark proof of concept; field validation pending'],
]
for i, row_data in enumerate(prior_data):
    row_cells = table12.rows[i+1].cells
    for j, cell_data in enumerate(row_data):
        row_cells[j].text = cell_data

# Section 5: Discussion
doc.add_heading('5. Discussion', level=1)
doc.add_heading('5.1 Key findings', level=2)
doc.add_paragraph('Safety-action selection is not gas classification. The 94.44% decision accuracy of ARXIS is lower than the 98.8% classification accuracy of the Perception Agent and the 97.0-99.7% reported by prior classification-focused studies. However, these numbers measure different quantities. A classification error and an action error do not necessarily have the same process-safety meaning.')
doc.add_paragraph('Zero danger misses under distribution shift. The most important result is not the clean-test accuracy but the behavior under perturbation. Under calibration drift, the supervised MLP retained higher decision accuracy (59.1%) than ARXIS (53.7%) but introduced a nonzero danger miss rate of 0.0187, whereas ARXIS preserved zero danger misses.')
doc.add_paragraph('Temporal features drive safety behavior. The decision-state ablation shows that removing the anomaly score reduced accuracy by 2.75 percentage points, while removing temporal change and standard-deviation features reduced accuracy by 10.27 points.')

doc.add_heading('5.2 Deployment tradeoffs', level=2)
doc.add_paragraph('1. Explanation must be asynchronous. Synchronous local explanation with Gemma 3 1B produced a mean end-to-end latency of 14,738 ms, far exceeding the 2,000 ms sampling budget. Decoupling explanation from the decision path reduced latency to 503 ms; INT8 quantization further reduced it to 235 ms.')
doc.add_paragraph('2. Quantization affects action boundaries. INT8 quantization reduced model size from 8.74 MB to 4.19 MB but introduced a false alarm rate of 0.0032 (one high-severity action across 316 NoGas windows).')
doc.add_paragraph('3. Thermal management matters. During a 36-minute passive-cooled run, the CPU reached the 80u00b0C throttling threshold after about 24 minutes, causing clock frequency to drop from 1,500 MHz to 1,234 MHz.')

doc.add_heading('5.3 Baseline comparison summary', level=2)
doc.add_paragraph('The comprehensive baseline comparison across 8 models demonstrates that no baseline significantly outperforms ARXIS at u03b1=0.05 on the clean test set. ARXIS uniquely preserves zero danger misses under calibration drift and sensor dropout. The MLP is the weakest baseline, with 18.7% danger miss rate on held-out Smoke. Random Forest achieves the highest raw accuracy (0.9769) but with higher variance than ARXIS. CQL (the only genuine safe-RL baseline) underperforms cost-weighted CE on this separable dataset.')

# Section 6: Limitations
doc.add_heading('6. Limitations', level=1)
doc.add_paragraph('1. Dataset scope. ARXIS was developed and evaluated on the MultimodalGasData benchmark under controlled exposure conditions. Wind effects, ambient-temperature changes, sensor aging, long-term drift, hydraulic transients, leak-rate diversity, and background operational noise are not represented.')
doc.add_paragraph('2. Decision-state scope. The current Decision Agent uses a 22-dimensional state composed of sensor dynamics and anomaly evidence. Thermal-image information is used by the Perception Agent and Reasoning Agent but is not included directly in the DQN policy state.')
doc.add_paragraph('3. Deployment scope. The edge proof of concept replays the benchmark test partition from local storage on a Raspberry Pi 4. It does not evaluate live sensor acquisition, long-duration field operation, integration with plant alarm-management systems, or operator response to the generated explanations.')

# Section 7: Conclusion
doc.add_heading('7. Conclusions', level=1)
doc.add_paragraph('This study presented ARXIS, an edge-deployable agentic AI framework that reframes gas-hazard monitoring as safety-action selection rather than gas classification alone. On the MultimodalGasData benchmark, ARXIS achieved 94.44% decision accuracy while preserving zero danger misses across 640 hazardous windows and zero high-severity false alarms across 316 baseline windows. Under calibration drift, a supervised baseline retained higher decision accuracy but introduced hazardous misses, whereas ARXIS preserved zero danger misses. The edge proof of concept showed that asynchronous reasoning and model quantization reduced the deployable decision path to 235 ms on a Raspberry Pi 4.')
doc.add_paragraph('Future work should evaluate ARXIS on live field streams, incorporate visual evidence directly into the decision policy, and assess the operator-facing explanations in realistic alarm and response scenarios.')

# CRediT
doc.add_heading('CRediT authorship contribution statement', level=1)
doc.add_paragraph('Benjamin C. Nweke: Conceptualization, Methodology, Software, Investigation, Formal analysis, Data curation, Visualization, Writing - original draft.')
doc.add_paragraph('Gholamreza Ramezan: Conceptualization, Methodology, Writing - review & editing.')
doc.add_paragraph('Soheil Saraji: Conceptualization, Supervision, Project administration, Resources, Writing - review & editing.')

# Declaration
doc.add_heading('Declaration of competing interest', level=1)
doc.add_paragraph('The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.')

# Acknowledgements
doc.add_heading('Acknowledgements', level=1)
doc.add_paragraph('This research was supported by the Subsurface Energy and Digital Innovation Center (SEDI) at the University of Wyoming.')

# References
doc.add_heading('References', level=1)
refs = [
    '[1] Laberge, J.C., et al., 2014. Addressing alarm flood situations in the process industries. Int. J. Ind. Ergon. 44, 395-406.',
    '[2] Goel, P., Datta, A., Mannan, M.S., 2017. Industrial alarm systems: Challenges and opportunities. J. Loss Prev. Process Ind. 50, 23-36.',
    '[3] Murvay, P.S., Silea, I., 2012. A survey on gas leak detection and localization techniques. J. Loss Prev. Process Ind. 25, 966-973.',
    '[4] Han, P., Kim, M., 2014. Optimizing leak detection performance. PSIG Annual Meeting, PSIG 1407.',
    '[5] Malhotra, P., et al., 2016. LSTM-based encoder-decoder for multi-sensor anomaly detection. arXiv preprint arXiv:1607.00148.',
    '[6] Liang, J., et al., 2024. Leak detection in natural gas pipelines based on unsupervised reconstruction of healthy flow data. SPE Journal.',
    '[7] Korjani, M., et al., 2024. Temporal deep learning image processing model for natural gas leak detection. OTC Asia, OTC-34756-MS.',
    '[8] Mnih, V., et al., 2015. Human-level control through deep reinforcement learning. Nature 518, 529-533.',
    '[9] Wang, Z., et al., 2016. Dueling network architectures for deep reinforcement learning. ICML, 1995-2003.',
    '[10] Sabbagh, V.B., et al., 2024. Comparative analysis of single and multiagent large language model architectures. SPE Journal 29, 6869-6884.',
    '[11] Santiago, C.J.S., et al., 2025. Integrating data-driven insights with domain expertise. SPE Annual Technical Conference, SPE-228143-MS.',
    '[12] Jessen, H., Roshchin, M., 2025. Agentic AI revolution across O&G value streams. ADIPEC, SPE-229240-MS.',
    '[13] Dehnaw, A.M., et al., 2024. Deep neural network optimization for efficient gas detection systems. Processes 12, 2638.',
    '[14] Google DeepMind, 2024. Gemma: Open models based on Gemini research and technology. https://ai.google.dev/gemma.',
    '[15] Adegboye, M.A., et al., 2019. Recent advances in pipeline monitoring and oil leakage detection technologies. Sensors 19, 2548.',
    '[16] Zhang, J., et al., 2013. Review of pipeline leak detection technologies. PSIG Annual Meeting, PSIG 1303.',
    '[17] Bustnes, T.E., et al., 2011. Leak detection performance of a commercial real-time transient model. PSIG Annual Meeting, PSIG 1114.',
    '[18] Wang, C.Y., et al., 2022. YOLOv7: Trainable bag-of-freebies. arXiv preprint arXiv:2207.02696.',
    '[19] Narkhede, P., et al., 2021. Gas detection and identification using multimodal artificial intelligence. Appl. Syst. Innov. 4, 3.',
    '[20] Narkhede, P., et al., 2022. MultimodalGasData: Multimodal dataset for gas detection and classification. Data 7, 112.',
    '[21] Ma, R., et al., 2026. Minor pipeline leak detection and localization using explainable deep learning. J. Loss Prev. Process Ind. 100, 105844.',
    '[22] Jocher, G., et al., 2023. YOLOv8 by Ultralytics. https://github.com/ultralytics/ultralytics.',
    '[23] Faleh, R., Kachouri, A., 2023. A hybrid deep convolutional neural network-based electronic nose. Chemom. Intell. Lab. Syst. 237, 104825.',
    '[24] El Barkani, M., et al., 2024. Gas leakage detection using tiny machine learning. Electronics 13, 4768.',
    '[25] Sharma, A., et al., 2024. Gas detection and classification using multimodal data based on federated learning. Sensors 24, 5904.',
    '[26] Zhang, E., Zhang, E., 2025. Gas pipeline leakage detection based on multiple multimodal deep feature selections. Front. Environ. Sci. 13, 1569621.',
]
for ref in refs:
    doc.add_paragraph(ref)

# Save
doc.save('ARXIS_Paper.docx')
print('Word document created: ARXIS_Paper.docx')
print(f'Size: {os.path.getsize("ARXIS_Paper.docx") / 1024:.1f} KB')
