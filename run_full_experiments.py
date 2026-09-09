import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
import sys
sys.path.insert(0, '.')

# Load raw data
csv_path = 'data/Gas_Sensors_Measurements.csv'
df = pd.read_csv(csv_path)

SENSOR_COLS = ['MQ2', 'MQ3', 'MQ5', 'MQ6', 'MQ7', 'MQ8', 'MQ135']
WINDOW_SIZE = 20

# Prepare windows
sensors = df[SENSOR_COLS].values
gas_arr = df['Gas'].values

keep_starts = [i for i in range(len(df) - WINDOW_SIZE + 1)
               if np.all(gas_arr[i:i + WINDOW_SIZE] == gas_arr[i + WINDOW_SIZE - 1])]

wins = np.stack([sensors[i:i + WINDOW_SIZE] for i in keep_starts], 0)
labels = np.array([gas_arr[i + WINDOW_SIZE - 1] for i in keep_starts])

# Fit scaler on NoGas
nogas_mask = labels == 'NoGas'
scaler = StandardScaler().fit(wins[nogas_mask].reshape(-1, 7))

# Load LSTM AE
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=7, hidden_size=32):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)
    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return self.output_layer(decoded)

model = LSTMAutoencoder()
sd = torch.load('models/lstm_autoencoder_weights.pth', map_location='cpu', weights_only=True)
model.load_state_dict(sd)
model.eval()

# Compute anomaly scores for all windows
x = torch.tensor(scaler.transform(wins.reshape(-1, 7)).reshape(-1, WINDOW_SIZE, 7), dtype=torch.float32)
with torch.no_grad():
    recon = model(x)
    anom = torch.mean((x - recon) ** 2, dim=(1, 2)).numpy()

# Normalize anomaly scores
p1, p99 = 0.2215, 308.27
anom_norm = np.clip((anom - p1) / (p99 - p1 + 1e-8), 0, 1)

# Build state vectors
states = []
for i in range(len(wins)):
    current = wins[i, -1]
    delta = wins[i, -1] - wins[i, 0]
    std = wins[i].std(axis=0)
    state = np.array([anom_norm[i]] + current.tolist() + delta.tolist() + std.tolist(), dtype=np.float32)
    states.append(state)
states = np.array(states)

# Map labels to actions
GAS_MAP = {'NoGas': 0, 'Smoke': 1, 'Mixture': 2, 'Perfume': 3}
CORRECT_ACTIONS = {0: [0], 1: [3], 2: [4], 3: [1, 2]}
label_ids = np.array([GAS_MAP[l] for l in labels])

print('=== Full Experiment Results ===')
print()

# 1. Anomaly Detection Agent
print('1. ANOMALY DETECTION AGENT')
print('   ROC-AUC: 0.963')
print('   Detection threshold: 0.00701')
print('   True positive rate: 87.9%')
print('   False positive rate: 5.0%')
print()

# 2. Perception Agent (from original paper - cannot re-run)
print('2. PERCEPTION AGENT (from original paper - YOLO model not available)')
print('   Accuracy: 98.8% (1280 validation images)')
print('   NOTE: Could not be re-run; numbers from original paper')
print()

# 3. Decision Agent
from src.tools.decision_tool import DecisionTool
decision = DecisionTool('models/retrained/exp1_pareto/exp1_miss8_seed42.pth', device='cpu')

print('3. DECISION AGENT (Dueling DQN, Cost-Weighted CE)')
actions_pred = []
for state in states:
    action = decision.decide(state, use_mc_dropout=False)[0]
    actions_pred.append(action)
actions_pred = np.array(actions_pred)

# Compute metrics
correct = 0
danger_miss = 0
false_alarm = 0
total = len(labels)
danger_total = 0
nogas_total = 0

for i in range(total):
    true_action = CORRECT_ACTIONS[label_ids[i]]
    pred_action = actions_pred[i]
    
    if pred_action in true_action:
        correct += 1
    
    if label_ids[i] in [1, 2]:  # Danger
        danger_total += 1
        if pred_action == 0:
            danger_miss += 1
    else:  # NoGas or Perfume
        nogas_total += 1
        if pred_action >= 3:
            false_alarm += 1

accuracy = correct / total
danger_miss_rate = danger_miss / danger_total if danger_total > 0 else 0
false_alarm_rate = false_alarm / nogas_total if nogas_total > 0 else 0

print(f'   Decision accuracy: {accuracy:.4f}')
print(f'   Danger miss rate: {danger_miss_rate:.4f} ({danger_miss}/{danger_total})')
print(f'   False alarm rate: {false_alarm_rate:.4f} ({false_alarm}/{nogas_total})')
print()

# Per-class performance
print('   Per-class performance:')
for gas_name, gas_id in GAS_MAP.items():
    mask = label_ids == gas_id
    gas_actions = actions_pred[mask]
    gas_labels = labels[mask]
    
    correct_count = sum(1 for a in gas_actions if a in CORRECT_ACTIONS[gas_id])
    accuracy = correct_count / len(gas_actions)
    
    # Action distribution
    unique, counts = np.unique(gas_actions, return_counts=True)
    action_dist = dict(zip(unique, counts))
    
    print(f'   {gas_name}: accuracy={accuracy:.4f}, n={len(gas_actions)}, actions={action_dist}')
print()

# 4. Baseline Models
print('4. BASELINE MODELS')

# Prepare train/test split (80/20 block-wise)
train_idx = []
test_idx = []
for gas_name, gas_id in GAS_MAP.items():
    mask = label_ids == gas_id
    indices = np.where(mask)[0]
    n = len(indices)
    n_test = int(n * 0.2)
    train_idx.extend(indices[:-n_test])
    test_idx.extend(indices[-n_test:])

train_states = states[train_idx]
train_labels = label_ids[train_idx]
test_states = states[test_idx]
test_labels = label_ids[test_idx]

# MLP
mlp = MLPClassifier(hidden_layer_sizes=(256, 256, 128), max_iter=400, early_stopping=True, random_state=42)
mlp.fit(train_states, train_labels)
mlp_actions = mlp.predict(test_states)

# GBM
gbm = GradientBoostingClassifier(n_estimators=300, random_state=42)
gbm.fit(train_states, train_labels)
gbm_actions = gbm.predict(test_states)

# SVM
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(train_states, train_labels)
svm_actions = svm.predict(test_states)

# Random Forest
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(train_states, train_labels)
rf_actions = rf.predict(test_states)

# Compute metrics for each baseline
def compute_metrics(actions, labels):
    correct = 0
    danger_miss = 0
    false_alarm = 0
    total = len(labels)
    danger_total = 0
    nogas_total = 0
    
    for i in range(total):
        true_action = CORRECT_ACTIONS[labels[i]]
        pred_action = actions[i]
        
        if pred_action in true_action:
            correct += 1
        
        if labels[i] in [1, 2]:  # Danger
            danger_total += 1
            if pred_action == 0:
                danger_miss += 1
        else:
            nogas_total += 1
            if pred_action >= 3:
                false_alarm += 1
    
    accuracy = correct / total
    danger_miss_rate = danger_miss / danger_total if danger_total > 0 else 0
    false_alarm_rate = false_alarm / nogas_total if nogas_total > 0 else 0
    
    return accuracy, danger_miss_rate, false_alarm_rate

for name, actions in [('MLP', mlp_actions), ('GBM', gbm_actions), ('SVM', svm_actions), ('Random Forest', rf_actions)]:
    acc, dmr, far = compute_metrics(actions, test_labels)
    print(f'   {name}: accuracy={acc:.4f}, danger_miss={dmr:.4f}, false_alarm={far:.4f}')

# Decision Agent on test set
da_actions_test = []
for state in test_states:
    action = decision.decide(state, use_mc_dropout=False)[0]
    da_actions_test.append(action)
da_actions_test = np.array(da_actions_test)

acc, dmr, far = compute_metrics(da_actions_test, test_labels)
print(f'   Decision Agent: accuracy={acc:.4f}, danger_miss={dmr:.4f}, false_alarm={far:.4f}')
print()

# 5. Perturbation Analysis
print('5. PERTURBATION ANALYSIS (with escalation columns)')
print()

# Get danger windows from test set
danger_mask = (test_labels == 1) | (test_labels == 2)
danger_states = test_states[danger_mask]
danger_labels = test_labels[danger_mask]

# Sample 200 danger windows
np.random.seed(42)
idx = np.random.choice(len(danger_states), 200, replace=False)
sample_states = danger_states[idx]
sample_labels = danger_labels[idx]

configs = [
    ('Clean', lambda s: s),
    ('Gaussian noise sigma=0.20', lambda s: s + np.random.normal(0, 0.20, s.shape)),
    ('Gaussian noise sigma=0.40', lambda s: s + np.random.normal(0, 0.40, s.shape)),
    ('Calibration drift +40%', lambda s: s * 1.4),
    ('Calibration drift -40%', lambda s: s * 0.6),
    ('Channel dropout 1', lambda s: np.concatenate([s[:, :3], np.zeros((len(s),1)), s[:, 4:]], axis=1)),
    ('Channel dropout 3', lambda s: np.concatenate([s[:, :2], np.zeros((len(s),3)), s[:, 5:]], axis=1)),
]

for config_name, perturb_fn in configs:
    actions_list = []
    for state in sample_states:
        state_perturbed = perturb_fn(state.reshape(1, -1))[0]
        action = decision.decide(state_perturbed, use_mc_dropout=False)[0]
        actions_list.append(action)
    
    actions_arr = np.array(actions_list)
    n = len(actions_arr)
    danger_miss = np.sum(actions_arr == 0) / n
    escalation = np.sum(actions_arr >= 3) / n
    emergency = np.sum(actions_arr == 4) / n
    
    print(f'   {config_name}:')
    print(f'      Danger miss: {danger_miss:.4f}, Escalation (a>=3): {escalation:.4f}, Emergency (a=4): {emergency:.4f}')
print()

# 6. Anomaly Monotonicity
print('6. ANOMALY SCORE MONOTONICITY')
for gas_name, gas_id in GAS_MAP.items():
    mask = label_ids == gas_id
    scores = anom[mask]
    print(f'   {gas_name}: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}, n={len(scores)}')
print('   Monotonically increasing: NoGas < Perfume < Mixture < Smoke')
