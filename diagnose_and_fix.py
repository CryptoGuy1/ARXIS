import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
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

# Compute anomaly scores
x = torch.tensor(scaler.transform(wins.reshape(-1, 7)).reshape(-1, WINDOW_SIZE, 7), dtype=torch.float32)
with torch.no_grad():
    recon = model(x)
    anom = torch.mean((x - recon) ** 2, dim=(1, 2)).numpy()

# Normalize using correct scale
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

# Train/test split (80/20 block-wise)
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

print(f'Train: {len(train_states)}, Test: {len(test_states)}')

# Train a simple MLP classifier as baseline
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(256, 256, 128), max_iter=500, early_stopping=True, random_state=42)
mlp.fit(train_states, train_labels)
mlp_actions = mlp.predict(test_states)

# Compute metrics
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

acc, dmr, far = compute_metrics(mlp_actions, test_labels)
print(f'MLP: accuracy={acc:.4f}, danger_miss={dmr:.4f}, false_alarm={far:.4f}')

# Per-class
for gas_name, gas_id in GAS_MAP.items():
    mask = test_labels == gas_id
    gas_actions = mlp_actions[mask]
    correct_count = sum(1 for a in gas_actions if a in CORRECT_ACTIONS[gas_id])
    acc = correct_count / len(gas_actions)
    unique, counts = np.unique(gas_actions, return_counts=True)
    action_dist = dict(zip(unique, counts))
    print(f'  {gas_name}: accuracy={acc:.4f}, actions={action_dist}')
