import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, '.')

# Load raw data
csv_path = 'data/Gas_Sensors_Measurements.csv'
df = pd.read_csv(csv_path)

SENSOR_COLS = ['MQ2', 'MQ3', 'MQ5', 'MQ6', 'MQ7', 'MQ8', 'MQ135']
WINDOW_SIZE = 20

sensors = df[SENSOR_COLS].values
gas_arr = df['Gas'].values

keep_starts = [i for i in range(len(df) - WINDOW_SIZE + 1)
               if np.all(gas_arr[i:i + WINDOW_SIZE] == gas_arr[i + WINDOW_SIZE - 1])]

wins = np.stack([sensors[i:i + WINDOW_SIZE] for i in keep_starts], 0)
labels = np.array([gas_arr[i + WINDOW_SIZE - 1] for i in keep_starts])

GAS_MAP = {'NoGas': 0, 'Smoke': 1, 'Mixture': 2, 'Perfume': 3}
ACTION_MAP = {0: 0, 1: 3, 2: 4, 3: 1}  # NoGas->Monitor, Smoke->Raise Alarm, Mixture->Shutdown, Perfume->Increase Sampling
CORRECT_ACTIONS = {0: [0], 1: [3], 2: [4], 3: [1, 2]}
label_ids = np.array([GAS_MAP[l] for l in labels])
action_targets = np.array([ACTION_MAP[l] for l in label_ids])

# Proper train/test split (80/20 block-wise)
train_idx = []
test_idx = []
for gas_name, gas_id in GAS_MAP.items():
    mask = label_ids == gas_id
    indices = np.where(mask)[0]
    n = len(indices)
    n_test = int(n * 0.2)
    train_idx.extend(indices[:-n_test])
    test_idx.extend(indices[-n_test:])

# Use full window (flattened)
train_flat = wins[train_idx].reshape(len(train_idx), -1)
test_flat = wins[test_idx].reshape(len(test_idx), -1)
train_actions = action_targets[train_idx]
test_actions = action_targets[test_idx]

scaler = StandardScaler().fit(train_flat)
train_scaled = scaler.transform(train_flat)
test_scaled = scaler.transform(test_flat)

print(f'Train: {len(train_scaled)}, Test: {len(test_scaled)}')

# Train MLP on correct ACTION targets
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(128, 128), max_iter=1000, early_stopping=True, random_state=42)
mlp.fit(train_scaled, train_actions)
mlp_actions = mlp.predict(test_scaled)

# Evaluate: check if predicted action is in the correct set for the true label
correct = sum(1 for a, l in zip(mlp_actions, action_targets[test_idx]) if a == l)
print(f'MLP (action targets) exact accuracy: {correct/len(test_actions):.4f}')

# Per-class
for gas_name, gas_id in GAS_MAP.items():
    mask = label_ids[test_idx] == gas_id
    gas_actions = mlp_actions[mask]
    correct_action = ACTION_MAP[gas_id]
    correct_count = sum(1 for a in gas_actions if a == correct_action)
    acc = correct_count / len(gas_actions)
    unique, counts = np.unique(gas_actions, return_counts=True)
    action_dist = dict(zip(unique, counts))
    print(f'  {gas_name}: accuracy={acc:.4f}, actions={action_dist}')

print()
print('=== Approach: Gradient Boosting on Action Targets ===')
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=300, random_state=42)
gb.fit(train_scaled, train_actions)
gb_actions = gb.predict(test_scaled)

correct = sum(1 for a, l in zip(gb_actions, test_actions) if a == l)
print(f'GBM (action targets) exact accuracy: {correct/len(test_actions):.4f}')

# Per-class
for gas_name, gas_id in GAS_MAP.items():
    mask = label_ids[test_idx] == gas_id
    gas_actions = gb_actions[mask]
    correct_action = ACTION_MAP[gas_id]
    correct_count = sum(1 for a in gas_actions if a == correct_action)
    acc = correct_count / len(gas_actions)
    unique, counts = np.unique(gas_actions, return_counts=True)
    action_dist = dict(zip(unique, counts))
    print(f'  {gas_name}: accuracy={acc:.4f}, actions={action_dist}')
