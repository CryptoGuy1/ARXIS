import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import sys
sys.path.insert(0, '.')

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
ACTION_MAP = {0: 0, 1: 3, 2: 4, 3: 1}
label_ids = np.array([GAS_MAP[l] for l in labels])
action_targets = np.array([ACTION_MAP[l] for l in label_ids])

train_idx = []
test_idx = []
for gas_name, gas_id in GAS_MAP.items():
    mask = label_ids == gas_id
    indices = np.where(mask)[0]
    n = len(indices)
    n_test = int(n * 0.2)
    train_idx.extend(indices[:-n_test])
    test_idx.extend(indices[-n_test:])

train_flat = wins[train_idx].reshape(len(train_idx), -1)
test_flat = wins[test_idx].reshape(len(test_idx), -1)
train_actions = action_targets[train_idx]
test_actions = action_targets[test_idx]

scaler = StandardScaler().fit(train_flat)
train_scaled = scaler.transform(train_flat)
test_scaled = scaler.transform(test_flat)

# Train Gradient Boosting on action targets
gb = GradientBoostingClassifier(n_estimators=300, random_state=42)
gb.fit(train_scaled, train_actions)
gb_actions = gb.predict(test_scaled)

correct = sum(1 for a, l in zip(gb_actions, test_actions) if a == l)
print(f'Gradient Boosting accuracy: {correct/len(test_actions):.4f}')

# Save the model and scaler
joblib.dump(gb, 'models/retrained/exp_gbm_action_targets.joblib')
joblib.dump(scaler, 'models/retrained/feat_scaler_action_targets.joblib')
print('Saved model: models/retrained/exp_gbm_action_targets.joblib')
print('Saved scaler: models/retrained/feat_scaler_action_targets.joblib')

# Verify
gb_loaded = joblib.load('models/retrained/exp_gbm_action_targets.joblib')
scaler_loaded = joblib.load('models/retrained/feat_scaler_action_targets.joblib')
test_scaled_loaded = scaler_loaded.transform(test_flat)
gb_actions_loaded = gb_loaded.predict(test_scaled_loaded)
correct_loaded = sum(1 for a, l in zip(gb_actions_loaded, test_actions) if a == l)
print(f'Verification accuracy: {correct_loaded/len(test_actions):.4f}')
