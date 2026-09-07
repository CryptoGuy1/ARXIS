"""Reward functions for the Decision Agent.

Two families:
  - GAS_ID based (ground truth available): always correct, matches the
    project's training notebook compute_reward, plus a cost-asymmetric
    version parameterized by `miss_cost` (penalty for missing a dangerous
    gas) used for the Reward-Asymmetry Pareto Frontier (Exp #1).
  - ANOMALY based (blind live mode): unreliable for this dataset, kept
    only for parity / documentation.

CANONICAL MAPPING (gas_id):
    0 NoGas, 1 Smoke, 2 Mixture, 3 Perfume
CORRECT ACTIONS:
    0 -> [0] Monitor
    1 -> [3] Raise Alarm
    2 -> [4] Emergency Shutdown
    3 -> [1,2] Increase Sampling / Request Verification
"""
import numpy as np

GAS_MAP = {"NoGas": 0, "Smoke": 1, "Mixture": 2, "Perfume": 3}
CORRECT_ACTIONS = {0: [0], 1: [3], 2: [4], 3: [1, 2]}
DANGER_GAS_IDS = {1, 2}  # Smoke, Mixture


# ---------------------------------------------------------------------------
# Symmetric reward (1:1 cost) -- the FLOOR anchor for Exp #1.
# This is the project's original compute_reward with asymmetry removed:
# every correct action gets +1, every wrong action gets -1, max_abs=1.
# ---------------------------------------------------------------------------
def reward_symmetric(gas_id: int, action: int, anomaly: float = 0.0) -> float:
    correct = CORRECT_ACTIONS.get(int(gas_id), [])
    if int(action) in correct:
        return 1.0
    return -1.0


# ---------------------------------------------------------------------------
# Asymmetric reward (Exp #1 sweep) -- cost ratio C = miss_cost : false_cost.
# Mirrors the original's structure but replaces the hardcoded
# (-10 miss, -4 false-alarm, +2 correct, +1 precision, ... ) with a
# clean cost ratio so the frontier is interpretable.
#
#   correct action              : +1.0
#   dangerous gas MISSED (act 0) : -miss_cost
#   clean air false alarm (>=3)  : -false_cost  (= miss_cost / C)
#   severe under/over response   : -false_cost/2 (minor miscalibration)
#   precision bonus (right severe action on danger): +1.0
#
# C = miss_cost / false_cost. We sweep miss_cost in {2,4,6,8,10,12,16,20}
# with false_cost = 1 -> C = 2:1 .. 20:1.
# ---------------------------------------------------------------------------
def reward_asymmetric(gas_id: int, action: int, anomaly: float = 0.0,
                      miss_cost: float = 2.0, false_cost: float = 1.0) -> float:
    gas_id = int(gas_id)
    action = int(action)
    correct = CORRECT_ACTIONS.get(gas_id, [])
    r = 0.0
    if action in correct:
        r += 1.0
        if gas_id in DANGER_GAS_IDS:
            r += 1.0  # precision bonus for correct severe response
    else:
        r -= false_cost  # wrong action costs the base false cost

    # Hard safety-critical skew (asymmetric core)
    if gas_id in DANGER_GAS_IDS and action == 0:
        r -= miss_cost  # missing a dangerous gas is the worst failure
    if gas_id == 0 and action >= 3:
        r -= false_cost  # false emergency alarm on clean air
    # minor miscalibration penalty (under-responding to a danger gas)
    if gas_id in DANGER_GAS_IDS and action not in (3, 4):
        r -= false_cost / 2.0
    return float(r)


# ---------------------------------------------------------------------------
# Original project reward (verbatim) -- used as the reference "honest" config
# and for the rule-oracle comparison baseline in Exp #1.
# ---------------------------------------------------------------------------
def reward_original(gas_id: int, action: int, anomaly: float = 0.0) -> float:
    reward = 0.0
    correct = CORRECT_ACTIONS.get(int(gas_id), [])
    if int(action) in correct:
        reward += 2.0
        if int(gas_id) in DANGER_GAS_IDS:
            reward += anomaly
    else:
        reward -= 2.0
    if int(gas_id) in DANGER_GAS_IDS and int(action) == 0:
        reward -= 10.0
    if int(gas_id) == 0 and int(action) >= 3:
        reward -= 4.0
    if int(gas_id) == 1 and int(action) == 3:
        reward += 1.0
    if int(gas_id) == 2 and int(action) == 4:
        reward += 1.0
    if int(gas_id) == 3:
        if anomaly > 0.5 and int(action) == 2:
            reward += 0.8
        elif anomaly > 0.5 and int(action) == 1:
            reward -= 0.5
        elif anomaly <= 0.5 and int(action) == 1:
            reward += 0.5
    return float(np.clip(reward / 3.0, -2.0, 2.0))


# Rule-oracle "agent": deterministic mapping gas_id -> action (ceiling anchor).
# This is the best achievable action choice given perfect knowledge.
RULE_ORACLE = {0: 0, 1: 3, 2: 4, 3: 1}  # NoGas->Monitor, Smoke->Alarm, Mixture->Shutdown, Perfume->IncreaseSampling


def rule_oracle_action(gas_id: int) -> int:
    return RULE_ORACLE.get(int(gas_id), 0)


# Backwards-compat alias
def compute_reward(gas_id, action, anomaly=0.0, **kwargs):
    return reward_original(gas_id, action, anomaly)
