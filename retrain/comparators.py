"""Comparison models for Exp #2 (near-boundary stress).

1. SupervisedMLP        - plain classifier, same hidden capacity as the DQN
                          features block, trained with cross-entropy.
2. HazardWeightedGBM    - gradient-boosted trees whose hazardous training rows
                          (Smoke, Mixture) carry 2.5 times the weight of the
                          nonhazardous rows (NoGas, Perfume). The primary
                          gradient-boosting comparator from the November 2026
                          rerun onwards.
   CleanAirWeightedGBM  - the configuration released up to v18.1, which weighted
                          by target action (NoGas 3.0, Perfume 1.0, Smoke and
                          Mixture 1.2), i.e. 2.5 to 1 toward clean air. It was
                          described as cost-sensitive but weighted the wrong
                          direction; it is kept only for provenance.
3. UnweightedDQN        - identical DuelingDQN architecture BUT trained with a
                          SYMMETRIC reward (+1/-1), isolating whether the
                          fail-safe skew comes from the REWARD, not the net.
4. SupervisedDQN        - identical DuelingDQN but trained as plain supervised
                          cross-entropy (no RL), for an apples-to-apples arch compare.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight

from retrain.agent_rl import DuelingDQN, STATE_DIM, N_ACTIONS
from retrain.rewards import reward_asymmetric, RULE_ORACLE



# --------------------------- Supervised MLP ---------------------------
class SupervisedMLP:
    def __init__(self, seed=42):
        self.clf = MLPClassifier(hidden_layer_sizes=(256, 256, 128),
                                 activation="relu", max_iter=400,
                                 early_stopping=True, random_state=seed)

    def fit(self, X, y_action):
        self.clf.fit(X, y_action)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


# ----------------------- Gradient-boosting comparators -----------------------
# Sample weights are keyed on the training target action, which is a deterministic
# function of the gas class (NoGas -> 0, Perfume -> 1, Smoke -> 3, Mixture -> 4),
# so an action key and a gas key give identical weights on training rows.
#
# HAZARD_ACTION_WEIGHT (primary): hazardous rows 2.5, nonhazardous rows 1.0.
#     NoGas 1.0, Perfume 1.0, Smoke 2.5, Mixture 2.5.
# CLEAN_AIR_ACTION_WEIGHT (released up to v18.1, provenance only):
#     NoGas 3.0, Perfume 1.0, Smoke 1.2, Mixture 1.2, i.e. 2.5 to 1 toward clean air.
HAZARD_ACTION_WEIGHT = {0: 1.0, 1: 1.0, 2: 1.0, 3: 2.5, 4: 2.5}
CLEAN_AIR_ACTION_WEIGHT = {0: 3.0, 1: 1.0, 2: 1.0, 3: 1.2, 4: 1.2}
ACTION_WEIGHT = CLEAN_AIR_ACTION_WEIGHT   # historical name, kept for the provenance probes


class _WeightedGBM:
    WEIGHTS = None

    def __init__(self, seed=42):
        self.clf = GradientBoostingClassifier(random_state=seed, n_estimators=300)

    def sample_weight(self, y_action):
        return np.array([self.WEIGHTS.get(int(a), 1.0) for a in y_action], dtype=float)

    def fit(self, X, y_action):
        self.clf.fit(X, y_action, sample_weight=self.sample_weight(y_action))

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


class HazardWeightedGBM(_WeightedGBM):
    """Primary gradient-boosting comparator: hazardous rows weighted 2.5 to 1."""
    WEIGHTS = HAZARD_ACTION_WEIGHT


class CleanAirWeightedGBM(_WeightedGBM):
    """Configuration released up to v18.1 (2.5 to 1 toward clean air); provenance only."""
    WEIGHTS = CLEAN_AIR_ACTION_WEIGHT


# Deprecated name. The deprecated drivers run_exp2_real, run_exp3_loco and
# run_expzoo still import it and keep their historical behaviour; no active
# driver uses it.
CostSensitiveGBM = CleanAirWeightedGBM


# --------------------- Supervised DuelingDQN (CE) ---------------------
class SupervisedDQN:
    """DuelingDQN trained with cross-entropy on actions (no RL, no reward).
    Optional per-sample `sample_weight` realizes the cost-asymmetry curve
    from Exp #1 as a weighted CE objective -- same architecture, same
    features, far cheaper than full RL and directly maps cost ratio -> weight.
    """
    def __init__(self, device="cpu", seed=42):
        self.device = device
        torch.manual_seed(seed)
        self.net = DuelingDQN().to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=2e-4)

    def fit(self, X, y_action, sample_weight=None, epochs=60, batch=256):
        X = torch.from_numpy(np.asarray(X, dtype=np.float32)).float().to(self.device)
        y = torch.from_numpy(np.asarray(y_action, dtype=np.int64)).to(self.device)
        if sample_weight is not None:
            sw = torch.from_numpy(np.asarray(sample_weight, dtype=np.float32)).to(self.device)
        else:
            sw = None
        self.net.train()
        n = len(X)
        for ep in range(epochs):
            perm = torch.randperm(n)
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                logits = self.net(X[idx])
                if sw is not None:
                    w = sw[idx]
                    loss = (F.cross_entropy(logits, y[idx], reduction="none") * w).mean()
                else:
                    loss = F.cross_entropy(logits, y[idx])
                self.opt.zero_grad(); loss.backward(); self.opt.step()

    @torch.no_grad()
    def predict_q(self, X):
        self.net.eval()
        X = torch.from_numpy(np.asarray(X, dtype=np.float32)).float().to(self.device)
        return self.net(X).cpu().numpy()

    def predict(self, X):
        return self.predict_q(X).argmax(1)

    def save(self, path, scales=None):
        """Persist weights + the normalization constants the model was trained
        with, so DecisionTool (src/tools/decision_tool.py) can reconstruct the
        exact 22-dim state at inference.

        scales: dict with keys anom_p1, anom_p99, feat_scaler_mean,
        feat_scaler_scale (all float arrays). Required for live serving.
        """
        ckpt = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
        if scales:
            for key in ("anom_p1", "anom_p99", "feat_scaler_mean", "feat_scaler_scale"):
                if key in scales and scales[key] is not None:
                    ckpt[key] = scales[key]
        torch.save(ckpt, path)

    @classmethod
    def load(cls, path, device="cpu", seed=42):
        obj = cls(device=device, seed=seed)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        sd = {k: v for k, v in ckpt.items()
              if not k.startswith(("anom_p", "feat_scaler"))}
        obj.net.load_state_dict(sd)
        obj.net.eval()
        return obj, ckpt


# --------------------- cost-weighted sample weights ---------------------
def cost_weighted_sample_weight(y_gas, miss_cost=2.0, false_cost=1.0):
    """Per-row weight that encodes the cost-asymmetry curve directly into a
    weighted-CE objective.

    The sweep parameter (miss_cost : false_cost) MUST reach the training
    weights for the Pareto frontier to be meaningful. We weight each row by
    the cost of *getting that row's gas wrong*:
      - DANGER gas (Smoke/Mixture): weight = miss_cost  (missing it is the
        worst failure, so up-weight it as miss_cost grows)
      - NON-DANGER gas (NoGas/Perfume): weight = false_cost  (a false alarm
        here is the 'false_cost' failure mode)
    Sweeping miss_cost in {2..20} with false_cost=1 thus moves the
    danger:clean weight ratio from 2:1 to 20:1 -- exactly the intended
    frontier. This replaces the earlier (buggy) version that evaluated
    reward_asymmetric only at the *correct* action, which made every ratio
    collapse to the same fixed weights.
    """
    w = np.ones(len(y_gas), dtype=np.float32)
    for i, g in enumerate(y_gas):
        g = int(g)
        w[i] = miss_cost if g in (1, 2) else false_cost
    return w


# --------------------- Unweighted DQN (symmetric reward) ---------------------
# Reuses RLAgent but with reward_symmetric so the only difference vs the
# asymmetric Decision Agent is the reward shape.
