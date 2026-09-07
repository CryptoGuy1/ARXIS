"""Exp #4 comparators: confidence/uncertainty estimators + calibration.

Baselines:
  - Raw softmax        : argmax Q, softmax(Q) as confidence (uncalibrated)
  - MC Dropout (20)     : DuelingDQN.predict_with_uncertainty (in agent_rl)
  - Deep Ensemble (5)   : 5 independently seeded DuelingDQN, mean logits
  - Temperature scaling : post-hoc T on the raw softmax (cheap calibrator)

All expose predict_q(X) and predict_proba(X) so ECE is comparable.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from retrain.agent_rl import DuelingDQN


class DeepEnsemble:
    def __init__(self, n=5, device="cpu", base_seed=42):
        self.device = device
        self.nets = []
        for k in range(n):
            torch.manual_seed(base_seed + 1000 * k)
            net = DuelingDQN().to(device)
            self.nets.append(net)

    def fit(self, X, y_action, epochs=40, batch=256):
        X = torch.from_numpy(np.asarray(X, dtype=np.float32)).float().to(self.device)
        y = torch.from_numpy(np.asarray(y_action, dtype=np.int64)).to(self.device)
        for net in self.nets:
            net.train()
            opt = torch.optim.Adam(net.parameters(), lr=2e-4)
            n = len(X)
            for ep in range(epochs):
                perm = torch.randperm(n)
                for s in range(0, n, batch):
                    idx = perm[s:s + batch]
                    loss = F.cross_entropy(net(X[idx]), y[idx])
                    opt.zero_grad(); loss.backward(); opt.step()

    @torch.no_grad()
    def predict_q(self, X):
        X = torch.from_numpy(np.asarray(X, dtype=np.float32)).float().to(self.device)
        stack = [net(X).unsqueeze(0) for net in self.nets]
        return torch.cat(stack, 0).mean(0).cpu().numpy()

    def predict(self, X):
        return self.predict_q(X).argmax(1)


class TemperatureScaling:
    """Post-hoc T on logits. Fit T on a held-out (train) set via NLL."""
    def __init__(self, device="cpu"):
        self.device = device
        self.T = torch.nn.Parameter(torch.ones(1) * 1.5)
        self.base_net_predict = None  # set externally to a q()-producing model

    def fit(self, q_logits, y_action, steps=200, lr=0.05):
        # q_logits: (N, 5) numpy; y_action: (N,)
        logits = torch.from_numpy(np.asarray(q_logits, dtype=np.float32)).to(self.device)
        y = torch.from_numpy(np.asarray(y_action, dtype=np.int64)).to(self.device)
        opt = torch.optim.LBFGS([self.T], lr=lr, max_iter=steps)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(logits / self.T, y)
            loss.backward()
            return loss
        opt.step(closure)

    @torch.no_grad()
    def scale(self, q_logits):
        logits = torch.from_numpy(np.asarray(q_logits, dtype=np.float32)).to(self.device)
        return (logits / self.T).cpu().numpy()


def softmax(probs):
    z = np.asarray(probs, dtype=float)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
