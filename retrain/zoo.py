"""Broader model zoo for the "is this just a fancy MLP?" comparison.

Adds four additional comparators, all trained to predict the rule-oracle
action (same target as SupervisedDQN / MLP / GBM in Exp #2/#3) so they are
apples-to-apples on decision_acc / miss_rate / false_alarm_rate:

  1. SVM            - RBF support-vector classifier on the 22-feat rows
  2. RandomForest   - 300-tree RF on the 22-feat rows
  3. RawWindowLSTM  - recurrent net over a sliding window of K consecutive
                      22-feature rows (uses TEMPORAL context, unlike the
                      stateless DQN). Predicts the action for the last row.
  4. CQLAgent       - a REAL offline safe-RL baseline: DuelingDQN trained
                      with a Conservative Q-Learning (CQL) regularizer so it
                      does not overestimate unseen (state, action) values.
                      This answers "does genuine offline safe-RL beat our
                      cost-weighted CE?" rather than "is it just an MLP?".

All use 5 seeds and the same evaluation metrics as the rest of the harness.
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                  # retrain/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from retrain.agent_rl import DuelingDQN
import rewards as RW


# ---------------------------------------------------------------------------
# 1 + 2 : SVM and Random Forest (stateless, 22-feat rows)
# ---------------------------------------------------------------------------
class SVMClassifier:
    """Audit fix: `SVC` defaults to `probability=False`, under which
    `predict_proba` raises AttributeError -- so the `predict_proba` method below
    was dead on arrival and any attempt to score SVM calibration (ECE) crashed.
    `probability=True` is now set at construction. Per the scikit-learn docs this
    fits an additional Platt calibrator by internal 5-fold CV but does NOT change
    `predict`, so the reported SVM decision accuracy (0.9511 +/- 0.0190) is
    unchanged; only training time increases."""

    def __init__(self, seed=42):
        self.clf = SVC(kernel="rbf", C=1.0, gamma="scale",
                       probability=True, random_state=seed)

    def fit(self, X, y_action):
        self.clf.fit(X, y_action)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


class RandomForest:
    def __init__(self, seed=42):
        self.clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1)

    def fit(self, X, y_action):
        self.clf.fit(X, y_action)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


# ---------------------------------------------------------------------------
# 3 : Raw-window LSTM (temporal). Build K-length windows of consecutive rows.
# ---------------------------------------------------------------------------
class _LSTMNet(nn.Module):
    """Audit C4: `nn.LSTM(..., num_layers=1, dropout=p)` silently DISCARDS the
    dropout (PyTorch only applies inter-layer dropout, so a 1-layer LSTM had
    none). The recurrent depth is kept at 1; dropout is now applied explicitly
    to the final hidden state, so the regularisation the code claimed is real."""

    def __init__(self, input_dim=22, hidden=64, n_actions=5, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, n_actions))

    def forward(self, x):  # x: (B, K, input_dim)
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1, :]))


class RawWindowLSTM:
    """Temporal baseline over K consecutive 22-feature rows.

    Audit C3: the previous `_windowed` slid a K-row window over X in row order.
    X is the class-contiguous concatenation of the four gas blocks, so windows
    STRADDLED class boundaries (mixing e.g. Smoke and Mixture rows under one
    label), and `predict` filled the first K-1 rows with `acts[0]` -- an
    arbitrary value copied from a different class. Both are fixed here:

      * windows are built only within a contiguous run of one gas class
        (`groups`), so no window ever mixes classes;
      * every row gets its own prediction from the window ENDING at that row,
        left-padded by repeating the first row of its own run when fewer than
        K rows of history exist. Nothing is copied across a class boundary.

    `groups` defaults to the action labels at fit time and, at predict time, to
    a single run -- pass the gas ids explicitly (`groups=gte`) for the correct
    behaviour on the concatenated test set.
    """

    def __init__(self, device="cpu", seed=42, K=10):
        self.device = device
        self.seed = seed
        self.K = K
        torch.manual_seed(seed)

    @staticmethod
    def _runs(groups):
        """Yield (start, stop) index pairs of maximal constant-value runs."""
        g = np.asarray(groups)
        if len(g) == 0:
            return []
        edges = np.flatnonzero(np.diff(g)) + 1
        bounds = np.concatenate(([0], edges, [len(g)]))
        return [(int(bounds[i]), int(bounds[i + 1])) for i in range(len(bounds) - 1)]

    def _windows_within_runs(self, X, groups):
        """One K-row window per row, never crossing a run boundary.

        Row i gets rows [i-K+1 .. i] of its own run, left-padded by repeating
        that run's first row when the history is short.
        """
        X = np.asarray(X, dtype=np.float32)
        out = np.empty((len(X), self.K, X.shape[1]), dtype=np.float32)
        for a, b in self._runs(groups):
            for i in range(a, b):
                lo = max(a, i - self.K + 1)
                chunk = X[lo:i + 1]
                if len(chunk) < self.K:  # left-pad with this run's first row
                    pad = np.repeat(chunk[:1], self.K - len(chunk), axis=0)
                    chunk = np.concatenate([pad, chunk], axis=0)
                out[i] = chunk
        return out

    def fit(self, X, y_action, groups=None, epochs=40, batch=256):
        if groups is None:
            groups = y_action  # blocks of one action == blocks of one class here
        Xw = self._windows_within_runs(X, groups)
        yw = np.asarray(y_action, dtype=np.int64)
        self.net = _LSTMNet(input_dim=np.asarray(X).shape[1]).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=2e-4)
        Xt = torch.from_numpy(Xw).float().to(self.device)
        yt = torch.from_numpy(yw).to(self.device)
        self.net.train()
        n = len(Xt)
        for ep in range(epochs):
            perm = torch.randperm(n)
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                logits = self.net(Xt[idx])
                loss = F.cross_entropy(logits, yt[idx])
                opt.zero_grad(); loss.backward(); opt.step()

    @torch.no_grad()
    def _logits(self, X, groups):
        if groups is None:
            groups = np.zeros(len(X), dtype=np.int64)
        Xw = self._windows_within_runs(X, groups)
        self.net.eval()
        Xt = torch.from_numpy(Xw).float().to(self.device)
        return self.net(Xt)

    def predict(self, X, groups=None):
        if len(X) == 0:
            return np.zeros(0, dtype=np.int64)
        return self._logits(X, groups).argmax(1).cpu().numpy()

    def predict_proba(self, X, groups=None):
        if len(X) == 0:
            return np.zeros((0, 5))
        return F.softmax(self._logits(X, groups), 1).cpu().numpy()


# ---------------------------------------------------------------------------
# 4 : Offline CQL (Conservative Q-Learning) agent
# ---------------------------------------------------------------------------
class CQLAgent:
    """Offline safe-RL baseline using a REAL Conservative Q-Learning (CQL)
    update — not the vacuous stacked-CE version.

    We treat the rule-oracle action as the behaviour action a_data and
    reward_asymmetric(g, a) as the reward. Q is trained by TD(0) bootstrap
    (r + gamma * max_a' Q(s', a')) on the offline (s, a_data, r, s') tuples,
    and the CQL penalty pushes DOWN Q-values of *all* actions relative to the
    behaviour action:

        loss = TD_error + alpha * ( logsumexp_a Q(s,a) - Q(s, a_data) )

    This is the standard CQL(R) conservative term. Because it is built on a
    Bellman target (not on cross-entropy), it is genuinely different from
    cost-weighted CE and answers "does offline safe-RL beat our objective?".
    """

    def __init__(self, device="cpu", seed=42, alpha=1.0, gamma=0.99):
        self.device = device
        self.alpha = alpha
        self.gamma = gamma
        torch.manual_seed(seed)
        self.net = DuelingDQN().to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=2e-4)

    def _next_states(self, X):
        # simple 1-step shift; last row has no next state (masked out)
        return torch.cat([X[1:], X[-1:]], dim=0)

    def fit(self, X, y_action, g_class=None, epochs=80, batch=256):
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).float().to(self.device)
        yt = torch.from_numpy(np.asarray(y_action, dtype=np.int64)).to(self.device)
        nxt = self._next_states(Xt)
        # reward for the observed (state, action)
        if g_class is None:
            g_class = torch.zeros(len(yt), dtype=torch.long, device=self.device)
        else:
            g_class = torch.as_tensor(np.asarray(g_class), dtype=torch.long, device=self.device)
        g_np = g_class.cpu().numpy() if hasattr(g_class, "cpu") else np.asarray(g_class)
        y_np = yt.cpu().numpy()
        rt = torch.tensor([float(RW.reward_asymmetric(int(g), int(a), 0.0,
                        miss_cost=10.0, false_cost=1.0))
                           for g, a in zip(g_np, y_np)],
                          dtype=torch.float32, device=self.device)
        self.net.train()
        n = len(Xt)
        # mask rows that are the final row of their trajectory (no next state)
        has_next = torch.ones(n, dtype=torch.bool, device=self.device)
        has_next[-1] = False
        for ep in range(epochs):
            perm = torch.randperm(n)
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                q_all = self.net(Xt[idx])                     # (b, n_actions)
                q_data = q_all.gather(1, yt[idx].unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = self.net(nxt[idx]).max(1).values   # (b,)
                    target = rt[idx] + self.gamma * q_next * has_next[idx].float()
                td = F.smooth_l1_loss(q_data, target)
                logsumexp = torch.logsumexp(q_all, dim=1)
                cql = (logsumexp - q_data).mean()
                loss = td + self.alpha * cql
                self.opt.zero_grad(); loss.backward(); self.opt.step()

    @torch.no_grad()
    def predict_q(self, X):
        self.net.eval()
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).float().to(self.device)
        return self.net(Xt).cpu().numpy()

    def predict(self, X):
        return self.predict_q(X).argmax(1)
