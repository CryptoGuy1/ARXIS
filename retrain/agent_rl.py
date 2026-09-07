"""Dueling DQN model + RL training harness.

Mirrors the project's training notebook architecture exactly:
  DuelingDQN(input_dim=22, output_dim=5, dropout=0.15)
    features: Linear(22->256, LN, ReLU, Dropout) x2, Linear(256->128, LN, ReLU)
    value_stream: Linear(128->64, ReLU), Linear(64->1)
    adv_stream:   Linear(128->64, ReLU), Linear(64->5)
  Trained via the SAME loop as the notebook:
    PER + n-step(3) + gamma 0.99 + eps-decay + target net (TAU 0.005)
  Reward is injected via `reward_fn(gas_id, action, anomaly)` so Exp #1
  can sweep cost ratios without touching the training mechanics.
"""
import math
import random
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

STATE_DIM = 22
N_ACTIONS = 5


class DuelingDQN(nn.Module):
    def __init__(self, input_dim=STATE_DIM, output_dim=N_ACTIONS, dropout=0.15):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(),
        )
        self.value_stream = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.adv_stream = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, output_dim))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.value_stream[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.adv_stream[-1].weight, gain=0.01)

    def forward(self, x):
        f = self.features(x)
        v = self.value_stream(f)
        a = self.adv_stream(f)
        return v + (a - a.mean(dim=1, keepdim=True))

    @torch.no_grad()
    def predict_with_uncertainty(self, x, n_samples=20):
        self.train()
        qs = torch.stack([self(x) for _ in range(n_samples)], dim=0)
        self.eval()
        mean_q = qs.mean(0); std_q = qs.std(0)
        action = mean_q.argmax(1)
        top2 = mean_q.topk(2, dim=1).values
        gap = top2[:, 0] - top2[:, 1]
        conf = (gap / (std_q.mean(1) + 1e-6)).clamp(0, 10) / 10.0
        return mean_q, std_q, action, conf


# ----------------------------- PER buffer -----------------------------
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.data = [None] * capacity
        self._write = 0
        self.n_entries = 0
        self._max_pri = 1.0
        self._eps = 1e-6

    def total(self):
        return self.tree[1]

    def add(self, priority, data):
        self.data[self._write] = data
        self.update(self._write + self.capacity, priority)
        self._write = (self._write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, leaf_idx, priority):
        self.tree[leaf_idx] = priority
        idx = leaf_idx >> 1
        while idx >= 1:
            self.tree[idx] = self.tree[idx * 2] + self.tree[idx * 2 + 1]
            idx >>= 1

    def get(self, s):
        idx = 1
        while idx < self.capacity:
            left = idx * 2
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        return idx, self.tree[idx], self.data[idx - self.capacity]

    def batch_update(self, indices, priorities):
        for i, p in zip(indices, priorities):
            self.tree[i] = p
        parents = set(i >> 1 for i in indices)
        while parents:
            nxt = set()
            for i in parents:
                if i >= 1:
                    self.tree[i] = self.tree[i * 2] + self.tree[i * 2 + 1]
                    if i > 1:
                        nxt.add(i >> 1)
            parents = nxt


class PERBuffer:
    def __init__(self, capacity=100_000, alpha=0.6, beta_start=0.4, anneal_steps=300_000):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.anneal_steps = anneal_steps
        self._step = 0

    def push(self, s, a, r, ns, done):
        self.tree.add(self.tree._max_pri,
                      (np.array(s, dtype=np.float32), int(a), float(r),
                       np.array(ns, dtype=np.float32), float(done)))

    def sample(self, batch_size, device):
        self._step += 1
        beta = min(1.0, self.beta_start + self._step * (1 - self.beta_start) / self.anneal_steps)
        indices, priorities, batch = [], [], []
        seg = self.tree.total() / batch_size
        for i in range(batch_size):
            s = random.uniform(seg * i, seg * (i + 1))
            idx, pri, data = self.tree.get(s)
            if data is None:
                continue
            indices.append(idx); priorities.append(max(pri, self.tree._eps)); batch.append(data)
        probs = np.array(priorities) / (self.tree.total() + self.tree._eps)
        weights = (self.tree.n_entries * probs) ** (-beta)
        weights /= weights.max() + self.tree._eps
        s, a, r, ns, d = zip(*batch)
        return (torch.from_numpy(np.stack(s)).float().to(device),
                torch.tensor(a, dtype=torch.long).to(device),
                torch.tensor(r, dtype=torch.float).to(device),
                torch.from_numpy(np.stack(ns)).float().to(device),
                torch.tensor(d, dtype=torch.float).to(device),
                indices, torch.tensor(weights, dtype=torch.float).to(device))

    def update_priorities(self, indices, td_errors):
        pris = [(abs(float(e)) + self.tree._eps) ** self.alpha for e in td_errors]
        self.tree.batch_update(indices, pris)
        self.tree._max_pri = max(self.tree._max_pri, max(pris))

    def __len__(self):
        return self.tree.n_entries


class RLAgent:
    GAMMA = 0.99; TAU = 0.005; LR = 2e-4; BATCH_SIZE = 128; GRAD_CLIP = 10.0
    N_STEP = 3; BUF_CAPACITY = 100_000; WARMUP_STEPS = 2_000; LEARN_EVERY = 4
    EPS_START = 1.0; EPS_END = 0.05; EPS_DECAY_EP = 150

    def __init__(self, df, reward_fn, device="cpu", seed=42):
        self.df = df.reset_index(drop=True)
        self.reward_fn = reward_fn
        self.device = device
        self.seed = seed
        self._rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        torch.manual_seed(seed)
        self.online = DuelingDQN().to(device)
        self.target = DuelingDQN().to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(self.online.parameters(), lr=self.LR, eps=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-5)
        self.buffer = PERBuffer(self.BUF_CAPACITY)
        self._n_buf = deque(maxlen=self.N_STEP)
        self._total_steps = 0
        self.episode = 0
        self.idx = 0

    @property
    def epsilon(self):
        return max(self.EPS_END, self.EPS_START - (self.EPS_START - self.EPS_END) * self.episode / self.EPS_DECAY_EP)

    def _state(self, i):
        return np.array([self.df.iloc[i]["anomaly"]] +
                        [self.df.iloc[i][c] for c in
                         ["MQ2", "MQ3", "MQ5", "MQ6", "MQ7", "MQ8", "MQ135",
                          "dMQ2", "dMQ3", "dMQ5", "dMQ6", "dMQ7", "dMQ8", "dMQ135",
                          "sMQ2", "sMQ3", "sMQ5", "sMQ6", "sMQ7", "sMQ8", "sMQ135"]],
                        dtype=np.float32)

    def select_action(self, state):
        if self._rng.random() < self.epsilon:
            return self._rng.randrange(N_ACTIONS)
        with torch.no_grad():
            return int(self.online(torch.from_numpy(state).float().unsqueeze(0).to(self.device)).argmax().item())

    def _step_env(self, action):
        row = self.df.iloc[self.idx]
        gas_id = int(row["gas_id"])
        anomaly = float(row["anomaly"])
        state = self._state(self.idx)
        reward = self.reward_fn(gas_id, action, anomaly)
        self.idx += 1
        done = (self.idx >= len(self.df))
        next_state = self._state(self.idx) if not done else state.copy()
        return next_state, reward, done

    def push(self, s, a, r, ns, done):
        self._n_buf.append((np.array(s, dtype=np.float32), a, r))
        if len(self._n_buf) == self.N_STEP or done:
            R = sum(ri * (self.GAMMA ** i) for i, (_, _, ri) in enumerate(self._n_buf))
            s0, a0 = self._n_buf[0][0], self._n_buf[0][1]
            self.buffer.push(s0, a0, R, np.array(ns, dtype=np.float32), float(done))
            if done:
                self._n_buf.clear()

    def learn(self):
        self._total_steps += 1
        if self._total_steps % self.LEARN_EVERY != 0:
            return None
        if len(self.buffer) < self.WARMUP_STEPS:
            return None
        s, a, r, ns, d, indices, weights = self.buffer.sample(self.BATCH_SIZE, self.device)
        with torch.no_grad():
            next_a = self.online(ns).argmax(1, keepdim=True)
            next_q = self.target(ns).gather(1, next_a).squeeze(1)
            target_q = r + self.GAMMA ** self.N_STEP * next_q * (1 - d)
        current_q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        td_errors = (current_q - target_q).detach().cpu().numpy()
        loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction="none")).mean()
        self.optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.GRAD_CLIP)
        self.optimizer.step()
        self.buffer.update_priorities(indices, np.abs(td_errors))
        for tp, op in zip(self.target.parameters(), self.online.parameters()):
            tp.data.copy_(self.TAU * op.data + (1 - self.TAU) * tp.data)
        return float(loss.item())

    def train(self, episodes=200, eval_every=10):
        best_miss = 1.0
        best_sd = None
        for ep in range(episodes):
            self.episode = ep
            self.online.train()
            order = np.arange(len(self.df))
            self.np_rng.shuffle(order)
            self.df = self.df.iloc[order].reset_index(drop=True)
            self.idx = 0
            state = self._state(0)
            done = False
            while not done:
                action = self.select_action(state)
                ns, r, done = self._step_env(action)
                self.push(state, action, r, ns, done)
                self.learn()
                state = ns
            self.scheduler.step()
            # only evaluate for best-checkpoint periodically (cheap)
            if ep % eval_every == 0 or ep == episodes - 1:
                miss = self.evaluate_df(self.df)["miss_rate"]
                if miss <= best_miss:
                    best_miss = miss
                    best_sd = copy.deepcopy(self.online.state_dict())
        if best_sd is not None:
            self.online.load_state_dict(best_sd)
        self.online.eval()
        return best_miss

    @torch.no_grad()
    def predict_actions(self, X):
        self.online.eval()
        X = torch.from_numpy(np.asarray(X, dtype=np.float32)).float().to(self.device)
        q = self.online(X)
        return q.cpu().numpy()

    def evaluate_df(self, df):
        from retrain.metrics import evaluate_actions
        X = np.stack([self._state(i) for i in range(len(df))], 0)
        q = self.predict_actions(X)
        actions = q.argmax(1)
        gas = df["gas_id"].astype(int).values
        return evaluate_actions(gas, actions)
