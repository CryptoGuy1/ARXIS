"""Baselines a reviewer will ask for and the original study omitted.

  * KNN            - k-nearest neighbours on the 22-feature state.
  * ThresholdRule  - a depth-3 decision tree on the seven RAW current-sensor
                     channels only. This is the deployable analogue of the
                     hand-tuned threshold logic fielded in practice, and unlike
                     the rule-oracle it does NOT consume the true label.
  * CusumDetector  - classical sequential change detection (Page 1954) on the
                     anomaly score. Two-class by construction: it decides
                     hazard / no-hazard, and cannot separate Smoke from
                     Mixture, so its decision accuracy is NOT comparable with
                     the multiclass models. Miss rate and escalation are.
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# column layout of the 22-dim state: [anomaly, current(7), delta(7), std(7)]
IDX_ANOM = 0
IDX_CURRENT = slice(1, 8)


class KNN:
    def __init__(self, seed=42, k=5):
        self.clf = KNeighborsClassifier(n_neighbors=k)

    def fit(self, X, y_action):
        self.clf.fit(X, y_action)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


class ThresholdRule:
    """Depth-3 tree on the seven raw current-sensor channels (interpretable,
    deployable, no label at inference). Incumbent-practice comparator."""

    def __init__(self, seed=42, depth=3):
        self.clf = DecisionTreeClassifier(max_depth=depth, random_state=seed)

    def fit(self, X, y_action):
        self.clf.fit(np.asarray(X)[:, IDX_CURRENT], y_action)

    def predict(self, X):
        return self.clf.predict(np.asarray(X)[:, IDX_CURRENT])

    def predict_proba(self, X):
        return self.clf.predict_proba(np.asarray(X)[:, IDX_CURRENT])


class CusumDetector:
    """One-sided CUSUM on the anomaly score.

        S_t = max(0, S_{t-1} + (x_t - mu0 - k))
        alarm when S_t > h

    mu0 and sigma are estimated from the NoGas TRAINING windows only; the
    drift parameter k is 0.5*sigma and h is chosen on the training partition
    to hit a target false-alarm rate on clean windows. Output is mapped onto
    the action ladder as: no alarm -> 0 (Monitor), alarm -> 3 (Raise Alarm).
    """

    def __init__(self, seed=42, k_sigma=0.5, target_fpr=0.01):
        self.k_sigma = k_sigma
        self.target_fpr = target_fpr

    def fit(self, X, y_action, g_class=None):
        x = np.asarray(X)[:, IDX_ANOM].astype(float)
        g = np.asarray(g_class) if g_class is not None else None
        base = x[g == 0] if g is not None and (g == 0).any() else x
        self.mu0 = float(base.mean())
        self.sigma = float(base.std() + 1e-9)
        self.k = self.k_sigma * self.sigma
        s = self._run(base)
        # threshold at the (1 - target_fpr) quantile of the clean CUSUM path
        self.h = float(np.quantile(s, 1.0 - self.target_fpr))
        return self

    def _run(self, x, groups=None):
        """Accumulate S_t, resetting to zero at the start of each evaluation block.

        Reviewer comment (Section 7): a sequential detector needs explicit reset
        semantics. S_t is zero at the first window of every call, and, when the
        class labels are supplied, at every change of class block. Without the
        reset the statistic accumulated across a block boundary, so the class
        following a hazardous block inherited evidence that did not belong to it.
        """
        s = np.zeros(len(x), dtype=float)
        acc = 0.0
        prev = None
        for i, xi in enumerate(x):
            if groups is not None and groups[i] != prev:
                acc = 0.0            # new block: no carry-over across the boundary
                prev = groups[i]
            acc = max(0.0, acc + (xi - self.mu0 - self.k))
            s[i] = acc
        return s

    def predict(self, X, groups=None):
        x = np.asarray(X)[:, IDX_ANOM].astype(float)
        g = np.asarray(groups) if groups is not None else None
        s = self._run(x, g)
        return np.where(s > self.h, 3, 0).astype(int)
