"""Safety-relevant metric set (extends retrain/metrics.py).

Adds the quantities the accuracy-only protocol omits:
  * escalation adequacy   : P(action >= 3 | hazardous window)
  * under-escalation      : P(action in {1,2} | hazardous window)
  * false-alarm rate      : reported under BOTH denominators, because the
                            manuscript and retrain/metrics.py disagree:
                              fa_per_clean  = (a>=3 & NoGas) / n_NoGas   <- manuscript
                              fa_per_alert  = (a>=3 & NoGas) / n_alerts  <- metrics.py
  * mean action severity on hazardous windows
  * raw counts, so Clopper-Pearson bounds can be attached to every zero.
"""
import numpy as np
from scipy.stats import beta

CORRECT_ACTIONS = {0: [0], 1: [3], 2: [4], 3: [1, 2]}
DANGER_GAS_IDS = {1, 2}
ALARM_GRADE = 3


def cp_upper(k, n, conf=0.95):
    """One-sided exact Clopper-Pearson upper bound, as a FRACTION."""
    if n <= 0:
        return float("nan")
    k, n = int(k), int(n)
    if k == 0:
        return 1.0 - (1.0 - conf) ** (1.0 / n)
    if k >= n:
        return 1.0
    return float(beta.ppf(conf, k + 1, n - k))


def cp_interval(k, n, conf=0.95):
    if n <= 0:
        return (float("nan"), float("nan"))
    k, n = int(k), int(n)
    a = 1.0 - conf
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2.0, k, n - k + 1))
    hi = 1.0 if k >= n else float(beta.ppf(1.0 - a / 2.0, k + 1, n - k))
    return (lo, hi)


def evaluate_safety(gas, action):
    """Full metric set. gas: gas_id per row; action: action per row."""
    g = np.asarray(gas).astype(int)
    a = np.asarray(action).astype(int)
    n = len(g)

    correct = sum(int(ai) in CORRECT_ACTIONS.get(int(gi), []) for gi, ai in zip(g, a))
    danger = np.isin(g, list(DANGER_GAS_IDS))
    clean = (g == 0)
    n_d = int(danger.sum())
    n_c = int(clean.sum())

    miss_k = int((danger & (a == 0)).sum())
    esc_k = int((danger & (a >= ALARM_GRADE)).sum())
    under_k = int((danger & (a >= 1) & (a < ALARM_GRADE)).sum())
    fa_k = int((clean & (a >= ALARM_GRADE)).sum())
    n_alerts = int((a >= ALARM_GRADE).sum())

    return dict(
        n=n,
        decision_acc=correct / max(n, 1),
        n_danger=n_d,
        miss_k=miss_k,
        miss_rate=miss_k / max(n_d, 1),
        miss_upper95=cp_upper(miss_k, n_d),
        esc_k=esc_k,
        escalation=esc_k / max(n_d, 1),
        under_esc_k=under_k,
        under_escalation=under_k / max(n_d, 1),
        n_clean=n_c,
        fa_k=fa_k,
        fa_per_clean=fa_k / max(n_c, 1),
        fa_per_clean_upper95=cp_upper(fa_k, n_c),
        n_alerts=n_alerts,
        fa_per_alert=fa_k / max(n_alerts, 1),
        mean_severity_danger=float(a[danger].mean()) if n_d else float("nan"),
    )


def agg(rows, keys):
    """mean/std across seeds for the given keys, plus pooled counts."""
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        out[k + "_mean"] = float(v.mean())
        out[k + "_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    for k in ("miss_k", "n_danger", "fa_k", "n_clean", "esc_k", "under_esc_k"):
        out[k + "_pooled"] = int(sum(r[k] for r in rows))
    out["miss_upper95_pooled"] = cp_upper(out["miss_k_pooled"], out["n_danger_pooled"])
    out["fa_upper95_pooled"] = cp_upper(out["fa_k_pooled"], out["n_clean_pooled"])
    return out


AGG_KEYS = ["decision_acc", "miss_rate", "escalation", "under_escalation",
            "fa_per_clean", "mean_severity_danger"]


def alarms_per_hour(rate, hz=0.5):
    """Convert a per-clean-window false-alarm rate into alarm burden."""
    return rate * hz * 3600.0
