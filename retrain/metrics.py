"""Shared evaluation metrics (matches project notebook vocabulary)."""
import numpy as np
from scipy import stats

CORRECT_ACTIONS = {0: [0], 1: [3], 2: [4], 3: [1, 2]}
DANGER_GAS_IDS = {1, 2}  # Smoke, Mixture
N_ACTIONS = 5


def evaluate_actions(y_true_gas, y_pred_action):
    """y_true_gas: gas_id per row; y_pred_action: action per row."""
    y_true_gas = np.asarray(y_true_gas)
    y_pred_action = np.asarray(y_pred_action)
    correct = sum(int(a) in CORRECT_ACTIONS.get(int(g), [])
                  for g, a in zip(y_true_gas, y_pred_action))
    decision_acc = correct / len(y_true_gas)

    danger_total = danger_missed = 0
    false_alarms = total_alerts = 0
    for g, a in zip(y_true_gas, y_pred_action):
        g = int(g)
        a = int(a)
        if g in DANGER_GAS_IDS:
            danger_total += 1
            if a == 0:
                danger_missed += 1
        if a >= 3:
            total_alerts += 1
            if g == 0:
                false_alarms += 1
    miss_rate = danger_missed / max(danger_total, 1)
    false_alarm_rate = false_alarms / max(total_alerts, 1)
    return dict(decision_acc=decision_acc, miss_rate=miss_rate,
                false_alarm_rate=false_alarm_rate)


def expected_calibration_error(probs, y_correct, n_bins=10):
    """probs: (N, n_actions) softmax; y_correct: bool correct-action flag."""
    probs = np.asarray(probs)
    y_correct = np.asarray(y_correct, dtype=bool)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc = y_correct[mask].mean()
        avg_conf = conf[mask].mean()
        ece += abs(avg_conf - acc) * mask.sum()
    return float(ece / max(len(probs), 1))


def paired_significance(a_vals, b_vals, alternative="greater"):
    """Paired test (e.g. miss-rate reduction) across seeds.
    Returns (stat, p). Uses Wilcoxon signed-rank."""
    a = np.asarray(a_vals, dtype=float)
    b = np.asarray(b_vals, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return (float("nan"), 1.0)
    # Guard on actual equality, NOT on "len(set(a-b))==1" — a constant
    # non-zero difference is the strongest evidence at this sample size and
    # must NOT be reported as (nan, 1.0).
    if np.allclose(a, b):
        return (float("nan"), 1.0)
    try:
        stat, p = stats.wilcoxon(a, b, alternative=alternative)
    except ValueError:
        return (float("nan"), 1.0)
    return (float(stat), float(p))


def mean_std(xs):
    xs = np.asarray(xs, dtype=float)
    return float(xs.mean()), float(xs.std(ddof=1) if len(xs) > 1 else 0.0)


def clopper_pearson_upper(k, n, conf=0.95):
    """One-sided upper confidence bound on a miss/error rate.

    Args:
        k: number of "failures" observed (e.g. danger-misses = 0).
        n: number of opportunities (e.g. danger rows in the test set).
        conf: confidence level (default 0.95).

    Returns:
        Upper bound on the true rate with confidence `conf`
        (i.e. P(true_rate <= bound) >= conf), using the exact
        Clopper-Pearson (beta) interval.

    Returned as a FRACTION in [0, 1] (multiply by 100 for a percentage).

    Method:
        U = BetaInv(conf; k + 1, n - k)
        with the closed form U = 1 - (1-conf)**(1/n) when k == 0.

    FIXED (post-audit): the previous implementation used
    `beta.ppf(1 - conf/2, k+1, n-k)` -- the 0.525 quantile at conf=0.95.
    That is close to the posterior MEDIAN, not an upper bound, and it
    UNDERSTATED the true bound by roughly 1.7-2x for small k
    (k=6, n=632: gave 1.08% where the correct bound is 1.87%).  Understating
    an upper bound on a danger-miss rate is a safety-relevant error.  The
    k == 0 branch was always correct and is unchanged, so every zero-miss
    bound already published (<=0.47% per seed, <=0.095% pooled, <=0.19% for
    the LOCO hold-out) still stands.
    """
    if n <= 0:
        return float("nan")
    k = int(k)
    n = int(n)
    if k < 0 or k > n:
        return float("nan")
    if k == 0:
        # closed form for zero observed failures
        return 1.0 - (1.0 - conf) ** (1.0 / n)
    if k == n:
        return 1.0
    from scipy.stats import beta
    return float(beta.ppf(conf, k + 1, n - k))


def clopper_pearson_interval(k, n, conf=0.95):
    """Exact two-sided Clopper-Pearson interval, returned as (lower, upper).

    Use this to put error bars on an observed non-zero rate; use
    `clopper_pearson_upper` when the claim is one-sided ("the miss rate is at
    most U").  Both are fractions in [0, 1].
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    from scipy.stats import beta
    k = int(k)
    n = int(n)
    alpha = 1.0 - conf
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return (lo, hi)


def miss_bound_row(miss_rate, n_danger, model=None, conf=0.95):
    """Turn an observed miss RATE + denominator into a report row.

    Returns dict(model, danger_rows, observed_misses, miss_rate, upper_pct)
    with `upper_pct` as a PERCENTAGE, so a caller cannot print a fraction
    under a column header that says "pct" -- the units ambiguity the audit
    flagged in exp3_clopper_pearson.csv.  Always carry `model`: a bound is a
    property of a (model, held-out class) pair, not of the class alone.
    """
    n_danger = int(n_danger)
    k = int(round(float(miss_rate) * n_danger))
    return dict(model=model,
                danger_rows=n_danger,
                observed_misses=k,
                miss_rate=float(miss_rate),
                upper_pct=100.0 * clopper_pearson_upper(k, n_danger, conf))
