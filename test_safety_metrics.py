"""Unit tests for the metric implementation every table in the paper runs through.

Plan item A11: "publish one authoritative safety-metric implementation with unit
tests", with a named acceptance test, that the run fails if the three hazardous-
window rates do not sum to one or if the false-alarm denominator admits Perfume.
Both are checked below, along with the partition discipline the protocol claims.

These are cheap and they guard the failures this project has actually had. Two
metric defects reached a released version of this manuscript: a false-alarm rate
divided by the number of alerts rather than by the number of clean windows, and a
one-sided embargo that left windows adjacent to the test band in training. Each
is now a test that fails rather than a paragraph in a change log.

    python3 test_safety_metrics.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrain.safety_metrics import (evaluate_safety, cp_upper, alarms_per_hour,
                                    CORRECT_ACTIONS, DANGER_GAS_IDS, ALARM_GRADE)
from retrain import raw_pipeline as rp

PASS, FAIL = [], []


def ok(label, cond, detail=""):
    (PASS if cond else FAIL).append(f"{label}{(': ' + detail) if detail else ''}")


def close(label, got, want, tol=1e-9):
    ok(label, abs(float(got) - float(want)) <= tol, f"got {got!r}, want {want!r}")


# ---------------------------------------------------------------- the action map
ok("Perfume is the only class with two acceptable actions",
   [g for g, a in CORRECT_ACTIONS.items() if len(a) > 1] == [3])
ok("the hazardous set is Smoke and Mixture", DANGER_GAS_IDS == {1, 2})
ok("the alarm boundary is a >= 3", ALARM_GRADE == 3)
ok("action 2 is never a single acceptable action on its own",
   not any(a == [2] for a in CORRECT_ACTIONS.values()))

# ------------------------------------------------- the three hazardous rates partition
# Every hazardous window lands in exactly one of: passive (a = 0), sub-alarm
# (a in {1,2}), alarm-grade (a >= 3). The three rates must therefore sum to 1.
rng = np.random.default_rng(0)
for trial in range(200):
    n = int(rng.integers(20, 400))
    gas = rng.integers(0, 4, size=n)
    act = rng.integers(0, 5, size=n)
    e = evaluate_safety(gas, act)
    if e["n_danger"] == 0:
        continue
    total = e["miss_rate"] + e["under_escalation"] + e["escalation"]
    if abs(total - 1.0) > 1e-9:
        FAIL.append(f"hazardous rates sum to {total!r} on trial {trial}, not 1")
        break
else:
    PASS.append("miss + under-escalation + escalation = 1 on every hazardous partition (200 trials)")

# ------------------------------------------------------------ metric denominators
gas = np.array([0, 0, 0, 1, 1, 2, 3, 3])          # 3 clean, 3 hazardous, 2 Perfume
act = np.array([4, 0, 0, 0, 3, 4, 1, 4])
e = evaluate_safety(gas, act)
close("hazardous denominator counts only Smoke and Mixture", e["n_danger"], 3)
close("clean denominator counts only NoGas", e["n_clean"], 3)
close("missed-hazard rate divides by hazardous windows", e["miss_rate"], 1 / 3)
close("false-alarm rate divides by clean windows, not by alerts", e["fa_per_clean"], 1 / 3)
ok("the false-alarm denominator excludes Perfume",
   e["n_clean"] == 3, "a Perfume window must not enter the clean denominator")
# the Perfume window sent to action 4 is a wrong action but not a false alarm
close("a high-severity action on Perfume is not counted as a false alarm",
      e["fa_k"], 1)
close("escalation counts a >= 3 on hazardous windows", e["esc_k"], 2)
close("under-escalation counts a in {1,2} on hazardous windows", e["under_esc_k"], 0)

# ------------------------------------------------------------ boundary handling
gas_b = np.array([1, 1, 1, 1])
close("a = 2 on a hazardous window is under-escalation, not escalation",
      evaluate_safety(gas_b, np.array([2, 2, 2, 2]))["under_escalation"], 1.0)
close("a = 3 on a hazardous window is escalation",
      evaluate_safety(gas_b, np.array([3, 3, 3, 3]))["escalation"], 1.0)
close("a = 0 on a hazardous window is a miss",
      evaluate_safety(gas_b, np.array([0, 0, 0, 0]))["miss_rate"], 1.0)

# ------------------------------------------------------------ the binomial bound
close("zero-count bound uses the closed form 1 - (1-c)^(1/n)",
      cp_upper(0, 632), 1 - 0.05 ** (1 / 632))
close("zero-count bound on 632 hazardous windows", 100 * cp_upper(0, 632), 0.4729, 6e-4)
close("zero-count bound on 316 clean windows", 100 * cp_upper(0, 316), 0.9435, 6e-4)
close("zero-count bound on 1,581 held-out-class windows", 100 * cp_upper(0, 1581), 0.1893, 6e-4)
close("zero-count bound on 31 disjoint-support windows", 100 * cp_upper(0, 31), 9.2114, 6e-4)
close("zero-count bound on 15 disjoint-support windows", 100 * cp_upper(0, 15), 18.1036, 6e-3)
# the nonzero branch is the one the averaged-pseudocount defect lived in
close("nonzero bound at the worst Smoke partition", 100 * cp_upper(234, 1581), 16.3498, 6e-3)
close("nonzero bound at the worst Mixture partition", 100 * cp_upper(988, 1581), 64.5081, 6e-3)
ok("the bound is monotone in the observed count",
   all(cp_upper(k, 1581) < cp_upper(k + 1, 1581) for k in range(0, 400, 37)))
ok("the bound is at least the observed rate",
   all(cp_upper(k, 1581) >= k / 1581 for k in range(0, 1500, 97)))

# ------------------------------------------------------------ burden conversion
close("burden conversion is rate x 1,000 x windows per hour / 1,000",
      alarms_per_hour(0.1284, hz=0.5), 0.1284 * 1800.0, 1e-6)

# ------------------------------------------------------------ partition discipline
# The protocol claims a two-sided 20-window embargo. A one-sided embargo shipped
# in an earlier version, so the separation is measured rather than assumed.
try:
    am, asc = rp.build_anomaly_model()
    ds = rp.build_dataset(am, asc, cache_path=os.path.join("retrain", "results", "real_features.csv"))
    for seed in (42, 1337, 7, 2024, 99, 1000, 1007):
        tr, te = rp.block_wise_holdout(ds, seed=seed)
        for cls in sorted(ds["gas_id"].unique()):
            a = np.sort(tr[tr.gas_id == cls]["win_id"].to_numpy())
            b = np.sort(te[te.gas_id == cls]["win_id"].to_numpy())
            if len(a) == 0 or len(b) == 0:
                continue
            below = a[a < b.min()]
            above = a[a > b.max()]
            if len(below) and b.min() - below.max() <= rp.WINDOW_SIZE:
                FAIL.append(f"embargo below the test band is {b.min()-below.max()} windows "
                            f"at seed {seed}, class {cls}; needs > {rp.WINDOW_SIZE}")
            if len(above) and above.min() - b.max() <= rp.WINDOW_SIZE:
                FAIL.append(f"embargo above the test band is {above.min()-b.max()} windows "
                            f"at seed {seed}, class {cls}; needs > {rp.WINDOW_SIZE}")
    PASS.append("the embargo is two-sided and exceeds one window length on every seed and class")
    n_tr, n_te = len(tr), len(te)
    ok("the partition is 4,900 train and 1,264 test", (n_tr, n_te) == (4900, 1264), f"{n_tr}/{n_te}")
    ok("no window appears in both partitions",
       set(tr["win_id"]).isdisjoint(set(te["win_id"])))
except Exception as exc:                                    # pragma: no cover
    FAIL.append(f"partition checks could not run: {exc}")

# ------------------------------------------------------------ report
print(f"{len(PASS) + len(FAIL)} assertions, {len(FAIL)} failure(s)\n")
for p in PASS:
    print("  ok  ", p)
for f in FAIL:
    print("  FAIL", f)
sys.exit(1 if FAIL else 0)
