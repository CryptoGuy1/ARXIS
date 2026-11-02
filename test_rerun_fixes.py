"""Regression tests for the code fixes made before the November 2026 rerun.

1. Training histories and bootstrapped successors never cross a gap in the
   recording (review comments 62 and 75): segment ids, CQL terminal masking,
   LSTM history construction, and the fit helpers' refusal to run without
   segment ids.
2. The primary gradient-boosting comparator weights hazardous rows, not clean
   air (review comments 158 and 216).

    python3 test_rerun_fixes.py
"""
import sys

import numpy as np
import pandas as pd

from retrain import raw_pipeline as rp
from retrain import rewards as RW
from retrain.comparators import (HazardWeightedGBM, CleanAirWeightedGBM,
                                 HAZARD_ACTION_WEIGHT, CLEAN_AIR_ACTION_WEIGHT)
from retrain.zoo import CQLAgent, RawWindowLSTM

N = [0]
FAIL = []


def check(label, ok, detail=""):
    N[0] += 1
    print(("  ok   " if ok else "  FAIL ") + label + (f": {detail}" if detail else ""))
    if not ok:
        FAIL.append(label)


# ---------------------------------------------------------------- segments
seg = rp.segment_ids(pd.DataFrame({"win_id": [5, 6, 7, 20, 21, 40]}))
check("segment ids break at every gap in win_id", seg.tolist() == [0, 0, 0, 1, 1, 2], str(seg.tolist()))
check("segment ids of an empty frame are empty", len(rp.segment_ids(pd.DataFrame({"win_id": []}))) == 0)

# a synthetic corpus laid out like the real one: four class blocks of
# consecutive window ids, with a gap of discarded boundary windows between them
rows, wid = [], 0
for label in rp.GAS_ORDER:
    for _ in range(300):
        rows.append(dict(win_id=wid, label=label, gas_id=rp.GAS_MAP[label]))
        wid += 1
    wid += 19
df = pd.DataFrame(rows)
for s in (42, 1337, 7, 2024, 99):
    tr, te = rp.block_wise_holdout(df, seed=s)
    tr_seg, te_seg = rp.segment_ids(tr), rp.segment_ids(te)
    per_class_tr = tr.assign(seg=tr_seg).groupby("gas_id")["seg"].nunique()
    per_class_te = te.assign(seg=te_seg).groupby("gas_id")["seg"].nunique()
    check(f"seed {s}: each class trains on at most two segments", bool((per_class_tr <= 2).all()),
          str(per_class_tr.to_dict()))
    check(f"seed {s}: each class tests on one contiguous segment", bool((per_class_te == 1).all()),
          str(per_class_te.to_dict()))
    one_class = all(tr.loc[tr_seg == k, "gas_id"].nunique() == 1 for k in np.unique(tr_seg))
    check(f"seed {s}: no training segment mixes classes", one_class)

# ---------------------------------------------------------------- CQL successors
hn = CQLAgent._has_next(6, segments=np.array([0, 0, 1, 1, 1, 2]))
check("CQL: last row of every segment is terminal",
      hn.tolist() == [True, False, True, True, False, False], str(hn.tolist()))
hn = CQLAgent._has_next(5, g_class=np.array([0, 0, 0, 1, 1]))
check("CQL: without segments, class boundaries are still terminal",
      hn.tolist() == [True, True, False, True, False], str(hn.tolist()))

# end to end on the synthetic layout: no successor crosses the removed block
tr, _ = rp.block_wise_holdout(df, seed=42)
segs = rp.segment_ids(tr)
hn = CQLAgent._has_next(len(tr), segments=segs)
ids = tr["win_id"].to_numpy()
bad = int((hn[:-1] & (np.diff(ids) != 1)).sum())
check("CQL: every live successor is the next window in time", bad == 0, f"{bad} crossing links")

# ---------------------------------------------------------------- LSTM histories
X = np.arange(7, dtype=np.float32).reshape(-1, 1)
w = RawWindowLSTM(K=3)._windows_within_runs(X, np.array([0, 0, 0, 1, 1, 1, 1]))
check("LSTM: a history never reaches into the previous segment",
      w[3, :, 0].tolist() == [3, 3, 3] and w[4, :, 0].tolist() == [3, 3, 4]
      and w[6, :, 0].tolist() == [4, 5, 6], str(w[:, :, 0].tolist()))

# ---------------------------------------------------------------- fit helpers refuse
from retrain.run_final import fit_pred
from retrain.run_all_v2 import fit_predict
Xs = np.random.default_rng(0).normal(size=(8, 22)).astype(np.float32)
ys = np.array([0, 0, 1, 1, 3, 3, 4, 4]); gs = np.array([0, 0, 3, 3, 1, 1, 2, 2])
for helper, nm in ((fit_pred, "run_final.fit_pred"), (fit_predict, "run_all_v2.fit_predict")):
    for model in (CQLAgent(seed=0), RawWindowLSTM(seed=0, K=3)):
        try:
            helper(model, Xs, ys, gs, Xs, gs, dict(epochs=1))
            ok = False
        except ValueError:
            ok = True
        check(f"{nm} refuses to fit {type(model).__name__} without segment ids", ok)

# ---------------------------------------------------------------- GBM weights
acts = [RW.rule_oracle_action(rp.GAS_MAP[c]) for c in ("NoGas", "Perfume", "Smoke", "Mixture")]
hw = HazardWeightedGBM().sample_weight(acts).tolist()
cw = CleanAirWeightedGBM().sample_weight(acts).tolist()
check("hazard-weighted GBM: NoGas 1.0, Perfume 1.0, Smoke 2.5, Mixture 2.5", hw == [1.0, 1.0, 2.5, 2.5], str(hw))
check("clean-air GBM (provenance): NoGas 3.0, Perfume 1.0, Smoke 1.2, Mixture 1.2", cw == [3.0, 1.0, 1.2, 1.2], str(cw))
gas = np.array([rp.GAS_MAP[c] for c in rp.GAS_ORDER] * 3)
by_gas = np.where(np.isin(gas, (rp.GAS_MAP["Smoke"], rp.GAS_MAP["Mixture"])), 2.5, 1.0)
by_act = HazardWeightedGBM().sample_weight([RW.rule_oracle_action(int(g)) for g in gas])
check("action-keyed hazard weights equal gas-keyed weights on training rows",
      np.allclose(by_gas, by_act))
check("the hazard weighting favors hazards by 2.5 to 1",
      HAZARD_ACTION_WEIGHT[3] / HAZARD_ACTION_WEIGHT[0] == 2.5
      and CLEAN_AIR_ACTION_WEIGHT[0] / CLEAN_AIR_ACTION_WEIGHT[3] == 2.5)

print(f"\n{N[0]} checks run, {len(FAIL)} failure(s)")
sys.exit(1 if FAIL else 0)
