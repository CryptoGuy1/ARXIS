"""What does the proposed metric set cost to compute, against accuracy alone?

Supplementary Section S8 claims the metric set is inexpensive to adopt. That is
the kind of claim that should be measured rather than asserted, and this driver
measures it: the wall-clock cost of `evaluate_safety`, which returns the whole
metric set of Table 5 including the independence-reference bounds, against the
cost of computing decision accuracy alone over the same predicted actions.

Both quantities are pure arithmetic over arrays the evaluation already holds, so
the comparison is a ratio of two small numbers and the ratio is the reportable
part. Timing is machine-dependent; the manifest written by `make_manifest.py`
records the host this was measured on.

The test partition is the real one, built by the same pipeline used everywhere
else, so the window count matches the manuscript rather than being a synthetic
array of convenient length. Actions are drawn from the deterministic oracle map
with a fixed fraction perturbed, because the metric cost depends on the array
sizes and not on which model produced the actions.

Writes results_v2/v3_metric_cost.json.

    python3 -m retrain.run_metric_cost
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrain import raw_pipeline as rp
from retrain.run_all_v2 import OUT, CACHE
from retrain import rewards as RW
from retrain.safety_metrics import evaluate_safety, CORRECT_ACTIONS

REPEATS = 200
WARMUP = 20
SEED = 42


def accuracy_only(gas, action):
    """Decision accuracy alone: what a conventional evaluation reports."""
    g = np.asarray(gas).astype(int)
    a = np.asarray(action).astype(int)
    correct = sum(int(ai) in CORRECT_ACTIONS.get(int(gi), []) for gi, ai in zip(g, a))
    return correct / max(len(g), 1)


def timeit(fn, *args, repeats=REPEATS, warmup=WARMUP):
    for _ in range(warmup):
        fn(*args)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(*args)
    return 1000.0 * (time.perf_counter() - t0) / repeats


def main():
    am, asc = rp.build_anomaly_model()
    ds = rp.build_dataset(am, asc, cache_path=CACHE)
    tr, te = rp.block_wise_holdout(ds, seed=SEED)
    tr, te, _, _ = rp.prepare(tr, te, anomaly_seed=SEED)
    _, gte = rp.to_arrays(te)

    rng = np.random.default_rng(SEED)
    acts = np.array([RW.rule_oracle_action(int(g)) for g in gte])
    flip = rng.random(len(acts)) < 0.05          # a realistic share of wrong actions
    acts[flip] = rng.integers(0, 5, size=int(flip.sum()))

    ms_acc = timeit(accuracy_only, gte, acts)
    ms_full = timeit(evaluate_safety, gte, acts)

    out = dict(n_windows=int(len(gte)),
               ms_accuracy_only=ms_acc,
               ms_full_metric_set=ms_full,
               ratio=ms_full / ms_acc,
               repeats=REPEATS,
               seed=SEED)
    with open(os.path.join(OUT, "v3_metric_cost.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"windows           {out['n_windows']}")
    print(f"accuracy only     {ms_acc:.4f} ms")
    print(f"full metric set   {ms_full:.4f} ms")
    print(f"ratio             {out['ratio']:.4f}x")
    print("wrote v3_metric_cost.json")


if __name__ == "__main__":
    main()
