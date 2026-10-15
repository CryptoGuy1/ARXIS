"""Summarize the thermal-classifier split study and check each run against itself.

Reads the per-run JSON files written by the Kaggle drivers (retrain/results_thermal/runs/),
the frame-level split tables (split_tables/), the per-epoch curves (curves/) and the
checkpoint hashes recorded in checkpoints.json. For every run it checks that the
top-1, per-class and hazard figures follow from the confusion matrix, that the
split counts and held-out block positions match the split table, and that held-out
test blocks are contiguous for the block-wise protocol. It then writes summary.json,
which verify_v15.py reads when it checks the perception numbers in the paper.

    python3 retrain/summarize_thermal.py

Weights are not in the repository (see Section S19); their hashes are recorded so a
copy obtained elsewhere can be matched to the run it came from.
"""
import csv, glob, json, os, sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ROOT, "results_thermal")
ORDER = ["NoGas", "Smoke", "Mixture", "Perfume"]
HAZARD = ["Smoke", "Mixture"]
N_TEST = 1280
PER_CLASS = 320

FAIL = []


def check(ok, msg):
    if not ok:
        FAIL.append(msg)


def load_split(path):
    rows = list(csv.DictReader(open(path, newline="")))
    per = defaultdict(list)
    for r in rows:
        per[r["label"]].append(r["split"])
    return rows, per


def run_checks(name, d):
    c = d["confusion_final_epoch"]
    total = sum(sum(c[t].values()) for t in c)
    diag = sum(c[k][k] for k in c)
    check(total == N_TEST, f"{name}: confusion total {total} != {N_TEST}")
    check(all(sum(c[t].values()) == PER_CLASS for t in c), f"{name}: confusion row sum != {PER_CLASS}")
    check(abs(diag / total - d["primary_top1_final_epoch"]) < 1e-9, f"{name}: top-1 does not match confusion")
    for k in ORDER:
        check(abs(c[k][k] / PER_CLASS - d["per_class_final_epoch"][k]) < 1e-9, f"{name}: per-class {k} mismatch")
    hz = sum(c[t][p] for t in HAZARD for p in c[t] if p not in HAZARD)
    if "hazard_called_nonhazard_final_epoch" in d:
        check(hz == d["hazard_called_nonhazard_final_epoch"], f"{name}: hazard count mismatch")
    d["_hazard_called_nonhazard"] = hz

    split_name = d.get("split") or ("randomized" if d["config"]["embargo_frames"] == 0 else "blockwise")
    table = os.path.join(D, "split_tables", f"split_table_{split_name}_seed{d['seed']}.csv")
    rows, per = load_split(table)
    cnt = Counter(r["split"] for r in rows)
    check(cnt["test"] == d["counts"]["heldout"], f"{name}: held-out count differs from split table")
    check(cnt["train"] == d["counts"]["train"], f"{name}: train count differs from split table")
    check(cnt.get("embargo", 0) == d["counts"]["embargo"], f"{name}: embargo count differs from split table")
    for lab, seq in per.items():
        check(sum(1 for s in seq if s == "test") == PER_CLASS, f"{name}: {lab} test count is not {PER_CLASS}")
    if split_name == "blockwise":
        for lab, seq in per.items():
            pos = [i for i, s in enumerate(seq) if s == "test"]
            check(pos == list(range(pos[0], pos[0] + len(pos))), f"{name}: {lab} test block not contiguous")
            check(pos[0] == d["held_out_block_start_per_class"][lab],
                  f"{name}: {lab} block start {pos[0]} != JSON {d['held_out_block_start_per_class'][lab]}")

    curve = os.path.join(D, "curves", f"{name}_results.csv")
    last_epoch = None
    if os.path.exists(curve):
        cr = list(csv.DictReader(open(curve)))
        last_epoch = float(cr[-1]["metrics/accuracy_top1"])
        check(len(cr) == d["config"]["epochs"], f"{name}: curve has {len(cr)} epochs")
    return {"split": split_name, "last_epoch_curve_top1": last_epoch}


def stats(vals):
    a = np.array(vals, dtype=float)
    return {"n": int(a.size), "mean": float(a.mean()),
            "sd": float(a.std(ddof=1)) if a.size > 1 else None,
            "min": float(a.min()), "max": float(a.max())}


def main():
    runs = {}
    for path in sorted(glob.glob(os.path.join(D, "runs", "*.json"))):
        d = json.load(open(path))
        name = d["run"]
        extra = run_checks(name, d)
        runs[name] = {"split": extra["split"], "seed": d["seed"],
                      "top1": d["primary_top1_final_epoch"],
                      "errors": N_TEST - round(d["primary_top1_final_epoch"] * N_TEST),
                      "best_checkpoint_reference": d["top1_best_checkpoint_reference_only"],
                      "per_class": d["per_class_final_epoch"],
                      "hazard_frames": 2 * PER_CLASS,
                      "hazard_called_nonhazard": d["_hazard_called_nonhazard"],
                      "last_epoch_curve_top1": extra["last_epoch_curve_top1"]}

    ck_path = os.path.join(D, "checkpoints.json")
    if os.path.exists(ck_path):
        ck = json.load(open(ck_path))
        for name in runs:
            check(name in ck, f"{name}: no checkpoint hash recorded")
            if name in ck:
                runs[name]["checkpoint_sha256"] = ck[name]["sha256"]
                runs[name]["checkpoint_stored_top1"] = ck[name]["stored_top1"]
                # The checkpoint stores validation top-1 during training (fp16 on the GPU), rounded to five
                # decimals; the JSON re-predicts the held-out frames in fp32 after training. A difference of a
                # few frames is expected; more than three would mean the JSON and the checkpoint disagree.
                frames = round((ck[name]["stored_top1"] - runs[name]["top1"]) * N_TEST)
                runs[name]["checkpoint_minus_json_frames"] = int(frames)
                check(abs(frames) <= 3, f"{name}: checkpoint and JSON differ by {frames} frames")
    else:
        FAIL.append("checkpoints.json missing")

    block = {k: v for k, v in runs.items() if v["split"] == "blockwise"}
    rand = {k: v for k, v in runs.items() if v["split"] == "randomized"}
    summary = {"runs": runs, "groups": {}}
    for label, grp in (("blockwise", block), ("randomized", rand)):
        if not grp:
            continue
        g = {"top1": stats([v["top1"] for v in grp.values()]),
             "errors": stats([v["errors"] for v in grp.values()]),
             "hazard_called_nonhazard": stats([v["hazard_called_nonhazard"] for v in grp.values()]),
             "per_class": {c: stats([v["per_class"][c] for v in grp.values()]) for c in ORDER}}
        summary["groups"][label] = g

    summary["gap_randomized_minus_blockwise_mean_points"] = (
        100 * (summary["groups"]["randomized"]["top1"]["mean"] - summary["groups"]["blockwise"]["top1"]["mean"]))
    with open(os.path.join(D, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    print(f"{'run':30s} {'top1%':>7s} {'err':>4s} {'hazard':>7s}  NoGas  Smoke  Mixt  Perf")
    for name in sorted(runs):
        v = runs[name]
        pc = v["per_class"]
        print(f"{name:30s} {100 * v['top1']:7.3f} {v['errors']:4d} {v['hazard_called_nonhazard']:4d}/640  "
              f"{100 * pc['NoGas']:5.1f} {100 * pc['Smoke']:6.1f} {100 * pc['Mixture']:5.1f} {100 * pc['Perfume']:5.1f}")
    for label, g in summary["groups"].items():
        t = g["top1"]
        sd = f"{100 * t['sd']:.2f}" if t["sd"] is not None else "n/a"
        print(f"{label}: n={t['n']} top1 mean {100 * t['mean']:.2f} sd {sd} range {100 * t['min']:.2f}-{100 * t['max']:.2f}")
    print(f"gap (randomized mean minus block-wise mean): {summary['gap_randomized_minus_blockwise_mean_points']:.2f} points")

    if FAIL:
        print(f"{len(FAIL)} failure(s):")
        for m in FAIL:
            print("  FAIL", m)
        sys.exit(1)
    print("all thermal checks pass")


if __name__ == "__main__":
    main()
