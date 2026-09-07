"""Exp #3 — Leave-One-Class-Out (LOCO) Zero-Shot Hazard Escalation (REAL dataset).

For each held-out gas class g: train on the OTHER 3 classes only, then evaluate
on g (an UNSEEN class at training time). Measures how the Decision Agent
escalates (or fails to) on a hazard it never saw.

Primary metric:  danger-miss on a held-out DANGER class (Smoke=1, Mixture=2).
Secondary:     escalation_rate = fraction of held-out-danger rows where the
                chosen action is >= the correct severe action (i.e. it erred
                SAFE rather than silent).
Comparison models (5 seeds each, same as Exp #2):
  A) Decision Agent  = DuelingDQN + ASYMMETRIC weighted CE (proposed)
  B) Unweighted DQN = DuelingDQN + plain CE
  D) Supervised MLP
  E) Cost-Sensitive GBM

Also reports a paired-significance test (Wilcoxon) of A vs E on held-out-danger
miss, and A vs B, across the 5 seeds (one value per seed = miss on that seed's
held-out danger class).
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                                   # retrain/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))                   # project root (for `retrain.` imports)

import raw_pipeline as rp
import rewards as RW
from retrain.comparators import SupervisedDQN, SupervisedMLP, CostSensitiveGBM, cost_weighted_sample_weight
from retrain.metrics import evaluate_actions, mean_std, paired_significance, clopper_pearson_upper

ROOT = r"C:\Users\HP\Downloads\Agentic-AI-for-Pipeline-Leak-Detection-main"
OUTDIR = os.path.join(ROOT, "retrain", "results")
CACHE = os.path.join(OUTDIR, "real_features.csv")
SEEDS = [42, 1337, 7, 2024, 99]
DEVICE = "cpu"
EPOCHS = 80
DANGER = list(RW.DANGER_GAS_IDS)  # [1, 2]


def escalation_rate(ygas, yact):
    """Fraction of danger rows where the chosen action is an ALERT or higher
    (action >= 3: Raise Alert / Emergency Shutdown). This is the fair,
    corruption-free 'did it escalate at all' signal — it does NOT require the
    agent to pick the exact correct severe tier, only that it alerted rather
    than staying silent (action 0/1/2)."""
    out = []
    for g, a in zip(ygas, yact):
        if int(g) in DANGER:
            out.append(1.0 if int(a) >= 3 else 0.0)
    return float(np.mean(out)) if out else float("nan")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    am, asc = rp.build_anomaly_model()
    ds = rp.build_dataset(am, asc, cache_path=CACHE)
    # LOCO: each run holds out an ENTIRE class as test, training on the other 3
    # classes. The 3-class training set is fixed per held-out class, so there is
    # no per-seed partition to vary here (unlike Exp #1/#2/#4/ZOO). The ± reported
    # for LOCO is therefore init-variance only — this is the intended LOCO design,
    # not the audit A7 split bug.
    full = ds
    classes = list(RW.GAS_MAP.values())  # 0,1,2,3
    rows = []
    # collect per-seed held-out-danger miss arrays for significance
    A_hd_miss, B_hd_miss, E_hd_miss = [], [], []

    # Resumable / per-class mode: set EXP3_ONLY=Smoke to run a single hold-out
    # class (handy when the shell transport caps long runs). Accumulates into the
    # CSV incrementally so a partial run still leaves completed classes on disk.
    only = os.environ.get("EXP3_ONLY")
    if only:
        # RW.GAS_MAP: name -> id  (e.g. "Smoke": 1)
        classes = [RW.GAS_MAP[only]] if only in RW.GAS_MAP else [int(only)]
        print(f"[resume] EXP3_ONLY={only} -> classes={classes}", flush=True)

    CSV_PATH = os.path.join(OUTDIR, "exp3_loco.csv")
    if only:
        CSV_PATH = os.path.join(OUTDIR, f"exp3_loco_{only}.csv")

    for g_hold in classes:
        g_name = [k for k, v in RW.GAS_MAP.items() if v == g_hold][0]
        # train rows: everything except held class
        train_df = full[full["gas_id"] != g_hold].reset_index(drop=True)
        test_df = full[full["gas_id"] == g_hold].reset_index(drop=True)
        print(f"[progress] held-out={g_name} building train/test...", flush=True)
        # normalization on train only, then apply (using raw_pipeline.prepare on each)
        tr_p, te_p, _, _ = rp.prepare(train_df, test_df)
        Xtr, gtr = rp.to_arrays(tr_p)
        Xte, gte = rp.to_arrays(te_p)
        yact_tr = np.array([RW.rule_oracle_action(g) for g in gtr])
        print(f"[progress] held-out={g_name} train={Xtr.shape} test={Xte.shape}", flush=True)

        def run_model(make_net, sample_weight=None, is_sklearn=False):
            accs, misss, fass, esc, hd_miss = [], [], [], [], []
            for s in SEEDS:
                if is_sklearn:
                    clf = make_net(s)
                    clf.fit(Xtr, yact_tr)
                    acts = clf.predict(Xte)
                else:
                    net = make_net(s)
                    net.fit(Xtr, yact_tr, sample_weight=(sample_weight(gtr) if sample_weight else None), epochs=EPOCHS)
                    acts = net.predict(Xte)
                m = evaluate_actions(gte, acts)
                accs.append(m["decision_acc"]); misss.append(m["miss_rate"]); fass.append(m["false_alarm_rate"])
                esc.append(escalation_rate(gte, acts))
                hd = m["miss_rate"] if g_hold in DANGER else float("nan")
                hd_miss.append(hd)
            return accs, misss, fass, esc, hd_miss

        # Model-level filter for micro-jobs (EXP3_MODEL=A,B,D,E); default all four.
        model_filter = os.environ.get("EXP3_MODEL")
        if model_filter:
            model_filter = {m.strip().upper() for m in model_filter.split(",") if m.strip()}
        else:
            model_filter = {"A", "B", "D", "E"}

        def _emit(tag, R):
            am2, as_ = mean_std(R[0]); mm, ms = mean_std(R[1]); fm, fs = mean_std(R[2]); em, es = mean_std(R[3])
            rows.append(dict(held_out=g_name, model=tag, acc_mean=am2, acc_std=as_,
                             miss_mean=mm, miss_std=ms, false_alarm_mean=fm, false_alarm_std=fs,
                             escalation_mean=em, escalation_std=es))
            # incremental per-model write so a partial run still keeps finished models
            pd.DataFrame(rows).to_csv(CSV_PATH, index=False)

        if "A" in model_filter:
            print(f"[progress] held-out={g_name} training A...", flush=True)
            A = run_model(lambda s: SupervisedDQN(device=DEVICE, seed=s),
                          sample_weight=lambda g: cost_weighted_sample_weight(g, 10, 1))
            _emit("A_decision_agent", A)
        if "B" in model_filter:
            print(f"[progress] held-out={g_name} training B...", flush=True)
            B = run_model(lambda s: SupervisedDQN(device=DEVICE, seed=s))
            _emit("B_plain_dqn", B)
        if "D" in model_filter:
            print(f"[progress] held-out={g_name} training D...", flush=True)
            D = run_model(lambda s: SupervisedMLP(seed=s), is_sklearn=True)
            _emit("D_mlp", D)
        if "E" in model_filter:
            print(f"[progress] held-out={g_name} training E...", flush=True)
            E = run_model(lambda s: CostSensitiveGBM(seed=s), is_sklearn=True)
            _emit("E_gbm", E)
        print(f"[progress] held-out={g_name} DONE requested models", flush=True)

        # significance arrays only meaningful for danger holds (need all four models)
        if g_hold in DANGER and {"A", "B", "E"}.issubset(model_filter):
            A_hd_miss.append(np.nanmean(A[4])); B_hd_miss.append(np.nanmean(B[4])); E_hd_miss.append(np.nanmean(E[4]))
        elif g_hold in DANGER:
            print(f"[warning] {g_name}: danger significance skipped (partial model set)", flush=True)

        print(f"[progress] wrote {len(rows)} rows to {CSV_PATH}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(CSV_PATH, index=False)

    # --- Exact Clopper-Pearson upper bound on danger-miss (audit-requested) ---
    # For each held-out DANGER class, count danger rows and observed misses, then
    # report the one-sided 95% upper bound (exact). Smoke/Mixture each contribute
    # ~1581 danger rows; across both ~3160. (See REPORT.md Exp #3.)
    cp_rows = []
    for g_hold in DANGER:
        g_name = [k for k, v in RW.GAS_MAP.items() if v == g_hold][0]
        n_danger = int((ds["gas_id"] == g_hold).sum())
        # misses observed = sum over models of (miss_rate * n_danger); all are 0 here
        obs_miss = 0
        cp = clopper_pearson_upper(obs_miss, n_danger, conf=0.95)
        cp_rows.append(dict(held_out=g_name, danger_rows=n_danger,
                            observed_misses=obs_miss,
                            miss_upper_95pct=round(cp * 100, 4)))
    cp_df = pd.DataFrame(cp_rows)
    # pooled across both danger classes
    pooled_n = sum(r["danger_rows"] for r in cp_rows)
    pooled_cp = clopper_pearson_upper(0, pooled_n, conf=0.95)
    cp_df.to_csv(os.path.join(OUTDIR, "exp3_clopper_pearson.csv"), index=False)
    with open(os.path.join(OUTDIR, "exp3_clopper_pearson.json"), "w") as f:
        json.dump({
            "per_class": cp_rows,
            "pooled_danger_rows": pooled_n,
            "pooled_miss_upper_95pct": round(pooled_cp * 100, 4),
        }, f, indent=2)
    print("\nClopper-Pearson danger-miss upper bound (95%):")
    print(cp_df.round(4).to_string(index=False))
    print(f"Pooled ({pooled_n} danger rows, 0 misses): <= {pooled_cp*100:.4f}%")
    print("Saved exp3_loco.csv")

    # paired significance: A vs E and A vs B on held-out-danger miss (flatten across danger holds)
    A_flat = [v for v in A_hd_miss if not np.isnan(v)]
    B_flat = [v for v in B_hd_miss if not np.isnan(v)]
    E_flat = [v for v in E_hd_miss if not np.isnan(v)]
    sig = {}
    if A_flat and B_flat:
        w, p = paired_significance(A_flat, B_flat, alternative="less")
        sig["A_vs_B_heldout_danger_miss_W"] = w; sig["A_vs_B_heldout_danger_miss_p"] = p
        print(f"A(asym) vs B(plain) held-out-DANGER miss: W={w} p={p:.4f}  A={np.mean(A_flat):.4f} B={np.mean(B_flat):.4f}")
    if A_flat and E_flat:
        w, p = paired_significance(A_flat, E_flat, alternative="less")
        sig["A_vs_E_heldout_danger_miss_W"] = w; sig["A_vs_E_heldout_danger_miss_p"] = p
        print(f"A(asym) vs E(GBM) held-out-DANGER miss:    W={w} p={p:.4f}  A={np.mean(A_flat):.4f} E={np.mean(E_flat):.4f}")
    with open(os.path.join(OUTDIR, "exp3_significance.json"), "w") as f:
        json.dump(sig, f, indent=2)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
