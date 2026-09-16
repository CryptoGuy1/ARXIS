"""Does the eventized alarm result depend on the persistence rule that produced it?

Five-part adversarial review, comment 3.12. The manuscript collapses contiguous
alarm-grade windows into annunciated episodes under one stated rule: an on-delay
of three windows and an off-delay of fifteen. That rule is a methodological
convention, not an industrially validated persistence specification, and the
eventized view is used in the manuscript to REORDER the models relative to the
raw indication view. If the ordering itself moves when the rule moves, the
reordering is a property of the convention rather than of the models, and the
manuscript's claim has to be weakened accordingly.

This driver answers that question directly. Models are fitted exactly once per
seed, the alarm-grade flag sequence is computed once per (model, seed,
condition), and the SAME flag sequences are then eventized under every point of
an on-delay x off-delay grid. Nothing is refitted between grid points, so any
difference in the output is caused by the persistence rule alone.

    on-delay   P in {1, 2, 3, 5, 10}  windows of persistence before annunciation
    off-delay  Q in {5, 10, 15, 30, 60} windows of quiet before the episode clears

P = 1, Q = 1 would reproduce the raw indication count and is excluded as
degenerate. The deployed screen, P = 3 and Q = 15, is one point of the grid so
the manuscript's numbers are reproduced exactly by this driver as a check.

Two things are reported. First, the episode rate at every grid point, so a
reader can see the spread. Second, and the point of the exercise, the rank
agreement between the model ordering at each grid point and the ordering at the
deployed setting, computed per condition with Kendall's tau, plus an explicit
record of every pairwise reversal. An ordering that survives the grid supports
the manuscript's claim; one that does not, does not.

Writes results_v2/v3_persistence_sweep.csv and results_v2/v3_persistence_rank.json.
"""
import os, sys, time, inspect, json, itertools, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch

torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrain import raw_pipeline as rp
from retrain.run_all_v2 import registry, SEEDS, OUT, CACHE, PERTURB_MODELS, perturb
from retrain.run_alarm_episodes import episodes, ALARM_GRADE, WINDOWS_PER_HOUR, CONDITIONS
from retrain import rewards as RW

ON_GRID = [1, 2, 3, 5, 10]
OFF_GRID = [5, 10, 15, 30, 60]
DEPLOYED = (3, 15)


def kendall_tau(a, b):
    """Kendall's tau-b between two rank vectors, written out to avoid a scipy
    version dependency in the released environment."""
    n = len(a)
    conc = disc = tie_a = tie_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 and db == 0:
                tie_a += 1; tie_b += 1
            elif da == 0:
                tie_a += 1
            elif db == 0:
                tie_b += 1
            elif (da > 0) == (db > 0):
                conc += 1
            else:
                disc += 1
    n0 = n * (n - 1) / 2
    den = np.sqrt((n0 - tie_a) * (n0 - tie_b))
    return float((conc - disc) / den) if den > 0 else float("nan")


def main():
    am, asc = rp.build_anomaly_model()
    ds = rp.build_dataset(am, asc, cache_path=CACHE)
    reg = registry(None)

    # flags[(model, condition, level)] = list over seeds of (n_clean, flag array)
    flags = {}
    for name in PERTURB_MODELS:
        mk, kwf = reg[name]
        t0 = time.time()
        fitted = []
        for s in SEEDS:
            tr, te = rp.block_wise_holdout(ds, seed=s)
            tr, te, _, _ = rp.prepare(tr, te, anomaly_seed=s)
            te = te.sort_values("win_id").reset_index(drop=True)
            Xtr, gtr = rp.to_arrays(tr)
            ytr = np.array([RW.rule_oracle_action(int(g)) for g in gtr])
            Xte, gte = rp.to_arrays(te)
            m = mk(s)
            kw = dict(kwf(s, gtr))
            if "groups" in inspect.signature(m.fit).parameters:
                kw.setdefault("groups", gtr)
            m.fit(Xtr, ytr, **kw)
            fitted.append((s, m, Xte, gte))

        for kind, lvl in CONDITIONS:
            per_seed = []
            for s, m, Xte, gte in fitted:
                rng = np.random.default_rng(s)
                Xp = Xte if kind == "clean" else perturb(Xte, kind, lvl, rng)
                if "groups" in inspect.signature(m.predict).parameters:
                    acts = m.predict(Xp, groups=gte)
                else:
                    acts = m.predict(Xp)
                clean = (gte == 0)
                per_seed.append((int(clean.sum()),
                                 (np.asarray(acts)[clean] >= ALARM_GRADE)))
            flags[(name, kind, lvl)] = per_seed
        print(f"  {name:16s} flags computed [{time.time()-t0:.0f}s]", flush=True)

    # ------------------------------------------------------------------ sweep
    rows = []
    for (name, kind, lvl), per_seed in flags.items():
        for on, off in itertools.product(ON_GRID, OFF_GRID):
            ind = np.array([int(f.sum()) for _, f in per_seed], float)
            ep = np.array([episodes(f, on, off) for _, f in per_seed], float)
            n_clean = float(np.mean([n for n, _ in per_seed]))
            rows.append(dict(
                model=name, perturbation=kind, level=lvl,
                on_delay=on, off_delay=off,
                n_clean_windows=n_clean,
                indications_mean=ind.mean(),
                episodes_mean=ep.mean(), episodes_std=ep.std(),
                indications_per_hour=WINDOWS_PER_HOUR * ind.mean() / n_clean,
                episodes_per_hour=WINDOWS_PER_HOUR * ep.mean() / n_clean,
                compression=(ind.mean() / ep.mean()) if ep.mean() > 0 else float("nan")))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "v3_persistence_sweep.csv"), index=False)
    print(f"\nwrote v3_persistence_sweep.csv  ({len(df)} rows)")

    # ------------------------------------------------- ordering stability
    out = {"on_grid": ON_GRID, "off_grid": OFF_GRID, "deployed": list(DEPLOYED),
           "conditions": {}}
    for kind, lvl in CONDITIONS:
        sub = df[(df.perturbation == kind) & (df.level == lvl)]
        base = sub[(sub.on_delay == DEPLOYED[0]) & (sub.off_delay == DEPLOYED[1])]
        base = base.set_index("model")["episodes_per_hour"]
        models = sorted(base.index)
        if len(models) < 2:
            continue
        b = [float(base[m]) for m in models]
        taus, reversals = [], []
        for on, off in itertools.product(ON_GRID, OFF_GRID):
            if (on, off) == DEPLOYED:
                continue
            cur = sub[(sub.on_delay == on) & (sub.off_delay == off)].set_index("model")["episodes_per_hour"]
            c = [float(cur[m]) for m in models]
            taus.append(kendall_tau(b, c))
            for i, j in itertools.combinations(range(len(models)), 2):
                if (b[i] - b[j]) * (c[i] - c[j]) < 0:
                    reversals.append(f"{models[i]} vs {models[j]} at on={on},off={off}")
        span = sub.groupby("model")["episodes_per_hour"].agg(["min", "max"])
        out["conditions"][f"{kind}@{lvl}"] = {
            "models": models,
            "episodes_per_hour_deployed": {m: float(base[m]) for m in models},
            "episodes_per_hour_min_over_grid": {m: float(span.loc[m, "min"]) for m in models},
            "episodes_per_hour_max_over_grid": {m: float(span.loc[m, "max"]) for m in models},
            "kendall_tau_vs_deployed_min": float(np.nanmin(taus)) if taus else None,
            "kendall_tau_vs_deployed_mean": float(np.nanmean(taus)) if taus else None,
            "n_grid_points_compared": len(taus),
            "n_pairwise_reversals": len(reversals),
            "n_model_pairs": len(models) * (len(models) - 1) // 2,
            "reversals": reversals[:40],
        }
        print(f"  {kind:8s}@{lvl:<4.1f} tau_min={out['conditions'][f'{kind}@{lvl}']['kendall_tau_vs_deployed_min']:.3f}"
              f"  reversals={len(reversals)} over {len(taus)} grid points", flush=True)

    with open(os.path.join(OUT, "v3_persistence_rank.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote v3_persistence_rank.json")


if __name__ == "__main__":
    main()
