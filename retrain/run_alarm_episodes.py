"""Alarm indications are not alarm events. This driver counts both.

Reviewer comment 4.9. Every false-alarm number elsewhere in this study is a rate
per clean WINDOW, and windows advance every 2 s. One sustained condition can
therefore produce hundreds of consecutive alarm-grade windows, which is an
indication rate, not the rate at which an operator would be interrupted. Real
alarm systems interpose persistence (on-delay), debounce (off-delay), latching
and suppression before a condition becomes an annunciated event.

Here contiguous runs of alarm-grade windows on CLEAN (NoGas) test windows are
collapsed into alarm episodes under an explicit rule:

    on-delay   P consecutive alarm-grade windows before an episode starts
    off-delay  Q consecutive non-alarm windows before it ends

With P = 1, Q = 1 the count is the raw indication count and the comparison is
degenerate by construction; the deployed screen uses P = 3 (6 s of persistence)
and Q = 15 (30 s of quiet before the condition is considered cleared), which are
conservative relative to typical process practice and are stated rather than
tuned. Episodes are reported per 1,000 clean windows and, under the same
illustrative duty cycle used in the manuscript (f = 1,800 windows/hour), per hour.

Windows are ordered by win_id, which is acquisition order, so contiguity in the
sequence is contiguity in time within a class block.

Writes results_v2/v3_alarm_episodes.csv.
"""
import os, sys, time, inspect, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch

torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrain import raw_pipeline as rp
from retrain.run_all_v2 import registry, SEEDS, OUT, CACHE, PERTURB_MODELS, perturb
from retrain import rewards as RW

ALARM_GRADE = 3
ON_DELAY, OFF_DELAY = 3, 15
WINDOWS_PER_HOUR = 1800.0
CONDITIONS = [("clean", 0.0), ("dropout", 1.0), ("dropout", 4.0), ("dropout", 7.0),
              ("drift", 0.3), ("drift", 0.5), ("noise", 0.3), ("noise", 0.5)]


def episodes(alarm_flags, on_delay=ON_DELAY, off_delay=OFF_DELAY):
    """Count annunciated episodes in a boolean sequence under on/off delays.

    An episode begins when `on_delay` consecutive True values have been seen and
    ends when `off_delay` consecutive False values have been seen. A run shorter
    than on_delay never annunciates; a gap shorter than off_delay does not split
    one episode into two.
    """
    n_ep, run_on, run_off, active = 0, 0, 0, False
    for f in alarm_flags:
        if f:
            run_off = 0
            run_on += 1
            if not active and run_on >= on_delay:
                active = True
                n_ep += 1
        else:
            run_on = 0
            if active:
                run_off += 1
                if run_off >= off_delay:
                    active = False
                    run_off = 0
    return n_ep


def main():
    am, asc = rp.build_anomaly_model()
    ds = rp.build_dataset(am, asc, cache_path=CACHE)
    reg = registry(None)
    rows = []
    for name in PERTURB_MODELS:
        mk, kwf = reg[name]
        t0 = time.time()
        fitted = []
        for s in SEEDS:
            tr, te = rp.block_wise_holdout(ds, seed=s)
            tr, te, _, _ = rp.prepare(tr, te, anomaly_seed=s)
            te = te.sort_values("win_id").reset_index(drop=True)   # acquisition order
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
                flags = (np.asarray(acts)[clean] >= ALARM_GRADE)
                n_clean = int(clean.sum())
                n_ind = int(flags.sum())
                n_ep = episodes(flags)
                per_seed.append((n_clean, n_ind, n_ep))
            n_clean = float(np.mean([p[0] for p in per_seed]))
            ind = np.array([p[1] for p in per_seed], float)
            ep = np.array([p[2] for p in per_seed], float)
            rows.append(dict(
                model=name, perturbation=kind, level=lvl,
                n_clean_windows=n_clean,
                indications_mean=ind.mean(), indications_std=ind.std(),
                episodes_mean=ep.mean(), episodes_std=ep.std(),
                indications_per_1k=1000.0 * ind.mean() / n_clean,
                episodes_per_1k=1000.0 * ep.mean() / n_clean,
                indications_per_hour=WINDOWS_PER_HOUR * ind.mean() / n_clean,
                episodes_per_hour=WINDOWS_PER_HOUR * ep.mean() / n_clean,
                compression=(ind.mean() / ep.mean()) if ep.mean() > 0 else float("nan"),
                on_delay=ON_DELAY, off_delay=OFF_DELAY))
            r = rows[-1]
            print(f"  {name:16s} {kind:8s} {lvl:>4.1f}  "
                  f"ind/h={r['indications_per_hour']:8.1f}  ep/h={r['episodes_per_hour']:7.2f}  "
                  f"compression={r['compression']:6.1f}x", flush=True)
        print(f"  {name:16s} done [{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "v3_alarm_episodes.csv"), index=False)
    print("\nwrote v3_alarm_episodes.csv")
    worst = df[df.perturbation != "clean"].sort_values("episodes_per_hour", ascending=False).head(3)
    print("\nhighest episode rates observed:")
    for _, r in worst.iterrows():
        print(f"  {r['model']:16s} {r['perturbation']}@{r['level']:.1f}  "
              f"{r['episodes_per_hour']:.2f} episodes/h against "
              f"{r['indications_per_hour']:.1f} indications/h")


if __name__ == "__main__":
    main()
