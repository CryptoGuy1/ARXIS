"""Episode-level hazard metrics and detection latency.

Reviewer Major Concerns 24 and 25, and Tier 1 item 5. The paper eventizes false
alarms but evaluates hazards at the window level, which is asymmetric: one
physical hazardous episode that produces 632 overlapping windows is counted as
632 hazard opportunities, when operationally it is one event that either gets a
response or does not. And a monitoring paper that motivates itself with "what to
do in the next minute" reports no latency anywhere.

This driver reports, per model and per hazardous class, treating each contiguous
run of same-class windows in the test partition as one hazard episode:

  episodes_detected      episodes reaching an alarm-grade action at any point
  detection_rate         that share of episodes
  latency_windows        windows from episode onset to the first alarm-grade action
  latency_seconds        the same at the documented 2 s logging interval
  persistence            share of the episode spent at alarm grade after first alarm
  sustained_detected     episodes where the alarm-grade response persists for
                         PERSIST consecutive windows, so a single flickering
                         window does not count as a detection

Latency is reported for detected episodes only, and the count of undetected
episodes is reported beside it, because a mean latency computed over detected
episodes alone is a conditional quantity and silently flatters a model that
detects few.

One structural caveat, stated here and in the manuscript: the corpus holds one
acquisition run per class, so a test partition contains one contiguous block per
class. Episode counts are therefore small, five per class across the five
partitions rather than hundreds, and the episode-level rates are correspondingly
coarse. That is a property of the corpus, not of the metric, and it is exactly
why the window-level numbers cannot be treated as independent evidence either.

Writes results_v2/v3_episode_metrics.csv.
"""
import os, sys, inspect, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch

torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrain import raw_pipeline as rp
from retrain import rewards as RW
from retrain.run_all_v2 import registry, SEEDS, OUT, CACHE, PERTURB_MODELS, perturb

ALARM_GRADE = 3
PERSIST = 3                 # consecutive alarm-grade windows for a sustained detection
SECONDS_PER_WINDOW = 2.0    # documented logging interval (Narkhede et al. 2022)
HAZARD_CLASSES = ("Smoke", "Mixture")
# In distribution every class block is a steady-state recording with no onset
# transient, so episode detection saturates and latency is structurally zero.
# The degraded conditions are where the metric can discriminate at all.
CONDITIONS = [("clean", 0.0), ("dropout", 1.0), ("dropout", 4.0), ("dropout", 7.0),
              ("drift", 0.5), ("noise", 0.5)]


def episodes_of(win_ids, labels, cls):
    """Contiguous runs of class `cls` in acquisition order, as index slices."""
    out, start = [], None
    for i, (w, g) in enumerate(zip(win_ids, labels)):
        contiguous = (start is not None and g == cls and win_ids[i - 1] == w - 1)
        if g == cls and not contiguous:
            if start is not None:
                out.append((start, i))
            start = i
        elif g != cls and start is not None:
            out.append((start, i)); start = None
    if start is not None:
        out.append((start, len(labels)))
    return out


def first_alarm(acts_slice):
    hits = np.flatnonzero(np.asarray(acts_slice) >= ALARM_GRADE)
    return int(hits[0]) if len(hits) else None


def sustained_alarm(acts_slice, persist=PERSIST):
    a = np.asarray(acts_slice) >= ALARM_GRADE
    run = 0
    for i, f in enumerate(a):
        run = run + 1 if f else 0
        if run >= persist:
            return i - persist + 1
    return None


def main():
    am, asc = rp.build_anomaly_model()
    ds = rp.build_dataset(am, asc, cache_path=CACHE)
    reg = registry(None)
    rows = []
    for name in PERTURB_MODELS:
        mk, kwf = reg[name]
        t0 = time.time()
        per = {(k, l, c): dict(n=0, det=0, sus=0, lat=[], slat=[], pers=[])
               for (k, l) in CONDITIONS for c in HAZARD_CLASSES}
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
            fitted.append((s, m, Xte, gte, te["win_id"].astype(int).values, te["label"].values))

        for kind, lvl in CONDITIONS:
            for s, m, Xte, gte, wid, lab in fitted:
                rng = np.random.default_rng(s)
                Xp = Xte if kind == "clean" else perturb(Xte, kind, lvl, rng)
                acts = np.asarray(m.predict(Xp, groups=gte)
                                  if "groups" in inspect.signature(m.predict).parameters
                                  else m.predict(Xp))
                for c in HAZARD_CLASSES:
                    for a, b in episodes_of(wid, lab, c):
                        seg = acts[a:b]
                        d = per[(kind, lvl, c)]
                        d["n"] += 1
                        k = first_alarm(seg)
                        if k is not None:
                            d["det"] += 1
                            d["lat"].append(k)
                            d["pers"].append(float((seg[k:] >= ALARM_GRADE).mean()))
                        j = sustained_alarm(seg)
                        if j is not None:
                            d["sus"] += 1
                            d["slat"].append(j)

        for (kind, lvl, c), d in per.items():
            lat = np.array(d["lat"], float); slat = np.array(d["slat"], float)
            rows.append(dict(
                model=name, perturbation=kind, level=lvl, hazard_class=c,
                episodes=d["n"], episodes_detected=d["det"], episodes_sustained=d["sus"],
                detection_rate=d["det"] / d["n"] if d["n"] else np.nan,
                sustained_rate=d["sus"] / d["n"] if d["n"] else np.nan,
                latency_windows_median=float(np.median(lat)) if len(lat) else np.nan,
                latency_windows_max=float(lat.max()) if len(lat) else np.nan,
                latency_seconds_median=float(np.median(lat) * SECONDS_PER_WINDOW) if len(lat) else np.nan,
                sustained_latency_windows_median=float(np.median(slat)) if len(slat) else np.nan,
                sustained_latency_seconds_median=float(np.median(slat) * SECONDS_PER_WINDOW) if len(slat) else np.nan,
                persistence_mean=float(np.mean(d["pers"])) if d["pers"] else np.nan,
                undetected=d["n"] - d["det"], persist_windows=PERSIST))
            r = rows[-1]
            if kind != "clean" or c == "Mixture":
                print(f"  {name:16s} {kind:8s}{lvl:>4.1f} {c:8s} "
                      f"det={r['episodes_detected']}/{r['episodes']} "
                      f"sus={r['episodes_sustained']}/{r['episodes']} "
                      f"persistence={r['persistence_mean']:.3f}", flush=True)
        print(f"  {name:16s} done [{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "v3_episode_metrics.csv"), index=False)
    print("\nwrote v3_episode_metrics.csv")
    for kind, lvl in CONDITIONS:
        sub = df[(df.perturbation == kind) & (df.level == lvl)]
        det = sub.groupby("model").episodes_detected.sum() / sub.groupby("model").episodes.sum()
        sus = sub.groupby("model").episodes_sustained.sum() / sub.groupby("model").episodes.sum()
        print(f"\n{kind} {lvl:.1f}: episode detection rate | sustained rate")
        for k in det.index:
            print(f"  {k:16s} {det[k]:.3f} | {sus[k]:.3f}")


if __name__ == "__main__":
    main()
