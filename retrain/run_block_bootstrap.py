"""A safety-rate interval that does not assume independent windows.

Reviewer Tier 1 item 4. The binomial upper bound reported elsewhere is exact
under a model this design violates: consecutive windows share 19 of 20 raw rows,
so 316 clean windows are nowhere near 316 independent trials. The binomial figure
is kept as a labelled reference point, but it should not be the primary interval.

A block bootstrap over *episodes* is the textbook answer and is unavailable here:
each class was acquired in one continuous run, so the corpus contains one episode
per class and there is nothing to resample at that level. What can be done, and
what this driver does, is a moving-block bootstrap over contiguous runs of windows
inside each test partition. Blocks of length L are drawn with replacement until
the resampled series matches the original length, which preserves dependence up to
lag L instead of destroying it. With L set to at least the window length, every
resampled block carries its own internal overlap structure intact.

The block length is swept rather than chosen, because the interval width depends
on it and a reader is entitled to see how much. L = 20 is the window length, the
shortest defensible choice; L = 100 and L = 316 probe longer-range dependence, the
last being a single contiguous block per class, which is the most conservative
resampling this partition admits.

Reported per model and per quantity: the point estimate, the percentile interval
at each block length, and the binomial reference. Where a count is zero the
bootstrap interval is degenerate at zero and the effective-sample-size estimate is
what carries the information; that case is reported explicitly rather than hidden.

Writes results_v2/v3_block_bootstrap.json.
"""
import json, os, sys, inspect, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch

torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrain import raw_pipeline as rp
from retrain import rewards as RW
from retrain.run_all_v2 import registry, SEEDS, OUT, CACHE, PERTURB_MODELS

ALARM_GRADE = 3
BLOCK_LENGTHS = [20, 100, 316]
B = 4000
BOOT_SEED = 20260911
CONF = 0.95


def moving_block_indices(n, L, rng):
    """Index set for one moving-block bootstrap replicate of a length-n series."""
    if L >= n:
        return np.arange(n)
    k = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, size=k)
    idx = np.concatenate([np.arange(s, s + L) for s in starts])
    return idx[:n]


def block_ci(flags, L, rng, B=B, conf=CONF):
    """Percentile interval for the mean of a dependent binary series."""
    n = len(flags)
    if n == 0:
        return (float("nan"), float("nan"))
    reps = np.empty(B)
    for b in range(B):
        reps[b] = flags[moving_block_indices(n, L, rng)].mean()
    a = (1.0 - conf) / 2.0
    return float(np.percentile(reps, 100 * a)), float(np.percentile(reps, 100 * (1 - a)))


def effective_n(flags, L):
    """Number of windows in the series with pairwise-disjoint raw support.

    Windows overlap by L-1 raw rows, so only every Lth window is free of shared
    readings with its neighbours: n//L of them. That count is the honest
    denominator to quote beside a binomial bound computed on all n windows. It is
    still an upper bound on the information available, because those n//L windows
    come from one acquisition episode on one device and are not independent in any
    physical sense either.
    """
    return max(1, int(len(flags) // L))


def binomial_upper(k, n, conf=CONF):
    from scipy.stats import beta
    if n <= 0:
        return float("nan")
    if k == 0:
        return 1.0 - (1.0 - conf) ** (1.0 / n)
    if k >= n:
        return 1.0
    return float(beta.ppf(conf, k + 1, n - k))


def main():
    am, asc = rp.build_anomaly_model()
    ds = rp.build_dataset(am, asc, cache_path=CACHE)
    reg = registry(None)
    rng = np.random.default_rng(BOOT_SEED)
    out = {"_meta": dict(block_lengths=BLOCK_LENGTHS, replicates=B, seed=BOOT_SEED,
                         confidence=CONF, window_size=rp.WINDOW_SIZE,
                         note="moving-block bootstrap within each test partition; "
                              "one acquisition episode per class, so episode-level "
                              "resampling is not available on this corpus")}

    for name in PERTURB_MODELS:
        mk, kwf = reg[name]
        per_seed = []
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
            acts = (m.predict(Xte, groups=gte)
                    if "groups" in inspect.signature(m.predict).parameters
                    else m.predict(Xte))
            acts = np.asarray(acts)
            haz = np.isin(gte, list(rp.GAS_MAP[c] for c in ("Smoke", "Mixture")))
            clean = gte == rp.GAS_MAP["NoGas"]
            per_seed.append(dict(
                miss=(acts[haz] == 0).astype(float),
                fa=(acts[clean] >= ALARM_GRADE).astype(float)))

        rec = {}
        for q in ("miss", "fa"):
            series = [d[q] for d in per_seed]
            point = float(np.mean([x.mean() for x in series]))
            k_tot = int(sum(x.sum() for x in series))
            n_one = int(len(series[0]))
            rec[q] = dict(point=point, k_per_partition=k_tot / len(series),
                          n_per_partition=n_one,
                          binomial_upper_pct=100 * binomial_upper(0 if point == 0 else
                                                                  int(round(point * n_one)), n_one))
            for L in BLOCK_LENGTHS:
                los, his = [], []
                for x in series:
                    lo, hi = block_ci(x, L, rng)
                    los.append(lo); his.append(hi)
                rec[q][f"L{L}"] = dict(ci_lo=float(np.mean(los)), ci_hi=float(np.mean(his)),
                                       effective_n=effective_n(series[0], L))
        out[name] = rec
        m_ = rec["miss"]; f_ = rec["fa"]
        print(f"  {name:16s} miss={m_['point']:.4f} "
              f"blockCI@20=[{m_['L20']['ci_lo']:.4f},{m_['L20']['ci_hi']:.4f}] "
              f"@316=[{m_['L316']['ci_lo']:.4f},{m_['L316']['ci_hi']:.4f}] "
              f"binom<={m_['binomial_upper_pct']:.3f}%  n_eff@20={m_['L20']['effective_n']}",
              flush=True)

    json.dump(out, open(os.path.join(OUT, "v3_block_bootstrap.json"), "w"), indent=1)
    print("\nwrote v3_block_bootstrap.json")
    n632 = 632 // rp.WINDOW_SIZE
    n316 = 316 // rp.WINDOW_SIZE
    print(f"\neffective sample size at the window length: {n632} hazardous blocks "
          f"and {n316} clean blocks per partition, against 632 and 316 nominal windows.")
    print(f"a zero over {n632} independent blocks supports at most "
          f"{100*binomial_upper(0, n632):.2f}%, against "
          f"{100*binomial_upper(0, 632):.2f}% over 632 nominal windows.")


if __name__ == "__main__":
    main()
