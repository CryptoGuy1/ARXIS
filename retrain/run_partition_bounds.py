"""Uncertainty with the partition, not the window, as the experimental unit.

Earlier revisions pooled every test window across seeds and applied a binomial
upper bound to the total. That treats the pooled count as independent Bernoulli
trials, which this design does not deliver: consecutive windows share 19 of 20
raw rows, and because the held-out block position moves with the seed, the same
underlying window is evaluated in more than one partition. The pooled bound
therefore claims more independent information than the experiment produced.

What is defensible without new data is the bound a SINGLE partition supports,
reported at the worst case over partitions. That still assumes independence
within a partition, which overlap also violates, so it remains an optimistic
bound; the manuscript says so rather than implying otherwise.
"""
import json, os
import numpy as np, pandas as pd
from scipy.stats import beta

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_v2")


def cp_upper(k, n, conf=0.95):
    if n <= 0:
        return float("nan")
    if k == 0:
        return 1.0 - (1.0 - conf) ** (1.0 / n)
    if k >= n:
        return 1.0
    return float(beta.ppf(conf, k + 1, n - k))


def report(label, k_pooled, n_pooled, n_runs):
    n_part = int(round(n_pooled / n_runs))
    k_part = int(round(k_pooled / n_runs))
    return dict(quantity=label, runs=n_runs, n_per_partition=n_part,
                k_pooled=int(k_pooled), n_pooled=int(n_pooled),
                pooled_bound_pct=100 * cp_upper(k_pooled, n_pooled),
                partition_bound_pct=100 * cp_upper(k_part, n_part))


out = []
pw = pd.read_csv(R + "/v3_powered.csv")
r = pw.iloc[0]
out.append(report("30-run in-distribution, hazardous windows", 0, r.n_danger_pooled, 30))
out.append(report("30-run in-distribution, clean windows", 0, r.n_clean_pooled, 30))

lo = pd.read_csv(R + "/v3_loco_all.csv")
out.append(report("leave-one-class-out, hazardous windows", 0, lo.iloc[0].n_danger_pooled, 5))

zo = pd.read_csv(R + "/v2_zoo.csv").iloc[0]
out.append(report("five-seed in-distribution, hazardous windows", 0, zo.n_danger_pooled, 5))
out.append(report("five-seed in-distribution, clean windows", 0, zo.n_clean_pooled, 5))

df = pd.DataFrame(out)
print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print()
print("alarm burden on an observed zero, per 1,000 clean windows:")
for n_runs, n_pooled, tag in [(5, zo.n_clean_pooled, "five-seed"), (30, r.n_clean_pooled, "30-run")]:
    n_part = int(round(n_pooled / n_runs))
    print(f"  {tag:10s} pooled n={int(n_pooled):5d} -> {1000*cp_upper(0,int(n_pooled)):5.2f} | "
          f"per partition n={n_part:4d} -> {1000*cp_upper(0,n_part):5.2f}")

# Per-partition bounds for the leave-one-class-out table.
#
# Plan item A1. These used to be computed by taking the mean miss RATE across
# the five seeds and rounding rate x n back into a count. That is the count an
# average partition produced, not the count the worst one produced, and where
# the seeds disagree it is much smaller: the multilayer perceptron on held-out
# Smoke misses 4, 23, 30, 216 and 234 windows across its five partitions, so the
# averaged pseudo-count gives 7.49% where the worst partition supports 16.35%.
# The stated rule in the manuscript is the worst case over partitions, so the
# integer counts are read from v3_loco_perseed.csv and the worst is used.
print("\nleave-one-class-out, per-partition bound at the worst observed count (n = 1,581):")
ps = pd.read_csv(R + "/v3_loco_perseed.csv")
rows = []
for _, s in lo.iterrows():
    g = ps[(ps.held_out == s.held_out) & (ps.model == s.model)]
    ks = sorted(int(x) for x in g.miss_k)
    n_part = int(g.n_danger.iloc[0])
    rows.append(dict(held_out=s.held_out, model=s.model, miss=s.miss_rate_mean,
                     k_per_partition=ks, k_part=max(ks),
                     pooled_pct=100 * s.miss_upper95_pooled,
                     mean_pseudocount_pct=100 * cp_upper(int(round(s.miss_rate_mean * n_part)), n_part),
                     partition_pct=100 * cp_upper(max(ks), n_part)))
b = pd.DataFrame(rows)
print(b.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
json.dump(dict(summary=out, loco=rows), open(R + "/v3_partition_bounds.json", "w"), indent=1, default=float)
print("\nwrote v3_partition_bounds.json")
