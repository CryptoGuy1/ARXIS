"""A resampling comparison that does not need an assumed run correlation.

The Bayesian correlated t-test needs a value for rho, and ours is imported from
a k-fold setting that is not quite this design. A paired bootstrap over the 30
runs avoids the question: the run is the experimental unit, runs are resampled
with replacement, and nothing is assumed about within-run correlation because
the run is never broken open.

Reported per comparison: the mean paired difference, a percentile interval, and
the share of bootstrap replicates in which the reference is ahead, behind, or
inside the same region of practical equivalence used elsewhere (1 accuracy
point). The question asked is the same; the assumption is weaker.
"""
import json, os
import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_v2")
ROPE, B, SEED = 0.01, 20000, 12345


def main():
    acc = json.load(open(R + "/v3_powered_bayes.json"))["per_run_acc"]
    ref = np.asarray(acc["A_cost_weighted"], float)
    n = len(ref)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(B, n))          # one index set, shared across models
    rows = {}
    print(f"paired bootstrap over {n} runs, {B} replicates, ROPE +/- {ROPE*100:.0f} accuracy point\n")
    print(f"{'model':16s} {'mean diff':>10s} {'95% CI':>20s} {'P(ref better)':>14s} "
          f"{'P(equiv)':>9s} {'P(other better)':>16s}")
    for name, vals in acc.items():
        if name == "A_cost_weighted":
            continue
        d = ref - np.asarray(vals, float)
        reps = d[idx].mean(axis=1)
        lo, hi = np.percentile(reps, [2.5, 97.5])
        p_ref = float((reps > ROPE).mean())
        p_eq = float((np.abs(reps) <= ROPE).mean())
        p_oth = float((reps < -ROPE).mean())
        rows[name] = dict(mean_diff=float(d.mean()), ci_lo=float(lo), ci_hi=float(hi),
                          p_ref_better=p_ref, p_equivalent=p_eq, p_other_better=p_oth)
        print(f"{name:16s} {d.mean():+10.4f} [{lo:+7.4f}, {hi:+7.4f}] "
              f"{p_ref:14.3f} {p_eq:9.3f} {p_oth:16.3f}")

    # does the bootstrap agree with the correlated t-test about direction?
    bay = json.load(open(R + "/v3_powered_bayes.json"))["vs_cost_weighted"]
    print("\nagreement with the Bayesian correlated t-test on the most probable outcome:")
    agree = 0
    for name, v in rows.items():
        b = bay[name]
        pick = lambda o, e, r: max((o, "other"), (e, "equiv"), (r, "ref"), key=lambda t: t[0])[1]
        boot = pick(v["p_other_better"], v["p_equivalent"], v["p_ref_better"])
        bays = pick(b["p_other_better"], b["p_practically_equivalent"], b["p_ref_better"])
        ok = boot == bays
        agree += ok
        print(f"  {name:16s} bootstrap={boot:6s} bayes={bays:6s} {'agree' if ok else 'DIFFER'}")
    print(f"\n{agree} of {len(rows)} comparisons agree on the most probable outcome")
    json.dump(rows, open(R + "/v3_paired_bootstrap.json", "w"), indent=1)
    print("wrote v3_paired_bootstrap.json")


if __name__ == "__main__":
    main()
