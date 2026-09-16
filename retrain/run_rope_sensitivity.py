"""How much of the equivalence interpretation rests on the width of the ROPE.

Reviewer comment 4.6. The region of practical equivalence used throughout is
+/- 1 accuracy point, which is an engineering judgment about what would change a
procurement decision, not a constant anyone derived. If the equivalence claims
survive only at exactly that width they are claims about the threshold rather
than about the models, and a reader is entitled to see which it is.

The sweep runs the same two analyses at +/- 0.5, +/- 1 and +/- 2 accuracy points:
the Bayesian correlated t-test of Benavoli et al. (2017) at the deployed rho, and
the paired bootstrap over runs. For each comparison and each width it records the
three posterior masses and the most probable outcome, so the table can show where
a conclusion flips.

Writes results_v2/v3_rope_sensitivity.json.
"""
import json, os
import numpy as np
from scipy import stats

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_v2")
ROPES = [0.005, 0.01, 0.02]
B, SEED = 20000, 12345
from retrain.run_split_counts import rho as _rho_from_counts
RHO_DEPLOYED = _rho_from_counts()   # plan item A5: read from v3_split_counts.json, never hard-coded


def bayes_masses(d, rho, rope):
    """Posterior masses under the correlated t-test: (ref better, equivalent, other better)."""
    n = len(d)
    mean = float(d.mean())
    s2 = float(d.var(ddof=1))
    s_post = np.sqrt(s2 * (1.0 / n + rho / (1.0 - rho)))
    if s_post <= 0:
        return (1.0, 0.0, 0.0) if mean > rope else ((0.0, 0.0, 1.0) if mean < -rope else (0.0, 1.0, 0.0))
    t = stats.t(df=n - 1, loc=mean, scale=s_post)
    p_lo = float(t.cdf(-rope))
    p_hi = float(1.0 - t.cdf(rope))
    return p_hi, float(1.0 - p_lo - p_hi), p_lo


def pick(p_ref, p_eq, p_oth):
    return max((p_oth, "other"), (p_eq, "equiv"), (p_ref, "ref"), key=lambda t: t[0])[1]


def main():
    acc = json.load(open(R + "/v3_powered_bayes.json"))["per_run_acc"]
    ref = np.asarray(acc["A_cost_weighted"], float)
    n = len(ref)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(B, n))

    out, flips = {}, []
    print(f"ROPE sensitivity, {n} runs, rho = {RHO_DEPLOYED:.4f}, {B} bootstrap replicates\n")
    hdr = f"{'model':16s} {'ROPE':>6s} {'bayes P(equiv)':>15s} {'boot P(equiv)':>14s} {'bayes':>7s} {'boot':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for name, vals in acc.items():
        if name == "A_cost_weighted":
            continue
        d = ref - np.asarray(vals, float)
        reps = d[idx].mean(axis=1)
        out[name] = {}
        picks = []
        for rope in ROPES:
            b_ref, b_eq, b_oth = bayes_masses(d, RHO_DEPLOYED, rope)
            k_ref = float((reps > rope).mean())
            k_eq = float((np.abs(reps) <= rope).mean())
            k_oth = float((reps < -rope).mean())
            pb, pk = pick(b_ref, b_eq, b_oth), pick(k_ref, k_eq, k_oth)
            picks.append(pb)
            out[name][f"{rope:.3f}"] = dict(
                bayes=dict(p_ref_better=b_ref, p_practically_equivalent=b_eq, p_other_better=b_oth,
                           most_probable=pb),
                bootstrap=dict(p_ref_better=k_ref, p_equivalent=k_eq, p_other_better=k_oth,
                               most_probable=pk))
            print(f"{name:16s} {rope*100:5.1f}p {b_eq:15.3f} {k_eq:14.3f} {pb:>7s} {pk:>7s}")
        if len(set(picks)) > 1:
            flips.append((name, picks))
        print()

    print("comparisons whose most probable outcome changes across the ROPE sweep:")
    if flips:
        for name, picks in flips:
            print(f"  {name:16s} " + " -> ".join(f"{r*100:.1f}p:{p}" for r, p in zip(ROPES, picks)))
    else:
        print("  none; every comparison keeps its most probable outcome from 0.5 to 2 accuracy points")
    out["_meta"] = dict(rope_widths=ROPES, rho=RHO_DEPLOYED, n_runs=n, bootstrap_replicates=B,
                        n_flipping=len(flips), flipping=[f[0] for f in flips])
    json.dump(out, open(R + "/v3_rope_sensitivity.json", "w"), indent=1)
    print("\nwrote v3_rope_sensitivity.json")


if __name__ == "__main__":
    main()
