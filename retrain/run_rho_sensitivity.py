"""Does the equivalence conclusion depend on the correlation term?

Benavoli et al. (2017) derive the correlated t-test for correlated resampling,
canonically k-fold cross-validation, where the correlation between runs comes
from overlapping training sets and is approximated by rho = n_test / n_total.
Our design is not k-fold: seed, held-out block position and a hyperparameter
draw all vary per run. Training partitions still overlap heavily, roughly 80%
of the same windows every run, so a positive correlation is real, but the
particular value of rho is a heuristic carried over rather than derived for
this design. A referee is entitled to ask what rides on it.

This sweeps rho from 0 (the naive independent t-test, which is known to be
anti-conservative here) to 0.5 and reports the three posterior probabilities
at each value, so the reader can see which conclusions are stable and which
are artifacts of the choice.
"""
import json, os
import numpy as np
from scipy.stats import t as student_t

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_v2")
ROPE = 0.01
# The deployed value is n_test/(n_train + n_test) computed exactly; using the
# rounded 0.202 here instead put the sweep's own column four decimal places
# away from the main comparison, which is a difference a reader would have to
# explain to themselves. Sweep the exact value.
from retrain.run_split_counts import rho as _rho_from_counts
RHO_DEPLOYED = _rho_from_counts()   # plan item A5: read from v3_split_counts.json, never hard-coded
RHOS = [0.0, 0.05, 0.10, RHO_DEPLOYED, 0.30, 0.40, 0.50]


def bayes_correlated_t(diff, rho, rope=ROPE):
    d = np.asarray(diff, float); n = len(d)
    m = d.mean(); v = d.var(ddof=1)
    if v <= 0:
        return (0.0, 1.0, 0.0) if abs(m) < rope else ((1.0, 0.0, 0.0) if m < 0 else (0.0, 0.0, 1.0))
    scale = np.sqrt(v * (1.0 / n + rho / (1.0 - rho)))
    lo = student_t.cdf((-rope - m) / scale, n - 1)
    hi = 1.0 - student_t.cdf((rope - m) / scale, n - 1)
    return float(lo), float(1.0 - lo - hi), float(hi)


def main():
    blob = json.load(open(os.path.join(R, "v3_powered_bayes.json")))
    acc = blob["per_run_acc"]
    ref = np.asarray(acc["A_cost_weighted"], float)
    rows = []
    for name, vals in acc.items():
        if name == "A_cost_weighted":
            continue
        diff = ref - np.asarray(vals, float)
        for rho in RHOS:
            o, e, r = bayes_correlated_t(diff, rho)
            rows.append(dict(model=name, rho=rho, p_other_better=o,
                             p_equivalent=e, p_ref_better=r))
    out = os.path.join(R, "v3_rho_sensitivity.json")
    json.dump(rows, open(out, "w"), indent=1)

    print(f"{'model':16s} " + " ".join(f"{r:>7.3f}" for r in RHOS))
    print(" " * 16 + "   P(practically equivalent to the cost-weighted policy)")
    for name in acc:
        if name == "A_cost_weighted":
            continue
        sub = [x for x in rows if x["model"] == name]
        print(f"{name:16s} " + " ".join(f"{x['p_equivalent']:7.3f}" for x in sub))
    print()
    print("Sign of the resolved comparisons across the whole sweep:")
    for name in ("RF", "ThresholdRule", "Ordinal"):
        sub = [x for x in rows if x["model"] == name]
        best = ["other" if x["p_other_better"] > max(x["p_equivalent"], x["p_ref_better"])
                else ("equiv" if x["p_equivalent"] > x["p_ref_better"] else "ref") for x in sub]
        print(f"  {name:16s} {best}")
    print(f"\nwrote {os.path.basename(out)}")


if __name__ == "__main__":
    main()
