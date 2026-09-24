"""Recompute the Bayesian posteriors at the corrected correlation term.

Pre-submission plan item A5, second half. Correcting rho changes every posterior
probability in Table 8, in the rho sweep and in the ROPE sweep. It does not
change a single model fit: the correlated t-test reads only the vector of
per-run decision accuracies, and those are stored in v3_powered_bayes.json from
the original 30-run execution. Retraining to change an arithmetic constant would
burn hours and, worse, would produce a different set of accuracies that nobody
could reconcile against the published table.

So the accuracies are held fixed and only the posterior arithmetic is redone.
The file's `rho` field is updated in place from results_v2/v3_split_counts.json,
and the previous value is recorded under `rho_superseded` so the change is
visible in the artifact rather than only in the paper.

Run `python3 -m retrain.run_split_counts` first. Then:

    python3 -m retrain.run_bayes_recompute
    python3 -m retrain.run_rho_sensitivity
    python3 -m retrain.run_rope_sensitivity
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import t as student_t

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrain.run_all_v2 import OUT
from retrain.run_split_counts import load as load_counts

ROPE = 0.01
REF = "A_cost_weighted"


def bayes_correlated_t(diff, rho, rope=ROPE):
    d = np.asarray(diff, float)
    n = len(d)
    m, v = d.mean(), d.var(ddof=1)
    if v <= 0:
        if abs(m) < rope:
            return 0.0, 1.0, 0.0
        return (1.0, 0.0, 0.0) if m < 0 else (0.0, 0.0, 1.0)
    scale = np.sqrt(v * (1.0 / n + rho / (1.0 - rho)))
    lo = float(student_t.cdf((-rope - m) / scale, n - 1))
    hi = float(1.0 - student_t.cdf((rope - m) / scale, n - 1))
    return lo, float(1.0 - lo - hi), hi


def main():
    counts = load_counts()
    rho_new = counts["rho"]
    path = os.path.join(OUT, "v3_powered_bayes.json")
    blob = json.load(open(path))
    rho_old = blob.get("rho")
    acc = blob["per_run_acc"]
    ref = np.asarray(acc[REF], float)

    print(f"rho  {rho_old!r}  ->  {rho_new!r}")
    print(f"     from n_test {counts['n_test']} / (n_train {counts['n_train']} + n_test)")
    print(f"{'model':17s} {'P(equiv) old':>13s} {'P(equiv) new':>13s} {'shift':>9s}")

    sig, moved = {}, 0
    for name, vals in acc.items():
        if name == REF:
            continue
        d = ref - np.asarray(vals, float)
        lo, mid, hi = bayes_correlated_t(d, rho_new)
        was = blob["vs_cost_weighted"].get(name, {}).get("p_practically_equivalent")
        sig[name] = dict(p_other_better=lo, p_practically_equivalent=mid,
                         p_ref_better=hi, mean_diff=float(d.mean()))
        if was is not None:
            shift = mid - was
            if abs(shift) > 5e-4:
                moved += 1
            print(f"{name:17s} {was:13.4f} {mid:13.4f} {shift:+9.4f}")

    blob["rho"] = rho_new
    blob["rho_superseded"] = rho_old
    blob["rho_source"] = "v3_split_counts.json"
    blob["vs_cost_weighted"] = sig
    json.dump(blob, open(path, "w"), indent=2)
    print(f"\n{moved} of {len(sig)} equivalence probabilities moved by more than 0.0005")
    print("wrote v3_powered_bayes.json")


if __name__ == "__main__":
    main()
