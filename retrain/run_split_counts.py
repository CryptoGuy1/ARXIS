"""The correlation term in Eq. 11 should come from the partition, not from a constant.

Pre-submission plan item A5. The manuscript reported rho = n_test/(n_train +
n_test) = 0.202, while the training and test counts it also reports, 4,900 and
1,264, give 0.20506. The discrepancy traced to a hard-coded pair of counts in the
comparison driver, one of which (4,980) never matched the partition the pipeline
actually produces. The numerical effect on the posteriors is small, but a formal
equation in a paper that spends its length on statistical care cannot disagree
with its own reported counts.

This driver removes the constant. It builds the partition the same way every
experiment does, checks that the counts do not move with the seed, and writes
them out. Every consumer of rho reads the result rather than carrying its own
copy, so the value in the manuscript, in the sweeps and in the comparison driver
cannot drift apart again.

Writes results_v2/v3_split_counts.json.

    python3 -m retrain.run_split_counts
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrain import raw_pipeline as rp
from retrain.run_all_v2 import OUT, CACHE, SEEDS

SEEDS30 = list(range(1000, 1030))
OUTFILE = "v3_split_counts.json"


def load():
    """The stored counts, for any driver that needs rho."""
    with open(os.path.join(OUT, OUTFILE)) as f:
        return json.load(f)


def rho():
    return load()["rho"]


def main():
    am, asc = rp.build_anomaly_model()
    ds = rp.build_dataset(am, asc, cache_path=CACHE)
    seen = {}
    for s in list(SEEDS) + SEEDS30:
        tr, te = rp.block_wise_holdout(ds, seed=s)
        seen.setdefault((len(tr), len(te)), []).append(s)

    if len(seen) != 1:
        print("partition size varies with the seed:")
        for k, v in seen.items():
            print(f"  train {k[0]}, test {k[1]}: seeds {v[:4]}...")
        raise SystemExit("rho is not a single number for this design; fix before quoting one")

    (n_tr, n_te), seeds = next(iter(seen.items()))
    out = dict(n_windows_total=int(len(ds)), n_train=int(n_tr), n_test=int(n_te),
               rho=n_te / (n_tr + n_te), n_seeds_checked=len(seeds),
               protocols=["five-seed", "thirty-run"])
    with open(os.path.join(OUT, OUTFILE), "w") as f:
        json.dump(out, f, indent=1)
    print(f"windows total  {out['n_windows_total']}")
    print(f"train          {out['n_train']}")
    print(f"test           {out['n_test']}")
    print(f"rho            {out['rho']:.17g}")
    print(f"checked across {out['n_seeds_checked']} seeds, identical in all")
    print(f"wrote {OUTFILE}")


if __name__ == "__main__":
    main()
