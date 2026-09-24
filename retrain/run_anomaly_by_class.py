"""Per-class mean reconstruction error under the partition-local anomaly model.

The descriptive "does the anomaly score track hazard?" figure was previously
computed with the corpus-wide autoencoder, which is the model reviewer comment
4.2 rules out. This recomputes it the way every other anomaly number in the paper
is now computed: the autoencoder is fitted on the training partition's nominal
windows only, then scores the held-out partition, and the per-class means are
averaged over the five seed-varying partitions.

Appends `per_class_mean_recon_partition` to results_v2/v2_anomaly.json.
"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch

torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrain import raw_pipeline as rp
from retrain.run_all_v2 import SEEDS, OUT, CACHE


def main():
    am, asc = rp.build_anomaly_model()
    ds = rp.build_dataset(am, asc, cache_path=CACHE)
    per_seed = {g: [] for g in rp.GAS_ORDER}
    for s in SEEDS:
        tr, te = rp.block_wise_holdout(ds, seed=s)
        tr, te = rp.partition_anomaly(tr, te, seed=s)
        for g in rp.GAS_ORDER:
            per_seed[g].append(float(te.loc[te.label == g, "anomaly"].mean()))
        print(f"  seed {s:5d} " + "  ".join(
            f"{g}={per_seed[g][-1]:.4g}" for g in rp.GAS_ORDER), flush=True)

    out = {g: dict(mean=float(np.mean(v)), std=float(np.std(v)), per_seed=v)
           for g, v in per_seed.items()}
    path = os.path.join(OUT, "v2_anomaly.json")
    d = json.load(open(path))
    d["per_class_mean_recon_partition"] = out
    json.dump(d, open(path, "w"), indent=1)

    print("\nper-class mean reconstruction error, held-out windows, "
          "autoencoder fitted on training-partition nominal windows only:")
    for g in rp.GAS_ORDER:
        print(f"  {g:9s} {out[g]['mean']:12.4f} +/- {out[g]['std']:.4f}")
    haz = np.mean([out['Smoke']['mean'], out['Mixture']['mean']])
    ben = np.mean([out['NoGas']['mean'], out['Perfume']['mean']])
    print(f"\nhazardous / benign ratio: {haz/ben:.0f}x")
    print("wrote per_class_mean_recon_partition into v2_anomaly.json")


if __name__ == "__main__":
    main()
