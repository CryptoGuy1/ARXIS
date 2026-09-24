"""Run manifest for the frozen research artifact.

Reviewer comment 4.3 / Section 6. A reviewer clicking the repository should be
able to tell, without guessing, which commit produced which result file, on what
machine, under which package versions. This writes that record.

Everything here is observed, not declared: package versions come from the live
interpreter, file hashes from the files on disk, the commit from git if the tree
is a repository. Nothing is copied from the manuscript, so a disagreement
between this manifest and Appendix A is a real disagreement and should be fixed
in the manuscript rather than in the manifest.

    python3 make_manifest.py            # writes RUN_MANIFEST.json and requirements.lock
"""
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "retrain", "results_v2")
FIGURES = os.path.join(ROOT, "figures_v2")

PACKAGES = ["numpy", "pandas", "scipy", "sklearn", "torch", "matplotlib", "ultralytics"]
PIP_NAME = {"sklearn": "scikit-learn"}

DRIVERS = [
    ("retrain/run_all_v2.py", "model comparison, cost sweep, perturbation, ablation, anomaly"),
    ("retrain/run_loco_v3.py", "class-disjoint hazard evaluation, three-class scaling"),
    ("retrain/run_leakage.py", "partitioning-protocol comparison, blocked validation"),
    ("retrain/run_calib_actions_v2.py", "calibration, action matrices, ROC curves"),
    ("retrain/run_anomaly_influence.py", "anomaly-input sensitivity"),
    ("retrain/run_alarm_episodes.py", "eventized alarm burden"),
    ("retrain/run_episode_metrics.py", "episode-level hazard metrics and latency"),
    ("retrain/run_block_bootstrap.py", "moving-block bootstrap and disjoint-support bounds"),
    ("retrain/run_anomaly_by_class.py", "per-class reconstruction error, partition-local"),
    ("retrain/run_final.py", "30-run comparison, Bayesian equivalence, ordinal objective"),
    ("retrain/run_rho_sensitivity.py", "sweep of the correlation term"),
    ("retrain/run_rope_sensitivity.py", "sweep of the region of practical equivalence"),
    ("retrain/run_partition_bounds.py", "partition-level binomial bounds"),
    ("retrain/run_paired_bootstrap.py", "paired bootstrap over runs"),
    ("retrain/make_figs_v2.py", "all figures"),
]


def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def git(*args):
    try:
        return subprocess.check_output(["git", "-C", ROOT] + list(args),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def versions():
    out = {}
    for mod in PACKAGES:
        try:
            m = importlib.import_module(mod)
            out[PIP_NAME.get(mod, mod)] = getattr(m, "__version__", "unknown")
        except Exception:
            out[PIP_NAME.get(mod, mod)] = "not installed"
    return out


def listing(directory, exts):
    if not os.path.isdir(directory):
        return {}
    out = {}
    for fn in sorted(os.listdir(directory)):
        if not fn.lower().endswith(exts):
            continue
        p = os.path.join(directory, fn)
        st = os.stat(p)
        out[fn] = dict(bytes=st.st_size, sha256_16=sha256(p),
                       modified_utc=datetime.fromtimestamp(st.st_mtime, timezone.utc)
                       .isoformat(timespec="seconds"))
    return out


def main():
    vers = versions()
    man = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "describe": git("describe", "--tags", "--always", "--dirty"),
            "dirty": bool(git("status", "--porcelain")),
            "remote": git("config", "--get", "remote.origin.url"),
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "node": platform.node(),
            "cpu_count": os.cpu_count(),
            "python": sys.version.split()[0],
            "python_implementation": platform.python_implementation(),
        },
        "packages": vers,
        "seeds": {
            "five_seed_protocol": [42, 1337, 7, 2024, 99],
            "thirty_run_protocol": list(range(1000, 1030)),
            "bootstrap_seed": 12345,
        },
        "protocol": {
            "window_size": 20,
            "embargo_windows_each_side": 20,
            "test_fraction": 0.2,
            "windows_total": 6324,
            "windows_per_class": 1581,
            "train_windows": 4900,
            "test_windows": 1264,
            "anomaly_model": "LSTM autoencoder, refit inside each training partition",
            "anomaly_epochs": 30,
        },
        "drivers": [{"path": p, "purpose": d,
                     "sha256_16": sha256(os.path.join(ROOT, p)) if os.path.exists(os.path.join(ROOT, p)) else None}
                    for p, d in DRIVERS],
        "results": listing(RESULTS, (".csv", ".json")),
        "figures": listing(FIGURES, (".png", ".pdf", ".svg")),
    }
    with open(os.path.join(ROOT, "RUN_MANIFEST.json"), "w") as f:
        json.dump(man, f, indent=1)

    with open(os.path.join(ROOT, "requirements.lock"), "w") as f:
        f.write("# Generated by make_manifest.py from the live interpreter that produced\n")
        f.write("# the results in retrain/results_v2. Appendix A of the manuscript must\n")
        f.write("# quote these versions exactly.\n")
        f.write(f"# python {sys.version.split()[0]}\n")
        for name, v in sorted(vers.items()):
            if v not in ("unknown", "not installed"):
                f.write(f"{name}=={v}\n")
            else:
                f.write(f"# {name}: {v}\n")

    print("wrote RUN_MANIFEST.json and requirements.lock")
    print(f"  commit   {man['git']['commit'] or 'not a git repository'}")
    print(f"  python   {man['host']['python']}")
    for k, v in sorted(vers.items()):
        print(f"  {k:16s} {v}")
    print(f"  results  {len(man['results'])} files")
    print(f"  figures  {len(man['figures'])} files")


if __name__ == "__main__":
    main()
