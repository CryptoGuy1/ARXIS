"""Run every experiment target, in one environment, in dependency order.

The same steps as `make all` up to the manifest, for machines without GNU Make
(Windows included). Resumable: each finished target is recorded in
run_records/pipeline_state.json, and a restart skips it, so an interrupted run
continues where it stopped. The 30-run comparison inside `final` also
checkpoints after every run.

    python run_pipeline.py              # run everything not yet done
    python run_pipeline.py --status     # show what is done and what is left
    python run_pipeline.py --from final # (re)run from one target onwards
    python run_pipeline.py --only rawfaults
    python run_pipeline.py --restart    # forget the state file and start over

Every target writes its own log to run_records/logs/<target>.log. A failing
target stops the run; fix the cause and run the same command again.

This deliberately stops before `make check`. The manuscript still quotes the
v18.1 numbers, so verification is expected to fail until it is revised from the
new result files.
"""
import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
REC = os.path.join(ROOT, "run_records")
LOGS = os.path.join(REC, "logs")
STATE = os.path.join(REC, "pipeline_state.json")

# (target, [command argument lists]) in an order that satisfies every Makefile
# dependency: final before splitcounts/bayes/rho/rope/bounds/bootstrap/rungcheck,
# calibration before nonhazard, comparison before everything that reads its
# cached features.
TARGETS = [
    ("comparison",     [["-m", "retrain.run_all_v2"]]),
    ("perseed",        [["-m", "retrain.run_perturb_perseed"]]),
    ("classdisjoint",  [["-m", "retrain.run_loco_v3"]]),
    ("leakage",        [["-m", "retrain.run_leakage"]]),
    ("calibration",    [["-m", "retrain.run_calib_actions_v2"]]),
    ("anomaly",        [["-m", "retrain.run_anomaly_influence"]]),
    ("episodes",       [["-m", "retrain.run_alarm_episodes"]]),
    ("hazardepisodes", [["-m", "retrain.run_episode_metrics"]]),
    ("blockboot",      [["-m", "retrain.run_block_bootstrap"]]),
    ("final",          [["-m", "retrain.run_final"]]),
    ("splitcounts",    [["-m", "retrain.run_split_counts"]]),
    ("bayes",          [["-m", "retrain.run_bayes_recompute"]]),
    ("rho",            [["-m", "retrain.run_rho_sensitivity"]]),
    ("rope",           [["-m", "retrain.run_rope_sensitivity"]]),
    ("bounds",         [["retrain/run_partition_bounds.py"]]),
    ("bootstrap",      [["retrain/run_paired_bootstrap.py"]]),
    ("anomalyclass",   [["-m", "retrain.run_anomaly_by_class"]]),
    ("metriccost",     [["-m", "retrain.run_metric_cost"]]),
    ("persistence",    [["-m", "retrain.run_persistence_sweep"]]),
    ("reject",         [["-m", "retrain.run_reject_option"]]),
    ("rawfaults",      [["-m", "retrain.run_raw_faults"]]),
    ("overlap",        [["-m", "retrain.run_overlap_isolation"]]),
    ("weighting",      [["-m", "retrain.run_weighting_probe"]]),
    ("nonhazard",      [["-m", "retrain.run_nonhazard_alarm"]]),
    ("rungcheck",      [["-m", "retrain.run_rung_check"]]),
    ("ratiocheck",     [["-m", "retrain.run_ratio_check"]]),
    ("figures",        [["-m", "retrain.make_figs_v2"], ["-m", "retrain.make_figs_arch"]]),
    ("manifest",       [["make_manifest.py"]]),
    ("test",           [["test_safety_metrics.py"], ["test_rerun_fixes.py"]]),
]
NAMES = [t for t, _ in TARGETS]


def now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def load_state():
    if os.path.isfile(STATE):
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    return {"done": {}, "environment": None}


def save_state(st):
    os.makedirs(REC, exist_ok=True)
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATE)


def environment():
    env = dict(python=sys.version.split()[0], executable=sys.executable,
               platform=platform.platform(), processor=platform.processor(),
               cpu_count=os.cpu_count())
    for pkg in ("numpy", "pandas", "scipy", "sklearn", "torch", "matplotlib"):
        try:
            mod = __import__(pkg)
            env[pkg] = getattr(mod, "__version__", "?")
        except Exception as exc:          # report, do not hide, a missing package
            env[pkg] = f"unavailable: {exc}"
    try:
        import torch
        env["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        pass
    try:
        env["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           capture_output=True, text=True).stdout.strip()
    except Exception:
        env["git_commit"] = None
    return env


def run_target(name, commands):
    os.makedirs(LOGS, exist_ok=True)
    path = os.path.join(LOGS, f"{name}.log")
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as log:
        for args in commands:
            cmd = [sys.executable] + args
            log.write(f"\n[{now()}] $ {' '.join(cmd)}\n"); log.flush()
            print(f"[{now()}] {name}: {' '.join(args)}  (log: {os.path.relpath(path, ROOT)})", flush=True)
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace", env=env)
            for line in p.stdout:
                log.write(line); log.flush()
                if line.strip():
                    print(f"    {line.rstrip()}", flush=True)
            rc = p.wait()
            log.write(f"[{now()}] exit {rc}\n"); log.flush()
            if rc != 0:
                return False, time.time() - t0
    return True, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--from", dest="start", choices=NAMES)
    ap.add_argument("--only", choices=NAMES)
    ap.add_argument("--restart", action="store_true")
    a = ap.parse_args()

    if a.restart and os.path.isfile(STATE):
        os.remove(STATE)
    st = load_state()

    if a.status:
        for name in NAMES:
            d = st["done"].get(name)
            print(f"  {'done' if d else 'todo':4s}  {name:15s}"
                  + (f"  {d['finished']}  {d['seconds'] / 3600:.2f} h" if d else ""))
        return 0

    env = environment()
    if st.get("environment") and any(st["environment"].get(k) != env.get(k)
                                     for k in ("python", "numpy", "pandas", "scipy", "sklearn", "torch")):
        print("This state file was started in a different environment:")
        print("  then:", {k: st["environment"].get(k) for k in ("python", "numpy", "sklearn", "torch")})
        print("  now: ", {k: env.get(k) for k in ("python", "numpy", "sklearn", "torch")})
        print("Every table must come from one environment. Run with --restart to begin again here,")
        print("or switch back to the environment the run started in.")
        return 2
    st["environment"] = st.get("environment") or env
    save_state(st)
    print("environment:", json.dumps({k: env[k] for k in ("python", "torch", "sklearn", "numpy", "cpu_count", "git_commit")}))

    if a.only:
        todo = [a.only]
    elif a.start:
        todo = NAMES[NAMES.index(a.start):]
        for name in todo:
            st["done"].pop(name, None)
        save_state(st)
    else:
        todo = [n for n in NAMES if n not in st["done"]]
    commands = dict(TARGETS)
    for name in todo:
        ok, secs = run_target(name, commands[name])
        if not ok:
            print(f"\nFAILED: {name} after {secs / 60:.1f} min. See run_records/logs/{name}.log.")
            print("Fix the cause and run the same command again; finished targets are skipped.")
            return 1
        st["done"][name] = dict(finished=now(), seconds=round(secs, 1))
        save_state(st)
        print(f"[{now()}] {name} done in {secs / 60:.1f} min", flush=True)
    left = [n for n in NAMES if n not in st["done"]]
    print("\nall targets finished." if not left else f"\nstill to run: {', '.join(left)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
