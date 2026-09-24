# Exp #3 LOCO — Provenance Note

**File:** `retrain/results/exp3_loco.csv` (16 rows: 4 held-out classes × 4 models)  
**Generated:** 2026-08-31

## What the data shows
Leave-one-class-out evaluation: for each held-out gas (NoGas, Smoke, Mixture, Perfume),
train all four comparator models (A=asym DQN, B=plain DQN, D=MLP, E=GBM) on the
other three classes and evaluate on the held-out one. The headline result: held-out
**danger** classes (Smoke, Mixture) achieve ~zero danger-miss across all models
(fail-safe escalation), while held-out accuracy is ~0 (the agent never saw that class).

## How it was produced
The LOCO grid (4 classes × 4 models × 5 seeds × 80 epochs) is too large to run as
one uninterrupted process on this Windows/Cygwin box: the Cygwin bash session
exhausts its fork/process-table capacity ~5–10 min into any training child and
kills the process tree (observed as `fork: Permission denied` / the python process
dying with no traceback). This is an environment limit, not a code bug — the
driver runs correctly up to the kill point.

To produce a complete result, Exp #3 was executed as **independent per-class
trainings** (one complete LOCO training per held-out gas, each ~4 min and
finishing all four comparator models before the limit hits). Each row in the CSV
comes from a **completed model training** (all 5 seeds, all epochs). The per-class
results were aggregated into the single 16-row `exp3_loco.csv`.

The Clopper–Pearson bounds (`exp3_clopper_pearson.csv`/`.json`) are deterministic
functions of the danger-row counts and observed misses — recomputed directly from
the final CSV, not from a separate run.

## Honest caveat
The numbers are correct (each from a finished training) and deterministic, but the
provenance is "aggregated from independent per-class trainings" rather than "one
monolithic run with exit=0." If a single uninterrupted run is required for
publication, it must be run on a machine without this Cygwin fork limit (Linux,
or a fresh long-lived shell). The data itself does not change — LOCO is
deterministic given the seed-varying split.
