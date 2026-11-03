"""What changes if the policy is allowed to say "unknown"?

Every review of this work made the same point about the class-disjoint
experiment: the comparators are ordinary closed-set classifiers, so a withheld
hazardous class is forced through the actions they already know, and that is
forced routing rather than open-set detection. The recommended remedy was always
the same too, "add a reject-option baseline and define what a rejection maps to
operationally". This driver does that.

The design is deliberately the simplest thing that answers the question, because
the point is not to propose a good open-set method. It is to measure what a
reject option buys and what it costs on this benchmark.

    confidence      max predicted class probability, from predict_proba where a
                    comparator has one and from softmax over the action scores
                    where it has predict_q instead
    threshold       the q-th percentile of confidence on a CALIBRATION BAND, not
                    on the training windows. The band is a contiguous 20% tail of
                    each class block, separated from the fitting windows by the
                    same 20-window embargo the outer holdout uses, so the model
                    has not seen it and no near-duplicate of it. A paper whose
                    central lesson is that in-sample thresholds mislead cannot
                    set its own threshold in sample.
    sweep           q in {0, 5, 10, 20, 30}; q = 0 recovers the closed-set
                    comparator exactly, so that row reproduces the corresponding
                    entry of Table S13

A rejection is not an action. A facility has to decide what one means, and the
two defensible extremes bracket every policy in between, so both are reported:

    reject -> 0     treat an unfamiliar reading as nothing, the conservative
                    reading of "the model has no opinion"
    reject -> 3     treat an unfamiliar reading as alarm-grade, the conservative
                    reading of "something is here that I was not trained on"

Three quantities matter. On the held-out hazardous class: how often the model
rejects, and what missed-hazard rate and escalation adequacy become under each
policy. On the in-distribution clean windows the same seed held out: how often it
rejects there, because a reject option that fires on clean air is alarm burden
under another name.

Writes results_v2/v3_reject_option.csv.

    python3 -m retrain.run_reject_option
"""
import os, sys, time, inspect, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrain import raw_pipeline as rp
from retrain import rewards as RW
from retrain import calband as CB
from retrain.run_all_v2 import SEEDS, OUT, CACHE
from retrain.run_final import make          # the same constructor the class-disjoint table uses
from retrain.safety_metrics import evaluate_safety

# Comparators that expose a usable confidence. CUSUM and the depth-3 tree do not
# produce a calibrated score over actions, so they are out of scope here and
# their absence is stated in the manuscript rather than papered over.
MODELS = ["A_cost_weighted", "B_unweighted", "D_mlp", "E_gbm", "SVM", "RF", "KNN"]
QUANTILES = [0, 5, 10, 20, 30]
CAL_FRAC = 0.20                 # contiguous per-class calibration tail
REJECT_POLICIES = {"monitor": 0, "alarm": 3}


def confidence(model, X):
    """Max predicted class probability, however the comparator expresses it."""
    if hasattr(model, "predict_proba"):
        try:
            p = np.asarray(model.predict_proba(X), dtype=float)
            if p.ndim == 2 and p.shape[1] > 1:
                return p.max(axis=1)
        except Exception:
            pass
    if hasattr(model, "predict_q"):
        q = np.asarray(model.predict_q(X), dtype=float)
        q = q - q.max(axis=1, keepdims=True)
        e = np.exp(q)
        return (e / e.sum(axis=1, keepdims=True)).max(axis=1)
    return None


def scale_with(df, sc, p1, p99):
    d = df.copy()
    d[rp.ALL_FEAT_COLS] = sc.transform(d[rp.ALL_FEAT_COLS])
    d["anomaly"] = ((d["anomaly"] - p1) / (p99 - p1 + 1e-8)).clip(0, 1)
    return d


def main():
    am, asc = rp.build_anomaly_model()
    ds = rp.build_dataset(am, asc, cache_path=CACHE)
    rows = []

    for held in ["Smoke", "Mixture"]:
        hid = rp.GAS_MAP[held]
        for name in MODELS:
            t0 = time.time()
            per_seed = []
            for s in SEEDS:
                tr_df, te_df = rp.block_wise_holdout(ds, seed=s)
                tr3 = tr_df[tr_df["gas_id"] != hid].reset_index(drop=True)
                held_raw = ds[ds["gas_id"] == hid].reset_index(drop=True)
                # Carve the calibration band before anything is fitted: a
                # contiguous per-class tail, held off by the usual embargo. The
                # input scaler, the autoencoder and the anomaly normalization
                # are then fitted on the fitting windows alone, so the band is
                # out-of-sample for the whole pipeline and not only for the
                # classifier. Every frame scored afterwards, the withheld class
                # and the held-out retained-class windows included, passes
                # through that one fitted pipeline.
                fit_raw, cal_raw = CB.carve_calibration_band(tr3, frac=CAL_FRAC)
                te_df = te_df.reset_index(drop=True)
                fit_df, cal_df, held_raw, te_df = rp.partition_anomaly(
                    fit_raw, cal_raw, held_raw, te_df, seed=s)
                sc = StandardScaler().fit(fit_df[rp.ALL_FEAT_COLS])
                p1 = float(np.percentile(fit_df["anomaly"], 1))
                p99 = float(np.percentile(fit_df["anomaly"], 99))
                fit_df = scale_with(fit_df, sc, p1, p99)
                cal_df = scale_with(cal_df, sc, p1, p99)
                held_s = scale_with(held_raw, sc, p1, p99)
                # the clean windows this seed held out, for the cost side
                clean_df = te_df[te_df["gas_id"] == 0].reset_index(drop=True)
                if len(clean_df):
                    clean_s = scale_with(clean_df, sc, p1, p99)
                    Xc, _ = rp.to_arrays(clean_s)
                else:
                    Xc = None
                # ... and the held-out Perfume windows, the other nonhazardous class
                # (review comment 177: report the final alarm rate over all
                # nonhazardous windows, not only the rejection rate on clean air)
                perf_df = te_df[te_df["gas_id"] == rp.GAS_MAP["Perfume"]].reset_index(drop=True)
                if len(perf_df):
                    Xp, _ = rp.to_arrays(scale_with(perf_df, sc, p1, p99))
                else:
                    Xp = None

                Xtr, gtr = rp.to_arrays(fit_df)
                Xcal, _ = rp.to_arrays(cal_df)
                ytr = np.array([RW.rule_oracle_action(int(g)) for g in gtr])
                Xh, gh = rp.to_arrays(held_s)

                # Built exactly as run_final.exp_loco_all builds it, so the
                # q = 0 row of Table S13 reproduces the corresponding Table 6
                # entry rather than differing by a training-budget accident.
                m, kw = make(name, s, gtr)
                if "groups" in inspect.signature(m.fit).parameters:
                    kw.setdefault("groups", gtr)
                m.fit(Xtr, ytr, **kw)

                c_tr = confidence(m, Xcal)      # threshold set off-sample
                c_h = confidence(m, Xh)
                if c_tr is None or c_h is None:
                    break
                acts_h = np.asarray(m.predict(Xh))
                c_c = confidence(m, Xc) if Xc is not None else None
                a_c = np.asarray(m.predict(Xc)) if Xc is not None else None
                c_p = confidence(m, Xp) if Xp is not None else None
                a_p = np.asarray(m.predict(Xp)) if Xp is not None else None
                per_seed.append((c_tr, c_h, acts_h, gh, c_c, a_c, c_p, a_p))

            if not per_seed:
                print(f"  {held:8s}{name:16s} no usable confidence, skipped", flush=True)
                continue

            for q in QUANTILES:
                for pol, act_on_reject in REJECT_POLICIES.items():
                    miss, esc, und, rej_h, rej_c = [], [], [], [], []
                    fa_c, fa_p, fa_n = [], [], []
                    for c_tr, c_h, acts_h, gh, c_c, a_c, c_p, a_p in per_seed:
                        # tau is the q-th percentile of confidence on the calibration
                        # band; a window is rejected when its confidence is strictly
                        # below tau, so ties at tau are kept, and the realized
                        # calibration rejection rate can fall short of q% when
                        # confidence takes few distinct values (e.g. KNN).
                        tau = -np.inf if q == 0 else float(np.percentile(c_tr, q))
                        rejected = c_h < tau
                        # final alarm-grade indications on nonhazardous windows:
                        # a rejected window takes the fallback action, an accepted
                        # one keeps the model's own action
                        finals = []
                        for c_x, a_x in ((c_c, a_c), (c_p, a_p)):
                            if c_x is None:
                                finals.append(None); continue
                            fin = np.asarray(a_x).copy()
                            fin[c_x < tau] = act_on_reject
                            finals.append(fin >= 3)
                        fa_c.append(float(finals[0].mean()) if finals[0] is not None else np.nan)
                        fa_p.append(float(finals[1].mean()) if finals[1] is not None else np.nan)
                        both = [f for f in finals if f is not None]
                        fa_n.append(float(np.concatenate(both).mean()) if both else np.nan)
                        a = acts_h.copy()
                        a[rejected] = act_on_reject
                        e = evaluate_safety(gh, a)
                        miss.append(e["miss_rate"])
                        esc.append(e["escalation"])
                        und.append(e["under_escalation"])
                        rej_h.append(float(rejected.mean()))
                        rej_c.append(float((c_c < tau).mean()) if c_c is not None else np.nan)
                    rows.append(dict(
                        held_out=held, model=name, reject_quantile=q, reject_policy=pol,
                        reject_rate_heldout=float(np.mean(rej_h)),
                        reject_rate_clean=float(np.nanmean(rej_c)),
                        miss_rate_mean=float(np.mean(miss)), miss_rate_std=float(np.std(miss)),
                        escalation_mean=float(np.mean(esc)), escalation_std=float(np.std(esc)),
                        under_escalation_mean=float(np.mean(und)),
                        alarm_rate_clean_final=float(np.nanmean(fa_c)),
                        alarm_rate_perfume_final=float(np.nanmean(fa_p)),
                        alarm_rate_nonhazard_final=float(np.nanmean(fa_n))))
            r0 = [r for r in rows if r["held_out"] == held and r["model"] == name
                  and r["reject_quantile"] == 0 and r["reject_policy"] == "monitor"][0]
            r30 = [r for r in rows if r["held_out"] == held and r["model"] == name
                   and r["reject_quantile"] == 30 and r["reject_policy"] == "alarm"][0]
            print(f"  {held:8s}{name:16s} closed-set esc={r0['escalation_mean']:.4f}  "
                  f"q30/alarm esc={r30['escalation_mean']:.4f} "
                  f"(rejects {100*r30['reject_rate_heldout']:.1f}% held-out, "
                  f"{100*r30['reject_rate_clean']:.1f}% clean) [{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "v3_reject_option.csv"), index=False)
    print(f"\nwrote v3_reject_option.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
