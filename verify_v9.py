"""End-to-end numerical audit: every table in the manuscript against the stored
result files. Parses the markdown tables directly so nothing is retyped, and
fails loudly on any cell that does not reproduce."""
import re, json, sys
import numpy as np, pandas as pd

MD = "ARXIS_Manuscript_v9.md"
R = "retrain/results_v2/"
TXT = open(MD).read()
FAIL = []
N = [0]


def check(label, got, want, tol=6e-4):
    N[0] += 1
    ok = (got is not None and want is not None
          and abs(float(got) - float(want)) <= tol)
    if not ok:
        FAIL.append(f"{label}: manuscript {got} vs results {want}")
    return ok


def table_after(caption_prefix):
    """Return the markdown table immediately preceding a given bold caption."""
    i = TXT.index(caption_prefix)
    block = TXT[:i].rstrip().split("\n\n")[-1]
    rows = [r for r in block.strip().split("\n") if r.startswith("|")]
    out = []
    for r in rows[2:]:
        cells = [c.strip() for c in r.strip("|").split("|")]
        out.append([re.sub(r"[*_`]", "", c).strip() for c in cells])
    return out


def num(s):
    s = s.replace("\\", "").replace("%", "").replace("≤", "").replace(",", "")
    s = s.replace("−", "-").replace("pp", "").replace("reference", "nan")
    m = re.search(r"-?\d+\.?\d*", s)
    return float(m.group()) if m else None


PRETTY2KEY = {
    "Cost-weighted policy": "A_cost_weighted", "Unweighted network": "B_unweighted",
    "Multilayer perceptron": "D_mlp", "Gradient boosting": "E_gbm",
    "Support-vector classifier": "SVM", "Random forest": "RF",
    "Recurrent (LSTM)": "LSTM", "Conservative Q-learning": "CQL",
    "k-nearest neighbors": "KNN", "Threshold rule": "ThresholdRule",
    "Ordinal cost objective": "Ordinal", "Cost-weighted": "A_cost_weighted",
}

# ---------------------------------------------------------------- Table 5
lk = pd.read_csv(R + "v3_leakage.csv").pivot_table(
    index="model", columns="protocol", values="decision_acc_mean")
cols = ["random", "blocked_noembargo", "blocked_embargo", "leave_one_block"]
for row in table_after("**Table 5. Decision accuracy"):
    if row[0].lower() == "mean":
        for j, c in enumerate(cols):
            check(f"T5 mean {c}", num(row[j + 1]), lk[c].mean())
        continue
    k = PRETTY2KEY[row[0]]
    for j, c in enumerate(cols):
        check(f"T5 {k} {c}", num(row[j + 1]), lk.loc[k, c])

# ---------------------------------------------------------------- Table 6
pw = pd.read_csv(R + "v3_powered.csv").set_index("model")
bay = json.load(open(R + "v3_powered_bayes.json"))["vs_cost_weighted"]
for row in table_after("**Table 6. Model comparison"):
    k = PRETTY2KEY[row[0]]
    check(f"T6 {k} acc", num(row[1]), pw.loc[k, "decision_acc_mean"])
    check(f"T6 {k} sd", num(row[2]), pw.loc[k, "decision_acc_std"])
    check(f"T6 {k} miss", num(row[3]), pw.loc[k, "miss_rate_mean"])
    check(f"T6 {k} esc", num(row[4]), pw.loc[k, "escalation_mean"], 6e-4)
    check(f"T6 {k} fa", num(row[5]), pw.loc[k, "fa_per_clean_mean"])
    check(f"T6 {k} cost", num(row[6]), pw.loc[k, "expected_cost_mean"])
    if k != "A_cost_weighted":
        check(f"T6 {k} P(eq)", num(row[7]), bay[k]["p_practically_equivalent"], 6e-4)

# ---------------------------------------------------------------- Table 7
rho_rows = json.load(open(R + "v3_rho_sensitivity.json"))
RHO_COLS = [0.0, 0.10, 0.202, 0.30, 0.50]
for row in table_after("**Table 7. Posterior probability"):
    k = PRETTY2KEY[row[0]]
    for j, rho in enumerate(RHO_COLS):
        want = next(x["p_equivalent"] for x in rho_rows
                    if x["model"] == k and abs(x["rho"] - rho) < 1e-9)
        check(f"T7 {k} rho={rho}", num(row[j + 1]), want, 6e-4)
# the claim that rides on the sweep: no comparison changes direction over it
for k in ("RF", "ThresholdRule", "Ordinal", "B_unweighted", "D_mlp"):
    sub = [x for x in rho_rows if x["model"] == k]
    winners = {max(("other", x["p_other_better"]), ("equiv", x["p_equivalent"]),
                   ("ref", x["p_ref_better"]), key=lambda t: t[1])[0] for x in sub}
    if len(winners) != 1:
        FAIL.append(f"rho sweep: {k} changes direction across rho ({winners})")
    N[0] += 1

# ---------------------------------------------------------------- Table 8
lo = pd.read_csv(R + "v3_loco_all.csv")
for row in table_after("**Table 8. Leave-one-class-out"):
    cls, k = row[0], PRETTY2KEY[row[1]]
    s = lo[(lo.held_out == cls) & (lo.model == k)].iloc[0]
    check(f"T8 {cls}/{k} miss", num(row[2]), s.miss_rate_mean)
    check(f"T8 {cls}/{k} bound", num(row[3]), 100 * s.miss_upper95_pooled, 6e-2)
    check(f"T8 {cls}/{k} esc", num(row[4]), s.escalation_mean)
    check(f"T8 {cls}/{k} under", num(row[5]), s.under_escalation_mean)

# ---------------------------------------------------------------- Table 8
cs = pd.read_csv(R + "v2_costsweep.csv").set_index("cost_ratio")
for row in table_after("**Table 9. Cost-asymmetry sweep"):
    key = "label_function" if "Label" in row[0] else row[0].split()[0]
    s = cs.loc[key]
    check(f"T9 {key} acc", num(row[1]), s.decision_acc_mean)
    check(f"T9 {key} sd", num(row[2]), s.decision_acc_std)
    check(f"T9 {key} miss", num(row[3]), s.miss_rate_mean)
    check(f"T9 {key} esc", num(row[4]), s.escalation_mean)

# ---------------------------------------------------------------- Table 9
ab = pd.read_csv(R + "v2_ablation.csv").set_index("state")
STATE = {"Full state": "full_22", "Without per-sensor σ": "no_std_15",
         "Without anomaly score": "no_anomaly_21", "Without per-sensor δ": "no_delta_15",
         "Without δ and σ": "no_delta_std_8", "Current readings only": "current_only_7",
         "Anomaly score only": "anomaly_only_1"}
for row in table_after("**Table 10. Decision-state ablation"):
    s = ab.loc[STATE[row[0]]]
    check(f"T10 {row[0]} d", num(row[1]), s.n_features)
    check(f"T10 {row[0]} acc", num(row[2]), s.decision_acc_mean)
    check(f"T10 {row[0]} sd", num(row[3]), s.decision_acc_std)
    if "reference" not in row[4]:
        check(f"T10 {row[0]} delta", num(row[4]), s.delta_pp_vs_full, 6e-3)

# --------------------------------------------------------------- Table 10
pt = pd.read_csv(R + "v2_perturb.csv")
COND = {"Clean": ("clean", 0.0), "Drift ±50%": ("drift", 0.5),
        "Noise σ = 0.5": ("noise", 0.5), "Dropout k = 1": ("dropout", 1.0),
        "Dropout k = 3": ("dropout", 3.0), "Dropout k = 7": ("dropout", 7.0)}
for row in table_after("**Table 11. Selected perturbation"):
    cond = row[0].replace("*", "").strip()
    p, lv = COND[cond]
    k = PRETTY2KEY[row[1]]
    s = pt[(pt.perturbation == p) & (pt.level == lv) & (pt.model == k)].iloc[0]
    check(f"T11 {cond}/{k} acc", num(row[2]), s.decision_acc_mean, 6e-4)
    check(f"T11 {cond}/{k} miss", num(row[3]), s.miss_rate_mean, 6e-4)
    check(f"T11 {cond}/{k} esc", num(row[4]), s.escalation_mean, 6e-4)
    check(f"T11 {cond}/{k} fa", num(row[5]), s.fa_per_clean_mean, 6e-4)
    if "≤" in row[6]:
        # clean rows carry the Clopper-Pearson bound on an observed zero
        check(f"T11 {cond}/{k} bound", num(row[6]),
              1000 * (1 - 0.05 ** (1 / s.n_clean_pooled)), 0.05)
    else:
        check(f"T11 {cond}/{k} burden", num(row[6]), 1000 * s.fa_per_clean_mean, 0.6)

# --------------------------------------------------------------- Table 11
ca = pd.read_csv(R + "v2_calibration.csv").set_index("estimator")
EST = {"Deep ensemble (5)": "deep_ensemble_5", "MC dropout (20)": "mc_dropout_20",
       "Raw softmax": "raw_softmax", "Temperature scaling": "temp_scaling"}
for row in table_after("**Table 12. Calibration"):
    s = ca.loc[EST[row[0]]]
    check(f"T12 {row[0]} ece", num(row[1]), s.ece_mean)
    check(f"T12 {row[0]} sd", num(row[2]), s.ece_std)
    check(f"T12 {row[0]} lo", num(row[3].split(" to ")[0]), s.ece_lo)
    check(f"T12 {row[0]} hi", num(row[3].split(" to ")[1]), s.ece_hi)
    check(f"T12 {row[0]} danger", num(row[4]), s.ece_danger_mean)
    check(f"T12 {row[0]} brier", num(row[5]), s.brier_mean)
    check(f"T12 {row[0]} nll", num(row[6]), s.nll_mean)
    check(f"T12 {row[0]} acc", num(row[7]), s.acc_mean)

# ------------------------------------------------- narrative claims in prose
an = json.load(open(R + "v2_anomaly.json"))
check("AUC held-out", 0.9928, an["auc_mean"])
check("TPR held-out", 0.7363, an["tpr_mean"])
check("AUC in-sample", 0.962, an["in_sample"]["auc"], 6e-4)
check("FPR in-sample", 0.05, an["in_sample"]["fpr"], 6e-4)
for cls, v in [("NoGas", 0.736), ("Perfume", 1.543), ("Mixture", 117.72), ("Smoke", 229.25)]:
    check(f"recon {cls}", v, an["per_class_mean_recon"][cls], 6e-3)

ai = pd.read_csv(R + "v2_anomaly_influence.csv")
frac = ai.groupby("model").frac_windows_action_changed.first()
for k, v in [("E_gbm", 0.606), ("RF", 0.161), ("A_cost_weighted", 0.080), ("ThresholdRule", 0.0)]:
    check(f"anomaly influence {k}", v, frac[k], 6e-4)
g0 = ai[(ai.model == "E_gbm") & (ai.level == 0.0)].iloc[0]
check("GBM miss at anomaly=0", 0.3405, g0.miss_rate_mean)
check("GBM esc at anomaly=0", 0.495, g0.escalation_mean, 6e-4)

cb = pd.read_csv(R + "v3_cost_blocked_validation.csv").set_index("cost_ratio")
check("blocked-val 2:1 val", 0.9606, cb.loc["2:1", "val_acc_mean"])
check("blocked-val 2:1 test", 0.9430, cb.loc["2:1", "test_acc_mean"])
check("blocked-val 6:1 val", 0.9604, cb.loc["6:1", "val_acc_mean"])

cv = pd.read_csv(R + "v3_cost_validation.csv").set_index("cost_ratio")
lowC = cv.loc[["1:1", "2:1", "4:1", "6:1", "8:1"], "val_acc_mean"]
check("random-val floor", 0.9948, lowC.min())
check("random-val ceiling", 0.9994, lowC.max())

ps = pd.read_csv(R + "v2_calibration_perseed.csv")
T = ps[ps.estimator == "temp_scaling"]["T"]
check("temperature mean", 0.839, T.mean(), 6e-4)
check("temperature sd", 0.044, T.std(ddof=1), 6e-4)

# derived quantities quoted in prose
mx = lo[lo.held_out == "Mixture"].set_index("model")
check("escalation ratio", 198, mx.escalation_mean.max() / mx.escalation_mean.min(), 1.0)
check("30-run CP hazard bound %", 0.016, 100 * (1 - 0.05 ** (1 / 18960)), 6e-4)
check("30-run CP clean bound %", 0.032, 100 * (1 - 0.05 ** (1 / 9480)), 6e-4)
check("LOCO CP bound %", 0.038, 100 * (1 - 0.05 ** (1 / 7905)), 6e-4)
check("spread of ten comparators (pts)", 5.66,
      100 * (pw.drop(index="Ordinal").decision_acc_mean.max()
             - pw.drop(index="Ordinal").decision_acc_mean.min()), 6e-3)
gap = (lk["random"] - lk["blocked_embargo"]) * 100
check("leakage gap min", 1.66, gap.drop("ThresholdRule").min(), 6e-3)
check("leakage gap max", 4.08, gap.drop("ThresholdRule").max(), 6e-3)
check("leakage gap mean (learned)", 3.09, gap.drop("ThresholdRule").mean(), 6e-3)
check("leakage gap mean (all six)", 2.45, gap.mean(), 6e-3)
check("threshold rule loses", -0.74, gap["ThresholdRule"], 6e-3)
check("alarms per hour at k=1", 211,
      1000 * pt[(pt.perturbation == "dropout") & (pt.level == 1.0)
                & (pt.model == "A_cost_weighted")].fa_per_clean_mean.iloc[0] * 1.8, 1.0)
check("clean-window burden bound", 1.9, 1000 * (1 - 0.05 ** (1 / 1580)), 0.05)
check("GBM anomaly factor", 35, g0.miss_rate_mean / 0.0098, 0.6)

# ------------------------------------------- submission-copy hygiene
if "CITATION NEEDED" in TXT:
    FAIL.append(f"{TXT.count('CITATION NEEDED')} unresolved CITATION NEEDED marker(s)")
N[0] += 1
for phrase in ("Items Still Requiring Author Action", "Author Verification Items",
               "needing author sign-off", "not intended for publication"):
    if phrase in TXT:
        FAIL.append(f"internal author-work text still in the manuscript: {phrase!r}")
    N[0] += 1
# the thermal-image claim the authors corrected: the images were used
for phrase in ("image corpus was not available", "image corpus was unavailable",
               "could not be corrected"):
    if phrase in TXT:
        FAIL.append(f"stale thermal-image availability claim: {phrase!r}")
    N[0] += 1
if "Liang et al. (2024)" in TXT:
    FAIL.append("Liang et al. is 2023, not 2024")
N[0] += 1

# structural checks on the document itself
caps = [int(m) for m in re.findall(r"^\*\*Fig\. (\d+)\.", TXT, re.M)]
tabs = [int(m) for m in re.findall(r"^\*\*Table (\d+)\.", TXT, re.M)]
if caps != list(range(1, len(caps) + 1)): FAIL.append(f"figure numbering: {caps}")
if tabs != list(range(1, len(tabs) + 1)): FAIL.append(f"table numbering: {tabs}")
if len(re.findall(r"!\[\]\(", TXT)) != len(caps): FAIL.append("figure/image count mismatch")
if "—" in TXT: FAIL.append("em-dash present")
if re.search(r"\*\*(?:Table|Fig)\. ?\d+\.\.", TXT): FAIL.append("double-period caption")

print(f"{N[0]} numerical checks run, {len(FAIL)} failure(s)")
for f in FAIL:
    print("  FAIL", f)
sys.exit(1 if FAIL else 0)
