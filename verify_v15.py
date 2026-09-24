"""Self-evaluation for the SPE-format manuscript.

Three families of check, all run against the file itself so nothing is taken on
trust: (A) every number in every table reproduces from the stored result files,
(B) the document conforms to the house style of the model paper, and (C) the
submission-hygiene rules hold. Exit code is nonzero if anything fails, so the
rewrite loop has an objective stopping condition rather than an opinion.
"""
import re, json, sys, os
import numpy as np, pandas as pd
from scipy.stats import beta


def cp_upper(k, n, conf=0.95):
    if n <= 0:
        return float("nan")
    if k == 0:
        return 1.0 - (1.0 - conf) ** (1.0 / n)
    if k >= n:
        return 1.0
    return float(beta.ppf(conf, k + 1, n - k))

MD = "ARXIS_Manuscript_v15_SPE.md"
SI = "ARXIS_Supporting_Information.md"
R = "retrain/results_v2/"
TXT = open(MD).read()
SITXT = open(SI).read()
FAIL, WARN, N = [], [], [0]


def check(label, got, want, tol=6e-4):
    N[0] += 1
    if got is None or want is None or abs(float(got) - float(want)) > tol:
        FAIL.append(f"[value] {label}: manuscript {got} vs results {want}")


def rule(label, ok, detail=""):
    N[0] += 1
    if not ok:
        FAIL.append(f"[style] {label}{(': ' + detail) if detail else ''}")


def table_after(prefix, doc=None):
    doc = TXT if doc is None else doc
    i = doc.index(prefix)
    block = doc[:i].rstrip().split("\n\n")[-1]
    rows = [r for r in block.strip().split("\n") if r.startswith("|")]
    return [[re.sub(r"[*_`]", "", c).strip() for c in r.strip("|").split("|")] for r in rows[2:]]


def num(s):
    s = s.replace("\\", "").replace("%", "").replace("≤", "").replace(",", "")
    s = s.replace("−", "-").replace("pp", "").replace("reference", "nan")
    m = re.search(r"-?\d+\.?\d*", s)
    return float(m.group()) if m else None


K = {"Cost-weighted policy": "A_cost_weighted", "Unweighted network": "B_unweighted",
     "Multilayer perceptron": "D_mlp", "Gradient boosting": "E_gbm",
     "Support-vector classifier": "SVM", "Random forest": "RF", "Recurrent (LSTM)": "LSTM",
     "Conservative Q-learning": "CQL", "k-nearest neighbors": "KNN",
     "Threshold rule": "ThresholdRule", "Shallow decision tree": "ThresholdRule",
     "Decision tree": "ThresholdRule", "Ordinal cost objective": "Ordinal",
     "Cost-weighted": "A_cost_weighted"}

# ============================ A. numbers ============================
lk = pd.read_csv(R + "v3_leakage.csv").pivot_table(index="model", columns="protocol",
                                                   values="decision_acc_mean")
cols = ["random", "blocked_noembargo", "blocked_embargo", "leave_one_block"]
for row in table_after("Table 7—Decision accuracy"):
    if row[0].lower() == "mean":
        for j, c in enumerate(cols):
            check(f"T8 mean {c}", num(row[j + 1]), lk[c].mean())
        continue
    for j, c in enumerate(cols):
        check(f"T8 {K[row[0]]} {c}", num(row[j + 1]), lk.loc[K[row[0]], c])

pw = pd.read_csv(R + "v3_powered.csv").set_index("model")
bay = json.load(open(R + "v3_powered_bayes.json"))["vs_cost_weighted"]
for row in table_after("Table 8—Model comparison"):
    k = K[row[0]]
    for j, col in enumerate(["decision_acc_mean", "decision_acc_std", "miss_rate_mean",
                             "escalation_mean", "fa_per_clean_mean", "expected_cost_mean"]):
        check(f"T9 {k} {col}", num(row[j + 1]), pw.loc[k, col])
    if k != "A_cost_weighted":
        check(f"T9 {k} P(eq)", num(row[7]), bay[k]["p_practically_equivalent"])

rho_rows = json.load(open(R + "v3_rho_sensitivity.json"))
for row in table_after("Table S4—Sensitivity of the Bayesian", SITXT):
    k = K[row[0]]
    _rho_dep = json.load(open(R + "v3_split_counts.json"))["rho"]
    for j, rho in enumerate([0.0, 0.05, 0.10, _rho_dep, 0.30, 0.40, 0.50]):
        want = next(x["p_equivalent"] for x in rho_rows
                    if x["model"] == k and abs(x["rho"] - rho) < 1e-9)
        check(f"T10 {k} rho={rho}", num(row[j + 1]), want)
# The manuscript claims nine of ten comparisons keep their most probable outcome
# across the rho sweep, and names the random forest as the single exception.
# Check exactly that, rather than assuming stability everywhere.
flip = []
for k in {x["model"] for x in rho_rows}:
    sub = [x for x in rho_rows if x["model"] == k]
    winners = {max(("o", x["p_other_better"]), ("e", x["p_equivalent"]),
                   ("r", x["p_ref_better"]), key=lambda t: t[1])[0] for x in sub}
    if len(winners) > 1:
        flip.append(k)
rule("two comparisons flip across the rho sweep", sorted(flip) == ["E_gbm", "RF"], str(sorted(flip)))
rule("both rho exceptions named where the sweep is reported",
     "Two comparisons change." in SITXT and "gradient boosting is equivalent up to" in SITXT)

# The ROPE sweep is the stronger sensitivity and the manuscript quotes its count.
rope = json.load(open(R + "v3_rope_sensitivity.json"))
n_flip = rope["_meta"]["n_flipping"]
rule("ROPE flip count as claimed", f"and {n_flip} of 10 do" in SITXT
     and f"seven of the ten changing outcome" in TXT, str(n_flip))
for k in ("B_unweighted", "ThresholdRule", "Ordinal"):
    picks = {rope[k][w]["bayes"]["most_probable"] for w in ("0.005", "0.010", "0.020")}
    rule(f"ROPE outcome stable for {k}", len(picks) == 1, str(picks))

lo = pd.read_csv(R + "v3_loco_all.csv")
# Plan item A1. The bound column is the worst-partition bound, so it has to be
# computed from the integer miss count each partition actually produced. The
# previous check reconstructed a pseudo-count from the averaged rate, which is a
# different and consistently smaller quantity: it put the worst Smoke bound at
# 7.49% where the worst partition supports 16.35%. Per-seed counts now come from
# v3_loco_perseed.csv and the manuscript's own counts column is checked against
# them as well, so neither can drift.
lps = pd.read_csv(R + "v3_loco_perseed.csv")
# Table 10 condenses six comparators that agree to four decimal places on every
# column into one grouped row. The grouping is checked against the result file
# rather than trusted: a hand-typed grouping is exactly how the Table 15 error
# got in, so every member is verified to carry the values the row shows, and the
# group is verified to be maximal for that held-out class.
_T10_SEEN = set()
for row in table_after("Table 10—Class-disjoint hazard"):
    members = [m.strip() for m in row[1].split(",")]
    for name in members:
        s = lo[(lo.held_out == row[0]) & (lo.model == K[name])].iloc[0]
        g = lps[(lps.held_out == row[0]) & (lps.model == K[name])]
        ks = sorted(int(x) for x in g.miss_k)
        n_part = int(g.n_danger.iloc[0])
        check(f"T12 {row[0]}/{name} miss", num(row[2]), s.miss_rate_mean)
        shown = row[3].strip()
        want = "0 in all five" if max(ks) == 0 else ", ".join(str(k) for k in ks)
        rule(f"T12 {row[0]}/{name} per-partition counts", shown == want, f"{shown!r} vs {want!r}")
        check(f"T12 {row[0]}/{name} worst-partition bound", num(row[4]),
              100 * cp_upper(max(ks), n_part), 6e-3)
        check(f"T12 {row[0]}/{name} esc", num(row[5]), s.escalation_mean)
        check(f"T12 {row[0]}/{name} under", num(row[6]), s.under_escalation_mean)
        _T10_SEEN.add((row[0], K[name]))
    if len(members) > 1:
        # maximality: no other comparator on this held-out class shares the row
        _same = [m for m in lo[lo.held_out == row[0]].model
                 if abs(float(lo[(lo.held_out == row[0]) & (lo.model == m)].miss_rate_mean.iloc[0])
                        - num(row[2])) < 5e-5
                 and abs(float(lo[(lo.held_out == row[0]) & (lo.model == m)].escalation_mean.iloc[0])
                         - num(row[5])) < 5e-4]
        rule(f"T10 grouped row on {row[0]} is maximal",
             sorted(_same) == sorted(K[m] for m in members),
             f"{sorted(_same)} vs {sorted(K[m] for m in members)}")
# every comparator still appears exactly once per held-out class
for _cls in ("Smoke", "Mixture"):
    _want = set((_cls, m) for m in lo[lo.held_out == _cls].model)
    rule(f"Table 10 lists every comparator on held-out {_cls}",
         _want <= _T10_SEEN, str(sorted(m for c, m in _want - _T10_SEEN)))
# Plan item A5: rho must come from the split counts and must agree everywhere.
_sc = json.load(open(R + "v3_split_counts.json"))
check("split counts: training windows", 4900, _sc["n_train"], 0)
check("split counts: test windows", 1264, _sc["n_test"], 0)
check("rho from the split counts", 0.2051, _sc["rho"], 6e-5)
check("rho stored in the Bayesian artifact", _sc["rho"],
      json.load(open(R + "v3_powered_bayes.json"))["rho"], 1e-12)
rule("rho is printed with its arithmetic and matches the counts",
     "ρ = *n*~test~/(*n*~train~ + *n*~test~) = 1,264/(4,900 + 1,264) = 0.2051" in TXT
     and "0.202." not in TXT.replace("1.423 ± 0.202.", ""), "printed value disagrees")
rule("the superseded rho is recorded in the artifact",
     abs(json.load(open(R + "v3_powered_bayes.json"))["rho_superseded"] - 0.20243433696348495) < 1e-12)

rule("the bound is never reconstructed from an averaged rate",
     "Bounds are computed from the counts, never from the averaged rate" in TXT)

cs = pd.read_csv(R + "v2_costsweep.csv").set_index("cost_ratio")
for row in table_after("Table S2—Cost-asymmetry sweep", SITXT):
    key = "label_function" if "Label" in row[0] else row[0].split()[0]
    s = cs.loc[key]
    for j, col in enumerate(["decision_acc_mean", "decision_acc_std",
                             "miss_rate_mean", "escalation_mean"]):
        check(f"T13 {key} {col}", num(row[j + 1]), s[col])

ab = pd.read_csv(R + "v2_ablation.csv").set_index("state")
STATE = {"Full state": "full_22", "Without per-sensor σ": "no_std_15",
         "Without anomaly score": "no_anomaly_21", "Without per-sensor δ": "no_delta_15",
         "Without δ and σ": "no_delta_std_8", "Current readings only": "current_only_7",
         "Anomaly score only": "anomaly_only_1"}
for row in table_after("Table S6—Decision-state ablation", SITXT):
    s = ab.loc[STATE[row[0]]]
    check(f"T14 {row[0]} d", num(row[1]), s.n_features)
    check(f"T14 {row[0]} acc", num(row[2]), s.decision_acc_mean)
    check(f"T14 {row[0]} sd", num(row[3]), s.decision_acc_std)
    if "reference" not in row[4]:
        check(f"T14 {row[0]} delta", num(row[4]), s.delta_pp_vs_full, 6e-3)

pt = pd.read_csv(R + "v2_perturb.csv")
COND = {"Clean": ("clean", 0.0), "Drift ±50%": ("drift", 0.5), "Noise σₙ = 0.5": ("noise", 0.5),
        "Dropout k = 1": ("dropout", 1.0), "Dropout k = 3": ("dropout", 3.0),
        "Dropout k = 7": ("dropout", 7.0)}
for row in table_after("Table 11—Selected perturbation"):
    cond = row[0].replace("*", "").strip()
    p_, lv = COND[cond]
    s = pt[(pt.perturbation == p_) & (pt.level == lv) & (pt.model == K[row[1]])].iloc[0]
    for j, col in enumerate(["decision_acc_mean", "miss_rate_mean",
                             "escalation_mean", "fa_per_clean_mean"]):
        check(f"T15 {cond}/{row[1]} {col}", num(row[j + 2]), s[col])
    if "≤" in row[6]:
        check(f"T15 {cond}/{row[1]} bound", num(row[6]),
              1000 * cp_upper(0, int(round(s.n_clean_pooled / 5))), 0.05)
    else:
        check(f"T15 {cond}/{row[1]} burden", num(row[6]), 1000 * s.fa_per_clean_mean, 0.6)

ca = pd.read_csv(R + "v2_calibration.csv").set_index("estimator")
EST = {"Deep ensemble (5)": "deep_ensemble_5", "MC dropout (20)": "mc_dropout_20",
       "Raw softmax": "raw_softmax", "Temperature scaling": "temp_scaling"}
for row in table_after("Table S3—Calibration", SITXT):
    s = ca.loc[EST[row[0]]]
    check(f"T16 {row[0]} ece", num(row[1]), s.ece_mean)
    check(f"T16 {row[0]} sd", num(row[2]), s.ece_std)
    check(f"T16 {row[0]} lo", num(row[3].split(" to ")[0]), s.ece_lo)
    check(f"T16 {row[0]} hi", num(row[3].split(" to ")[1]), s.ece_hi)
    check(f"T16 {row[0]} danger", num(row[4]), s.ece_danger_mean)
    check(f"T16 {row[0]} brier", num(row[5]), s.brier_mean)
    check(f"T16 {row[0]} nll", num(row[6]), s.nll_mean)
    check(f"T16 {row[0]} acc", num(row[7]), s.acc_mean)

an = json.load(open(R + "v2_anomaly.json"))
check("AUC held-out", 0.9251, an["auc_mean"], 6e-3); check("TPR held-out", 0.8015, an["tpr_mean"], 6e-3)
check("FPR held-out", 0.1101, an["fpr_mean"], 6e-3)
check("AUC in-sample", 0.962, an["in_sample"]["auc"])
check("FPR in-sample", 0.05, an["in_sample"]["fpr"])
for cls, v in [("NoGas", 0.023), ("Perfume", 0.107), ("Mixture", 80.55), ("Smoke", 164.65)]:
    check(f"recon {cls}", v, an["per_class_mean_recon_partition"][cls]["mean"], 6e-3)
# The anomaly numbers must appear in the prose in the form the result file holds
# them, not merely be consistent with it. A ROC-AUC of "0.9928 +- 0.0053 with no
# observed false positive" survived three revisions here: it is in no result file
# at all, and 0.9928 turns out to be a decision accuracy in v3_leakage.csv.
rule("the whole-corpus anomaly fit is quoted from the result file",
     f"ROC-AUC {an['in_sample']['auc']:.4f} at a false-positive rate of "
     f"{100 * an['in_sample']['fpr']:.1f}%" in TXT,
     f"{an['in_sample']['auc']:.4f} / {100 * an['in_sample']['fpr']:.1f}%")
rule("the unsourced 0.9928 anomaly claim is gone",
     "0.9928" not in TXT and "no observed false positive" not in TXT)
rule("the partition-local anomaly fit is quoted from the result file",
     f"ROC-AUC {an['auc_mean']:.4f} ± {an['auc_std']:.4f}" in TXT
     and f"TPR {an['tpr_mean']:.4f} ± {an['tpr_std']:.4f}" in TXT
     and f"FPR {an['fpr_mean']:.4f} ± {an['fpr_std']:.4f}" in TXT)
_gap = an["in_sample"]["auc"] - an["auc_mean"]
rule("the discrimination the leak was worth is described, not overstated",
     "Just under four points of discrimination" in TXT and 0.030 < _gap < 0.040,
     f"{_gap:.4f}")
rule("the false-positive direction is stated correctly",
     an["in_sample"]["fpr"] < an["fpr_mean"]
     and "the false-positive rate more than doubles" in TXT)
# Fig. 7a must plot the same fit the prose quotes.
_pc = an["per_class_mean_recon_partition"]
rule("the per-class reconstruction figures in the prose are the partition-local fit",
     all(f"{_pc[c]['mean']:.3f}" in TXT or f"{_pc[c]['mean']:.1f}" in TXT
         for c in ("NoGas", "Perfume", "Mixture", "Smoke")),
     str({c: round(_pc[c]["mean"], 3) for c in _pc}))
rule("Fig. 7a is drawn from the partition-local fit",
     "per_class_mean_recon_partition" in open(
         os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "retrain", "make_figs_v2.py"), encoding="utf-8").read())

# One name and one membership for the evaluation set. The Summary used to call
# it a five-quantity *safety* metric set while the body says decision accuracy is
# not a safety metric, and Fig. 1 listed a different five from Table 5.
rule("the evaluation set has one name", "five-quantity evaluation set" in TXT
     and "five-quantity safety metric set" not in TXT)
_t5 = [r[0] for r in table_after("Table 5—The proposed evaluation set")]
rule("Table 5 carries five quantities", len(_t5) == 5, str(_t5))
_fig1 = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "retrain", "spe_flow.py"), encoding="utf-8").read()
_cells = re.search(r"METRIC_CELLS = \[(.*?)\]", _fig1, re.S).group(1)
_cells = [s.strip().strip('"') for s in _cells.replace("\n", " ").split(",") if s.strip()]
def _norm(s):
    return re.sub(r"[^a-z]", "", s.lower().replace("rate", ""))
rule("Fig. 1 names the same five quantities as Table 5",
     sorted(_norm(c) for c in _cells) == sorted(_norm(r) for r in _t5),
     f"{_cells} vs {_t5}")
rule("alarm burden is declared a unit of the false-alarm rate, not a sixth quantity",
     "not a sixth quantity" in _fig1 and "not a sixth quantity" in TXT)

ai = pd.read_csv(R + "v2_anomaly_influence.csv")
fr = ai.groupby("model").frac_windows_action_changed.first()
for k, v in [("E_gbm", 0.6422), ("RF", 0.2359), ("A_cost_weighted", 0.0665), ("ThresholdRule", 0.0)]:
    check(f"anomaly influence {k}", v, fr[k])
g0 = ai[(ai.model == "E_gbm") & (ai.level == 0.0)].iloc[0]
check("GBM miss at anomaly=0", 0.4203, g0.miss_rate_mean)
check("GBM esc at anomaly=0", 0.497, g0.escalation_mean)
cb = pd.read_csv(R + "v3_cost_blocked_validation.csv").set_index("cost_ratio")
check("blocked-val 2:1 val", 0.9549, cb.loc["2:1", "val_acc_mean"])
check("blocked-val 2:1 test", 0.9384, cb.loc["2:1", "test_acc_mean"])
check("blocked-val 6:1 val", 0.9584, cb.loc["6:1", "val_acc_mean"])
cv = pd.read_csv(R + "v3_cost_validation.csv").set_index("cost_ratio")
low = cv.loc[["1:1", "2:1", "4:1", "6:1", "8:1"], "val_acc_mean"]
check("random-val floor", 0.9948, low.min(), 6e-3); check("random-val ceiling", 0.9980, low.max())
ps = pd.read_csv(R + "v2_calibration_perseed.csv")
T = ps[ps.estimator == "temp_scaling"]["T"]
check("temperature mean", 1.423, T.mean(), 6e-3); check("temperature sd", 0.202, T.std(ddof=1), 6e-3)
rule("the network is reported over-confident under the blocked calibration split",
     "the network is over-confident" in TXT and T.min() > 1.0, f"min T = {T.min():.2f}")
mx = lo[lo.held_out == "Mixture"].set_index("model")
check("escalation ratio", 203, mx.escalation_mean.max() / mx.escalation_mean.min(), 1.0)
check("30-run hazard bound %", 0.016, 100 * (1 - 0.05 ** (1 / 18960)))
check("30-run clean bound %", 0.032, 100 * (1 - 0.05 ** (1 / 9480)))
check("LOCO bound %", 0.038, 100 * (1 - 0.05 ** (1 / 7905)))
pw10 = pw.drop(index="Ordinal")
check("ten-comparator spread", 4.43, 100 * (pw10.decision_acc_mean.max()
                                            - pw10.decision_acc_mean.min()), 6e-3)
gap = (lk["random"] - lk["blocked_embargo"]) * 100
check("leakage gap min", 3.23, gap.drop("ThresholdRule").min(), 6e-3)
check("leakage gap max", 5.57, gap.drop("ThresholdRule").max(), 6e-3)
check("leakage gap mean learned", 4.04, gap.drop("ThresholdRule").mean(), 6e-3)
check("leakage gap mean all", 3.24, gap.mean(), 6e-3)
check("threshold rule loses", -0.74, gap["ThresholdRule"], 6e-3)
check("alarms/hour at k=1", 231, 1000 * pt[(pt.perturbation == "dropout") & (pt.level == 1.0)
      & (pt.model == "A_cost_weighted")].fa_per_clean_mean.iloc[0] * 1.8, 1.0)
check("clean burden bound", 1.9, 1000 * (1 - 0.05 ** (1 / 1580)), 0.05)
check("GBM anomaly factor", 43, g0.miss_rate_mean / 0.0092, 3.0)

# ====================== B. SPE house style ==========================
# Captions are no longer set in bold, so they are recognised by their SPE form:
# a line that opens with "Fig. N—" or "Table N—". A cross-reference in running
# prose never carries the em-dash, and never starts a line.
figs = [int(m) for m in re.findall(r"^Fig\. (\d+)—", TXT, re.M)]
tabs = [int(m) for m in re.findall(r"^Table (\d+)—", TXT, re.M)]
# Eq. 2 is split into 2a (the single-valued training target) and 2b (the
# acceptable set) after plan item A6, so the sequence check works on the number
# and the lettered pair counts once.
eq_labels = re.findall(r"\.{4,}\s*\((\d+[a-z]?)\)", TXT)
eqs = []
for lbl in eq_labels:
    n = int(re.match(r"\d+", lbl).group())
    if not eqs or eqs[-1] != n:
        eqs.append(n)
rule("figure captions numbered sequentially", figs == list(range(1, len(figs) + 1)), str(figs))
rule("table captions numbered sequentially", tabs == list(range(1, len(tabs) + 1)), str(tabs))
rule("equations numbered sequentially", eqs == list(range(1, len(eqs) + 1)), str(eqs))
rule("caption count matches image count", len(figs) == len(re.findall(r"!\[\]\(", TXT)))
_sfigs = re.findall(r"^Fig\. S(\d+)—", SITXT, re.M)
rule("at least as many figures as the model paper", len(figs) + len(_sfigs) >= 19,
     f"{len(figs)} body + {len(_sfigs)} supplementary")
rule("at least as many tables as the model paper", len(tabs) >= 4, f"{len(tabs)}")
rule("at least as many equations as the model paper",
     len(eq_labels) + SITXT.count("(S1)") >= 13, f"{len(eq_labels)} numbered displays")

for n in figs:
    rule(f"Fig. {n} cited in text", len(re.findall(rf"Fig(?:s)?\. \*{{0,2}}{n}(?![0-9])", TXT)) >= 2)
for n in tabs:
    rule(f"Table {n} cited in text", len(re.findall(rf"\*{{0,2}}Table(?:s)? {n}\b", TXT)) >= 2)
for n in eqs:
    rule(f"Eq. {n} cited in text", f"Eq. {n}" in TXT or f"Eqs. {n}" in TXT
         or re.search(rf"Eq\. \d+ (?:and|to) {n}\b", TXT) is not None)

# each figure caption must FOLLOW its image, as SPE sets them
for m in re.finditer(r"^\*\*Fig\. (\d+)—", TXT, re.M):
    before = TXT[:m.start()].rstrip().split("\n")[-1]
    rule(f"Fig. {m.group(1)} caption sits under its image", before.strip().startswith("!["))

HEADS = ["## Summary", "## Introduction", "## Related Work and Positioning", "## Methodology",
         "## Results", "## Engineering Implications",
         "## Limitations and Threats to Validity", "## Conclusions",
         "## Acknowledgments", "## References"]
for h in HEADS:
    rule(f"section present: {h[3:]}", h in TXT)
rule("no numbered section headings (SPE style)",
     not re.search(r"^## \d", TXT, re.M))
rule("references alphabetical", True)
_refs_end = TXT.find("\n## ", TXT.index("## References") + 1)
refs = TXT[TXT.index("## References"):_refs_end if _refs_end != -1 else len(TXT)]
names = re.findall(r"^([A-Z][^,\n]*),", refs, re.M)
sorted_names = sorted(names, key=lambda x: x.lower())
if names != sorted_names:
    off = [n for n, s_ in zip(names, sorted_names) if n != s_][:3]
    FAIL.append(f"[style] references not alphabetical near {off}")
N[0] += 1

# ---- cross-reference proof pass -------------------------------------
# The scripted edits that produced this draft broke several "Eq. n" and
# "Table n" pointers without breaking any number, so every cross-reference is
# now proved against what actually sits at the target.
eq_body = {}
for m in re.finditer(r"&nbsp;.*?\.{4,}\s*\((\d+)[a-z]?\)", TXT):
    n = int(m.group(1))
    eq_body[n] = eq_body.get(n, "") + m.group(0)   # 2a and 2b share the slot
# Eq. 12 of v10, the ordinal objective, moved to the supporting information as
# Eq. S1 when that experiment did; the duty-cycle conversion renumbers to 12.
EQ_SYMBOL = {1: "φₜ", 2: "NoGas", 3: "log", 4: "Acc", 5: "~miss~", 6: "~esc~",
             7: "~under~", 8: "~fa~", 9: "1000", 10: "BetaInv", 11: "~post~",
             12: "*f* / 1000"}
for n, sym in EQ_SYMBOL.items():
    rule(f"Eq. {n} displays the quantity the text names",
         n in eq_body and sym in eq_body[n], eq_body.get(n, "missing")[:70])

CAPTION = {int(m.group(1)): m.group(2).lower()
           for m in re.finditer(r"^\*\*Table (\d+)—([^*]{0,70})", TXT, re.M)}
STOP = {"the", "and", "of", "for", "with", "over", "under", "each", "its",
        "which", "from", "into", "that", "this", "five", "four", "seeds"}
def stem(w):
    return w[:-1] if w.endswith("s") and len(w) > 4 else w
for m in re.finditer(r"\*\*Table (\d+)\*\*", TXT):
    n = int(m.group(1))
    ctx = TXT[max(0, m.start() - 400): m.start() + 400].lower()
    words = {stem(w) for w in re.findall(r"[a-z]{4,}", CAPTION.get(n, ""))} - STOP
    hit = any(stem(w) in ctx for w in words)
    rule(f"reference to Table {n} lands on a topically related table", (not words) or hit,
         f"caption {CAPTION.get(n, '?')!r}")

rule("no leftover edit tokens", not re.search(r"@[A-Z0-9_]{1,12}@", TXT))
body_only = TXT[:TXT.index("## References")]
rule("no sentence ends then continues lowercase",
     not re.search(r"[a-z]\.\s+(?:and|but|or|which|so|because)\s", body_only))
rule("no doubled horizontal rule", "---\n\n---" not in TXT)

# every reference must be cited, and every citation must have a reference
ref_block = TXT[TXT.index("## References"):]
entries = [l for l in ref_block.split("\n") if len(l) > 60 and re.match(r"^[A-Z]", l)]
for line in entries:
    m = re.match(r"^(.*?)\.?\s+(?:19|20)\d\d[a-z]?\.", line)
    if not m:
        continue
    authors = m.group(1)
    first = re.split(r",", authors)[0].strip()
    surname = first.split()[-1] if " " in first else first
    N[0] += 1
    if surname not in body_only:
        FAIL.append(f"[style] reference never cited in the body: {surname}")

# ---- coverage against the pre-restructure manuscript ----------------
# The SPE rewrite reorganised the whole document, so this proves nothing was
# silently dropped on the way: every figure v9 carried, every table subject it
# carried, and every distinctive claim, must still be here.
# The v9 manuscript itself is not in the repository: it quotes values this paper
# has since corrected, and a reader who found it would be reading superseded
# numbers. Its continuity list lives in continuity_v9.json instead, so the check
# survives the manuscript being retired. A missing list is a failure, not a
# silent skip: seventy checks disappearing quietly is how a lost figure escapes.
_V9J = os.path.join(os.path.dirname(os.path.abspath(__file__)), "continuity_v9.json")
rule("the v9 continuity list is present", os.path.exists(_V9J),
     "continuity_v9.json is missing; the carry-over checks cannot run")
if os.path.exists(_V9J):
    import json as _cj
    V9_FIGURES = _cj.load(open(_V9J, encoding="utf-8"))["figures"]
else:
    V9_FIGURES = []
# One v9 figure is deliberately retired rather than carried: the pipeline
# flowchart fig_architecture, replaced by fig_setup after the supervisor asked
# for fewer box-and-arrow diagrams and a picture of the measurement instead.
# The replacement has to be present for the exemption to hold, so that a
# genuine accidental deletion still fails here.
# The action ladder is drawn in two figures and tabulated in Table 4. A figure
# that labels action 4 "Emergency shutdown" contradicts the paper's own statement
# that it is deliberately not called that, so the drawn labels are gated against
# the table rather than trusted.
_LADDER = [r[1] for r in table_after("Table 4—Safety-action space")]
# Table 9 had no gate at all, which is how its seven rows came to disagree with
# v2_action_matrix.csv in every cell while Fig. 11, drawn from the same file, was
# right. The acceptable set is A(NoGas)={0}, A(Smoke)={3}, A(Mixture)={4} and
# A(Perfume)={1,2}, so the share is a sum over those actions and nothing else.
_AM = pd.read_csv(R + "v2_action_matrix.csv")
_ACCEPT = {"NoGas": {0}, "Smoke": {3}, "Mixture": {4}, "Perfume": {1, 2}}
_T9 = table_after("Table 9—Per-class share")
_KEY9 = {v: k for k, v in {
    "E_gbm": "Gradient boosting", "RF": "Random forest",
    "SVM": "Support-vector classifier", "D_mlp": "Multilayer perceptron",
    "KNN": "k-nearest neighbors", "A_cost_weighted": "Cost-weighted policy",
    "ThresholdRule": "Shallow decision tree"}.items()}
for _r in _T9:
    if _r[0] == "Range":
        continue
    _m = _KEY9.get(_r[0])
    rule(f"Table 9 names a model in the result file: {_r[0]}", _m is not None)
    if _m is None:
        continue
    for _c, _g in enumerate(["NoGas", "Smoke", "Mixture", "Perfume"], start=1):
        _sub = _AM[(_AM.model == _m) & (_AM.gas == _g)]
        _want = float(_sub[_sub.action.isin(_ACCEPT[_g])].share.sum())
        check(f"Table 9 {_r[0]} / {_g}", num(_r[_c]), _want, 5e-4)
rule("Table 9 lists every model in the action matrix",
     len([r for r in _T9 if r[0] != "Range"]) == _AM.model.nunique(),
     f"{len(_T9) - 1} rows against {_AM.model.nunique()} models")

rule("Table 4 lists the five actions in order",
     _LADDER == ["Monitor", "Increase sampling", "Request verification",
                 "Raise alarm", "Recommend ESD assessment"], str(_LADDER))
# make_fig_setup.py was retired with the figure it drew. Reading whatever figure
# modules exist, and failing when none do, keeps these two from being skipped
# quietly because a filename changed.
_figsrc = ""
for _fm in ("retrain/make_figs_v2.py", "retrain/make_figs_arch.py"):
    if os.path.exists(_fm):
        _figsrc += open(_fm, encoding="utf-8").read()
rule("the figure modules are present", bool(_figsrc), "no figure module found")
rule("no figure still labels action 4 an emergency shutdown",
     "Emergency shutdown" not in _figsrc and "Emergency Shutdown" not in _figsrc)
rule("the ladder figure uses Table 4's label for action 4",
     '(4, "Recommend ESD assessment"' in _figsrc)

# Both the v9 pipeline flowchart and the acquisition schematic that briefly
# replaced it are retired: the editorial review reframed the paper as a
# benchmark study, where neither earns a place. A retired figure must be
# absent from both documents, so an accidental deletion still fails here.
RETIRED = {"fig_architecture": None, "fig_setup": None, "fig_edge": None}
if V9_FIGURES:
    for f in V9_FIGURES:
        if f in RETIRED:
            rule(f"retired v9 figure {f} is absent", f not in TXT and f not in SITXT)
            continue
        rule(f"figure carried over from v9: {f}", f in TXT or f in SITXT)
    for subj in ["prior work", "MOX array", "Safety-action", "Perception component",
                 "partitioning protocols", "Model comparison", "Posterior probability",
                 "Leave-one-class-out", "Cost-asymmetry", "Decision-state ablation",
                 "perturbation results", "Calibration"]:
        rule(f"table carried over from v9: {subj}", subj.lower() in (TXT + SITXT).lower())
    CLAIMS = ["275 alarms", "six alarms per hour", "2 s interval", "6,324 windows",
              "4,900 training", "label-function", "0.4535", "does not command a trip",
              "protection layer", "dueling value and advantage", "Gemma 3 1B",
              "150 tokens", "hidden dimension 32", "95th percentile", "0.0625",
              "region of practical equivalence", "42, 1337, 7, 2024, 99", "1000 to 1029",
              "determinism guards", "2,995", "not monotone in target severity", "0.962",
              "never selected", "predictive uncertainty", "*C* = 6:1", "silence costs 10",
              "elevated sampling", "Graceful, then brittle",
              "231 alarm-grade indications per hour", "17 per hour", "60% accuracy", "under-confident",
              "5 to 30 bins", "conformal prediction", "Cross-session and cross-device",
              "controlled hydrocarbon release", "91.7%", "optimistic upper bound",
              "no visual term", "0.19%"]
    for c in CLAIMS:
        rule(f"claim carried over from v9: {c!r}", c in TXT or c in SITXT)

# ======================= C. hygiene =================================
rule("no unresolved citation markers", "CITATION NEEDED" not in TXT)
# SPE's submission form takes one to five keywords; the printed line and the
# form entry have to agree, so the count is checked here rather than trusted.
kw = re.search(r"^Keywords:(.+)$", TXT, re.M)
rule("keywords line present", kw is not None)
if kw:
    n_kw = len([k for k in kw.group(1).split(";") if k.strip()])
    rule("keyword count within SPE's one to five", 1 <= n_kw <= 5, f"{n_kw} listed")
for phrase in ("Items Still Requiring Author Action", "Author Verification Items",
               "not intended for publication", "image corpus was unavailable",
               "image corpus was not available", "Liang et al. (2024)"):
    rule(f"no stale text: {phrase!r}", phrase not in TXT)

# SPE sets captions as "Fig. 1—..." and appendix headings as "Appendix A—...",
# so the em-dash is house style in exactly those places and banned everywhere else.
prose = "\n".join(l for l in TXT.split("\n")
                  if not re.match(r"^(?:Table |Fig\. )(?:\d+|[A-Z]-\d+)—", l.strip())
                  and not re.match(r"^## Appendix [A-Z]—", l.strip()))
rule("no em-dash in prose (captions and appendix headings excepted)", "—" not in prose,
     f"{prose.count('—')} found")

def prose_words(t):
    keep = [l for l in t.split("\n") if not l.strip().startswith("|")
            and not re.match(r"^(?:Table |Fig\. )(?:\d+|[A-Z]-\d+)—", l.strip())
            and not l.strip().startswith("![")]
    return len(" ".join(keep).split())

ALLH = HEADS + ["## Limitations and Threats to Validity", "## Nomenclature"]
bounds = sorted(TXT.index(h) for h in ALLH if h in TXT)
pos = sorted((TXT.index(h), h) for h in HEADS[:7] if h in TXT)
# Absolute word bands for the sections whose job is fixed, and for the results
# section a DENSITY test instead: the model paper spends 3,462 words on two
# experiments, or 1,731 each. A results section carrying seven experiments is
# not required to fit the model's absolute length, but it should not be more
# verbose per experiment than the paper it is modelled on.
# Summary and Conclusion raised once, with the experiment count, from seven
# experiments to nine. Experiment 8 answers the reject-option question every
# review asked for and Experiment 9 the raw-space fault modes the pre-submission
# plan called the preferred path; each contributes one finding that a reader
# should not have to reach the Results section to learn. Both sections were
# compressed before the band was moved, and the model paper's ratio of summary
# to experiment count is still respected at nine.
# Bands recalibrated for the structure the editorial review asked for: all
# experiments now precede the implications section, the two eventization
# analyses moved from Model Application into Results, and the Summary was cut
# from a mini-Discussion to an abstract. The Summary ceiling is the reviewer's
# stated target rather than an internal preference. Every other band came down,
# because the same review asked for a 25 to 30 percent reduction and a band
# that does not bind is not a check.
BAND = {"## Summary": (300, 400), "## Introduction": (800, 1250),
        "## Related Work and Positioning": (700, 1400),
        "## Methodology": (1800, 3600),
        "## Engineering Implications": (700, 1800),
        "## Limitations and Threats to Validity": (500, 1200),
        "## Conclusions": (300, 560)}
N_EXPERIMENTS = 9   # six answering a research question, three supporting
MODEL_RESULTS_DENSITY = 3462 / 2
for i, (p_, h) in enumerate(pos):
    end = min([b for b in bounds if b > p_] + [len(TXT)])
    w = prose_words(TXT[p_:end])
    if h == "## Results":
        dens = w / N_EXPERIMENTS
        rule("results density vs the model paper", dens <= MODEL_RESULTS_DENSITY,
             f"{dens:.0f} words/experiment vs the model's {MODEL_RESULTS_DENSITY:.0f}")
        # Raised with the experiment count, from seven to nine: Experiment 8
        # adds the reject-option baseline every review asked for and Experiment 9
        # the raw-space fault modes the pre-submission plan called the preferred
        # path. The density check below is the one that does the real work, and
        # it tightened rather than loosened when the two were added.
        rule("results section not bloated in absolute terms", w <= 7400, f"{w}")
        continue
    lo_, hi_ = BAND[h]
    rule(f"word budget {h[3:]}", lo_ <= w <= hi_, f"{w} outside [{lo_}, {hi_}]")


# ============ E. checks added after the second referee pass ============
# Every one of these exists because a specific number drifted or a specific
# claim outran its evidence. They fail loudly rather than silently.

import json as _json

# --- the 117/128 contradiction: the burden quoted in prose must equal the
#     perturbation result, and the hourly figure must equal 1.8x it
_pt = pd.read_csv(R + "v2_perturb.csv")
_k1 = _pt[(_pt.perturbation == "dropout") & (_pt.level == 1.0) &
          (_pt.model == "A_cost_weighted")].fa_per_clean_mean.iloc[0]
rule("alarm burden at one lost channel quoted consistently",
     f"from 0 to between {1000*_k1:.0f} and 200 per 1,000 windows" in TXT, f"{1000*_k1:.1f}")
# The random forest is not untouched at one lost channel; it moves to 7 per 1,000.
# The earlier text said "untouched" in two places, which the screen table and the
# conclusions both repeated.
_k1rf = _pt[(_pt.perturbation == "dropout") & (_pt.level == 1.0) &
            (_pt.model == "RF")].fa_per_clean_mean.iloc[0]
rule("the random forest is reported as moving, not as untouched, at one lost channel",
     f"the random forest moves only to {1000*_k1rf:.1f}" in TXT
     and f"the random forest only to {1000*_k1rf:.0f}." in TXT
     and "left the random forest untouched" not in TXT, f"{1000*_k1rf:.2f}")
# Total sensor loss: gradient boosting stops alarming and misses 19.5%, not 60%.
_k7gbm = _pt[(_pt.perturbation == "dropout") & (_pt.level == 7.0) &
             (_pt.model == "E_gbm")].miss_rate_mean.iloc[0]
rule("the silent-failure miss rate at total sensor loss is quoted from the result file",
     f"misses {100*_k7gbm:.1f}% of hazards" in TXT
     and f"missing {100*_k7gbm:.1f}% of hazards" in TXT
     and "misses 60% of hazards" not in TXT, f"{100*_k7gbm:.2f}")
# Section names that no longer exist must not survive as cross-references.
rule("no cross-reference to the retired Proposed Method heading",
     "Proposed Method" not in TXT and "Proposed Method" not in SITXT)
rule("hourly figure equals 1.8x the per-1000 burden",
     f"about {1800*_k1:.0f} alarm-grade indications per hour" in TXT, f"{1800*_k1:.1f}")
rule("the superseded 117 figure is gone", "117 alarm-grade" not in TXT and
     "to 117 and 200" not in TXT)

# --- the anomaly-probe contradiction between prose and figure caption
_ai = pd.read_csv(R + "v2_anomaly_influence.csv")
_g0 = _ai[(_ai.model == "E_gbm") & (_ai.level == 0.0)].iloc[0]
_gb = _ai[(_ai.model == "E_gbm") & (_ai.level == 1.0)].iloc[0]
rule("anomaly probe quoted identically in the manuscript and the supplement",
     (TXT + SITXT).count(f"{_gb.miss_rate_mean:.4f} to {_g0.miss_rate_mean:.4f}") >= 2,
     f"{_gb.miss_rate_mean:.4f} to {_g0.miss_rate_mean:.4f}")
rule("the superseded 0.3405 figure is gone", "0.3405" not in TXT)

# --- the 2.48 / 2.42 spread
_lk2 = pd.read_csv(R + "v3_leakage.csv").pivot_table(
    index="model", columns="protocol", values="decision_acc_mean")
_temporal = [m for m in _lk2.index if m != "ThresholdRule"]
_spread = 100 * (_lk2.loc[_temporal, "blocked_embargo"].max() -
                 _lk2.loc[_temporal, "blocked_embargo"].min())
rule("temporal model spread quoted consistently",
     f"{_spread:.2f}-point spread" in TXT, f"{_spread:.2f}")

# --- headline ratios must be reconstructible from a displayed value
_lo3 = pd.read_csv(R + "v3_loco_all.csv")
_mix = _lo3[_lo3.held_out == "Mixture"]
_ratio = _mix.escalation_mean.max() / _mix.escalation_mean.min()
rule("escalation ratio stated as about 200, not a false precision",
     "a factor of about 200" in TXT and "factor of 203" not in TXT, f"{_ratio:.1f}")
rule("the escalation minimum is displayed to the precision the ratio needs",
     f"{_mix.escalation_mean.min():.4f}" in TXT, f"{_mix.escalation_mean.min():.4f}")
_bd2 = _json.load(open(R + "v3_partition_bounds.json"))
_sk = {r["model"]: r["partition_pct"] for r in _bd2["loco"] if r["held_out"] == "Smoke"}
_bratio = max(_sk.values()) / min(_sk.values())

# --- the effective-sample-size correction, the reviewer's Tier 1 statistical item
_bb = _json.load(open(R + "v3_block_bootstrap.json"))
_neff_h = _bb["A_cost_weighted"]["miss"]["L20"]["effective_n"]
_neff_c = _bb["A_cost_weighted"]["fa"]["L20"]["effective_n"]
rule("disjoint-support counts quoted", f"{_neff_h} of the 632 hazardous" in TXT
     and f"{_neff_c} of the 316 clean windows" in TXT, f"{_neff_h}/{_neff_c}")
check("disjoint-support bound, hazardous", 9.21, 100 * (1 - 0.05 ** (1.0 / _neff_h)), 6e-3)
check("disjoint-support bound, clean", 18.10, 100 * (1 - 0.05 ** (1.0 / _neff_c)), 6e-2)
# The emphasis markers were removed from prose on the supervisor's instruction
# that a journal does not embolden text mid-sentence, so the check is on the
# figures rather than on their formatting.
rule("the widened bounds appear in the text",
     "from 0.47% to 9.21%" in TXT and "from 0.94% to 18.10%" in TXT)
rule("bold survives only on caption leads and run-in headings",
     not re.search(r"(?<!^)\*\*(?:Table|Fig\.|Eq\.) ?\d", TXT, re.M), "a bolded cross-reference remains")

# --- episode-level hazard metrics
_em = pd.read_csv(R + "v3_episode_metrics.csv")
for _m, _lv, _want in [("D_mlp", 7.0, 5), ("E_gbm", 7.0, 7), ("E_gbm", 1.0, 9)]:
    _sub = _em[(_em.model == _m) & (_em.perturbation == "dropout") & (_em.level == _lv)]
    check(f"episodes detected {_m}@k={_lv:.0f}", _want, _sub.episodes_detected.sum(), 0.5)
rule("in-distribution episode detection is reported as saturated",
     "every model detects every hazard episode, and every detection latency is zero" in TXT)
rule("no latency claim is made", "No claim about detection latency can be made from this corpus" in TXT)

# --- framing requirements from the second review
rule("action labels declared constructed targets before Eq. 2",
     "constructed evaluation targets" in TXT and
     TXT.index("constructed evaluation targets") < TXT.index("*y*(NoGas) = 0"))
# Plan item A6: the training target and the acceptable set are different objects
# and the loss is only defined against the single-valued one. The previous text
# used one symbol for both, which left Eq. 3 taking the log-probability of a set.
rule("training target and acceptable set are separate objects",
     "*y*(NoGas) = 0" in TXT and "*A*(NoGas) = {0}" in TXT
     and "............ (2a)" in TXT and "............ (2b)" in TXT)
rule("the loss is written against the single-valued target",
     "log *p*~θ~( *y*(*g*ₜ) | φₜ )" in TXT and "*p*~θ~( *a*\\*(" not in TXT)
rule("decision accuracy scores membership in the acceptable set",
     "1[ *a*ₜ ∈ *A*(*g*ₜ) ]" in TXT)
rule("the old conflated symbol is gone everywhere",
     "a*\\*(" not in TXT and "a*\\*(" not in SITXT)
rule("action 2 is declared structurally untrained",
     "action 2 is never a positive training target" in TXT)
rule("RQ4 asks only what the experiment answers",
     "no hardware was faulted and no physical failure induced" in TXT)
rule("the miss-only comparison is not stated as a ranking",
     "would rank the support-vector classifier above gradient boosting" not in TXT
     and "would clear both and separate neither" in TXT)
# The contributions were a bullet list and are running prose now, as the model
# papers set them. The six research questions stay a list, numbered i to vi, each
# item carrying the RQ label the rest of the paper cross-references.
_RQ_BLOCK = TXT[TXT.index("Six questions follow."):
                TXT.index("## Related Work and Positioning")]
_RQ_ITEMS = re.findall(r"^(i{1,3}|iv|v|vi)\.\s+RQ(\d)\.", _RQ_BLOCK, re.M)
rule("the research questions are counted correctly",
     [n for _, n in _RQ_ITEMS] == ["0", "1", "2", "3", "4", "5"]
     and [r for r, _ in _RQ_ITEMS] == ["i", "ii", "iii", "iv", "v", "vi"],
     str(_RQ_ITEMS))
rule("the contributions are prose, not a list",
     not re.search(r"^\s*[-*+]\s",
                   TXT[TXT.index("The first contribution is"):
                       TXT.index("## Related Work and Positioning")], re.M))
rule("the experiment count is stated correctly in both places",
     "reports nine experiments" in TXT and "Each of the nine experiments" in TXT)
# The stale figures the 14 September review caught, each now pinned to its source.
_ptx = pd.read_csv(R + "v2_perturb.csv")
_pwx = pd.read_csv(R + "v3_powered.csv").set_index("model")
_lox = pd.read_csv(R + "v3_loco_all.csv")
def _pv(m, k, l, col):
    return float(_ptx[(_ptx.model == m) & (_ptx.perturbation == k) & (_ptx.level == l)][col].iloc[0])
check("GBM miss under total sensor loss, quoted in the Discussion", 0.195,
      _pv("E_gbm", "dropout", 7.0, "miss_rate_mean"), 6e-4)
rule("the Discussion quotes the current GBM figure",
     "missing 19.5% of hazards" in TXT and "missing 60% of hazards" not in TXT)
check("cost-weighted accuracy at total sensor loss", 0.350,
      _pv("A_cost_weighted", "dropout", 7.0, "decision_acc_mean"), 6e-4)
rule("the failure-mode sentence quotes the current accuracy",
     "at 35.0% accuracy describes a failure mode" in TXT)
_gap = 100 * (_pwx.loc["A_cost_weighted", "decision_acc_mean"] - _pwx.loc["ThresholdRule", "decision_acc_mean"])
check("shallow-tree accuracy gap", 3.90, _gap, 6e-3)
rule("the tree gap is quoted correctly", "3.90 accuracy points lower" in TXT)
_mlp = float(_lox[(_lox.held_out == "Smoke") & (_lox.model == "D_mlp")].miss_rate_mean.iloc[0])
check("MLP miss on held-out Smoke, quoted in the Discussion", 0.0641, _mlp, 6e-4)
check("GBM miss under 50% drift", 0.0111, _pv("E_gbm", "drift", 0.5, "miss_rate_mean"), 6e-4)
rule("both Discussion comparisons quote current figures",
     "the perceptron misses 6.41%" in TXT and "gradient boosting misses 1.11%" in TXT)
_ord = _lox[(_lox.held_out == "Mixture")].set_index("model")
rule("the ordinal-objective shift is quoted from Table 10",
     f"rises from {_ord.loc['A_cost_weighted','escalation_mean']:.3f} ± {_ord.loc['A_cost_weighted','escalation_std']:.3f} "
     f"to {_ord.loc['Ordinal','escalation_mean']:.3f} ± {_ord.loc['Ordinal','escalation_std']:.3f}" in TXT,
     f"{_ord.loc['A_cost_weighted','escalation_mean']:.3f}->{_ord.loc['Ordinal','escalation_mean']:.3f}")
rule("class exclusion is not called demonstrated domain shift",
     "not demonstrated domain shift" in TXT)
rule("no functional-safety terminology for the failure taxonomy",
     "fail-loud" not in TXT.lower() and "fail-silent" not in TXT.lower()
     and "fails safe" not in TXT.lower())
rule("alarm hierarchy announced where the metric is defined",
     "Three levels, announced in advance" in TXT)
rule("EEMUA comparison marked contextual",
     "the comparison is contextual and not a compliance test" in TXT)
# The indication-density figure carries no EEMUA reference line: that figure is a
# budget for a whole operator position and the axis is one detector's density, so
# a line across the panel invited a comparison the two quantities do not support.
rule("the supporting information carries the manuscript title",
     TXT.split("\n")[0].lstrip("# ").strip() in SITXT)
rule("the indication-density figure carries no reference line",
     "Fig. 15 carries no reference line" in TXT)
# A repository released with a paper and no licence is not usable by a reader,
# whatever the availability statement says.
rule("the repository carries a licence file",
     os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "LICENSE")), "LICENSE is missing")
rule("expected cost marked study-defined", "Study-defined expected cost" in TXT)
rule("no deployable-policy claim for the decision tree",
     "deployable decision tree" not in TXT)
rule("the rho sweep is described consistently wherever it appears",
     "Two comparisons change" in SITXT
     and "direction of every comparison is stable" not in (TXT + SITXT)
     and "eight of the ten comparisons keep their most probable outcome" in TXT.lower())

# ============ F. submission gates from the third referee pass ============
# Each of these blocks submission until it is resolved. They are gates, not
# warnings, because the manuscript claims every quoted value is machine-checked
# and a soft warning is how the earlier contradictions survived.

# Blocker 1: the anomaly-probe ratio must be reconstructible from the two
# displayed values, to the precision the text claims.
_ratio = _g0.miss_rate_mean / _gb.miss_rate_mean
_disp = round(_g0.miss_rate_mean, 4) / round(_gb.miss_rate_mean, 4)
check("anomaly probe ratio, exact", 46, _ratio, 0.8)
check("anomaly probe ratio, from displayed values", 46, _disp, 0.8)
# The ratio itself was dropped: it divides a rate by a rate, which is sound, but
# the corrected-weighting figure belongs beside it, so both are quoted instead.
rule("the anomaly-input result shows both values and the corrected one",
     "0.0092 to 0.4203" in TXT and "0.3307" in TXT,
     f"exact {_ratio:.2f}, displayed {_disp:.2f}")
rule("the superseded factor of forty-three is gone", "forty-three" not in TXT)

# Blocker 2: the manuscript must not contain an unfinished-work note to its
# own authors. The placeholder fails until the real version is supplied.
rule("no to-do note left in the manuscript",
     "should be stated before submission" not in TXT and "before submission" not in TXT)

# Blocker 3: availability must not sound as though reproducibility stops early.
rule("availability statement covers every table",
     "every table and figure in the manuscript and the supporting information" in TXT
     and "Tables 7 through 13" not in TXT)

# Strong recommendations from the same pass
rule("no claim of field results", "in the field" not in TXT)
rule("opening sentence is supported rather than universal",
     "far more abnormal indications in a week than any operator can act on" not in TXT)
rule("no rhetorical asides about the cost of an argument", "cheap to make" not in TXT)
rule("nomenclature names the bound the way the method does",
     "one-sided independence-reference binomial upper bound on a rate" in TXT)
rule("the experimental units are distinguished in the nomenclature",
     all(k in TXT for k in ("*N*~test~", "*N*~opp~", "*N*~run~")))
rule("the five-level / four-action distinction is made once",
     TXT.count("four-action") <= 1 and TXT.count("five-level") == 0)

# ============ G. systematic ratio and count audit (A5, A4) ============
# Every derived ratio the manuscript quotes, recomputed from the result files and
# also from the values as displayed, so a rounded-but-wrong ratio cannot pass.

def ratio_rule(label, stated, exact, displayed, tol):
    """A quoted ratio must match both the exact computation and the displayed one."""
    N[0] += 1
    if abs(stated - exact) > tol or abs(stated - displayed) > tol:
        FAIL.append(f"[ratio] {label}: manuscript {stated}, exact {exact:.2f}, "
                    f"from displayed values {displayed:.2f}")

# Plan item A2: the headline ratio follows the corrected Table 10, so it is
# checked against the exact worst-partition bounds AND against what a reader
# recomputes from the two printed figures.
_lo_b, _hi_b = min(_sk.values()), max(_sk.values())
# The headline used to be the ratio of those two bounds, quoted as 86. A bound on
# an observed zero is almost entirely a property of the convention that produced
# it: the same zero is 0.19% under the independence reference and 3.72% under the
# disjoint-support convention, so the ratio is 86 or 5.9 with no model changing.
# The counts are what separates the models, so the counts are what is quoted, and
# where a ratio appears both conventions must appear with it.
rule("the headline separation is stated as counts, not as a ratio of bounds",
     "factor of about 86" not in TXT
     and "507 of 7,905" in TXT
     and "no missed hazard in any of five partitions" in TXT)
rule("a quoted bound ratio carries both conventions",
     ("86" not in TXT.split("## References")[0]
      or ("5.9 under the conservative disjoint-support convention" in TXT
          and "3.72%" in TXT)))
rule("the two extreme bounds are still printed",
     f"{_lo_b:.2f}%" in TXT and f"{_hi_b:.2f}%" in TXT, f"{_bratio:.1f}")
rule("the worst-partition rule is named where the ratio is used",
     "worst-partition bound" in TXT and "worst-partition independence-reference bound" in TXT)
# and the counts behind the two extremes are printed, not just the bounds
rule("the per-partition counts behind the worst row are shown",
     "4, 23, 30, 216 and 234 misses" in TXT and "0, 0, 489, 489 and 988" in TXT)

# protocol effect against model spread (the 1.7x claim)
_g2 = (_lk2["random"] - _lk2["blocked_embargo"]) * 100
_temporal2 = [m for m in _lk2.index if m != "ThresholdRule"]
ratio_rule("protocol effect / model spread", 1.7,
           _g2[_temporal2].mean() / _spread, _g2[_temporal2].mean() / _spread, 0.15)

# escalation spread on the held-out Mixture class
ratio_rule("escalation spread", 200, _ratio_esc := _mix.escalation_mean.max() / _mix.escalation_mean.min(),
           1.0 / round(_mix.escalation_mean.min(), 4), 6)


# the disjoint-support widening factor
_w_h = (1 - 0.05 ** (1.0 / _neff_h)) / (1 - 0.05 ** (1.0 / 632))
ratio_rule("disjoint-support widening", 20, _w_h, _w_h, 1.5)

# The body tables for the two eventization analyses are condensed: some rows
# name a count of models instead of one model. That is a claim about every
# comparator not named separately for that condition, so it is checked here
# rather than trusted, and the full tables live in the supporting information.
_MODELS6 = ["A_cost_weighted", "B_unweighted", "D_mlp", "E_gbm", "RF", "ThresholdRule"]
_emg = pd.read_csv(R + "v3_episode_metrics.csv").groupby(
    ["model", "perturbation", "level"]).agg(
    det=("episodes_detected", "sum"), sus=("episodes_sustained", "sum"),
    tot=("episodes", "sum")).reset_index()
for _pt, _lv, _named, _n in [("clean", 0.0, ["E_gbm"], 5),
                             ("dropout", 1.0, ["E_gbm"], 5),
                             ("dropout", 4.0, ["D_mlp", "E_gbm"], 4),
                             ("dropout", 7.0, ["D_mlp", "E_gbm"], 4)]:
    _rest = [m for m in _MODELS6 if m not in _named]
    _sub = _emg[(_emg.perturbation == _pt) & (_emg.level == _lv)
                & (_emg.model.isin(_rest))]
    rule(f"T15 grouped row {_pt}@{_lv:.0f} covers {_n} models at 10/10",
         len(_sub) == _n and bool((_sub.det == _sub.tot).all())
         and bool((_sub.sus == _sub.tot).all()),
         f"{len(_sub)} models, det {sorted(_sub.det)}")
_pb = pd.read_csv(R + "v2_perturb.csv")

# A reference entry that lands in the body instead of the reference list is a
# failure the ordering gate cannot see, because it only reads the References
# section. It happened once: a Kumar et al. entry was inserted at the first
# occurrence of "Laberge" in the file, which was an in-text citation in the
# Introduction, splitting a sentence in half. These two gates catch the shape of
# that mistake rather than that one instance.
_BODY_TXT = TXT[:TXT.index("## References")]
_REFS_TXT = TXT[TXT.index("## References"):]
rule("no reference-style entry appears in the body",
     not re.search(r"\([A-Z][A-Za-z.\- ]+, [A-Z]\.[^)]{0,80}\d{4}\. [A-Z][^)]{20,}\*",
                   _BODY_TXT),
     "a full reference entry looks to be inline in the body")
# The availability statement is where the repository URL belongs, so the whole
# section is exempt, not just the line the heading sits on. Matching the heading
# text against each line exempted the heading and nothing under it, which is why
# putting the real repository URL into the statement tripped this gate.
_AVAIL_HEAD = "## Data and Code Availability"
if _AVAIL_HEAD in _BODY_TXT:
    _a = _BODY_TXT.index(_AVAIL_HEAD)
    _b = _BODY_TXT.find("\n## ", _a + 1)
    _PROSE_TXT = _BODY_TXT[:_a] + _BODY_TXT[_b if _b != -1 else len(_BODY_TXT):]
else:
    _PROSE_TXT = _BODY_TXT
rule("no DOI or URL appears in the body outside a table or the availability note",
     all("https://doi.org" not in ln and "https://github" not in ln
         for ln in _PROSE_TXT.split("\n")
         if not ln.startswith("|") and "onepetro" not in ln),
     "a bare link is sitting in body prose")
rule("the availability statement names the repository",
     "https://github.com/" in TXT[TXT.index(_AVAIL_HEAD):]
     if _AVAIL_HEAD in TXT else False)
# The cited commit has to be a commit, not a shortened or mistyped one, and it
# has to be the commit the tag points at wherever that can be resolved. The
# manuscript inside the tagged commit cannot contain its own hash, so the tagged
# copy names the tag and the submitted copy names the hash the tag resolves to.
_AVAIL = TXT[TXT.index(_AVAIL_HEAD):] if _AVAIL_HEAD in TXT else ""
_cited = re.findall(r"`([0-9a-f]{40})`", _AVAIL)
_defers = "git rev-parse v15-submission" in _AVAIL
rule("the release state is cited as a full hash, or as the tag that resolves to it",
     len(_cited) == 1 or _defers, f"{_cited}")
if _cited:
    import subprocess as _sp2
    try:
        _resolved = _sp2.run(["git", "rev-parse", "v15-submission^{commit}"],
                             capture_output=True, text=True, timeout=30,
                             cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except Exception:
        _resolved = ""
    if re.fullmatch(r"[0-9a-f]{40}", _resolved or ""):
        rule("the cited commit is the one the v15-submission tag points at",
             _resolved == _cited[0], f"tag is {_resolved}, paper says {_cited[0]}")
# every surname cited in the body must have an entry, and every entry must be cited
_ORG = {"EEMUA", "IEC", "ISA", "Gemma"}
_ref_surnames = set()
for _ln in _REFS_TXT.split("\n"):
    _m = re.match(r"^((?:El |de |van |Van )?[A-Z][A-Za-z\u00C0-\u017F'\-]+), ", _ln)
    if _m:
        _ref_surnames.add(_m.group(1))
    elif re.match(r"^(EEMUA|IEC|ISA|Gemma Team)\b", _ln):
        _ref_surnames.add(_ln.split(".")[0].split()[0])
_cited = set(re.findall(r"\((?:e\.g\., )?((?:El |de |van )?[A-Z][A-Za-z\u00C0-\u017F'\-]+)"
                        r"(?: et al\.| and [A-Z][A-Za-z]+)? \d{4}", _BODY_TXT))
_cited |= set(re.findall(r"\b((?:El |de |van )?[A-Z][A-Za-z\u00C0-\u017F'\-]+) et al\. \(\d{4}\)",
                         _BODY_TXT))
_orphans = sorted(c for c in _cited if c not in _ref_surnames and c not in _ORG)
rule("every surname cited in the body has a reference entry", not _orphans, str(_orphans))

# Figure resolution. The four SPE Journal model papers carry 300 to 430 dpi
# after the publisher downsamples, from source art delivered higher; 600 dpi is
# the floor adopted here. Checked on the rendered files, because a savefig call
# that loses its dpi argument is exactly the kind of drift nothing else catches.
try:
    from PIL import Image as _PILImage
    _FIGDIR = "figures_v2"
    _figs = sorted(f for f in os.listdir(_FIGDIR) if f.endswith(".png"))
    _thin = []
    _offwidth = []
    for _f in _figs:
        with _PILImage.open(os.path.join(_FIGDIR, _f)) as _im:
            # every figure in this paper is authored at most 11.5 in wide, so a
            # 600 dpi render is at least 600 * its own width; the conservative
            # check is against the narrowest author width used, 3.35 in.
            if _im.size[0] < 600 * 3.35:
                _thin.append((_f, _im.size[0]))
            # Authored width. Every figure is placed at the 6.50 in text width,
            # so its type renders at (authored font) x 6.50 / (authored width):
            # a figure authored much wider than the rest arrives with smaller
            # labels than its neighbours, which is the defect this catches. The
            # earlier drafts authored the result figures at 11.5 in and Word
            # scaled them to half size.
            _w_in = _im.size[0] / float(_im.info.get("dpi", (600, 600))[0])
            if not (6.4 <= _w_in <= 7.2):
                _offwidth.append((_f, round(_w_in, 2)))
    rule("every figure is rendered at 600 dpi or better", not _thin, str(_thin))
    rule("every figure is authored within 6.4 to 7.2 in, so labels render at a "
         "common size once placed at the 6.50 in text width",
         not _offwidth, str(_offwidth))
    rule("the figure inventory is complete", len(_figs) >= 20, f"{len(_figs)} files")
    # Retired figures must be gone from the figure directory too, not just from
    # the documents, so a stale file cannot be picked up by a later build.
    _stale = sorted(f for f in _figs if f[:-4] in RETIRED)
    rule("no retired figure file survives in the figure directory", not _stale, str(_stale))
    # Clipping. A long axis label or a wide legend can run past the canvas and be
    # cut off in the saved file; the tight bounding box does not always catch it.
    # Ink in the outermost two pixel rows or columns is the signature.
    import numpy as _np
    _clipped = []
    for _f in _figs:
        with _PILImage.open(os.path.join(_FIGDIR, _f)) as _im:
            _a = _np.asarray(_im.convert("L"))
        if ((_a[:, :2] < 210).any() or (_a[:, -2:] < 210).any()
                or (_a[:2, :] < 210).any() or (_a[-2:, :] < 210).any()):
            _clipped.append(_f)
    rule("no figure has content cut off at a canvas edge", not _clipped, str(_clipped))
    # Overlapping text. Twenty-four figures is more than anyone re-reads by eye
    # after every regeneration, so every pair of text artists in every figure is
    # tested for real overlap (rotated boxes, off-view tick labels excluded).
    try:
        import subprocess as _sp
        _r = _sp.run([sys.executable, "check_figure_text.py"], capture_output=True,
                     text=True, timeout=900, cwd=os.path.dirname(os.path.abspath(__file__)))
        _bad = [ln.strip() for ln in _r.stdout.splitlines() if ln.strip().startswith("FAIL")]
        rule("no figure has overlapping text", _r.returncode == 0 and not _bad,
             "; ".join(_bad)[:400] or f"exit {_r.returncode}")
    except Exception as _exc:
        rule("the figure text check runs", False, repr(_exc))
    # Placement. A figure with no explicit width is sized by pandoc from its dpi
    # metadata and capped at the reference document's text width, which is how
    # the figures came out at 5.83 in rather than 6.50 in.
    _unpinned = sorted(set(re.findall(r"!\[\]\(figures_v2/[a-z_0-9]+\.png\)(?!\{)",
                                      TXT + SITXT)))
    rule("every figure is placed at an explicit width", not _unpinned, str(_unpinned))
    rule("figures are placed at the 6.50 in text width",
         (TXT + SITXT).count("{width=6.50in}") == 22,
         str((TXT + SITXT).count("{width=6.50in}")))
except ImportError:
    pass

# eventization compression range
_ep = pd.read_csv(R + "v3_alarm_episodes.csv")
_epn = _ep[_ep.episodes_mean > 0]
check("compression minimum", 5, _epn.compression.min(), 1.0)
check("compression maximum", 316, _epn.compression.max(), 1.0)
rule("compression range quoted as 5 to 316",
     "between 5 and 316 indications per episode" in TXT,
     f"{_epn.compression.min():.1f} to {_epn.compression.max():.1f}")

# counts the manuscript asserts about its own design
rule("model counts consistent",
     "Eleven comparators" in TXT and len(pw.drop(index="Ordinal")) == 10,
     f"{len(pw)} rows in the 30-run table")
check("seeds in the five-seed protocol", 5, len(pd.read_csv(R + "v2_calibration_perseed.csv")
                                               .seed.unique()), 0.5)
check("runs in the powered protocol", 30,
      len(_json.load(open(R + "v3_powered_bayes.json"))["per_run_acc"]["A_cost_weighted"]), 0.5)
rule("window and partition counts consistent",
     all(k in TXT for k in ("6,324 windows", "1,581 per class", "4,900 training", "1,264 test")))

# A6: no claim that results carry beyond the corpus
rule("no external-validity claim in the Summary",
     "carry outside this corpus" not in TXT)
# A7: framework, not an established qualification test
rule("contribution described as an evaluation framework",
     ("pre-deployment evaluation framework" in TXT
      or "pre-deployment evaluation methodology" in TXT
      or "benchmark study of learned gas-monitoring decision models" in TXT)
     and "a pre-deployment screen" not in TXT)
# A13, A15
rule("escalator claims scoped to the evaluated models",
     "strongest escalators" not in TXT or "Among the evaluated models" in TXT)
# The old rule banned "would clear" outright, to stop the paper implying a metric
# certifies a model. The sentence that now uses it says the opposite: a miss-only
# screen would clear BOTH comparators and separate neither, which is the failure
# being demonstrated. The rule is narrowed to the certifying sense.
rule("no metric is described as clearing a model for service",
     not any(x in TXT for x in ("would clear it for", "clears the model", "would clear the model")))
# C1
rule("defensive register reduced",
     not any(x in TXT for x in ("understates the field case", "cheap to honor",
                                "is blunt:", "otherwise unfalsifiable", "cheap to adopt")))
# A2: the verification artifact matches the manuscript version
rule("verification artifact version matches the manuscript",
     "verify_v15.py" in TXT and "verify_v12" not in TXT)

# --- B8: equation rendering. The underscore form printed literally in Word
# ("R_miss", "Sigma_{t in H}"), so every symbol carries a real subscript now.
rule("no literal underscore subscripts survive in the equations",
     not re.search(r"\*[A-Za-z]\*_[a-z]", TXT) and "Σ_{" not in TXT)
rule("Eq. 2 is not run together with the following sentence",
     "............ (2) Each" not in TXT)


# ============ H. checks added after the five-part adversarial review ============
# Each of these exists because the review showed a specific claim reaching past
# what the data support, or a specific artifact missing from the release.

# --- 3.12: the persistence rule is a convention, so its influence is measured.
_ps = pd.read_csv(R + "v3_persistence_sweep.csv")
_pr = _json.load(open(R + "v3_persistence_rank.json"))
rule("the persistence sweep is reported, not just promised",
     "Table S7" in SITXT and "S9 Sensitivity of the Eventized Alarm Result" in SITXT
     and "bounded below" not in TXT)
_grid = len(_pr["on_grid"]) * len(_pr["off_grid"])
rule("the number of persistence settings is stated correctly",
     _grid == 25 and "twenty-five" in TXT and "twenty-five" in SITXT, str(_grid))

# The single-channel reordering is the one eventization claim the manuscript
# leans on. It is asserted to survive every grid point, so check that directly
# rather than trusting the sentence.
_d1 = _ps[(_ps.perturbation == "dropout") & (_ps.level == 1.0)]
_tree = _d1[_d1.model == "ThresholdRule"].set_index(["on_delay", "off_delay"])["episodes_per_hour"]
_cost = _d1[_d1.model == "A_cost_weighted"].set_index(["on_delay", "off_delay"])["episodes_per_hour"]
_strict = int((_tree < _cost).sum()); _rev = int((_tree > _cost).sum())
rule("the single-channel eventized reordering survives the whole grid",
     _rev == 0 and _strict == 20 and f"strictly less at twenty of them" in TXT,
     f"strict {_strict}, reversed {_rev}")

# The EEMUA compliance reading must be stated as rule-conditional, and the two
# numbers that break it must match the sweep.
_noise5 = _ps[(_ps.perturbation == "noise") & (_ps.level == 0.5)].episodes_per_hour.max()
_noise3 = _ps[(_ps.perturbation == "noise") & (_ps.level == 0.3)].episodes_per_hour.max()
_noise3_dep = _ps[(_ps.perturbation == "noise") & (_ps.level == 0.3)
                  & (_ps.on_delay == 3) & (_ps.off_delay == 15)].episodes_per_hour.max()
check("persistence sweep noise 0.5 maximum", 109.0, _noise5, 0.5)
check("persistence sweep noise 0.3 maximum", 50.0, _noise3, 0.2)
rule("the deployed rule really does report zero at noise 0.3", _noise3_dep == 0.0, str(_noise3_dep))
rule("the EEMUA reading is stated as conditional on the rule",
     "Under this persistence rule every condition tested falls within or near" in TXT
     and "not a property of the models" in TXT)

# --- 5.x: the metric-cost claim now has a driver and the numbers come from it.
_mc = _json.load(open(R + "v3_metric_cost.json"))
rule("the metric-cost claim has a released driver",
     "retrain/run_metric_cost.py" in TXT or "retrain/run_metric_cost.py" in SITXT)
check("S8 metric-set cost", 0.455, _mc["ms_full_metric_set"], 6e-3)
check("S8 accuracy-only cost", 0.290, _mc["ms_accuracy_only"], 6e-3)
check("S8 cost ratio", 1.57, _mc["ratio"], 6e-3)

# --- A.7: the provenance table, and the two directions of the release audit.
rule("a provenance table maps every reported object to its source",
     "Table A-1—Provenance" in TXT and "| Reported object | Driver |" in TXT)
for _f in ("v3_persistence_sweep.csv", "v3_persistence_rank.json", "v3_metric_cost.json"):
    rule(f"{_f} is named in the provenance table", _f in TXT, "missing")

# --- 5.6: stale drivers carry a deprecation notice and produce nothing here.
for _dep in ("run_exp1_real", "run_exp3_loco", "run_exp4_calibration", "run_expzoo"):
    _h = open(f"retrain/{_dep}.py", encoding="utf-8").read(40)
    rule(f"{_dep}.py is marked deprecated", _h.startswith('"""DEPRECATED'), _h[:30])
rule("the deprecated drivers are declared in the appendix",
     "Deprecated paths." in TXT and "run_expzoo.py" in TXT)

# --- 3.3: the bound must read as a bound, never as an estimate of the rate.
rule("the missed-hazard bound is not phrased as an estimate",
     "consistent with a true rate near one in eleven" not in TXT
     and "does not exclude a true rate of roughly 9%" in TXT)
rule("the direction of the bound is stated explicitly",
     "what the data fail to exclude under a conservative counting rule, not an estimate" in TXT
     and "nothing here puts the missed-hazard rate at 9.21%" in TXT)
# Plan item A28: the disjoint-support counts are a conservative reference, not
# an effective sample size, and the manuscript must say so where they are built.
_ess = TXT.count("effective sample size") + SITXT.count("effective sample size")
rule("disjoint-support counts are not called an effective sample size",
     _ess == 1 and "rather than an effective sample size" in TXT, f"{_ess} uses")

# --- 3.9: ten models are not a sample, so the Fisher intervals are not CIs.
rule("the cross-model correlation intervals are labelled exploratory",
     "exploratory reference intervals rather than confidence intervals" in TXT
     and "with a 95% confidence interval of" not in TXT)

# --- 2.3: the class-disjoint experiment is forced closed-set classification.
rule("the absence of a reject option is stated",
     "No comparator has a reject option" in TXT
     and "neither an open-set method nor a scoring of one" in TXT)

# --- 2.6 and plan item A4: the ten Table 14 entries are five runs times two
# classes over two physical acquisitions. Attaching a binomial interval to them
# is pseudo-replication. An earlier revision did exactly that, quoting lower
# bounds of 0.39 and 0.74, so the check is that the inference is gone and stays
# gone rather than that its arithmetic is right.
rule("no binomial inference is attached to the hazard episodes",
     not any(x in TXT for x in ("lower bound on detection", "establishes only that detection exceeds",
                                "one-sided 95% independence-reference lower bound")))
rule("the episode counts are declared run-by-class, not physical episodes",
     "no episode-detection probability is inferred" in TXT
     and "Ten is five runs times two hazardous classes, not ten physical hazard episodes" in TXT)

# --- 3.4, 3.6, 3.8: computational replication is not physical replication.
rule("run-level SDs are declared computational",
     "quantify computational variability across seeds" in TXT)
rule("the Bayesian test is offered as a sensitivity analysis",
     "a sensitivity analysis over an assumed dependence structure rather than a calibrated posterior" in TXT)
rule("bootstrap agreement is not claimed as independent confirmation",
     "not that independent evidence confirms it, since the two analyses share every observation" in TXT)

# Prose that quotes a posterior must agree with the artifact. The rho correction
# left five stale values in a sentence outside any table, which the table-cell
# checks could not see. Every posterior named in prose is now checked.
_bay_all = _json.load(open(R + "v3_powered_bayes.json"))["vs_cost_weighted"]
for _k, _nm in [("LSTM", "recurrent network"), ("CQL", "conservative Q-learning"),
                ("D_mlp", "multilayer perceptron"), ("E_gbm", "gradient boosting"),
                ("RF", "random forest")]:
    _v = f"{_bay_all[_k]['p_practically_equivalent']:.3f}"
    rule(f"prose posterior for the {_nm} matches the artifact",
         f"{_nm} ({_v})" in TXT, f"expected {_v}")
rule("the perceptron posterior quoted in Experiment 3 matches the artifact",
     f"posterior probability {_bay_all['D_mlp']['p_practically_equivalent']:.3f}" in TXT)

# --- the persistence rule is named as a limitation in its own right.
rule("the persistence convention appears in Limitations",
     "*The Persistence Rule Is a Convention.*" in TXT)

# ============ I. Experiments 8 and 9, and the overlap control ============
# Every cell of the two new body tables is recomputed from its result file, and
# the claims made about them in prose are checked against the same files.

_rej = pd.read_csv(R + "v3_reject_option.csv")
_RK = {v: k for k, v in K.items()}
for row in table_after("Table 12—Escalation adequacy on the held-out Mixture"):
    k = K[row[0]]
    g = _rej[(_rej.held_out == "Mixture") & (_rej.model == k) & (_rej.reject_policy == "alarm")]
    g = g.set_index("reject_quantile")
    check(f"T12 {k} closed-set escalation", num(row[1]), g.loc[0, "escalation_mean"])
    check(f"T12 {k} escalation at q=10", num(row[2]), g.loc[10, "escalation_mean"])
    check(f"T12 {k} escalation at q=30", num(row[3]), g.loc[30, "escalation_mean"])
    check(f"T12 {k} held-out rejection at q=30", num(row[4]),
          100 * g.loc[30, "reject_rate_heldout"], 6e-2)
    check(f"T12 {k} clean rejection at q=30", num(row[5]),
          100 * g.loc[30, "reject_rate_clean"], 6e-2)

# the two claims the reject-option discussion rests on
_svm = _rej[(_rej.held_out == "Mixture") & (_rej.model == "SVM") & (_rej.reject_policy == "alarm")].set_index("reject_quantile")
rule("the reject-option repair is quoted correctly",
     f"reaches {_svm.loc[30, 'escalation_mean']:.4f} escalation once it may decline" in TXT)
_sep = (_rej[(_rej.reject_quantile == 30) & (_rej.reject_policy == "alarm")]
        .assign(sep=lambda d: 100 * (d.reject_rate_heldout - d.reject_rate_clean)))
rule("the separation range is quoted correctly",
     f"+{_sep.sep.max():.1f} points" in TXT and f"−{abs(_sep.sep.min()):.1f} points" in TXT,
     f"{_sep.sep.min():.1f} to {_sep.sep.max():.1f}")
rule("the reject option is not offered as a fix",
     "The intervention does not remove the trade-off the paper documents" in TXT)

_rf = pd.read_csv(R + "v3_raw_faults.csv")
_LBL = {"None": ("clean", 0.0), "Stuck at last value": ("stuck", None),
        "Forced raw zero": ("zero", None), "High-end clamp": ("saturate", None),
        "Raw additive noise": ("rawnoise", None)}
for row in table_after("Table 13—Faults injected in raw sensor space"):
    fault = _LBL[row[0]][0]
    lvl = 0.0 if fault == "clean" else float(re.search(r"[\d.]+", row[1]).group())
    g = _rf[(_rf.fault == fault) & (_rf.level == lvl)]
    check(f"T13 {fault}@{lvl} worst miss", num(row[2]), g.miss_rate_mean.max())
    check(f"T13 {fault}@{lvl} worst under-escalation", num(row[3]), g.under_escalation_mean.max())
    check(f"T13 {fault}@{lvl} worst false alarm", num(row[4]), g.fa_per_clean_mean.max())
    check(f"T13 {fault}@{lvl} normalized anomaly feature", num(row[5]), g.anomaly_mean_mean.mean(), 6e-3)

# the saturation finding is the point of Experiment 9, so it is gated hard
_clean_anom = _rf[_rf.fault == "clean"].anomaly_mean_mean.mean()
_sat4 = _rf[(_rf.fault == "saturate") & (_rf.level == 4.0)]
_sat7 = _rf[(_rf.fault == "saturate") & (_rf.level == 7.0)]
rule("the high-end clamp lowers the normalized anomaly feature below its clean value",
     _sat4.anomaly_mean_mean.mean() < _clean_anom and _sat7.anomaly_mean_mean.mean() < _clean_anom,
     f"clean {_clean_anom:.3f}, k=4 {_sat4.anomaly_mean_mean.mean():.3f}, k=7 {_sat7.anomaly_mean_mean.mean():.3f}")
rule("a total high-end clamp is a silent failure on every comparator",
     bool((_sat7.escalation_mean == 0).all() and (_sat7.miss_rate_mean == 0).all()
          and (_sat7.fa_per_clean_mean == 0).all()))
check("the worst raw-fault missed-hazard rate", 0.3930, _rf.miss_rate_mean.max(), 6e-4)
rule("the silent-failure reading is stated",
     "Three of the four reported metrics read as perfect while nothing whatsoever reaches an operator." in TXT)
rule("the electrical-zero case is no longer described as untested",
     "the electrical-zero case is untested" not in TXT)

_ov = pd.read_csv(R + "v3_overlap_isolation.csv")
_eo = _ov[_ov.arm == "effect_overlap_pp"].decision_acc_mean
_ep = _ov[_ov.arm == "effect_position_pp"].decision_acc_mean
check("overlap effect, mean over models", 1.56, _eo.mean(), 6e-3)
check("position effect, mean over models", -1.30, _ep.mean(), 6e-3)
rule("the isolation control is reported as a null result",
     "a null result rather than a decomposition" in TXT
     and "not separable on this corpus" in SITXT)
rule("the matched training budget is stated",
     "256-window training budget" in TXT and str(int(_ov.n_train_budget.iloc[0])) == "256")

# The self-audit and the metric tests are part of the release and are named in
# Appendix A, so their existence is gated here rather than taken on trust.
import os as _os
for _f in ("self_audit.py", "test_safety_metrics.py"):
    rule(f"{_f} ships with the release", _os.path.exists(_f))
# Submission blockers: facts only the authors hold. Each fails loudly until it is
# supplied, rather than sitting in the paper as a promise to add it later.
for _tok, _what in (("⟨ULTRALYTICS_VERSION⟩", "the Ultralytics version"),
                    ("⟨REPOSITORY_URL⟩", "the repository URL"),
                    ("⟨COMMIT_SHA⟩", "the release commit SHA"),
                    ("⟨ARCHIVE_DOI⟩", "the archival DOI")):
    rule(f"{_what} is supplied, not deferred", _tok not in TXT,
         "placeholder still present")
rule("no promise to add the DOI later survives the availability statement",
     "once minted" not in TXT and "will carry a DOI" not in TXT)

# Publishers now ask that use of a generative AI tool be disclosed rather than
# left to be inferred. The disclosure has to say what the tool did, what it did
# not do, and who is answerable for the result.
_ACK = TXT[TXT.index("## Acknowledgments"):TXT.index("## Author Contributions")] \
    if "## Acknowledgments" in TXT and "## Author Contributions" in TXT else ""
rule("the use of a generative AI tool is disclosed",
     "Use of artificial intelligence" in _ACK)
rule("the disclosure says what the tool was not used for",
     "not used to generate data, results or references" in _ACK)
rule("the disclosure places responsibility with the authors",
     "take full responsibility" in _ACK)

rule("the checking scripts are described in Appendix A",
     "self_audit.py" in TXT and "test_safety_metrics.py" in TXT
     and "audit_release.py" in TXT and "check_figure_text.py" in TXT
     and "check_page_proof.py" in TXT and "build_docx.py" in TXT)

# --- The page proof. Three defects found in the final proof existed only after
# the sources had become a paginated document: a caption symbol dropped by the
# renderer, words broken mid-word inside narrow table columns, and table rows
# breaking across a page. The build and the proof are scripted so they cannot
# recur unnoticed.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _script in ("build_docx.py", "check_page_proof.py"):
    rule(f"{_script} is present", os.path.exists(os.path.join(_HERE, _script)),
         "missing")
rule("the document build is scripted, not typed by hand",
     "docx:" in open(os.path.join(_HERE, "Makefile"), encoding="utf-8").read()
     and "proof:" in open(os.path.join(_HERE, "Makefile"), encoding="utf-8").read())
# Inline TeX becomes an OMML run that Word renders and several other readers drop
# silently, taking the symbol with it. Fig. 5's caption lost its V and its A that
# way, and read " = value stream" in every renderer but Word.
_math = re.findall(r"(?<!\\)\$[^$\n]+\$", TXT + SITXT)
rule("no caption or sentence carries inline TeX math", not _math, str(_math[:3]))

# The ordinal objective's action vocabulary, read out of the sentence that
# states it and checked against the file that sentence cites. This is the gate
# that was missing when the denominator was written as the corpus window count
# rather than the count in v3_ordinal_actions.csv; the two differ by four.
_oa = pd.read_csv(os.path.join(R, "v3_ordinal_actions.csv"))
_m_ord = re.search(r"assigns action 0 to no window[^,]*, 0 of ([\d,]+)", TXT)
rule("the ordinal action claim states its denominator", _m_ord is not None)
if _m_ord:
    _claimed = int(_m_ord.group(1).replace(",", ""))
    _actual = int(_oa.groupby(["seed", "gas"]).n.first().sum())
    rule("the ordinal denominator is the count in the file it cites",
         _claimed == _actual, f"manuscript says {_claimed}, file gives {_actual}")
rule("action 0 really is emitted zero times",
     int(_oa[_oa.action == 0]["count"].sum()) == 0,
     f"{int(_oa[_oa.action == 0]['count'].sum())} emissions of action 0")
_m_pf = re.search(r"and ([\d,]+) of ([\d,]+) Perfume windows to action 4", TXT)
rule("the Perfume action-4 count is stated", _m_pf is not None)
if _m_pf:
    _pf_claim = int(_m_pf.group(1).replace(",", ""))
    _pf_n = int(_m_pf.group(2).replace(",", ""))
    _pf_rows = _oa[_oa.gas == "Perfume"]
    rule("the Perfume action-4 count matches the file",
         int(_pf_rows[_pf_rows.action == 4]["count"].sum()) == _pf_claim,
         f"manuscript says {_pf_claim}, file gives {int(_pf_rows[_pf_rows.action == 4]['count'].sum())}")
    rule("the Perfume denominator matches the file",
         int(_pf_rows.groupby('seed').n.first().sum()) == _pf_n,
         f"manuscript says {_pf_n}, file gives {int(_pf_rows.groupby('seed').n.first().sum())}")

print(f"{N[0]} checks run, {len(FAIL)} failure(s)")
for f in FAIL:
    print("  FAIL", f)
sys.exit(1 if FAIL else 0)
