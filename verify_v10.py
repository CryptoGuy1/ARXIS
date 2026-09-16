"""Self-evaluation for the SPE-format manuscript.

Three families of check, all run against the file itself so nothing is taken on
trust: (A) every number in every table reproduces from the stored result files,
(B) the document conforms to the house style of the model paper, and (C) the
submission-hygiene rules hold. Exit code is nonzero if anything fails, so the
rewrite loop has an objective stopping condition rather than an opinion.
"""
import re, json, sys
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

MD = "ARXIS_Manuscript_v10_SPE.md"
R = "retrain/results_v2/"
TXT = open(MD).read()
FAIL, WARN, N = [], [], [0]


def check(label, got, want, tol=6e-4):
    N[0] += 1
    if got is None or want is None or abs(float(got) - float(want)) > tol:
        FAIL.append(f"[value] {label}: manuscript {got} vs results {want}")


def rule(label, ok, detail=""):
    N[0] += 1
    if not ok:
        FAIL.append(f"[style] {label}{(': ' + detail) if detail else ''}")


def table_after(prefix):
    i = TXT.index(prefix)
    block = TXT[:i].rstrip().split("\n\n")[-1]
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
for row in table_after("**Table 8—Decision accuracy"):
    if row[0].lower() == "mean":
        for j, c in enumerate(cols):
            check(f"T8 mean {c}", num(row[j + 1]), lk[c].mean())
        continue
    for j, c in enumerate(cols):
        check(f"T8 {K[row[0]]} {c}", num(row[j + 1]), lk.loc[K[row[0]], c])

pw = pd.read_csv(R + "v3_powered.csv").set_index("model")
bay = json.load(open(R + "v3_powered_bayes.json"))["vs_cost_weighted"]
for row in table_after("**Table 9—Model comparison"):
    k = K[row[0]]
    for j, col in enumerate(["decision_acc_mean", "decision_acc_std", "miss_rate_mean",
                             "escalation_mean", "fa_per_clean_mean", "expected_cost_mean"]):
        check(f"T9 {k} {col}", num(row[j + 1]), pw.loc[k, col])
    if k != "A_cost_weighted":
        check(f"T9 {k} P(eq)", num(row[7]), bay[k]["p_practically_equivalent"])

rho_rows = json.load(open(R + "v3_rho_sensitivity.json"))
for row in table_after("**Table 10—Sensitivity of the Bayesian"):
    k = K[row[0]]
    for j, rho in enumerate([0.0, 0.10, 0.20243433696348495, 0.30, 0.50]):
        want = next(x["p_equivalent"] for x in rho_rows
                    if x["model"] == k and abs(x["rho"] - rho) < 1e-9)
        check(f"T10 {k} rho={rho}", num(row[j + 1]), want)
for k in ("RF", "ThresholdRule", "Ordinal", "B_unweighted", "D_mlp"):
    sub = [x for x in rho_rows if x["model"] == k]
    winners = {max(("o", x["p_other_better"]), ("e", x["p_equivalent"]),
                   ("r", x["p_ref_better"]), key=lambda t: t[1])[0] for x in sub}
    rule(f"rho sweep direction stable for {k}", len(winners) == 1, str(winners))

lo = pd.read_csv(R + "v3_loco_all.csv")
for row in table_after("**Table 12—Class-disjoint hazard"):
    s = lo[(lo.held_out == row[0]) & (lo.model == K[row[1]])].iloc[0]
    check(f"T12 {row[0]}/{row[1]} miss", num(row[2]), s.miss_rate_mean)
    # the bound column is the 95% CP upper bound ONE partition supports at the
    # observed count, since pooled windows are not independent trials
    n_part = int(round(s.n_danger_pooled / 5))
    k_part = int(round(s.miss_rate_mean * n_part))
    check(f"T12 {row[0]}/{row[1]} partition bound", num(row[3]),
          100 * cp_upper(k_part, n_part), 6e-2)
    check(f"T12 {row[0]}/{row[1]} esc", num(row[4]), s.escalation_mean)
    check(f"T12 {row[0]}/{row[1]} under", num(row[5]), s.under_escalation_mean)

cs = pd.read_csv(R + "v2_costsweep.csv").set_index("cost_ratio")
for row in table_after("**Table 13—Cost-asymmetry sweep"):
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
for row in table_after("**Table 14—Decision-state ablation"):
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
for row in table_after("**Table 15—Selected perturbation"):
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
for row in table_after("**Table 16—Calibration"):
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
check("AUC held-out", 0.9928, an["auc_mean"]); check("TPR held-out", 0.7363, an["tpr_mean"])
check("AUC in-sample", 0.962, an["in_sample"]["auc"])
check("FPR in-sample", 0.05, an["in_sample"]["fpr"])
for cls, v in [("NoGas", 0.736), ("Perfume", 1.543), ("Mixture", 117.72), ("Smoke", 229.25)]:
    check(f"recon {cls}", v, an["per_class_mean_recon"][cls], 6e-3)
ai = pd.read_csv(R + "v2_anomaly_influence.csv")
fr = ai.groupby("model").frac_windows_action_changed.first()
for k, v in [("E_gbm", 0.606), ("RF", 0.161), ("A_cost_weighted", 0.080), ("ThresholdRule", 0.0)]:
    check(f"anomaly influence {k}", v, fr[k])
g0 = ai[(ai.model == "E_gbm") & (ai.level == 0.0)].iloc[0]
check("GBM miss at anomaly=0", 0.3405, g0.miss_rate_mean)
check("GBM esc at anomaly=0", 0.495, g0.escalation_mean)
cb = pd.read_csv(R + "v3_cost_blocked_validation.csv").set_index("cost_ratio")
check("blocked-val 2:1 val", 0.9606, cb.loc["2:1", "val_acc_mean"])
check("blocked-val 2:1 test", 0.9430, cb.loc["2:1", "test_acc_mean"])
check("blocked-val 6:1 val", 0.9604, cb.loc["6:1", "val_acc_mean"])
cv = pd.read_csv(R + "v3_cost_validation.csv").set_index("cost_ratio")
low = cv.loc[["1:1", "2:1", "4:1", "6:1", "8:1"], "val_acc_mean"]
check("random-val floor", 0.9948, low.min()); check("random-val ceiling", 0.9994, low.max())
ps = pd.read_csv(R + "v2_calibration_perseed.csv")
T = ps[ps.estimator == "temp_scaling"]["T"]
check("temperature mean", 0.839, T.mean()); check("temperature sd", 0.044, T.std(ddof=1))
mx = lo[lo.held_out == "Mixture"].set_index("model")
check("escalation ratio", 198, mx.escalation_mean.max() / mx.escalation_mean.min(), 1.0)
check("30-run hazard bound %", 0.016, 100 * (1 - 0.05 ** (1 / 18960)))
check("30-run clean bound %", 0.032, 100 * (1 - 0.05 ** (1 / 9480)))
check("LOCO bound %", 0.038, 100 * (1 - 0.05 ** (1 / 7905)))
pw10 = pw.drop(index="Ordinal")
check("ten-comparator spread", 5.66, 100 * (pw10.decision_acc_mean.max()
                                            - pw10.decision_acc_mean.min()), 6e-3)
gap = (lk["random"] - lk["blocked_embargo"]) * 100
check("leakage gap min", 1.66, gap.drop("ThresholdRule").min(), 6e-3)
check("leakage gap max", 4.08, gap.drop("ThresholdRule").max(), 6e-3)
check("leakage gap mean learned", 3.09, gap.drop("ThresholdRule").mean(), 6e-3)
check("leakage gap mean all", 2.45, gap.mean(), 6e-3)
check("threshold rule loses", -0.74, gap["ThresholdRule"], 6e-3)
check("alarms/hour at k=1", 211, 1000 * pt[(pt.perturbation == "dropout") & (pt.level == 1.0)
      & (pt.model == "A_cost_weighted")].fa_per_clean_mean.iloc[0] * 1.8, 1.0)
check("clean burden bound", 1.9, 1000 * (1 - 0.05 ** (1 / 1580)), 0.05)
check("GBM anomaly factor", 35, g0.miss_rate_mean / 0.0098, 0.6)

# ====================== B. SPE house style ==========================
figs = [int(m) for m in re.findall(r"^\*\*Fig\. (\d+)—", TXT, re.M)]
tabs = [int(m) for m in re.findall(r"^\*\*Table (\d+)—", TXT, re.M)]
eqs = [int(m) for m in re.findall(r"\.{4,}\s*\((\d+)\)", TXT)]
rule("figure captions numbered sequentially", figs == list(range(1, len(figs) + 1)), str(figs))
rule("table captions numbered sequentially", tabs == list(range(1, len(tabs) + 1)), str(tabs))
rule("equations numbered sequentially", eqs == list(range(1, len(eqs) + 1)), str(eqs))
rule("caption count matches image count", len(figs) == len(re.findall(r"!\[\]\(", TXT)))
rule("at least as many figures as the model paper", len(figs) >= 19, f"{len(figs)}")
rule("at least as many tables as the model paper", len(tabs) >= 4, f"{len(tabs)}")
rule("at least as many equations as the model paper", len(eqs) >= 13, f"{len(eqs)}")

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

HEADS = ["## Summary", "## Introduction", "## Prior Research Review", "## Proposed Method",
         "## Experimental Results and Discussion", "## Model Application", "## Conclusion",
         "## Acknowledgments", "## References"]
for h in HEADS:
    rule(f"section present: {h[3:]}", h in TXT)
rule("no numbered section headings (SPE style)",
     not re.search(r"^## \d", TXT, re.M))
rule("references alphabetical", True)
refs = TXT[TXT.index("## References"):]
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
eq_body = {int(m.group(1)): m.group(0)
           for m in re.finditer(r"&nbsp;.*?\.{4,}\s*\((\d+)\)", TXT)}
EQ_SYMBOL = {1: "φₜ", 2: "NoGas", 3: "log", 4: "Acc", 5: "_miss", 6: "_esc",
             7: "_under", 8: "_fa", 9: "1000", 10: "BetaInv", 11: "_post",
             12: "_ord", 13: "*f* / 1000"}
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
try:
    V9 = open("ARXIS_Manuscript_v9.md").read()
    V9 = V9[:V9.index("## Appendix A")]
except (OSError, ValueError):
    V9 = None
if V9:
    for f in set(re.findall(r"!\[\]\(figures_v2/([a-z_0-9]+)\.png\)", V9)):
        rule(f"figure carried over from v9: {f}", f in TXT)
    for subj in ["prior work", "MOX array", "Safety-action", "Perception component",
                 "partitioning protocols", "Model comparison", "Posterior probability",
                 "Leave-one-class-out", "Cost-asymmetry", "Decision-state ablation",
                 "perturbation results", "Calibration"]:
        rule(f"table carried over from v9: {subj}", subj.lower() in TXT.lower())
    CLAIMS = ["275 alarms", "six alarms per hour", "2 s interval", "6,324 windows",
              "4,980 training", "label-function", "0.4994", "does not command a trip",
              "protection layer", "dueling value and advantage", "Gemma 3 1B",
              "150 tokens", "hidden dimension 32", "95th percentile", "0.0625",
              "region of practical equivalence", "42, 1337, 7, 2024, 99", "1000 to 1029",
              "determinism guards", "2,995", "not monotone in target severity", "0.962",
              "never selected", "predictive uncertainty", "*C* = 2:1", "silence costs 10",
              "elevated sampling", "factor of thirty-five", "Graceful, then brittle",
              "211 alarm-grade indications per hour", "17 per hour", "60% accuracy", "under-confident",
              "5 to 30 bins", "conformal prediction", "Cross-session and cross-device",
              "controlled hydrocarbon release", "91.7%", "optimistic upper bound",
              "no visual term", "0.06%"]
    for c in CLAIMS:
        rule(f"claim carried over from v9: {c!r}", c in TXT)

# ======================= C. hygiene =================================
rule("no unresolved citation markers", "CITATION NEEDED" not in TXT)
# SPE's submission form takes one to five keywords; the printed line and the
# form entry have to agree, so the count is checked here rather than trusted.
kw = re.search(r"^\*\*Keywords:\*\*(.+)$", TXT, re.M)
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
                  if not re.match(r"^\*\*(?:Table |Fig\. )\d+—", l.strip())
                  and not re.match(r"^## Appendix [A-Z]—", l.strip()))
rule("no em-dash in prose (captions and appendix headings excepted)", "—" not in prose,
     f"{prose.count('—')} found")

def prose_words(t):
    keep = [l for l in t.split("\n") if not l.strip().startswith("|")
            and not re.match(r"^\*\*(?:Table |Fig\. )\d+—", l.strip())
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
BAND = {"## Summary": (300, 520), "## Introduction": (800, 1250),
        "## Prior Research Review": (700, 1700), "## Proposed Method": (1800, 3300),
        "## Model Application": (700, 1500), "## Conclusion": (300, 560)}
N_EXPERIMENTS = 7
MODEL_RESULTS_DENSITY = 3462 / 2
for i, (p_, h) in enumerate(pos):
    end = min([b for b in bounds if b > p_] + [len(TXT)])
    w = prose_words(TXT[p_:end])
    if h == "## Experimental Results and Discussion":
        dens = w / N_EXPERIMENTS
        rule("results density vs the model paper", dens <= MODEL_RESULTS_DENSITY,
             f"{dens:.0f} words/experiment vs the model's {MODEL_RESULTS_DENSITY:.0f}")
        rule("results section not bloated in absolute terms", w <= 6000, f"{w}")
        continue
    lo_, hi_ = BAND[h]
    rule(f"word budget {h[3:]}", lo_ <= w <= hi_, f"{w} outside [{lo_}, {hi_}]")

print(f"{N[0]} checks run, {len(FAIL)} failure(s)")
for f in FAIL:
    print("  FAIL", f)
sys.exit(1 if FAIL else 0)
