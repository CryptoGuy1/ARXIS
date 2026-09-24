"""Self-audit: the checks a referee would run by hand, run mechanically instead.

`verify_v15.py` answers "does every number match the result files". This answers
a different question: "is the document consistent with itself, and does it still
contain anything a previous revision superseded". Those are the failures that
have actually escaped in this project. Twice now a number was corrected in five
places and missed in a sixth, because nothing was checking the sixth.

Seven families, corresponding to the five publication-readiness review prompts:

  A. superseded values           nothing a prior revision replaced may survive
  B. number traceability        every figure quoted in the Summary, the research
                                question table or the Conclusion must appear
                                somewhere the reader can check it
  C. controlled vocabulary      one name per thing, everywhere
  D. cross-reference integrity  every object cited exists and is cited
  E. notation                   Nomenclature and body agree in both directions
  F. prose risk                 length, repetition, superlatives, defensiveness
  G. claim coherence            Summary, Results and Conclusion tell one story

Exit code is nonzero on any failure.

    python3 self_audit.py
"""
import re
import sys
from collections import Counter

MD = "ARXIS_Manuscript_v15_SPE.md"
SI = "ARXIS_Supporting_Information.md"
TXT = open(MD, encoding="utf-8").read()
SITXT = open(SI, encoding="utf-8").read()

FAIL, WARN = [], []
N = [0]


def bad(fam, msg):
    FAIL.append(f"[{fam}] {msg}")


def warn(fam, msg):
    WARN.append(f"[{fam}] {msg}")


def rule(fam, label, ok, detail=""):
    N[0] += 1
    if not ok:
        bad(fam, f"{label}{(': ' + detail) if detail else ''}")


def section(doc, head, nexts):
    i = doc.index(head)
    ends = [doc.index(n) for n in nexts if n in doc and doc.index(n) > i]
    return doc[i:min(ends)] if ends else doc[i:]


def prose_only(doc):
    keep = []
    for ln in doc.split("\n"):
        st = ln.strip()
        if not st or st.startswith(("|", "![", "#", ">", "&nbsp;")):
            continue
        if re.match(r"^(?:Table|Fig)\.? ?[0-9A-Z]+(?:-[0-9]+)?—", st):
            continue
        if st.startswith(("- ", "* ", "^", "\\*")):     # bullet items and the title block
            continue
        if st.startswith("Keywords:") or st.startswith("*Prepared for"):
            continue
        keep.append(st)
    # Paragraphs and display equations are separated in the source by blank
    # lines. Joining with a space would run the last sentence of one paragraph
    # into the first of the next and report a sentence nobody wrote.
    return " | ".join(keep).replace(" | ", ". ") if False else " ".join(
        (x if x.endswith((".", "!", "?", ":", ";")) else x + ".") for x in keep)


# =============================== A. superseded values ===============================
# Every entry here was a real value in some revision and was replaced. If one
# comes back, a correction was applied incompletely. The context string keeps a
# legitimate reuse of the same digits from tripping the check.
SUPERSEDED = [
    ("factor of 40", None, "the held-out-class separation is reported as counts, 0 against 507 of 7,905"),
    ("factor of about 86", None, "a ratio of bounds on an observed zero is a convention artifact; the counts are the finding"),
    (" 40 times the best", None, "same"),
    ("7.49", None, "old Smoke/MLP bound from an averaged pseudo-count; now 16.35"),
    ("26.71", None, "old Mixture/tree bound; now 64.51"),
    ("ten of eleven", None, "eight of eleven hold worst-partition bounds at or below 3.3%"),
    ("Ten of eleven", None, "same"),
    ("pass a miss-rate screen", None, "no engineering acceptance threshold exists, so nothing passes"),
    ("between 27 and 29 points", None, "the early/late losses are 41.2, 37.9 and 29.5"),
    ("Fig. 18", None, "the paper has fifteen figures"),
    # "Table 15" was superseded when the paper had fourteen tables. Experiments 8
    # and 9 took it to sixteen, so the entry is retired rather than left to fire
    # on a table that now exists. Cross-reference integrity is family D's job.
    ("exact upper bound", None, "the bound is an independence-reference quantity, not exact"),
    ("a*\\*(", None, "replaced by y(.) for the training target and A(.) for the acceptable set"),
    ("consistent with a true rate near", None, "the bound says what the data fail to exclude"),
    ("assumes nothing about within-run correlation", None, "the paired bootstrap does not remove corpus dependence"),
    ("lower bound on detection", None, "binomial inference on ten run-by-class episodes is pseudo-replication"),
    ("bounded below", None, "a forward reference that was never fulfilled; Section S9 now fulfils it"),
    ("Under Distribution Shift", None, "class exclusion is not demonstrated domain shift"),
    ("Safety-Relevant Evaluation Protocol", None, "the title says Safety-Oriented after the scope narrowing"),
]
for token, ctx, why in SUPERSEDED:
    N[0] += 1
    hits = TXT.count(token) + SITXT.count(token)
    if hits:
        bad("A", f"superseded value {token!r} still present ({hits}x): {why}")

# rho: 0.202 is superseded as a correlation, but 0.202 is also a real temperature SD
N[0] += 1
_rho_hits = len(re.findall(r"ρ\s*=\s*0\.202\b", TXT + SITXT)) + (TXT + SITXT).count("ρ = 0.202")
if _rho_hits:
    bad("A", f"superseded correlation 0.202 still quoted as rho ({_rho_hits}x); the split gives 0.2051")

# "effective sample size" is allowed exactly once, in the sentence that denies it
N[0] += 1
_ess = TXT.count("effective sample size") + SITXT.count("effective sample size")
if _ess != 1 or "rather than an effective sample size" not in TXT:
    bad("A", f"'effective sample size' used {_ess}x; exactly one negating use is allowed")


# ============================ B. number traceability ============================
# A figure quoted in the Summary, the research-question table or the Conclusion
# must be checkable somewhere else in the document. This is the check that would
# have caught the research-question table keeping "40 times" after the headline
# ratio moved to 86.
SUMMARY = section(TXT, "## Summary", ["## Introduction"])
CONCL = section(TXT, "## Conclusions", ["## Nomenclature"])
RQTABLE = TXT[:TXT.index("Table 6—Research questions")].rstrip().split("\n\n")[-1]
BODY = TXT[TXT.index("## Results"):]

NUM = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?![\w])")
IGNORE = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
          "13", "14", "15", "20", "30", "100", "1000", "1,000", "2016", "2017",
          "2021", "2022", "2023", "2024", "2025", "2026", "95", "61508", "61511", "191"}

for name, blob in [("Summary", SUMMARY), ("research-question table", RQTABLE),
                   ("Conclusion", CONCL)]:
    for tok in sorted(set(NUM.findall(blob))):
        if tok in IGNORE:
            continue
        N[0] += 1
        elsewhere = (TXT.count(tok) + SITXT.count(tok)) - blob.count(tok)
        if elsewhere < 1:
            bad("B", f"{name} quotes {tok} and nothing else in the paper does")


# ============================ C. controlled vocabulary ============================
# One name per thing. The left spelling is the one the paper uses; the right ones
# are variants that must not appear.
VOCAB = [
    ("block-wise", ["blockwise", "block wise"]),
    ("held-out hazardous", ["unseen hazard", "unseen-hazard", "unknown-gas detection"]),
    ("gradient boosting", []),
    ("support-vector classifier", []),
    ("alarm-grade indication", []),
]
for canon, variants in VOCAB:
    for v in variants:
        N[0] += 1
        n = len(re.findall(rf"\b{re.escape(v)}\b", TXT, re.I)) + len(re.findall(rf"\b{re.escape(v)}\b", SITXT, re.I))
        if n:
            bad("C", f"variant spelling {v!r} appears {n}x; the paper uses {canon!r}")

# model names must be spelled the same in prose and in every table
MODEL_NAMES = ["Cost-weighted policy", "Unweighted network", "Multilayer perceptron",
               "Gradient boosting", "Support-vector classifier", "Random forest",
               "Recurrent (LSTM)", "Conservative Q-learning", "k-nearest neighbors",
               "Shallow decision tree", "Ordinal cost objective"]
for m in MODEL_NAMES:
    N[0] += 1
    if TXT.count(m) < 1:
        bad("C", f"model name {m!r} never appears; a table may use a different spelling")

# a quantity should not be given as a decimal and a percentage of the same value
# in the same breath without reason; flag the known offender pattern
N[0] += 1
if "0.0049" in TXT and "0.49%" in TXT:
    warn("C", "escalation minimum given as both 0.0049 and 0.49%; pick one form")


# ============================ D. cross-reference integrity ============================
figs = [int(m) for m in re.findall(r"^Fig\. (\d+)—", TXT, re.M)]
tabs = [int(m) for m in re.findall(r"^Table (\d+)—", TXT, re.M)]
eqs = sorted({int(re.match(r"\d+", m).group())
              for m in re.findall(r"\.{4,}\s*\((\d+[a-z]?)\)", TXT)})
rule("D", "figure numbering is a clean run", figs == list(range(1, len(figs) + 1)), str(figs))
rule("D", "table numbering is a clean run", tabs == list(range(1, len(tabs) + 1)), str(tabs))
rule("D", "equation numbering is a clean run", eqs == list(range(1, len(eqs) + 1)), str(eqs))

for n in set(int(x) for x in re.findall(r"Figs?\.\s*\*{0,2}(\d+)", TXT)):
    N[0] += 1
    if n not in figs:
        bad("D", f"Fig. {n} is cited and has no caption")
for n in set(int(x) for x in re.findall(r"Tables?\s*\*{0,2}(\d+)", TXT)):
    N[0] += 1
    if n not in tabs:
        bad("D", f"Table {n} is cited and has no caption")
for n in set(int(x) for x in re.findall(r"Eq\.\s*\*{0,2}(\d+)", TXT)):
    N[0] += 1
    if n not in eqs:
        bad("D", f"Eq. {n} is cited and is not displayed")

# supplementary objects cited from the body must exist in the supplement
for lbl in sorted(set(re.findall(r"\*{0,2}(Table S\d+|Fig\. S\d+)\*{0,2}", TXT))):
    N[0] += 1
    if f"{lbl}—" not in SITXT:
        bad("D", f"{lbl} is cited from the body and has no caption in the supplement")
for sec in sorted(set(re.findall(r"Supplementary Section (S\d+)", TXT))):
    N[0] += 1
    if f"## {sec} " not in SITXT:
        bad("D", f"Supplementary Section {sec} is cited and does not exist")

# and nothing in the supplement should be orphaned
for lbl in sorted(set(re.findall(r"^(Table S\d+|Fig\. S\d+)—", SITXT, re.M))):
    N[0] += 1
    if lbl not in TXT and SITXT.count(lbl) < 2:
        warn("D", f"{lbl} is never cited from the body or the supplement text")


# ============================ E. notation ============================
NOM = section(TXT, "## Nomenclature", ["## Acknowledgments"])
sym_rows = re.findall(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$", NOM, re.M)
declared = [s.strip() for s, _ in sym_rows if s.strip() not in ("Symbol", "---", ":---")]
rule("E", "the nomenclature has entries", len(declared) > 10, f"{len(declared)} rows")
# Symbols are written with markdown emphasis and subscript markers, so both
# sides of the comparison are stripped before matching.
def bare(x):
    return re.sub(r"[*~`\\_ ]", "", x)

FLAT = bare(TXT) + bare(SITXT)
NOMFLAT = bare(NOM)
for sym in declared:
    for part in sym.split(","):                      # a row may declare a pair
        core = bare(part)
        if not core or len(core) > 24:
            continue
        N[0] += 1
        used = FLAT.count(core) - NOMFLAT.count(core)
        if used < 1:
            bad("E", f"nomenclature declares {part.strip()!r} and the paper never uses it")


# ============================ F. prose risk ============================
# The reference list is other people's titles, not this paper's prose, so the
# vocabulary and sentence-length checks skip it. Appendix A is this paper's
# own writing and stays in.
_NOREFS = TXT[:TXT.index("## References")]
if "## Appendix A" in TXT:
    _NOREFS += "\n\n" + TXT[TXT.index("## Appendix A"):]
PROSE = prose_only(_NOREFS)
sents = [s for s in re.split(r"(?<=[.!?])\s+", PROSE) if s.strip()]

long_ones = [s for s in sents if len(s.split()) > 52]
rule("F", "no sentence runs past 52 words", not long_ones,
     f"{len(long_ones)} found, longest {max((len(s.split()) for s in long_ones), default=0)}")

SUPERLATIVE = ["the best possible", "unprecedented", "the most important", "by far",
               "dramatically", "vastly", "perfectly", "completely solves", "state-of-the-art"]
for w in SUPERLATIVE:
    N[0] += 1
    n = len(re.findall(rf"\b{re.escape(w)}\b", PROSE, re.I))
    if n:
        bad("F", f"unsupported superlative {w!r} appears {n}x")

DEFENSIVE = ["we want to be explicit", "it is worth stressing", "we emphasize again",
             "as we have already said", "to be absolutely clear", "let us be clear",
             "the table earns its place", "that cuts both ways", "the pattern is not subtle"]
for w in DEFENSIVE:
    N[0] += 1
    n = len(re.findall(re.escape(w), PROSE, re.I))
    if n:
        bad("F", f"defensive meta-commentary {w!r} appears {n}x")

AIISH = ["delve", "robust", "innovative", "leverage", "crucial", "pivotal",
         "furthermore", "moreover", "in conclusion", "it is important to note",
         "comprehensive", "underscore", "showcase", "realm", "landscape",
         "paradigm", "seamless", "holistic", "intricate", "testament", "cutting-edge"]
for w in AIISH:
    N[0] += 1
    n = len(re.findall(rf"\b{re.escape(w)}\b", PROSE, re.I))
    if n:
        bad("F", f"overused academic vocabulary {w!r} appears {n}x")

openings = Counter(" ".join(s.split()[:2]) for s in sents if len(s.split()) > 3)
for op, n in openings.items():
    N[0] += 1
    if n > 16:
        bad("F", f"{n} sentences open with {op!r}; vary them")

rule("F", "no em-dash in prose", "—" not in PROSE,
     f"{PROSE.count('—')} found")

# a vague pronoun opening a paragraph has nothing to refer back to
# A paragraph opening "This paper" or "That boundary" names its referent. One
# opening with a bare pronoun and no noun does not, and the reader has to look
# back a paragraph to find out what is being discussed.
vague = [p.strip() for p in TXT.split("\n\n")
         if re.match(r"^(This|That|These|Those|They|It)\s+(is|are|was|were|has|have|does|do|would|will|can|also|then|leaves|means)\b",
                     p.strip())]
rule("F", "no paragraph opens with a bare pronoun and no noun", not vague,
     "; ".join(p[:50] for p in vague[:4]))


# ============================ G. claim coherence ============================
# The three places a reader looks for the headline findings must agree with each
# other and with the results section.
COHERENCE = [
    ("the held-out-class miss counts", "507 of 7,905", [SUMMARY, RQTABLE, CONCL, BODY]),
    ("the escalation spread", "about 200", [SUMMARY, RQTABLE, CONCL, BODY]),
    ("the count of models under the 3.3% line", "eight of eleven", [SUMMARY, CONCL, BODY]),
    ("the protocol effect", "3.23", [SUMMARY, CONCL, BODY]),
    ("the in-distribution accuracy span", "4.43", [SUMMARY, CONCL, BODY]),
]
for label, token, blobs in COHERENCE:
    missing = [i for i, b in enumerate(blobs) if token.lower() not in b.lower()]
    N[0] += 1
    if len(missing) == len(blobs):
        bad("G", f"{label}: {token!r} appears in none of Summary, RQ table, Conclusion, Results")
    elif missing:
        warn("G", f"{label}: {token!r} missing from section index {missing} of Summary/RQ/Conclusion/Results")

# scope statements that must survive in all three places
for phrase, where in [("not hydrocarbons", "Summary"),
                      ("constructed", "Summary"),
                      ("class exclusion", "Summary")]:
    N[0] += 1
    if phrase not in SUMMARY:
        bad("G", f"the Summary no longer states the scope limit {phrase!r}")

rule("G", "the Conclusion carries a scope statement",
     "not a field validation or safety qualification" in CONCL)
rule("G", "the Conclusion does not claim deployment readiness",
     not any(x in CONCL for x in ("ready for deployment", "fit for service", "qualifies")))


# ============================ report ============================
print(f"{N[0]} checks run, {len(FAIL)} failure(s), {len(WARN)} warning(s)\n")
for f in FAIL:
    print("  FAIL", f)
if WARN:
    print()
for w in WARN:
    print("  warn", w)
sys.exit(1 if FAIL else 0)
