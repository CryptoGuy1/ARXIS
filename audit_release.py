"""Does the release actually contain everything the manuscript claims?

Reviewer A3. Data and Code Availability says every artifact needed to reproduce
the paper is released together. That is a strong claim and it is checkable: this
script reads the file names out of the manuscript's own Appendix A and Data and
Code Availability sections and confirms each one exists on disk, then checks the
reverse direction, that no result file the drivers produce has been left out of
the manuscript's listing.

Run it before tagging a release. A missing artifact is a submission blocker,
because a reviewer who clones the repository and finds a gap can say the paper
claims reproducibility it does not deliver.

    python3 audit_release.py
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(ROOT, "ARXIS_Manuscript_v15_SPE.md")
SI = os.path.join(ROOT, "ARXIS_Supporting_Information.md")
RESULTS = os.path.join(ROOT, "retrain", "results_v2")

# Files the manuscript names in backticks. Paths are relative to the repository
# root; a few are expected only in the public release and are listed separately.
RELEASE_ONLY = {"models/yolov8n_cls_thermal.pt"}


def named_in(path):
    """Every `backticked` file name in the document."""
    txt = open(path, encoding="utf-8").read()
    out = set()
    for m in re.finditer(r"`([^`]+)`", txt):
        tok = m.group(1).strip()
        if re.search(r"\.(py|csv|json|lock|md|pt|pth)$", tok) or tok in ("Makefile",):
            out.add(tok)
        elif tok.endswith("/"):
            out.add(tok)
    return out


_OBJ = re.compile(r"\b(Fig|Table)s?\.?\s+(S?\d+(?:\s*(?:,|and|to)\s*S?\d+)*)")


def declared_objects(text):
    """Every figure and table a piece of prose names, with ranges expanded.

    Handles the forms the documents actually use: "Fig. 9", "Figs. 15 and 16",
    "Figs. 1, 2, 6 to 17", "Tables S4 and S5". The S prefix of a supplementary
    object is kept, so "Table S4" never collapses onto "Table 4".
    """
    out = set()
    for kind, body in _OBJ.findall(text):
        tokens = re.findall(r"S?\d+|to|and|,", body)
        prev, pending_range = None, False
        for tok in tokens:
            if tok == "to":
                pending_range = True
                continue
            if tok in (",", "and"):
                pending_range = False
                continue
            pre = "S" if tok.startswith("S") else ""
            num = int(tok.lstrip("S"))
            if pending_range and prev is not None and prev[0] == pre:
                for n in range(prev[1] + 1, num + 1):
                    out.add(f"{kind}. {pre}{n}" if kind == "Fig" else f"{kind} {pre}{n}")
            else:
                out.add(f"{kind}. {pre}{num}" if kind == "Fig" else f"{kind} {pre}{num}")
            prev, pending_range = (pre, num), False
    return out


def provenance_map(txt):
    """Table A-1 as {make target: {"Table 7", "Fig. 9", ...}}."""
    i = txt.find("| Reported object | Driver |")
    if i < 0:
        return {}
    end = txt.find("\n\nTable A-1", i)
    block = txt[i:end if end != -1 else len(txt)]
    out = {}
    for line in block.split("\n")[2:]:
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 5:
            continue
        objects = declared_objects(cells[0])
        for target in re.findall(r"`([a-z]+)`", cells[4]):
            out.setdefault(target, set()).update(objects)
    return out


def audit_target_comments(txt):
    """The Makefile and the README must claim what Table A-1 claims.

    The Makefile's comments and the README's target table were both written by
    hand against an older numbering and had drifted: `leakage` was documented as
    producing Fig. 7 when Table A-1 says Fig. 9, and five other rows were out by
    the same kind of shift. A reviewer auditing reproducibility reads those two
    files first, so a wrong mapping there is worse than a wrong mapping anywhere.
    """
    truth = provenance_map(txt)
    problems = []

    mk = os.path.join(ROOT, "Makefile")
    if os.path.exists(mk) and truth:
        lines = open(mk, encoding="utf-8").read().split("\n")
        for n, line in enumerate(lines):
            m = re.match(r"^([a-z]+):", line)
            if not m or m.group(1) not in truth:
                continue
            comment = []
            k = n - 1
            while k >= 0 and lines[k].startswith("#"):
                comment.insert(0, lines[k])
                k -= 1
            claimed = declared_objects(" ".join(comment))
            wrong = sorted(claimed - truth[m.group(1)])
            if wrong:
                problems.append(f"Makefile `{m.group(1)}` claims {wrong}, "
                                f"Table A-1 says {sorted(truth[m.group(1)])}")

    rd = os.path.join(ROOT, "README.md")
    if os.path.exists(rd) and truth:
        for line in open(rd, encoding="utf-8").read().split("\n"):
            if not line.startswith("| `make "):
                continue
            targets = re.findall(r"`make ([a-z]+)`", line)
            allowed = set()
            for t in targets:
                allowed |= truth.get(t, set())
            if not allowed:
                continue
            claimed = declared_objects(line.split("|")[2] if line.count("|") > 2 else "")
            wrong = sorted(claimed - allowed)
            if wrong:
                problems.append(f"README `{'/'.join(targets)}` claims {wrong}, "
                                f"Table A-1 says {sorted(allowed)}")
    return problems


def audit_manifest():
    """The manifest claims a hash for every figure and result file. Check it.

    Appendix A says a disagreement between RUN_MANIFEST.json and this table is a
    real disagreement, which only holds if something reads the manifest. Nothing
    did, and it drifted: twenty of twenty-one figure hashes were stale, it still
    listed a retired figure, and it did not know about the three architecture
    figures. `make manifest` rewrites it.
    """
    import hashlib
    path = os.path.join(ROOT, "RUN_MANIFEST.json")
    if not os.path.exists(path):
        return ["RUN_MANIFEST.json is missing; run `make manifest`"]
    import json
    man = json.load(open(path, encoding="utf-8"))
    problems = []
    for key, folder in (("figures", os.path.join(ROOT, "figures_v2")),
                        ("results", RESULTS)):
        for name, meta in (man.get(key) or {}).items():
            f = os.path.join(folder, name)
            if not os.path.exists(f):
                problems.append(f"{key}: {name} is in the manifest and not on disk")
                continue
            digest = hashlib.sha256(open(f, "rb").read()).hexdigest()[:16]
            if digest != meta.get("sha256_16"):
                problems.append(f"{key}: {name} has changed since the manifest was written")
        if os.path.isdir(folder):
            listed = set(man.get(key) or {})
            for f in sorted(os.listdir(folder)):
                if key == "results" and not f.endswith((".csv", ".json")):
                    continue
                if key == "figures" and not f.endswith(".png"):
                    continue
                if f not in listed:
                    problems.append(f"{key}: {f} is on disk and not in the manifest")
    return problems


def main():
    claimed = named_in(MD) | named_in(SI)
    missing, present = [], []
    for name in sorted(claimed):
        if name in RELEASE_ONLY:
            continue
        if name.endswith("/"):
            ok = os.path.isdir(os.path.join(ROOT, name.rstrip("/")))
        else:
            ok = os.path.exists(os.path.join(ROOT, name))
            if not ok:
                ok = os.path.exists(os.path.join(RESULTS, os.path.basename(name)))
        (present if ok else missing).append(name)

    produced = set()
    if os.path.isdir(RESULTS):
        produced = {f for f in os.listdir(RESULTS) if f.endswith((".csv", ".json"))}
    text = open(MD, encoding="utf-8").read() + open(SI, encoding="utf-8").read()
    unlisted = sorted(f for f in produced if f not in text)

    print(f"artifacts named in the manuscript: {len(claimed)}")
    print(f"  present on disk: {len(present)}")
    for n in present:
        print(f"    ok      {n}")
    if missing:
        print(f"  MISSING: {len(missing)}")
        for n in missing:
            print(f"    MISSING {n}")
    if RELEASE_ONLY:
        print("  expected in the public release only, not in this workspace:")
        for n in sorted(RELEASE_ONLY):
            print(f"    release {n}")
    if unlisted:
        print(f"\nresult files produced but NOT named in the manuscript: {len(unlisted)}")
        for n in unlisted:
            print(f"    unlisted {n}")

    stale = audit_manifest()
    if stale:
        print(f"\nrun manifest disagreeing with the files on disk: {len(stale)}")
        for s in stale:
            print(f"    stale   {s}")

    drift = audit_target_comments(open(MD, encoding="utf-8").read())
    if drift:
        print(f"\ntarget documentation disagreeing with Table A-1: {len(drift)}")
        for d in drift:
            print(f"    drift   {d}")

    problems = len(missing) + len(unlisted) + len(drift) + len(stale)
    print(f"\n{problems} problem(s)")
    if problems:
        print("Do not tag a release until each line above is resolved: either the file "
              "ships, or the manuscript stops claiming it.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
