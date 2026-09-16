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
MD = os.path.join(ROOT, "ARXIS_Manuscript_v13_SPE.md")
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

    problems = len(missing) + len(unlisted)
    print(f"\n{problems} problem(s)")
    if problems:
        print("Do not tag a release until each line above is resolved: either the file "
              "ships, or the manuscript stops claiming it.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
