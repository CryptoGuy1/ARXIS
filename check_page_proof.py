"""Read the built document the way a production editor would, and fail on what they would send back.

`check_figure_text.py` looks inside each figure. Nothing looked at the page the
figure lands on. The final pre-submission proof turned up three defects that only
appear once the Markdown has become a paginated document:

  * `$V$` in a caption became an OMML run that several renderers drop silently,
    so the caption read " = value stream" with the symbol missing;
  * table columns were all the same width, so `Conservative` and
    `retrain/run_overlap_isolation.py` were broken mid-word in narrow columns;
  * table rows broke across page boundaries.

This script builds the documents, renders them, and checks the result. Every word
the sources contain must survive to the page unbroken, every figure must be
placed at exactly the authored width, and every figure and table must be
captioned exactly once, in order, with every cross-reference resolving.

    python3 check_page_proof.py

Requires pandoc, LibreOffice and poppler-utils. If LibreOffice is absent the
render-dependent checks are skipped and reported as skipped, never as passing.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(ROOT, "ARXIS_Manuscript_v15_SPE.md")
SI = os.path.join(ROOT, "ARXIS_Supporting_Information.md")

FIG_IN = 6.50
EMU_PER_IN = 914400
N_FIGS = 16
N_TABLES = 16

# Tokens the PDF text layer joins from a base and a subscript, so they are not in
# the source as one word. They are not broken words.
SUBSCRIPTED = {
    "nopp", "nrun", "ntest", "ntrain", "resc", "rfa", "rmiss", "runder",
    "cmiss", "cfalse", "spost", "lord",
}

failures: list[str] = []
skipped: list[str] = []
checks = 0


def rule(name: str, ok: bool, detail: str = "") -> None:
    global checks
    checks += 1
    if ok:
        print(f"  ok   {name}")
    else:
        failures.append(name)
        print(f"  FAIL {name}" + (f": {detail}" if detail else ""))


def skip(name: str, why: str) -> None:
    skipped.append(name)
    print(f"  skip {name}: {why}")


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


def source_text() -> str:
    out = []
    for p in (MD, SI):
        if os.path.exists(p):
            out.append(open(p, encoding="utf-8").read())
    return "\n".join(out)


def check_sources() -> None:
    print("sources")
    for path in (MD, SI):
        if not os.path.exists(path):
            continue
        txt = open(path, encoding="utf-8").read()
        name = os.path.basename(path)
        # Inline TeX becomes an OMML run. Word renders it; LibreOffice, Google
        # Docs and several PDF converters drop it, taking the symbol with it.
        math = re.findall(r"(?<!\\)\$[^$\n]+\$", txt)
        rule(f"{name} contains no inline TeX math", not math, f"{math[:3]}")
        # An unclosed ^ is not superscript, it is a caret. The author line read
        # "Nweke^1,2,*" on the page, and three exponents read "R^(20x7)".
        stray = re.findall(r"\^[^\s^]{0,40}", re.sub(r"\^[^\s^]+\^", "", txt))
        rule(f"{name}: every superscript marker is closed", not stray,
             f"{sorted(set(stray))[:4]}")
        # The author asked for no bold anywhere in the writeup. Captions are set
        # apart by size and centring instead, in build_docx.py.
        bold = re.findall(r"\*\*[^*]+\*\*", txt)
        rule(f"{name}: nothing is set in bold", not bold, f"{bold[:3]}")
        # Two affiliations in one Markdown paragraph run together on the page.
        aff = re.search(r"^\^1\^ .*$", txt, re.M)
        if aff:
            rule(f"{name}: the affiliation lines break", aff.group(0).endswith("\\"),
                 "no hard line break after the first affiliation")


def check_docx(docx: str) -> str | None:
    """Structural checks on the built file. Returns document.xml, or None."""
    name = os.path.basename(docx)
    if not os.path.exists(docx):
        rule(f"{name} was built", False, "missing")
        return None
    rule(f"{name} was built", True)
    with zipfile.ZipFile(docx) as z:
        xml = z.read("word/document.xml").decode("utf-8")

    widths = [int(cx) / EMU_PER_IN
              for cx, _ in re.findall(r'<wp:extent cx="(\d+)" cy="(\d+)"', xml)]
    off = [round(w, 4) for w in widths if abs(w - FIG_IN) > 1e-4]
    rule(f"{name}: every figure is placed at exactly {FIG_IN} in", not off, f"{off}")

    paras = re.findall(r"<w:p>(?:(?!</w:p>).)*?</w:p>", xml, re.S)
    loose = [p for p in paras if "<w:drawing>" in p and "<w:keepNext />" not in p]
    rule(f"{name}: every figure is bound to the caption below it", not loose,
         f"{len(loose)} figure(s) without w:keepNext")

    # These two hold for what is submitted. The change review is an internal
    # note and is not held to the manuscript's typographic rules.
    if "Change_Review" not in name:
        body = re.sub(r"<w:tbl>.*?</w:tbl>", "", xml, flags=re.S)
        rule(f"{name}: no run is set in bold",
             "<w:b />" not in body and "<w:b/>" not in body,
             f"{body.count('<w:b />')} bold run(s)")

        caps = [p for p in re.findall(r"<w:p>(?:(?!</w:p>).)*?</w:p>", xml, re.S)
                if re.match(r"^(?:Fig\.|Table)\s(?:\d+|[A-Z]-\d+|S\d+)—",
                            "".join(re.findall(r"<w:t(?: [^>]*)?>(.*?)</w:t>", p, re.S)).strip())]
        unsized = [p for p in caps if 'w:val="22"' not in p]
        rule(f"{name}: every caption is set apart from the body", caps and not unsized,
             f"{len(caps)} caption(s), {len(unsized)} unstyled")

    rows = re.findall(r"<w:tr(?: [^>]*)?>", xml)
    splits = len(rows) - xml.count("<w:cantSplit />")
    rule(f"{name}: no table row may break across a page", splits <= 0,
         f"{splits} row(s) without w:cantSplit")

    tables = re.findall(r"<w:tbl>.*?</w:tbl>", xml, re.S)
    uniform = [i + 1 for i, t in enumerate(tables)
               if len(set(re.findall(r'<w:gridCol w:w="(\d+)"', t))) == 1
               and len(re.findall(r'<w:gridCol', t)) > 2]
    rule(f"{name}: column widths are fitted, not uniform", not uniform,
         f"tables {uniform} still have equal columns")
    return xml


def check_pdf(pdf: str, src: str) -> None:
    name = os.path.basename(pdf)
    txt = subprocess.run(["pdftotext", "-layout", pdf, "-"],
                         capture_output=True, text=True, check=True).stdout

    vocab = {w.lower() for w in re.findall(r"[A-Za-z]{2,}", src)}
    broken: dict[str, int] = {}
    for page_no, page in enumerate(txt.split("\f"), 1):
        for tok in re.findall(r"[A-Za-z]{3,}", page):
            low = tok.lower()
            if low not in vocab and low not in SUBSCRIPTED:
                broken.setdefault(tok, page_no)
    rule(f"{name}: no word is broken across a line",
         not broken, f"{sorted(broken)[:8]}")

    figs = [int(n) for n in re.findall(r"Fig\.\s*(\d+)—", txt)]
    tabs = [int(n) for n in re.findall(r"Table\s*(\d+)—", txt)]
    if "Manuscript" in name:
        rule(f"{name}: {N_FIGS} figure captions, once each, in order",
             figs == list(range(1, N_FIGS + 1)), f"{figs}")
        rule(f"{name}: {N_TABLES} table captions, once each, in order",
             tabs == list(range(1, N_TABLES + 1)), f"{tabs}")

        bad_f = sorted({int(n) for n in re.findall(r"Fig\.\s*(\d+)(?!—)", txt)}
                       - set(range(1, N_FIGS + 1)))
        bad_t = sorted({int(n) for n in re.findall(r"Table\s*(\d+)(?!—)", txt)}
                       - set(range(1, N_TABLES + 1)))
        rule(f"{name}: every figure cross-reference resolves", not bad_f, f"{bad_f}")
        rule(f"{name}: every table cross-reference resolves", not bad_t, f"{bad_t}")

        try:
            import pdfplumber
        except ImportError:
            skip(f"{name}: each caption sits on its figure's page", "no pdfplumber")
        else:
            with pdfplumber.open(pdf) as doc:
                images, captions = [], {}
                for i, page in enumerate(doc.pages, 1):
                    images += [i] * len(page.images)
                    for m in re.finditer(r"Fig\.\s*(\d+)—", page.extract_text() or ""):
                        captions[int(m.group(1))] = i
            split = [n for (n, cp), ip in zip(sorted(captions.items()), images)
                     if cp != ip]
            rule(f"{name}: each caption sits on its figure's page", not split,
                 f"Fig. {split} separated from its caption")

    stray = sorted(set(re.findall(r"[⟨⟩]|\bTODO\b|\bTBD\b", txt)))
    if stray:
        print(f"  note {name}: {len(re.findall(chr(10216), txt))} placeholder(s) "
              f"still awaiting a value")


def main() -> int:
    print("page proof\n")
    check_sources()

    if not have("pandoc"):
        skip("the documents build", "pandoc not installed")
        return report()

    out = tempfile.mkdtemp(prefix="pageproof-")
    try:
        print("\nbuild")
        subprocess.run([sys.executable, os.path.join(ROOT, "build_docx.py"),
                        "--outdir", out], check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL)
        built = []
        for f in sorted(os.listdir(out)):
            if f.endswith(".docx"):
                if check_docx(os.path.join(out, f)) is not None:
                    built.append(f)

        if not have("soffice") or not have("pdftotext"):
            skip("the rendered page", "LibreOffice or poppler-utils not installed")
            return report()

        print("\nrendered page")
        src = source_text()
        for f in built:
            if "Change_Review" in f:
                continue
            subprocess.run(["soffice", "--headless", "--convert-to", "pdf",
                            "--outdir", out, os.path.join(out, f)],
                           check=True, capture_output=True, timeout=900)
            pdf = os.path.join(out, f[:-5] + ".pdf")
            if os.path.exists(pdf):
                check_pdf(pdf, src)
            else:
                rule(f"{f} renders", False, "no PDF produced")
    finally:
        shutil.rmtree(out, ignore_errors=True)
    return report()


def report() -> int:
    print(f"\n{checks} checks run, {len(failures)} failure(s), "
          f"{len(skipped)} skipped")
    for f in failures:
        print(f"  FAIL {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
