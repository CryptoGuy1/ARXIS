#!/usr/bin/env python3
"""Build the submission documents from the Markdown sources.

The document build used to be a bare pandoc call typed by hand. Three defects
found in the final pre-submission page proof came from that: tables were set at
body size and broke long words across lines inside narrow columns, table rows
split across page boundaries, and nothing checked that every figure reached the
page at the width it was authored for.

This script is the build, and it post-processes the docx pandoc writes:

  * table text is set at TABLE_PT, below body size, as SPE sets it;
  * every table row carries w:cantSplit, so a row cannot break across a page;
  * every table header row carries w:tblHeader, so the header repeats;
  * image extents are asserted to be exactly FIG_IN wide.

Usage:  python3 build_docx.py [--outdir DIR]
"""

from __future__ import annotations

import argparse
import re
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "spe_reference.docx"

TABLE_PT = 9.0          # table text size, points: the size a table is set at
TABLE_PT_FLOOR = 7.0    # unless a column is too narrow for it, down to this
TABLE_PT_STEP = 0.5
CAPTION_PT = 11.0      # captions, one point under the body
FIG_IN = 6.50           # every figure is placed at exactly this width
TEXT_IN = 6.50          # text width of a Letter page with 1 in margins
CELL_PAD_IN = 0.08      # Word's default cell margin, each side
EMU_PER_IN = 914400

# The fonts the reference document asks for, and the metric-compatible faces a
# renderer substitutes when they are absent. Measuring against the substitute is
# conservative: Liberation Mono is wider per character than Consolas.
SERIF_FONT = "/usr/share/fonts/truetype/crosextra/Caladea-Regular.ttf"
SERIF_BOLD = "/usr/share/fonts/truetype/crosextra/Caladea-Bold.ttf"
MONO_FONT = "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"

DOCUMENTS = [
    ("ARXIS_Manuscript_v15_SPE.md", "ARXIS_Manuscript_v15_SPE.docx"),
    ("ARXIS_Supporting_Information.md", "ARXIS_Supporting_Information.docx"),
    ("CHANGE_REVIEW_v15_presubmission.md", "ARXIS_Change_Review_v15_presubmission.docx"),
]

TBL = re.compile(r"<w:tbl>.*?</w:tbl>", re.S)
RUN = re.compile(r"<w:r>(?:(?!</w:r>).)*?</w:r>", re.S)
RPR = re.compile(r"<w:rPr>(.*?)</w:rPr>", re.S)
SZ = re.compile(r"<w:sz(?:Cs)? w:val=\"\d+\"\s*/>")
TR = re.compile(r"<w:tr>|<w:tr [^>]*>")
ROW = re.compile(r"<w:tr(?: [^>]*)?>.*?</w:tr>", re.S)
CELL = re.compile(r"<w:tc>.*?</w:tc>", re.S)
TEXT = re.compile(r"<w:t(?: [^>]*)?>(.*?)</w:t>", re.S)
GRIDCOL = re.compile(r'<w:gridCol w:w="(\d+)"')

_FONT_CACHE: dict[tuple[str, int], object] = {}


def _font(path: str, pt: float):
    """A PIL font at `pt`, measured at 4x and scaled down for sub-point sizes."""
    from PIL import ImageFont
    key = (path, int(round(pt * 4)))
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(path, int(round(pt * 4)))
    return _FONT_CACHE[key]


def _width_in(text: str, path: str, pt: float) -> float:
    """Width of `text` in inches, at `pt`, in the face at `path`."""
    f = _font(path, pt)
    return f.getlength(text) / 4.0 / 72.0


def _unbreakable(text: str) -> list[str]:
    """The pieces a renderer cannot split: whitespace and a hyphen may break."""
    pieces = []
    for tok in text.split():
        parts = tok.split("-")
        for i, part in enumerate(parts):
            pieces.append(part + ("-" if i < len(parts) - 1 else ""))
    return [p for p in pieces if p]


def _column_widths(table_xml: str) -> list[float]:
    """Column widths in inches, from the grid, normalized to the text width."""
    grid = [int(w) for w in GRIDCOL.findall(table_xml)]
    if not grid:
        return []
    total = sum(grid) or 1
    return [TEXT_IN * w / total for w in grid]


def _widest_by_column(table_xml: str, ncols: int) -> list[list[tuple[str, bool]]]:
    """Per column, the unbreakable pieces and whether each is set in mono."""
    out: list[list[tuple[str, bool]]] = [[] for _ in range(ncols)]
    for row in ROW.findall(table_xml):
        for j, cell in enumerate(CELL.findall(row)):
            if j >= ncols:
                continue
            for run in RUN.findall(cell):
                mono = "VerbatimChar" in run
                body = "".join(TEXT.findall(run))
                body = (body.replace("&amp;", "&").replace("&lt;", "<")
                            .replace("&gt;", ">").replace("&quot;", '"'))
                for piece in _unbreakable(body):
                    out[j].append((piece, mono))
    return out


def _demand(table_xml: str, ncols: int, pt: float) -> tuple[list[float], list[float]]:
    """Per column: the width its longest unbreakable piece needs, and its volume.

    Volume is the mean width of a cell's whole text and is only used to share
    out slack, so a column of long sentences grows before a column of numbers.
    """
    need = [0.0] * ncols
    volume = [0.0] * ncols
    rows = 0
    for row in ROW.findall(table_xml):
        rows += 1
        for j, cell in enumerate(CELL.findall(row)):
            if j >= ncols:
                continue
            for run in RUN.findall(cell):
                font = MONO_FONT if "VerbatimChar" in run else SERIF_FONT
                body = "".join(TEXT.findall(run))
                body = (body.replace("&amp;", "&").replace("&lt;", "<")
                            .replace("&gt;", ">").replace("&quot;", '"'))
                volume[j] += _width_in(body, font, pt)
                for piece in _unbreakable(body):
                    need[j] = max(need[j], _width_in(piece, font, pt))
    pad = 2 * CELL_PAD_IN
    need = [n + pad for n in need]
    volume = [v / max(rows, 1) + pad for v in volume]
    return need, volume


def fit_table(table_xml: str) -> tuple[float, list[float]]:
    """Size and column widths: the largest size at which no word has to break.

    Pandoc gives every column an equal share of the text width, which is why a
    long model name or a driver path was being broken mid-word in a narrow
    column while a column of four-digit numbers sat half empty. Each column is
    given the width its widest unbreakable piece needs, and what is left over is
    shared out in proportion to how much text each column actually carries.
    """
    ncols = len(GRIDCOL.findall(table_xml))
    if not ncols:
        return TABLE_PT, []
    pt = TABLE_PT
    while True:
        need, volume = _demand(table_xml, ncols, pt)
        if sum(need) <= TEXT_IN or pt <= TABLE_PT_FLOOR:
            break
        pt -= TABLE_PT_STEP
    slack = TEXT_IN - sum(need)
    if slack <= 0:
        total = sum(need) or 1.0
        return pt, [TEXT_IN * n / total for n in need]
    share = sum(volume) or 1.0
    return pt, [n + slack * v / share for n, v in zip(need, volume)]


def _apply_widths(table_xml: str, widths: list[float]) -> str:
    """Write the widths into the grid and into every cell, for both renderers."""
    if not widths:
        return table_xml
    twips = [max(int(round(w * 1440)), 1) for w in widths]
    total = sum(twips)
    grid = "".join(f'<w:gridCol w:w="{t}" />' for t in twips)
    table_xml = re.sub(r"<w:tblGrid>.*?</w:tblGrid>",
                       f"<w:tblGrid>{grid}</w:tblGrid>", table_xml, count=1, flags=re.S)
    pct = [int(round(5000 * t / total)) for t in twips]

    def fix_row(m: re.Match) -> str:
        row = m.group(0)
        j = [0]

        def fix_cell(c: re.Match) -> str:
            cell = c.group(0)
            i = j[0]
            j[0] += 1
            if i >= len(pct):
                return cell
            w = f'<w:tcW w:type="pct" w:w="{pct[i]}" />'
            if "<w:tcPr />" in cell:
                return cell.replace("<w:tcPr />", f"<w:tcPr>{w}</w:tcPr>", 1)
            if "<w:tcPr>" in cell:
                return cell.replace("<w:tcPr>", f"<w:tcPr>{w}", 1)
            return cell.replace("<w:tc>", f"<w:tc><w:tcPr>{w}</w:tcPr>", 1)

        return CELL.sub(fix_cell, row)

    return ROW.sub(fix_row, table_xml)


def _size_tags(pt: float) -> str:
    half = int(round(pt * 2))
    return f'<w:sz w:val="{half}" /><w:szCs w:val="{half}" />'


def _shrink_runs(table_xml: str, pt: float) -> str:
    """Set every run inside one table to `pt`, keeping bold/italic as they are."""
    tags = _size_tags(pt)

    def fix(m: re.Match) -> str:
        run = m.group(0)
        if "<w:drawing>" in run:
            return run
        rpr = RPR.search(run)
        if rpr:
            inner = SZ.sub("", rpr.group(1))
            return run.replace(rpr.group(0), f"<w:rPr>{inner}{tags}</w:rPr>")
        return run.replace("<w:r>", f"<w:r><w:rPr>{tags}</w:rPr>", 1)

    return RUN.sub(fix, table_xml)


PARA = re.compile(r"<w:p>(?:(?!</w:p>).)*?</w:p>", re.S)
CAPTION = re.compile(r"^(?:Fig\.|Table)\s(?:\d+|[A-Z]-\d+|S\d+)—")


def _style_captions(xml: str) -> int:
    """Set captions apart without bold.

    The captions used to be bold, which is what made them read as captions. They
    are plain now, so they are set one point smaller than the body and centred
    when they fit on a single line, which is how the model paper sets them. A
    caption too long to centre stays left-aligned, because a centred five-line
    caption is worse than no distinction at all.
    """
    tags = _size_tags(CAPTION_PT)
    count = 0

    def fix(m: re.Match) -> str:
        nonlocal count
        para = m.group(0)
        if "<w:drawing>" in para or "<w:tc>" in para:
            return para
        body = "".join(TEXT.findall(para))
        body = (body.replace("&amp;", "&").replace("&lt;", "<")
                    .replace("&gt;", ">").replace("&quot;", '"')).strip()
        if not CAPTION.match(body):
            return para
        count += 1
        centred = _width_in(body, SERIF_FONT, CAPTION_PT) <= TEXT_IN - 2 * CELL_PAD_IN
        para = _shrink_runs(para, CAPTION_PT)
        if centred:
            jc = '<w:jc w:val="center" />'
            if "<w:pPr>" in para:
                para = para.replace("<w:pPr>", f"<w:pPr>{jc}", 1)
            else:
                para = para.replace("<w:p>", f"<w:p><w:pPr>{jc}</w:pPr>", 1)
        return para

    out = PARA.sub(fix, xml)
    return count, out


def _keep_figures_with_captions(xml: str) -> int:
    """A figure and its caption must not be separated by a page break.

    Fig. 2's image sat at the foot of one page and its caption opened the next.
    The image paragraph gets w:keepNext, which binds it to the caption below it.
    """
    count = 0

    def fix(m: re.Match) -> str:
        nonlocal count
        para = m.group(0)
        if "<w:drawing>" not in para or "<w:keepNext />" in para:
            return para
        count += 1
        if "<w:pPr>" in para:
            return para.replace("<w:pPr>", "<w:pPr><w:keepNext />", 1)
        return para.replace("<w:p>", "<w:p><w:pPr><w:keepNext /></w:pPr>", 1)

    out = PARA.sub(fix, xml)
    return count, out


def _row_properties(table_xml: str) -> str:
    """Give every row w:cantSplit, and the first row a repeating header."""
    out = []
    pos = 0
    for i, m in enumerate(TR.finditer(table_xml)):
        out.append(table_xml[pos:m.end()])
        props = "<w:cantSplit />" + ("<w:tblHeader />" if i == 0 else "")
        rest = table_xml[m.end():]
        existing = re.match(r"\s*<w:trPr>(.*?)</w:trPr>", rest, re.S)
        if existing:
            inner = existing.group(1)
            inner = inner.replace("<w:cantSplit />", "").replace("<w:tblHeader />", "")
            out.append(f"<w:trPr>{props}{inner}</w:trPr>")
            pos = m.end() + existing.end()
        else:
            out.append(f"<w:trPr>{props}</w:trPr>")
            pos = m.end()
    out.append(table_xml[pos:])
    return "".join(out)


def postprocess(docx: Path) -> dict:
    """Rewrite word/document.xml in place. Returns a small report."""
    with zipfile.ZipFile(docx) as z:
        names = z.namelist()
        blobs = {n: z.read(n) for n in names}

    xml = blobs["word/document.xml"].decode("utf-8")

    sizes: list[float] = []

    def fix_table(m: re.Match) -> str:
        pt, widths = fit_table(m.group(0))
        sizes.append(pt)
        return _row_properties(_apply_widths(_shrink_runs(m.group(0), pt), widths))

    xml = TBL.sub(fix_table, xml)
    kept, xml = _keep_figures_with_captions(xml)
    captions, xml = _style_captions(xml)

    extents = [int(cx) for cx, _ in re.findall(r'<wp:extent cx="(\d+)" cy="(\d+)"', xml)]
    wrong = [cx / EMU_PER_IN for cx in extents if abs(cx / EMU_PER_IN - FIG_IN) > 1e-4]
    if wrong:
        raise SystemExit(f"{docx.name}: figure widths not {FIG_IN} in: {wrong}")

    blobs["word/document.xml"] = xml.encode("utf-8")

    tmp = docx.with_suffix(".tmp.docx")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, blobs[n])
    tmp.replace(docx)
    return {"tables": len(sizes), "figures": len(extents), "sizes": sizes,
            "kept": kept, "captions": captions}


def build(src: Path, dst: Path) -> dict:
    cmd = [
        "pandoc", str(src), "-o", str(dst),
        f"--reference-doc={REFERENCE}",
        f"--resource-path={ROOT}",
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    return postprocess(dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "build"),
        help="where to write the built documents (default: ./build)")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not REFERENCE.exists():
        print(f"missing reference document: {REFERENCE}", file=sys.stderr)
        return 1

    for src_name, dst_name in DOCUMENTS:
        src = ROOT / src_name
        if not src.exists():
            print(f"skipped (absent): {src_name}")
            continue
        dst = outdir / dst_name
        report = build(src, dst)
        sizes = report["sizes"]
        stepped = [f"T{i + 1}@{p:g}pt" for i, p in enumerate(sizes) if p < TABLE_PT]
        note = f"; stepped down: {', '.join(stepped)}" if stepped else ""
        print(f"{dst_name}: {report['figures']} figures, {report['tables']} tables "
              f"at {TABLE_PT:g} pt, {report['captions']} captions at {CAPTION_PT:g} pt{note}")
        if src_name.startswith("ARXIS_"):
            # Ship the source beside the built file, under the built file's name,
            # so the two are obviously the same revision.
            shutil.copy2(src, outdir / (dst.stem + ".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
