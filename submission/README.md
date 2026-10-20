# Reviewed JLPPI submission files

The reviewed submission documents accompanying `v18.1-jlppi-reviewed-2026-10-20` are identified by
SHA-256 in `SHA256SUMS`. The Word/PDF binaries are distributed beside the Git
bundle and as submission documents, rather than tracked in the source repository.
Use those reviewed copies for the submission. `make docx` builds Word documents
from the aligned Markdown sources, with pagination dependent on fonts and renderer.

The main manuscript has 48 pages, the supplement 66, and the cover letter and
highlights one page each. All 33 numerical tables match the source tables, all
22 embedded figures match the source PNGs, and all 11 native manuscript equations
are preserved. The workflow figure has a portable font fallback and a larger
protocol box. Experimental CSV/JSON contents and the existing `v18.1-final` tag
are unchanged.

Verification on 20 October 2026: 1,068 numerical/style/figure checks passed;
670 consistency and regression checks passed; release-file audit reported zero
problems. Reviewed Word files were exported using Microsoft Word and their PDF
page images inspected. The packaged LibreOffice renderer was unavailable.
The complete PyTorch-dependent runtime suite, retraining and `make all` were not
run in this editorial environment. The workflow dependencies were checked and
the regression audit checks that the new analyses are reachable from `all`.

Before journal submission, the authors must complete the requested Fides postal
address placeholder, confirm authorship and exclusive-submission declarations,
and supply the archival DOI when the deposit is made. The deposit is still pending.
