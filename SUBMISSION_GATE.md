# Submission gate: one string outstanding

`verify_v13.py` reports **770 checks, 1 failure**, and the failure is deliberate:

```
FAIL [style] the Ultralytics version is supplied, not deferred:
     placeholder still present: insert the version string from the training environment
```

`audit_release.py` reports **0 problems**. Everything else passes.

---

## What to do

On the machine where the thermal classifier was trained:

```bash
python -c "import ultralytics; print(ultralytics.__version__)"
# or
pip show ultralytics | grep Version
```

Then, in `ARXIS_Manuscript_v13_SPE.md`, replace the placeholder:

```
version ⟨ULTRALYTICS_VERSION⟩
```

with the string that command prints, for example `version 8.0.196`.

Then:

```bash
make verify        # must report 770 checks, 0 failures
python3 audit_release.py   # must report 0 problems
```

Do not submit before both read clean.

---

## Why this is a hard gate rather than a note

The previous version contained a sentence telling a referee that a
reproducibility item was unfinished, inside the appendix that claims every value
is machine-checked. Replacing it with a softer note would reproduce the failure
mode that let the 117/128, 0.0098/0.3405 and "factor of 43" contradictions
survive three rounds of review: a warning nobody acts on.

The placeholder fails the verification run instead.

---

## If the version genuinely cannot be recovered

Do not guess. Replace the whole clause with:

> The thermal classifier was trained separately with Ultralytics YOLOv8n-cls.
> The exact package version was not recorded at training time and the classifier
> has not been retrained, so that one element of the environment cannot be
> reproduced exactly.

and change the gate in `verify_v13.py` to require that wording instead. That
costs a reproducibility point and costs nothing in credibility. A plausible
version number invented for a reproducibility appendix costs the reverse.

---

## The rest of the final sequence, already done

| Step | Status |
|---|---|
| 1. Manuscript frozen as `ARXIS_Manuscript_v13_SPE.md` | done |
| 2. Tables regenerated from the result files | done |
| 3. Figures regenerated (21 files, 300 dpi) | done |
| 4. `verify_v13.py` run against that exact manuscript | done, 770 checks |
| 5. Zero discrepancies | **pending this one string** |
| 6. Release audit: every artifact the manuscript names exists | done, 0 problems |
| 7. Equation rendering checked in the rendered PDF, not the text extract | done |
| 8. Ratio audit: every quoted ratio recomputed exactly and from displayed values | done |

Step 6 found and fixed one real gap: `v3_metric_cost.json` was produced but never
named in the manuscript. It is the measurement behind the claim that the metric
set is inexpensive to compute, which had been asserted without citing it. The
supporting information now quotes the measured figures and names the file.
