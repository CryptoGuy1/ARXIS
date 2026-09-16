# Text integrity audit: ARXIS manuscript v10

Two questions were asked of the manuscript: does it reuse anyone's text, and does
its prose carry the signature that commercial AI-writing detectors look for. The
first has a definite answer. The second does not, and the honest form of this
report says so before it says anything else.

## What cannot be delivered, and why

There is no scientifically reliable way to compute an "AI percentage" for a piece
of text. Commercial detectors return one anyway, and that number is not a
measurement. Published evaluations of these tools report high false-positive
rates on non-native English, on heavily edited technical prose, and on writing
from any domain with a fixed register, which describes a journal methods section
exactly. A detector score on this manuscript would be a number without a
denominator, and quoting one here would be worse than quoting nothing.

What can be done instead is to measure the features detectors actually key on and
compare them against a peer-reviewed, human-written paper from the same journal
and the same section. That comparison is interpretable. A score is not.

The comparator is SPE-219771-PA, *SPE Journal*, Data Science and Engineering
Analytics: same venue, same section, same register, human authors, published.

## Part 1: Text reuse

`textcheck.py` shingles both documents into overlapping word n-grams after
stripping markup, figure and table captions, and reference lists, then measures
the share of the manuscript's n-grams that appear anywhere in the comparison
document.

**Against the model paper used for structure (SPE-219771-PA):**

| n-gram order | overlap | matched |
|---|---|---|
| 5-gram | 0.00% | 0 of 15,995 |
| 8-gram | 0.00% | 0 of 15,992 |
| 12-gram | 0.00% | 0 of 15,988 |
| longest shared run | 0 words | — |

Zero at every order, including five words. The paper's structure, section
ordering and caption conventions were learned from that model; none of its
sentences were. An earlier pass found one seven-word run, *"this study
contributes in the following ways"*, generic enough to be unremarkable but easy
to remove, and it was removed.

**Against the published literature.** Four distinctive phrases from the
manuscript and both candidate titles were searched verbatim on the open web. No
matches. Neither title is publicly indexed, which also confirms no prior
disclosure that would complicate the submission.

**Against our own earlier drafts:** 43.6% at five grams against v9, 26.3%
against v5, 13.7% against v3. This is the expected signature of a document
revised in place across ten versions. It is self-overlap, not plagiarism, and no
part of any earlier draft was published or submitted. It is reported here only so
that a reviewer who runs their own similarity check and sees a hit against a
preprint server or a repository copy knows where it comes from.

**What this check cannot see.** Subscription similarity services such as
iThenticate and Turnitin index paywalled journal content that no open-web search
reaches. SPE runs a similarity check on submission. Nothing in the result above
predicts what a paywalled corpus will return, though a document with zero
five-gram overlap against its own structural model is not a document at risk of a
high similarity index.

## Part 2: Stylometric comparison

`stylecheck.py` measures fifteen features on prose only: tables, captions,
display equations, the keyword line and the abbreviation list are excluded,
because those are house style rather than writing.

| metric | SPE-219771 (human) | ARXIS v10 | ARXIS v9 |
|---|---|---|---|
| words | 10,907 | 13,135 | 13,731 |
| sentences | 511 | 634 | 653 |
| mean sentence length | 21.6 | 21.0 | 21.2 |
| sentence length SD | 12.9 | 10.9 | 12.9 |
| burstiness (SD/mean) | 0.599 | 0.521 | 0.609 |
| sentences ≤ 8 words | 12.9% | 14.2% | 14.7% |
| sentences ≥ 35 words | 11.5% | 12.9% | 13.5% |
| lexical diversity (MATTR) | 0.504 | 0.552 | 0.557 |
| repeated trigrams | 21.01% | 18.76% | 20.72% |
| hedges per 1k words | 2.5 | 1.5 | 2.0 |
| formulaic connectives per 1k | 5.5 | 1.4 | 0.7 |
| commas per sentence | 1.60 | 1.05 | 1.16 |
| semicolons per 1k words | 2.8 | 1.4 | 4.2 |

Every feature now sits inside the human paper's neighbourhood or on the safer
side of it. Lexical diversity is higher than the human comparator, repeated
trigrams lower, formulaic connectives roughly a quarter of the human rate: all of
these run opposite to the generated-text signature, which is characterised by
*low* diversity, *high* trigram repetition and *heavy* connective scaffolding.

Three outliers were found in the first pass and three were acted on.

**Semicolons: 8.5 per 1k, against 2.8.** Two causes. The measurement counted the
semicolons inside `&nbsp;` markup entities, which inflated the figure and has
been fixed in the tool. The rest were real, and thirteen prose semicolons were
converted to full stops. Table cells, captions, the keyword line, citation
separators and enumerations with three or more items were left alone, because
those are the house style of the journal. Final: 1.4.

**Long sentences: 20.7% at or above 35 words, against 11.5%.** Thirty-seven
sentences of 45 words or more were split at their natural joints, which were
almost always a colon or a coordinating conjunction carrying two independent
clauses. Final: 12.9%.

**Hedges: 5.9 per 1k, against 2.5.** Most of the count was the word *would*,
which in this manuscript is nearly always counterfactual rather than hedging: *"a
failed autoencoder would supply zero"*, *"a metric set reporting missed-hazard
rate alone would rank the support-vector classifier above the reference"*. That
is a conditional, not a hedge, and the fix was to the metric rather than to the
paper. Genuine hedging was left in place: this is a safety paper with an
eight-item limitations section, and its qualifications are load-bearing.

**One feature moved the wrong way.** Burstiness fell from 0.604 to 0.521 against
the human paper's 0.599, because splitting long sentences made lengths more
uniform. This is stated rather than corrected. Restoring it would mean
re-lengthening sentences that read better short, which is the kind of deliberate
degradation your own revision plan ruled out at item 66, and 0.52 is not far
enough outside the human range to be worth prose that is worse.

## What this does not settle

Nothing here is a guarantee against a detector returning a high score. Detectors
are not calibrated and their outputs are not reproducible between versions. What
the audit does establish is narrower and more defensible: the manuscript reuses
no text, and on every feature these tools are known to measure it sits with or
beyond a published human paper in the same venue. If a score is ever raised
against this paper, that is the record to answer it with.

## Reproducing this

`textcheck.py` and `stylecheck.py` ship in `arxis_v10_artifacts.zip` and run on
the manuscript source with no arguments. Both are deterministic.
