# Self-analysis: what I broke, why, and what now stops it

Written because the criticism was fair. Across three revision rounds I corrected real defects and introduced new ones while doing it, and the pattern is consistent enough to name.

---

## The churn, itemized

**1. I added a statistical error, then removed it.**

Round two: the adversarial review said Table 14's episode counts were being read as probability estimates. I "fixed" that by computing one-sided binomial lower bounds, 0.39 for 7/10 and 0.74 for 10/10, and writing them into the paper as a caution.

Round three: the pre-submission plan pointed out that those bounds are pseudo-replication. Ten entries are five runs times two hazardous classes over **two** physical acquisitions. Repeating a partition with a new seed does not sample a new event, so there is no population to bound. I removed them.

Net effect on the manuscript: zero. Cost: one round, and a version of the paper went out with an invalid interval in it. The reviewer's complaint had been about over-reading the counts, and I answered it by adding a more sophisticated way to over-read them.

**2. I corrected a number in five places and missed the sixth.**

Round three moved the headline missed-hazard bound ratio from 40 to about 86. I updated the Summary, the contributions list, Experiment 3, the Conclusion and the model-application screen. I did not update the research-question table, which went on saying "the worst missed-hazard bound is 40 times the best" in a version I delivered and described as verified.

**3. The correlation correction left five stale numbers in prose.**

Round three corrected ρ from 0.202 to 0.2051 and regenerated every posterior. I updated Table 8's final column, Table S4 and Table S5. A sentence in Experiment 2 quotes five of those same posteriors inline, as "the recurrent network (0.782), conservative Q-learning (0.758), the multilayer perceptron (0.693), gradient boosting (0.589) and the random forest (0.542)". All five were superseded. A sixth stale copy sat in Experiment 3.

**4. A smaller one, caught by my own new test.**

Writing the metric unit tests this round, I asserted that the zero-count bound on 15 windows is 18.0966%. It is 18.1036%. The manuscript had it right; my test expectation was wrong. The test failed, which is what it was for.

---

## Why it kept happening

`verify_v13.py` had 847 checks and every one of them compared **a table cell to a result file**. That is a good defense against a number drifting away from the data that produced it. It is no defense at all against:

- a number in prose disagreeing with the table it quotes;
- one table disagreeing with another;
- a value that was correct in the previous revision surviving into this one;
- a claim in the Summary that no longer matches the Results.

Every miss above falls into one of those four. Items 2 and 3 are prose-versus-table. Item 1 is a claim that was internally consistent and externally wrong, which no consistency check would have caught, but which a stricter reading of the reviewer's actual complaint would have.

There is a second, softer cause. Each round I worked from a reviewer's list, and a list invites treating each item as a local edit. A number is not local. Changing 40 to 86 is a change to every sentence that ever quoted it, and I was not enumerating those sentences.

---

## What now stops it

**`self_audit.py`, 989 checks, none of which reads a result file.** It checks the document against itself, in the four categories that were unguarded:

| Family | What it catches | Would it have caught |
|---|---|---|
| A. superseded values | any value a prior revision replaced, anywhere in either document | items 2 and 3 |
| B. number traceability | a figure quoted in the Summary, the research-question table or the Conclusion that appears nowhere else | item 2 |
| C. controlled vocabulary | variant spellings, model names, one name per thing | terminology drift |
| D. cross-references | figures, tables, equations and supplementary objects cited but absent, or present but never cited | Fig. 18 and Table 15 |
| E. notation | nomenclature and body disagreeing in either direction | symbol drift |
| F. prose risk | sentences past 52 words, repeated openings, superlatives, defensive meta-commentary, overused vocabulary, em-dashes, bare-pronoun paragraph openings | the style brief |
| G. claim coherence | Summary, research-question table, Conclusion and Results telling different stories | items 2 and 3 |

Family A is the important one and it is deliberately blunt. It carries a list of eighteen strings that were once true and are now wrong, and it fails if any of them reappears anywhere: `factor of 40`, `7.49`, `26.71`, `ten of eleven`, `pass a miss-rate screen`, `between 27 and 29 points`, `Fig. 18`, `Table 15`, `exact upper bound`, the old `a*(·)` symbol, `consistent with a true rate near`, `assumes nothing about within-run correlation`, `lower bound on detection`, `bounded below`, `Under Distribution Shift`, the old title, ρ = 0.202, and any affirmative use of "effective sample size". The list only grows.

**`verify_v13.py` now checks prose, not only tables.** Every posterior named in a sentence is compared against the artifact, which is the check that item 3 needed and did not have.

**`test_safety_metrics.py`, 30 assertions.** The plan asked for this and I had skipped it. It tests the identity that missed-hazard, under-escalation and escalation rates sum to one over 200 random hazardous partitions; that the false-alarm denominator admits no Perfume window; the zero-count and nonzero branches of the bound at the exact values the paper quotes; boundary handling at *a* = 2 against *a* = 3; the burden conversion; and that the embargo is two-sided with more than one window of separation on every seed and class. Two of those guard defects that actually shipped in earlier versions.

**`make check` runs all four.** Current state:

```
verify_v13.py         847 checks, 1 failure   (the deliberate Ultralytics gate)
self_audit.py         989 checks, 0 failures
test_safety_metrics.py 30 assertions, 0 failures
audit_release.py        0 problems
```

---

## What this round actually found and fixed

Running the new audit against the manuscript as I had left it:

- the research-question table still said "40 times" (item 2 above);
- five posteriors in Experiment 2's prose and one in Experiment 3 were stale (item 3);
- "unseen hazard" appeared twice where the paper's own vocabulary is "held-out hazardous class";
- "perfectly" twice and "the most important" once, as superlatives standing in for numbers that were available, now replaced by the numbers;
- one grammar error, "a eleventh" for "an eleventh";
- a paragraph opening on a bare "It is" with no referent;
- fifteen sentences past 52 words, the longest 57, now split;
- the Conclusion omitted the escalation ratio that the Summary and the research-question table both quote.

None of these is fatal on its own. Together they are what makes a referee stop trusting the arithmetic, which is the thing this paper can least afford, since its whole argument is that people are insufficiently careful with evaluation numbers.

---

## What I am not claiming

The self-audit checks consistency, not correctness. It would not have caught item 1, the invalid binomial interval, because that claim was perfectly consistent with everything around it and simply should not have been made. Nothing mechanical substitutes for reading a reviewer's objection carefully enough to answer the objection rather than its surface form.

Two things also remain outside any check here: the Ultralytics version string, which is a gate that fails by design until someone supplies it, and the four references located but not bibliographically verified from this environment.
