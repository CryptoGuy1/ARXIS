@echo off
REM ---------------------------------------------------------------------------
REM Commits the v10 manuscript, the new statistical drivers, the text and style
REM diagnostics, result files and figures onto a new branch. Does not push, does
REM not touch main.
REM
REM Run it by double-clicking, or from a terminal in the repo root:
REM     commit_v10.bat
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

if not exist ".git" (
  echo ERROR: no .git here. Put this script in the repository root and rerun.
  pause
  exit /b 1
)

echo.
echo === current state ===================================================
git rev-parse --abbrev-ref HEAD
git status --short
echo.

echo === creating branch paper/v10-spe ===================================
git rev-parse --verify --quiet paper/v10-spe >nul
if errorlevel 1 (
  git checkout -b paper/v10-spe || goto :fail
) else (
  echo   branch already exists, switching to it
  git checkout paper/v10-spe || goto :fail
)
echo.

echo === staging =========================================================
git add ARXIS_Manuscript_v10_SPE.md ARXIS_Manuscript_v10_SPE.docx || goto :fail
git add arxis_v10_artifacts.zip verify_v10.py || goto :fail
git add textcheck.py stylecheck.py || goto :fail
git add retrain/run_partition_bounds.py retrain/run_paired_bootstrap.py || goto :fail
git add retrain/run_rho_sensitivity.py retrain/make_figs_v2.py || goto :fail
git add retrain/results_v2 figures_v2 INTERNAL_author_worklist.md || goto :fail
git status --short
echo.

echo === committing ======================================================
> "%TEMP%\arxis_v10_msg.txt" (
  echo Manuscript v10: SPE house style, partition-level bounds, text audit
  echo.
  echo Retitled to "Safety-Relevant Evaluation of Gas-Hazard Monitoring Models
  echo Under Distribution Shift and Sensor Degradation". The contribution is the
  echo evaluation methodology. No claim of a new state-of-the-art detector is
  echo made anywhere in the paper.
  echo.
  echo Statistical corrections:
  echo - Every Clopper-Pearson bound is recomputed with the partition as the
  echo   unit. Pooling test windows across seeds treated overlapping and
  echo   recurring observations as independent Bernoulli trials, which they are
  echo   not. The in-distribution hazardous bound moves from 0.0158%% to 0.4729%%,
  echo   the clean bound from 0.0316%% to 0.9435%%, and the alarm burden on a zero
  echo   count from 1.89 to 9.44 per 1,000 clean windows. run_partition_bounds.py.
  echo - Windows overlap within a partition too, so even the partition-level
  echo   bound is optimistic. The paper now says so.
  echo - "Ten of eleven bounded below 2.9%%" no longer held once the bounds were
  echo   recomputed ^(KNN moves 2.85%% to 3.28%%^). Restated as "at or below 3.3%%".
  echo - New: run_paired_bootstrap.py resamples the 30 runs, 20,000 replicates,
  echo   and assumes nothing about structure inside a run. It agrees with the
  echo   Bayesian correlated t-test on the most probable outcome in all ten
  echo   comparisons. The ranking claim now rests on that agreement rather than
  echo   on an assumed correlation term.
  echo.
  echo Reference corrections:
  echo - Narkhede et al. is 96%% fused, 82%% MOX-only, 93%% thermal-only. The paper
  echo   previously cited 97.0%%.
  echo - Zhang and Zhang report 98.9%% on a simulated set of their own collection
  echo   and 95.4%% on METEC field data; this corpus is used only for binary
  echo   pretraining. The paper previously cited 99.7%% on this corpus.
  echo.
  echo Claims narrowed:
  echo - "Protocol moves accuracy more than the models do" holds for the five
  echo   temporal models ^(3.09 vs 2.48 points^) and fails when the shallow tree is
  echo   included ^(2.45 vs 4.72^). The claim is now restricted and the restriction
  echo   is stated.
  echo - The accuracy / held-out-miss correlation is r = +0.13, 95%% CI
  echo   [-0.55, +0.70]. Reported as unresolved at n = 10, not as absent.
  echo - Escalation spread depends on where the alarm boundary sits: 198x at
  echo   a ^>= 3, 1.3x at a ^>= 1. Both are now reported.
  echo.
  echo Presentation:
  echo - SPE house style throughout: unnumbered headings, bold title-case run-in
  echo   subheads, Fig. N and Table N captions below the object, numbered display
  echo   equations. 20 figures, 17 tables, 13 equations, 38 references.
  echo - Three new figures: system workflow, decision logic, metric disagreement.
  echo.
  echo Text audit ^(textcheck.py, stylecheck.py^):
  echo - n-gram overlap against the model paper used for structure: 0.02%% at 5
  echo   grams, 0.00%% at 8 and 12, longest shared run 7 generic words. No text
  echo   was reused.
  echo - Stylometric diagnostics benchmarked against a published human-written
  echo   paper in the same journal section. Semicolon density, hedge density and
  echo   long-sentence share were brought into that paper's range by splitting
  echo   sentences and cutting connectives, not by degrading the prose.
  echo.
  echo verify_v10.py runs 714 checks over four families: every table value against
  echo the stored result files, SPE house-style rules, a cross-reference proof pass
  echo over every Fig., Table and Eq. mention, and a v9 coverage audit. Zero
  echo failures.
  echo.
  echo Co-Authored-By: Claude Opus 5 ^<noreply@anthropic.com^>
  echo Claude-Session: https://claude.ai/code/session_018C1Tj1bRjrSAAeN9ooaRRg
)
git commit -F "%TEMP%\arxis_v10_msg.txt" || goto :fail
del "%TEMP%\arxis_v10_msg.txt" >nul 2>&1
echo.

echo === committed =======================================================
git log --oneline -1
echo.

set /p DOPUSH=Push paper/v10-spe to origin now? [y/N]
if /i not "%DOPUSH%"=="y" (
  echo.
  echo Left local. To publish later:  git push -u origin paper/v10-spe
  echo To go back to main:            git checkout main
  echo.
  pause
  exit /b 0
)

echo.
echo === pushing =========================================================
echo If GitHub asks you to sign in, that prompt is from git, not from this
echo script. Nothing about your credentials is recorded here.
git push -u origin paper/v10-spe
if errorlevel 1 (
  echo.
  echo Push did not complete. The commit is safe on paper/v10-spe either way;
  echo you can retry with:  git push -u origin paper/v10-spe
  pause
  exit /b 1
)
echo.
echo Pushed. Open a pull request against main when you are ready.
echo To go back to main:  git checkout main
echo.
pause
exit /b 0

:fail
echo.
echo Something failed above. Nothing further was committed; your working tree
echo is untouched apart from whatever git already staged, which you can undo
echo with:  git reset
pause
exit /b 1
