@echo off
REM ---------------------------------------------------------------------------
REM Commits the v9 manuscript, experiment drivers, result files and figures onto
REM a new branch. Does not push, does not touch main, does not modify any file
REM that is already tracked apart from retrain\metrics.py.
REM
REM Run it by double-clicking, or from a terminal in the repo root:
REM     commit_v9.bat
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

echo === creating branch paper/v9-final ==================================
git rev-parse --verify --quiet paper/v9-final >nul
if errorlevel 1 (
  git checkout -b paper/v9-final || goto :fail
) else (
  echo   branch already exists, switching to it
  git checkout paper/v9-final || goto :fail
)
echo.

echo === staging =========================================================
git add ARXIS_Manuscript_v9.md ARXIS_Manuscript_v9_SPE.docx arxis_v9_artifacts.zip verify_v9.py || goto :fail
git add retrain/metrics.py retrain/safety_metrics.py retrain/new_baselines.py || goto :fail
git add retrain/run_all_v2.py retrain/run_loco_v3.py retrain/run_calib_actions_v2.py || goto :fail
git add retrain/run_anomaly_influence.py retrain/run_final.py retrain/run_leakage.py || goto :fail
git add retrain/make_figs_v2.py retrain/run_rho_sensitivity.py || goto :fail
git add retrain/results_v2 figures_v2 INTERNAL_author_worklist.md || goto :fail
git status --short
echo.

echo === committing ======================================================
> "%TEMP%\arxis_v9_msg.txt" (
  echo Manuscript v9: submission copy, resolved references, rho sensitivity
  echo.
  echo Final revision of the SPE Journal submission. Every number in the body
  echo now reproduces from a stored result file, and the audit that proves it
  echo ships with the paper.
  echo.
  echo New experiments:
  echo - run_final.py: 30-run model comparison with the Bayesian correlated
  echo   t-test of Benavoli et al. ^(2017^), ROPE +/- 1 accuracy point; extended
  echo   leave-one-class-out grid to eleven comparators; ordinal cost-matrix
  echo   objective; cost-ratio selection.
  echo - run_leakage.py: four partitioning protocols x six models x five seeds,
  echo   plus cost-ratio selection on a blocked validation band.
  echo - run_anomaly_influence.py: counterfactual sweep of the anomaly input
  echo   with the other twenty-one features held fixed.
  echo.
  echo Findings added to the paper:
  echo - A random split over overlapping windows adds 1.66 to 4.08 accuracy
  echo   points to every learned model. The 20-window embargo contributes
  echo   almost nothing; the block structure does the work.
  echo - The cost-weighted policy is practically equivalent to the unweighted
  echo   network with posterior probability 0.960.
  echo - The ordinal cost matrix raises escalation on unseen Mixture from 0.245
  echo   to 0.393, at a cost of 21 accuracy points.
  echo - Gradient boosting's missed-hazard rate moves from 0.0098 to 0.3405
  echo   when the anomaly channel is forced to zero.
  echo.
  echo Corrections:
  echo - retrain/metrics.py divided false alarms by the number of alerts rather
  echo   than by clean windows. The two agree only at zero, which is why it went
  echo   unnoticed. Now matches retrain/safety_metrics.py, the module behind
  echo   every published number.
  echo - Roughly thirty stale figures and cross-references in the manuscript,
  echo   left over from the five-seed, eight-comparator draft.
  echo - fig_dissociation read five-seed accuracies while the text cited the
  echo   30-run table, and clipped the threshold rule off the axis.
  echo - fig_disposition read the eight-model LOCO file against a table of
  echo   eleven.
  echo.
  echo Submission hygiene:
  echo - Removed the two internal appendices ^(author work list, verification
  echo   items^) from the manuscript. They are kept in INTERNAL_author_worklist.md,
  echo   which is not manuscript content.
  echo - Resolved all 14 CITATION NEEDED markers against the sources. Liang et al.
  echo   is 2023, not 2024; Sabbagh is 6869-6882 SPE-223612-PA; Sharma's author
  echo   list, Dehnaw's author list and the Gemma 3 identifier were wrong.
  echo - Corrected the thermal-image account. The images were collected and used,
  echo   and the classifier trained on them is part of the pipeline; the real
  echo   limitation is that their split is randomized rather than block-wise, and
  echo   the resulting asymmetry between the two paths is now stated explicitly.
  echo - Bounded the ordinal objective as a methodological result rather than a
  echo   model, in Section 4.5 and Section 5.3.
  echo.
  echo New: run_rho_sensitivity.py sweeps the correlation term of the Bayesian
  echo test from 0 to 0.5 ^(Table 7^). Every comparison keeps its direction across
  echo the range; the strength of the equivalence claims does depend on rho, and
  echo the paper now says so.
  echo.
  echo verify_v9.py parses the manuscript's own markdown tables and checks all
  echo 479 quoted values against retrain/results_v2, plus the submission-hygiene
  echo rules above. Currently zero failures.
  echo.
  echo Co-Authored-By: Claude Opus 5 ^<noreply@anthropic.com^>
  echo Claude-Session: https://claude.ai/code/session_018C1Tj1bRjrSAAeN9ooaRRg
)
git commit -F "%TEMP%\arxis_v9_msg.txt" || goto :fail
del "%TEMP%\arxis_v9_msg.txt" >nul 2>&1
echo.

echo === committed =======================================================
git log --oneline -1
echo.

set /p DOPUSH=Push paper/v9-final to origin now? [y/N]
if /i not "%DOPUSH%"=="y" (
  echo.
  echo Left local. To publish later:  git push -u origin paper/v9-final
  echo To go back to main:            git checkout main
  echo.
  pause
  exit /b 0
)

echo.
echo === pushing =========================================================
echo If GitHub asks you to sign in, that prompt is from git, not from this
echo script. Nothing about your credentials is recorded here.
git push -u origin paper/v9-final
if errorlevel 1 (
  echo.
  echo Push did not complete. The commit is safe on paper/v9-final either way;
  echo you can retry with:  git push -u origin paper/v9-final
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
