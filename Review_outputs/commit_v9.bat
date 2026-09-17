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
git add retrain/make_figs_v2.py retrain/results_v2 figures_v2 || goto :fail
git status --short
echo.

echo === committing ======================================================
> "%TEMP%\arxis_v9_msg.txt" (
  echo Manuscript v9: audited results, ordinal objective, leakage measurement
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
  echo verify_v9.py parses the manuscript's own markdown tables and checks all
  echo 415 quoted values against retrain/results_v2. Currently zero failures.
  echo.
  echo Co-Authored-By: Claude Opus 5 ^<noreply@anthropic.com^>
  echo Claude-Session: https://claude.ai/code/session_018C1Tj1bRjrSAAeN9ooaRRg
)
git commit -F "%TEMP%\arxis_v9_msg.txt" || goto :fail
del "%TEMP%\arxis_v9_msg.txt" >nul 2>&1
echo.

echo === done ============================================================
git log --oneline -1
echo.
echo Branch paper/v9-final is committed locally. Nothing has been pushed.
echo To publish it:   git push -u origin paper/v9-final
echo To go back:      git checkout main
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
