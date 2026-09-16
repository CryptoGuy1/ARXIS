@echo off
REM ============================================================================
REM  ARXIS v13 - commit the reviewer-driven revision to the repository.
REM
REM  Run this from the repository root:
REM
REM      cd C:\Users\HP\Downloads\Agentic-AI-for-Pipeline-Leak-Detection-main
REM      commit_v13.bat
REM
REM  The v13 files are already in the working tree. This script only stages,
REM  commits and (after you confirm) pushes them. It does not overwrite
REM  anything and it does not touch files it did not add.
REM
REM  The commit message lives in commit_v13_message.txt next to this script.
REM ============================================================================

setlocal
cd /d "%~dp0"

set BRANCH=arxis-v13-referee-round4

echo.
echo === repository ===
git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
  echo ERROR: this folder is not a git repository.
  goto :end
)
git rev-parse --abbrev-ref HEAD
git remote -v
echo.

echo === current state ===
git status --short
echo.

if not exist commit_v13_message.txt (
  echo ERROR: commit_v13_message.txt is missing. It must sit next to this script.
  goto :end
)

echo === branch %BRANCH% ===
git rev-parse --verify %BRANCH% >nul 2>&1
if errorlevel 1 (
  git checkout -b %BRANCH%
) else (
  git checkout %BRANCH%
)
if errorlevel 1 goto :end
echo.

echo === staging the v13 file set ===

REM manuscript and supporting information
git add ARXIS_Manuscript_v13_SPE.md
git add ARXIS_Manuscript_v13_SPE.docx
git add ARXIS_Supporting_Information.md
git add ARXIS_Supporting_Information.docx

REM verification and release gates
git add verify_v13.py
git add audit_release.py
git add make_manifest.py
REM The Makefile could not be written into this folder from the session
REM (Windows protects that name against remote writes). Save the copy that
REM was sent in the chat into this folder as "Makefile" before running this
REM script, or the `make verify` target will be missing from the release.
if exist Makefile (
  git add Makefile
) else (
  echo WARNING: Makefile is not present. Save it from the chat into this folder.
)
git add RUN_MANIFEST.json
git add requirements.lock
git add SUBMISSION_GATE.md
git add RELEASE_CHECKLIST.md

REM documentation of what changed and why
REM the artifact README is added under its own name so it does not displace
REM the repository's existing README.md
git add ARTIFACT_README.md
git add RESPONSE_TO_REVIEWER.md
git add CHANGE_REVIEW_v10_to_v11.md
git add CHANGE_REVIEW_v11_to_v12.md
git add ARXIS_Change_Review_v11_to_v12.docx
git add CHANGE_REVIEW_v13_adversarial.md
git add ARXIS_Change_Review_v13_adversarial.docx
git add CHANGE_REVIEW_v13_presubmission.md
git add ARXIS_Change_Review_v13_presubmission.docx

REM this script and its commit message, so the record is self-describing
git add commit_v13.bat
git add commit_v13_message.txt

REM experiment drivers - the four corrected defects live in here
git add retrain/*.py

REM regenerated result files and figures
git add retrain/results_v2
git add figures_v2

echo.
echo === what will be committed ===
git status --short
echo.
echo Review the list above.
pause

git commit -F commit_v13_message.txt
if errorlevel 1 (
  echo.
  echo Commit failed. Nothing was pushed.
  goto :end
)

echo.
echo === commit created ===
git log -1 --stat
echo.

echo Ready to push branch %BRANCH% to origin.
echo Close this window now if you would rather push by hand.
pause

git push -u origin %BRANCH%

echo.
echo === done ===
git log --oneline -3

:end
echo.
pause
endlocal
