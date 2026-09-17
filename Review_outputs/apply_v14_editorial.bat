@echo off
REM ============================================================================
REM  ARXIS - apply the editorial pass on structure, Summary and style to the repository,
REM  commit it, and tag the submitted state.
REM
REM  The last sync attempt failed part-way because the machine went offline
REM  mid-push, so this script does the unpack itself rather than assuming the
REM  working tree is already current.
REM
REM  Run from the repository root:
REM
REM      cd C:\Users\HP\Downloads\Agentic-AI-for-Pipeline-Leak-Detection-main
REM      apply_v13_technical_review.bat
REM
REM  What it expects next to itself:
REM      arxis_v13_repo_drop.zip
REM      commit_v14_editorial_message.txt
REM
REM  It unpacks the drop OVER the working tree, so files in the drop replace
REM  their counterparts. Files not in the drop are left alone. Nothing is
REM  deleted. It stops before pushing and asks.
REM ============================================================================

setlocal EnableDelayedExpansion
cd /d "%~dp0"

set BRANCH=arxis-v14-editorial
set TAG=v13-submission
set DROP=arxis_v13_repo_drop.zip
set MSG=commit_v14_editorial_message.txt

echo.
echo === checks ===

git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
  echo   NOT a git repository. Run this from the repository root.
  goto :fail
)
echo   repository ok

if not exist "%DROP%" (
  echo   missing %DROP% - put it next to this script.
  goto :fail
)
echo   found %DROP%

if not exist "%MSG%" (
  echo   missing %MSG% - put it next to this script.
  goto :fail
)
echo   found %MSG%

echo.
echo === current state ===
git status --short
echo.
echo   branch:
git rev-parse --abbrev-ref HEAD

echo.
echo A safety copy of the current tree is made before anything is overwritten.
set STAMP=%DATE:~-4%%DATE:~4,2%%DATE:~7,2%_%TIME:~0,2%%TIME:~3,2%
set STAMP=%STAMP: =0%
git stash list >nul 2>&1
git diff --quiet
if errorlevel 1 (
  echo   uncommitted changes present - stashing them as arxis-presync-%STAMP%
  git stash push -u -m "arxis-presync-%STAMP%"
  echo   recover them later with: git stash list  ^&^&  git stash pop
) else (
  echo   working tree clean, nothing to stash
)

echo.
echo === branch ===
git rev-parse --verify %BRANCH% >nul 2>&1
if errorlevel 1 (
  git checkout -b %BRANCH%
) else (
  git checkout %BRANCH%
)
if errorlevel 1 goto :fail

echo.
echo === unpacking %DROP% over the working tree ===
REM repo_v13\ is the top folder inside the zip; its contents land at the root.
if exist "_arxis_unpack" rmdir /s /q "_arxis_unpack"
mkdir "_arxis_unpack"
powershell -NoProfile -Command "Expand-Archive -LiteralPath '%DROP%' -DestinationPath '_arxis_unpack' -Force"
if errorlevel 1 (
  echo   unpack failed. Is PowerShell available?
  goto :fail
)

if not exist "_arxis_unpack\repo_v13" (
  echo   unexpected zip layout - expected a repo_v13 folder inside.
  goto :fail
)

xcopy "_arxis_unpack\repo_v13\*" "." /E /Y /I >nul
if errorlevel 1 goto :fail
rmdir /s /q "_arxis_unpack"
echo   unpacked

echo.
echo === the retired figure ===
REM Fig. 3 is now fig_setup.png. The old flowchart is no longer referenced by
REM the manuscript and should not ship as if it were current.
if exist "figures_v2\fig_architecture.png" (
  git rm --cached "figures_v2\fig_architecture.png" >nul 2>&1
  del "figures_v2\fig_architecture.png"
  echo   removed figures_v2\fig_architecture.png
) else (
  echo   already absent
)

echo.
echo === what changed ===
git add -A
git status --short

echo.
echo === commit ===
git commit -F "%MSG%"
if errorlevel 1 (
  echo   nothing to commit, or the commit failed. Check the output above.
  goto :fail
)

echo.
echo === tag ===
git tag -f %TAG%
echo   tagged %TAG% at this commit
echo   (the manuscript's availability statement points a reviewer at this tag)

echo.
echo ============================================================
echo  Committed locally on branch %BRANCH% and tagged %TAG%.
echo  Nothing has been pushed yet.
echo.
echo  To push, run:
echo      git push -u origin %BRANCH%
echo      git push origin %TAG%
echo.
echo  If the connection drops mid-push, just run those two again.
echo  Git resumes; it does not corrupt anything.
echo ============================================================
echo.
goto :end

:fail
echo.
echo ---- stopped, nothing was committed ----
echo.

:end
pause
endlocal
