@echo off
title Deploy VisionCalc to GitHub
echo ======================================================================
echo   VisionCalc: Deploying to GitHub (varshikakureti842-blip)
echo ======================================================================
echo.
cd /d "C:\Users\varsh.VARSHI\gesture_math_calculator"

echo [1/3] Verifying Git repository status...
git status

echo.
echo [2/3] Setting remote origin...
git remote remove origin 2>nul
git remote add origin https://github.com/varshikakureti842-blip/visioncalc.git
git remote -v

echo.
echo [3/3] Pushing to GitHub (main branch)...
git push -u origin main

echo.
if %ERRORLEVEL% EQU 0 (
    echo ======================================================================
    echo  SUCCESS! Project deployed to:
    echo  https://github.com/varshikakureti842-blip/visioncalc
    echo ======================================================================
) else (
    echo ======================================================================
    echo  NOTE: If you haven't created the repository on GitHub yet:
    echo  1. Open: https://github.com/new
    echo  2. Repository name: visioncalc
    echo  3. Click "Create repository" (leave it empty)
    echo  4. Re-run this script!
    echo ======================================================================
)
pause
