@echo off
title Deploy VisionCalc to GitHub (vision-calc)
echo ======================================================================
echo   VisionCalc: Deploying to GitHub (varshikakureti842-blip/vision-calc)
echo ======================================================================
echo.
cd /d "C:\Users\varsh.VARSHI\gesture_math_calculator"

echo [1/3] Verifying Git repository status...
git status

echo.
echo [2/3] Setting remote origin...
git remote remove origin 2>nul
git remote add origin https://github.com/varshikakureti842-blip/vision-calc.git
git remote -v

echo.
echo [3/3] Pushing to GitHub (main branch)...
git push -u origin main

echo.
if %ERRORLEVEL% EQU 0 (
    echo ======================================================================
    echo  SUCCESS! Project deployed to:
    echo  https://github.com/varshikakureti842-blip/vision-calc
    echo ======================================================================
) else (
    echo [ERROR] Push failed. If needed, try force push:
    echo git push -u origin main --force
)
pause
