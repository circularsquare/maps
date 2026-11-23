@echo off
rem T-3688 Manila redraw: mai's fringes to its neighbours, San Pablo City and Lipa City new; one PH join. notes/T-3688.md "Manila".
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\anita\projects\maps\trainworld\notes\T-3688\kit\redraws-ph.ps1" -Want 6 %*
