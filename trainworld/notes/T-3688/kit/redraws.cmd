@echo off
rem T-3688 overnight redraws: Muzaffarpur split; one IN join. notes/T-3688.md.
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\anita\projects\maps\trainworld\notes\T-3688\kit\redraws.ps1" -Want 6 %*
