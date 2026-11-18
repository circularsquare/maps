@echo off
rem Night of 2026-11-18: the 11-17 queue at 9 cores (Anita wants a few free for daytime work).
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\anita\projects\maps\trainworld\notes\T-3455\kit\night1117.ps1" -Want 9 %*
