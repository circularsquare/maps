@echo off
rem Night of 2026-11-17 resumed (nightfix): the queue from batch 1 wave 1's rerun, 9 cores, no batch 3.
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\anita\projects\maps\trainworld\notes\T-3455\kit\night1117b.ps1" -Want 9 -NoCif %*
