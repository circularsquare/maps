@echo off
rem Batch 2 (T-3455) finished by nightfix3: the 8 parked rows again, the join, the after-steps, start tiers. 9 cores.
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\anita\projects\maps\trainworld\notes\T-3455\kit\night1118fix.ps1" -Want 9 %*
