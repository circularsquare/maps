@echo off
rem T-4364 region redraws, restarted by redraws4 from the US's wave 2 (his) after bos's hand steps and gate (notes/T-4364.md "The job", "Stop at us.w1").
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\anita\projects\maps\trainworld\notes\T-4364\kit\redraws3.ps1" -From us.w2 %*
