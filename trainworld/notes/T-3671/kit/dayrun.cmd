@echo off
rem T-3671 daytime rebuild batch: cai, elm (+slw), clt (+anu); one EG+GT join. notes/T-3671.md.
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\anita\projects\maps\trainworld\notes\T-3671\kit\dayrun.ps1" -Want 6 %*
