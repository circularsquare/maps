@echo off
rem T-4364 region redraws (agent redraws3): every 11-26 yes, a country at a time, Egypt and Taiwan behind her answer. notes/T-4364.md "The job".
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\anita\projects\maps\trainworld\notes\T-4364\kit\redraws3.ps1" %*
