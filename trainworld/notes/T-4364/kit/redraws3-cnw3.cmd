@echo off
rem T-4364 region redraws, restarted by redraws4 from China's wave 3 after zhu's origin fix (notes/T-4364.md "The job", "Stop at cn.w1").
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\anita\projects\maps\trainworld\notes\T-4364\kit\redraws3.ps1" -From cn.w3 %*
