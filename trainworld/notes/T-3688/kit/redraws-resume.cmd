@echo off
rem T-3688 restart without Manila (Anita wants to see it first): waits for the first start's retire of uji, then waves, join, after.
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\anita\projects\maps\trainworld\notes\T-3688\kit\redraws.ps1" -Want 6 -From wave1 -WaitPid 19704 %*
