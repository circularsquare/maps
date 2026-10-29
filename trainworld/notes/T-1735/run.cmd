@echo off
rem usage: run.cmd <tag> <save> <workers> <view lng,lat,zoom> <edit lng,lat> [extra query]
rem env passed through: EXPRESS, EDIT2, NOWHOLE
set "QS=&demandWorkers=%3%~6"
set VIEW=%~4
set EDIT=%~5
if not exist C:\Users\anita\projects\maps\artifacts\trainworld\t1735 mkdir C:\Users\anita\projects\maps\artifacts\trainworld\t1735
node %~dp0edit.mjs http://localhost:6123/ %2 C:\Users\anita\projects\maps\artifacts\trainworld\t1735\%1.json 75 > C:\Users\anita\projects\maps\artifacts\trainworld\t1735\%1.log 2>&1
