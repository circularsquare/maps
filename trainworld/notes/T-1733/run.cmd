@echo off
rem usage: run.cmd <tag> <save> <workers> <view lng,lat,zoom> <edit lng,lat> [edit2 lng,lat]
set "QS=&demandWorkers=%3"
set VIEW=%~4
set EDIT=%~5
set EDIT2=%~6
node C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\f9a7f88d-0531-4963-9df6-6771326e513f\scratchpad\t1733\edit.mjs http://localhost:6122/ %2 C:\Users\anita\projects\maps\artifacts\trainworld\t1733\%1.json 75 > C:\Users\anita\projects\maps\artifacts\trainworld\t1733\%1.log 2>&1
