@echo off
rem as run.cmd, with ?perfOff=firm (the old cancelling behaviour)
set "QS=&demandWorkers=%3&perfOff=firm"
set VIEW=%~4
set CENTER=%~5
set EVERY=%~6
set MINUTES=%~7
node C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\f9a7f88d-0531-4963-9df6-6771326e513f\scratchpad\t1734\steady.mjs http://localhost:6122/ %2 C:\Users\anita\projects\maps\artifacts\trainworld\t1734\%1.json > C:\Users\anita\projects\maps\artifacts\trainworld\t1734\%1.log 2>&1
