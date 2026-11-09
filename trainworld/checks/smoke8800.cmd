@echo off
rem Smoke 8800 light then dark, only before a publish (T-2831); the checks pin Chrome themselves (TW_CORES, checks/cores.mjs).
rem Logs: artifacts\trainworld\smoke8800-light.log and smoke8800-dark.log. About 3 minutes a shade.
set MAPS=%~dp0..\..
cd /d %MAPS%
set TW_CORES=4
node trainworld\checks\smoke.mjs http://localhost:8800/trainworld/ > "%MAPS%\artifacts\trainworld\smoke8800-light.log" 2>&1
set SMOKE_SHADE=dark
node trainworld\checks\smoke.mjs http://localhost:8800/trainworld/ > "%MAPS%\artifacts\trainworld\smoke8800-dark.log" 2>&1
echo done
