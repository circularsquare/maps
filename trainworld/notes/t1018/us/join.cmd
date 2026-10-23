@echo off
rem The joint solve for every gated US row, from this folder's sim copy: join.cmd <log tag> <jobs>
rem (back up first: backup.py <task> prejoin US)
cd /d C:\Users\anita\projects\maps
set PYTHONIOENCODING=utf-8
set TW_FAULT_S=900
set TW_SIM_DIR=%~dp0simroot\sim
set TW_CARGO_TARGET_DIR=%~dp0simroot\sim\target
set PATH=%USERPROFILE%\.cargo\bin;%PATH%
venv\Scripts\python.exe -I trainworld\pipeline\world_run.py join --countries US --jobs %2 > C:\Users\anita\projects\maps\artifacts\trainworld\%1-join.log 2>&1
