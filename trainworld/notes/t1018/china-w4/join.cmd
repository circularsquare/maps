@echo off
rem T-897 wave 4: the joint solve for every gated CN row (backup data/work/pre-T897/w4)
cd /d C:\Users\anita\projects\maps
set PYTHONIOENCODING=utf-8
set TW_FAULT_S=900
set TW_SIM_DIR=C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\2e31d80c-814f-4eb1-abc9-7e758da09685\scratchpad\china-w4\simroot\sim
set TW_CARGO_TARGET_DIR=C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\2e31d80c-814f-4eb1-abc9-7e758da09685\scratchpad\china-w4\simroot\sim\target
set PATH=%USERPROFILE%\.cargo\bin;%PATH%
venv\Scripts\python.exe -I trainworld\pipeline\world_run.py join --countries CN --jobs 9 > C:\Users\anita\projects\maps\artifacts\trainworld\t897-w4-join.log 2>&1
