@echo off
rem T-1185, run H: run E's 5 drive_factor parks (relaxed, set built: gate only), no join
cd /d C:\Users\anita\projects\maps
set PYTHONIOENCODING=utf-8
set TW_QUIET_S=3600
set TW_SIM_DIR=C:\Users\anita\projects\maps\trainworld\notes\t1150\run\simroot\sim
set TW_CARGO_TARGET_DIR=C:\Users\anita\projects\maps\trainworld\notes\t1150\run\simroot\sim\target
set PATH=%USERPROFILE%\.cargo\bin;%PATH%
venv\Scripts\python.exe -I trainworld\pipeline\world_run.py run --rows siu,lzo,yzn,nji,xhn --no-join --jobs 3 --port 6177 --coarse 1000 > C:\Users\anita\projects\maps\artifacts\trainworld\t1185-h.log 2>&1
