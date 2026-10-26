@echo off
rem T-1185, run E: the 16 rows across two groups, D's 5 relaxed parks (gate only) and qie again, no join
cd /d C:\Users\anita\projects\maps
set PYTHONIOENCODING=utf-8
set TW_QUIET_S=3600
set TW_SIM_DIR=C:\Users\anita\projects\maps\trainworld\notes\t1150\run\simroot\sim
set TW_CARGO_TARGET_DIR=C:\Users\anita\projects\maps\trainworld\notes\t1150\run\simroot\sim\target
set PATH=%USERPROFILE%\.cargo\bin;%PATH%
venv\Scripts\python.exe -I trainworld\pipeline\world_run.py run --rows ych,wih,llg,qad,sbb,cgn,ynu,nnx,siu,lzo,yzn,nji,cli,xhn,god,ysn,biq,wdu,ksg,aty,qim,qie --no-join --jobs 4 --port 6175 --coarse 1000 > C:\Users\anita\projects\maps\artifacts\trainworld\t1185-e.log 2>&1
