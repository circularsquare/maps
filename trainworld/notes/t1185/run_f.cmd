@echo off
rem T-1185, run F: runs B and C's 22 parked rows again (relaxed, set built: gate only), no join
cd /d C:\Users\anita\projects\maps
set PYTHONIOENCODING=utf-8
set TW_QUIET_S=3600
set TW_SIM_DIR=C:\Users\anita\projects\maps\trainworld\notes\t1150\run\simroot\sim
set TW_CARGO_TARGET_DIR=C:\Users\anita\projects\maps\trainworld\notes\t1150\run\simroot\sim\target
set PATH=%USERPROFILE%\.cargo\bin;%PATH%
venv\Scripts\python.exe -I trainworld\pipeline\world_run.py run --rows lnd,hgx,enh,hch,lgs,soz,lgh,xhx,wga,gia,xya,sim,yoz,lyu,tyu,hgj,xup,axa,lhn,syf,bdn,yyg --no-join --jobs 3 --port 6176 --coarse 1000 > C:\Users\anita\projects\maps\artifacts\trainworld\t1185-f.log 2>&1
