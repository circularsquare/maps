@echo off
rem T-1185, run G: run A's 75 parked rows again (relaxed, set built: gate only), no join
cd /d C:\Users\anita\projects\maps
set PYTHONIOENCODING=utf-8
set TW_QUIET_S=3600
set TW_SIM_DIR=C:\Users\anita\projects\maps\trainworld\notes\t1150\run\simroot\sim
set TW_CARGO_TARGET_DIR=C:\Users\anita\projects\maps\trainworld\notes\t1150\run\simroot\sim\target
set PATH=%USERPROFILE%\.cargo\bin;%PATH%
venv\Scripts\python.exe -I trainworld\pipeline\world_run.py run --rows fln,bzu,xha,yzo,jsh,jgx,gnn,rne,qxx,zex,xau,sba,mlx,xxu,qxn,paz,jno,gli,wau,hoa,qji,xgy,ylo,zos,lcg,xno,xud,yab,cui,yna,zay,wiy,zji,yyn,dch,wig,yxi,ynn,fej,fni,lpg,mng,yay,axi,lup,jnj,kia,lcx,kii,yoe,gei,duy,lpx,hie,jgh,kih,fax,hus,roh,lgo,lgg,zaj,bji,ery,mji,nao,cnz,lgl,zeg,bzx,kru,naq,tnr,nga,ygi --no-join --jobs 3 --port 6177 --coarse 1000 > C:\Users\anita\projects\maps\artifacts\trainworld\t1185-g.log 2>&1
