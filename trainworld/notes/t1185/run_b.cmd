@echo off
rem T-1185, run B: east (152 Chinese zone rows), no join
cd /d C:\Users\anita\projects\maps
set PYTHONIOENCODING=utf-8
set TW_SIM_DIR=C:\Users\anita\projects\maps\trainworld\notes\t1150\run\simroot\sim
set TW_CARGO_TARGET_DIR=C:\Users\anita\projects\maps\trainworld\notes\t1150\run\simroot\sim\target
set PATH=%USERPROFILE%\.cargo\bin;%PATH%
venv\Scripts\python.exe -I trainworld\pipeline\world_run.py run --rows wuj,cau,yoq,hun,qio,taj,yuh,zhc,pid,hng,tnc,lnq,dot,hay,juo,quf,zou,lnx,qic,guo,yng,xox,fna,snx,yhu,nnh,xny,big,caf,xna,dux,zif,sce,toc,wue,jsu,fic,lca,sya,xuo,txi,toa,ydu,szn,xgh,jaf,pya,keh,dny,gxi,syn,byi,lnd,rua,fng,pzh,sso,sxi,lgb,lba,xne,mua,sgu,xus,jua,hiy,aqi,sgr,lbc,gnx,rnc,jaj,kue,hoi,hba,tiy,hbb,wug,lha,lsu,lge,zoy,ywu,yox,dof,hgx,sgx,hio,diy,hja,lyg,pxx,aya,dgy,jxi,ysi,sgh,qhe,tui,lpn,lqi,lzu,whe,ruj,goh,nny,yad,gaf,lqu,pii,hbd,diu,fya,yga,zuj,zas,gom,cgi,seh,nae,lbh,gic,ygx,yaw,gxx,zaq,lnk,nnd,jxa,fny,gur,nnu,goo,jaw,sbd,ynj,dga,ysx,rud,pnx,fnx,cny,jnd,bnh,sbe,taw,xyc,pnh,jsa,fii,lbj,fnh --no-join --jobs 3 --port 6172 --coarse 1000 > C:\Users\anita\projects\maps\artifacts\trainworld\t1185-b.log 2>&1
