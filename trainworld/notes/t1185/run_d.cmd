@echo off
rem T-1185, run D: north and west (143 Chinese zone rows), no join
cd /d C:\Users\anita\projects\maps
set PYTHONIOENCODING=utf-8
set TW_SIM_DIR=C:\Users\anita\projects\maps\trainworld\notes\t1150\run\simroot\sim
set TW_CARGO_TARGET_DIR=C:\Users\anita\projects\maps\trainworld\notes\t1150\run\simroot\sim\target
set PATH=%USERPROFILE%\.cargo\bin;%PATH%
venv\Scripts\python.exe -I trainworld\pipeline\world_run.py run --rows wuq,cnx,jix,waf,feg,hig,hna,fux,jut,hic,wec,cac,giz,snh,dmi,wxx,yax,cgu,gnu,lhe,hnn,hzu,zag,qug,scn,hoq,zug,lit,yla,rnq,nne,lnj,jla,yni,hnh,blx,kux,xni,qan,cak,fyx,ada,lgz,pns,bln,pnu,lik,qnz,yaz,hnx,lbb,lsx,jnc,xln,jap,dih,nnj,tbe,bto,nna,yit,biq,bdi,koo,dsh,tch,ysu,krx,dwa,pah,til,nnn,hei,dns,xij,gae,krm,wch,klx,lta,nee,gna,fea,jga,yrk,xui,aku,mhe,lgi,kiu,xxx,pey,mnz,hln,hsn,zxx,jug,acn,lci,zuh,szx,yik,wdu,lyx,ksg,waa,goe,szu,jaq,lbg,ygn,tgx,yal,suq,xgt,rau,fji,lxh,yam,hlu,gnz,gou,jul,dto,sgd,oro,gca,sgs,dwu,kes,zab,sge,gyn,kgi,myu,aty,cgj,kei,hnz,lbk,gnh,alf,qim --no-join --jobs 2 --port 6174 --coarse 1000 > C:\Users\anita\projects\maps\artifacts\trainworld\t1185-d.log 2>&1
