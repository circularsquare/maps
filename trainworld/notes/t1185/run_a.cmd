@echo off
rem T-1185, run A: south and south-west (180 Chinese zone rows), no join
cd /d C:\Users\anita\projects\maps
set PYTHONIOENCODING=utf-8
set TW_SIM_DIR=C:\Users\anita\projects\maps\trainworld\notes\t1150\run\simroot\sim
set TW_CARGO_TARGET_DIR=C:\Users\anita\projects\maps\trainworld\notes\t1150\run\simroot\sim\target
set PATH=%USERPROFILE%\.cargo\bin;%PATH%
venv\Scripts\python.exe -I trainworld\pipeline\world_run.py run --rows leh,sih,jig,dia,gau,heh,yoc,gug,lij,fuq,gay,kag,tis,zon,haf,xun,xnn,bbu,hzh,hii,beb,jny,dzu,wea,ziz,dnz,wag,lae,fln,bzu,xha,yzo,gii,jsh,jgx,gnn,rne,hji,cus,qxx,rns,zex,xau,sba,mlx,yce,jgy,xnb,xxu,ync,qxn,paz,dui,jno,gli,wau,hoa,qji,xcn,zej,xgy,bba,lgc,ylo,sni,zos,zoa,dni,zix,lfn,lcg,ldi,pnn,xsa,qln,qie,xno,xud,lgn,yab,fue,dgp,jne,cui,yna,zay,wiy,siw,zji,joc,bxx,yyn,dch,wig,zoi,jnu,yzu,blo,hbc,cga,mxi,ybi,lgx,hpu,yxi,fsu,ynn,xnl,fej,aye,fni,wmi,lpg,puj,mng,yay,lbe,gnb,axi,lup,qon,fes,jnj,lbf,kia,lcx,kii,yoe,gei,duy,whu,lpx,pnl,hie,zoh,dij,jgh,teo,cei,kih,fax,cti,hus,emi,roh,hdo,lgo,lgg,zaj,bji,ery,yid,dlx,mji,rax,cgh,nao,wcu,cnz,lgl,lbi,sut,jgn,quh,hya,qnc,peg,xyi,mnn,zeg,bzx,zau,dix,kru,naq,jat,tnr,nga,ygi,lji --no-join --jobs 3 --port 6171 --coarse 1000 > C:\Users\anita\projects\maps\artifacts\trainworld\t1185-a.log 2>&1
