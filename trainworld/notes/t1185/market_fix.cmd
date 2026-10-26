@echo off
rem T-1185: rewrite the 88 market packs naming China's old zone places (stale.py)
cd /d C:\Users\anita\projects\maps
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=10
set TW_SIM_DIR=C:\Users\anita\projects\maps\trainworld\notes\t1150\run\simroot\sim
set TW_CARGO_TARGET_DIR=C:\Users\anita\projects\maps\trainworld\notes\t1150\run\simroot\sim\target
set PATH=%USERPROFILE%\.cargo\bin;%PATH%
venv\Scripts\python.exe -I trainworld\pipeline\world_run.py step market_zones ans any bao bei bin chc chg cho chs chz cix dal dat deg dog fos fuy fuz gan gua gui had hag har hef hog hoh hon hua hui jia jih jil jin jiw kai kun lan lif lin liu luo luy mcu mns nac nag nan nat nin put qin qua sha shi shn shu shx shy shz suz tag tay taz ten tia tin uru wen wuh wuu xia xim xin xiu xuc xuz yac yic yih yin yue zha zhe zho zhu zhz zib > C:\Users\anita\projects\maps\artifacts\trainworld\t1185-market-fix.log 2>&1
echo exit %ERRORLEVEL% >> C:\Users\anita\projects\maps\artifacts\trainworld\t1185-market-fix.log
