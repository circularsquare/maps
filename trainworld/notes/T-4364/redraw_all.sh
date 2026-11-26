#!/bin/sh
# redraw the T-4364 sheets with the density layer, one after another (one core)
cd /c/Users/anita/projects/maps
T=trainworld/notes/T-4364
R=artifacts/trainworld/review
for pair in cn-guilin:guilin cn-wuhan:wuhan cn-huiyang:huiyang cn-nanning:nanning cn-liaoyang:liaoyang mex:yucatan isr:israel nor-stavanger:norway pol-skierniewice:skierniewice gbr-brighton:brighton deu-mettmann:mettmann twn-taitung:taitung ind-patna:patna egy-delta:nile-delta; do
  s=${pair%%:*}; n=${pair##*:}
  echo "== $s -> $n"
  venv/Scripts/python.exe -I $T/kit/dry.py $T/t4364-$s.json --out $T/dry --png $R/redraws-1125-$n.png 2>&1 | tail -2
done
venv/Scripts/python.exe -I $T/boston_sheet.py $R/redraws-1125-boston.png 2>&1 | tail -1
venv/Scripts/python.exe -I $T/cn_hengqin.py $T/t4364-cn-guangdong.json --out $T/dry --png $R/redraws-1125-guangdong.png 2>&1 | tail -1
echo DONE
