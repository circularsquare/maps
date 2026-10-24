$env:PYTHONIOENCODING = 'utf-8'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = 'C:/Users/anita/projects/maps/venv/Scripts/python.exe'
$m = "$here/mock.py"
$o = 'C:/Users/anita/projects/maps/artifacts/trainworld/t1168'
$nyc = 'C:/Users/anita/projects/maps/trainworld/T-007-real-nyc.save'
$w = 'C:/Users/anita/projects/maps/artifacts/trainworld/world-map-day73.save'
& $py $m $nyc 1 "$o/2.html" "--hover=col:3@100" "--hover=down:1@30" "--hover=cross:149 St@60" "--hover=dot:Franklin St@100"
& $py $m $nyc 3 "$o/4.html" "--hover=col:6@70" "--hover=dot:23 St@100"
& $py $m $nyc 7 "$o/6x.html" "--hover=col:0@96" "--hover=dot:Cypress Av@96" "--hover=down:2@30"
& $py $m $w 18 "$o/g.html" "--hover=down:0@30" "--hover=dot:4 Av@80"
& $py $m $w 117 "$o/keikyu.html" "--hover=col:0@80" "--hover=up:0@40"
node "$here/shot.mjs" "$o/2.html" "$o/2.png" "$o/4.html" "$o/4.png" "$o/6x.html" "$o/6x.png" "$o/g.html" "$o/g.png" "$o/keikyu.html" "$o/keikyu.png"
