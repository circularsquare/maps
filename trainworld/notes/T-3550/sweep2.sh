#!/bin/sh
# sweep2.sh <exe A> <tag A> <exe B> <tag B>: sweep.sh's runs, A and B one after the other per line, so both meet the same machine load.
S=/c/Users/anita/AppData/Local/Temp/claude/dce1fefcc1d43aef/f4ff94a8-c06c-4457-a593-caeb3269d7c2/scratchpad/trainrate
mkdir -p $S/$2 $S/$4
for l in $(seq 0 170); do
  $1 $S/lon.net $S/$2/lon-$l count,count_undo,count2,count2_undo $l > $S/$2/lon-$l.log 2>&1
  $3 $S/lon.net $S/$4/lon-$l count,count_undo,count2,count2_undo $l > $S/$4/lon-$l.log 2>&1
done
for l in 16 0; do
  $1 $S/nyc.net $S/$2/nyc-$l count,count_undo,count2,count2_undo $l > $S/$2/nyc-$l.log 2>&1
  $3 $S/nyc.net $S/$4/nyc-$l count,count_undo,count2,count2_undo $l > $S/$4/nyc-$l.log 2>&1
done
echo done > $S/$4/DONE
