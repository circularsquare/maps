import json, sys
city = sys.argv[1]
d = json.load(open(f'C:/Users/anita/projects/maps/artifacts/trainworld/t1750/{city}_dump.json', encoding='utf-8'))
PH = [4, 6, 4, 4, 6]; PHF = d['view']['peakHourFactor']
rail = d['view']['railTripsPerDay']
riders = sum(l['dm']['ridersPerDay'] for l in d['lines'] if l['dm'])
st = d['stationDemand']
entries = sum(s[2] for s in st); xfer = sum(s[4] for s in st)
print(f'rail trips {rail/1e6:.2f}M  line riders {riders/1e6:.2f}M ({riders/rail:.2f}/trip)  station boardings {entries/1e6:.2f}M  of which transfers {xfer/1e6:.2f}M ({xfer/entries:.2f})  first boardings (entries) {(entries-xfer)/1e6:.2f}M')
# load/crush distribution at the busiest hour of each high period, weighted by riders-km? : share of line-segment-hours over thresholds
import collections
bins = collections.Counter(); tot = 0.0
arr = brd = 0.0
for l in d['lines']:
    dm = l['dm']
    if not dm or l['stat']['status'] != 'running': continue
    n = len(l['stops']); tph = dm['tph']; cars = dm['cars']
    for q in range(5):
        lev = 0 if q < 3 else (1 if q == 3 else 2)
        cap = tph[lev] * cars * 160
        if cap <= 0: continue
        for x in dm['loads'][q]:
            hour = x / PH[q] * PHF[q]
            r = hour / cap
            w = x  # rider-segments
            b = 'lt70' if r < 0.7 else ('70-85' if r < 0.85 else ('85-100' if r < 1.0 else 'ge100'))
            bins[b] += w; tot += w
    if dm.get('arrive'):
        arr += sum(sum(a) for a in dm['arrive']); brd += sum(sum(a) for a in dm['board'])
print('rider-segments by load/crush (busiest hour):', {k: round(v / tot, 3) for k, v in bins.items()})
print(f'arrive {arr/1e6:.3f}M board {brd/1e6:.3f}M turned away at platforms {100*(arr-brd)/arr:.2f}% of line boardings')
