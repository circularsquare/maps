import json, sys
C = sys.argv[1]
d = json.load(open(f'C:/Users/anita/projects/maps/artifacts/trainworld/t1750/{C}_dump.json', encoding='utf-8'))
PH = [4, 6, 4, 4, 6]; PHF = d['view']['peakHourFactor']
rows = []
for l in d['lines']:
    dm = l['dm']
    if not dm or l['stat']['status'] != 'running': continue
    tph, cars = dm['tph'][0], dm['cars']
    best = (0, None, None)
    for q in (0, 1, 2):
        ld = dm['loads'][q]
        k = max(range(len(ld)), key=lambda i: ld[i])
        h = ld[k] / PH[q] * PHF[q]
        if h > best[0]: best = (h, q, k)
    h, q, k = best
    n = len(l['stops'])
    seg = k if k < n - 1 else k - (n - 1)
    rows.append((l['name'], n, tph, cars, h, h / (tph * cars * 160) if tph * cars else 0, h / (tph * cars * 143) if tph * cars else 0, q, 'run0' if k < n - 1 else 'run1', l['stopNames'][seg], dm['ridersPerDay']))
rows.sort(key=lambda r: -r[10])
out = []
for r in rows[:int(sys.argv[2])]:
    out.append(f"{r[0]:12s} tph={r[2]:.1f} cars={r[3]} peak-hour load {r[4]:8.0f}/h  = {100*r[5]:3.0f}% of crush(160/car) = {100*r[6]:3.0f}% on MLIT scale(143/car)  period {r[7]} {r[8]} after {r[9]}")
sys.stdout.buffer.write(('\n'.join(out) + '\n').encode('utf-8'))
