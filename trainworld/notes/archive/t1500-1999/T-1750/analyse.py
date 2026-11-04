"""T-1750: per-line speeds, service and loads from a dump_real.mjs dump.
usage: analyse.py <city> [name-substring ...]   (no names = the 25 busiest lines)"""
import json, sys
city = sys.argv[1]
names = sys.argv[2:]
d = json.load(open(f'C:/Users/anita/projects/maps/artifacts/trainworld/t1750/{city}_dump.json', encoding='utf-8'))
PH = [4, 6, 4, 4, 6]          # period hours
PHF = d['view']['peakHourFactor']
print('view', {k: d['view'][k] for k in ('round', 'final', 'tripsPerDay', 'railTripsPerDay', 'split')}, 'PHF', PHF)

def sm(l):
    st = l['stat']; n = len(l['stops']); dm = l['dm']
    if not dm or st['status'] != 'running': return None
    t = st['fromStartByLevel'][0]
    km = st['km']
    tmin = t[-1] / 60
    dwell = st['dwellS']
    run_s = t[-1] - dwell * (n - 2)
    top = max((s['reached'] for s in st['stretches']), default=0)
    why = [0, 0, 0, 0]
    for s in st['stretches']: why[s['why']] += 1
    stopS = st['stopS'][0]
    # peak hour load vs capacity at crush, worst segment over the three high periods
    tph = dm['tph'][0]; cars = dm['cars']
    worst = 0; worst_q = None; mean_peak = 0
    nseg = n - 1
    for q in (0, 2):
        loads = dm['loads'][q]
        hour = [x / PH[q] * PHF[q] for x in loads]
        # per direction capacity = tph * cars * 160
        cap = tph * cars * 160
        w = max(hour) / cap if cap else 0
        if w > worst: worst, worst_q = w, q
    arrive = sum(sum(a) for a in dm['arrive']) if dm.get('arrive') else 0
    board = sum(sum(a) for a in dm['board']) if dm.get('board') else 0
    return dict(id=l['id'], name=l['name'], n=n, km=km, tmin=tmin, v_avg=km / (t[-1] / 3600) if t[-1] else 0, v_run=km / (run_s / 3600) if run_s > 0 else 0,
                dwell_share=dwell * (n - 2) / t[-1] if t[-1] else 0, top=top, why=why, spacing=km / (n - 1), tph_save=l['saveTph'], tph=dm['tph'], cars=cars,
                trains=st['trainsNeeded'], delay=st['delayS'], rt=st['roundTripByLevel'], riders=dm['ridersPerDay'], fullest=dm['fullestPct'], worst=worst, worst_q=worst_q,
                arrive=arrive, board=board, passed=len(st['passed']), loop=st['loop'])

rows = [r for r in (sm(l) for l in d['lines']) if r]
if names:
    sel = [r for r in rows if any(s in r['name'] for s in names)]
else:
    sel = sorted(rows, key=lambda r: -r['riders'])[:25]
out = []
for r in sel:
    out.append(f"#{r['id']} {r['name']} n={r['n']} {r['km']:.1f}km sp={r['spacing']:.2f} t={r['tmin']:.1f}min v={r['v_avg']:.1f}/{r['v_run']:.1f}(run) dwell={r['dwell_share']*100:.0f}% top={r['top']:.0f} why={r['why']} "
               f"tphSave={[round(x,1) for x in r['tph_save'].values()] if isinstance(r['tph_save'], dict) else r['tph_save']} tphSolved={[round(x,1) for x in r['tph']]} cars={r['cars']} trains={r['trains']} delay={[round(x) for x in r['delay']]} "
               f"riders={r['riders']:.0f} fullest={r['fullest']}% peakload/cap={r['worst']:.2f}(q{r['worst_q']}) arrive/board={r['arrive']:.0f}/{r['board']:.0f} loop={r['loop']}")
sys.stdout.buffer.write(('\n'.join(out) + '\n').encode('utf-8'))
