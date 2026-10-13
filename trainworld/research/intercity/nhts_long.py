"""T-276: weekday trips per person by straight-line distance, non-commute and commute, 2017 NHTS.

The regional market's trip rate, onset and distance curve are fitted to these (pipeline/market_fit.py).

    python -I nhts_long.py <nhts csv dir> <out json>

Linked trips (a change of vehicle, WHYTO 7 then WHYFROM 7, is one trip), weekday travel days (TDWKND
02), persons weighted by WTPERFIN, so each band's figure is trips a weekday per person. Distances are
the reported route miles (TRPMILES) over 1.25, the market's road detour, as straight-line km. Commute
= NHTS's home-based work (TRIPPURP HBW). Northeast = census divisions 1 and 2 (New England, Middle
Atlantic); the whole US is given beside it.
"""
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS', '3')
import numpy as np, pandas as pd

d, out = sys.argv[1], sys.argv[2]
per = pd.read_csv(os.path.join(d, 'perpub.csv'), usecols=['HOUSEID', 'PERSONID', 'TRAVDAY', 'WTPERFIN', 'CENSUS_D', 'GCDWORK', 'WORKER'],
                  dtype={'HOUSEID': str, 'PERSONID': str})
trp = pd.read_csv(os.path.join(d, 'trippub.csv'), usecols=['HOUSEID', 'PERSONID', 'TDTRPNUM', 'TRPMILES', 'WHYFROM', 'WHYTO', 'TRAVDAY', 'TDWKND', 'TRIPPURP', 'CENSUS_D', 'TRPTRANS'],
                  dtype={'HOUSEID': str, 'PERSONID': str, 'TDWKND': str, 'TRPTRANS': str})
trp = trp.merge(per[['HOUSEID', 'PERSONID', 'WTPERFIN']], on=['HOUSEID', 'PERSONID'], how='left', validate='m:1')
trp = trp.sort_values(['HOUSEID', 'PERSONID', 'TDTRPNUM']).reset_index(drop=True)
samep = (trp.HOUSEID == trp.HOUSEID.shift()) & (trp.PERSONID == trp.PERSONID.shift())
cont = samep & (trp.WHYTO.shift() == 7) & (trp.WHYFROM == 7)
trp['LID'] = (~cont).cumsum()
trp['MI'] = trp.TRPMILES.where(trp.TRPMILES >= 0)
# a linked trip's purpose: NHTS's TRIPPURP of its last segment (the activity it ends at), and HBW if any segment is
trp['HBW'] = trp.TRIPPURP == 'HBW'
trp['AIR'] = trp.TRPTRANS == '19'
g = trp.groupby('LID', sort=False)
L = g.agg(HOUSEID=('HOUSEID', 'first'), PERSONID=('PERSONID', 'first'), TDWKND=('TDWKND', 'first'), CENSUS_D=('CENSUS_D', 'first'),
          W=('WTPERFIN', 'first'), MI=('MI', lambda s: s.sum(min_count=1)), HBW=('HBW', 'max'), AIR=('AIR', 'max'))
L = L[L.MI.notna()]
L['KM'] = L.MI * 1.609344 / 1.25

EDGES = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500, 700, 1000, 1500, 2500, 5000]
res = {'edges_km': EDGES, 'note': 'trips a weekday per person by straight-line km band (route miles x 1.609 / 1.25); linked trips'}
for name, divs in (('northeast', [1, 2]), ('us', list(range(1, 10)))):
    pw = per[(per.TRAVDAY.between(2, 6)) & per.CENSUS_D.isin(divs)]
    persons = pw.WTPERFIN.sum()
    t = L[(L.TDWKND == '02') & L.CENSUS_D.isin(divs)]
    r = {'persons_weight': float(persons), 'sample_persons': int(len(pw)), 'sample_trips': int(len(t))}
    for kind, sel in (('noncommute', ~t.HBW), ('commute', t.HBW), ('noncommute_air', ~t.HBW & t.AIR)):
        tt = t[sel]
        h, _ = np.histogram(tt.KM, bins=EDGES, weights=tt.W)
        n, _ = np.histogram(tt.KM, bins=EDGES)
        r[kind] = {'rate': (h / persons).tolist(), 'n': n.tolist()}
    # commuters' straight-line home-work distance (GCDWORK, miles), workers on any day
    wk = per[per.CENSUS_D.isin(divs) & (per.WORKER == 1) & (per.GCDWORK >= 0)]
    h, _ = np.histogram(wk.GCDWORK * 1.609344, bins=EDGES, weights=wk.WTPERFIN)
    r['workers_gcdwork_share'] = (h / h.sum()).tolist()
    res[name] = r
with open(out, 'w') as f:
    json.dump(res, f, indent=1)
for name in ('northeast', 'us'):
    r = res[name]
    print(name, 'persons', r['sample_persons'], 'trips', r['sample_trips'])
    for i in range(len(EDGES) - 1):
        print(f"  {EDGES[i]:>5}-{EDGES[i+1]:<5} km  noncommute {r['noncommute']['rate'][i]:.5f} (n {r['noncommute']['n'][i]:>6})  air {r['noncommute_air']['rate'][i]:.5f}  commute {r['commute']['rate'][i]:.5f} (n {r['commute']['n'][i]:>5})  gcdwork {r['workers_gcdwork_share'][i]:.4f}")
