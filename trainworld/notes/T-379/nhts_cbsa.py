"""T-379: NHTS 2017 weekday non-commute trips per person by straight-line band, by household CBSA
(the same linking and distance rule as research/intercity/nhts_long.py).

    python -I nhts_cbsa.py <csv.zip> <out.json>
"""
import os, sys, json, zipfile
os.environ.setdefault('OMP_NUM_THREADS', '3')
import numpy as np, pandas as pd

z = zipfile.ZipFile(sys.argv[1])
per = pd.read_csv(z.open('perpub.csv'), usecols=['HOUSEID', 'PERSONID', 'TRAVDAY', 'WTPERFIN', 'HH_CBSA', 'CENSUS_D'], dtype={'HOUSEID': str, 'PERSONID': str, 'HH_CBSA': str})
trp = pd.read_csv(z.open('trippub.csv'), usecols=['HOUSEID', 'PERSONID', 'TDTRPNUM', 'TRPMILES', 'WHYFROM', 'WHYTO', 'TDWKND', 'TRIPPURP', 'TRPTRANS'],
                  dtype={'HOUSEID': str, 'PERSONID': str, 'TDWKND': str, 'TRPTRANS': str})
trp = trp.merge(per[['HOUSEID', 'PERSONID', 'WTPERFIN', 'HH_CBSA', 'CENSUS_D']], on=['HOUSEID', 'PERSONID'], how='left', validate='m:1')
trp = trp.sort_values(['HOUSEID', 'PERSONID', 'TDTRPNUM']).reset_index(drop=True)
samep = (trp.HOUSEID == trp.HOUSEID.shift()) & (trp.PERSONID == trp.PERSONID.shift())
cont = samep & (trp.WHYTO.shift() == 7) & (trp.WHYFROM == 7)
trp['LID'] = (~cont).cumsum()
trp['MI'] = trp.TRPMILES.where(trp.TRPMILES >= 0)
trp['HBW'] = trp.TRIPPURP == 'HBW'
g = trp.groupby('LID', sort=False)
L = g.agg(TDWKND=('TDWKND', 'first'), CBSA=('HH_CBSA', 'first'), D=('CENSUS_D', 'first'), W=('WTPERFIN', 'first'), MI=('MI', lambda s: s.sum(min_count=1)), HBW=('HBW', 'max'))
L = L[L.MI.notna()]
L['KM'] = L.MI * 1.609344 / 1.25
EDGES = [0, 25, 50, 80, 100, 120, 150, 200, 250, 300, 500, 1000, 5000]
res = {'edges_km': EDGES}
groups = {'New York CBSA': ('35620',), 'Philadelphia CBSA': ('37980',), 'Boston CBSA': ('14460',), 'Washington CBSA': ('47900',), 'Northeast': None}
for name, cb in groups.items():
    if cb is None:
        pw = per[per.TRAVDAY.between(2, 6) & per.CENSUS_D.isin([1, 2])]
        t = L[(L.TDWKND == '02') & L.D.isin([1, 2]) & ~L.HBW]
    else:
        pw = per[per.TRAVDAY.between(2, 6) & per.HH_CBSA.isin(cb)]
        t = L[(L.TDWKND == '02') & L.CBSA.isin(cb) & ~L.HBW]
    persons = pw.WTPERFIN.sum()
    h, _ = np.histogram(t.KM, bins=EDGES, weights=t.W)
    n, _ = np.histogram(t.KM, bins=EDGES)
    res[name] = {'persons_weight': float(persons), 'sample_persons': int(len(pw)), 'rate': (h / persons).tolist(), 'n': n.tolist()}
    print(name, f"persons {len(pw)}")
    for i in range(len(EDGES) - 1):
        print(f"  {EDGES[i]:>4}-{EDGES[i+1]:<5} {h[i] / persons:.5f} (n {n[i]})")
json.dump(res, open(sys.argv[2], 'w'), indent=1)
