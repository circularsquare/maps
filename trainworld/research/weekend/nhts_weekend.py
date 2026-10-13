"""Weekend vs weekday trip rates by game purpose, 2017 NHTS.
usage: python -I nhts_weekend.py <csv dir> <out json>
"""
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS', '6')
import numpy as np, pandas as pd

d, out = sys.argv[1], sys.argv[2]
per = pd.read_csv(os.path.join(d, 'perpub.csv'), usecols=['HOUSEID', 'PERSONID', 'TRAVDAY', 'WTPERFIN', 'R_AGE_IMP'], dtype={'HOUSEID': str, 'PERSONID': str})
trp = pd.read_csv(os.path.join(d, 'trippub.csv'), usecols=['HOUSEID', 'PERSONID', 'TDTRPNUM', 'STRTTIME', 'TRPMILES', 'WHYFROM', 'WHYTO', 'TRAVDAY', 'WTTRDFIN', 'R_AGE_IMP', 'TRIPPURP', 'WHYTRP1S'],
                  dtype={'HOUSEID': str, 'PERSONID': str, 'STRTTIME': str})

info = {}
info['persons'] = len(per); info['trips_raw'] = len(trp)
info['per_travday_counts'] = per.TRAVDAY.value_counts().sort_index().to_dict()
info['per_travday_wt_millions'] = (per.groupby('TRAVDAY').WTPERFIN.sum() / 1e6).round(2).to_dict()
info['trip_travday_counts'] = trp.TRAVDAY.value_counts().sort_index().to_dict()
info['whyto_codes'] = sorted(trp.WHYTO.unique().tolist())
info['whyfrom_codes'] = sorted(trp.WHYFROM.unique().tolist())
info['trippurp_codes'] = sorted(trp.TRIPPURP.unique().tolist())
info['strttime_min_max'] = [trp.STRTTIME.min(), trp.STRTTIME.max()]
info['strttime_len'] = trp.STRTTIME.str.len().value_counts().to_dict()
info['trpmiles_neg'] = int((trp.TRPMILES < 0).sum())
info['age_min_max'] = [int(per.R_AGE_IMP.min()), int(per.R_AGE_IMP.max())]
# cross-check codes: WHYTRP1S 10 should equal WHYTO in (3,4); TRIPPURP HBW should be home(1,2)<->work(3)
info['check_whytrp1s10_vs_whyto34'] = [int((trp.WHYTRP1S == 10).sum()), int(trp.WHYTO.isin([3, 4]).sum())]
info['check_HBW_vs_home_work'] = [int((trp.TRIPPURP == 'HBW').sum()),
                                  int(((trp.WHYFROM.isin([1, 2]) & (trp.WHYTO == 3)) | (trp.WHYTO.isin([1, 2]) & (trp.WHYFROM == 3))).sum())]

# join person weight to trips; check travday matches
trp = trp.merge(per[['HOUSEID', 'PERSONID', 'WTPERFIN', 'TRAVDAY']].rename(columns={'TRAVDAY': 'PDAY'}), on=['HOUSEID', 'PERSONID'], how='left', validate='m:1')
info['trips_without_person'] = int(trp.WTPERFIN.isna().sum())
info['trip_person_travday_mismatch'] = int((trp.TRAVDAY != trp.PDAY).sum())
# ratio WTTRDFIN / WTPERFIN (expect ~365)
r = trp.WTTRDFIN / trp.WTPERFIN
info['wttrdfin_over_wtperfin_quantiles'] = r.quantile([0.01, 0.5, 0.99]).round(2).tolist()

# --- link transfer segments (WHYTO=7 followed by WHYFROM=7) into whole trips
trp = trp.sort_values(['HOUSEID', 'PERSONID', 'TDTRPNUM']).reset_index(drop=True)
samep = (trp.HOUSEID == trp.HOUSEID.shift()) & (trp.PERSONID == trp.PERSONID.shift())
cont = samep & (trp.WHYTO.shift() == 7) & (trp.WHYFROM == 7)
trp['LID'] = (~cont).cumsum()
trp['MI'] = trp.TRPMILES.where(trp.TRPMILES >= 0)
g = trp.groupby('LID', sort=False)
L = g.agg(HOUSEID=('HOUSEID', 'first'), PERSONID=('PERSONID', 'first'), STRTTIME=('STRTTIME', 'first'),
          WHYFROM=('WHYFROM', 'first'), WHYTO=('WHYTO', 'last'), TRAVDAY=('TRAVDAY', 'first'),
          WTPERFIN=('WTPERFIN', 'first'), WTTRDFIN=('WTTRDFIN', 'first'), AGE=('R_AGE_IMP', 'first'),
          MI=('MI', lambda s: s.sum(min_count=1)), NSEG=('MI', 'size'))
info['linked_trips'] = len(L)
info['segments_merged'] = int(cont.sum())
info['linked_still_ending_in_transfer'] = int((L.WHYTO == 7).sum())

HOME = [1, 2]  # regular home activities, work from home (both at home)
hr = L.STRTTIME.str[:2].astype(int)
L['PER'] = np.select([(hr >= 6) & (hr < 10), (hr >= 10) & (hr < 16), (hr >= 16) & (hr < 20), (hr >= 20)], ['6-10', '10-16', '16-20', '20-24'], '0-6')
PERIODS = ['0-6', '6-10', '10-16', '16-20', '20-24']

def kind(day):
    return np.where(day == 1, 'Sun', np.where(day == 7, 'Sat', 'Wkday'))
L['K'] = kind(L.TRAVDAY.values)
per['K'] = kind(per.TRAVDAY.values)

PURP = {
    'work': lambda w, a: w == 3,
    'errands': lambda w, a: w.isin([11, 12, 14]),
    'social': lambda w, a: w.isin([13, 15, 16, 17, 19]),
    'school': lambda w, a: w.isin([8, 9]) & (a < 18),
    'university': lambda w, a: (w == 8) & (a >= 18),
    'medical': lambda w, a: w == 18,
}
fromhome = L.WHYFROM.isin(HOME)
tohome = L.WHYTO.isin(HOME)
sel = {}
for p, f in PURP.items():
    to_act = f(L.WHYTO, L.AGE)
    from_act = f(L.WHYFROM, L.AGE)
    sel[(p, 'out_hb')] = to_act & fromhome
    sel[(p, 'out_nhb')] = to_act & ~fromhome
    sel[(p, 'out')] = to_act                      # headline outward: every trip arriving at the activity
    sel[(p, 'ret')] = from_act & tohome           # return: activity -> home only
    sel[(p, 'nhb_from')] = from_act & ~tohome     # leaving the activity for somewhere not home (not counted as return)
long_ = L.MI >= 50
sel[('long', 'all')] = long_
sel[('long', 'out')] = long_ & ~tohome
sel[('long', 'ret')] = long_ & tohome
sel[('all', 'all')] = pd.Series(True, index=L.index)

# denominators
KINDS = {'Wkday': ['Wkday'], 'Sat': ['Sat'], 'Sun': ['Sun'], 'Wkend': ['Sat', 'Sun']}
den = {k: per.loc[per.K.isin(v), 'WTPERFIN'].sum() for k, v in KINDS.items()}
den_n = {k: int(per.K.isin(v).sum()) for k, v in KINDS.items()}
# alt denominator for WTTRDFIN: annual person-days of that kind = sum WTPERFIN(kind) * 365
res = {'info': info, 'persons_by_kind': den_n, 'cells': {}}
for (p, dirn), m in sel.items():
    cell = {}
    for k, v in KINDS.items():
        mm = m & L.K.isin(v)
        sub = L[mm]
        rate = sub.WTPERFIN.sum() / den[k]
        rate_tw = sub.WTTRDFIN.sum() / (den[k] * 365)
        sh = sub.groupby('PER').WTPERFIN.sum().reindex(PERIODS).fillna(0)
        sh = (100 * sh / sh.sum()).round(1) if sh.sum() > 0 else sh
        cell[k] = {'rate': round(float(rate), 4), 'rate_tripweight': round(float(rate_tw), 4), 'n': int(mm.sum()),
                   'shares': {q: float(sh[q]) for q in PERIODS}}
    for k in ['Sat', 'Sun', 'Wkend']:
        cell['ratio_' + k] = round(cell[k]['rate'] / cell['Wkday']['rate'], 3) if cell['Wkday']['rate'] else None
    res['cells'][p + '/' + dirn] = cell

json.dump(res, open(out, 'w'), indent=1)

# print report
print(json.dumps(info, indent=1, default=str))
print('persons by kind', den_n, {k: round(v / 1e6, 1) for k, v in den.items()})
print('\nRATES (trips/person/day, person-weighted) [n unweighted]; ratios vs weekday')
print(f"{'cell':24s}{'Wkday':>16s}{'Sat':>16s}{'Sun':>16s}{'Wkend':>16s}{'Sat/Wd':>8s}{'Sun/Wd':>8s}{'We/Wd':>8s}  trip-wt We/Wd")
for c, v in res['cells'].items():
    s = ''.join(f"{v[k]['rate']:9.4f}[{v[k]['n']:5d}]" for k in KINDS)
    twr = v['Wkend']['rate_tripweight'] / v['Wkday']['rate_tripweight'] if v['Wkday']['rate_tripweight'] else 0
    print(f"{c:24s}{s}{v['ratio_Sat']:8.3f}{v['ratio_Sun']:8.3f}{v['ratio_Wkend']:8.3f}  {twr:.3f}")
print('\nPERIOD SHARES % (0-6, 6-10, 10-16, 16-20, 20-24)')
for c, v in res['cells'].items():
    if c.split('/')[1] in ('out_hb', 'out_nhb', 'nhb_from'):
        continue
    for k in KINDS:
        sh = v[k]['shares']
        flag = ' LOW-N' if v[k]['n'] < 200 else ''
        print(f"{c:18s}{k:6s} " + ' '.join(f"{sh[q]:5.1f}" for q in PERIODS) + f"  n={v[k]['n']}{flag}")
