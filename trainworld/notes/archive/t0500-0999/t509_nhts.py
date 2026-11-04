"""NHTS 2017 weekday trips per person by the game's purposes, and trip lengths by purpose.

Convention (T-509 stage 4): the game's outing rate for a purpose = legs with that purpose / 2,
where a trip home takes the purpose of the activity it leaves (NHTS WHYTRP1S does this), so legs
are kept. Per person of all ages: NHTS persons are 5+, so x (share 5+) from ACS.
"""
import io
import sys
import zipfile

import numpy as np
import pandas as pd

Z = r"C:\Users\anita\projects\maps\trainworld\data\raw\nhts2017\csv.zip"
zf = zipfile.ZipFile(Z)
if len(sys.argv) > 1 and sys.argv[1] == "ls":
    print(zf.namelist())
    sys.exit()

def read(name, cols):
    with zf.open(name) as f:
        return pd.read_csv(f, usecols=cols)

per = read("perpub.csv", ["HOUSEID", "PERSONID", "WTPERFIN", "TRAVDAY", "HH_CBSA", "MSASIZE", "URBRUR"])
trp = read("trippub.csv", ["HOUSEID", "PERSONID", "WTTRDFIN", "TRAVDAY", "WHYTRP1S", "WHYFROM", "WHYTO",
                           "TRPMILES", "HH_CBSA", "TRPTRANS", "HHSTFIPS"])
# weekday = TRAVDAY 2-6
wd_p = per[per.TRAVDAY.between(2, 6)]
wd_t = trp[trp.TRAVDAY.between(2, 6)]

# game purposes from WHYTRP1S (1 home, 10 work, 20 school/daycare/religious, 30 medical, 40 shopping/errands,
# 50 social/recreational, 70 transport someone, 80 meals, 97 other). WHYTO details for 20: 08 school as a
# student, 09 child care, 10 adult care, 19 religious/community. NHTS 2017 WHYTO codes: 01 home 02 work from
# home 03 work 04 work-related 05 volunteer 06 drop off/pick up 07 change mode 08 school as student 09 child
# care 10 adult care 11 buy goods 12 buy services 13 buy meals 14 other general errands 15 recreational 16
# exercise 17 visit friends 18 health care 19 religious/community 97 other.
def game(row_why, to, frm):
    # the activity end: WHYTO unless it is home, then WHYFROM
    act = np.where(np.isin(to, [1, 2]), frm, to)
    out = np.full(len(act), "none", dtype=object)
    out[np.isin(act, [11, 12, 14, 6])] = "errands"
    out[np.isin(act, [13, 15, 16, 17, 19])] = "social"
    out[act == 18] = "medical"
    out[np.isin(act, [3, 4, 5])] = "work"
    out[np.isin(act, [8, 9, 10])] = "school"
    out[np.isin(act, [1, 2])] = "home-home"
    out[np.isin(act, [7, 97])] = "other"
    return out

def rates(p, t, label):
    t = t.copy()
    t["g"] = game(t.WHYTRP1S.values, t.WHYTO.values, t.WHYFROM.values)
    # weights: WTTRDFIN is annual trips (x365 days); WTPERFIN persons. Weekday per person per day:
    # sum trip weights / (sum person weights x 365 x 5/7)... simpler: unweighted by day, weighted by person
    # weekday persons' weights sum to the population x 5/7 and trip weights are annual, so
    # person-weekdays a year = the weekday persons' weight x 365
    days = p.WTPERFIN.sum() * 365
    print(f"== {label}: {len(p)} persons, {len(t)} trips, persons weight {p.WTPERFIN.sum()/1e6:.1f}M")
    tot = t.WTTRDFIN.sum() / days
    print(f"all trips per person 5+ per weekday {tot:.3f}")
    out = {}
    for g, d in t.groupby("g"):
        legs = d.WTTRDFIN.sum() / days
        ok = (d.TRPMILES >= 0) & (d.TRPMILES * 1.609 <= 60)  # network km; the local layer tapers past 40-80 km straight
        sl = d.TRPMILES[ok] * 1.609 / 1.3  # straight line at the game's road detour
        w = d.WTTRDFIN[ok]
        km_mean = (sl * w).sum() / w.sum()
        km_med = np.interp(0.5, np.cumsum(w.values[np.argsort(sl.values)]) / w.sum(), np.sort(sl.values))
        rail = d.WTTRDFIN[d.TRPTRANS.isin([15, 16])].sum() / d.WTTRDFIN.sum()
        walk = d.WTTRDFIN[d.TRPTRANS == 1].sum() / d.WTTRDFIN.sum()
        out[g] = (legs, km_mean, km_med)
        print(f"  {g:10s} legs {legs:.3f}  outings {legs/2:.3f}  straight km under 46: mean {km_mean:5.2f} median {km_med:5.2f}"
              f"  rail {rail:5.1%} walk {walk:5.1%}")
    return out

rates(wd_p, wd_t, "US weekday")
# New York-Newark-Jersey City CBSA 35620
rates(wd_p[wd_p.HH_CBSA.astype(str) == "35620"], wd_t[wd_t.HH_CBSA.astype(str) == "35620"], "NY CBSA weekday")
