"""T-196: where Americans' leisure outings go, from the American Time Use Survey 2023 microdata
(BLS, public domain): https://www.bls.gov/tus/datafiles/atusact-2023.zip and atusresp-2023.zip,
cached in data/raw/t196_atus/. Read-only.

    ..\\venv\\Scripts\\python.exe trainworld/notes/t196_atus.py

An outing is a travel episode whose purpose (ATUS tier 2 under 18, travel) is eating and drinking
(1811), socializing, relaxing and leisure (1812) or sports, exercise and recreation (1813); its
destination is where the next episode happens (TEWHERE). Weighted by TUFINLWGT (person-days), so
the shares are of all such trips made in 2023 by people 15 and over. Trips home are left out.
"""
import os
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(HERE), "data", "raw", "t196_atus")

WHERE = {1: "own home or yard", 2: "workplace", 3: "someone else's home", 4: "restaurant or bar", 5: "place of worship",
         6: "grocery store", 7: "other store or mall", 8: "school", 9: "outdoors away from home", 10: "library",
         11: "other place", 30: "bank", 31: "gym or health club", 32: "post office", 89: "unspecified place"}
PURPOSE = {11: "eating and drinking", 12: "socializing, relaxing, leisure", 13: "sports, exercise, recreation"}
# the game's three social/leisure destination kinds
KIND = {3: "friends and family (population)", 9: "outdoors (parks)"}


def read(name):
    with zipfile.ZipFile(os.path.join(RAW, name + ".zip")) as z:
        dat = [n for n in z.namelist() if n.lower().endswith(".dat")][0]
        with z.open(dat) as f:
            return pd.read_csv(f)


def main():
    act = read("atusact-2023")
    resp = read("atusresp-2023")[["TUCASEID", "TUFINLWGT"]]
    act = act.sort_values(["TUCASEID", "TUACTIVITY_N"]).reset_index(drop=True)
    tier1 = act["TRCODE"] // 10000
    tier2 = (act["TRCODE"] // 100) % 100
    # the destination is the first following episode that is not travel (a trip with a change of
    # mode is several travel episodes in a row)
    act = act.assign(tier1=tier1, tier2=tier2)
    stay = act["TEWHERE"].where(act.tier1 != 18)
    act["dest"] = stay.groupby(act["TUCASEID"]).bfill().groupby(act["TUCASEID"]).shift(-1)
    act["dest_code"] = act["TRCODE"].where(act.tier1 != 18).groupby(act["TUCASEID"]).bfill().groupby(act["TUCASEID"]).shift(-1)
    trips = act[(act.tier1 == 18) & act.tier2.isin(list(PURPOSE)) & act.dest.notna()]
    trips = trips[trips.dest != 1].merge(resp, on="TUCASEID")
    unk = trips[trips.dest < 0]
    print(f"destination where not recorded (TEWHERE < 0): {unk.TUFINLWGT.sum() / trips.TUFINLWGT.sum():.1%} of outings; "
          f"their activity codes: {unk.dest_code.astype(int).value_counts().head(6).to_dict()}; left out")
    trips = trips[trips.dest > 0]
    trips["dest"] = trips["dest"].astype(int)
    trips["kind"] = trips["dest"].map(KIND).fillna("other places (leisure jobs)")
    w = trips["TUFINLWGT"]
    print(f"{len(trips):,} outings sampled from {trips.TUCASEID.nunique():,} respondents\n")
    print("Destination of leisure outings (eating out, socializing/leisure, sports/recreation), weighted share:")
    by = trips.groupby("dest")["TUFINLWGT"].sum() / w.sum()
    for d, s in by.sort_values(ascending=False).items():
        print(f"  {WHERE.get(d, d):<28} {s:6.1%}")
    print("\nBy the game's kinds:")
    for k, s in (trips.groupby("kind")["TUFINLWGT"].sum() / w.sum()).sort_values(ascending=False).items():
        print(f"  {k:<36} {s:6.1%}")
    print("\nBy purpose, share going outdoors / to someone's home / elsewhere:")
    for p, g in trips.groupby("tier2"):
        gw = g["TUFINLWGT"].sum()
        o = g.loc[g.dest == 9, "TUFINLWGT"].sum() / gw
        h = g.loc[g.dest == 3, "TUFINLWGT"].sum() / gw
        print(f"  {PURPOSE[p]:<32} {gw / w.sum():6.1%} of outings: outdoors {o:5.1%}, someone's home {h:5.1%}, elsewhere {1 - o - h:5.1%}")
    # sampling spread: bootstrap over respondents
    rng = np.random.default_rng(1)
    ids = trips.TUCASEID.unique()
    g = trips.groupby("TUCASEID")
    out_w = g.apply(lambda x: x.loc[x.dest == 9, "TUFINLWGT"].sum()).reindex(ids).to_numpy()
    all_w = g["TUFINLWGT"].sum().reindex(ids).to_numpy()
    bs = [out_w[i].sum() / all_w[i].sum() for i in (rng.integers(0, len(ids), len(ids)) for _ in range(200))]
    print(f"\noutdoors share 95% interval (bootstrap over respondents): {np.percentile(bs, 2.5):.1%} - {np.percentile(bs, 97.5):.1%}")


if __name__ == "__main__":
    main()
