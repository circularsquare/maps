"""LODES OD 2023 (Pennsylvania, main file) against ACS for the Cambria ring round Pittsburgh: shares of the
zone's resident jobs worked inside the zone and in Pittsburgh's counties, all and county leavers."""
import os
import pickle
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8")
D = pickle.load(open(os.path.join(HERE, "us_data.pkl"), "rb"))
R = pickle.load(open(os.path.join(HERE, "us_out.pkl"), "rb"))
zone = set(R["before"]["groups"]["cmb"])
pit = set(next(r["units"] for r in D["rows"] if r["name"] == "Pittsburgh"))
od = pd.read_csv(r"C:\Users\anita\projects\maps\trainworld\data\raw\pa_od_main_JT00_2023.csv.gz",
                 usecols=["w_geocode", "h_geocode", "S000"], dtype={"w_geocode": str, "h_geocode": str})
od["h"] = od.h_geocode.str[:5]
od["w"] = od.w_geocode.str[:5]
od = od[od.h.isin(zone)]
g = od.groupby(["h", "w"]).S000.sum().reset_index()
tot = g.S000.sum()
inside = g[g.w.isin(zone)].S000.sum()
metro = g[g.w.isin(pit)].S000.sum()
x = g[g.h != g.w]
xt = x.S000.sum()
print(f"LODES (PA main): {tot:,} jobs; inside {inside / tot:.0%}, Pittsburgh {metro / tot:.0%}; "
      f"leaving their county: inside {x[x.w.isin(zone)].S000.sum() / xt:.0%}, Pittsburgh {x[x.w.isin(pit)].S000.sum() / xt:.0%}")
s = R["before"]["sc"]["cmb"]
print(f"ACS: inside {s['within']:.0%}, Pittsburgh {s['to_metro']:.0%}; leaving their county: inside {s['x_within']:.0%}, "
      f"Pittsburgh {s['x_metro']:.0%}")
print("zone states:", sorted({U[:2] for U in zone}))
