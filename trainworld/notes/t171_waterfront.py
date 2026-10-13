"""Rail share of commuters, Hudson County groups: ACS 2019-2023 tracts vs the model's day."""
import json, sys, os
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
T = r"C:\Users\anita\projects\maps\trainworld"
sys.path.insert(0, T + r"\notes"); sys.path.insert(0, T + r"\pipeline")
os.environ["OMP_NUM_THREADS"] = "3"
import read_pack, h3
from t171_gravity import groups
S = r"C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\87037fc9-7e8f-4e0f-abce-7678aadf6b68\scratchpad\agent-demand\t171"
b = pd.read_csv(T + r"\data\work\nyc.blocks.csv.gz", dtype={"geoid": str}, usecols=["geoid", "lat", "lon", "pop"])
b["geoid"] = b["geoid"].str.zfill(15)
b = b[b["geoid"].str[:5].isin(["34017", "34013", "36061", "36047", "34003"])].reset_index(drop=True)
b["g"] = groups("nyc", b)
b.loc[b["g"] == "other", "g"] = b["geoid"].str[:5]
b["tract"] = b["geoid"].str[:11]
# tract -> group by population majority
tg = b.groupby(["tract", "g"])["pop"].sum().reset_index().sort_values("pop").drop_duplicates("tract", keep="last").set_index("tract")["g"]
cols = ["GEO_ID", "B08301_E001", "B08301_E011", "B08301_E012", "B08301_E013", "B08301_E014", "B08301_E015", "B08301_E019", "B08301_E021"]
rows = []
for ch in pd.read_csv(T + r"\data\raw\acsdt5y2023-b08301.dat", sep="|", usecols=cols, dtype={"GEO_ID": str}, chunksize=200_000):
    ch = ch[ch["GEO_ID"].str.startswith("1400000US")]
    ch = ch.assign(tract=ch["GEO_ID"].str[9:20])
    rows.append(ch[ch["tract"].isin(tg.index)])
a = pd.concat(rows)
a["g"] = a["tract"].map(tg)
s = a.groupby("g")[cols[1:]].sum()
comm = s["B08301_E001"] - s["B08301_E021"]
acs = pd.DataFrame({"commuters": comm, "rail": (s["B08301_E012"] + s["B08301_E013"] + s["B08301_E014"]) / comm,
                    "subway(PATH)": s["B08301_E012"] / comm, "bus": s["B08301_E011"] / comm, "ferry": s["B08301_E015"] / comm, "walk": s["B08301_E019"] / comm})
# model cells -> nearest block -> group
header, arr = read_pack.load_pack("nyc")
ll = np.array([h3.cell_to_latlng(h3.int_to_str(int(i))) for i in arr["h3"]])
k = np.cos(np.radians(40.75))
tree = cKDTree(np.c_[b["lat"], b["lon"] * k])
dist, nb = tree.query(np.c_[ll[:, 0], ll[:, 1] * k])
cg = b["g"].to_numpy()[nb]
cg[dist > 0.01] = "far"
for f in sys.argv[1:] or [S + r"\day_all2.json"]:
    res = json.load(open(f))
    rail = sum(np.array(p["rail_cell"]) for p in res["periods"])
    trips = sum(np.array(p["trips_cell"]) for p in res["periods"])
    m = pd.DataFrame({"g": cg, "r": rail, "t": trips}).groupby("g").sum()
    acs[os.path.basename(f)] = m["r"] / m["t"]
print(acs.to_string(float_format=lambda v: f"{v:.3f}" if v < 10 else f"{v:,.0f}"))
