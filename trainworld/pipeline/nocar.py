"""T-511: residents without a car, per cell, for the local mode choice (docs/demand.md 4.3).

    python pipeline/nocar.py <id> [<id> ...] | --all      patch these packs
    python pipeline/nocar.py <id> --dry                    the shares only, nothing written

Adds the per-cell array `nocar` (f32, people living in a household with no car) and a `nocar`
header block to a finished pack: first to its res-9 copy in data/work/res9, then the same array
summed into the merged pack in data/packs by merge_cells.py's own rule, after checking that rule
still reproduces every other per-cell array of the live pack. Both writes only append to the .bin
(.bin replaced before .json, each through a temporary file), so a reader never sees a half-written
pack. No other array changes; nothing needs a re-solve.

A cell's share of people without a car comes from the finest open source:

- **US**: ACS 2019-23 5-year B08201 (households by size by vehicles available) by tract: people in
  households with no vehicle over people in households (sizes 1, 2, 3 and 4+ at 4.6 people). A
  cell takes the tract of its nearest populated 2020 block (the pack's blocks CSV).
- **England**: Census 2021 TS045 (car or van availability, households) by LSOA 2021, households
  made people as below.
- **Everywhere else**: cars per household `c` gives the households with no car by
  `h0 = exp(-1.30 c^1.33)`, fitted on English LSOAs and US tracts of 50 households or more
  together (the US alone gives 1.34 and 1.29, England 1.24 and 1.29; rmse 0.04 against 0.11 for
  the Poisson exp(-c)), households weighted. Japan: AIRIA
  private cars per household by prefecture (March 2025). Other countries: passenger cars per
  1,000 people from national registers times the mean household size (COUNTRIES).
- **Households to people**: a household with no car is smaller, 0.693 times the size of one with a
  car (ACS 2019-23 national: 1.74 against 2.51 people), so the people's share is
  `h0 k / (h0 k + 1 - h0)` with k = 0.693 (England 23.5% of households, 17.6% of people).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import json
import sys

import h3
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from read_pack import PACKS, ROOT, load_pack  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw", "t511")
WORK = os.path.join(ROOT, "data", "work")
RES9 = os.path.join(WORK, "res9")
CURVE = (1.30, 1.33)  # h0 = exp(-a c^b)
SIZE_RATIO = 0.693  # people per no-car household over people per car household (ACS 2019-23, US)
ACS_URL = "https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/data/5YRData/acsdt5y2023-b08201.dat"
TS045 = ("Census 2021 TS045 car or van availability by LSOA 2021 (Nomis NM_2063_1), "
         "households made people at a no-car household size ratio of 0.693")
# passenger cars per 1,000 people (national register, year) and people per household
COUNTRIES = {
    "CA": (587, 2.40, "Statistics Canada vehicle registrations 2023, light-duty 23.55M / 40.1M; census 2021 household size"),
    "FR": (577, 2.17, "Eurostat road_eqs_carhab 2024 (provisional); INSEE household size"),
    "DE": (590, 2.03, "Eurostat road_eqs_carhab 2024; Destatis household size 2023"),
    "KR": (421, 2.21, "MOLIT registrations end 2024, passenger cars 21.77M / 51.7M (index.go.kr 1257); census 2024 household size"),
    "CN": (251, 2.62, "Ministry of Public Security 2024, 353M cars (汽车) / 1,408M; census 2020 household size"),
    "IN": (35, 4.44, "MoRTH 2022 cars, jeeps and taxis 49.05M / 1,417M; NFHS-5 household size"),
    "BR": (298, 2.79, "Senatran end 2024, 63.3M automóveis / 212.6M; census 2022 household size"),
    "NG": (24, 4.70, "NBS road transport data 2018, 4.74M private vehicles / 198M; NDHS 2018 household size"),
}
# AIRIA, 令和7年3月末: private passenger cars (registered and kei) per household, prefectures 01-47
AIRIA_JP = [0.986, 1.209, 1.369, 1.234, 1.360, 1.617, 1.510, 1.508, 1.537, 1.558, 0.915, 0.921, 0.405, 0.664,
            1.497, 1.616, 1.450, 1.670, 1.494, 1.532, 1.506, 1.344, 1.209, 1.414, 1.318, 0.784, 0.611, 0.880,
            1.060, 1.221, 1.431, 1.384, 1.331, 1.086, 1.233, 1.337, 1.311, 1.129, 1.129, 1.032, 1.470, 1.099,
            1.283, 1.261, 1.271, 1.183, 1.286]
AIRIA = "AIRIA private passenger cars per household by prefecture, March 2025 (kenbetsu2025.pdf)"
PACK_COUNTRY = {"lon": "GB-ENG", "par": "FR", "ber": "DE", "seo": "KR", "sha": "CN", "mum": "IN", "sao": "BR",
                "lag": "NG", "tor": "CA"}


def households_no_car(c):
    a, b = CURVE
    return np.exp(-a * np.power(np.maximum(c, 0.0), b))


def people_share(h0):
    k = SIZE_RATIO
    return h0 * k / (h0 * k + 1.0 - h0)


def country_of(cid, header):
    if header.get("country") == "JP":
        return "JP"
    if cid in PACK_COUNTRY:
        return PACK_COUNTRY[cid]
    if os.path.exists(os.path.join(WORK, f"{cid}.blocks.csv.gz")):
        return "US"
    sys.exit(f"{cid}: no country for the no-car share")


def cell_lonlat(a):
    ll = np.array([h3.cell_to_latlng(h3.int_to_str(int(c))) for c in a["h3"]])
    return ll[:, 1], ll[:, 0]


def units_with_nearest(lat, lon, units):
    """Index of the unit holding each point; a point in none takes the nearest unit."""
    import shapely
    from shapely import STRtree
    import england
    idx, _ = england.cell_units(lat, lon, units)
    miss = np.nonzero(idx < 0)[0]
    if len(miss):
        tree = STRtree([g for _, g in units])
        idx[miss] = tree.nearest(shapely.points(lon[miss], lat[miss]))
    return idx, len(miss)


def us_tracts():
    """{tract GEOID: (people in no-vehicle households, people in households)}."""
    path = os.path.join(RAW, "acsdt5y2023-b08201.dat")
    if not os.path.exists(path):
        sys.exit(f"download {ACS_URL} to {path}")
    cols = ["GEO_ID"] + [f"B08201_E{i:03d}" for i in (7, 8, 13, 14, 19, 20, 25, 26)]
    t = pd.read_csv(path, sep="|", usecols=cols, dtype={"GEO_ID": str})
    t = t[t.GEO_ID.str.startswith("1400000US")]
    e = lambda i: t[f"B08201_E{i:03d}"].to_numpy(float)  # noqa: E731
    sizes = ((7, 1.0), (13, 2.0), (19, 3.0), (25, 4.6))
    no = sum(e(b + 1) * s for b, s in sizes)
    al = sum(e(b) * s for b, s in sizes)
    return pd.DataFrame({"geoid": t.GEO_ID.str[9:].to_numpy(), "no": no, "all": al})


def share_us(cid, lat, lon):
    from scipy.spatial import cKDTree
    b = pd.read_csv(os.path.join(WORK, f"{cid}.blocks.csv.gz"), dtype={"geoid": str}, usecols=["geoid", "pop", "lat", "lon"])
    b = b[b["pop"] > 0]
    t = us_tracts()
    tract = b.geoid.str[:11]
    by = t.set_index("geoid")
    # CT's 2020 tracts sit in 2022 planning regions in ACS 2023: state + tract code (T-042)
    ct = t[t.geoid.str[:2] == "09"].assign(k=lambda d: d.geoid.str[:2] + d.geoid.str[5:]).drop_duplicates("k").set_index("k")
    no = tract.map(by["no"])
    al = tract.map(by["all"])
    m = no.isna() & (tract.str[:2] == "09")
    k = tract[m].str[:2] + tract[m].str[5:]
    no[m], al[m] = k.map(ct["no"]), k.map(ct["all"])
    # a tract with no figure (renumbered, or no households): its county's
    county = t.assign(c=t.geoid.str[:5]).groupby("c")[["no", "all"]].sum()
    m = no.isna() | ~(al > 0)
    unmatched = float(b["pop"][m].sum())
    no[m], al[m] = tract[m].str[:5].map(county["no"]), tract[m].str[:5].map(county["all"])
    s = (no / al).to_numpy(float)
    s = np.where(np.isfinite(s), s, np.nan)
    mean = np.nansum(s * b["pop"]) / b["pop"][np.isfinite(s)].sum()
    s = np.where(np.isfinite(s), s, mean)
    c = np.cos(np.radians(np.mean(lat)))
    _, i = cKDTree(np.c_[b.lon.to_numpy() * c, b.lat.to_numpy()]).query(np.c_[lon * c, lat])
    src = (f"ACS 2019-23 5-year B08201, people in households with no vehicle by tract (2020 tracts; Connecticut by "
           f"state and tract code), a cell taking its nearest populated 2020 block's tract; {unmatched:,.0f} block "
           f"people in tracts with no figure take their county's")
    return s[i], src


def share_england(lat, lon):
    path = os.path.join(ROOT, "data", "raw", "tier0", "lsoa2021_bgc_lon.json")
    with open(path, encoding="utf-8") as f:
        feats = json.load(f)
    from shapely import make_valid
    from shapely.geometry import shape
    units = [(ft["properties"]["LSOA21CD"], make_valid(shape(ft["geometry"]))) for ft in feats]
    t = pd.read_csv(os.path.join(RAW, "ts045_lsoa2021.csv")).set_index("GEOGRAPHY_CODE")
    h0 = (t.c0 / t["all"]).reindex([u for u, _ in units]).to_numpy(float)
    mean = float(t.c0.sum() / t["all"].sum())
    h0 = np.where(np.isfinite(h0), h0, mean)
    idx, miss = units_with_nearest(lat, lon, units)
    return people_share(h0[idx]), f"{TS045}; {miss} cells in no LSOA take the nearest"


def share_japan(cid, lat, lon):
    import build_city as bc
    import japan
    prefs = bc.CITIES[cid]["prefectures"]
    ms = japan.municipalities(prefs)
    idx, miss = units_with_nearest(lat, lon, [(m[0], m[3]) for m in ms])
    pref = np.array([int(ms[i][0][:2]) for i in idx])
    c = np.array(AIRIA_JP)[pref - 1]
    return people_share(households_no_car(c)), f"{AIRIA}, households with no car by exp(-1.30 c^1.33), made people at 0.693"


def shares(cid, header, a):
    country = country_of(cid, header)
    lon, lat = cell_lonlat(a)
    if country == "US":
        s, src = share_us(cid, lat, lon)
    elif country == "GB-ENG":
        s, src = share_england(lat, lon)
    elif country == "JP":
        s, src = share_japan(cid, lat, lon)
    else:
        v, hh, ref = COUNTRIES[country]
        c = v / 1000.0 * hh
        s = np.full(len(lat), people_share(households_no_car(c)))
        src = (f"{country}: {v} passenger cars per 1,000 people x {hh} people per household = {c:.2f} cars per "
               f"household, households with no car by exp(-1.30 c^1.33), made people at 0.693 ({ref})")
    return country, np.clip(s, 0.0, 1.0), src


def append_array(stem, header, name, values, block):
    """Append (or replace) a per-cell f32 array and set header[name] = block, .bin first."""
    with open(stem + ".bin", "rb") as f:
        buf = bytearray(f.read())
    header = json.loads(json.dumps(header))
    old = [e for e in header["arrays"] if e["name"] == name]
    v = np.ascontiguousarray(values, dtype="<f4").tobytes()
    if old and old[0]["count"] * 4 == len(v):
        buf[old[0]["offset"]:old[0]["offset"] + len(v)] = v
    else:
        header["arrays"] = [e for e in header["arrays"] if e["name"] != name]
        buf += b"\0" * (-len(buf) % 8)
        header["arrays"].append({"name": name, "dtype": "f32", "per": "cell", "offset": len(buf), "count": len(values)})
        buf += v
    header[name] = block
    with open(stem + ".bin.part", "wb") as f:
        f.write(buf)
    with open(stem + ".json.part", "w") as f:
        json.dump(header, f, indent=2)
    os.replace(stem + ".bin.part", stem + ".bin")
    os.replace(stem + ".json.part", stem + ".json")
    return header


def run(cid, dry=False):
    import merge_cells as mc
    h9, a9 = load_pack(cid, RES9)
    live, la = load_pack(cid, PACKS)
    assert "merged" in live and h9["h3_res"] == 9
    if mc.written(cid) != mc.stamp(PACKS, cid):
        sys.exit(f"{cid}: the merged pack changed after merging; merge it again first")
    country, s, src = shares(cid, h9, a9)
    pop = a9["pop"].astype(np.float64)
    nocar = (pop * s).astype(np.float32)
    line = f"{cid}: {country}, {nocar.sum():,.0f} of {pop.sum():,.0f} people without a car ({nocar.sum() / pop.sum():.1%})"
    if dry:
        print(line + f"; cells {s.min():.2f}-{s.max():.2f}", flush=True)
        return
    block = {"task": "T-511", "what": "people living in a household with no car, per cell", "source": src,
             "share": round(float(nocar.sum() / pop.sum()), 4)}
    # the merged values by merge_cells' rule, checked against the live pack first
    cells, _ = mc.merge(dict(h9, arrays=h9["arrays"] + [{"name": "nocar", "dtype": "f32", "per": "cell", "count": h9["cells"]}]),
                        dict(a9, nocar=nocar))
    for spec in live["arrays"]:
        k = spec["name"]
        if spec.get("per") == "cell" and k != "nocar":
            assert np.array_equal(cells[k], la[k]), f"{cid}: merging again changes {k}; merge_cells.py {cid} first"
    append_array(os.path.join(RES9, cid), h9, "nocar", nocar, block)
    live = json.loads(json.dumps(live))
    live["merged"]["source"] = mc.stamp(RES9, cid)
    append_array(os.path.join(PACKS, cid), live, "nocar", cells["nocar"], block)
    with open(os.path.join(RES9, f"{cid}.merged.json"), "w") as f:
        json.dump(mc.stamp(PACKS, cid), f)
    print(line, flush=True)


if __name__ == "__main__":
    import merge_cells as mc
    args = sys.argv[1:]
    ids = mc.registered() if "--all" in args else [x for x in args if not x.startswith("--")]
    if not ids:
        sys.exit(__doc__)
    for cid in ids:
        run(cid, dry="--dry" in args)
