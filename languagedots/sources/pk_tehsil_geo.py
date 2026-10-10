"""Pakistan's placement layer at tehsil grain -> data/geo/pk/pk_hexes.gpkg (the place layer)
and data/geo/pk/pk_tehsil_units.csv (Table 11 tehsil id -> drawn unit).

    python sources/pk_tehsil_geo.py --fetch     Karachi's 21 towns from polygons.openstreetmap.fr, then build
    python sources/pk_tehsil_geo.py             rebuild

Run after sources/pk_t11.py (tehsil rows) and sources/pk_north_geo.py (district hexes with
Gilgit-Baltistan, pk_district_hexes.gpkg). Gilgit-Baltistan and Azad Kashmir keep their district
units untouched; only the 136 Table 11 districts are re-keyed.

THE POLYGONS. The 2023 census has 591 tehsils (tehsils, sub-divisions, talukas, sub-tehsils).
No boundary file has them all: COD-AB v01 ADM3 (valid_on 2022-09-09, religiondots' copy, read
in place) has 521 in the four provinces and Islamabad, and many 2023 units are newer (Balochistan
added dozens of sub-tehsils; Lahore went from 2 tehsils to 5) or older (COD splits Upper Dir
into 7 where the census has 4). Karachi's COD ADM3 is the 2001 towns, so Karachi's polygons are
OpenStreetMap's admin_level=7 towns (2023 layout; ODbL), as religiondots took its districts.
religiondots' districts are COD tehsils dissolved, so every COD tehsil falls in exactly one
census district (asserted, by a point inside it).

THE JOIN, inside each district: census tehsil names against polygon names (folded; a substring
or a SequenceMatcher ratio of 0.75 or more), globally greedy by score so each side is used once,
plus ALIAS for renamings no fold catches. Then every hex of religiondots' district layer goes to
the polygon its centroid falls in, inside its own district (nearest polygon of the district for
a hex that falls outside them), and the polygon's Kontur people are compared with its census
tehsil, normalised by the district's own ratio.

THE GROUPING. A tehsil is drawn on its own polygon when its pair's Kontur/census is within a
factor of BAND of the district's own ratio, and, in a district that has gained tehsils since
COD's vintage, when the polygon's area is also within AREA_BAND of Table 1's printed area (a
parent that lost a sub-tehsil keeps its name: Lahore City is 214 km2 in the census and 670 in
COD). The rest of the district (unpaired tehsils, unpaired polygons and failed pairs) is one
unit, which takes further pairs, best first, until its own Kontur ratio is within BAND; when no
pair is left it is the whole district. BAND (1.5) was set before reading the numbers; the first
run applied it only where names did not pair off, and 35 pairs in one-to-one districts were then
out of band (Kohat's Gumbat 0.08 beside Lachi 1.79, Hyderabad's three urban talukas), so it
applies everywhere. AREA_BAND (2) is wider because Table 1's areas are loose (pairs in one-to-one
districts: median 1.00, 5-95% 0.67-1.63, Balochistan worst). Paired names are the key; Kontur
and area only decide what is grouped, never what is paired.

CHECKS: every census tehsil in exactly one unit; units add up to Table 11's districts; every unit
has a populated hex; the singleton units' Kontur against census beats the same with the census
populations shuffled WITHIN each district (the null that a right district with wrong tehsils
would pass).
"""
import difflib
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD, RD_GEO  # noqa: E402

NORM = HERE / "data" / "normalized" / "pk.csv"
DIST_HEXES = HERE / "data" / "geo" / "pk" / "pk_district_hexes.gpkg"
OUT = HERE / "data" / "geo" / "pk" / "pk_hexes.gpkg"
OUT_LUT = HERE / "data" / "geo" / "pk" / "pk_tehsil_units.csv"
OUT_POLYS = HERE / "data" / "geo" / "pk" / "pk_tehsil_polygons.gpkg"
RD_DISTRICTS = RD_GEO / "pk2023" / "pk_districts.gpkg"
COD_A3 = RD / "data" / "raw" / "pk2023" / "cod" / "pak_admin3.shp"
RD_OSM_DISTRICTS = RD / "data" / "raw" / "pk2023" / "osm_karachi_districts.json"
OSM_TOWNS = HERE / "data" / "raw" / "pk" / "osm_karachi_towns.geojson"

BAND = 1.5
AREA_BAND = 2.0
COD_A3_IN_SCOPE = 521          # COD ADM3 in Punjab, Sindh, KP, Balochistan, Islamabad
CENSUS_TEHSILS = 591
UTM = 32642

# Karachi's 2023 towns, OSM admin_level=7 (relation ids from religiondots'
# osm_karachi_admin.json); the district each belongs to is checked against the subarea
# members of religiondots' seven district relations.
KARACHI_TOWNS = {
    "Baldia Town": 16347666, "Bin Qasim Town": 16351914, "Gadap Town": 16351915,
    "Gulberg Town": 16349280, "Gulshan-e-Iqbal Town": 16350240, "Jamshed Town": 16350241,
    "Keamari Town": 16351020, "Korangi Town": 16350630, "Landhi Town": 16350631,
    "Liaquatabad Town": 16349279, "Lyari Town": 16350833, "Malir Town": 16351913,
    "Manghopir Town": 16351171, "Mauripur Town": 16351021, "Mominabad Town": 16351170,
    "New Karachi Town": 16349277, "North Nazimabad Town": 16349278, "Orangi Town": 16347665,
    "SITE Town": 16347664, "Saddar Town": 16350835, "Shah Faisal Town": 16350629,
}
# religiondots' sources/pk_2023_geo.py OSM_KARACHI: census district -> OSM relation
OSM_KARACHI_DISTRICTS = {
    "KARACHI WEST DISTRICT": 16347667, "KARACHI CENTRAL DISTRICT": 16349281,
    "KARACHI EAST DISTRICT": 16350242, "KORANGI DISTRICT": 16350632,
    "KARACHI SOUTH DISTRICT": 16350836, "KEAMARI DISTRICT": 16351022, "MALIR DISTRICT": 16351916,
}

# (district id, census tehsil id's last part) -> polygon name, for renamings no fold catches.
# Each with its reason.
ALIAS = {
    # Bori is Loralai tehsil's own name (Loralai town is its seat)
    ("PK23-balochistan/loralai-district", "bori-sub-division"): "Loralai",
    # Golarchi taluka was renamed Shaheed Fazil Rahu; the census prints both
    ("PK23-sindh/badin-district", "golarchi-s-f-rahu-taluka"): "Shaheed Fazil Rahu",
    # Mirwah taluka's seat is Thari Mirwah
    ("PK23-sindh/khairpur-district", "mirwah-taluka"): "Thari Meer Wah",
    # Table 9 and 11 print "AI"; Allai is Batagram's other tehsil, the census area agrees
    # (804 km2 against COD's 971)
    ("PK23-khyber-pakhtunkhwa/batagram-district", "ai-tehsil"): "Allai",
    # religiondots' TEHSIL_NAMES_DIFFER: the census names the sub-divisions, COD the seats
    ("PK23-khyber-pakhtunkhwa/malakand-protected-area", "sam-rani-zai-sub-division"): "Dargai",
    ("PK23-khyber-pakhtunkhwa/malakand-protected-area", "swat-rani-zai-sub-division"): "Bat Khela",
    # the de-excluded (former tribal) areas of the two districts
    ("PK23-punjab/dera-ghazi-khan-district", "koh-e-suleman-tehsil"): "D.G Khan (Tribal Area)",
    ("PK23-punjab/rajanpur-district", "de-excluded-area-rajanpur"): "Rajanpur (Tribal Area)",
}

_WORDS = re.compile(r"\b(SUB-DIVISION|SUB DIVISION|SUB-TEHSIL|TEHSIL|TALUKA|TOWN)\b", re.I)


def fold(s):
    return re.sub(r"[^a-z]", "", _WORDS.sub(" ", str(s)).lower())


def score(a, b):
    a, b = fold(a), fold(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if len(a) >= 4 and len(b) >= 4 and (a in b or b in a):
        return 0.9
    return difflib.SequenceMatcher(None, a, b).ratio()


def fetch():
    import requests
    OSM_TOWNS.parent.mkdir(parents=True, exist_ok=True)
    # [[reference_overpass_user_agent]]: a project UA, no personal details
    # Overpass answered 504/500 on 2026-10-07 from three endpoints; polygons.openstreetmap.fr
    # serves one relation's polygon at a time, which is all this needs. The subarea witness
    # still reads religiondots' Overpass copy of the district relations.
    feats = []
    cache = OSM_TOWNS.with_suffix("")
    cache.mkdir(parents=True, exist_ok=True)
    ua = {"User-Agent": "languagedots-map-research/1.0"}
    for name, rid in KARACHI_TOWNS.items():
        f = cache / f"{rid}.geojson"
        for attempt in range(4):
            if f.exists() and f.stat().st_size > 100:
                break
            try:
                # the index page makes the server build a relation it has not built yet
                requests.get(f"https://polygons.openstreetmap.fr/index.py?id={rid}", timeout=300,
                             headers=ua)
                r = requests.get(f"https://polygons.openstreetmap.fr/get_geojson.py?id={rid}"
                                 f"&params=0", timeout=300, headers=ua)
                if r.ok and r.content.lstrip().startswith(b"{"):
                    f.write_bytes(r.content)
            except requests.RequestException as e:
                print(f"  {name} ({rid}) attempt {attempt + 1}: {e}")
        if not f.exists():
            raise SystemExit(f"could not fetch {name} ({rid})")
        g = json.loads(f.read_text(encoding="utf-8"))
        if g.get("type") not in ("Polygon", "MultiPolygon", "GeometryCollection"):
            raise SystemExit(f"{name} ({rid}): {r.content[:120]!r}")
        feats.append({"type": "Feature", "properties": {"pname": name, "rid": rid}, "geometry": g})
    OSM_TOWNS.write_text(json.dumps({"type": "FeatureCollection", "features": feats}),
                         encoding="utf-8")
    print(f"wrote {OSM_TOWNS} ({len(feats)} towns)")


def census():
    df = pd.read_csv(NORM)
    if set(df["geo_level"]) != {"tehsil"}:
        raise SystemExit(f"{NORM} is not at tehsil grain; run sources/pk_t11.py")
    t = df.groupby(["geo_id", "geo_name"], as_index=False)["count"].sum()
    t["district"] = t["geo_id"].str.rsplit("/", n=1).str[0]
    t["slug"] = t["geo_id"].str.rsplit("/", n=1).str[1]
    if len(t) != CENSUS_TEHSILS or not t["geo_id"].is_unique:
        raise SystemExit(f"{len(t)} census tehsils, expected {CENSUS_TEHSILS}")
    # Table 1's printed area per tehsil, keyed by district, name and total population
    sys.path.insert(0, str(HERE / "sources"))
    import pk_t11
    a = pk_t11.read_table("TABLE_01")
    a = a[(a["REGION"] == "OVERALL") & a["TEHSIL"].notna()].copy()
    known = a.dropna(subset=["PROVINCE"]).drop_duplicates("DISTRICT").set_index("DISTRICT")["PROVINCE"]
    known = {**{"TANDO ALLAHYAR": "SINDH"}, **known.to_dict()}
    a["PROVINCE"] = a["PROVINCE"].astype(object).where(a["PROVINCE"].notna(), a["DISTRICT"].map(known))
    a["district"] = [f"PK23-{pk_t11.PROVINCE[p]}/"
                     f"{pk_t11.ALIAS.get(pk_t11.slug(d), pk_t11.slug(d) + '-district')}"
                     for p, d in zip(a["PROVINCE"], a["DISTRICT"])]
    # Table 1 counts everyone, Table 11 leaves out people counted by head only, so the
    # population is only a tie-break between same-named tehsils (Rajanpur's two)
    area = []
    for d, n, c in zip(t["district"], t["geo_name"], t["count"]):
        hit = a[(a["district"] == d) & (a["TEHSIL"].astype(str) == n)]
        if hit.empty:
            area.append(np.nan)
            continue
        i = (hit["ALL_SEXES"].fillna(0) - c).abs().idxmin()
        area.append(float(hit.loc[i, "AREA_SQKM"]))
    t["area"] = area
    print(f"  Table 1 area found for {int(t['area'].notna().sum())} of {len(t)} tehsils "
          f"(on district and name); missing: {t.loc[t['area'].isna(), 'geo_id'].tolist()}")
    return t


def polygons(dists):
    """COD ADM3 outside Karachi, OSM towns inside it; each with its census district."""
    a3 = gpd.read_file(COD_A3)
    a3 = a3[a3["adm1_name"].isin(["Punjab", "Sindh", "Khyber Pakhtunkhwa", "Balochistan",
                                  "Islamabad"])].copy()
    if len(a3) != COD_A3_IN_SCOPE:
        raise SystemExit(f"COD ADM3 in scope: {len(a3)}, expected {COD_A3_IN_SCOPE}")
    karachi = set(dists.loc[dists["built_from"].str.startswith("OSM"), "unit"])
    if len(karachi) != 7:
        raise SystemExit(f"religiondots has {len(karachi)} OSM-built Karachi districts, expected 7")
    pt = a3.to_crs(UTM).representative_point().to_crs(4326)
    j = gpd.sjoin(gpd.GeoDataFrame(geometry=pt, crs=4326), dists[["unit", "geometry"]],
                  how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(a3.index)
    if j["unit"].isna().any():
        raise SystemExit(f"COD tehsils in no census district: "
                         f"{a3.loc[j['unit'].isna(), 'adm3_name'].tolist()}")
    a3["district"] = j["unit"]
    cod = a3[~a3["district"].isin(karachi)]
    print(f"  COD ADM3: {len(a3)} tehsils, each inside one census district; "
          f"{len(a3) - len(cod)} in Karachi set aside (2001 towns)")
    cod = gpd.GeoDataFrame({"district": cod["district"], "pname": cod["adm3_name"],
                            "source": "COD-AB v01 ADM3"}, geometry=cod.geometry, crs=a3.crs)

    if not OSM_TOWNS.exists():
        raise SystemExit(f"missing {OSM_TOWNS}: run with --fetch")
    towns = gpd.read_file(OSM_TOWNS)
    if sorted(towns["rid"].astype(int)) != sorted(KARACHI_TOWNS.values()) or \
            towns.geometry.is_empty.any():
        raise SystemExit(f"{OSM_TOWNS} does not hold the {len(KARACHI_TOWNS)} towns")
    towns["rid"] = towns["rid"].astype(int)
    towns = towns.set_crs(4326, allow_override=True)
    kd = dists[dists["unit"].isin(karachi)]
    # clip to religiondots' Karachi districts (OSM's coastal units run out to sea)
    towns = gpd.overlay(towns, gpd.GeoDataFrame(geometry=[kd.union_all()], crs=4326),
                        how="intersection")
    tp = towns.to_crs(UTM).representative_point().to_crs(4326)
    tj = gpd.sjoin(gpd.GeoDataFrame(geometry=tp, crs=4326), kd[["unit", "geo_name", "geometry"]],
                   how="left", predicate="within")
    tj = tj[~tj.index.duplicated(keep="first")].reindex(towns.index)
    towns["district"] = tj["unit"]
    # witness: OSM's own district relations list their towns as subarea members
    rdj = json.load(open(RD_OSM_DISTRICTS, encoding="utf-8"))
    sub = {}
    for e in rdj["elements"]:
        for m in e.get("members", []):
            if m["type"] == "relation":
                sub[m["ref"]] = e["id"]
    name_of = dict(zip(kd["geo_name"], kd["unit"]))
    rel_unit = {rid: name_of[n] for n, rid in OSM_KARACHI_DISTRICTS.items()}
    bad = [(r.pname, r.district, rel_unit.get(sub.get(r.rid))) for r in towns.itertuples()
           if rel_unit.get(sub.get(r.rid)) != r.district]
    if bad or towns["district"].isna().any() or set(towns["district"]) != karachi:
        raise SystemExit(f"Karachi towns -> districts disagree with OSM's subareas: {bad}")
    print(f"  OSM Karachi: {len(towns)} towns, each in the district whose OSM relation lists it "
          f"as a subarea; all 7 districts have towns")
    towns["source"] = "OSM admin_level=7, clipped to religiondots' Karachi"
    polys = pd.concat([cod, towns[["district", "pname", "source", "geometry"]]], ignore_index=True)
    polys = gpd.GeoDataFrame(polys, geometry="geometry", crs=4326)
    missing = sorted(set(dists["unit"]) - set(polys["district"]))
    if missing:
        raise SystemExit(f"census districts with no polygon: {missing}")
    polys["pid"] = np.arange(len(polys))
    polys["km2"] = polys.to_crs("ESRI:54034").area / 1e6
    return polys


def rekey(hexes, polys):
    """Each hex to a polygon of its own district: centroid within, else nearest."""
    cen = gpd.GeoDataFrame(geometry=hexes.to_crs(UTM).geometry.centroid, crs=UTM).to_crs(4326)
    j = gpd.sjoin(cen, polys[["pid", "district", "geometry"]], how="left", predicate="within")
    j = j[j["district"].to_numpy() == hexes.loc[j.index, "unit"].to_numpy()]
    j = j[~j.index.duplicated(keep="first")]
    pid = pd.Series(j["pid"], index=j.index).reindex(hexes.index)
    left = pid.isna()
    near_pop = 0.0
    if left.any():
        cu = cen.to_crs(UTM)
        pu = polys.to_crs(UTM)
        for d, idx in hexes[left].groupby("unit").groups.items():
            n = gpd.sjoin_nearest(cu.loc[idx], pu[pu["district"] == d][["pid", "geometry"]],
                                  how="left")
            n = n[~n.index.duplicated(keep="first")]
            pid.loc[n.index] = n["pid"]
        near_pop = float(hexes.loc[left, "pop"].sum())
    print(f"  hexes re-keyed: {len(hexes):,}; {int(left.sum()):,} ({near_pop:,.0f} Kontur people, "
          f"{100 * near_pop / hexes['pop'].sum():.2f}%) outside every polygon of their district, "
          f"given the nearest")
    if pid.isna().any():
        raise SystemExit("hexes left with no polygon")
    return pid.astype(int)


def pair_district(d, ct, dp):
    """Greedy name pairing inside one district. Returns {tehsil geo_id: pid}."""
    alias = {}
    for r in ct.itertuples():
        want = ALIAS.get((d, r.slug))
        if want:
            hit = dp[dp["pname"] == want]
            if len(hit) != 1:
                raise SystemExit(f"ALIAS {d} {r.slug} -> {want}: {len(hit)} polygons")
            alias[r.geo_id] = int(hit["pid"].iloc[0])
    cand = []
    for r in ct.itertuples():
        if r.geo_id in alias:
            continue
        for p in dp.itertuples():
            if p.pid in alias.values():
                continue
            s = score(r.geo_name, p.pname)
            if s >= 0.75:
                cand.append((s, r.geo_id, int(p.pid)))
    cand.sort(key=lambda x: (-x[0], x[1], x[2]))
    pairs, used_t, used_p = dict(alias), set(alias), set(alias.values())
    for s, t, p in cand:
        if t in used_t or p in used_p:
            continue
        pairs[t] = p
        used_t.add(t)
        used_p.add(p)
    return pairs


def group_district(d, ct, dp, pairs, kpop):
    """Units for one district: {tehsil geo_id: unit id}, and the pids of each unit."""
    C = dict(zip(ct["geo_id"], ct["count"]))
    K = {int(p): float(kpop.get(p, 0.0)) for p in dp["pid"]}
    dr = sum(K.values()) / sum(C.values())
    lt = set(C) - set(pairs)
    lp = set(K) - set(pairs.values())

    def ratio(ts, ps):
        c = sum(C[t] for t in ts)
        return (sum(K[p] for p in ps) / c / dr) if c else float("inf")

    def inband(x):
        return 1 / BAND <= x <= BAND

    pr = {t: ratio([t], [p]) for t, p in pairs.items()}
    A = dict(zip(ct["geo_id"], ct["area"]))
    PA = dict(zip(dp["pid"].astype(int), dp["km2"]))
    ar = {t: PA[p] / A[t] if A[t] and A[t] > 0 else np.nan for t, p in pairs.items()}

    def area_ok(t):
        # only where the district gained tehsils since COD: a carved-out parent keeps its name
        return not lt or np.isnan(ar[t]) or 1 / AREA_BAND <= ar[t] <= AREA_BAND

    rem_t, rem_p = set(lt), set(lp)
    for t, p in pairs.items():
        if not inband(pr[t]) or not area_ok(t):
            rem_t.add(t)
            rem_p.add(p)
    free = {t: p for t, p in pairs.items() if t not in rem_t}
    if rem_t or rem_p:
        while free and not (rem_t and rem_p and inband(ratio(rem_t, rem_p))):
            best = min(free, key=lambda t: (abs(np.log(max(ratio(rem_t | {t}, rem_p | {free[t]}),
                                                           1e-9))), t))
            rem_t.add(best)
            rem_p.add(free.pop(best))
    groups = [([t], [p]) for t, p in free.items()]
    if rem_t or rem_p:
        groups.append((sorted(rem_t), sorted(rem_p)))
    note = (("names pair off one to one" if not lt and not lp else
             f"{len(lt)} tehsils and {len(lp)} polygons unpaired")
            + f"; {len(free)} drawn alone, {len(rem_t)} grouped")
    out, pid_unit, rows = {}, {}, []
    slugs = dict(zip(ct["geo_id"], ct["slug"]))
    for ts, ps in groups:
        if len(ts) == len(C):
            u = d
        elif len(ts) == 1:
            u = ts[0]
        else:
            u = d + "/" + "+".join(sorted(slugs[t] for t in ts))
        for t in ts:
            out[t] = u
        for p in ps:
            pid_unit[p] = u
        rows.append((u, len(ts), len(ps), sum(C[t] for t in ts), sum(K[p] for p in ps),
                     ratio(ts, ps)))
    return out, pid_unit, rows, note, pr


def main():
    if "--fetch" in sys.argv:
        fetch()
    t = census()
    dists = gpd.read_file(RD_DISTRICTS)
    if set(dists["unit"]) - {u for u in dists["unit"] if "azad" in u} != set(t["district"]):
        raise SystemExit("religiondots' districts differ from Table 11's")
    dists = dists[dists["unit"].isin(set(t["district"]))].copy()
    polys = polygons(dists)

    hexes = gpd.read_file(DIST_HEXES)
    mine = hexes["unit"].isin(set(t["district"]))
    print(f"  district layer: {len(hexes):,} hexes; {int(mine.sum()):,} in Table 11's 136 districts, "
          f"{int((~mine).sum()):,} in Gilgit-Baltistan and Azad Kashmir (kept as they are)")
    h = hexes[mine].copy()
    h["pid"] = rekey(h, polys)
    kpop = h.groupby("pid")["pop"].sum()

    lut, pid_unit, report, oob_bij = {}, {}, [], []
    n_pairs = n_alias = 0
    for d, ct in t.groupby("district"):
        dp = polys[polys["district"] == d]
        pairs = pair_district(d, ct, dp)
        n_pairs += len(pairs)
        n_alias += sum(1 for r in ct.itertuples() if (d, r.slug) in ALIAS)
        o, pu, rows, note, pr = group_district(d, ct, dp, pairs, kpop)
        lut.update(o)
        pid_unit.update(pu)
        report.append((d, len(ct), len(dp), note, rows))
        if note.startswith("names pair"):
            oob_bij += [(t_, round(x, 2)) for t_, x in pr.items() if not 1 / BAND <= x <= BAND]

    # ---- checks
    if set(lut) != set(t["geo_id"]):
        raise SystemExit("a census tehsil is in no unit")
    if set(pid_unit) != set(polys["pid"]):
        raise SystemExit("a polygon is in no unit")
    t["unit"] = t["geo_id"].map(lut)
    cross = t.groupby("unit")["district"].nunique()
    if (cross != 1).any() or not all(u.startswith(d) for u, d in zip(t["unit"], t["district"])):
        raise SystemExit(f"units crossing a district line: {list(cross[cross != 1].index)}")
    if t.groupby("district")["count"].sum().sum() != t["count"].sum():
        raise SystemExit("tehsils do not add up")
    h["unit"] = h["pid"].map(pid_unit)
    per = h.groupby("unit")["pop"].sum()
    cen = t.groupby("unit")["count"].sum()
    empty = sorted(set(cen.index) - set(per.index[per > 0]))
    if empty:
        raise SystemExit(f"units with no populated hex: {empty}")

    n_units = cen.size
    single = [u for u in cen.index if u in set(t["geo_id"])]
    whole = [u for u in cen.index if u in set(t["district"])]
    print(f"\n  {CENSUS_TEHSILS} census tehsils in 136 districts -> {n_units} units: "
          f"{len(single)} tehsils on their own polygon, {n_units - len(single) - len(whole)} "
          f"groups inside a district, {len(whole)} whole districts")
    print(f"  {n_pairs} names paired ({n_alias} through ALIAS); band {BAND}")
    nb = sum(1 for r in report if r[3].startswith("names pair"))
    print(f"  {nb} of 136 districts pair off one to one; pairs there out of band, so grouped "
          f"(Kontur's noise or a line that moved, which nothing here tells apart): {len(oob_bij)}")
    for x in sorted(oob_bij, key=lambda x: x[1]):
        print(f"      {x[1]:5.2f}  {x[0]}")

    # Kontur against census per unit, and the within-district shuffle
    df = pd.DataFrame({"census": cen, "kontur": per.reindex(cen.index)})
    df["district"] = [u if u in set(t["district"]) else "/".join(u.split("/")[:2]) for u in df.index]
    nat = df["kontur"].sum() / df["census"].sum()
    df["r"] = df["kontur"] / df["census"] / nat
    q = df["r"].quantile([.1, .5, .9])
    print(f"\n  Kontur / census nationally {nat:.3f}; per unit normalised p10 {q[.1]:.2f}, "
          f"median {q[.5]:.2f}, p90 {q[.9]:.2f}")
    s = df[df.index.isin(single)]
    lc, lk = np.log(s["census"].to_numpy()), np.log(s["kontur"].to_numpy())
    r0 = np.corrcoef(lc, lk)[0, 1]
    rng = np.random.default_rng(20261007)
    groups = [np.flatnonzero((s["district"] == d).to_numpy()) for d in s["district"].unique()]
    best = 0.0
    for _ in range(500):
        perm = np.arange(len(s))
        for g in groups:
            perm[g] = rng.permutation(g)
        best = max(best, np.corrcoef(lc[perm], lk)[0, 1])
    print(f"  {len(s)} single-tehsil units: log correlation of Kontur with census r = {r0:.3f}; "
          f"census shuffled within each district, best of 500 {best:.3f}")
    if not r0 > best:
        raise SystemExit("the tehsil join carries no information beyond the district")

    # report
    lines = []
    for d, nt, npg, note, rows in report:
        lines.append(f"{d}: {nt} tehsils, {npg} polygons; {note}")
        for u, a, b, c, k, x in rows:
            lines.append(f"    {x:5.2f}  {a} tehsil(s) / {b} polygon(s)  census {c:>10,}  {u}")
    rep = OUT_LUT.with_name("pk_tehsil_report.txt")
    rep.write_text("\n".join(lines), encoding="utf-8")

    rest = hexes[~mine]
    newh = gpd.GeoDataFrame({"unit": h["unit"].to_numpy(), "pop": h["pop"].to_numpy()},
                            geometry=h.geometry.to_numpy(), crs=hexes.crs)
    out = gpd.GeoDataFrame(pd.concat([newh, rest[["unit", "pop", "geometry"]]], ignore_index=True),
                           geometry="geometry", crs=hexes.crs)
    if len(out) != len(hexes) or abs(out["pop"].sum() - hexes["pop"].sum()) > 1:
        raise SystemExit("hex count or population changed in the re-key")
    tmp = OUT.with_suffix(".tmp.gpkg")
    out.to_file(tmp, layer="hexes", driver="GPKG")
    os.replace(tmp, OUT)
    t[["geo_id", "geo_name", "district", "unit", "count"]].rename(
        columns={"count": "census_2023"}).to_csv(OUT_LUT, index=False)
    polys["unit"] = polys["pid"].map(pid_unit)
    polys["kontur"] = polys["pid"].map(kpop).fillna(0)
    polys.drop(columns="pid").to_file(OUT_POLYS, layer="polygons", driver="GPKG")
    print(f"\n  wrote {OUT} ({len(out):,} hexes, {out['unit'].nunique()} units), {OUT_LUT.name}, "
          f"{OUT_POLYS.name}, {rep.name}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
