"""Transnistria and Bender, 2015 census, as extra units of `countries/md.py`: the ground Moldova's
2024 census did not enumerate.

    python sources/md_pmr.py --fetch   -> data/raw/md/ (the table, OSM raion polygons)
    python sources/md_pmr.py           -> data/normalized/md_pmr.csv,
                                          data/geo/md/md_pmr_units.gpkg, md_plus_hexes.gpkg

GEOGRAPHY FOLLOWS religiondots: Transnistria stays inside Moldova's entry (religiondots' `md`
leaves it blank; its not_drawn.py names it from Natural Earth's disputed-areas layer). Units
`PMR-*`.

THE TABLE. The 2015 census of the Transnistrian authorities (State Statistics Service, results
published 2016-17), nationality by city and raion, 8 units, read from Tim Bespyatov's
transcription, pop-stat.mashke.org/pmr-ethnic2015.htm (it sums to 475,007 and is asserted to;
the authorities' headline was 475,665, the gap being people without usual residence in a unit).
The census asked native language too, but no table of it by raion was published that could be
found (searched: the ru.wikipedia articles on the census and the population, the statistics
service's site through search engines, pop-stat.mashke.org); the census is read as nationality.

RETENTION (AGENT_BRIEF §2, ethnicity only; every row `derived`): each nationality is spread over
languages in the shares Moldova's own 2024 census gives that nationality nationally (BNS, table
5.33, nationality by mother tongue, data/raw/md/Anexa_Caracteristici_Etnoculturale_RPL2024.xlsx,
the file sources/md_census.py reads). That is the right bank. On the left bank Russian is the
language of 89% of schooling and nearly all official business, so these shares probably overstate
how many Ukrainians, Bulgarians and Moldovans there name their own language; no left-bank source
says by how much. The Moldovenească and Română answers are summed and drawn as Moldovan: the
Transnistrian authorities' official language is "Moldovan" in Cyrillic, and Moldova's split
between the two names is a right-bank question.
  "Transnistrians" (1,013): no language of their own; Russian, the territory's lingua franca.
  "Other": `other`. Undeclared and refused (68,406, 14.4%): not drawn, md's `gap`.

UNITS: OpenStreetMap's relations (polygons.openstreetmap.fr; ids in UNITS, found with Nominatim).
OSM puts Dnestrovsk inside Tiraspol's city council; the census counts it apart, so it is cut out.

PLACEMENT: the Kontur MD hexes that md_hexes.gpkg does not hold (Moldova's census units), each to
the unit its centroid falls in. A Kontur hex already in md_hexes.gpkg stays Moldova's: BNS counted
those villages (Cocieri, Molovata Nouă, Doroţcaia and the other left-bank communes under
Chişinău's administration).
"""
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
RAW = ROOT / "data" / "raw" / "md"
GEO = ROOT / "data" / "geo" / "md"
NORM = ROOT / "data" / "normalized" / "md_pmr.csv"
UA = {"User-Agent": "languagedots-research/1.0"}
URL = "http://pop-stat.mashke.org/pmr-ethnic2015.htm"
FILE = RAW / "mashke_pmr-ethnic2015.htm"
BNS = RAW / "Anexa_Caracteristici_Etnoculturale_RPL2024.xlsx"
POLY_URL = "https://polygons.openstreetmap.fr/get_geojson.py?id={}&params=0"
TOTAL = 475_007

UNITS = {"mun. Bender": ("PMR-BENDER", 944727), "mun. Tiraspol": ("PMR-TIRASPOL", 1702219),
         "or. Dnestrovsc": ("PMR-DNESTROVSK", 8290840), "Grigoriopol": ("PMR-GRIGORIOPOL", 1702214),
         "Dubăsari": ("PMR-DUBOSSARY", 1702215), "Camenca": ("PMR-KAMENKA", 1702216),
         "Rîbnița": ("PMR-RYBNITSA", 1702217), "Slobozia": ("PMR-SLOBODZEYA", 1702218)}
# census column -> BNS 5.33 nationality row (None: a fixed rule)
NAT = {"Russians": "Rus", "Moldovans": "Moldovean", "Ukrainians": "Ucrainean",
       "Bulgarians": "Bulgar", "Gagauzians": "Găgăuz", "Belorussians": "Belorus",
       "Germans": "German / Neamț", "Poles": "Polonez", '"Transnistrians"': None, "Other": None}
NOT_DRAWN = ("Undeclared", "Refused to answer")
# BNS 5.33 language column -> label
LANG = {"Moldovenească sau Română": "Moldovan", "Ucraineană": "Ukrainian", "Rusă": "Russian",
        "Găgăuză": "Gagauz", "Bulgară": "Bulgarian", "Romani (Țigănească)": "Romani",
        "Belorusă": "Belarusian", "Germană": "German", "Poloneză": "Polish"}


def fetch():
    import requests
    if not FILE.exists():
        r = requests.get(URL, headers=UA, timeout=300)
        r.raise_for_status()
        FILE.write_bytes(r.content)
    (RAW / "osm").mkdir(parents=True, exist_ok=True)
    for _, rel in UNITS.values():
        p = RAW / "osm" / f"rel_{rel}.geojson"
        if not p.exists():
            r = requests.get(POLY_URL.format(rel), headers=UA, timeout=300)
            r.raise_for_status()
            json.loads(r.text)
            p.write_text(r.text, encoding="utf-8")
            print(f"  {p.name}: {len(r.text):,} bytes")


def _num(s):
    s = str(s).strip()
    return 0 if s in ("-", "–", "—", "", "…") else int(s)


def read_table():
    t = FILE.read_text(encoding="utf-8-sig")
    rows = []
    for tr in re.split(r"<tr>", t, flags=re.I)[1:]:
        cells = [re.sub(r"<[^>]+>", "", c) for c in re.split(r"<t[dh][^>]*>", tr, flags=re.I)[1:]]
        rows.append([c.strip().split("\n")[0].strip() for c in cells])
    head = next(r for r in rows if len(r) > 3 and r[2] == "Total" and r[3] == "Russians")
    cols = head[3:]
    assert cols == ["Russians", "Moldovans", "Ukrainians", "Bulgarians", "Gagauzians",
                    "Belorussians", "Germans", '"Transnistrians"', "Poles", "Other", "Undeclared",
                    "Refused to answer"], cols
    out, total = {}, None
    for r in rows:
        if len(r) != len(head) or not re.fullmatch(r"\d+", r[2]):
            continue
        vals = [_num(x) for x in r[2:]]
        # "Refused to answer" is a subset of "Undeclared": the row sums without it
        if sum(vals[1:-1]) != vals[0] or vals[-1] > vals[-2]:
            raise SystemExit(f"{r[1]}: {sum(vals[1:-1])} != {vals[0]}")
        if r[1] == "Republica Moldovenească Nistreană":
            total = vals
        elif r[1] in UNITS:
            out[UNITS[r[1]][0]] = dict(zip(cols, vals[1:]))
        else:
            raise SystemExit(f"unknown row {r[1]!r}")
    if len(out) != 8 or total[0] != TOTAL:
        raise SystemExit(f"{len(out)} units, total {total and total[0]}")
    for i, c in enumerate(cols):
        # "Refused" (a subset of Undeclared, not drawn) is 4,603 over the units against 1,974 in
        # the total row; the transcription and the census disagree there, and nothing reads it
        if c != "Refused to answer" and sum(u[c] for u in out.values()) != total[i + 1]:
            raise SystemExit(f"{c}: units do not sum to the total row")
    print(f"  8 units, {TOTAL:,}; units sum to the total row in 11 of 12 columns (not refusals, which are not drawn)")
    return out


def bns_shares():
    import openpyxl
    rows = list(openpyxl.load_workbook(BNS, read_only=True)["5.33"].iter_rows(values_only=True))
    head = [str(c).strip() if c else "" for c in rows[7]]
    nd =[str(c).strip() if c else "" for c in rows[6]].index("Nu au declarat limba maternă")
    langcols = [i for i in range(3, nd) if head[i]]
    shares = {}
    for r in rows[10:40]:
        nat = str(r[1]).strip() if r[1] else ""
        if nat.startswith("în %"):
            break          # the sheet's second half is the same table in percent
        if nat not in [v for v in NAT.values() if v]:
            continue
        tot = sum(_num(r[i]) for i in langcols)
        sh = {}
        for i in langcols:
            lab = LANG.get(head[i], "Other")
            sh[lab] = sh.get(lab, 0) + _num(r[i]) / tot
        shares[nat] = sh
        top = sorted(sh.items(), key=lambda x: -x[1])[:3]
        print(f"    {nat:<16} {tot:>9,}: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in top))
    missing = {v for v in NAT.values() if v} - set(shares)
    if missing:
        raise SystemExit(f"BNS 5.33 lacks {missing}")
    return shares


def rows():
    data = read_table()
    print("  Moldova 2024, nationality by mother tongue (BNS 5.33), shares used:")
    sh = bns_shares()
    out = []
    for unit, nats in data.items():
        for nat, n in nats.items():
            if nat in NOT_DRAWN or not n:      # refused is inside undeclared
                continue
            key = NAT[nat]
            split = sh[key] if key else {"Russian" if nat == '"Transnistrians"' else "Other": 1.0}
            for lang, s in split.items():
                if s:
                    out.append(dict(geo_level="unit", geo_id=unit, nationality=nat,
                                    source_category=lang, count=n * s, tier="derived"))
    return out


def build(census):
    import geopandas as gpd
    import pandas as pd
    import shapely
    from shapely.geometry import shape
    from _grid import kontur_path
    geoms = {}
    for unit, rel in UNITS.values():
        g = shape(json.loads((RAW / "osm" / f"rel_{rel}.geojson").read_text(encoding="utf-8")))
        geoms[unit] = g if g.is_valid else shapely.make_valid(g)
    inside = shapely.area(shapely.intersection(geoms["PMR-TIRASPOL"], geoms["PMR-DNESTROVSK"])
                          ) / geoms["PMR-DNESTROVSK"].area
    print(f"  Dnestrovsk: {inside:.0%} inside OSM's Tiraspol city council, cut out of it")
    geoms["PMR-TIRASPOL"] = shapely.difference(geoms["PMR-TIRASPOL"], geoms["PMR-DNESTROVSK"])
    u = gpd.GeoDataFrame({"unit": list(geoms)}, geometry=list(geoms.values()), crs=4326)
    pr = u.to_crs(32635)
    over = [(u["unit"][a], u["unit"][b], shapely.area(shapely.intersection(pr.geometry[a], pr.geometry[b])) / 1e6)
            for a in range(len(u)) for b in range(a + 1, len(u))]
    over = [o for o in over if o[2] > 0.05]
    print("  overlaps: " + (", ".join(f"{a}/{b} {x:.1f} km2" for a, b, x in over) or "none"))
    if sum(o[2] for o in over) > 5:
        raise SystemExit("the raion polygons overlap")
    u.to_file(GEO / "md_pmr_units.gpkg", layer="units", driver="GPKG")

    k = gpd.read_file(kontur_path("md"))
    cen = k.geometry.centroid.to_crs(4326)
    key = lambda s: [f"{x:.5f},{y:.5f}" for x, y in zip(s.x, s.y)]  # noqa: E731
    md = gpd.read_file(GEO / "md_hexes.gpkg")
    hexrows = md[md.geometry.geom_type == "Polygon"]
    taken = set(key(hexrows.to_crs(3857).geometry.centroid.to_crs(4326)))
    kk = pd.Series(key(cen), index=k.index)
    free = ~kk.isin(taken)
    print(f"  Kontur MD: {len(k):,} hexes, {int((~free).sum()):,} in md_hexes.gpkg, "
          f"{int(free.sum()):,} free ({k.loc[free, 'population'].sum():,.0f} people)")
    pts = gpd.GeoDataFrame({"pop": k.loc[free, "population"].to_numpy(dtype=float)},
                           geometry=cen[free].to_numpy(), crs=4326, index=k.index[free])
    j = gpd.sjoin(pts, u, how="left", predicate="within")
    j = j[~j.index.duplicated()].reindex(pts.index)
    # a free hex just outside a unit sits on a boundary the two layers draw differently (no UAT
    # holds it either); within SNAP_M it goes to the nearest unit. The rest of the free hexes are
    # far away (md_geo.py's own border drops)
    rest = pts[j["unit"].isna()]
    nn = gpd.sjoin_nearest(rest.to_crs(32635), u.to_crs(32635), distance_col="d")
    nn = nn[~nn.index.duplicated()].reindex(rest.index)
    snap = (nn["d"] <= 1000).to_numpy()
    j.loc[rest.index[snap], "unit"] = nn.loc[snap, "unit"].to_numpy()
    print(f"  {int(snap.sum())} free hexes ({rest.loc[snap, 'pop'].sum():,.0f} people) within 1 km "
          "of a unit, snapped to it")
    keep = j["unit"].notna()
    add = gpd.GeoDataFrame({"unit": j.loc[keep, "unit"].to_numpy(), "pop": pts.loc[keep, "pop"].to_numpy()},
                           geometry=k.geometry[free.to_numpy()][keep.to_numpy()].to_crs(4326).to_numpy(),
                           crs=4326)
    print(f"  {int(keep.sum()):,} free hexes ({add['pop'].sum():,.0f} people) inside the 8 units; "
          f"{int((~keep).sum()):,} ({pts.loc[~keep, 'pop'].sum():,.0f}) outside them, not used")
    # md hexes inside the units: Moldova's census counted those places, so they stay Moldova's
    mc = gpd.GeoDataFrame(md[["unit", "pop"]], geometry=md.to_crs(3857).geometry.centroid.to_crs(4326), crs=4326)
    mj = gpd.sjoin(mc, u.rename(columns={"unit": "pmr"}), predicate="within")
    print(f"  {len(mj):,} hexes of Moldova's own layer ({mj['pop'].sum():,.0f} Kontur people, "
          f"{mj['unit'].nunique()} UATs) sit inside the units and stay Moldova's")
    per = add.groupby("unit")["pop"].sum()
    ratio = per.sum() / sum(census.values())
    print(f"  Kontur 2023 / census 2015 per unit, normalised (nationally {ratio:.2f}):")
    for x in sorted(census):
        print(f"    {x:<16} census {census[x]:>8,.0f}  Kontur {per.get(x, 0):>8,.0f}  "
              f"{per.get(x, 0) / census[x] / ratio:5.2f}")
    if set(census) - set(per.index[per > 0]):
        raise SystemExit(f"units with no hex: {set(census) - set(per.index)}")
    plus = gpd.GeoDataFrame(pd.concat([md[["unit", "pop", "geometry"]], add], ignore_index=True), crs=4326)
    plus.to_file(GEO / "md_plus_hexes.gpkg", layer="hexes", driver="GPKG")
    print(f"  wrote md_plus_hexes.gpkg ({len(plus):,} = {len(md):,} + {len(add):,})")


def main():
    import pandas as pd
    if "--fetch" in sys.argv:
        fetch()
    df = pd.DataFrame(rows())
    df.to_csv(NORM, index=False)
    print(f"  wrote {NORM.name}: {len(df)} rows, {df['count'].sum():,.0f} people")
    print(df.groupby("source_category")["count"].sum().round().astype(int)
          .sort_values(ascending=False).to_string())
    build(df.groupby("geo_id")["count"].sum().to_dict())


if __name__ == "__main__":
    main()
