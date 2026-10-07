"""Abkhazia (2011 census) and South Ossetia (2015 census): the two parts of Georgia that Georgia's
2024 census did not enumerate, as rows and a placement layer for `countries/ge.py`.

    python sources/ge_breakaway.py --fetch   -> data/raw/ge/ (tables, PDF, OSM district polygons)
    python sources/ge_breakaway.py           -> data/normalized/ge_breakaway.csv,
                                                data/geo/ge/ge_breakaway_units.gpkg,
                                                data/geo/ge/ge_plus_hexes.gpkg

GEOGRAPHY FOLLOWS religiondots: both territories stay inside Georgia's entry (religiondots' `ge`
leaves them blank; its country_shapes.CLIP names them from Natural Earth's disputed-areas layer).
They are extra units of `ge`, ids `AB-*` and `SO-*`, with their own `parts` line.

ABKHAZIA, 2011 census (Department of State Statistics of the Republic of Abkhazia, published in
"Itogi perepisi naseleniya Respubliki Abkhaziya 2011 goda", Sukhum 2012): nationality by district.
The table is read from Tim Bespyatov's transcription, pop-stat.mashke.org/abkhazia-ethnic2011.htm
(8 units, 15 nationalities and "other"; it sums to the published 240,705 and is asserted to).
The census published no native-language table, so each nationality is read as a language with a
retention share (AGENT_BRIEF §2, ethnicity only), every row `derived`:
  Abkhaz        97% Abkhaz, the rest Russian: the 1989 Soviet census, all Abkhaz (Hewitt and
                Watson, Encyclopedia of World Cultures, "Abkhazians": "As of 1989, 97 percent of
                Abkhazians claimed Abkhaz as their native tongue").
  Armenians, Ukrainians, Greeks, Roma
                Russia's 2021 census, Volume 5 Table 7, Krasnodar Krai: the same Black Sea coast
                and, for the Armenians, the same Hamshen community as Gagra and Gulripsh.
                The remainder goes to Russian.
  Turks, Ossetians, Tatars, Belarusians
                the same table for the Russian Federation as a whole; remainder to Russian.
  Georgians, Mingrelians, Svans
                their own language, 100%. The census prints Georgians and Mingrelians apart
                (43,248 and 3,207), and that split is kept: Mingrelians on Mingrelian, Georgians
                on Georgian. Most of Gal's Georgians speak Mingrelian at home, but so do the
                people of Samegrelo across the Enguri, whom Georgia's native-language census
                draws as Georgian; drawing the Gal Georgians as Mingrelian would put a language
                border on the ceasefire line that is not on the ground. No retention source exists
                for this compact community, and the Krasnodar rate (59% for a scattered diaspora)
                does not fit it.
  Russians      Russian. Estonians (the Salme and Sulevo villages): Estonian, no source.
  Other         `other`.

SOUTH OSSETIA, 2015 census (State Statistics Directorate, "Itogi Vseobshchei perepisi naseleniya
Respubliki Yuzhnaya Osetiya 2015 goda", ugosstat.ru): tables 4.2.1-4.2.5 give native language by
nationality for each district and Tskhinval city. MEASURED: the language totals per unit are read
directly (the "stated a nationality" row plus the "nationality not stated" row), not modelled.
Asserted against table 4.2's republic totals (Greek and Romani, which only two units print,
folded into "other languages" for that check). Not stated: 93 people, not drawn.

UNITS: OpenStreetMap's district relations, fetched as GeoJSON from polygons.openstreetmap.fr
(ids in REL; found with Nominatim). Sukhum city is cut out of Sukhum district. Each polygon has
Georgia's own 64 units (sources/ge_geo.py, COD-AB) subtracted, so no ground is placed twice.

PLACEMENT: the Kontur GE hexes that ge_hexes.gpkg does not hold, each to the district its
centroid falls in; centroids just offshore (Abkhazia's coast) go to the nearest district within
SNAP_M. ge_plus_hexes.gpkg is ge_hexes.gpkg plus these, and is what countries/ge.py reads; it is
rebuilt here, so re-run this script after sources/ge_geo.py.
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
RAW = ROOT / "data" / "raw" / "ge"
RU_TAB7 = ROOT / "data" / "raw" / "ru" / "Tom5_tab7_VPN-2020.xlsx"
NORM = ROOT / "data" / "normalized" / "ge_breakaway.csv"
GEO = ROOT / "data" / "geo" / "ge"
UA = {"User-Agent": "languagedots-research/1.0"}

AB_URL = "http://pop-stat.mashke.org/abkhazia-ethnic2011.htm"
AB_FILE = RAW / "mashke_abkhazia-ethnic2011.htm"
SO_URL = "http://ugosstat.ru/wp-content/uploads/2017/06/Itogi-perepisi-RYUO.pdf"
SO_FILE = RAW / "ugosstat_itogi_perepisi_2015.pdf"
POLY_URL = "https://polygons.openstreetmap.fr/get_geojson.py?id={}&params=0"

AB_TOTAL = 240_705
SO_TOTAL = 53_532
SNAP_M = 5000

# table row (English name as printed) -> unit id, OSM relation
AB_UNITS = {"ak. Aqwa": ("AB-SUKHUM-CITY", 2027324), "Aqwa": ("AB-SUKHUM", 2027323),
            "Gagra": ("AB-GAGRA", 1245702), "Gal": ("AB-GAL", 2624773),
            "Gwdouta": ("AB-GUDAUTA", 2027322), "Gwylrypsh": ("AB-GULRIPSH", 2027325),
            "Ochamchyra": ("AB-OCHAMCHIRA", 2624774), "Tqwarchal": ("AB-TKVARCHELI", 2624775)}
# PDF page -> unit id, OSM relation
SO_UNITS = {105: ("SO-DZAU", 2027316), 106: ("SO-ZNAUR", 2537350), 107: ("SO-LENINGOR", 2027320),
            108: ("SO-TSKHINVAL", 2027319), 109: ("SO-TSKHINVAL-CITY", 2027318)}
SO_REPUBLIC_PAGE = 104
NAMES = {"AB-SUKHUM-CITY": "Sukhum (city)", "AB-SUKHUM": "Sukhum district", "AB-GAGRA": "Gagra",
         "AB-GAL": "Gal", "AB-GUDAUTA": "Gudauta", "AB-GULRIPSH": "Gulripsh",
         "AB-OCHAMCHIRA": "Ochamchira", "AB-TKVARCHELI": "Tkvarcheli", "SO-DZAU": "Dzau",
         "SO-ZNAUR": "Znaur", "SO-LENINGOR": "Leningor", "SO-TSKHINVAL": "Tskhinval district",
         "SO-TSKHINVAL-CITY": "Tskhinval (city)"}

# Abkhazia: nationality -> how its people are read as languages. ("own", lang, share) is a fixed
# share; ("tab7", sheet, nationality column, language row, lang) is Russia 2021's own-language
# share for that sheet; the rest goes to Russian in both cases.
KRASNODAR, RF = "Краснодарский край", "Российская Федерация"
AB_RULES = {
    "Abkhazians": ("own", "Abkhaz", 0.97),
    "Georgians": ("own", "Georgian", 1.0),
    "Mingrelians": ("own", "Mingrelian", 1.0),
    "Svans": ("own", "Svan", 1.0),
    "Armenians": ("tab7", KRASNODAR, "Армяне", "Армянский", "Armenian"),
    "Russians": ("own", "Russian", 1.0),
    "Ukrainians": ("tab7", KRASNODAR, "Украинцы", "Украинский", "Ukrainian"),
    "Greeks": ("tab7", KRASNODAR, "Греки", "Греческий", "Greek"),
    "Turks": ("tab7", RF, "Турки", "Турецкий", "Turkish"),
    "Ossetians": ("tab7", RF, "Осетины", "Осетинский", "Ossetian"),
    "Estonians": ("own", "Estonian", 1.0),
    "Tatars": ("tab7", RF, "Татары", "Татарский", "Tatar"),
    "Belorussians": ("tab7", RF, "Белорусы", "Белорусский", "Belarusian"),
    "Romanies": ("tab7", KRASNODAR, "Цыгане", "Цыганский", "Romani"),
    "Other": ("own", "Other", 1.0),
}
# South Ossetia: the PDF's language heads -> labels
SO_LANG = {"осетинский": "Ossetian", "русский": "Russian", "армянский": "Armenian",
           "азербайджанский": "Azerbaijani", "грузинский": "Georgian", "украинский": "Ukrainian",
           "греческий": "Greek", "цыганский": "Romani", "другие языки": "Other",
           "другиет языки": "Other"}


def fetch():
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    for url, path in ((AB_URL, AB_FILE), (SO_URL, SO_FILE)):
        if not path.exists():
            r = requests.get(url, headers=UA, timeout=300)
            r.raise_for_status()
            path.write_bytes(r.content)
            print(f"  {path.name}: {len(r.content):,} bytes")
    (RAW / "osm").mkdir(exist_ok=True)
    for _, rel in list(AB_UNITS.values()) + list(SO_UNITS.values()):
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


def read_abkhazia():
    """{unit: {nationality: count}} from the transcription, asserted to sum to the total row."""
    t = AB_FILE.read_text(encoding="utf-8-sig")
    rows = []
    for tr in re.split(r"<tr>", t, flags=re.I)[1:]:
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in re.split(r"<t[dh][^>]*>", tr, flags=re.I)[1:]]
        cells = [re.split(r"</t", c)[0].strip().split("\n")[0].strip() for c in cells]
        rows.append(cells)
    head = rows[0]
    assert head[1] == "Araion" and head[2] == "Total", head[:3]
    nats = head[3:]
    assert len(nats) == 15 and nats[-1] == "Other", nats
    total = None
    out = {}
    for r in rows[1:]:
        if len(r) < len(head):
            continue
        name, vals = r[1], [_num(x) for x in r[2:len(head)]]
        if name == "Apsny Ahwyntkarra":
            total = vals
            continue
        if name not in AB_UNITS:
            raise SystemExit(f"Abkhazia: unknown row {name!r}")
        if sum(vals[1:]) != vals[0]:
            raise SystemExit(f"Abkhazia {name}: nationalities {sum(vals[1:])} != total {vals[0]}")
        out[AB_UNITS[name][0]] = dict(zip(nats, vals[1:]))
    if set(out) != {u for u, _ in AB_UNITS.values()}:
        raise SystemExit(f"Abkhazia: units {sorted(out)}")
    if total[0] != AB_TOTAL:
        raise SystemExit(f"Abkhazia total {total[0]} != {AB_TOTAL}")
    for i, n in enumerate(nats):
        s = sum(u[n] for u in out.values())
        if s != total[i + 1]:
            raise SystemExit(f"Abkhazia {n}: districts sum {s} != total {total[i + 1]}")
    print(f"  Abkhazia: 8 units, 15 nationalities + other, {AB_TOTAL:,}; districts sum to the "
          "total row in every column")
    return out


def tab7_share(sheet, nat, lang_row):
    import openpyxl
    wb = openpyxl.load_workbook(RU_TAB7, read_only=True)
    rows = list(wb[sheet].iter_rows(values_only=True))
    head = [str(c).strip() if c else "" for c in rows[5]]
    col = head.index(nat)
    stated = next(r for r in rows if r[0] and str(r[0]).startswith("Указавшие родной язык"))
    own = next(r for r in rows if r[0] and str(r[0]).strip() == lang_row)
    s, o = _num(stated[col]), _num(own[col])
    return o / s, o, s


def abkhazia_rows():
    data = read_abkhazia()
    shares, notes = {}, []
    for nat, rule in AB_RULES.items():
        if rule[0] == "own":
            shares[nat] = {rule[1]: rule[2], "Russian": 1 - rule[2]} if rule[2] < 1 else {rule[1]: 1.0}
            continue
        _, sheet, col, row, lang = rule
        sh, o, s = tab7_share(sheet, col, row)
        shares[nat] = {lang: sh, "Russian": 1 - sh}
        notes.append(f"{nat}: {lang} {o:,} of {s:,} ({100 * sh:.1f}%) in {sheet}")
    print("  retention (Russia 2021, Table 7):\n    " + "\n    ".join(notes))
    rows = []
    for unit, nats in data.items():
        for nat, n in nats.items():
            for lang, sh in shares[nat].items():
                if n and sh:
                    rows.append(dict(geo_id=unit, territory="Abkhazia", nationality=nat,
                                     source_category=lang, count=n * sh, tier="derived"))
    return rows


def _pdf_table(page):
    import fitz
    toks = [t.strip() for t in fitz.open(SO_FILE)[page - 1].get_text().split("\n") if t.strip()]
    iu = next(i for i, t in enumerate(toks) if t.startswith("Указавшие национальную"))
    langs, j = [], iu - 1
    while toks[j] in SO_LANG:
        langs.insert(0, SO_LANG[toks[j]])
        j -= 1
    k = len(langs) + 2
    if toks[iu + 1] != "принадлежность":
        raise SystemExit(f"page {page}: layout changed at {toks[iu:iu + 3]}")
    stated = []
    for x in toks[iu + 2: iu + 2 + k]:
        if not re.fullmatch(r"[\d–\-]+", x):
            break      # table 4.2 leaves the row's last cell (not stated) empty, not "–"
        stated.append(_num(x))
    stated += [0] * (k - len(stated))
    il = next(i for i, t in enumerate(toks) if t.startswith("Лица, в переписных"))
    m = il
    while not re.fullmatch(r"[\d–\-]+", toks[m]):
        m += 1
    nonat = [_num(x) for x in toks[m: m + k]]
    if len(nonat) != k or m + k != len(toks):
        raise SystemExit(f"page {page}: the last row is not {k} numbers at the end of the page")
    for row in (stated, nonat):
        if sum(row[1:-1]) != row[0]:
            raise SystemExit(f"page {page}: languages {sum(row[1:-1])} != stated {row[0]}")
    langs_n = {lang: stated[i + 1] + nonat[i + 1] for i, lang in enumerate(langs)}
    return langs_n, stated[-1] + nonat[-1]


def ossetia_rows():
    rows, sums, nstated = [], {}, 0
    for page, (unit, _) in SO_UNITS.items():
        langs, ns = _pdf_table(page)
        nstated += ns
        for lang, n in langs.items():
            key = lang if lang in ("Ossetian", "Russian", "Armenian", "Azerbaijani", "Georgian",
                                   "Ukrainian") else "Other"
            sums[key] = sums.get(key, 0) + n
            if n:
                rows.append(dict(geo_id=unit, territory="South Ossetia", nationality="",
                                 source_category=lang, count=n, tier="measured"))
    rep, rep_ns = _pdf_table(SO_REPUBLIC_PAGE)
    if rep != sums or rep_ns != nstated:
        raise SystemExit(f"South Ossetia: units {sums} / {nstated} != republic {rep} / {rep_ns}")
    tot = sum(r["count"] for r in rows) + nstated
    if tot != SO_TOTAL:
        raise SystemExit(f"South Ossetia: {tot} people, expected {SO_TOTAL}")
    print(f"  South Ossetia: 5 units sum to table 4.2 in every language; "
          f"{tot - nstated:,} with a native language + {nstated} not stated = {SO_TOTAL:,}")
    return rows


def build_units():
    import geopandas as gpd
    import shapely
    from shapely.geometry import shape
    recs = []
    for unit, rel in list(AB_UNITS.values()) + list(SO_UNITS.values()):
        g = shape(json.loads((RAW / "osm" / f"rel_{rel}.geojson").read_text(encoding="utf-8")))
        if not g.is_valid:      # Tskhinval district's relation is; difference() fails silently on it
            g = shapely.make_valid(g)
            g = shapely.union_all([p for p in shapely.get_parts(g) if p.geom_type in
                                   ("Polygon", "MultiPolygon")])
        recs.append((unit, rel, g))
    u = gpd.GeoDataFrame({"unit": [r[0] for r in recs], "osm_rel": [r[1] for r in recs],
                          "name": [NAMES[r[0]] for r in recs]},
                         geometry=[r[2] for r in recs], crs=4326)
    # a city drawn inside its district's relation is cut out of the district
    geoms = dict(zip(u["unit"], u.geometry))
    for dist, city in (("AB-SUKHUM", "AB-SUKHUM-CITY"), ("SO-TSKHINVAL", "SO-TSKHINVAL-CITY")):
        inside = shapely.area(shapely.intersection(geoms[dist], geoms[city])) / geoms[city].area
        print(f"  {city}: {inside:.0%} of it inside OSM's {dist}, cut out of the district")
        geoms[dist] = shapely.difference(geoms[dist], geoms[city])
    u = u.set_geometry([geoms[x] for x in u["unit"]], crs=4326)
    pr = u.to_crs(32638)
    pairs = [(u["unit"].iloc[a], u["unit"].iloc[b],
              shapely.area(shapely.intersection(pr.geometry.iloc[a], pr.geometry.iloc[b])) / 1e6)
             for a in range(len(pr)) for b in range(a + 1, len(pr))]
    pairs = [p for p in pairs if p[2] > 0.05]
    over = sum(p[2] for p in pairs)
    print(f"  overlap between the 13 polygons after the cuts: {over:.2f} km2 "
          + ", ".join(f"{a}/{b} {x:.1f}" for a, b, x in pairs))
    if over > 20:
        raise SystemExit("district polygons overlap; look at the OSM relations")
    ge = gpd.read_file(GEO / "ge_units.gpkg").to_crs(4326)
    ov = gpd.overlay(u[["unit", "geometry"]], ge[["unit", "geometry"]].rename(
        columns={"unit": "ge_unit"}), how="intersection", keep_geom_type=True).to_crs(32638)
    ov["km2"] = ov.area / 1e6
    ov = ov[ov["km2"] > 1].sort_values("km2", ascending=False)
    print("  OSM districts against COD-AB's census units (km2 both claim; the OSM line is the "
          "line of control, so the district keeps it): "
          + ", ".join(f"{a}/{b} {x:.0f}" for a, b, x in ov[["unit", "ge_unit", "km2"]].values))
    # against Natural Earth's breakaway polygons (religiondots' copy, read only)
    nep = ROOT.parent / "religiondots" / "data" / "geo" / "ne_10m_admin_0_disputed_areas.geojson"
    ne = {f["properties"]["BRK_NAME"]: shape(f["geometry"])
          for f in json.loads(nep.read_text(encoding="utf-8"))["features"]
          if f["properties"].get("BRK_NAME") in ("Abkhazia", "South Ossetia")}
    for name, pre in (("Abkhazia", "AB-"), ("South Ossetia", "SO-")):
        mine = shapely.union_all(u.loc[u["unit"].str.startswith(pre), "geometry"].values)
        a = gpd.GeoSeries([mine, ne[name], shapely.intersection(mine, ne[name])], crs=4326
                          ).to_crs(32638).area / 1e6
        print(f"  {name}: OSM districts {a[0]:,.0f} km2, Natural Earth {a[1]:,.0f} km2, "
              f"shared {a[2]:,.0f} km2")
        if a[2] < 0.85 * min(a[0], a[1]):
            raise SystemExit(f"{name}: the districts and Natural Earth disagree")
    u.to_file(GEO / "ge_breakaway_units.gpkg", layer="units", driver="GPKG")
    return u


def build_hexes(units, census):
    import geopandas as gpd
    import pandas as pd
    from _grid import kontur_path
    k = gpd.read_file(kontur_path("ge"))
    cen = k.geometry.centroid.to_crs(4326)
    key = lambda s: [f"{x:.5f},{y:.5f}" for x, y in zip(s.x, s.y)]  # noqa: E731
    ge = gpd.read_file(GEO / "ge_hexes.gpkg")
    taken = set(key(ge.to_crs(3857).geometry.centroid.to_crs(4326)))
    kk = pd.Series(key(cen), index=k.index)
    free = ~kk.isin(taken)
    print(f"  Kontur GE: {len(k):,} hexes; {int((~free).sum()):,} already in ge_hexes.gpkg "
          f"(of its {len(ge):,}), {int(free.sum()):,} free with {k.loc[free, 'population'].sum():,.0f} people")
    if int((~free).sum()) != len(ge):
        raise SystemExit("ge_hexes.gpkg holds hexes this script cannot match to Kontur GE")
    pts = gpd.GeoDataFrame({"pop": k["population"].to_numpy(dtype=float)},
                           geometry=cen.to_numpy(), crs=4326, index=k.index)
    j = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()].reindex(pts.index)
    # Kontur hexes inside a district that Georgia's layer holds: COD-AB's municipalities run over
    # part of the line of control (Gori's polygon takes in Tskhinval), so sources/ge_geo.py placed
    # some of Georgia's census people there. They leave Georgia's layer here.
    moved = (~free) & j["unit"].notna()
    gk = pd.Series(key(ge.to_crs(3857).geometry.centroid.to_crs(4326)), index=ge.index)
    gone = gk.isin(set(kk[moved]))
    by = ge[gone].groupby("unit")["pop"].sum().sort_values(ascending=False)
    print(f"  {int(gone.sum())} hexes of Georgia's layer ({ge.loc[gone, 'pop'].sum():,.0f} Kontur "
          "people) lie inside the districts and leave it: "
          + ", ".join(f"{u_} {p:,.0f}" for u_, p in by.items()))
    left = ge[~gone].groupby("unit")["pop"].sum()
    for u_ in by.index:
        tot = ge[ge["unit"] == u_]["pop"].sum()
        print(f"    {u_}: keeps {left.get(u_, 0):,.0f} of {tot:,.0f} Kontur people")
    if set(ge["unit"]) - set(left.index[left > 0]):
        raise SystemExit(f"a Georgian unit would lose every hex: {set(ge['unit']) - set(left.index)}")
    ge = ge[~gone]
    rest = pts[free & j["unit"].isna()]
    nn = gpd.sjoin_nearest(rest.to_crs(32638), units[["unit", "geometry"]].to_crs(32638),
                           how="left", distance_col="d")
    nn = nn[~nn.index.duplicated()].reindex(rest.index)
    # a free hex near a district is offshore (Abkhazia's coast) or across the Psou or the Roki
    # tunnel; only the sea is snapped, so a Russian town is never drawn as Abkhazia
    ne = gpd.read_file(ROOT.parent / "religiondots" / "data" / "geo" /
                       "ne_10m_admin_0_countries.geojson")[["ADM0_A3", "geometry"]].to_crs(4326)
    on = gpd.sjoin(rest, ne, how="left", predicate="within")
    on = on[~on.index.duplicated()].reindex(rest.index)
    sea = on["ADM0_A3"].isna().to_numpy() | (on["ADM0_A3"] == "GEO").to_numpy()
    snap = sea & (nn["d"] <= SNAP_M).to_numpy()
    j.loc[rest.index[snap], "unit"] = nn.loc[snap, "unit"].to_numpy()
    print(f"  {int(snap.sum())} hexes ({rest.loc[snap, 'pop'].sum():,.0f} people) just offshore or "
          f"in a gap, snapped to the nearest district within {SNAP_M} m")
    keep = j["unit"].notna()
    add = gpd.GeoDataFrame({"unit": j.loc[keep, "unit"].to_numpy(),
                            "pop": pts.loc[keep, "pop"].to_numpy()},
                           geometry=k.geometry[keep.to_numpy()].to_crs(4326).to_numpy(), crs=4326)
    per = add.groupby("unit")["pop"].sum()
    ratio = per.sum() / sum(census.values())
    print("  Kontur / census per unit, normalised (Kontur 2023 against 2011 and 2015 censuses; "
          f"nationally {ratio:.2f}):")
    for u_ in sorted(census):
        print(f"    {u_:<18} census {census[u_]:>7,.0f}  Kontur {per.get(u_, 0):>8,.0f}  "
              f"{per.get(u_, 0) / census[u_] / ratio:5.2f}")
    missing = sorted(set(census) - set(per.index[per > 0]))
    if missing:
        raise SystemExit(f"units with no populated hex: {missing}")
    add.to_file(GEO / "ge_breakaway_hexes.gpkg", layer="hexes", driver="GPKG")
    plus = gpd.GeoDataFrame(pd.concat([ge[["unit", "pop", "geometry"]], add], ignore_index=True),
                            crs=4326)
    plus.to_file(GEO / "ge_plus_hexes.gpkg", layer="hexes", driver="GPKG")
    print(f"  wrote ge_breakaway_hexes.gpkg ({len(add):,}) and ge_plus_hexes.gpkg ({len(plus):,})")


def main():
    import pandas as pd
    if "--fetch" in sys.argv:
        fetch()
    rows = abkhazia_rows() + ossetia_rows()
    df = pd.DataFrame(rows)
    df["geo_level"] = "unit"
    NORM.parent.mkdir(parents=True, exist_ok=True)
    df[["geo_level", "geo_id", "territory", "nationality", "source_category", "count", "tier"]
       ].to_csv(NORM, index=False)
    ab = df[df["territory"] == "Abkhazia"]["count"].sum()
    if abs(ab - AB_TOTAL) > 0.5:
        raise SystemExit(f"Abkhazia rows sum to {ab}, expected {AB_TOTAL}")
    print(f"  wrote {NORM.name}: {len(df)} rows, {df['count'].sum():,.0f} people")
    print(df.groupby(["territory", "source_category"])["count"].sum().round().astype(int)
          .sort_values(ascending=False).to_string())
    units = build_units()
    census = df.groupby("geo_id")["count"].sum().to_dict()
    build_hexes(units, census)


if __name__ == "__main__":
    main()
