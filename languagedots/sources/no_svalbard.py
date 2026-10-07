"""Svalbard, drawn inside Norway as three units (religiondots draws it as Norway's unit NO-21).

    python sources/no_svalbard.py --fetch    SSB 07430, 12622, the 2026 article and its chart
    python sources/no_svalbard.py            -> data/normalized/no_svalbard.csv, and Svalbard's
                                                rows appended to data/geo/no/no_hexes.gpkg

Svalbard is in no kommune, so the mainland build (sources/no_ssb.py) has nobody there. This adds
it without touching a mainland row.

UNITS AND COUNTS (1 January 2026, SSB):
  NO-21-LYR  Longyearbyen and Ny-Alesund, 2,512 (07430). By citizenship from the Flourish chart
             in SSB's article of 3 March 2026 ("Hoy befolkningsutskiftning og okende mangfold pa
             Svalbard ved inngangen av 2026", figur 2): 13 named citizenships and "Resten". The
             chart is checked against table 12622's six citizenship groups for the same date.
  NO-21-BAR  Barentsburg and Pyramiden, 392 (07430). SSB publishes no citizenship table for the
             Russian settlements, but the same article gives the largest citizenships on all of
             Svalbard (Russland 305, Tadsjikistan 83, Ukraina 78, ...); Svalbard less the chart's
             Longyearbyen figures leaves Russia 248, Ukraine 56, Tajikistan 83 (the chart has no
             Tajik row and its Africa-and-Asia remainder is only 32, so all 83 are put here),
             and 5 unassigned.
  NO-21-HOR  Hornsund, 10 (07430): the Polish Polar Station of the Polish Academy of Sciences,
             drawn as Polish (no citizenship figure; the station's overwintering crew is Polish).

LANGUAGES. Norwegian citizens on Norwegian, as the mainland. Each citizenship through
origin_mix.mix(iso, "no"), as the mainland's country backgrounds. Two exceptions:
  * Ukrainians in Barentsburg: Arktikugol's Ukrainian miners mostly came from the Donbas
    (Wikipedia, "Arktikugol" and "Barentsburg"), so they take Donetsk oblast's native-language
    split in Ukraine's 2001 census (this map's ua.csv), not Ukraine's national mix.
  * "Resten" (249 in Longyearbyen) and Barentsburg's 5 unassigned are on `other`: 50-odd
    citizenships, the languages unknown.
All rows `derived`. Of Longyearbyen's 2,512, 1,648 are registered in a mainland kommune and so
are counted there too (07430's own split); they are not removed from the mainland, whose rows this
build leaves alone. 0.03% of Norway.

PLACEMENT. religiondots' Svalbard hexes (its sources/no_geo.py: Kontur SJ inside Natural Earth's
Svalbard, weighted to SSB's settlement counts), read-only, re-keyed: Longyearbyen's hexes and
Ny-Alesund's (35 of the unit, uncited: Kings Bay's usual winter figure of 30-40) to LYR,
Barentsburg's and Pyramiden's (10, uncited: a winter caretaker crew) to BAR, Hornsund's to HOR.
Every other Svalbard hex (Sveagruva, closed 2017, where Kontur puts 1,900) joins LYR at weight 0:
placement ground with no dots.
"""
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from rdlink import RD_GEO  # noqa: E402

RAW = ROOT / "data" / "raw" / "no"
OUT = ROOT / "data" / "normalized" / "no_svalbard.csv"
HEX = ROOT / "data" / "geo" / "no" / "no_hexes.gpkg"
RD_LAU = RD_GEO / "no" / "no_lau.gpkg"
KONTUR_SJ = RD_GEO / "kontur" / "kontur_population_SJ_20231101.gpkg"
API = "https://data.ssb.no/api/v0/en/table/"
ARTICLE = ("https://www.ssb.no/befolkning/folketall/statistikk/befolkningen-pa-svalbard/artikler/"
           "hoy-befolkningsutskiftning-og-okende-mangfold-pa-svalbard-ved-inngangen-av-2026")
FLOURISH = "https://flo.uri.sh/visualisation/27693460/embed"
HALF = "2026H1"
UA = {"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36"}

NORWEGIAN = "indoeuropean.germanic.north.norwegian"
POLISH = "indoeuropean.slavic.west.polish"
RUSSIAN = "indoeuropean.slavic.east.russian"
UKRAINIAN = "indoeuropean.slavic.east.ukrainian"
OTHER = "other"

LYR, BAR, HOR = "NO-21-LYR", "NO-21-BAR", "NO-21-HOR"
NAMES = {LYR: "Longyearbyen and Ny-Alesund", BAR: "Barentsburg and Pyramiden", HOR: "Hornsund"}
# the chart's Norwegian column names -> ISO
CHART_ISO = {"Norge": "NO", "Sverige": "SE", "Thailand": "TH", "Russland": "RU",
             "Filippinene": "PH", "Danmark": "DK", "Tyskland": "DE", "Ukraina": "UA",
             "Polen": "PL", "Storbritannia": "GB", "USA": "US", "Finland": "FI",
             "Frankrike": "FR", "Resten": None}
# the article's all-Svalbard figures, "De mest representerte statsborgerskapene" (parsed and
# asserted against the saved page)
ARTICLE_ISO = {"Russland": "RU", "Filippinene": "PH", "Thailand": "TH", "Tadsjikistan": "TJ",
               "Ukraina": "UA", "Tyskland": "DE", "Sverige": "SE"}
# placement-only shares inside a unit (module docstring), uncited
NY_ALESUND = 35
PYRAMIDEN = 10
POINTS = {"Ny-Alesund": (78.925, 11.93, 3.0), "Pyramiden": (78.655, 16.33, 3.0)}


def fetch():
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    item = lambda code, vals: {"code": code, "selection": {"filter": "item", "values": vals}}
    alls = lambda code: {"code": code, "selection": {"filter": "all", "values": ["*"]}}
    for table, q in (("07430", [alls("Bosetting"), item("Tid", [HALF])]),
                     ("12622", [alls("Statsbrgskap"), item("Tid", [HALF])])):
        r = requests.post(API + table, json={"query": q, "response": {"format": "json-stat2"}},
                          timeout=120)
        r.raise_for_status()
        (RAW / f"ssb_{table}.json").write_bytes(r.content)
        print(f"  {table}: {len(r.content):,} bytes")
    h = {"User-Agent": next(iter(UA))}
    for url, name in ((ARTICLE, "ssb_svalbard_2026_article.html"),
                      (FLOURISH, "ssb_svalbard_2026_flourish.html")):
        r = requests.get(url, headers=h, timeout=120)
        r.raise_for_status()
        (RAW / name).write_bytes(r.content)
        print(f"  {name}: {len(r.content):,} bytes")


def jsonstat1(name):
    d = json.loads((RAW / name).read_text(encoding="utf-8"))
    dim = d["id"][0]
    cat = d["dimension"][dim]["category"]
    order = sorted(cat["index"], key=cat["index"].get)
    if int(pd.Series(d["size"]).prod()) != len(order):
        raise SystemExit(f"{name}: expected one value per {dim}")
    return pd.Series(d["value"], index=order).fillna(0).astype(int), cat["label"]


def chart():
    t = (RAW / "ssb_svalbard_2026_flourish.html").read_text(encoding="utf-8", errors="replace")
    data = json.loads(re.search(r"_Flourish_data = (\{.*?\}\]\}),", t).group(1))
    cols = json.loads(re.search(r"_Flourish_data_column_names = (\{.*?\}\}),", t).group(1))
    names = cols["data"]["value"]
    row = [r for r in data["data"] if r["label"] == 2026]
    if len(row) != 1 or len(row[0]["value"]) != len(names):
        raise SystemExit("Flourish chart: no single 2026 row")
    s = pd.Series(row[0]["value"], index=names, dtype=int)
    if set(s.index) != set(CHART_ISO):
        raise SystemExit(f"Flourish chart columns changed: {list(s.index)}")
    return s


def article():
    t = (RAW / "ssb_svalbard_2026_article.html").read_text(encoding="utf-8", errors="replace")
    t = re.sub(r"<[^>]+>", " ", t).replace("&nbsp;", " ")
    m = re.search(r"De mest representerte statsborgerskapene.{0,80}?er:(.{0,400}?)\.\s", t, re.S)
    if not m:
        raise SystemExit("article: the citizenship sentence is gone")
    got = {k: int(v) for k, v in re.findall(r"([A-Za-zæøå]+)\s*\((\d+)\)", m.group(1))}
    if set(got) != set(ARTICLE_ISO):
        raise SystemExit(f"article citizenships changed: {got}")
    if not re.search(r"Av\s*2\s*904 bosatte p.{1,3} Svalbard", t):
        raise SystemExit("article: the 2 904 total is gone")
    return {ARTICLE_ISO[k]: v for k, v in got.items()}


def donetsk_mix():
    """Russian and Ukrainian as native language in Donetsk oblast, 2001 census (ua.csv)."""
    ua = pd.read_csv(ROOT / "data" / "normalized" / "ua.csv")
    d = ua[ua["oblast"].str.startswith("DONETS")]
    ru = d.loc[d["source_category"] == "російську", "count"].sum()
    uk = d.loc[d["source_category"] == "українську", "count"].sum()
    if not 3_500_000 < ru + uk < 5_000_000:
        raise SystemExit(f"ua.csv Donetsk: {ru + uk:,} Russian and Ukrainian speakers?")
    print(f"  Donetsk oblast 2001: Russian {ru:,}, Ukrainian {uk:,} ({ru / (ru + uk):.1%} Russian)")
    return {RUSSIAN: ru / (ru + uk), UKRAINIAN: uk / (ru + uk)}


def written_ids():
    """origin_mix.mix() returns DRAWN ids (regrouped, taxonomy/regroup.py); the CSV keeps
    WRITTEN ids, as no.csv and every fragment do. Inverts regroup's move over the written tree."""
    sys.path.insert(0, str(ROOT / "taxonomy"))
    from regroup import move
    tx = ROOT / "taxonomy"
    ids = set()
    for f in [tx / "tree.txt", *sorted((tx / "tree.d").glob("*.txt"))]:
        # generated borrowed-node blocks repeat other fragments' ids and are left out
        text = re.sub(r"# --- origin_mix borrowed nodes.*?# --- end origin_mix borrowed nodes ---",
                      "", f.read_text(encoding="utf-8"), flags=re.S)
        for ln in text.splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#") and "|" in ln:
                ids.add(ln.split("|")[0].strip())
    inv = {}
    for i in ids:
        inv.setdefault(move(i), set()).add(i)

    def back(node):
        if node in ids or node == OTHER:
            return node
        w = inv.get(node, set())
        if len(w) != 1:
            raise SystemExit(f"no single written id for drawn {node!r}: {sorted(w)}")
        return next(iter(w))
    return back


def counts():
    from origin_mix import mix as _mix
    back = written_ids()

    def mix(iso, dest):
        out = {}
        for node, s in _mix(iso, dest).items():
            out[back(node)] = out.get(back(node), 0) + s
        return out

    sett, slab = jsonstat1("ssb_07430.json")
    print("07430, " + HALF + ": " + ", ".join(f"{slab[k]} {v:,}" for k, v in sett.items()))
    lyr_total = int(sett["Svalb01"] + sett["Svalb00"])
    bar_total, hor_total = int(sett["Svalb02"]), int(sett["Svalb03"])
    grp, glab = jsonstat1("ssb_12622.json")
    if int(grp["000-999"]) != lyr_total or int(grp.drop("000-999").sum()) != lyr_total:
        raise SystemExit(f"12622 total {grp['000-999']:,} is not 07430's {lyr_total:,}")
    c = chart()
    if int(c.sum()) != lyr_total:
        raise SystemExit(f"chart sums to {c.sum():,}, not {lyr_total:,}")
    if int(c["Norge"]) != int(grp["000"]):
        raise SystemExit("chart's Norway differs from 12622's")
    # each 12622 group must hold at least the chart's named countries in it
    named_in = {"00-": ["Sverige", "Danmark", "Finland"], "194": ["Tyskland", "Frankrike"],
                "UC": ["Russland", "Ukraina", "Polen", "Storbritannia"],
                "2+4": ["Thailand", "Filippinene"], "698d": ["USA"]}
    rest = {}
    for g, names in named_in.items():
        rest[g] = int(grp[g]) - int(c[names].sum())
        if rest[g] < 0:
            raise SystemExit(f"12622 group {glab[g]} is smaller than the chart's countries in it")
    if sum(rest.values()) + int(grp["990"]) != int(c["Resten"]):
        raise SystemExit("12622 groups less the chart's countries do not make the chart's Resten")
    print(f"  chart (2026) matches 12622: {lyr_total:,}; Resten {c['Resten']} by group "
          + ", ".join(f"{glab[g]} {v}" for g, v in rest.items()))

    sv = article()
    bar = {"RU": sv["RU"] - int(c["Russland"]), "UA": sv["UA"] - int(c["Ukraina"]), "TJ": sv["TJ"]}
    for iso, name in (("PH", "Filippinene"), ("TH", "Thailand"), ("DE", "Tyskland"),
                      ("SE", "Sverige")):
        if sv[iso] != int(c[name]):
            print(f"  !! {iso}: {sv[iso]} on Svalbard, {c[name]} in Longyearbyen")
    left = bar_total - sum(bar.values())
    if not 0 <= left <= 20:
        raise SystemExit(f"Barentsburg: {bar} leaves {left} of {bar_total}")
    print(f"  Barentsburg and Pyramiden {bar_total}: {bar}, {left} unassigned")

    rows = []
    for name, n in c.items():
        iso = CHART_ISO[name]
        m = {NORWEGIAN: 1.0} if iso == "NO" else ({OTHER: 1.0} if iso is None else mix(iso, "no"))
        rows += [(LYR, node, n * s) for node, s in m.items()]
    don = donetsk_mix()
    for iso, n in bar.items():
        m = don if iso == "UA" else mix(iso, "no")
        rows += [(BAR, node, n * s) for node, s in m.items()]
    rows.append((BAR, OTHER, left))
    rows.append((HOR, POLISH, hor_total))
    df = pd.DataFrame(rows, columns=["geo_id", "node", "count"])
    df = df.groupby(["geo_id", "node"], as_index=False)["count"].sum()

    out = []
    for u, g in df.groupby("geo_id"):
        want = {LYR: lyr_total, BAR: bar_total, HOR: hor_total}[u]
        s = g.set_index("node")["count"]
        fl = s.apply(int)
        fl[(s - fl).sort_values(ascending=False).index[:want - int(fl.sum())]] += 1
        for node, n in fl.items():
            if n > 0:
                out.append(dict(geo_id=u, geo_level="svalbard", geo_name=NAMES[u],
                                source_category=node, count=int(n), tier="derived", year=2026,
                                source_id="ssb_07430_12622_2026"))
    out = pd.DataFrame(out)
    tot = out.groupby("geo_id")["count"].sum().to_dict()
    if tot != {LYR: lyr_total, BAR: bar_total, HOR: hor_total}:
        raise SystemExit(f"unit totals moved: {tot}")
    t = out.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(f"Svalbard {int(t.sum()):,} people, {len(t)} nodes:")
    for node, v in t.head(10).items():
        print(f"   {v:>6,}  {node}")
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(out)} rows")
    return out


def place(totals):
    import geopandas as gpd
    import os as _os

    sv = gpd.read_file(RD_LAU, where="unit='NO-21'").to_crs(4326)
    if round(sv["pop"].sum()) != sum(totals.values()):
        raise SystemExit(f"religiondots' Svalbard weights sum to {sv['pop'].sum():,.0f}, not "
                         f"{sum(totals.values()):,}: its 07430 vintage differs")
    k = gpd.read_file(KONTUR_SJ).to_crs(4326)
    sv["kontur"] = gpd.sjoin(gpd.GeoDataFrame(geometry=sv.representative_point(), crs=4326),
                             k[["population", "geometry"]], how="left",
                             predicate="within")["population"].groupby(level=0).first()
    sv["kontur"] = sv["kontur"].fillna(0.0)
    cent = sv.to_crs(32633).centroid
    sv["unit"], sv["w"] = LYR, 0.0
    for name, (lat, lon, km) in POINTS.items():
        pt = gpd.GeoSeries(gpd.points_from_xy([lon], [lat]), crs=4326).to_crs(32633).iloc[0]
        near = (cent.distance(pt) <= km * 1000) & (sv["name"] == "Svalbard")
        if not near.any() or sv.loc[near, "kontur"].sum() <= 0:
            raise SystemExit(f"no populated hex near {name}")
        sv.loc[near, "name"] = name
    lyr, nya = sv["name"] == "Longyearbyen", sv["name"] == "Ny-Alesund"
    bar, pyr = sv["name"] == "Barentsburg", sv["name"] == "Pyramiden"
    hor = sv["name"] == "Hornsund"

    def spread(mask, n, by):
        sv.loc[mask, "w"] = sv.loc[mask, by] / sv.loc[mask, by].sum() * n

    spread(lyr, totals[LYR] - NY_ALESUND, "pop")
    spread(nya, NY_ALESUND, "kontur")
    spread(bar, totals[BAR] - PYRAMIDEN, "pop")
    spread(pyr, PYRAMIDEN, "kontur")
    sv.loc[bar | pyr, "unit"] = BAR
    sv.loc[hor, "unit"] = HOR
    spread(hor, totals[HOR], "pop")
    got = sv.groupby("unit")["w"].sum().round().astype(int).to_dict()
    if got != totals:
        raise SystemExit(f"placement weights {got} differ from the counts {totals}")
    print("  placement: " + ", ".join(f"{n} {int(m.sum())} hexes" for n, m in
                                       (("Longyearbyen", lyr), ("Ny-Alesund", nya),
                                        ("Barentsburg", bar), ("Pyramiden", pyr),
                                        ("Hornsund", hor)))
          + f"; {int((sv['w'] == 0).sum())} other hexes at weight 0")

    main = gpd.read_file(HEX, layer="hexes")
    main = main[~main["unit"].astype(str).str.startswith("NO-21")]
    if main["unit"].nunique() != 356:
        raise SystemExit(f"{HEX.name}: {main['unit'].nunique()} mainland units, expected 356")
    add = sv.rename(columns={"w": "pop_new"})[["unit", "pop_new", "geometry"]] \
        .rename(columns={"pop_new": "pop"})
    out = gpd.GeoDataFrame(pd.concat([main, add], ignore_index=True), geometry="geometry",
                           crs=4326)
    tmp = HEX.with_name(HEX.stem + ".part.gpkg")
    out.to_file(tmp, driver="GPKG", layer="hexes")
    _os.replace(tmp, HEX)
    print(f"  {HEX.relative_to(ROOT)}: {len(main):,} mainland hexes kept, {len(add)} Svalbard "
          "hexes appended")


def main():
    out = counts()
    place({u: int(v) for u, v in out.groupby("geo_id")["count"].sum().items()})


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    main()
