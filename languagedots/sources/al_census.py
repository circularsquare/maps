"""Albania: INSTAT, Census of Population and Housing 2011, mother tongue (gjuha amtare).

    python sources/al_census.py --fetch    PxWeb tables + INSTAT's ArcGIS layers into data/raw/al/
    python sources/al_census.py            normalise from data/raw/al/

-> data/normalized/al.csv, two levels:
   `qark`  12 prefectures x 11 categories (table 1.1.14 of the prefecture branch), the DRAWN counts
   `au`    373 administrative units (2011 communes and municipalities) x Albanian, Greek and
           Macedonian, rebuilt from INSTAT's ArcGIS share layers; used only to PLACE those three
           languages inside their qark (countries/al.py), never summed with `qark`

WHY 2011 AND NOT 2023. The 2023 census asked "language usually spoken at home" and published it
(table 1.14, national CE16 and per qark) as Albanian / Other language / Mixed languages / Prefer
not to answer / Not available, nothing else, and 21% of the country falls in the last two
(Berat 26%). No minority language is named at any level. 2011 asked mother tongue and printed
eight languages by name per prefecture with only 0.14% invalid or undetermined. sources/al.md.

THE DATABASE (as religiondots/sources/al.py found it): PxWeb 21.1 at
https://databaza.instat.gov.al:8083/pxweb/sq/DST/, Albanian interface only (the English one is
empty in places), no REST API, so a table is read by posting the selection form back, every value
of every variable selected, and parsing the HTML table that comes back. Numbers print with a dot
as the thousands separator.

THE ARCGIS LAYERS: INSTAT's ArcGIS Online organisation (services7.arcgis.com/E9FE1JuiACmTPbPv)
has administrativeunit_p_mtong{alb,gre,mac}_2011_view, each one full-precision percentage of the
unit's 2011 resident population, and administrativeunit_p_distrib_2011_view, that population.
share * population / 100 comes back within 0.01 of an integer on every unit (asserted), so these
are the census's own counts, not an estimate.

CHECKS: every qark's categories sum to its printed total; the twelve sum to the national table
1.1.15 (Census2011/Census1115) in every category; the 373 unit populations sum to each qark's
printed total; the rebuilt unit counts of Albanian, Greek and Macedonian sum to the qark table's
figures exactly.
"""

import csv
import html
import json
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "al")
OUT = os.path.join(ROOT, "data", "normalized", "al.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
PX = "https://databaza.instat.gov.al:8083/pxweb/sq/DST/"
TABLES = {
    "p1114.html": "START__Census2011__Census_Prefecture/CENSUS_P_1114/",  # by prefecture
    "n1115.html": "START__Census2011/Census1115/",                          # national
}
AGOL = "https://services7.arcgis.com/E9FE1JuiACmTPbPv/arcgis/rest/services"
LAYERS = {
    "distrib": "administrativeunit_p_distrib_2011_view",
    "Shqip": "administrativeunit_p_mtongalb_2011_view",
    "Greqisht": "administrativeunit_p_mtonggre_2011_view",
    "Maqedonisht": "administrativeunit_p_mtongmac_2011_view",
}
FIELD = {"Shqip": "P_MTONGALB", "Greqisht": "P_MTONGGRE", "Maqedonisht": "P_MTONGMAC"}

SOURCE_ID = "al_census_2011_mt"
YEAR = 2011
NATIONAL = 2_800_138
COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count", "tier", "year",
           "source_id", "note"]

# The qark codes INSTAT uses as CODE_PREFECTURE on every ArcGIS layer, the same ids religiondots'
# al_hexes.gpkg carries as `unit` (religiondots/sources/al.py proved them against population).
QARK = {"Berat": "01", "Dibër": "02", "Durrës": "03", "Elbasan": "04", "Fier": "05",
        "Gjirokastër": "06", "Korçë": "07", "Kukës": "08", "Lezhë": "09", "Shkodër": "10",
        "Tiranë": "11", "Vlorë": "12"}
TOTAL = "Gjithsej"
# The national table spells Aromanian `Arumanisht`, the prefecture table `Rumanisht`. Both are
# the census's Aromanian answer (the national figure equals the prefecture sum, asserted below);
# Romanian proper is not a category. One label in al.csv.
RENAME = {"Arumanisht": "Rumanisht",
          # national table prints it without the space
          "E pavlefshme/e papërcaktuar": "E pavlefshme /e papërcaktuar"}
CATS = ["Shqip", "Greqisht", "Maqedonisht", "Rome", "Rumanisht", "Turqisht", "Italisht",
        "Serbokroatisht", "Tjetër", "E pavlefshme /e papërcaktuar"]


def px_fetch(path, dest):
    """Post PxWeb's selection form back with every value selected; save the result page."""
    import requests
    import urllib3
    urllib3.disable_warnings()
    s = requests.Session()
    s.headers.update(UA)
    s.verify = False   # INSTAT's port-8083 certificate does not validate everywhere
    url = PX + path
    t = s.get(url, timeout=120).text
    data = []
    for m in re.finditer(r'<input[^>]*type="hidden"[^>]*>', t):
        n = re.search(r'name="([^"]+)"', m.group(0))
        v = re.search(r'value="([^"]*)"', m.group(0))
        if n:
            data.append((html.unescape(n.group(1)), html.unescape(v.group(1)) if v else ""))
    for m in re.finditer(r'<select[^>]*name="([^"]+)"[^>]*>(.*?)</select>', t, re.S):
        name = html.unescape(m.group(1))
        opts = re.findall(r'<option[^>]*value="([^"]*)"[^>]*>(.*?)</option>', m.group(2), re.S)
        for v, label in opts:
            if "Perqindja" in label or "Përqindja" in label:   # percentages: counts only
                continue
            data.append((name, html.unescape(v)))
    btn = re.search(r'name="([^"]*ButtonViewTable)"[^>]*value="([^"]*)"', t)
    data.append((html.unescape(btn.group(1)), html.unescape(btn.group(2))))
    r = s.post(url, data=data, timeout=180)
    r.raise_for_status()
    if "tableViewLayout" not in r.url:
        raise SystemExit(f"PxWeb did not return a table for {path} (landed on {r.url})")
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write(r.text)
    print(f"  {dest}: {len(r.text):,} chars")


def agol_fetch(layer, dest, geometry=False):
    import requests
    r = requests.get(f"{AGOL}/{layer}/FeatureServer/0/query",
                     params={"where": "1=1", "outFields": "*", "f": "geojson" if geometry else "json",
                             "returnGeometry": "true" if geometry else "false",
                             "outSR": 4326, "resultRecordCount": 2000},
                     headers=UA, timeout=180)
    r.raise_for_status()
    js = r.json()
    n = len(js["features"])
    if n != 373:
        raise SystemExit(f"{layer}: {n} features, expected 373")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(js, fh, ensure_ascii=False)
    print(f"  {dest}: {n} features")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for name, path in TABLES.items():
        dest = os.path.join(RAW, name)
        if not os.path.exists(dest):
            px_fetch(path, dest)
    for key, layer in LAYERS.items():
        geom = key == "distrib"
        dest = os.path.join(RAW, f"{layer}.{'geojson' if geom else 'json'}")
        if not os.path.exists(dest):
            agol_fetch(layer, dest, geometry=geom)


def num(s):
    s = s.strip()
    if not re.fullmatch(r"\d{1,3}(\.\d{3})*", s):
        raise SystemExit(f"not a PxWeb count: {s!r}")
    return int(s.replace(".", ""))


def table_rows(path):
    t = open(path, encoding="utf-8").read()
    out = []
    for tab in re.findall(r"<table.*?</table>", t, re.S):
        for row in re.findall(r"<tr.*?</tr>", tab, re.S):
            cells = [" ".join(html.unescape(re.sub("<[^>]+>", " ", c)).split())
                     for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.S)]
            if cells:
                out.append(cells)
    return out


def read_prefectures():
    """{qark name: {category: count}} from table 1.1.14 by prefecture."""
    got, cur = {}, None
    for cells in table_rows(os.path.join(RAW, "p1114.html")):
        label = cells[0]
        if label in QARK and (len(cells) == 1 or cells[1] == ""):
            cur = label
            got[cur] = {}
        elif cur and len(cells) == 2 and (label == TOTAL or RENAME.get(label, label) in CATS):
            got[cur][RENAME.get(label, label)] = num(cells[1])
    if sorted(got) != sorted(QARK):
        raise SystemExit(f"prefecture table: qarqe {sorted(got)}")
    for q, d in got.items():
        missing = set(CATS + [TOTAL]) - set(d)
        if missing:
            raise SystemExit(f"{q}: missing {missing}")
        s = sum(d[c] for c in CATS)
        if s != d[TOTAL]:
            raise SystemExit(f"{q}: categories sum to {s:,}, printed total {d[TOTAL]:,}")
    return got


def read_national():
    got = {}
    for cells in table_rows(os.path.join(RAW, "n1115.html")):
        label = RENAME.get(cells[0], cells[0])
        if len(cells) == 2 and (label == TOTAL or label in CATS):
            got[label] = num(cells[1])
    return got


def read_units():
    """{au id: dict(qark, name, pop, Shqip, Greqisht, Maqedonisht)} from the ArcGIS layers."""
    units = {}
    gj = json.load(open(os.path.join(RAW, LAYERS["distrib"] + ".geojson"), encoding="utf-8"))
    for f in gj["features"]:
        a = f["properties"]
        units[a["ID_ADMUNIT"]] = dict(qark=a["CODE_PREFECTURE"], name=a["NAME_ADMINUNIT"],
                                      pop=int(a["P_DISTRIB"]))
    worst = 0.0
    for cat, field in FIELD.items():
        js = json.load(open(os.path.join(RAW, LAYERS[cat] + ".json"), encoding="utf-8"))
        seen = set()
        for f in js["features"]:
            a = f["attributes"]
            u = units[a["ID_ADMUNIT"]]
            if a["CODE_PREFECTURE"] != u["qark"]:
                raise SystemExit(f"{a['ID_ADMUNIT']}: qark {a['CODE_PREFECTURE']} vs {u['qark']}")
            x = (a[field] or 0.0) * u["pop"] / 100.0
            worst = max(worst, abs(x - round(x)))
            u[cat] = int(round(x))
            seen.add(a["ID_ADMUNIT"])
        if seen != set(units):
            raise SystemExit(f"{cat}: units differ from the population layer")
    if worst > 0.01:
        raise SystemExit(f"share * population is {worst:.4f} from an integer somewhere: these "
                         "are not counts")
    print(f"  373 units; share x population lands within {worst:.2e} of an integer everywhere")
    return units


def main():
    if "--fetch" in sys.argv:
        fetch()
    pref = read_prefectures()
    nat = read_national()

    # national check
    bad = []
    for c in CATS + [TOTAL]:
        s = sum(pref[q][c] for q in pref) if c != TOTAL else sum(pref[q][TOTAL] for q in pref)
        if s != nat.get(c):
            bad.append((c, s, nat.get(c)))
    if bad or nat[TOTAL] != NATIONAL:
        raise SystemExit(f"prefectures against national table 1.1.15: {bad}")
    print(f"  12 qarqe sum to national table 1.1.15 in all {len(CATS)} categories; "
          f"total {NATIONAL:,}")
    for c in CATS:
        print(f"    {c:<32}{nat[c]:>10,}  {nat[c] / NATIONAL:7.3%}")

    units = read_units()
    code2name = {v: k for k, v in QARK.items()}
    for code, name in code2name.items():
        us = [u for u in units.values() if u["qark"] == code]
        p = sum(u["pop"] for u in us)
        if p != pref[name][TOTAL]:
            raise SystemExit(f"{name}: units sum to {p:,}, table says {pref[name][TOTAL]:,}")
        for cat in FIELD:
            s = sum(u[cat] for u in us)
            if s != pref[name][cat]:
                raise SystemExit(f"{name} {cat}: units sum to {s:,}, table says {pref[name][cat]:,}")
    print("  unit populations and unit Albanian/Greek/Macedonian sum to every qark's figures exactly")

    rows = []
    for name, d in pref.items():
        for c in CATS:
            rows.append(dict(geo_id=QARK[name], geo_level="qark", geo_name=name,
                             source_category=c, count=d[c], tier="measured", year=YEAR,
                             source_id=SOURCE_ID, note="INSTAT PxWeb CENSUS_P_1114"))
    for au, u in sorted(units.items()):
        for c in FIELD:
            rows.append(dict(geo_id=au, geo_level="au", geo_name=u["name"],
                             source_category=c, count=u[c], tier="measured", year=YEAR,
                             source_id=SOURCE_ID,
                             note=f"qark={u['qark']}; pop2011={u['pop']}; ArcGIS {LAYERS[c]}"))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}: {len(rows):,} rows")

    # where the minorities are, for the record
    for c in ("Greqisht", "Maqedonisht"):
        top = sorted(units.items(), key=lambda kv: -kv[1][c])[:8]
        print(f"  {c}: " + ", ".join(f"{u['name']} {u[c]:,} ({u[c] / u['pop']:.0%})"
                                      for _, u in top))


if __name__ == "__main__":
    main()
