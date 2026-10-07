"""Belarus: Belstat's 2019 census, native language (and home language) by raion and city.

    python sources/by_census.py --fetch    ~175 small POSTs to census.belstat.gov.by, one PDF
    python sources/by_census.py            normalise from data/raw/by/

-> data/normalized/by.csv, one row per (unit, question, language) with a count above zero.
   `question` is `native` (родной язык, what is drawn) or `home` (язык, на котором обычно
   разговаривают дома, kept for the record and the note).

THE TABLE. Belstat's census database, https://census.belstat.gov.by/ (a React app over a Saiku
OLAP server; the bulletins on belstat.gov.by stop at the oblast). Its cube **F503 / `tb503`**,
"population by nationality, native language and language usually spoken at home", is the only
cube with a language dimension below the oblast: dimensions `tb034` (ATE level 3: raions, cities
of oblast subordination and Minsk's nine city districts), `tb002` urban/rural, `tb001` sex,
`tb059` nationality, `tb016` language (171 values); indicators `rodn_kol_19` (native language,
2019) and `razg_kol_19` (spoken at home, 2019), plus the 2009 pair. The app's choropleth endpoint
answers an anonymous POST:

    POST /sdpn/rest/api/map/l3
    {"cubeKeys": {"table": "tb503", "indicatorKeys": ["rodn_kol_19", "razg_kol_19"],
                  "dimensions": [{"table": "tb016", "tablePK": "iazyk_id", "value": ["29"]}]}}

and returns one value per level-3 unit, summed over every dimension not named (sex, urban/rural,
nationality). One POST per language, plus one with no language filter for the unit totals. Code
lists come from `/sdpn/rest/nsi/simple?code=tb016` and `?code=tb034` (SOATO codes included).
The route was found by reading the app's JS bundle (`static/js/0.*.chunk.js` embeds the cube
definitions). The host's certificate chains to Belarus's national root, which Python does not
trust, so verification is off for this host only; the data is public and nothing is sent.

UNITS. `tb034` lists 142 units; five of them are 2009-only (Polotsk city, and the Vitebsk,
Novopolotsk, Orsha and Gomel city councils) and hold nobody in 2019, asserted by name. That
leaves 137: 118 raions, 10 cities of oblast subordination (Brest, Baranovichi, Pinsk, Vitebsk,
Novopolotsk, Gomel, Grodno, Zhodino, Mogilev, Bobruisk) and Minsk's 9 districts.

CHECKS: the national total 9,413,446; every unit's languages sum to its own total, for both
questions; the five empty units are exactly the five named; units sum to the oblast figures and
the oblast figures equal the census bulletin's table "Распределение населения по использованию
белорусского и русского языков по областям и г.Минску" (Belstat, statistical bulletin on the
2019 census, p.44: population, native Belarusian, native Russian, Belarusian at home, Russian at
home, per oblast and Minsk) to the person, 35 figures; and the cube's level-2 (oblast) endpoint
gives the same oblast totals as the sum of its units.
"""
import csv
import json
import os
import ssl
import sys
import time
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "by")
OUT = os.path.join(ROOT, "data", "normalized", "by.csv")

HOST = "https://census.belstat.gov.by"
CUBE_FILE = os.path.join(RAW, "belstat_tb503_l3.json")
L2_FILE = os.path.join(RAW, "belstat_tb503_l2_total.json")
NSI = {"tb016": os.path.join(RAW, "belstat_nsi_tb016.json"),
       "tb034": os.path.join(RAW, "belstat_nsi_tb034.json")}
BULLETIN_URL = "https://www.belstat.gov.by/upload/iblock/471/471b4693ab545e3c40d206338ff4ec9e.pdf"
BULLETIN = os.path.join(RAW, "belstat_bulletin_2019_obshchaya.pdf")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}

SOURCE_ID = "by_census_2019_tb503"
YEAR = 2019
NATIONAL = 9_413_446
IND = {"rodn_kol_19": "native", "razg_kol_19": "home"}
EMPTY_2019 = {"г. Полоцк", "Витебский горсовет", "Новополоцкий горсовет", "Оршанский горсовет",
              "Гомельский горсовет"}
EXPECTED_UNITS = 137
CITIES = {1401, 1410, 1445, 2401, 2418, 3401, 4401, 6413, 7401, 7410}   # SOATO / 10^6

# The bulletin's p.44 table, transcribed: population; native Belarusian, native Russian;
# Belarusian at home, Russian at home. Keyed by the first digit of SOATO.
BULLETIN_P44 = {
    1: ("Брестская", 1_348_115, 1_037_661, 270_525, 218_686, 1_103_858),
    2: ("Витебская", 1_135_731, 514_029, 585_644, 142_820, 965_571),
    3: ("Гомельская", 1_388_512, 652_694, 693_744, 201_154, 1_156_257),
    4: ("Гродненская", 1_026_816, 559_931, 427_764, 389_116, 616_705),
    5: ("г.Минск", 2_018_281, 982_084, 980_064, 689_115, 1_292_596),
    6: ("Минская", 1_471_240, 876_103, 548_299, 549_638, 884_621),
    7: ("Могилевская", 1_024_751, 472_426, 477_725, 257_235, 698_949),
}
BULLETIN_NATIONAL = (9_413_446, 5_094_928, 3_983_765, 2_447_764, 6_718_557)
BELARUSIAN, RUSSIAN = 29, 108

COLUMNS = ["geo_id", "geo_level", "geo_name", "oblast", "question", "lang_id",
           "source_category", "count", "tier", "year", "source_id"]


def _ctx():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE        # Belarus's national root CA; see the docstring
    return ctx


def _post(path, body):
    req = urllib.request.Request(HOST + path, data=json.dumps(body).encode(),
                                 headers={**UA, "content-type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, context=_ctx(), timeout=180) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:                      # noqa: BLE001
            if attempt == 3:
                raise
            print(f"    retry {path}: {e}")
            time.sleep(5 * (attempt + 1))


def _query(level, lang_id=None):
    dims = [] if lang_id is None else [{"table": "tb016", "tablePK": "iazyk_id",
                                        "value": [str(lang_id)]}]
    body = {"cubeKeys": {"table": "tb503", "indicatorKeys": list(IND), "dimensions": dims}}
    res = _post(f"/sdpn/rest/api/map/{level}", body)
    out = {}
    for ind in res:
        out[ind["indicatorKey"]] = {str(v["id"]): v["value"] or 0 for v in ind["mapValues"]}
    if set(out) != set(IND):
        raise SystemExit(f"language {lang_id}: indicators {sorted(out)}")
    return out


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for code, path in NSI.items():
        if not os.path.exists(path):
            doc = _post(f"/sdpn/rest/nsi/simple?code={code}", {})
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(doc, fh, ensure_ascii=False)
            print(f"  got {code}: {len(doc['data'])} values")
    if not os.path.exists(CUBE_FILE):
        langs = json.load(open(NSI["tb016"], encoding="utf-8"))["data"]
        cube = {"_total": _query("l3")}
        for n, lang in enumerate(langs, 1):
            cube[str(lang["id"])] = _query("l3", lang["id"])
            if n % 25 == 0:
                print(f"  {n}/{len(langs)} languages")
            time.sleep(0.3)
        with open(CUBE_FILE + ".part", "w", encoding="utf-8") as fh:
            json.dump(cube, fh, ensure_ascii=False)
        os.replace(CUBE_FILE + ".part", CUBE_FILE)
        print(f"  wrote {CUBE_FILE}")
    if not os.path.exists(L2_FILE):
        with open(L2_FILE, "w", encoding="utf-8") as fh:
            json.dump(_query("l2"), fh)
    if not os.path.exists(BULLETIN):
        req = urllib.request.Request(BULLETIN_URL, headers=UA)
        with urllib.request.urlopen(req, context=_ctx(), timeout=300) as r:
            body = r.read()
        with open(BULLETIN, "wb") as fh:
            fh.write(body)
        print(f"  got the bulletin ({len(body):,} bytes)")


def bulletin_text_check():
    """The transcription of p.44 against the PDF's own text, number by number."""
    try:
        import fitz
    except ImportError:
        print("  (PyMuPDF missing; p.44 transcription not re-read)")
        return
    t = fitz.open(BULLETIN)[43].get_text()
    nums = [int(x.replace(" ", "")) for x in
            __import__("re").findall(r"\d{1,3}(?: \d{3})+|\d+", t.split("Городское население")[0])]
    want = list(BULLETIN_NATIONAL) + [v for k in sorted(BULLETIN_P44) for v in BULLETIN_P44[k][1:]]
    got = [n for n in nums if n >= 1000]
    if got[:len(want)] != want:
        raise SystemExit(f"p.44 transcription differs from the PDF text:\n {got[:len(want)]}\n {want}")
    print(f"  bulletin p.44: all {len(want)} transcribed figures re-read from the PDF's text")


def main():
    if "--fetch" in sys.argv:
        fetch()
    for p in [CUBE_FILE, L2_FILE, BULLETIN, *NSI.values()]:
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}; run with --fetch")
    langs = {str(x["id"]): x for x in json.load(open(NSI["tb016"], encoding="utf-8"))["data"]}
    units = {str(x["id"]): x for x in json.load(open(NSI["tb034"], encoding="utf-8"))["data"]}
    cube = json.load(open(CUBE_FILE, encoding="utf-8"))
    total = cube.pop("_total")
    ok = True

    if set(cube) != set(langs):
        raise SystemExit(f"cube languages {len(cube)} against the code list's {len(langs)}")
    empty = {units[u]["ru"] for u, v in total["rodn_kol_19"].items() if v == 0}
    good = empty == EMPTY_2019
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} units empty in 2019 are exactly the five 2009-only ones "
          f"({sorted(empty)})")
    live = sorted((u for u, v in total["rodn_kol_19"].items() if v > 0),
                  key=lambda u: units[u]["soato_k3"])
    good = len(live) == EXPECTED_UNITS and set(total["rodn_kol_19"]) == set(units)
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {len(live)} populated units (expected {EXPECTED_UNITS}), "
          f"the cube's unit list equals the code list's {len(units)}")

    rows = []
    for ind, question in IND.items():
        nat = sum(total[ind][u] for u in live)
        good = nat == NATIONAL
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {question}: national total {nat:,} ({NATIONAL:,})")
        bad = []
        for u in live:
            s = sum(cube[l][ind].get(u, 0) for l in cube)
            if s != total[ind][u]:
                bad.append((units[u]["ru"], s, total[ind][u]))
        ok &= not bad
        print(f"  {'OK ' if not bad else 'BAD'} {question}: every unit's languages sum to its "
              f"total {bad[:5] if bad else ''}")
        for u in live:
            k = units[u]["soato_k3"] // 1_000_000
            level = "city" if k in CITIES else ("minsk_district" if k // 100 == 52 else "raion")
            for l, vals in cube.items():
                n = vals[ind].get(u, 0)
                if n:
                    rows.append(dict(geo_id=units[u]["soato_k3"], geo_level=level,
                                     geo_name=units[u]["ru"], oblast=units[u]["parent_ru"],
                                     question=question, lang_id=int(l),
                                     source_category=langs[l]["ru"], count=n, tier="measured",
                                     year=YEAR, source_id=SOURCE_ID))
    lv = {}
    for r in rows:
        if r["question"] == "native" and r["lang_id"] == BELARUSIAN:
            lv[r["geo_level"]] = lv.get(r["geo_level"], 0) + 1
    good = lv == {"raion": 118, "city": 10, "minsk_district": 9}
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} 118 raions, 10 cities, 9 Minsk districts ({lv})")

    # oblast sums against the bulletin's p.44 and against the cube's own level-2 endpoint
    l2 = json.load(open(L2_FILE, encoding="utf-8"))
    bulletin_text_check()
    bad = []
    for ob, (name, pop, nbe, nru, hbe, hru) in BULLETIN_P44.items():
        mine = [u for u in live if units[u]["soato_k3"] // 1_000_000_000 == ob]
        got = (sum(total["rodn_kol_19"][u] for u in mine),
               sum(cube[str(BELARUSIAN)]["rodn_kol_19"].get(u, 0) for u in mine),
               sum(cube[str(RUSSIAN)]["rodn_kol_19"].get(u, 0) for u in mine),
               sum(cube[str(BELARUSIAN)]["razg_kol_19"].get(u, 0) for u in mine),
               sum(cube[str(RUSSIAN)]["razg_kol_19"].get(u, 0) for u in mine))
        if got != (pop, nbe, nru, hbe, hru):
            bad.append((name, got, (pop, nbe, nru, hbe, hru)))
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} units sum to the bulletin's p.44 in all 35 oblast "
          f"figures (population, native and home Belarusian and Russian) {bad or ''}")
    l2pop = sorted(l2["rodn_kol_19"].values())
    good = l2pop == sorted(v[1] for v in BULLETIN_P44.values())
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the cube's oblast endpoint gives the bulletin's seven "
          "populations")

    print("\n  national, native language (largest 15) | at home:")
    nat = {}
    for r in rows:
        nat.setdefault(r["source_category"], {}).setdefault(r["question"], 0)
        nat[r["source_category"]][r["question"]] += r["count"]
    for lab, d in sorted(nat.items(), key=lambda kv: -kv[1].get("native", 0))[:15]:
        print(f"    {d.get('native', 0):>10,}  {100 * d.get('native', 0) / NATIONAL:6.2f}%  "
              f"| {d.get('home', 0):>10,}  {lab}")
    print(f"    {sum(1 for d in nat.values() if d.get('native'))} languages named as native, "
          f"{sum(1 for d in nat.values() if d.get('home'))} at home")
    if not ok:
        raise SystemExit("reconciliation FAILED")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT} ({len(rows):,} rows)")


if __name__ == "__main__":
    main()
