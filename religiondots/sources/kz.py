"""Kazakhstan — BNS, National Population Census 2021. **Religion by oblast, COUNTED.**

Writes data/normalized/kz.csv from BNS's own religion × oblast figures. Kazakhstan was drawn
as a modelled country from 2026-09-07 to 2026-09-08; it is not one any more.

**WHERE THE FIGURES COME FROM, AND WHY NOBODY FOUND THEM.** `sources.md` §11u established
four ways that no BNS *publication* cuts religion by region, and all four checks were right.
What none of them touched is the census's own **interactive dashboard**, which is not a
picture: `stat.gov.kz/ru/instuments/dashboards/28424/` embeds sheets served by a **Qlik Sense
server at `qap.stat.gov.kz`**, and a Qlik app carries its data model, not its charts. This
one's data model is the census itself —

    table `Население 2009_2021`   35,195,612 rows   = 16,009,597 (2009) + 19,186,015 (2021)

one row per enumerated person, keyed on a hashed IIN, carrying `Вероисповедание` on the same
row as `Область`, `КАТО РАЙОН`, `Тип местности`, `Национальность` and a hundred more. The
engine will cross-tabulate any pair of them, so **religion × oblast is a query, not a
publication.** The engine's JSON API is open to `anonymous` (`mustAuthenticate: false`); no
key, no login, no terms gate.

**FOUR MARGINS PROVE IT IS THE CENSUS AND NOT SOMEBODY'S MODEL.**

  1. every one of the nine religion totals reproduces the published volume **to the person**
     — Ислам 13,297,775, Православие 3,269,143, ... Иудаизм 7,192;
  2. they sum to 19,186,015, the census population, with no residual;
  3. the urban and rural margins are 11,741,342 and 7,444,673, the volume's own constants;
  4. the 2009 and 2021 row counts sum to the two censuses' published populations exactly.

And it **disagrees** with the ethnicity model that used to draw this country by 7.50% of the
country, which is the fifth proof: a table that matched the published national figures to the
person while differing regionally from anything derived from them can only be their source.

**THE JOIN IS PROVED ON POPULATION, NOT ON NAMES.** The engine labels oblasts in caps
(`СЕВЕРО-КАЗАХСТАНСКАЯ`) and carries no KATO code for them, while the geography is keyed on
KATO. Names would join sixteen of seventeen and fail on the capital, which the census
enumerated as Nur-Sultan and the dashboard calls Astana — the silent single-row miss
`[[reference_name_join_wrong_neighbour]]` is about. So the seventeen oblast populations are
asserted distinct and the join is made **on the population to the person**, with the names
checked afterwards and the one legitimate disagreement named in `CAPITAL_RENAME`.

**WHAT CHANGED ON THE MAP.** The model put 1,439,367 people — 7.50% of Kazakhstan — in the
wrong (oblast, religion) cell. Islam and Orthodoxy were within 4.5% of the truth, so the
north-south pattern was real; refusal (27.8% misplaced) and non-belief (23.0%) were not, and
the model's near-flat refusal layer was an artefact. Measured, refusal runs from 1.19% in
East Kazakhstan to 22.07% in Mangystau. See `sources/kz_model.py`, which still builds the
model and now scores it, and spec §14.25.

**THE CATEGORIES ARE THE SAME NINE** the volume prints, so taxonomy/kz2021.py is unchanged;
the engine spells them lowercase and `NORMALISE` restores the volume's forms. The refusal
cell is drawn on `unknown` for the reason kz2021.py gives (Question 11 offers it as printed
option six, so it is an answer and not a blank) and Kazakhstan is 100.00% drawn.

Usage:
    python sources/kz.py --fetch    query the engine, ~30 s, caches to data/raw/kz/
    python sources/kz.py            rebuild data/normalized/kz.csv from the cache
"""

import csv
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "kz")
OUT = os.path.join(ROOT, "data", "normalized", "kz.csv")

SOURCE_ID = "kz_census_2021"
YEAR = 2021
BASIS = "counted"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# ---- the engine ------------------------------------------------------------------------
HOST = "qap.stat.gov.kz"
APP = "4c82a5bb-b3c9-49ba-bf4a-2eceb365084f"    # "Итоги переписи населения 2021"
SHEET_RELIGION = "063175c3-9502-4a72-a506-69ec3d5f3a99"   # tab 28443, Вероисповедание
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/128.0.0.0 Safari/537.36")

# Rows in the person table are one per person, so the measure counts rows. Set analysis
# rather than a selection: an anonymous session has its own state, and a query that carries
# its own filter cannot be left holding a stale selection between calls.
MEASURE = "Sum({<[Год переписи]={'%d'}>} 1)" % YEAR

F_OBLAST = "Область"
F_RELIGION = "Вероисповедание"
F_LOCALITY = "Тип местности"

F_RAYON = "КАТО РАЙОН"

CACHE_OBLAST = os.path.join(RAW, "kz_qlik_oblast_religion.csv")
CACHE_URBAN = os.path.join(RAW, "kz_qlik_oblast_locality_religion.csv")
CACHE_META = os.path.join(RAW, "kz_qlik_meta.json")
# Not used by the build. The engine will cut religion at 218 rayons as readily as at 17
# oblasts, and the app carries the rayon boundaries as well (`карта_район`, 190 rows with a
# `Район.Line` geometry), so the finer country is a geography job rather than a data hunt.
# Cached so the next session starts from a file instead of rediscovering the engine.
CACHE_RAYON = os.path.join(RAW, "kz_qlik_rayon_religion.csv")

# ---- the volume's own national table, ch.12 p506. Every one of these must come back from
# the engine to the person or the pull is not the census and the build stops (spec §12).
PUBLISHED = {
    "Ислам": 13_297_775, "Православие": 3_269_143, "Католицизм": 18_988,
    "Протестантизм": 9_419, "Иудаизм": 7_192, "Буддизм": 15_458, "Другое": 23_247,
    "Отказались указать": 2_112_653, "Неверующие": 432_140,
}
NATIONAL = 19_186_015
URBAN = 11_741_342
RURAL = 7_444_673
EXPECTED_REGIONS = 17          # the 2021 vintage: 14 oblasts + Astana, Almaty, Shymkent

# The engine lowercases the volume's labels and uses the masculine singular for the
# non-believers. taxonomy/kz2021.py is keyed on the volume's forms, so restore them here
# rather than touching the mapping.
NORMALISE = {
    "ислам": "Ислам", "православие": "Православие", "католицизм": "Католицизм",
    "протестантизм": "Протестантизм", "иудаизм": "Иудаизм", "буддизм": "Буддизм",
    "другое": "Другое", "отказались указать": "Отказались указать",
    "неверующий": "Неверующие",
}
DRAWN = ["Ислам", "Православие", "Католицизм", "Протестантизм", "Иудаизм", "Буддизм",
         "Другое", "Отказались указать", "Неверующие"]

# The one oblast whose name legitimately differs between the two files. The census enumerated
# the capital on 1 September 2021, when it was Nur-Sultan; it was renamed Astana on 17
# September 2022 and the dashboard uses the current name. Both are the same 1,234,042 people
# and the population join is what actually pairs them; this only stops the name check firing.
CAPITAL_RENAME = {"Г.АСТАНА": "г.Нур-Султан"}


def fold(s):
    """Latin/Cyrillic confusables -> Cyrillic. BNS mixes them inside single headers.

    sources/kz.md records the case that cost half an hour: the ethnos workbook types the
    German column as `Hемцы` with a Latin H. Any Cyrillic source can do this and a name
    lookup misses exactly one row without printing differently.
    """
    return (str(s).replace("A", "А").replace("E", "Е")
            .replace("H", "Н").replace("K", "К")
            .replace("M", "М").replace("O", "О")
            .replace("P", "Р").replace("C", "С")
            .replace("T", "Т").replace("X", "Х")
            .replace("a", "а").replace("e", "е")
            .replace("область", " ").strip())


def norm_oblast(s):
    """Fold to a comparable oblast key: caps, no `область`, no `г.`, no punctuation."""
    s = fold(str(s).upper())
    for junk in ("ОБЛАСТЬ", "Г.", "Г ", "  "):
        s = s.replace(junk, " ")
    return " ".join(s.split())


# ---- fetch -------------------------------------------------------------------------------
def _cube(call, doc, dims, page_rows=5000):
    """A straight hypercube over `dims`, measure = people counted in YEAR."""
    o = call(doc, "CreateSessionObject", [{
        "qInfo": {"qType": "religiondots-cube"},
        "qHyperCubeDef": {
            "qDimensions": [{"qDef": {"qFieldDefs": [d]}, "qNullSuppression": False}
                            for d in dims],
            "qMeasures": [{"qDef": {"qDef": MEASURE}}],
            "qInitialDataFetch": [],
            "qSuppressZero": False,
            "qSuppressMissing": False,
            "qMode": "S",
        },
    }])["result"]["qReturn"]["qHandle"]
    size = call(o, "GetLayout", [])["result"]["qLayout"]["qHyperCube"]["qSize"]
    w, h = size["qcx"], size["qcy"]
    rows, top = [], 0
    while top < h:
        pg = call(o, "GetHyperCubeData", ["/qHyperCubeDef", [{
            "qTop": top, "qLeft": 0, "qWidth": w,
            "qHeight": max(1, min(page_rows // max(w, 1), h - top))}]])
        mat = pg["result"]["qDataPages"][0]["qMatrix"]
        if not mat:
            break
        for r in mat:
            rows.append([c.get("qText") for c in r[:-1]] + [int(r[-1].get("qNum") or 0)])
        top += len(mat)
    if len(rows) != h:
        raise SystemExit(f"engine returned {len(rows)} of {h} rows for {dims}")
    return rows


def fetch():
    """Query the engine and cache two cross-tabs. Needs `websocket-client` and `requests`."""
    import requests
    import websocket

    s = requests.Session()
    s.headers["User-Agent"] = UA
    # The single/ page is what mints the X-Qlik-Session cookie the socket needs.
    r = s.get(f"https://{HOST}/single/?appid={APP}&sheet={SHEET_RELIGION}", timeout=60)
    if r.status_code != 200:
        raise SystemExit(f"{HOST}/single/ -> {r.status_code}; the dashboard has moved")
    cookie = "; ".join(f"{k}={v}" for k, v in s.cookies.get_dict().items())

    ws = websocket.create_connection(f"wss://{HOST}/app/{APP}", timeout=300,
                                     header=[f"User-Agent: {UA}"], cookie=cookie,
                                     origin=f"https://{HOST}")
    hello = json.loads(ws.recv())
    auth = hello.get("params", {})
    if auth.get("mustAuthenticate"):
        raise SystemExit("the engine now demands authentication; this route is closed")
    print(f"engine open as {auth.get('userId', '?')} ({auth.get('userDirectory')})")

    counter = [0]

    def call(handle, method, params):
        counter[0] += 1
        ws.send(json.dumps({"jsonrpc": "2.0", "id": counter[0], "handle": handle,
                            "method": method, "params": params}))
        while True:
            m = json.loads(ws.recv())
            if m.get("id") == counter[0]:
                if "error" in m:
                    raise SystemExit(f"{method}: {m['error']}")
                return m

    doc = call(-1, "OpenDoc", [APP, "", "", "", False])["result"]["qReturn"]["qHandle"]

    ob = _cube(call, doc, [F_OBLAST, F_RELIGION])
    ur = _cube(call, doc, [F_OBLAST, F_LOCALITY, F_RELIGION])
    pop = _cube(call, doc, [F_OBLAST])
    ry = _cube(call, doc, [F_OBLAST, F_RAYON, F_RELIGION])
    ws.close()

    os.makedirs(RAW, exist_ok=True)
    _write(CACHE_OBLAST, ["oblast", "religion", "n"], ob)
    _write(CACHE_URBAN, ["oblast", "locality", "religion", "n"], ur)
    _write(CACHE_RAYON, ["oblast", "rayon_kato", "religion", "n"], ry)
    with open(CACHE_META, "w", encoding="utf-8") as fh:
        json.dump({"host": HOST, "app": APP, "sheet": SHEET_RELIGION, "measure": MEASURE,
                   "user": auth.get("userId"), "oblast_pop": {r[0]: r[1] for r in pop},
                   "rayon_units": len({r[1] for r in ry}),
                   "rayon_people": sum(r[-1] for r in ry)},
                  fh, ensure_ascii=False, indent=1)
    print(f"cached {len(ob)} oblast rows, {len(ur)} oblast x locality rows and {len(ry)} "
          f"rayon rows ({len({r[1] for r in ry})} rayons, not used by the build)")


def _write(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


# ---- read and check ----------------------------------------------------------------------
def read_cache():
    if not os.path.exists(CACHE_OBLAST):
        raise SystemExit(f"missing {CACHE_OBLAST} -- run with --fetch first")
    out = {}
    with open(CACHE_OBLAST, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            cat = NORMALISE.get(r["religion"].strip().lower())
            if cat is None:
                raise SystemExit(f"unmapped religion label {r['religion']!r} -- the engine's "
                                 "category list changed; check it against volume ch.12 "
                                 "before touching NORMALISE")
            out.setdefault(r["oblast"].strip(), {})[cat] = int(r["n"])
    if len(out) != EXPECTED_REGIONS:
        raise SystemExit(f"{len(out)} oblasts, expected {EXPECTED_REGIONS} -- has the 2022 "
                         "three-oblast reform reached the dashboard? See sources/kz_geo.md")
    for ob, cats in out.items():
        missing = [c for c in DRAWN if c not in cats]
        if missing:
            raise SystemExit(f"{ob} is missing {missing}")
    return out


def check_national(measured):
    """Every published national total, to the person. This is the whole proof (spec §12)."""
    bad = []
    for cat, want in PUBLISHED.items():
        got = sum(v[cat] for v in measured.values())
        if got != want:
            bad.append(f"  {cat}: engine {got:,} vs volume ch.12 {want:,}")
    total = sum(sum(v.values()) for v in measured.values())
    if total != NATIONAL:
        bad.append(f"  TOTAL: engine {total:,} vs census {NATIONAL:,}")
    if bad:
        raise SystemExit("the engine no longer reproduces the published table:\n"
                         + "\n".join(bad))
    print(f"national margins: all {len(PUBLISHED)} categories reproduce volume ch.12 "
          f"to the person, and sum to {NATIONAL:,}")

    if os.path.exists(CACHE_URBAN):
        loc = {}
        with open(CACHE_URBAN, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                loc[r["locality"]] = loc.get(r["locality"], 0) + int(r["n"])
        if loc.get("Город") != URBAN or loc.get("Село") != RURAL:
            raise SystemExit(f"urban/rural margins {loc} != published {URBAN:,}/{RURAL:,}")
        print(f"urban/rural margins: {URBAN:,} / {RURAL:,}, the volume's own constants")


def join_to_kato(measured):
    """oblast label -> (KATO geo_id, the geo_name the geography is keyed on).

    Proved on the population and not on the name. The seventeen oblast populations are
    asserted distinct first, which is what makes a population join a proof rather than a
    coincidence; the names are then checked for agreement and are allowed to differ only
    where CAPITAL_RENAME says so.
    """
    sys.path.insert(0, HERE)
    from kz_model import read_magnitude          # the ethnos workbook: KATO, name, Всего

    _, regions, _ = read_magnitude()
    ref = {r["kato"]: (r["name"], r["Всего"]) for r in regions}
    if len(ref) != EXPECTED_REGIONS:
        raise SystemExit(f"{len(ref)} KATO regions in the workbook")

    pops = [p for _, p in ref.values()]
    if len(set(pops)) != len(pops):
        raise SystemExit("two oblasts share a population; the population join is not a "
                         "proof any more and this needs a real key (spec §12)")
    by_pop = {p: (k, n) for k, (n, p) in ref.items()}

    out, unmatched = {}, []
    for ob, cats in measured.items():
        pop = sum(cats.values())
        if pop not in by_pop:
            unmatched.append(f"  {ob}: {pop:,} matches no oblast in the workbook")
            continue
        kato, name = by_pop[pop]
        out[ob] = (kato, name)
    if unmatched:
        raise SystemExit("the population join failed:\n" + "\n".join(unmatched))
    if len(set(out.values())) != EXPECTED_REGIONS:
        raise SystemExit("the population join is not one-to-one")

    # now the names, as a second opinion on a join that is already proved
    for ob, (kato, name) in sorted(out.items()):
        want = CAPITAL_RENAME.get(ob)
        if want is not None:
            if name != want:
                raise SystemExit(f"{ob} paired with {name!r}, expected {want!r}")
            print(f"  {kato}  {ob:<24} -> {name}   (renamed since the census; population "
                  f"{sum(measured[ob].values()):,} pairs them)")
            continue
        if norm_oblast(ob) != norm_oblast(name):
            raise SystemExit(
                f"population pairs {ob!r} with {name!r} but the names disagree. One of them "
                "is wrong and the totals will not show it "
                "([[reference_name_join_wrong_neighbour]]); resolve it before building.")
    print(f"join: {EXPECTED_REGIONS} oblasts paired on population to the person, "
          f"{EXPECTED_REGIONS - len(CAPITAL_RENAME)} confirmed by name")
    return out


def main():
    if "--fetch" in sys.argv:
        fetch()
    measured = read_cache()
    check_national(measured)
    pairing = join_to_kato(measured)

    note = ("level=region; COUNTED. BNS's own religion x oblast cross-tabulation of the 2021 "
            "census, taken from the census dashboard's Qlik engine (qap.stat.gov.kz, app "
            f"{APP}), whose data model is the 19,186,015 enumerated person records "
            "themselves. Every national religion total reproduces the published volume's "
            "chapter 12 to the person")

    rows = []
    for ob, (kato, name) in sorted(pairing.items(), key=lambda kv: kv[1][0]):
        cats = measured[ob]
        rows.append({"geo_id": kato, "geo_level": "region", "geo_name": name,
                     "source_category": "Всего", "count": sum(cats.values()),
                     "basis": BASIS, "year": YEAR, "source_id": SOURCE_ID,
                     "note": note + "; universe total, not a religion category"})
        for c in DRAWN:
            rows.append({"geo_id": kato, "geo_level": "region", "geo_name": name,
                         "source_category": c, "count": cats[c], "basis": BASIS,
                         "year": YEAR, "source_id": SOURCE_ID, "note": note})

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT} ({len(rows):,} rows, {EXPECTED_REGIONS} oblasts, "
          f"{len(DRAWN)} categories, basis {BASIS})")


if __name__ == "__main__":
    main()
