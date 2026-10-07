"""Kazakhstan, National Population Census 2021: native language by oblast -> data/normalized/kz.csv.

    python sources/kz_census.py --fetch    query the census engine (~1 min), cache to data/raw/kz/
    python sources/kz_census.py            rebuild data/normalized/kz.csv from the cache

THE SOURCE IS THE CENSUS DASHBOARD'S ENGINE, NOT A PUBLICATION. BNS's printed volume
(«Национальный состав, вероисповедание и владение языками», 2023, table 6, pp. 316-319) prints
native language only nationally and only as "language of one's own nationality" / "language of
another nationality". The census dashboard (stat.gov.kz/ru/instuments/dashboards/28424/) embeds
a Qlik Sense app on qap.stat.gov.kz whose data model is the person records themselves (table
`Население 2009_2021`, one row per enumerated person, 2009 and 2021), and the engine is open to
anonymous callers. religiondots found the route (../religiondots/sources/kz.py, read-only here).
Two fields carry the answer to the native-language question:

  `Родной язык`   on the person table: 17 named languages and "Другой язык" (another language),
                  2021 only (2009 rows read "-"). This is what is drawn.
  `Родной язык2`  on the linked `Владение_языками` table: own nationality's language / another
                  nationality's language, the printed table's two columns.

CHECKS, all asserted:
  1. `Национальность_краткая` x `Родной язык2` reproduces printed table 6's national block for
     all 18 nationality rows, both columns, TO THE PERSON (PRINTED below). That proves the engine
     is the census and that its language answer is the one the volume printed.
  2. The 17 oblasts' totals equal religiondots' census oblast totals to the person; the join to
     KATO is made on that population (all 17 distinct) and then checked by name, the capital
     (Nur-Sultan in 2021, Astana in the engine) being the one allowed disagreement.
  3. Named languages sum to 19,186,015, the census population.
  4. Each oblast's rayons (`КАТО РАЙОН`, 218) sum to it, language by language. The rayon rows
     are what is drawn (geo_level `rayon`, geo_id `R` + the engine's rayon number, which is not
     a KATO code); their names are OWNER3_NAME, joined to boundaries in sources/kz_geo.py.
  5. Printed, not asserted: how often the named field and the own/other flag agree.
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
RD_NORM = os.path.join(ROOT, "..", "religiondots", "data", "normalized", "kz.csv")

HOST = "qap.stat.gov.kz"
APP = "4c82a5bb-b3c9-49ba-bf4a-2eceb365084f"          # "Итоги переписи населения 2021"
SHEET = "063175c3-9502-4a72-a506-69ec3d5f3a99"        # any sheet mints the session cookie
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/128.0.0.0 Safari/537.36")
M2021 = "Sum({<[Год переписи]={'2021'}>} 1)"           # one row per person
DISTINCT = "Count(DISTINCT HASH_IIN)"                   # for the fields on linked tables

F_OBL, F_RAY, F_LANG, F_OWN = "Область", "КАТО РАЙОН", "Родной язык", "Родной язык2"
F_NAT, F_LOC = "Национальность_краткая", "Тип местности"

C_OBL = os.path.join(RAW, "kz_qlik_oblast_language.csv")
C_NAT = os.path.join(RAW, "kz_qlik_oblast_nationality_language.csv")
C_OWN = os.path.join(RAW, "kz_qlik_nationality_own.csv")
C_RAY = os.path.join(RAW, "kz_qlik_rayon_language.csv")
C_LOC = os.path.join(RAW, "kz_qlik_locality_language.csv")
C_META = os.path.join(RAW, "kz_qlik_meta.json")
C_RAYN = os.path.join(RAW, "kz_qlik_rayon_names.csv")
F_RAYNAME = "OWNER3_NAME"
C_VIL = os.path.join(RAW, "kz_qlik_rayon_villages.csv")

NATIONAL = 19_186_015
# Printed volume table 6, "Все население" block (pp. 316-317 of the PDF): own, other.
PRINTED = {
    "Казахи": (13_380_107, 117_784), "Русские": (2_853_224, 128_722),
    "Узбеки": (583_456, 30_591), "Украинцы": (77_258, 310_069),
    "Уйгуры": (247_322, 43_015), "Татары": (129_314, 89_339),
    "Немцы": (77_252, 148_840), "Корейцы": (49_222, 69_228),
    "Турки": (71_935, 13_543), "Азербайджанцы": (108_538, 37_077),
    "Белорусы": (42_299, 34_185), "Дунгане": (74_107, 4_710),
    "Курды": (41_770, 6_110), "Таджики": (45_950, 3_877),
    "Поляки": (7_737, 27_582), "Чеченцы": (26_452, 7_105),
    "Кыргызы": (26_264, 7_920), "Другие национальности": (111_666, 152_445),
}
OWN, OTHER = "язык своей национальности", "язык другой национальности"
LANGS = ["Казахский", "Русский", "Узбекский", "Уйгурский", "Татарский", "Азербайджанский",
         "Турецкий", "Немецкий", "Украинский", "Дунганский", "Корейский", "Таджикский",
         "Белорусский", "Курдский", "Чеченский", "Кыргызский", "Польский", "Другой язык"]
CAPITAL_RENAME = {"Г.АСТАНА": "г.Нур-Султан"}


# ---- fetch ---------------------------------------------------------------------------------
class Engine:
    def __init__(self):
        import requests
        import websocket
        s = requests.Session()
        s.headers["User-Agent"] = UA
        r = s.get(f"https://{HOST}/single/?appid={APP}&sheet={SHEET}", timeout=60)
        if r.status_code != 200:
            raise SystemExit(f"{HOST}/single/ -> {r.status_code}; the dashboard has moved")
        cookie = "; ".join(f"{k}={v}" for k, v in s.cookies.get_dict().items())
        self.ws = websocket.create_connection(f"wss://{HOST}/app/{APP}", timeout=600,
                                              header=[f"User-Agent: {UA}"], cookie=cookie,
                                              origin=f"https://{HOST}")
        auth = json.loads(self.ws.recv()).get("params", {})
        if auth.get("mustAuthenticate"):
            raise SystemExit("the engine now demands authentication; this route is closed")
        self.user = auth.get("userId")
        self.n = 0
        self.doc = self.call(-1, "OpenDoc", [APP, "", "", "", False])["result"]["qReturn"]["qHandle"]

    def call(self, h, method, params):
        self.n += 1
        self.ws.send(json.dumps({"jsonrpc": "2.0", "id": self.n, "handle": h,
                                 "method": method, "params": params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("id") == self.n:
                if "error" in m:
                    raise SystemExit(f"{method}: {m['error']}")
                return m

    def cube(self, dims, measure):
        o = self.call(self.doc, "CreateSessionObject", [{
            "qInfo": {"qType": "languagedots-cube"},
            "qHyperCubeDef": {
                "qDimensions": [{"qDef": {"qFieldDefs": [d]}, "qNullSuppression": False}
                                for d in dims],
                "qMeasures": [{"qDef": {"qDef": measure}}],
                "qInitialDataFetch": [], "qSuppressZero": True, "qSuppressMissing": False,
                "qMode": "S"}}])["result"]["qReturn"]["qHandle"]
        size = self.call(o, "GetLayout", [])["result"]["qLayout"]["qHyperCube"]["qSize"]
        w, h = size["qcx"], size["qcy"]
        rows, top = [], 0
        while top < h:
            mat = self.call(o, "GetHyperCubeData", ["/qHyperCubeDef", [{
                "qTop": top, "qLeft": 0, "qWidth": w,
                "qHeight": max(1, min(5000 // w, h - top))}]])["result"]["qDataPages"][0]["qMatrix"]
            if not mat:
                break
            rows += [[c.get("qText") for c in r[:-1]] + [int(r[-1].get("qNum") or 0)] for r in mat]
            top += len(mat)
        if len(rows) != h:
            raise SystemExit(f"engine returned {len(rows)} of {h} rows for {dims}")
        return rows


def _write(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def fetch():
    e = Engine()
    print(f"engine open as {e.user}")
    os.makedirs(RAW, exist_ok=True)
    _write(C_OBL, ["oblast", "language", "n"], e.cube([F_OBL, F_LANG], M2021))
    _write(C_NAT, ["oblast", "nationality", "language", "n"], e.cube([F_OBL, F_NAT, F_LANG], M2021))
    _write(C_OWN, ["nationality", "language", "own", "n"], e.cube([F_NAT, F_LANG, F_OWN], DISTINCT))
    _write(C_RAY, ["oblast", "rayon_kato", "language", "n"], e.cube([F_OBL, F_RAY, F_LANG], M2021))
    _write(C_LOC, ["locality", "language", "n"], e.cube([F_LOC, F_LANG], M2021))
    # `КАТО РАЙОН` is an engine number, not a KATO code; OWNER3_NAME is the rayon's name
    _write(C_RAYN, ["oblast", "rayon_kato", "rayon_name", "n"],
           e.cube([F_OBL, F_RAY, F_RAYNAME], M2021))
    # villages per rayon: kz_geo.py's witness that a 2022 district's seat was in the 2021 rayon
    # it is dissolved back into
    _write(C_VIL, ["rayon", "village", "n"], e.cube([F_RAY, "Село КАТО"], M2021))
    with open(C_META, "w", encoding="utf-8") as fh:
        json.dump({"host": HOST, "app": APP, "measure": M2021, "user": e.user},
                  fh, ensure_ascii=False, indent=1)
    e.ws.close()
    print("cached", ", ".join(os.path.basename(p) for p in (C_OBL, C_NAT, C_OWN, C_RAY, C_LOC)))


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---- checks --------------------------------------------------------------------------------
def check_printed():
    own = {}
    agree = total = 0
    for r in _read(C_OWN):
        n = int(r["n"])
        k = own.setdefault(r["nationality"], {OWN: 0, OTHER: 0})
        k[r["own"]] += n
        total += n
        # the named field agrees with the flag when "own" goes with the group's own language
        if (r["own"] == OWN) == (_own_lang(r["nationality"]) == r["language"]):
            agree += n
    bad = []
    for nat, (a, b) in PRINTED.items():
        got = own.get(nat, {})
        if (got.get(OWN), got.get(OTHER)) != (a, b):
            bad.append(f"  {nat}: engine {got} vs printed {a:,} / {b:,}")
    if set(own) != set(PRINTED):
        bad.append(f"  nationality rows differ: {sorted(set(own) ^ set(PRINTED))}")
    if bad:
        raise SystemExit("the engine no longer reproduces printed table 6:\n" + "\n".join(bad))
    print(f"check 1: all {len(PRINTED)} nationality rows of printed table 6 reproduce to the "
          f"person, own and other ({total:,} people)")
    print(f"check 5: the named field agrees with the own/other flag for {agree:,} "
          f"({100 * agree / total:.2f}%); 'Другие национальности' counts as agreeing only "
          "when flagged other")


def _own_lang(nat):
    return {"Казахи": "Казахский", "Русские": "Русский", "Узбеки": "Узбекский",
            "Украинцы": "Украинский", "Уйгуры": "Уйгурский", "Татары": "Татарский",
            "Немцы": "Немецкий", "Корейцы": "Корейский", "Турки": "Турецкий",
            "Азербайджанцы": "Азербайджанский", "Белорусы": "Белорусский",
            "Дунгане": "Дунганский", "Курды": "Курдский", "Таджики": "Таджикский",
            "Поляки": "Польский", "Чеченцы": "Чеченский",
            "Кыргызы": "Кыргызский"}.get(nat)


def norm_oblast(s):
    s = str(s).upper().replace("ОБЛАСТЬ", " ").replace("Г.", " ")
    return " ".join(s.split())


def join_kato(by_obl):
    ref = {}
    with open(RD_NORM, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["source_category"] == "Всего":
                ref[int(r["count"])] = (r["geo_id"], r["geo_name"])
    if len(ref) != 17:
        raise SystemExit(f"religiondots' kz.csv has {len(ref)} distinct oblast totals, not 17")
    out = {}
    for ob, langs in by_obl.items():
        pop = sum(langs.values())
        if pop not in ref:
            raise SystemExit(f"{ob}: {pop:,} matches no census oblast total")
        kato, name = ref[pop]
        want = CAPITAL_RENAME.get(ob)
        if want is not None:
            if name != want:
                raise SystemExit(f"{ob} paired with {name}, expected {want}")
        elif norm_oblast(ob) != norm_oblast(name):
            raise SystemExit(f"population pairs {ob} with {name} but the names disagree")
        out[ob] = (kato, name)
    if len({v[0] for v in out.values()}) != 17:
        raise SystemExit("the population join is not one-to-one")
    print("check 2: 17 oblasts joined to KATO on population to the person; 16 confirmed by "
          "name, the capital by CAPITAL_RENAME")
    return out


def main():
    if "--fetch" in sys.argv:
        fetch()
    check_printed()

    by_obl = {}
    for r in _read(C_OBL):
        if r["language"] not in LANGS:
            raise SystemExit(f"unknown language label {r['language']!r}")
        by_obl.setdefault(r["oblast"], {})[r["language"]] = int(r["n"])
    if len(by_obl) != 17:
        raise SystemExit(f"{len(by_obl)} oblasts, expected 17")
    tot = sum(sum(v.values()) for v in by_obl.values())
    if tot != NATIONAL:
        raise SystemExit(f"languages sum to {tot:,}, not {NATIONAL:,}")
    print(f"check 3: 18 language answers sum to {NATIONAL:,}")
    kato = join_kato(by_obl)

    ray = {}
    units = set()
    for r in _read(C_RAY):
        ray[(r["oblast"], r["language"])] = ray.get((r["oblast"], r["language"]), 0) + int(r["n"])
        units.add(r["rayon_kato"])
    bad = [k for k in ray if ray[k] != by_obl[k[0]].get(k[1], 0)]
    bad += [(ob, l) for ob, v in by_obl.items() for l in v if (ob, l) not in ray]
    if bad:
        raise SystemExit(f"rayons do not sum to oblasts for {bad[:5]}")
    print(f"check 4: {len(units)} rayons sum to their oblasts, language by language")

    # rayon names (OWNER3_NAME): one per engine rayon number, the same population, and unique
    # inside each oblast (the same name recurs across oblasts: Abay, Zhambyl, Esil, Aktogay)
    names = {}
    for r in _read(C_RAYN):
        if r["rayon_kato"] in names:
            raise SystemExit(f"rayon {r['rayon_kato']} has two names")
        names[r["rayon_kato"]] = (r["oblast"], r["rayon_name"], int(r["n"]))
    ray_pop = {}
    for r in _read(C_RAY):
        ray_pop[r["rayon_kato"]] = ray_pop.get(r["rayon_kato"], 0) + int(r["n"])
    if set(names) != set(ray_pop) or any(names[k][2] != ray_pop[k] for k in ray_pop):
        raise SystemExit("rayon names and rayon counts disagree on units or populations")
    seen = {}
    for k, (ob, nm, _n) in names.items():
        if (ob, nm) in seen:
            raise SystemExit(f"{nm} twice in {ob}")
        seen[(ob, nm)] = k
    print(f"check 4b: {len(names)} rayon names, one each, unique inside every oblast")

    rows = []
    for ob, langs in sorted(by_obl.items(), key=lambda kv: kato[kv[0]][0]):
        g, name = kato[ob]
        for l in LANGS:
            if langs.get(l):
                rows.append({"geo_id": g, "geo_level": "oblast", "unit_name": name,
                             "parent": "", "source_category": l, "count": langs[l],
                             "tier": "measured"})
    for r in sorted(_read(C_RAY), key=lambda r: (kato[r["oblast"]][0], int(r["rayon_kato"]),
                                                 LANGS.index(r["language"]))):
        if int(r["n"]) > 0:
            rows.append({"geo_id": f"R{int(r['rayon_kato']):03d}", "geo_level": "rayon",
                         "unit_name": names[r["rayon_kato"]][1], "parent": kato[r["oblast"]][0],
                         "source_category": r["language"], "count": int(r["n"]),
                         "tier": "measured"})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["geo_id", "geo_level", "unit_name", "parent",
                                           "source_category", "count", "tier"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")
    if sum(r["count"] for r in rows if r["geo_level"] == "rayon") != NATIONAL:
        raise SystemExit("rayon rows do not sum to the census population")
    nat = {l: sum(v.get(l, 0) for v in by_obl.values()) for l in LANGS}
    for l, n in sorted(nat.items(), key=lambda kv: -kv[1]):
        print(f"  {l:<16} {n:>11,}  {100 * n / NATIONAL:6.2f}%")


if __name__ == "__main__":
    main()
