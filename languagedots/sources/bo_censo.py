"""Bolivia: INE, Censo de Poblacion y Vivienda 2024, variable PERSONA.IDIOMAT, question 34.1
"Primer idioma o lengua en el que aprendio a hablar en su ninez" -- the language a person first
learned to speak as a child, i.e. mother tongue. Asked of everyone.

    python sources/bo_censo.py --fetch     five REDATAM queries, a few minutes
    python sources/bo_censo.py             normalise from data/raw/bo/

Writes data/normalized/bo.csv, one row per (municipality, census label), and
data/raw/bo/bo_munic_pop.csv, every municipality's census count (all ages), for the hex checks.

THE SOURCE IS INE'S OWN REDATAM SERVER (https://redatam.ine.gob.bo, base PHCCEN24ESPV1), the
route religiondots already uses for Bolivia's 2024 count (../religiondots/sources/bo_census.py):
open, no login, no terms gate. Its certificate chain is incomplete, so verification is off.
PROGRED (a free Redatam program) refuses AREALIST on this variable ("Too many categories": it is a
recode of a sparse code list), so every table here is the web form's FREQUENCY with AREABREAK,
which prints one LABELLED table per area: nothing is paired by position.

THE UNIVERSE IS INE'S OWN TABULATION UNIVERSE: aged 4 and over AND usually resident in Bolivia
(question 36 "here, in this municipality" or "in another municipality of the country"). The
variable was asked of everyone, but INE publishes it for that universe, and a run on it
reproduces every national figure INE released (checked below) to the person. Infants are mostly
"does not speak" or "not specified", so drawing every age would add little but noise.

LABELS. The variable has 79 categories: 36 constitutional native languages, Castellano, 37
foreign languages, "Afroboliviano", "Joaquiniano", "Otras declaraciones", "Otro idioma
extranjero", sign language and "Sin especificar"; "No Aplica" is "does not speak" (its name in
the derived variable IDIOMA_MAT). Some labels come off the server double-encoded (UTF-8 read as
Latin-1, "KabineÃ±a"); `fix` repairs them, and the repaired set is asserted.

CHECKS
  * 343 municipality areas: COD-AB's 339 plus exactly four the 2024 census counts on their own
    (three indigenous territories, TIOC, and San Pedro de Macha), asserted by code and name.
  * every area's labels sum to its own Total.
  * municipalities rebuild the PROVIN break and the DEPTO break on every cell (separate queries),
    and the national RESUMEN.
  * INE's published national figures for this universe (INE's 2024 language release, as
    reported in the press, Sept. 2025: eju.tv, La Epoca, El Fulgor) are reproduced exactly: 24
    labels plus "no language" and "not specified".
  * municipality populations (SEXO by MUNIC, all ages) sum to the 11,365,333 counted.
"""

import csv
import html
import os
import re
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "bo")
OUT = os.path.join(ROOT, "data", "normalized", "bo.csv")
POP_OUT = os.path.join(RAW, "bo_munic_pop.csv")

HOST = "https://redatam.ine.gob.bo"
BASE = "PHCCEN24ESPV1"
VAR = "PERSONA.IDIOMAT"
UNIVERSE = "PERSONA.EDAD >= 4 AND PERSONA.LUGRES <= 2"
CENSUS_POPULATION = 11_365_333
EXPECTED = {"DEPTO": 9, "PROVIN": 113, "MUNIC": 343}

# In the server's order, after `fix`.
CATEGORIES = [
    "Araona", "Aymara", "Baure", "Bésiro", "Canichana", "Castellano", "Kabineña", "Cayubaba",
    "Chácobo", "Tsimane´", "Ese Ejja", "Guaraní", "Guarasu´we", "Gwarayu", "Itonama", "Leco",
    "Macha´juyay Kallawaya", "Machineri", "Maropa", "Mojeño Ignaciano", "Mojeño Trinitario",
    "Moré", "Mosetén", "Movima", "Pacahuara", "Puquina", "Quechua", "Sirionó", "Tacana",
    "Tapiete", "Uru-Chipaya", "Weenhayek", "Yaminawa", "Yuqui", "Yurakaré", "Zamuco",
    "Albanés", "Alemán", "Árabe", "Búlgaro", "Catalán", "Chino", "Coreano", "Croata", "Danés",
    "Finlandés", "Francés", "Holandés", "Húngaro", "Inglés", "Italiano", "Japonés", "Noruego",
    "Portugués", "Rumano", "Ruso", "Serbio", "Sueco", "Tailandés", "Turco", "Ucraniano", "Vasco",
    "Vietnamés", "Hebreo", "Polaco", "Checo", "Griego", "Persa", "Suizo", "Latin", "Taiwanés",
    "Gallego", "Valenciano", "Afroboliviano", "Joaquiniano", "Otras declaraciones",
    "Otro idioma extranjero", "Lenguaje de señas", "Sin especificar",
]
NO_SPEAK = "No Aplica"      # "No habla" in IDIOMA_MAT

# Areas the census counts that COD-AB (339 municipalities) has no polygon for; bo_geo.py folds
# each into the municipality it was carved from.
EXTRA_AREAS = {
    "031304": "TIOC-Raqaypampa",
    "050405": "San Pedro de Macha",
    "051204": "TIOC-Jatun Ayllu Yura",
    "080901": "TIOC-Territorio Indígena Multiétnico",
}

# INE's figures for this universe as released in Sept. 2025 (eju.tv, 2025-09, "Censo 2024: El
# castellano, quechua y aymara predominan..."; the same text in La Epoca and El Fulgor).
PUBLISHED = {
    "Castellano": 8_141_600, "Quechua": 1_395_229, "Aymara": 774_874, "Guaraní": 43_870,
    "Tsimane´": 16_556, "Weenhayek": 4_515, "Mojeño Trinitario": 1_835, "Canichana": 9,
    "Moré": 9, "Cayubaba": 7, "Macha´juyay Kallawaya": 7, "Joaquiniano": 8, "Guarasu´we": 1,
    "Baure": 11, "Machineri": 11, "Pacahuara": 23, "Tapiete": 59, "Leco": 62, "Puquina": 94,
    "Yaminawa": 118, "Yuqui": 246, "Lenguaje de señas": 1_822, "Sin especificar": 74_589,
    NO_SPEAK: 21_443,
}


def fix(s):
    """Undo UTF-8-read-as-Latin-1 in the labels the base stores double-encoded."""
    s = html.unescape(s).replace("\xa0", " ").strip()
    # pair by pair, since one string can hold both a correct "á" and a broken "Ã±" (ZudáÃ±ez)
    return re.sub(r"[\xc2\xc3][\x80-\xbf]",
                  lambda m: m.group(0).encode("latin-1").decode("utf-8"), s)


def _session():
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    return s


def _frequency(s, row, areabreak, universe):
    fields = {"MAIN": "WebServerMain.inl", "BASE": BASE, "LANG": "ESP", "CODIGO": "XXUSUARIOXX",
              "ITEM": "RESPER", "MODE": "RUN", "inputTitle": "", "ROW": row,
              "AREABREAK": areabreak, "SELECTION": "ALL", "INLINESELECTION": "",
              "UNIVERSE": universe, "FILTER": "", "TEXT_FILTER": "", "PERCENT": "OFF",
              "FORMAT": "HTML", "Submit": "Ejecutar"}
    r = s.post(f"{HOST}/binbol/RpWebStats.exe/Frequency?", data=fields, timeout=1800,
               verify=False)
    r.raise_for_status()
    pages = [r.text]
    for link in sorted(set(re.findall(
            r'(?:src|href)="((?:https://redatam\.ine\.gob\.bo)?/redbol/+tempo/[^"]+)"', r.text))):
        t = s.get(link if link.startswith("http") else HOST + link, timeout=1800, verify=False)
        t.raise_for_status()
        pages.append(t.text)
    hits = [p for p in pages if "RESUMEN" in p or ("Total" in p and "<table" in p.lower()
                                                   and "Casos" in p)]
    if len(hits) != 1:
        raise SystemExit(f"{row} by {areabreak or 'nation'}: {len(hits)} result pages; the "
                         "server's output has changed")
    return hits[0]


def fetch():
    os.makedirs(RAW, exist_ok=True)
    s = _session()
    jobs = [(f"bo_{lvl.lower()}_lang.htm", VAR, lvl, UNIVERSE) for lvl in EXPECTED]
    jobs += [("bo_nat_lang.htm", VAR, "", UNIVERSE),
             ("bo_munic_pop.htm", "PERSONA.SEXO", "MUNIC", "")]
    for name, row, brk, uni in jobs:
        dest = os.path.join(RAW, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 2_000:
            print("already have", dest)
            continue
        print("RUN", name)
        body = _frequency(s, row, brk, uni)
        with open(dest, "w", encoding="utf-8", newline="") as fh:
            fh.write(body)
        print(f"  {os.path.getsize(dest):,} bytes -> {dest}")
        time.sleep(1)


def _rows(body):
    out = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S | re.I):
        cells = [fix(re.sub(r"<[^>]+>", "", c))
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S | re.I)]
        cells = [c for c in cells if c]
        if cells:
            out.append(cells)
    return out


def _int(tok, where):
    t = tok.replace(" ", "").replace(".", "").replace(",", "")
    if not re.fullmatch(r"\d+", t):
        raise SystemExit(f"{where}: {tok!r} is not a figure")
    return int(t)


def parse(name, labels):
    """FREQUENCY [with AREABREAK] -> {code: (name, {label: n, 'Total': n, NO_SPEAK: n})};
    the national RESUMEN (or an unbroken table) under code 'NATIONAL'."""
    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run with --fetch")
    areas, cur = {}, None
    for r in _rows(open(path, encoding="utf-8").read()):
        m = re.match(r"AREA # (\d+)$", r[0])
        if m:
            cur = m.group(1)
            if cur in areas:
                raise SystemExit(f"{name}: area {cur} twice")
            areas[cur] = (r[1] if len(r) > 1 else "", {})
            continue
        if r[0] == "RESUMEN" or (cur is None and r[0] in labels):
            cur = "NATIONAL"
            areas.setdefault(cur, ("national", {}))
            if r[0] == "RESUMEN":
                continue
        if cur is None:
            continue
        cnt = areas[cur][1]
        lab = r[0].rstrip(" :")
        if lab in labels or lab in ("Total", NO_SPEAK):
            if lab in cnt:
                raise SystemExit(f"{name} {cur}: {lab} twice")
            cnt[lab] = _int(r[1], f"{name} {cur} {lab}")
        elif len(r) >= 2 and re.fullmatch(r"[\d ]+", r[1]) and cur is not None \
                and not r[0].startswith(("Fuente", "Procesado")):
            raise SystemExit(f"{name} {cur}: unknown label {r[0]!r} -- the category list "
                             "changed; revisit taxonomy/bo2024.py first")
    return areas


def main():
    if "--fetch" in sys.argv:
        fetch()
    ok = True
    lv = {lvl: parse(f"bo_{lvl.lower()}_lang.htm", CATEGORIES) for lvl in EXPECTED}
    nat = parse("bo_nat_lang.htm", CATEGORIES)["NATIONAL"][1]
    allc = CATEGORIES + ["Total", NO_SPEAK]

    for lvl, n in EXPECTED.items():
        got = {k: v for k, v in lv[lvl].items() if k != "NATIONAL"}
        good = len(got) == n
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {lvl}: {len(got)} areas (expected {n})")
    mun = {k: v for k, v in lv["MUNIC"].items() if k != "NATIONAL"}

    bad = [c for c, (_, v) in mun.items()
           if sum(v.get(x, 0) for x in CATEGORIES) != v.get("Total")]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} labels sum to Total in all {len(mun)} municipalities "
          f"({len(bad)} failures) {bad[:3]}")

    for lvl, k in (("PROVIN", 4), ("DEPTO", 2)):
        coarse = {c: v for c, (_, v) in lv[lvl].items() if c != "NATIONAL"}
        rolled = {}
        for code, (_, v) in mun.items():
            acc = rolled.setdefault(code[:k], dict.fromkeys(allc, 0))
            for c in allc:
                acc[c] += v.get(c, 0)
        bad = ([("codes", sorted(set(rolled) ^ set(coarse)))] if set(rolled) != set(coarse)
               else [(x, c) for x in coarse for c in allc if rolled[x][c] != coarse[x].get(c, 0)])
        ok &= not bad
        print(f"  {'OK ' if not bad else 'BAD'} municipalities rebuild the {lvl} break on all "
              f"{len(coarse) * len(allc):,} cells ({len(bad)} failures) {bad[:3]}")

    tot = {c: sum(v.get(c, 0) for _, v in mun.values()) for c in allc}
    for label, ref in (("the MUNIC break's RESUMEN", lv["MUNIC"]["NATIONAL"][1]),
                       ("an unbroken national run", nat)):
        bad = [c for c in allc if tot[c] != ref.get(c, 0)]
        ok &= not bad
        print(f"  {'OK ' if not bad else 'BAD'} municipalities sum to {label} on all {len(allc)} "
              f"cells ({len(bad)} failures) {bad[:3]}")

    bad = [(c, tot[c], v) for c, v in PUBLISHED.items() if tot[c] != v]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} INE's {len(PUBLISHED)} published national figures "
          f"for this universe reproduced ({len(bad)} failures) {bad}")

    extra = {c: n for c, (n, _) in mun.items() if c in EXTRA_AREAS}
    good = extra == EXTRA_AREAS
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the four areas COD-AB lacks, by code and name: {extra}")

    pop = parse("bo_munic_pop.htm", ["Mujer", "Hombre"])
    pop = {c: v["Total"] for c, (_, v) in pop.items() if c != "NATIONAL"}
    good = set(pop) == set(mun) and sum(pop.values()) == CENSUS_POPULATION
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {len(pop)} municipality populations, all ages, sum to "
          f"{sum(pop.values()):,} (census {CENSUS_POPULATION:,})")
    bad = [c for c in mun if mun[c][1]["Total"] + mun[c][1].get(NO_SPEAK, 0) > pop.get(c, 0)]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} every municipality's universe is within its "
          f"population ({len(bad)} failures)")

    if not ok:
        raise SystemExit("reconciliation FAILED")

    print(f"\n  national, aged 4+ and resident ({tot['Total']:,} answered, {tot[NO_SPEAK]:,} "
          f"do not speak):")
    for c in sorted(CATEGORIES, key=lambda c: -tot[c]):
        print(f"    {tot[c]:>11,}  {100 * tot[c] / tot['Total']:7.4f}%  {c}")

    rows = []
    for code in sorted(mun):
        nm, v = mun[code]
        for c in CATEGORIES:
            if v.get(c):
                rows.append({"geo_id": code, "geo_level": "municipio", "geo_name": nm,
                             "source_category": c, "count": v[c]})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(POP_OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_id", "name", "pop"])
        for code in sorted(pop):
            w.writerow([code, mun[code][0], pop[code]])
    print(f"\nwrote {OUT} ({len(rows):,} rows) and {POP_OUT}")


if __name__ == "__main__":
    main()
