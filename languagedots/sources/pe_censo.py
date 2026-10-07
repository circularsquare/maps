"""Peru: INEI, Censos Nacionales 2017 (XII de Poblacion, VII de Vivienda y III de Comunidades
Indigenas), variable C5P11, "P3a+: Idioma o lengua con el que aprendio hablar" -- the language a
person learned to speak in childhood, i.e. mother tongue, asked of everyone aged 3 and over.

    python sources/pe_censo.py --fetch     five REDATAM queries, a few minutes
    python sources/pe_censo.py             normalise from data/raw/pe/

Writes data/normalized/pe.csv: one row per (district, census label), 1,874 districts.

THE SOURCE IS INEI'S OWN REDATAM SERVER, running a program we send it, the route religiondots
already uses for Peru's religion question (../religiondots/sources/pe.py; open, no login, no
terms gate). The precoded form had 13 answers plus "other native language, specify"; INEI coded
the write-ins into 31 more named languages, so the variable has 39 named native languages, a
small uncoded native remainder, Spanish, Portuguese, other foreign, Peruvian Sign Language,
"does not hear or speak" and "does not know": 46 categories.

CHECKS
  * 1,874 districts, 196 provinces, 25 departments; districts rebuild provinces and provinces
    rebuild departments on every cell (three separate queries).
  * the labelled categories sum to the row's own Total on every district.
  * a second endpoint (the web form's FREQUENCY with AREABREAK, one labelled table per district)
    gives the same count for every (district, label): this is what would catch a column landing
    under the wrong label in the AREALIST output, since that pairing is by position.
  * INEI's published national figures for the population aged 5 and over (Peru: Perfil
    sociodemografico, Censo 2017, ch. on language): Castellano 22,209,686, Quechua 3,735,682,
    Aimara 444,389, otra lengua nativa 210,017, otro tipo de lengua 83,981. A 5+ query is run and
    must reproduce all five, which also shows how INEI grouped the labels.
"""

import csv
import html
import os
import re
import sys
import time
import urllib.parse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "pe")
OUT = os.path.join(ROOT, "data", "normalized", "pe.csv")

HOST = "https://censos2017.inei.gob.pe/bininei"
BASE = "CPV2017DI"
VAR = "Poblacio.C5P11"
AGE = "Poblacio.C5P41"

# In REDATAM's order. The 31 after "No escucha, ni habla" are INEI's coding of the write-ins.
CATEGORIES = [
    "Quechua", "Aimara", "Ashaninka", "Awajún / Aguaruna", "Shipibo - Konibo",
    "Shawi/Chayahuita", "Matsigenka/Machiguenga", "Achuar", "Otra lengua nativa u originaria",
    "Castellano", "Portugués", "Otra lengua extranjera", "Lengua de señas peruanas",
    "No escucha, ni habla",
    "Wampis", "Kichwa", "Nomatsigenga", "Tikuna", "Urarina", "Yine", "Yanesha",
    "Kandozi-Chapra", "Kakataibo", "Matses", "Kukama kukamiria", "Yagua", "Secoya", "Harakbut",
    "Yaminahua", "Jaqaru", "Murui-Muinani", "Kakinte", "Amahuaca", "Arabela", "Nahua",
    "Ese Eja", "Capanahua", "Maijuna", "Ocaina", "Sharanahua", "Cauqui", "Shiwilu",
    "Cashinahua", "Isconahua", "Omagua",
    "No sabe / No responde",
]
TOTAL = "Total"
EXPECTED = {"dep": 25, "prov": 196, "dist": 1874}
NATIONAL_3PLUS = 27_946_060
CENSUS_POPULATION = 29_381_884

# INEI, Peru: Perfil sociodemografico (Censo 2017), population aged 5 and over.
PUBLISHED_5PLUS = {
    "Castellano": 22_209_686,
    "Quechua": 3_735_682,
    "Aimara": 444_389,
    "otra lengua nativa": 210_017,
    "otro tipo de lengua": 83_981,
}
OTRO_TIPO = ["Portugués", "Otra lengua extranjera", "Lengua de señas peruanas",
             "No escucha, ni habla"]
NOT_NATIVE = {"Quechua", "Aimara", "Castellano", "No sabe / No responde"} | set(OTRO_TIPO)


def _session():
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    return s


def _collect(s, endpoint, fields, timeout=900):
    r = s.post(f"{HOST}/RpWebStats.exe/{endpoint}?", data=fields, timeout=timeout, verify=False)
    r.raise_for_status()
    m = re.search(r"LFN=([^\"'&<>]+?\.htm)", r.text)
    if not m:
        raise SystemExit("REDATAM returned no output file:\n"
                         + re.sub(r"<[^>]+>", " ", r.text)[:800])
    lfn = html.unescape(m.group(1))
    t = s.get(f"{HOST}/RpWebUtilities.exe/Text?LFN=" + urllib.parse.quote(lfn) + "&TYPE=TMP",
              timeout=timeout, verify=False)
    t.raise_for_status()
    return t.text


def _progred(s, prog):
    return _collect(s, "CmdSet", {
        "MAIN": "WebServerMain.inl", "BASE": BASE, "LANG": "esp", "CODIGO": "XXUSUARIOXX",
        "ITEM": "PROGRED", "MODE": "RUN", "CMDSET": prog, "Submit": "Ejecutar"})


def _save(dest, body):
    if "Tabla vac" in body or "<table" not in body.lower():
        raise SystemExit(f"{dest}: REDATAM returned no table\n{body[:600]}")
    with open(dest, "w", encoding="utf-8", newline="") as fh:
        fh.write(body)
    print(f"  {os.path.getsize(dest):,} bytes -> {dest}")
    time.sleep(1)


def fetch():
    os.makedirs(RAW, exist_ok=True)
    s = _session()
    jobs = [(f"pe_{k}_lang.htm", "arealist", lvl)
            for k, lvl in (("dep", "DEPARTAM"), ("prov", "PROVINCI"), ("dist", "DISTRITO"))]
    jobs += [("pe_dist_break.htm", "break", "Distrito"), ("pe_nat_5plus.htm", "5plus", None)]
    for name, kind, lvl in jobs:
        dest = os.path.join(RAW, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 2_000:
            print("already have", dest)
            continue
        print("RUN", name)
        if kind == "arealist":
            body = _progred(s, f"RUNDEF Job\n SELECTION ALL\nTABLE T\n AS AREALIST\n"
                               f" OF {lvl}, {VAR}\n")
        elif kind == "5plus":
            body = _progred(s, f"RUNDEF Job\n SELECTION ALL\n UNIVERSE {AGE} >= 5\nTABLE T\n"
                               f" AS FREQUENCY\n OF {VAR}\n")
        else:
            body = _collect(s, "Frequency", {
                "MAIN": "WebServerMain.inl", "BASE": BASE, "LANG": "esp",
                "CODIGO": "XXUSUARIOXX", "ITEM": "FREQPOB", "MODE": "RUN", "ROW": VAR,
                "AREABREAK": lvl, "SELECTION": "ALL", "FORMAT": "HTML", "PERCENT": "OFF",
                "WEIGHT": "", "UNIVERSE": "", "FILTER": "", "inputTitle": "",
                "INLINESELECTION": "", "Submit": "Ejecutar"})
        _save(dest, body)


def _rows(body):
    out = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S | re.I):
        cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).replace("\xa0", " ").strip()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S | re.I)]
        cells = [c for c in cells if c]
        if cells:
            out.append(cells)
    return out


def _int(tok, where):
    t = tok.replace(" ", "").replace(",", "").replace(".", "")
    if not re.fullmatch(r"\d+", t):
        raise SystemExit(f"{where}: {tok!r} is not a figure")
    return int(t)


def _read(name):
    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run with --fetch")
    return open(path, encoding="utf-8").read()


def _areal(name, expected):
    rows = _rows(_read(name))
    want = CATEGORIES + [TOTAL]
    header = next((r for r in rows if r[0] == "Código"), None)
    if header is None or header[1:] != want:
        raise SystemExit(f"{name}: columns are {header and header[1:]}, expected {want} -- the "
                         "category list changed; revisit taxonomy/pe2017.py first")
    out = {}
    for r in rows:
        if len(r) != len(want) + 1 or not re.fullmatch(r"\d+", r[0].replace(" ", "")):
            continue
        code = r[0].strip()
        if code in out:
            raise SystemExit(f"{name}: area {code} twice")
        out[code] = dict(zip(want, (_int(t, f"{name} {code}") for t in r[1:])))
    if len(out) != expected:
        raise SystemExit(f"{name}: {len(out)} areas, expected {expected}")
    return out


def _break(name, expected):
    """FREQUENCY with AREABREAK: {code: ((dept, prov, dist), {label: count})}."""
    body = _read(name)
    parts = re.split(r"AREA\s*#\s*(\d+)\s*</td>\s*<td[^>]*>([^<]*)", body)
    out = {}
    for i in range(1, len(parts) - 2, 3):
        code, label, chunk = parts[i], parts[i + 1], parts[i + 2]
        label = html.unescape(label).replace("\xa0", " ").strip()
        m = re.match(r"^(.*?),\s*(.*?),\s*distrito:\s*(.*)$", label)
        if m:
            nm = (m.group(1).strip(), m.group(2).strip(), m.group(3).strip())
        else:
            m = re.match(r"^(.*?),\s*distrito:\s*(.*)$", label)   # Callao: dept = province
            if not m:
                raise SystemExit(f"{name}: cannot parse area label {label!r}")
            nm = (m.group(1).strip(), m.group(1).strip(), m.group(2).strip())
        # the last district's chunk runs on into the national RESUMEN table
        chunk = chunk.split("RESUMEN")[0]
        counts = {}
        for r in _rows(chunk):
            # this endpoint mangles the ñ of señas into U+FFFD
            if r[0].startswith("Lengua de se") and r[0].endswith("as peruanas"):
                r[0] = "Lengua de señas peruanas"
            # first occurrence only: the last district's chunk runs on into the national
            # summary table, which repeats every label
            if r[0] in CATEGORIES and len(r) >= 2 and r[0] not in counts:
                counts[r[0]] = _int(r[1], f"{name} {code} {r[0]}")
        if code in out:
            raise SystemExit(f"{name}: {code} twice")
        out[code] = (nm, counts)
    if len(out) != expected:
        raise SystemExit(f"{name}: {len(out)} area headings, expected {expected}")
    return out


def _national_5plus():
    rows = _rows(_read("pe_nat_5plus.htm"))
    got = {r[0]: _int(r[1], "5plus") for r in rows if r[0] in CATEGORIES and len(r) >= 2}
    if set(got) - set(CATEGORIES) or len(got) < 40:
        raise SystemExit(f"5plus: unexpected labels {sorted(got)}")
    return got


def main():
    if "--fetch" in sys.argv:
        fetch()
    dep = _areal("pe_dep_lang.htm", EXPECTED["dep"])
    prov = _areal("pe_prov_lang.htm", EXPECTED["prov"])
    dist = _areal("pe_dist_lang.htm", EXPECTED["dist"])
    brk = _break("pe_dist_break.htm", EXPECTED["dist"])
    nat5 = _national_5plus()
    ok = True
    allc = CATEGORIES + [TOTAL]

    nat = {c: sum(v[c] for v in dist.values()) for c in allc}
    good = nat[TOTAL] == NATIONAL_3PLUS
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} national universe (3+) {nat[TOTAL]:,} "
          f"(expected {NATIONAL_3PLUS:,})")

    bad = [c for c, v in dist.items() if sum(v[x] for x in CATEGORIES) != v[TOTAL]]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} 46 categories sum to Total on all {len(dist)} "
          f"districts ({len(bad)} failures)")

    for label, finer, coarser, k in [("districts rebuild provinces", dist, prov, 4),
                                     ("provinces rebuild departments", prov, dep, 2)]:
        rolled = {}
        for code, v in finer.items():
            acc = rolled.setdefault(code[:k], dict.fromkeys(allc, 0))
            for c in allc:
                acc[c] += v[c]
        bad = ([("codes", sorted(set(rolled) ^ set(coarser)))] if set(rolled) != set(coarser)
               else [(x, c) for x in coarser for c in allc if rolled[x][c] != coarser[x][c]])
        ok &= not bad
        print(f"  {'OK ' if not bad else 'BAD'} {label} on all {len(coarser) * len(allc)} "
              f"cells ({len(bad)} failures) {bad[:3]}")

    # second endpoint, paired by LABEL, against the AREALIST, paired by position
    bad = []
    if set(brk) != set(dist):
        bad.append(("codes", sorted(set(brk) ^ set(dist))[:5]))
    for code, (_, cnt) in brk.items():
        for c in CATEGORIES:
            if cnt.get(c, 0) != dist.get(code, {}).get(c):
                bad.append((code, c, cnt.get(c, 0), dist.get(code, {}).get(c)))
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the AREABREAK tables (labelled rows) equal the "
          f"AREALIST (positional columns) on all {len(dist) * len(CATEGORIES):,} cells "
          f"({len(bad)} failures) {bad[:3]}")

    got = {"Castellano": nat5.get("Castellano", 0), "Quechua": nat5.get("Quechua", 0),
           "Aimara": nat5.get("Aimara", 0),
           "otra lengua nativa": sum(v for c, v in nat5.items() if c not in NOT_NATIVE),
           "otro tipo de lengua": sum(nat5.get(c, 0) for c in OTRO_TIPO)}
    bad = [(c, got[c], PUBLISHED_5PLUS[c]) for c in PUBLISHED_5PLUS if got[c] != PUBLISHED_5PLUS[c]]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} INEI's 5 published national figures (age 5+) "
          f"reproduced by a 5+ query ({len(bad)} failures) {bad}")
    print("      'otra lengua nativa' = every native label but Quechua and Aimara (Jaqaru and "
          "Cauqui included);\n      'otro tipo de lengua' = Portuguese + other foreign + sign "
          "language + 'does not hear or speak'.")

    if not ok:
        raise SystemExit("reconciliation FAILED")

    print(f"\n  national, population aged 3 and over ({nat[TOTAL]:,}; "
          f"{CENSUS_POPULATION - nat[TOTAL]:,} under-3s not asked):")
    for c in sorted(CATEGORIES, key=lambda c: -nat[c]):
        print(f"    {nat[c]:>11,}  {100 * nat[c] / nat[TOTAL]:6.3f}%  {c}")

    rows = []
    for code in sorted(dist):
        (dept, prov_nm, dist_nm), _ = brk[code]
        for c in CATEGORIES:
            if dist[code][c]:
                rows.append({"geo_id": code, "geo_level": "distrito", "geo_name": dist_nm,
                             "province": prov_nm, "department": dept,
                             "source_category": c, "count": dist[code][c]})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT} ({len(rows):,} rows)")


if __name__ == "__main__":
    main()
