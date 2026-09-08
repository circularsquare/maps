"""Peru — INEI, Censos Nacionales 2017: XII de Población, VII de Vivienda y III de
Comunidades Indígenas, variable C5P26.

Reads (or fetches) data/raw/pe/ and writes data/normalized/pe.csv.

**THE SOURCE IS INEI'S OWN REDATAM SERVER RUNNING A QUERY WE WROTE** (`sources.md` §11x,
§11y). `censos2017.inei.gob.pe/bininei/` is unauthenticated, has no terms gate, and will run
an arbitrary Redatam+SP program against the 2017 census microdata. The religion variable is
`Poblacio.C5P26`, *"P12a+: Religión que profesa"*, and it is served at three geographies:

    DEPARTAM ....... 25 units ......... 928,000 people each
    PROVINCI ...... 196 units ......... 118,000 people each
    DISTRITO .... 1,874 units .......... 12,378 people each   <- what is drawn

**THE PRIZE IS EIGHT CATEGORIES, NOT FOUR, AND THAT IS THE WHOLE REASON THE COUNTRY IS HERE.**
`sources.md` §11t declined Peru because the UNSD oracle reports four categories for it. Four
is what INEI *forwarded* to UNSD; the microdata holds eight. §11y is the finding and this file
is the proof:

    Católica | Evangélica | Otra | Ninguna | Cristiano | Adventista | Testigo de Jehová | Mormones

**AND THE PUBLISHED FOUR-CATEGORY TABLE IS WHERE THE OTHER FIVE WENT.** INEI's own headline
release prints Católica, Evangélica, *otra religión* and Ninguna — and its `otra religión` of
1,115,872 is **exactly** Otra + Cristiano + Adventista + Testigo de Jehová + Mormones. So the
oracle's four is not a different count, it is this count with six columns collapsed into one.
That identity is asserted below, and it is the cleanest demonstration on this map of §11y's
rule: **the oracle ranks what was reported, and a country can be deeper than its own return.**

**THE UNIVERSE IS AGE 12 AND OVER, NOT THE WHOLE CENSUS.** 23,196,391 against a 2017 census
population of 29,381,884 — the question was asked of people aged 12+, as the variable's own
label says. The 6.2M under-twelves were never asked, which is a different thing from a §3.5
undercount; `countries.py` carries it in `gap=` and it is not drawn as a hole.

**THE CHECK IS INEI'S OWN PUBLISHED RELEASE, AND IT IS GENUINELY INDEPENDENT.** Every internal
identity here reconciles whichever order the columns are read in — the three geographies agree
to the person, which is reassuring and would not catch a column landing in the wrong place.
The four published national figures would, and they are reproduced exactly by a 2026 query.

**`Adventista` IS THE CONTENT, AND §11y OVERSTATED IT BY A FACTOR OF FOUR.** §11y says
"Adventists are 1.5M nationally"; the department run actually gives **353,430, which is
1.52%** — a percentage misread as millions. The correction does not weaken the case, because
the case was never the national total: Adventists are **23.4% of San Antón and 22.1% of
Crucero**, both in Puno, and the Adventist mission on the Aymara altiplano has been running
schools there since 1898. Nothing else in the Americas on this map can show that.

Usage:
    python sources/pe.py --fetch    six REDATAM queries, ~90s
    python sources/pe.py            normalise from data/raw/pe/
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

SOURCE_ID = "pe_cpv_2017"
YEAR = 2017
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

HOST = "https://censos2017.inei.gob.pe/bininei"
BASE = "CPV2017DI"
VAR = "Poblacio.C5P26"

# Two different engine endpoints, deliberately.
#
#   CmdSet/PROGRED  runs a Redatam+SP program; `AS AREALIST` gives one compact row per area,
#                   keyed by code, and is where the COUNTS come from.
#   Frequency       is the web form's own route; with AREABREAK it emits one table per area
#                   under a heading that carries the code AND the name together --
#                   `AREA # 010101 | Amazonas, Chachapoyas, distrito: Chachapoyas`.
#
# The names have to come from the second, because a FREQUENCY on the name variable
# (`Distrito.NCCDI`) returns names with no codes beside them and an AREALIST of it returns
# `Tabla vacía` -- it is a string variable. Pairing the two by row order would be a §12
# shape-2 join, so it is not done: the AREABREAK heading pairs them at source.
AREALIST = {
    "dep_religion": "DEPARTAM",
    "prov_religion": "PROVINCI",
    "dist_religion": "DISTRITO",
}
AREABREAK = {
    "dep_names": "Departam",
    "prov_names": "Provinci",
    "dist_names": "Distrito",
}

# In the order REDATAM returns them. `Total` last is the row's own universe, not a category.
CATEGORIES = ["Católica", "Evangélica", "Otra", "Ninguna",
              "Cristiano", "Adventista", "Testigo de Jehová", "Mormones"]
TOTAL_CAT = "Total"

EXPECTED = {"dep": 25, "prov": 196, "dist": 1874}

# INEI's published national release for the 2017 census, population aged 12 and over. This is
# the independent witness -- see the module docstring. It names FOUR categories, and
# `otra religión` is the other five of ours added together.
PUBLISHED_2018 = {
    "Católica": 17_635_339,
    "Evangélica": 3_264_819,
    "otra religión": 1_115_872,
    "Ninguna": 1_180_361,
}
COLLAPSED_INTO_OTRA = ["Otra", "Cristiano", "Adventista", "Testigo de Jehová", "Mormones"]
NATIONAL = 23_196_391

# The 2017 census counted 29,381,884 people. The religion question was asked of 12+.
CENSUS_POPULATION = 29_381_884


def _session():
    import requests
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    return s


def _collect(s, endpoint, fields, timeout=900):
    """POST to a RpWebStats endpoint and fetch the temp file it mints."""
    r = s.post(f"{HOST}/RpWebStats.exe/{endpoint}?", data=fields,
               timeout=timeout, verify=False)
    r.raise_for_status()
    # The engine answers with a shell whose iframe names a per-session temp file.
    m = re.search(r"LFN=([^\"'&<>]+?\.htm)", r.text)
    if not m:
        raise SystemExit("REDATAM returned no output file -- the program or a variable name "
                         f"has changed.\n{r.text[:600]}")
    lfn = html.unescape(m.group(1))
    t = s.get(f"{HOST}/RpWebUtilities.exe/Text?LFN=" + urllib.parse.quote(lfn) + "&TYPE=TMP",
              timeout=timeout, verify=False)
    t.raise_for_status()
    return t.text


def fetch():
    os.makedirs(RAW, exist_ok=True)
    s = _session()

    for name, level in AREALIST.items():
        dest = os.path.join(RAW, f"pe_{name}.htm")
        if os.path.exists(dest) and os.path.getsize(dest) > 2_000:
            print("already have", dest)
            continue
        print("RUN", name, f"({level}, AREALIST)")
        body = _collect(s, "CmdSet", {
            "MAIN": "WebServerMain.inl", "BASE": BASE, "LANG": "esp",
            "CODIGO": "XXUSUARIOXX", "ITEM": "PROGRED", "MODE": "RUN",
            "CMDSET": f"RUNDEF Job\n SELECTION ALL\nTABLE T\n AS AREALIST\n"
                      f" OF {level}, {VAR}\n",
            "Submit": "Ejecutar"})
        _save(dest, body, name)

    for name, level in AREABREAK.items():
        dest = os.path.join(RAW, f"pe_{name}.htm")
        if os.path.exists(dest) and os.path.getsize(dest) > 2_000:
            print("already have", dest)
            continue
        print("RUN", name, f"({level}, AREABREAK)")
        body = _collect(s, "Frequency", {
            "MAIN": "WebServerMain.inl", "BASE": BASE, "LANG": "esp",
            "CODIGO": "XXUSUARIOXX", "ITEM": "FREQPOB", "MODE": "RUN",
            "ROW": VAR, "AREABREAK": level, "SELECTION": "ALL", "FORMAT": "HTML",
            "PERCENT": "OFF", "WEIGHT": "", "UNIVERSE": "", "FILTER": "",
            "inputTitle": "", "INLINESELECTION": "", "Submit": "Ejecutar"})
        _save(dest, body, name)


def _save(dest, body, name):
    # §5a: a 200 is not a result. REDATAM answers a bad program with `Tabla vacía` and
    # HTTP 200, which is the shape that would silently write an empty csv.
    if "Tabla vac" in body or "<table" not in body.lower():
        raise SystemExit(f"{name}: REDATAM returned no table -- the program or a variable "
                         f"name has changed.\n{body[:600]}")
    with open(dest, "w", encoding="utf-8", newline="") as fh:
        fh.write(body)
    print(f"  {os.path.getsize(dest):,} bytes")
    time.sleep(1)


def _cells(path):
    """Every <tr> of a REDATAM output table, as a list of non-empty cell strings."""
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run with --fetch first")
    body = open(path, encoding="utf-8").read()
    out = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S | re.I):
        cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).replace("\xa0", " ").strip()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S | re.I)]
        cells = [c for c in cells if c]
        if cells:
            out.append(cells)
    return out


def _areal(name, expected):
    """An AREALIST table -> {code: {category: count}}, with the header asserted in order."""
    path = os.path.join(RAW, f"pe_{name}.htm")
    rows = _cells(path)
    want = CATEGORIES + [TOTAL_CAT]

    header = next((r for r in rows if r and r[0] == "Código"), None)
    if header is None:
        raise SystemExit(f"{path}: no header row starting `Código`")
    if header[1:] != want:
        raise SystemExit(
            f"{path}: REDATAM's columns are {header[1:]}, expected {want} -- variable "
            "C5P26's category list has changed and taxonomy/pe2017.py must be revisited "
            "before anything here is drawn")

    out = {}
    for r in rows:
        if len(r) != len(want) + 1 or not re.fullmatch(r"\d+", r[0].replace(" ", "")):
            continue
        code = r[0].strip()
        vals = []
        for tok in r[1:]:
            t = tok.replace(" ", "").replace(",", "")
            if not re.fullmatch(r"\d+", t):
                raise SystemExit(f"{path}: {tok!r} in row {code} is not a figure")
            vals.append(int(t))
        if code in out:
            raise SystemExit(f"{path}: area {code} appears twice")
        out[code] = dict(zip(want, vals))
    if len(out) != expected:
        raise SystemExit(f"{path}: {len(out)} areas, expected {expected}")
    return out


def _names(name, expected):
    """An AREABREAK run -> {code: (department, province, district)}.

    The heading row is `AREA # <code> | <dept>, <prov>, distrito: <name>`, which is the only
    place on this server where a geographic code and its name arrive together.
    """
    path = os.path.join(RAW, f"pe_{name}.htm")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run with --fetch first")
    body = open(path, encoding="utf-8").read()
    heads = re.findall(r"AREA\s*#\s*(\d+)\s*</td>\s*<td[^>]*>([^<]*)", body)
    out = {}
    for code, label in heads:
        label = html.unescape(label).replace("\xa0", " ").strip()
        # `Amazonas, Chachapoyas, distrito: Chachapoyas`
        m = re.match(r"^(.*?),\s*(.*?),\s*(?:distrito|provincia):\s*(.*)$", label)
        if m:
            parts = (m.group(1).strip(), m.group(2).strip(), m.group(3).strip())
        else:
            # Callao is the `Provincia Constitucional del Callao` and carries a two-part
            # label; its department and province are the same thing.
            m = re.match(r"^(.*?),\s*(?:distrito|provincia):\s*(.*)$", label)
            if m:
                parts = (m.group(1).strip(), m.group(1).strip(), m.group(2).strip())
            else:
                # A department heading: `Departamento: Amazonas`.
                m = re.match(r"^\s*Departamento:\s*(.*)$", label)
                if m:
                    parts = (m.group(1).strip(),) * 3
                elif ":" not in label and label:
                    # `Provincia Constitucional del Callao` -- the one province that is its
                    # own department carries no marker at all at province level.
                    parts = (label,) * 3
                else:
                    raise SystemExit(f"{path}: cannot parse area label {label!r}")
        if code in out and out[code] != parts:
            raise SystemExit(f"{path}: {code} labelled both {out[code]} and {parts}")
        out[code] = parts
    if len(out) != expected:
        raise SystemExit(f"{path}: {len(out)} area headings, expected {expected}")
    return out


def read():
    dep = _areal("dep_religion", EXPECTED["dep"])
    prov = _areal("prov_religion", EXPECTED["prov"])
    dist = _areal("dist_religion", EXPECTED["dist"])
    names = _names("dist_names", EXPECTED["dist"])
    prov_names = _names("prov_names", EXPECTED["prov"])

    missing = sorted(set(dist) - set(names))
    if missing:
        raise SystemExit(f"districts with counts and no name: {missing[:8]}")

    rows = []
    for code in sorted(dist):
        dept, prov_nm, dist_nm = names[code]
        for cat in CATEGORIES + [TOTAL_CAT]:
            note = (f"level=distrito; province={prov_nm}; department={dept}; "
                    "universe is population aged 12 and over")
            if cat == TOTAL_CAT:
                note += "; universe total, not a religion category"
            rows.append({"geo_id": code, "geo_level": "distrito",
                         "geo_name": dist_nm, "source_category": cat,
                         "count": dist[code][cat], "basis": BASIS, "year": YEAR,
                         "source_id": SOURCE_ID, "note": note})
    return rows, dep, prov, dist, names, prov_names


def check(rows, dep, prov, dist, names, prov_names):
    ok = True
    allcats = CATEGORIES + [TOTAL_CAT]

    for label, tbl, want in [("department", dep, EXPECTED["dep"]),
                             ("province", prov, EXPECTED["prov"]),
                             ("district", dist, EXPECTED["dist"])]:
        good = len(tbl) == want
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {label:<11} {len(tbl):>5} units "
              f"(expected {want})")

    nat = {c: sum(v[c] for v in dist.values()) for c in allcats}
    good = nat[TOTAL_CAT] == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} national universe {nat[TOTAL_CAT]:,} "
          f"(expected {NATIONAL:,})")

    # The categories are an exact partition of the 12+ population: REDATAM tabulates every
    # record and C5P26 has no missing code, so there is no `not stated` cell to lose.
    bad = [c for c, v in dist.items() if sum(v[x] for x in CATEGORIES) != v[TOTAL_CAT]]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the 8 categories sum to Total on all "
          f"{len(dist)} districts ({len(bad)} failures) {bad[:4]}")

    # ---- the 1,874 must rebuild the 196 and the 25, on every column ----
    # Three separate queries; the engine aggregated each independently, so this is what
    # would catch a district dropped or double-counted.
    for label, finer, coarser, keylen in [
            ("districts rebuild the 196 provinces", dist, prov, 4),
            ("provinces rebuild the 25 departments", prov, dep, 2)]:
        rolled = {}
        for code, v in finer.items():
            acc = rolled.setdefault(code[:keylen], {c: 0 for c in allcats})
            for c in allcats:
                acc[c] += v[c]
        bad = []
        if set(rolled) != set(coarser):
            bad.append(("codes", sorted(set(rolled) ^ set(coarser))))
        else:
            for k in coarser:
                for c in allcats:
                    if rolled[k][c] != coarser[k][c]:
                        bad.append((k, c, rolled[k][c], coarser[k][c]))
        ok &= not bad
        n_cells = len(coarser) * len(allcats)
        print(f"  {'OK ' if not bad else 'BAD'} the {label} on all {n_cells} cells "
              f"({len(bad)} failures)")
        for b in bad[:5]:
            print(f"        {b}")

    # ---- the independent witness: INEI's own published national release ----
    got = {"Católica": nat["Católica"], "Evangélica": nat["Evangélica"],
           "Ninguna": nat["Ninguna"],
           "otra religión": sum(nat[c] for c in COLLAPSED_INTO_OTRA)}
    bad = [(c, got[c], PUBLISHED_2018[c]) for c in PUBLISHED_2018
           if got[c] != PUBLISHED_2018[c]]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} all 4 published national figures match INEI's "
          f"2017 census release ({len(bad)} failures)")
    for c, g, w in bad:
        print(f"        {c}: query {g:,} vs published {w:,}")
    print("      AND THE FOURTH ONE IS THE FINDING. INEI publishes `otra religión` = "
          f"{PUBLISHED_2018['otra religión']:,},\n      which is EXACTLY Otra + Cristiano + "
          "Adventista + Testigo de Jehová + Mormones.\n      The oracle's four categories "
          "are this table with six columns collapsed into one\n      (sources.md §11y); the "
          "microdata has all eight.")

    # every finer code must sit under its parent
    for label, finer, coarser, keylen in [("district", dist, prov, 4),
                                          ("province", prov, dep, 2)]:
        bad = sorted(c for c in finer if c[:keylen] not in coarser)
        ok &= not bad
        print(f"  {'OK ' if not bad else 'BAD'} every {label} code sits under its parent "
              f"({len(bad)} orphans) {bad[:4]}")

    print(f"\n  {len(rows):,} rows. Categories, national:")
    for cat in allcats:
        n = nat[cat]
        mark = "  <- universe" if cat == TOTAL_CAT else ""
        print(f"    {n:>11,}  {100.0 * n / NATIONAL:6.2f}%  {cat}{mark}")

    under12 = CENSUS_POPULATION - NATIONAL
    print(f"\n  universe {NATIONAL:,} of a {CENSUS_POPULATION:,} census population — "
          f"{under12:,} people\n  ({100.0 * under12 / CENSUS_POPULATION:.1f}%) are under 12 "
          "and were never asked. NOT a §3.5 undercount.")

    # Adventists are the reason this country is here; show that they have a geography.
    adv = sorted(((v["Adventista"] / v[TOTAL_CAT], names[c][2], names[c][0],
                   v["Adventista"], v[TOTAL_CAT])
                  for c, v in dist.items() if v[TOTAL_CAT] >= 2000), reverse=True)
    print("\n  `Adventista` is 1.52% nationally and is an altiplano, not a country:")
    for share, nm, dept, n, tot in adv[:6]:
        print(f"    {nm:<28} {dept:<14} {n:>6,} of {tot:>8,}  {100 * share:6.2f}%")
    print(f"    ... and {sum(1 for v in dist.values() if v['Adventista'] == 0)} districts "
          "have none at all.")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows, dep, prov, dist, names, prov_names = read()
    check(rows, dep, prov, dist, names, prov_names)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
