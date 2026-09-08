"""Nicaragua — INIDE, VIII Censo de Población y IV de Vivienda 2005, variable P13.

Reads (or fetches) data/raw/ni/ and writes data/normalized/ni.csv.

**THE SOURCE IS NOT A PUBLICATION, IT IS INIDE'S OWN REDATAM SERVER RUNNING A QUERY WE
WROTE** (`sources.md` §11x). `redatam.inide.gob.ni` is linked from INIDE's main navigation as
*Sistema en línea Redatam*, is unauthenticated, and will run an arbitrary Redatam+SP program
against the 2005 census microdata. That matters for what can be drawn:

    what INIDE PRINTS   Volume I, CUADRO 12, religion x DEPARTMENT ....... 17 units
    what INIDE SERVES   this file, religion x MUNICIPALITY .............. 153 units
                        (and x comarca, 2,579 units — see below)

Volume IV is 546 pages of municipal tables and contains no religion table at all; its only
hit is the glossary. So the published route caps at 17 and the served route is 9x finer, on
the same website, for the same census. §12's rule, added with this country: **ask whether the
office runs a REDATAM instance before reading its PDFs.**

**THE COMARCA TABLE IS REAL AND IS NOT DRAWN.** `OF COMR05, PERS05.P13` returns 2,579 units
at 1,759 people each — among the finest geographies anywhere on this map — and reconciles to
the same national total. It is not used because **no comarca boundaries are published**:
geoBoundaries NIC 404s on ADM3 and OCHA's `cod-ab-nic` says in its own words *"structured
into 2 levels"*. §11w's rule, that the one thing the oracle cannot see is the geography,
biting on the resolution rather than on the country. If census cartography ever surfaces this
file changes one identifier.

**THE UNIVERSE IS AGE 5 AND OVER, NOT THE WHOLE CENSUS.** 4,537,200 against a 2005 census
population of 5,142,098 — the question was asked of people aged 5+, exactly as CUADRO 12's
own title says (*POBLACIÓN DE 5 AÑOS Y MÁS, POR RELIGIÓN*). The 604,898 under-fives are not a
§3.5 undercount and must not be drawn as one; they were never asked. countries.py carries
this in `gap=`.

**EIGHT CATEGORIES, AND `Morava` IS WHY THIS COUNTRY IS WORTH DRAWING.** The Moravian Church
is 73,902 people, 1.6% nationally, and is not a national phenomenon at all: it is the
Caribbean coast. The department run puts 8,995 Moravians in the RAAN against 27 in Nueva
Segovia — a 300x spread that the national figure hides completely. No other source on this
map counts Moravians at a geography that can show one; Trinidad's 3,526 and Jamaica's are
national-scale rounding.

**THE CHECK IS THE PRINTED VOLUME, WHICH IS A GENUINELY INDEPENDENT WITNESS.** The nine
national figures below were read off page 192 of `Vol.I Poblacion-Caracteristicas
Generales.pdf` — a PDF typeset in 2006 from a tabulation run then — and this file asserts
that a query written in 2026 against the microdata reproduces all nine exactly. The two share
no code path and nothing else here would catch a column landing in the wrong place: every
internal identity reconciles whichever order the columns are read in.

Usage:
    python sources/ni.py --fetch    four REDATAM queries, ~30s
    python sources/ni.py            normalise from data/raw/ni/
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
RAW = os.path.join(ROOT, "data", "raw", "ni")
OUT = os.path.join(ROOT, "data", "normalized", "ni.csv")

SOURCE_ID = "ni_cpv_2005"
YEAR = 2005
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

HOST = "http://redatam.inide.gob.ni/redbin"
BASE = "VIVPOB05"

# Redatam+SP. `AREALIST` gives one row per area of the named entity; `FREQUENCY` on the name
# variable gives the `CODE-Name` strings, which is where the municipality names come from —
# §12's Chile rule, take the name from the statistical source and not the boundary file.
PROGRAMS = {
    "mun_religion": "RUNDEF Job\n SELECTION ALL\nTABLE T\n AS AREALIST\n OF MUN05, PERS05.P13\n",
    "dep_religion": "RUNDEF Job\n SELECTION ALL\nTABLE T\n AS AREALIST\n OF DEP05, PERS05.P13\n",
    "mun_names": "RUNDEF Job\n SELECTION ALL\nTABLE T\n AS FREQUENCY\n OF MUN05.CMuni\n",
    "dep_names": "RUNDEF Job\n SELECTION ALL\nTABLE T\n AS FREQUENCY\n OF DEP05.CDepto\n",
}

# In the order REDATAM returns them. `Total` last is the row's own universe, not a category.
CATEGORIES = ["Ninguna", "Católica", "Evangélica", "Morava",
              "Testigo de Jehová", "Judaísmo", "Musulmán", "Otra"]
TOTAL_CAT = "Total"

EXPECTED_MUN = 153
EXPECTED_DEP = 17

# Read off page 192 of Vol.I Poblacion-Caracteristicas Generales.pdf, the `LA REPÚBLICA /
# Ambos Sexos` row of CUADRO 12. This is the independent witness — see the module docstring.
PRINTED_2006 = {
    "Total": 4_537_200,
    "Católica": 2_652_985,
    "Evangélica": 981_795,
    "Morava": 73_902,
    "Testigo de Jehová": 42_587,
    "Judaísmo": 199,
    "Musulmán": 321,
    "Otra": 74_101,
    "Ninguna": 711_310,
}
NATIONAL = PRINTED_2006["Total"]

# The 2005 census counted 5,142,098 people. The religion question was asked of 5+.
CENSUS_POPULATION = 5_142_098


def _post(program, timeout=600):
    import requests

    r = requests.post(
        HOST + "/RpWebStats.exe/CmdSet?",
        data={"MAIN": "WebServerMain.inl", "BASE": BASE, "LANG": "esp",
              "CODIGO": "XXUSUARIOXX", "ITEM": "PROGRED", "MODE": "RUN",
              "CMDSET": program, "Submit": "Ejecutar"},
        headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    r.raise_for_status()
    # The engine answers with a shell naming a per-session temp file it has just minted.
    tmps = sorted(set(re.findall(r"(RpBases[^\"'&<>]*?\.htm)", r.text)))
    if not tmps:
        raise SystemExit("REDATAM returned no output file. Body starts:\n"
                         + r.text[:600])
    url = HOST + "/RpWebUtilities.exe/Text?LFN=" + urllib.parse.quote(tmps[0]) + "&TYPE=TMP"
    t = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    t.raise_for_status()
    return t.text


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for name, program in PROGRAMS.items():
        dest = os.path.join(RAW, f"ni_{name}.htm")
        if os.path.exists(dest) and os.path.getsize(dest) > 2_000:
            print("already have", dest)
            continue
        print("RUN", name)
        body = _post(program)
        # §5a: a 200 is not a result. REDATAM answers a bad program with `Tabla vacía` and
        # HTTP 200, which is the shape that would silently write an empty csv.
        if "Tabla vac" in body or "<table" not in body.lower():
            raise SystemExit(f"{name}: REDATAM returned no table -- the program or a "
                             f"variable name has changed.\n{body[:600]}")
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
    path = os.path.join(RAW, f"ni_{name}.htm")
    rows = _cells(path)
    want = CATEGORIES + [TOTAL_CAT]

    header = next((r for r in rows if r and r[0] == "Código"), None)
    if header is None:
        raise SystemExit(f"{path}: no header row starting `Código`")
    got = header[1:]
    if got != want:
        raise SystemExit(
            f"{path}: REDATAM's columns are {got}, expected {want} -- variable P13's "
            "category list has changed and taxonomy/ni2005.py must be revisited before "
            "anything here is drawn")

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
    """A FREQUENCY table on a name variable -> {code: name}, from its `CODE-Name` labels."""
    path = os.path.join(RAW, f"ni_{name}.htm")
    out = {}
    for r in _cells(path):
        m = re.fullmatch(r"(\d+)\s*-\s*(.+)", r[0].strip())
        if m:
            code, label = m.group(1), m.group(2).strip()
            if code in out and out[code] != label:
                raise SystemExit(f"{path}: {code} named both {out[code]!r} and {label!r}")
            out[code] = label
    if len(out) != expected:
        raise SystemExit(f"{path}: {len(out)} names, expected {expected}")
    return out


def read():
    mun = _areal("mun_religion", EXPECTED_MUN)
    dep = _areal("dep_religion", EXPECTED_DEP)
    mun_names = _names("mun_names", EXPECTED_MUN)
    dep_names = _names("dep_names", EXPECTED_DEP)

    missing = sorted(set(mun) - set(mun_names))
    if missing:
        raise SystemExit(f"municipalities with counts and no name: {missing[:8]}")

    rows = []
    for code in sorted(mun):
        for cat in CATEGORIES + [TOTAL_CAT]:
            note = "level=municipio; universe is population aged 5 and over"
            if cat == TOTAL_CAT:
                note += "; universe total, not a religion category"
            rows.append({"geo_id": code, "geo_level": "municipio",
                         "geo_name": mun_names[code], "source_category": cat,
                         "count": mun[code][cat], "basis": BASIS, "year": YEAR,
                         "source_id": SOURCE_ID, "note": note})
    return rows, mun, dep, mun_names, dep_names


def check(rows, mun, dep, mun_names, dep_names):
    ok = True

    good = len(mun) == EXPECTED_MUN
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} municipio  {len(mun):>4} units "
          f"(expected {EXPECTED_MUN})")

    nat = {c: sum(v[c] for v in mun.values()) for c in CATEGORIES + [TOTAL_CAT]}
    good = nat[TOTAL_CAT] == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} national universe {nat[TOTAL_CAT]:,} "
          f"(expected {NATIONAL:,})")

    # The categories are an exact partition of the 5+ population: REDATAM tabulates every
    # record and P13 has no missing code, so there is no `not stated` cell to lose.
    bad = [c for c, v in mun.items()
           if sum(v[x] for x in CATEGORIES) != v[TOTAL_CAT]]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the 8 categories sum to Total on all "
          f"{len(mun)} municipios ({len(bad)} failures) {bad[:4]}")

    # ---- the 153 must rebuild the 17, on every column ----
    # This is the check that a municipality has not been dropped or double-counted: the two
    # tables are separate queries and the engine aggregated each independently.
    rolled = {}
    for code, v in mun.items():
        d = code[:2]
        acc = rolled.setdefault(d, {c: 0 for c in CATEGORIES + [TOTAL_CAT]})
        for c in CATEGORIES + [TOTAL_CAT]:
            acc[c] += v[c]
    bad = []
    if set(rolled) != set(dep):
        bad.append(("codes", sorted(set(rolled) ^ set(dep))))
    else:
        for d in dep:
            for c in CATEGORIES + [TOTAL_CAT]:
                if rolled[d][c] != dep[d][c]:
                    bad.append((d, c, rolled[d][c], dep[d][c]))
    ok &= not bad
    n_cells = EXPECTED_DEP * (len(CATEGORIES) + 1)
    print(f"  {'OK ' if not bad else 'BAD'} the {EXPECTED_MUN} municipios rebuild the "
          f"{EXPECTED_DEP} departments on all {n_cells} cells ({len(bad)} failures)")
    for b in bad[:5]:
        print(f"        {b}")

    # ---- the independent witness: the 2006 printed volume ----
    bad = [(c, nat[c], PRINTED_2006[c]) for c in PRINTED_2006 if nat[c] != PRINTED_2006[c]]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} all 9 national figures match CUADRO 12 as "
          f"PRINTED in 2006 ({len(bad)} failures)")
    for c, got, want in bad[:9]:
        print(f"        {c}: query {got:,} vs printed {want:,}")
    print("      a 2026 microdata query against a PDF typeset in 2006 — no shared code path,\n"
          "      and the only check here that would catch a column landing in the wrong place.")

    # every municipality code must be its department's code + two digits
    bad = sorted(c for c in mun if c[:2] not in dep)
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} every municipio code sits under a department "
          f"code ({len(bad)} orphans) {bad[:4]}")

    print(f"\n  {len(rows):,} rows. Categories, national:")
    for cat in CATEGORIES + [TOTAL_CAT]:
        n = nat[cat]
        mark = "  <- universe" if cat == TOTAL_CAT else ""
        print(f"    {n:>11,}  {100.0 * n / NATIONAL:6.2f}%  {cat}{mark}")

    under5 = CENSUS_POPULATION - NATIONAL
    print(f"\n  universe {NATIONAL:,} of a {CENSUS_POPULATION:,} census population — "
          f"{under5:,} people\n  ({100.0 * under5 / CENSUS_POPULATION:.1f}%) are under 5 and "
          "were never asked. NOT a §3.5 undercount.")

    # Morava is the reason this country is here; show that it has a geography.
    mor = sorted(((v["Morava"] / v[TOTAL_CAT], mun_names[c], v["Morava"], v[TOTAL_CAT])
                  for c, v in mun.items() if v[TOTAL_CAT] > 0), reverse=True)
    print("\n  `Morava` is 1.63% nationally and is a coast, not a country:")
    for share, nm, n, tot in mor[:5]:
        print(f"    {nm:<24} {n:>7,} of {tot:>8,}  {100 * share:6.2f}%")
    print(f"    ... and {sum(1 for c, v in mun.items() if v['Morava'] == 0)} municipios "
          "have none at all.")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows, mun, dep, mun_names, dep_names = read()
    check(rows, mun, dep, mun_names, dep_names)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
