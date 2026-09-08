"""Chile — INE, Censo de Población y Vivienda 2024, religion by comuna.

Reads (or fetches) data/raw/cl/ and writes data/normalized/cl.csv.

One workbook, `P6_Religion-o-credo.xlsx`, sheet **2**: **13 named categories plus Ninguna
and a non-response line, on 346 comunas**, 15,205,784 people aged 15 or over.

**Chile asked this question for the first time since 2002.** The 2017 census was abbreviated
and left it out, so this is a 22-year gap closing, and the category list was written with the
Oficina Nacional de Asuntos Religiosos rather than inherited. That shows: it names Jehovah's
Witnesses, the Latter-day Saints, the Orthodox and the Bahá'í separately, at four and three
figures, which most censuses of this size do not.

THE UNIVERSE IS PEOPLE AGED 15 OR OVER, AND THAT IS NOT SCALED UP — DECIDED 2026-09-04.
Question 31 was put to residents 15+, so 3,274,648 people — 17.7% of Chile — are outside the
table. They are left outside it. Multiplying each comuna's counts by
`population / population 15+` was considered and rejected, because Chile's own data says what
that would get wrong: the share professing a religion runs **96.0% at 65+, 79.3% at 45-64,
69.4% at 30-44 and 63.9% at 15-29**, against a 15+ average of 75.1%. Under-15s are the
children of the two youngest bands, so a flat scale-up would assign them the 75.1% and
overstate religion among children by six to eleven points.

§3.5a scales Pew's adult answers onto American children, and the difference is the whole
argument: there, the alternative was drawing 51.6% of the country as nothing at all. Here the
source publishes an exact, complete partition of its own universe, so scaling would invent a
magnitude that is already published — §14.4. Chile is therefore drawn at **81.8% of its
population**, which is the same kind of declared partial coverage North Macedonia has at
92.5%, and `note_public` says so.

Usage:
    python sources/cl.py --fetch    two xlsx, ~0.7 MB
    python sources/cl.py            normalise from data/raw/cl/
"""

import csv
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "cl")
OUT = os.path.join(ROOT, "data", "normalized", "cl.csv")

SOURCE_ID = "cl_censo_2024"
YEAR = 2024
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

BASE = "https://censo2024.ine.gob.cl/wp-content/uploads/"
FILES = {
    # the religion tables; sheet 2 is the comuna one
    "P6_Religion-o-credo.xlsx": BASE + "2025/06/P6_Religion-o-credo.xlsx",
    # population by comuna and five-year age band — NOT used to scale anything (see the
    # module docstring). It is the independent check in sources/cl_geo.py and the source
    # of the coverage figure quoted in countries.py.
    "D1_Poblacion-por-sexo-y-edad.xlsx":
        BASE + "2025/03/D1_Poblacion-censada-por-sexo-y-edad-en-grupos-quinquenales.xlsx",
}

SHEET = "2"
HEADER_ROW = 3            # 0-based; row 4 is the País total, comunas start at row 5

# INE's own column order on sheet 2, asserted rather than assumed.
GEO_COLS = ["Código región", "Región", "Código provincia", "Provincia",
            "Código comuna", "Comuna"]
UNIVERSE = "Población de 15 años o más"
CATEGORIES = [
    "Católica",
    "Evangélica o protestante",
    "Judía",
    "Musulmana",
    "Iglesia de Jesucristo de los Santos de los Últimos Días",
    "Católica Ortodoxa",
    "Budista",
    "Hinduista",
    "Fe Bahá'í",
    "Testigo de Jehová",
    "Otros cristianos y tradiciones relacionadas con Cristo",
    "Otras religiones o credos",
    "Ninguna",
    "Religión o credo no declarado",
]

NATIONAL_15PLUS = 15_205_784
NATIONAL_ALL_AGES = 18_480_432      # D1, for the coverage statement only
EXPECTED_COMUNAS = 346


def fetch():
    import requests
    import urllib3
    import zipfile
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    os.makedirs(RAW, exist_ok=True)
    for name, url in FILES.items():
        dest = os.path.join(RAW, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
            print("already have", dest)
            continue
        print("GET", url)
        r = requests.get(url, timeout=300, verify=False,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(dest, "wb") as fh:
            fh.write(r.content)
        # §5a: assert it is really an xlsx, not a 200-with-an-error-page.
        if not zipfile.is_zipfile(dest):
            raise SystemExit(f"{dest} is not a readable xlsx")
        print(f"  {os.path.getsize(dest):,} bytes -> {dest}")


def _cell(v, where):
    """INE publishes this table complete — no suppression marks, no blanks. Anything that is
    not an integer is therefore new and must stop the build rather than be coerced (§12)."""
    if isinstance(v, (int, float)) and v == v:
        if float(v) != int(v):
            raise SystemExit(f"{where}: non-integer count {v!r}")
        return int(v)
    raise SystemExit(f"{where}: unrecognised cell {v!r} -- INE has changed the table")


def read():
    import pandas as pd

    p = os.path.join(RAW, "P6_Religion-o-credo.xlsx")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run with --fetch first")
    df = pd.read_excel(p, sheet_name=SHEET, header=HEADER_ROW, dtype=object)

    want = GEO_COLS + [UNIVERSE] + CATEGORIES
    got = [str(c).strip() for c in df.columns]
    if got != want:
        raise SystemExit("sheet 2 columns are not what this script expects:\n"
                         f"  got  {got}\n  want {want}")
    df.columns = want

    rows, national = [], {}
    for r in df.itertuples(index=False):
        d = dict(zip(want, r))
        cut = d["Código comuna"]
        if cut is None or cut != cut:
            continue
        cut = f"{int(cut):05d}"
        name = str(d["Comuna"]).strip()
        if cut == "00000":                     # the País row
            national[UNIVERSE] = _cell(d[UNIVERSE], "national/universe")
            for c in CATEGORIES:
                national[c] = _cell(d[c], f"national/{c}")
            continue
        note = (f"region={str(d['Región']).strip()}; "
                f"provincia={str(d['Provincia']).strip()}; universe=15+")
        for c in [UNIVERSE] + CATEGORIES:
            n = _cell(d[c], f"{cut} {name}/{c}")
            note_c = note
            if c == UNIVERSE:
                note_c += "; universe total, not a religion category"
            rows.append({"geo_id": cut, "geo_level": "comuna", "geo_name": name,
                         "source_category": c, "count": n, "basis": BASIS,
                         "year": YEAR, "source_id": SOURCE_ID, "note": note_c})
    if not national:
        raise SystemExit("no País row found on sheet 2")
    return rows, national


def check(rows, national):
    import collections
    ok = True

    per = collections.defaultdict(dict)
    for r in rows:
        per[r["geo_id"]][r["source_category"]] = r["count"]

    good = len(per) == EXPECTED_COMUNAS
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} comunas {len(per)} (expected {EXPECTED_COMUNAS})")

    # Every comuna's categories partition its own 15+ population. INE publishes both, so
    # this is an equality per comuna, 346 separate checks.
    bad = [g for g, d in per.items()
           if sum(d[c] for c in CATEGORIES) != d[UNIVERSE]]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} all {len(per)} comunas: the {len(CATEGORIES)} "
          f"categories sum to the comuna's own 15+ total ({len(bad)} failures)")

    print()
    for c in [UNIVERSE] + CATEGORIES:
        s = sum(d[c] for d in per.values())
        good = s == national[c]
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {c[:46]:46s} {s:>11,}  national {national[c]:>11,}")

    good = national[UNIVERSE] == NATIONAL_15PLUS
    ok &= good
    print(f"\n  {'OK ' if good else 'BAD'} national 15+ total {national[UNIVERSE]:,} "
          f"(published {NATIONAL_15PLUS:,})")

    drawn = national[UNIVERSE] - national["Religión o credo no declarado"]
    print(f"\n  {len(rows):,} rows. Categories, national, as a share of the 15+ universe:")
    for c in CATEGORIES:
        n = national[c]
        print(f"    {n:>10,}  {100.0 * n / national[UNIVERSE]:5.2f}%  {c}")
    print(f"\n  COVERAGE. The question was put to people aged 15 or over, and nothing here "
          f"scales that\n  up (see the module docstring). Drawn: {drawn:,} of "
          f"{NATIONAL_ALL_AGES:,} people, "
          f"**{100.0 * drawn / NATIONAL_ALL_AGES:.1f}%** of Chile —\n  the 15+ universe "
          f"({100.0 * national[UNIVERSE] / NATIONAL_ALL_AGES:.1f}%) less the "
          f"{national['Religión o credo no declarado']:,} who did not answer.")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows, national = read()
    check(rows, national)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
