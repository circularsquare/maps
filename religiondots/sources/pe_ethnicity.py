"""Peru — religion crossed with self-identified ethnicity. A CHECK ON A CLAIM, NOT A LAYER.

Reads (or fetches) data/raw/pe/ and writes nothing. Nothing here places a dot, changes a
count, or enters `pe.csv`. It exists because `countries.py`'s `note_public` for Peru makes an
assertion about *why* `Ninguna` peaks in Amazonia, and INEI serves the variable that tests it.

**THE CLAIM UNDER TEST**, written before this file existed:

> No religion is 5.1%, and reading it as secularity would be wrong. It peaks in indigenous
> Amazonia. The census offers no box for Amazonian indigenous religion, and 'none' is where a
> form with no box for your religion puts you.

**THE VERDICT: the association is real and large, and the explanation was overstated.**
`C5P25` — *"P12a+: Por sus costumbres y sus antepasados Ud. se considera"* — is asked of the
same 23,196,391 people as `C5P26`, so the two cross exactly with no modelling at all:

    among people who self-identify as `Nativo o indígena de la amazonía` (210,612):

        Evangélica   41.54%   <- the PLURALITY, and 2.95x the national rate
        Católica     34.14%       against 76.03% nationally
        Ninguna      18.74%   <- 3.68x the national 5.09%. Real, large, and THIRD.
        Adventista    4.34%
        ...
        about 81% give a Christian answer of some kind

So `Ninguna` among Amazonian indigenous people is genuinely 3.7x the national rate — the claim
is not wrong — but it is **not the main story**, and the note as first written implied it was.
The dominant fact about religion among Amazonian indigenous Peruvians is **evangelical
conversion**, which is the documented history of the Awajún, Wampís and Asháninka missions.
`note_public` now says both.

**AND IT REFUTES A CANDIDATE THIS MAP HAD LISTED FOR `other.pe`.** `taxonomy/branches.py`
offered "the indigenous religions of the Amazon" as plausible content for the `Otra` cell.
Measured, `Otra` is **flat across ethnicity**:

    Otra among Amazonian indigenous  0.40%   vs  0.41% nationally  =  0.98x
    district correlation between Amazonian-indigenous share and Otra share:  r = 0.07

**Zero.** Whatever is in `Otra`, it is not Amazonian indigenous respondents, which is what
`Ninguna` was already hinting at and this measures. It leaves the Israelitas del Nuevo Pacto
Universal — a mestizo and Andean-migrant church — as the reading, and `Otra`'s largest groups
by share turn out to be **Tusán (5.51%) and Nikkei (3.50%)**, the Chinese-Peruvian and
Japanese-Peruvian populations, which is a different and much more legible content.

**A THIRD THING, UNLOOKED FOR: THE TWO ADVENTIST REGIONS ARE TWO PEOPLES.** `sources/pe_geo.py`
found that Peru's Adventists are the Puno altiplano *and* the Alto Mayo, geographically. At the
person level the same split is ethnic and sharper than the geography:

    Adventista among Aimara ................. 6.58%   =  4.32x the national rate
    Adventista among Amazonian indigenous ... 4.34%   =  2.85x
    Adventista among Quechua ................ 1.60%   =  1.05x  (i.e. nothing)

The altiplano cluster is **Aymara and not merely southern** — Quechua Peru, four times the size
of Aymara Peru and largely in the same highlands, is at the national average. That is the 1898
Platería mission showing up in a variable it has nothing to do with.

**WHY THIS IS A SEPARATE FILE.** Crossing religion with ethnicity is a different act from
drawing religion, and spec §14 says who a map depicts and how finely is not a technical
question. Keeping it out of `pe.py` keeps the boundary explicit: **`pe.csv` contains no
ethnicity, no dot is placed by it, and this module is only ever run to check a sentence.**

Usage:
    python sources/pe_ethnicity.py --fetch    two REDATAM queries, ~30s
    python sources/pe_ethnicity.py            report from data/raw/pe/
"""

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

HOST = "https://censos2017.inei.gob.pe/bininei"
BASE = "CPV2017DI"
RELIGION = "Poblacio.C5P26"
ETHNICITY = "Poblacio.C5P25"

# `AS FREQUENCY OF <a> BY <b>` is the crosstab; the engine labels its own output "Crosstab".
# `AS CROSSTABLE ...` — the syntax the manual suggests — returns HTTP 500 on this build.
PROGRAMS = {
    "eth_crosstab": f"RUNDEF Job\n SELECTION ALL\nTABLE T\n AS FREQUENCY\n"
                    f" OF {RELIGION} BY {ETHNICITY}\n",
    "dist_ethnicity": f"RUNDEF Job\n SELECTION ALL\nTABLE T\n AS AREALIST\n"
                      f" OF DISTRITO, {ETHNICITY}\n",
}

NATIONAL = 23_196_391
EXPECTED_DISTRICTS = 1874

RELIGIONS = ["Católica", "Evangélica", "Otra", "Ninguna",
             "Cristiano", "Adventista", "Testigo de Jehová", "Mormones"]
ETHNICITIES = ["Quechua", "Aimara", "Nativo o indígena de la amazonía",
               "Parte de otro pueblo indígena u originario",
               "Negro, moreno, zambo, mulato / pueblo afroperuano o afrodescendiente",
               "Blanco", "Mestizo", "Otro", "No sabe /  No responde", "Nikkei", "Tusán"]
AMAZON = "Nativo o indígena de la amazonía"

# Districts below this are excluded from the district-level statistics only: one extended
# family answering together swings a share, and it is the correlation that is being measured.
MIN_POP = 2000


def _session():
    import requests
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    return s


def fetch():
    os.makedirs(RAW, exist_ok=True)
    s = _session()
    for name, program in PROGRAMS.items():
        dest = os.path.join(RAW, f"pe_{name}.htm")
        if os.path.exists(dest) and os.path.getsize(dest) > 2_000:
            print("already have", dest)
            continue
        print("RUN", name)
        r = s.post(f"{HOST}/RpWebStats.exe/CmdSet?", data={
            "MAIN": "WebServerMain.inl", "BASE": BASE, "LANG": "esp",
            "CODIGO": "XXUSUARIOXX", "ITEM": "PROGRED", "MODE": "RUN",
            "CMDSET": program, "Submit": "Ejecutar"}, timeout=900, verify=False)
        r.raise_for_status()
        m = re.search(r"LFN=([^\"'&<>]+?\.htm)", r.text)
        if not m:
            raise SystemExit(f"{name}: REDATAM returned no output file.\n{r.text[:600]}")
        t = s.get(f"{HOST}/RpWebUtilities.exe/Text?LFN="
                  + urllib.parse.quote(html.unescape(m.group(1))) + "&TYPE=TMP",
                  timeout=900, verify=False)
        t.raise_for_status()
        # §5a: a 200 is not a result.
        if "Tabla vac" in t.text or "<table" not in t.text.lower():
            raise SystemExit(f"{name}: REDATAM returned no table.\n{t.text[:600]}")
        with open(dest, "w", encoding="utf-8", newline="") as fh:
            fh.write(t.text)
        print(f"  {os.path.getsize(dest):,} bytes")
        time.sleep(1)


def _cells(path):
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


def read_crosstab():
    """-> {religion: {ethnicity: count}}, header asserted in order."""
    rows = _cells(os.path.join(RAW, "pe_eth_crosstab.htm"))
    want = ETHNICITIES + ["Total"]
    header = next((r for r in rows if r and r[0] == "Quechua" and len(r) == len(want)), None)
    if header is None or header != want:
        raise SystemExit(f"crosstab columns are {header}, expected {want} -- C5P25's "
                         "category list has changed")
    out = {}
    for r in rows:
        if len(r) != len(want) + 1 or r[0] not in RELIGIONS + ["Total"]:
            continue
        vals = []
        for tok in r[1:]:
            t = tok.replace(" ", "").replace(",", "")
            if not re.fullmatch(r"\d+", t):
                raise SystemExit(f"{r[0]}: {tok!r} is not a figure")
            vals.append(int(t))
        out[r[0]] = dict(zip(want, vals))
    missing = [k for k in RELIGIONS if k not in out]
    if missing:
        raise SystemExit(f"crosstab is missing religion rows: {missing}")
    return out


def _areal(name, cats, expected):
    rows = _cells(os.path.join(RAW, name))
    want = cats + ["Total"]
    out = {}
    for r in rows:
        if len(r) != len(want) + 1 or not re.fullmatch(r"\d+", r[0].replace(" ", "")):
            continue
        out[r[0]] = dict(zip(want, [int(x.replace(" ", "")) for x in r[1:]]))
    if len(out) != expected:
        raise SystemExit(f"{name}: {len(out)} areas, expected {expected}")
    return out


def main():
    if "--fetch" in sys.argv:
        fetch()

    import numpy as np

    X = read_crosstab()
    ok = True

    # ---- the two variables must cover exactly the same people ----
    col = {e: sum(X[r][e] for r in RELIGIONS) for e in ETHNICITIES + ["Total"]}
    good = col["Total"] == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the crosstab covers {col['Total']:,} people "
          f"(C5P26's universe is {NATIONAL:,})")

    bad = [e for e in ETHNICITIES if sum(X[r][e] for r in RELIGIONS) != col[e]]
    good = sum(col[e] for e in ETHNICITIES) == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the 11 ethnicity columns sum to the same universe")

    # The religion margin must equal what sources/pe.py already reconciled against INEI's
    # published figures -- which is what makes this crosstab the SAME table, not a new one.
    margin = {r: X[r]["Total"] for r in RELIGIONS}
    published = {"Católica": 17_635_339, "Evangélica": 3_264_819, "Ninguna": 1_180_361}
    bad = [(r, margin[r], published[r]) for r in published if margin[r] != published[r]]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} its religion margin matches INEI's published "
          f"figures ({len(bad)} failures)")

    if not ok:
        raise SystemExit("crosstab reconciliation FAILED")

    N = NATIONAL
    natl = {r: margin[r] / N for r in RELIGIONS}

    # ---- 1. the claim under test ----
    a = col[AMAZON]
    print(f"\n  1. THE CLAIM: `Ninguna` peaks in Amazonia because the form has no box for "
          f"Amazonian\n     indigenous religion. Among the {a:,} who self-identify as "
          f"`{AMAZON}`:\n")
    print(f"     {'':<20} {'share':>8} {'national':>9} {'ratio':>7}")
    for r in sorted(RELIGIONS, key=lambda r: -X[r][AMAZON]):
        s = X[r][AMAZON] / a
        print(f"     {r:<20} {100 * s:7.2f}% {100 * natl[r]:8.2f}% {s / natl[r]:6.2f}x")
    christian = sum(X[r][AMAZON] for r in RELIGIONS if r not in ("Ninguna", "Otra"))
    print(f"\n     VERDICT: the association is REAL and LARGE — `Ninguna` is "
          f"{X['Ninguna'][AMAZON] / a / natl['Ninguna']:.2f}x the national rate —\n"
          f"     but it is the THIRD answer, not the story. {100 * christian / a:.1f}% give a "
          "Christian answer and\n     `Evangélica` is the plurality at "
          f"{100 * X['Evangélica'][AMAZON] / a:.1f}%. The note overstated it and now says both.")

    # ---- 2. the candidate this refutes ----
    print(f"\n  2. `Otra` IS FLAT ACROSS ETHNICITY, which removes a candidate branches.py "
          "listed:\n")
    for e in sorted(ETHNICITIES, key=lambda e: -X["Otra"][e] / max(col[e], 1)):
        s = X["Otra"][e] / col[e]
        mark = "   <- the cell's largest groups" if e in ("Tusán", "Nikkei") else ""
        star = "  <- 0.98x: NOT indigenous Amazonians" if e == AMAZON else ""
        print(f"     {e[:54]:<56} {100 * s:6.2f}%  ({X['Otra'][e]:>6,}){mark}{star}")

    # ---- 3. unlooked for: the two Adventist regions are two peoples ----
    print("\n  3. THE TWO ADVENTIST REGIONS ARE TWO PEOPLES, which the geography could not "
         "show:\n")
    for e in ["Aimara", AMAZON, "Quechua", "Mestizo"]:
        s = X["Adventista"][e] / col[e]
        print(f"     Adventista among {e[:38]:<40} {100 * s:6.2f}%  "
              f"{s / natl['Adventista']:5.2f}x")
    print("     Quechua Peru is four times the size of Aymara Peru and largely the same "
          "highlands,\n     and it sits at the national average. The altiplano cluster is "
          "AYMARA, not southern.")

    # ---- 4. and the same thing geographically, as a second witness ----
    rel = _areal("pe_dist_religion.htm", RELIGIONS, EXPECTED_DISTRICTS)
    eth = _areal("pe_dist_ethnicity.htm", ETHNICITIES, EXPECTED_DISTRICTS)
    gids = sorted(set(rel) & set(eth))
    if len(gids) != EXPECTED_DISTRICTS:
        raise SystemExit(f"{len(gids)} districts in both tables, "
                         f"expected {EXPECTED_DISTRICTS}")
    tot = np.array([rel[g]["Total"] for g in gids], float)
    bad = [g for g in gids if rel[g]["Total"] != eth[g]["Total"]]
    if bad:
        raise SystemExit(f"{len(bad)} districts where the two variables disagree on the "
                         f"district total: {bad[:5]}")
    m = tot >= MIN_POP
    amz = np.array([eth[g][AMAZON] for g in gids], float) / tot
    print(f"\n  4. AND THE SAME THING GEOGRAPHICALLY, on the {int(m.sum()):,} districts of "
          f"{MIN_POP:,}+ people:\n")
    for r in ["Ninguna", "Evangélica", "Otra", "Católica"]:
        s = np.array([rel[g][r] for g in gids], float) / tot
        print(f"     corr(Amazonian-indigenous share, {r:<11} share) = "
              f"{np.corrcoef(amz[m], s[m])[0, 1]:7.4f}")
    hi, lo = amz[m] > 0.5, amz[m] < 0.02
    for label, sel in [(">50% Amazonian indigenous", hi), (" <2% Amazonian indigenous", lo)]:
        n = np.array([rel[g]["Ninguna"] for g in gids], float)[m][sel].sum()
        e = np.array([rel[g]["Evangélica"] for g in gids], float)[m][sel].sum()
        o = np.array([rel[g]["Otra"] for g in gids], float)[m][sel].sum()
        t = tot[m][sel].sum()
        print(f"     districts {label}: {int(sel.sum()):>4}  Ninguna {100 * n / t:5.1f}%  "
              f"Evangélica {100 * e / t:5.1f}%  Otra {100 * o / t:4.2f}%")
    print("\n     The district picture and the person picture agree, which they need not "
          "have:\n     an ecological correlation can survive an individual-level one being "
          "absent.")

    print("\n  NOTHING HERE IS DRAWN. pe.csv contains no ethnicity and no dot is placed by "
          "it;\n  this module exists to check a sentence in countries.py. See the docstring.")


if __name__ == "__main__":
    main()
