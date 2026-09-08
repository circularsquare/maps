"""Paraguay — DGEEC/INE, Censo Nacional de Población y Viviendas 2002, variable P17.

Reads (or fetches) data/raw/py/ and writes data/normalized/py.csv.

**FIFTY-FOUR RELIGION CATEGORIES ACROSS 229 DISTRICTS, WHICH IS THE DEEPEST LIST ON THIS
MAP.** Austria's 31 is what `queue.md` called "the deepest undrawn religion list in the
world"; Paraguay 2002 has 54, and it was sitting behind a REDATAM instance nobody had opened.
Fifteen Protestant bodies are named separately (Asamblea de Dios, Bautista, Hermanos Libres,
Iglesia de Dios, Luterana, **Mennonita**, Metodista, Nazarena, Neotestamentaria, Pentecostal,
Presbiteriana and more), the Orthodox split three ways, and the tail counts **Umbanda** (108),
**Reyukai** (144) and **Sintoismo** (60) — the last two the Japanese colonies at La Colmena
and Yguazú, which no other census on this map counts at all.

**AND IT COUNTS SYNCRETISM AS ITS OWN ANSWER.** Five categories are indigenous religion
crossed with a church: `Indígena + católica` (446), `+ anglicana` (58), `+ evangélica`
(2,406), `+ mennonita` (16), `+ otras religiones` (30). Nothing else drawn here lets a person
be counted as both, and the Anglican and Mennonite pairings are the Chaco missions in the
census's own words. `sources/py.md` §4 has what that costs the mapping, because the tree
cannot put one person in two places.

**THE ROUTE, AND WHY FOUR EARLIER PASSES MISSED IT.** `sources.md` §9k left Paraguay
"unresolved", §11t resolved the oracle row and never opened the office, and `queue.md` had it
at "§11t resolved the row and never opened the office". All of that was true of the *published*
output: INE's 2002 library serves 52 national tables and 20 district ones, and religion appears
in exactly one of them, **CUADRO P11**, which is national with an urban/rural split and only
four named categories. The microdata tabulator is a different thing entirely, and it is open:

    https://prod.redatam.org/redpry/                     the index, which names its own cgi dir
    /binpry/RpWebEngine.exe/Portal?BASE=CPV2002          mints a session, returns 4 IFRAMES
    /binpry/RpWebStats.exe/Frequency?...&ITEM=FREQPOB    the form; POST it with MODE=RUN

**THE IFRAME IS THE WHOLE TRICK.** The portal response looks empty — 4,611 bytes, no visible
content — and a frame-walker written for the older R+SP servers (Venezuela's, which this
project also probed) finds nothing, because those emit `<frame>` and this emits `<iframe>`.
One character. It reads exactly like a dead deployment, and a browser was launched to prove
otherwise before the regex was re-read.

**THE PARTITION IS EXACT AND IS ASSERTED THREE WAYS**, in `check()`:

    every district's categories sum to its own printed Total ......  229 / 229
    the national total ..........................................  3,892,603
      = UNSD Demographic Yearbook table 28, Paraguay 2002, to the person
    religion universe + No Aplica ...............................  5,163,198
      = the 2002 census population, to the person

**`No Aplica` IS NOT A RELIGION AND IS NOT A REFUSAL.** P17 was asked of people **10 and
over** (the questionnaire, Capítulo G, prints the restriction above the question), so the
1,270,595 people under 10 are outside the universe. They are dropped here rather than carried
as not-stated, and `countries.py` states the universe in `grain`. `No especificado` (74,412)
is the real non-response and IS carried.

Usage:
    python sources/py.py --fetch    one POST to prod.redatam.org, plus the DEPTO witness
    python sources/py.py            normalise from data/raw/py/
"""

import argparse
import csv
import html
import os
import re
import sys
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "py")
OUT = os.path.join(ROOT, "data", "normalized", "py.csv")

SOURCE_ID = "py_cpv_2002"
YEAR = 2002
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

REDATAM = "https://prod.redatam.org"
CGI = REDATAM + "/binpry"
BASE = "CPV2002"

# The three totals this file refuses to write without. See the module docstring.
NATIONAL_10PLUS = 3_892_603      # UNSD table 28, Paraguay 2002
UNDER_10 = 1_270_595             # the census's own `No Aplica`
CENSUS_POP = 5_163_198           # CNPV 2002 total population
N_DISTRICTS = 229
N_CATEGORIES = 54

# Outside P17's universe (people under 10), not a religion and not a non-response.
NOT_IN_UNIVERSE = ("No Aplica",)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")


# --------------------------------------------------------------------------- fetch

def fetch():
    """POST the Frequency form once per area break and keep the raw HTML."""
    import requests
    import urllib3
    urllib3.disable_warnings()

    os.makedirs(RAW, exist_ok=True)
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Referer": REDATAM + "/redpry/"})

    # The portal mints the per-session scratch files the form and its output live in.
    # Without it the Frequency POST answers 500.
    s.get(f"{CGI}/RpWebEngine.exe/Portal?BASE={BASE}", timeout=90, verify=False)
    s.get(f"{CGI}/RpWebStats.exe/Frequency?BASE={BASE}&ITEM=FREQPOB&lang=ESP",
          timeout=120, verify=False)

    for areabreak in ("DISTRITO", "DEPTO"):
        # FORMAT and Submit are not optional: without them the engine answers 500 with
        # `Número de Error 1 en función : setOutputFormats`, which reads like a dead host.
        data = {
            "MAIN": "WebServerMain.inl", "BASE": BASE, "LANG": "ESP",
            "CODIGO": "XXUSUARIOXX", "ITEM": "FREQPOB", "MODE": "RUN",
            "inputTitle": "", "PERCENT": "PCTGRAF",
            "ROW": "PERSONA.religion", "AREABREAK": areabreak,
            "SELECTION": "ALL", "INLINESELECTION": "",
            "UNIVERSE": "", "FILTER": "", "TEXT_FILTER": "",
            "FORMAT": "HTML", "Submit": "Ejecutar",
        }
        r = s.post(f"{CGI}/RpWebStats.exe/Frequency?", data=data,
                   timeout=600, verify=False)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "latin-1"
        blob = r.text

        # The result is a frameset onto ~tmp_*.htm under /redpry/tempo/<session>/.
        # <IFRAME>, not <frame> -- see the docstring.
        frames = re.findall(r'<i?frame[^>]+src="([^"]+)"', blob, re.I)
        if not frames:
            raise SystemExit(f"{areabreak}: no iframes in the response; the engine "
                             f"returned {len(blob)} chars. Read it before retrying.")
        for f in frames:
            f = f.replace("&amp;", "&")
            u = f if f.startswith("http") else REDATAM + f
            rr = s.get(u, timeout=600, verify=False)
            rr.encoding = rr.apparent_encoding or "latin-1"
            blob += f"\n<!--FRAME {u}-->\n" + rr.text

        path = os.path.join(RAW, f"py_freq_{areabreak}.html")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(blob)
        print(f"  {areabreak}: {len(blob):,} chars -> {path}")


# --------------------------------------------------------------------------- parse

def _cells(tr):
    c = [html.unescape(re.sub(r"<[^>]+>", "", x)).replace("\xa0", " ").strip()
         for x in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S | re.I)]
    return [x for x in c if x]


def _num(s):
    s = s.replace(" ", "").replace(" ", "").replace(".", "")
    return int(s) if s.isdigit() else None


def read(areabreak="DISTRITO"):
    """-> (rows, totals, noaplica, names).

    Each area block is

        AREA # <code> | <name>
        Religión que profesa | Casos | % | Acumulado %
        <category> | <n> | <pct> | <cum>    x N
        Total | <n>
        No Aplica : | <n>

    and the file ends with a WHOLE-COUNTRY block carrying no AREA header. An area must
    therefore be CLOSED at its `No Aplica` row; leaving it open makes the last district
    swallow the national totals, which doubles the national sum and is the one error here
    that still looks like a plausible number.
    """
    path = os.path.join(RAW, f"py_freq_{areabreak}.html")
    with open(path, encoding="utf-8") as fh:
        blob = fh.read()

    names, rows = {}, []
    totals, noaplica = {}, {}
    cur = None
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", blob, re.S | re.I):
        c = _cells(tr)
        if not c:
            continue
        m = re.match(r"AREA\s*#\s*(\S+)$", c[0])
        if m:
            cur = m.group(1)
            names[cur] = c[1] if len(c) > 1 else ""
            continue
        if cur is None or len(c) < 2:
            continue
        label, v = c[0], _num(c[1])
        if v is None:
            continue
        if label.lower().startswith("total"):
            totals[cur] = v
            continue
        if label.startswith(NOT_IN_UNIVERSE):
            noaplica[cur] = v
            cur = None
            continue
        rows.append((cur, names[cur], label, v))
    return rows, totals, noaplica, names


def check(rows, totals, noaplica, names):
    per = defaultdict(int)
    for code, _n, _r, v in rows:
        per[code] += v

    bad = [c for c in names if per[c] != totals.get(c)]
    assert not bad, (f"{len(bad)} areas whose categories do not sum to their printed "
                     f"Total, first: {bad[:5]}")

    grand = sum(per.values())
    assert grand == NATIONAL_10PLUS, f"national {grand:,} != {NATIONAL_10PLUS:,}"

    under = sum(noaplica.values())
    assert under == UNDER_10, f"No Aplica {under:,} != {UNDER_10:,}"
    assert grand + under == CENSUS_POP, \
        f"{grand:,} + {under:,} != census {CENSUS_POP:,}"

    assert len(names) == N_DISTRICTS, f"{len(names)} districts, expected {N_DISTRICTS}"
    cats = {r[2] for r in rows}
    assert len(cats) == N_CATEGORIES, f"{len(cats)} categories, expected {N_CATEGORIES}"

    # The department run is an independent witness on the same engine: summing districts
    # into departments must reproduce it exactly.
    dep_path = os.path.join(RAW, "py_freq_DEPTO.html")
    if os.path.exists(dep_path):
        drows, dtot, dna, dnames = read("DEPTO")
        by_dep = defaultdict(int)
        for code, _n, _r, v in rows:
            by_dep[code[:2]] += v
        dsum = defaultdict(int)
        for code, _n, _r, v in drows:
            dsum[code.zfill(2)] += v
        mismatch = {k: (by_dep.get(k), dsum.get(k)) for k in set(by_dep) | set(dsum)
                    if by_dep.get(k) != dsum.get(k)}
        assert not mismatch, f"district sums != department run: {mismatch}"
        print(f"  departments agree: {len(dsum)} of them, district sums identical")

    print(f"  {len(names)} districts, {len(cats)} categories, "
          f"{grand:,} people aged 10+, partition exact")


def write(rows):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for code, name, cat, n in sorted(rows):
            w.writerow([code, "distrito", name, cat, n,
                        BASIS, YEAR, SOURCE_ID, "level=distrito; universe=10+"])
    print(f"  wrote {OUT} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true",
                    help="POST the REDATAM Frequency form into data/raw/py/")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    rows, totals, noaplica, names = read()
    check(rows, totals, noaplica, names)
    write(rows)


if __name__ == "__main__":
    main()
