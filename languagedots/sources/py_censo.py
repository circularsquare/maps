"""Paraguay, Censo Nacional de Poblacion y Viviendas 2002 (DGEEC, now INE): HOGAR.idiohog,
"Idioma del hogar", the language spoken in the household most of the time, counted in PEOPLE.

    python sources/py_censo.py [--fetch]

Writes data/normalized/py.csv: one row per (2002 census district, label), every person in the
2002 census (5,163,198) in exactly one row.

WHY 2002 AND NOT 2022. The 2022 census asks every person yes/no for each of Guarani,
Castellano, Portugues, Aleman, Ingles, Frances, "an indigenous language" and "another
language" (PERSONA.idiogrn ... lenotr): languages spoken, several allowed, and the indigenous
languages are not named. 2002 asked two things: PERSONA.P16A-E, languages spoken (up to five,
again several allowed), and HOGAR.idiohog, ONE language per household, "the language spoken
in this household most of the time", with all twenty indigenous languages named. Home language
outranks languages spoken (spec 1), and a single answer beats a shared one (AGENT_BRIEF 2), so
the map draws 2002's household language. sources/py.md has the argument and the 2022 figures.

THE QUESTION IS ASKED OF THE HOUSEHOLD, NOT THE PERSON. Every member of a household is drawn
in the household's language, so a Guarani-speaking grandmother in a household that answered
Castellano is drawn as Castellano. Counting people rather than households is the engine's own
cross of the household variable with PERSONA.sexo (CrossTab ITEM=CRUZCOMBI), which counts
persons; the household frequency (FREQHOG) counts 1,109,536 households instead.

People with no household language: "Viv. Colectivas" (40,216 people in collective dwellings:
barracks, convents, hospitals, prisons), "No especificado" (303), "No habla" (156) and "Psv"
(51; the engine does not expand it, probably people without a dwelling). Kept here as labels;
the mapping does not draw them.

THE SOURCE is the same REDATAM deployment religiondots read religion from
(religiondots/sources/py.py, which documents the route and its traps): prod.redatam.org's
/binpry/ cgi dir, base CPV2002. The portal call must come first; FORMAT=HTML and Submit are
required; the result is in <iframe>s.

CHECKS (the script stops unless all hold):
  1. 229 census districts (Asuncion's six barrios count separately); every district's label
     rows sum to its own printed Total, and Varon + Mujer = Total on every row.
  2. the districts sum to 5,163,198, the 2002 census population, to the person.
  3. a second run of the same cross by DEPTO (18 departments) equals the district sums on
     every label.
  4. per district, the total equals religiondots' independent tabulation of a different
     variable on the same microdata (religion: 10+ universe + its No Aplica), read-only from
     religiondots/data/raw/py/py_freq_DISTRITO.html.
"""
import argparse
import html
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "py"
NORM = HERE / "data" / "normalized"
RD_RAW = HERE.parent / "religiondots" / "data" / "raw" / "py" / "py_freq_DISTRITO.html"

R = "https://prod.redatam.org"
CGI = R + "/binpry"
BASE = "CPV2002"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

CENSUS_POP = 5_163_198
N_DISTRICTS = 229
N_DEPTOS = 18


def fetch():
    import requests
    import urllib3
    urllib3.disable_warnings()
    RAW.mkdir(parents=True, exist_ok=True)
    for ab in ("DISTRITO", "DEPTO"):
        s = requests.Session()
        s.headers.update({"User-Agent": UA, "Referer": R + "/redpry/"})
        s.get(f"{CGI}/RpWebEngine.exe/Portal?BASE={BASE}", timeout=90, verify=False)
        s.get(f"{CGI}/RpWebStats.exe/CrossTab?BASE={BASE}&ITEM=CRUZCOMBI&lang=ESP",
              timeout=120, verify=False)
        data = {"MAIN": "WebServerMain.inl", "BASE": BASE, "LANG": "ESP",
                "CODIGO": "XXUSUARIOXX", "ITEM": "CRUZCOMBI", "MODE": "RUN",
                "inputTitle": "", "ROW": "HOGAR.idiohog", "COLUMN": "PERSONA.sexo",
                "CONTROL": "", "AREABREAK": ab, "SELECTION": "ALL", "INLINESELECTION": "",
                "UNIVERSE": "", "FILTER": "", "TEXT_FILTER": "", "PERCENT": "PCT_1",
                "FORMAT": "HTML", "Submit": "Ejecutar"}
        r = s.post(f"{CGI}/RpWebStats.exe/CrossTab?", data=data, timeout=600, verify=False)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "latin-1"
        blob = r.text
        frames = re.findall(r'<i?frame[^>]+src="([^"]+)"', blob, re.I)
        if not frames:
            raise SystemExit(f"{ab}: no iframes in the response ({len(blob)} chars)")
        for f in frames:
            f = f.replace("&amp;", "&")
            u = f if f.startswith("http") else R + f
            rr = s.get(u, timeout=600, verify=False)
            rr.encoding = rr.apparent_encoding or "latin-1"
            blob += f"\n<!--FRAME {u}-->\n" + rr.text
        out = RAW / f"py_idiohog_sexo_{ab}.html"
        out.write_text(blob, encoding="utf-8")
        print(f"  {ab}: {len(blob):,} chars -> {out}")


def _cells(tr):
    c = [html.unescape(re.sub(r"<[^>]+>", "", x)).replace("\xa0", " ").strip()
         for x in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S | re.I)]
    return [x for x in c if x]


def _num(s):
    s = s.replace(" ", "").replace(".", "")
    if s == "-":            # the engine prints a zero cell as a dash
        return 0
    return int(s) if s.isdigit() else None


def read(ab):
    """-> rows [(code, name, label, total)], printed totals {code: n}, names {code: name}.

    Each area block is `AREA # <code> | <name>`, a two-row header, then `<label> | varon |
    mujer | total` rows and a `Total` row. The file ends with a whole-country block carrying
    no AREA header, so an area is CLOSED at its Total row (religiondots' py.py hit the same).
    """
    blob = (RAW / f"py_idiohog_sexo_{ab}.html").read_text(encoding="utf-8")
    rows, totals, names = [], {}, {}
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
        if cur is None or len(c) != 4:
            continue
        v = [_num(x) for x in c[1:]]
        if None in v:
            continue
        assert v[0] + v[1] == v[2], f"{cur} {c[0]}: {v[0]} + {v[1]} != {v[2]}"
        if c[0] == "Total":
            totals[cur] = v[2]
            cur = None
            continue
        rows.append((cur, names[cur], c[0], v[2]))
    return rows, totals, names


def _religion_population():
    """religiondots' religion run, per district: its 10+ Total plus its No Aplica (under 10s)."""
    blob = RD_RAW.read_text(encoding="utf-8")
    pop, cur = defaultdict(int), None
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", blob, re.S | re.I):
        c = _cells(tr)
        if not c:
            continue
        m = re.match(r"AREA\s*#\s*(\S+)$", c[0])
        if m:
            cur = m.group(1)
            continue
        if cur is None or len(c) < 2:
            continue
        v = _num(c[1])
        if v is None:
            continue
        if c[0].lower().startswith("total"):
            pop[cur] += v
        elif c[0].startswith("No Aplica"):
            pop[cur] += v
            cur = None
    return pop


def check(rows, totals, names):
    per = defaultdict(int)
    for code, _n, _l, v in rows:
        per[code] += v
    assert len(names) == N_DISTRICTS, f"{len(names)} districts, expected {N_DISTRICTS}"
    bad = [c for c in names if per[c] != totals.get(c)]
    assert not bad, f"{len(bad)} districts whose rows do not sum to their Total: {bad[:5]}"
    grand = sum(per.values())
    assert grand == CENSUS_POP, f"national {grand:,} != census {CENSUS_POP:,}"
    print(f"  1-2. {len(names)} districts, rows sum to their Totals; {grand:,} people "
          f"= the 2002 census population")

    drows, dtot, dnames = read("DEPTO")
    assert len(dnames) == N_DEPTOS, f"{len(dnames)} departments, expected {N_DEPTOS}"
    a, b = defaultdict(int), defaultdict(int)
    for code, _n, lab, v in rows:
        a[(code[:2], lab)] += v
    for code, _n, lab, v in drows:
        b[(code.zfill(2), lab)] += v
    diff = {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert not diff, f"district sums != department run: {list(diff.items())[:5]}"
    print(f"  3. the {len(dnames)}-department run equals the district sums on all "
          f"{len(b)} (department, label) cells")

    if RD_RAW.exists():
        rel = _religion_population()
        miss = [c for c in names if per[c] != rel.get(c)]
        assert not miss, (f"{len(miss)} districts differ from religiondots' religion run: "
                          f"{[(c, per[c], rel.get(c)) for c in miss[:5]]}")
        print(f"  4. all {len(names)} district totals equal religiondots' religion "
              f"tabulation (10+ and No Aplica) to the person")
    else:
        print(f"  4. skipped: {RD_RAW} not found")

    nat = defaultdict(int)
    for _c, _n, lab, v in rows:
        nat[lab] += v
    for lab, v in sorted(nat.items(), key=lambda kv: -kv[1]):
        print(f"     {lab:<20} {v:>10,}  {100 * v / grand:6.2f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    rows, totals, names = read("DISTRITO")
    check(rows, totals, names)
    NORM.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["geo_id", "geo_name", "source_category", "count"])
    df.insert(1, "geo_level", "distrito")
    df.sort_values(["geo_id", "source_category"]).to_csv(NORM / "py.csv", index=False)
    print(f"  wrote {NORM / 'py.csv'} ({len(df)} rows)")


if __name__ == "__main__":
    main()
