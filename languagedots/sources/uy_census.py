"""Uruguay, Censos 2011 (INE): country of birth of the foreign-born, per department and, in
Montevideo, per barrio -> data/normalized/uy.csv.

    python sources/uy_census.py [--fetch]

NO LANGUAGE QUESTION, and the census's ethnic-ancestry question (afro, indigenous, white,
Asian) names no group with a language of its own in use: Uruguay's indigenous languages are
long gone. Built under AGENT_BRIEF section 2's rulings for countries with no language question:
the Uruguayan-born on Spanish, the foreign-born on their origin's languages
(sources/origin_mix.py). No source counts Portuguese (or Portunol) as a first language on the
Brazilian border: INE's Encuesta Telefonica de Idiomas 2019 measures knowledge of Portuguese
(29.7%, highest in Artigas, Rivera and Cerro Largo), a learned-language figure the 2026-10-05
ruling keeps off the map (sources/uy.md).

THE SOURCE is INE's REDATAM base CPV2011 on CEPAL's server (prod.redatam.org/binury, open). The
unit U is the department (DEPTO.DEPTO, INE's code; Montevideo is 1) outside Montevideo and
100 + VIVIENDA.CBAR, INE's barrio number 1-62, inside it: religiondots' Uruguay units
(18 departments + 62 barrios, religiondots/data/geo/uy/*_lookup.csv).

CHECKS: the crosstab's unit totals equal a separate FREQUENCY of U; its country totals equal a
separate national FREQUENCY of PAISNAC; units sum to the census population 3,286,314; 80 units;
every country label has an ISO code (COUNTRY below) or is listed as a remainder.
"""
import html
import re
import sys
import time
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "uy"
NORM = HERE / "data" / "normalized"
HOST = "https://prod.redatam.org/binury"
BASE = "CPV2011"
CENSUS_POPULATION = 3_285_877    # the public base; INE publishes 3,286,314 (437 more)

# a variable in DEFAULT evaluates to 0 on this engine, so each department has its own INCASE
U = ("DEFINE PERSONA.U\n AS SWITCH\n  INCASE DEPTO.DEPTO = 1\n   ASSIGN 100 + VIVIENDA.CBAR\n"
     + "".join(f"  INCASE DEPTO.DEPTO = {k}\n   ASSIGN {k}\n" for k in range(2, 20))
     + "  DEFAULT 0\n TYPE INTEGER\n RANGE 0-199\n")
PROGRAMS = {
    "uy_pais_by_u": "RUNDEF Job\n SELECTION ALL\n" + U + "TABLE T\n AS CROSSTABS\n OF PERSONA.PAISNAC BY PERSONA.U\n",
    "uy_u": "RUNDEF Job\n SELECTION ALL\n" + U + "TABLE T\n AS FREQUENCY\n OF PERSONA.U\n",
    "uy_pais": "RUNDEF Job\n SELECTION ALL\nTABLE T\n AS FREQUENCY\n OF PERSONA.PAISNAC\n",
}


def fetch():
    import requests
    import urllib3
    urllib3.disable_warnings()
    RAW.mkdir(parents=True, exist_ok=True)
    for name, program in PROGRAMS.items():
        dest = RAW / f"{name}.htm"
        if dest.exists() and dest.stat().st_size > 2_000:
            print("already have", dest.name)
            continue
        print("RUN", name)
        r = requests.post(f"{HOST}/RpWebStats.exe/CmdSet?", data={
            "MAIN": "WebServerMain.inl", "BASE": BASE, "LANG": "esp",
            "CODIGO": "XXUSUARIOXX", "ITEM": "PROGRED", "MODE": "RUN",
            "CMDSET": program, "Submit": "Ejecutar"},
            headers={"User-Agent": "Mozilla/5.0"}, timeout=1800, verify=False)
        r.raise_for_status()
        m = re.search(r"<iframe src=\"([^\"]+)\"", r.text)
        if not m:
            raise SystemExit(f"{name}: no output frame\n{r.text[:800]}")
        t = requests.get("https://prod.redatam.org" + m.group(1),
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=1800, verify=False)
        t.raise_for_status()
        body = t.content.decode("utf-8-sig", errors="replace")
        if "<table" not in body.lower():
            raise SystemExit(f"{name}: no table\n{body[:600]}")
        (RAW / f"{name}.program.txt").write_text(program, encoding="utf-8")
        dest.write_text(body, encoding="utf-8")
        print(f"  {dest.stat().st_size:,} bytes")
        time.sleep(1)


def _rows(name):
    body = (RAW / f"{name}.htm").read_text(encoding="utf-8")
    out = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S | re.I):
        cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).replace("\xa0", " ").strip()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S | re.I)]
        cells = [c for c in cells if c]
        if cells:
            out.append(cells)
    return out


def _num(tok):
    t = tok.replace(" ", "").replace(",", "")
    if t == "-":
        return 0
    if not re.fullmatch(r"\d+", t):
        raise ValueError(tok)
    return int(t)


def _freq(name):
    out = {}
    for r in _rows(name):
        if len(r) == 4 and r[2].endswith("%"):
            try:
                out[r[0]] = _num(r[1])
            except ValueError:
                pass
        elif len(r) == 2 and r[0].startswith("No Aplica"):
            out["No Aplica"] = _num(r[1])
    return out


def _crosstab(name):
    rows = _rows(name)
    hi = next(i for i, r in enumerate(rows) if r[-1] == "Total" and len(r) > 10)
    cols = rows[hi][:-1]
    xt = {}
    for r in rows[hi + 1:]:
        if len(r) == len(cols) + 2:
            xt[r[0]] = dict(zip(cols + ["Total"], (_num(x) for x in r[1:])))
    return xt, cols


def main():
    if "--fetch" in sys.argv:
        fetch()
    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'OK ' if cond else 'BAD'} {msg}")

    import uy2011
    xt, cols = _crosstab("uy_pais_by_u")
    ufreq = _freq("uy_u")
    pfreq = _freq("uy_pais")
    tot_u = {k: v for k, v in ufreq.items() if k not in ("Total", "No Aplica")}
    check(len(cols) == 80 and set(cols) == set(tot_u), f"{len(cols)} units in the crosstab, "
          f"{len(tot_u)} in FREQUENCY of U")
    pop = sum(tot_u.values())
    check(pop == CENSUS_POPULATION, f"units sum to {pop:,} ({CENSUS_POPULATION:,})")
    foreign = {c: xt["Total"][c] for c in cols}
    check(all(foreign[c] <= tot_u[c] for c in cols), "foreign-born never exceed a unit's people")
    bad = [k for k, v in xt.items() if k != "Total" and pfreq.get(k) != v["Total"]]
    check(not bad, f"country totals equal the national FREQUENCY of PAISNAC ({bad[:4]})")
    labels = [k for k in xt if k != "Total"]
    unk = [k for k in labels if k not in uy2011.COUNTRY and k not in uy2011.REMAINDER]
    check(not unk, f"every country label has an ISO code or is a listed remainder ({unk})")
    if not ok:
        raise SystemExit("reconciliation FAILED")
    rows = []
    for c in cols:
        native = tot_u[c] - foreign[c]
        rows.append(dict(unit=int(c), origin="UY", count=native))
        for k in labels:
            n = xt[k][c]
            if n:
                rows.append(dict(unit=int(c), origin=uy2011.COUNTRY.get(k, "rest"), count=n))
    df = pd.DataFrame(rows).groupby(["unit", "origin"], as_index=False)["count"].sum()
    df.to_csv(NORM / "uy.csv", index=False)
    fb = sum(foreign.values())
    print(f"\nwrote {NORM / 'uy.csv'} ({len(df):,} rows); foreign-born {fb:,} "
          f"({fb / pop:.1%}); top: " + ", ".join(
              f"{k} {xt[k]['Total']:,}" for k in sorted(labels, key=lambda k: -xt[k]["Total"])[:8]))


if __name__ == "__main__":
    sys.path.insert(0, str(HERE / "taxonomy"))
    main()
