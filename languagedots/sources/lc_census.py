"""Saint Lucia, Population and Housing Census 2022 (CSO): ethnicity and country of birth, per
district -> data/normalized/lc.csv. A copy of sources/tt_census.py with Saint Lucia's
variables (P1_4 ethnicity, P2_2 born in Saint Lucia or abroad, P2_2_2ABROAD_R country of birth).

    python sources/lc_census.py [--fetch]

NO LANGUAGE QUESTION (VARLIST read). Built as Dominica (sources/dm.md) and Barbados: the
native-born on Kweyol, white natives on English, the foreign-born on their origin's languages
(taxonomy/lc2022.py).

THE SOURCE is CSO's REDATAM base PHC2022 on CEPAL's server (prod.redatam.org/binlca, open). The
base is the unweighted enumeration; CSO weighted each district up for a 23% undercount before
publishing (religiondots/countries/lc.py), so countries/lc.py scales each district's mix to the
published district total.

CHECKS: 10 districts in every table; crosstab margins within the one-way tables; countries of
birth within the born-abroad count; persons agree across two tables.
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
RAW = HERE / "data" / "raw" / "lc"
NORM = HERE / "data" / "normalized"
ROOT = "https://prod.redatam.org"
HOST = ROOT + "/binlca"
BASE = "PHC2022"
FOREIGN = "Born elsewhere"  # P2_2's born-abroad label (asserted below)
N_AREAS = 12                # Castries as City, Suburban and Rural; religiondots' LC13 is all three

PROGRAMS = {
    "lc_break_ethnic": "RUNDEF Job\n SELECTION ALL\nTABLE T\n AS FREQUENCY\n OF PERSON.P1_4\n AREABREAK DISTRICT\n",
    "lc_break_plabirth": "RUNDEF Job\n SELECTION ALL\nTABLE T\n AS FREQUENCY\n OF PERSON.P2_2\n AREABREAK DISTRICT\n",
    "lc_break_cnt": "RUNDEF Job\n SELECTION ALL\nTABLE T\n AS FREQUENCY\n OF PERSON.P2_2_2ABROAD_R\n AREABREAK DISTRICT\n",
    "lc_eth_by_plabirth": "RUNDEF Job\n SELECTION ALL\nTABLE T\n AS CROSSTABS\n OF PERSON.P1_4 BY PERSON.P2_2\n AREABREAK DISTRICT\n",
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
            "MAIN": "WebServerMain.inl", "BASE": BASE, "LANG": "eng",
            "CODIGO": "XXUSUARIOXX", "ITEM": "PROGRED", "MODE": "RUN",
            "CMDSET": program, "Submit": "Ejecutar"},
            headers={"User-Agent": "Mozilla/5.0"}, timeout=1800, verify=False)
        r.raise_for_status()
        m = re.search(r"<iframe src=\"([^\"]+)\"", r.text)
        if not m:
            raise SystemExit(f"{name}: no output frame\n{r.text[:800]}")
        t = requests.get(ROOT + m.group(1), headers={"User-Agent": "Mozilla/5.0"},
                         timeout=1800, verify=False)
        t.raise_for_status()
        body = t.content.decode("utf-8-sig", errors="replace")
        if "<table" not in body.lower():
            raise SystemExit(f"{name}: no table\n{body[:600]}")
        (RAW / f"{name}.program.txt").write_text(program, encoding="utf-8")
        dest.write_text(body, encoding="utf-8")
        print(f"  {dest.stat().st_size:,} bytes")
        time.sleep(1)


def rows(name):
    body = (RAW / f"{name}.htm").read_text(encoding="utf-8")
    out = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S | re.I):
        cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).replace("\xa0", " ").strip()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S | re.I)]
        cells = [c for c in cells if c]
        if cells:
            out.append(cells)
    return out


def num(tok):
    t = tok.replace(" ", "").replace(",", "")
    if t == "-":
        return 0
    if not re.fullmatch(r"\d+", t):
        raise ValueError(tok)
    return int(t)


def read_break(name):
    data, names, cur = {}, {}, None
    for r in rows(name):
        if r[0] in ("RESUMEN", "SUMMARY"):
            break
        m = re.fullmatch(r"AREA # (\d+)", r[0])
        if m:
            cur = m.group(1)
            names[cur] = r[1] if len(r) > 1 else ""
            data[cur] = {}
            continue
        if cur and len(r) == 4 and r[2].endswith("%") and r[0] != "Total":
            try:
                data[cur][r[0]] = num(r[1])
            except ValueError:
                continue
    return data, names


def read_break_xt(name):
    """CROSSTABS ... AREABREAK -> {area: {(row, col): n}}"""
    data, cur, cols = {}, None, None
    for r in rows(name):
        if r[0] in ("RESUMEN", "SUMMARY"):
            break
        m = re.fullmatch(r"AREA # (\d+)", r[0])
        if m:
            cur, cols = m.group(1), None
            data[cur] = {}
            continue
        if cur and r[-1] == "Total" and cols is None and len(r) > 2:
            cols = r[:-1]
            continue
        if cur and cols and len(r) == len(cols) + 2 and r[0] != "Total":
            for c, x in zip(cols, r[1:-1]):
                data[cur][(r[0], c)] = num(x)
    return data


def base_persons(name, area):
    """Every person of an area (all areas if None): the table's Total + NotApp + Missing."""
    tot, cur = 0, None
    for r in rows(name):
        if r[0] in ("RESUMEN", "SUMMARY"):
            break
        m = re.fullmatch(r"AREA # (\d+)", r[0])
        if m:
            cur = m.group(1)
            continue
        if area is not None and cur != area:
            continue
        if (r[0] == "Total" and len(r) == 4) or r[0] in ("NotApp :", "Missing :"):
            tot += num(r[1])
    return tot


def main():
    if "--fetch" in sys.argv:
        fetch()
    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'OK ' if cond else 'BAD'} {msg}")

    eth, names = read_break("lc_break_ethnic")
    pla, _ = read_break("lc_break_plabirth")
    cnt, _ = read_break("lc_break_cnt")
    xt = read_break_xt("lc_eth_by_plabirth")
    print("     districts:", names)
    print("     P1_4:", sorted({k for v in eth.values() for k in v}))
    print("     P2_2:", sorted({k for v in pla.values() for k in v}))
    check(len(eth) == N_AREAS and set(eth) == set(pla) == set(cnt) == set(xt),
          f"{len(eth)} districts in every table")
    check(all(FOREIGN in v for v in pla.values()), f"P2_2 has a {FOREIGN!r} answer everywhere")
    pop = sum(sum(v.values()) for v in eth.values())
    print(f"     population {pop:,}")
    bad = []
    for m in eth:
        e = {}
        p = {}
        for (r, c), n in xt[m].items():
            e[r] = e.get(r, 0) + n
            p[c] = p.get(c, 0) + n
        # the crosstab drops anyone missing either answer, so its margins sit at or under each
        # one-way table, by at most that table's own Missing
        if any(n > eth[m].get(k, 0) for k, n in e.items()):
            bad.append((m, "ETHNIC"))
        if any(n > pla[m].get(k, 0) for k, n in p.items()):
            bad.append((m, "PLABIRTH"))
        fb = sum(n for c, n in cnt[m].items())
        if fb > pla[m].get(FOREIGN, 0):
            bad.append((m, "country", fb, pla[m].get(FOREIGN)))
        print(f"     {m}: country of birth given for {fb:,} of {pla[m].get(FOREIGN, 0):,} foreign-born")
    check(not bad, f"crosstab margins within the ETHNIC and PLABIRTH tables, and countries of "
                   f"birth within PLABIRTH's foreign-born ({bad[:4]})")
    persons = {m: base_persons("lc_break_plabirth", m) for m in pla}
    tot = sum(persons.values())
    check(tot == base_persons("lc_break_ethnic", None),
          f"persons per municipality (answers + NotApp + Missing) sum to {tot:,} in both tables")
    if not ok:
        raise SystemExit("reconciliation FAILED")
    out = []
    for m in sorted(xt):
        out.append(dict(geo_id=m, geo_name=names[m], kind="persons", ethnic="", place="",
                        country="", count=persons[m]))
        for (e, p), n in xt[m].items():
            if n:
                out.append(dict(geo_id=m, geo_name=names[m], kind="ethnic_x_birthplace",
                                ethnic=e, place=p, country="", count=n))
        for c, n in cnt[m].items():
            if n:
                out.append(dict(geo_id=m, geo_name=names[m], kind="country", ethnic="",
                                place="", country=c, count=n))
    df = pd.DataFrame(out)
    df.to_csv(NORM / "lc.csv", index=False)
    nat = df[df.kind == "country"].groupby("country")["count"].sum().sort_values(ascending=False)
    print(f"wrote {NORM / 'lc.csv'}({len(df):,} rows)\ncountry of birth, top:\n{nat.head(25).to_string()}")


if __name__ == "__main__":
    main()
