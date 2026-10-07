"""Panama, XII Censo de Poblacion y VIII de Vivienda 2023 (INEC): indigenous people (P08) and
Afro-descendant group (P09), per corregimiento.

    python sources/pa_censo.py [--fetch]

Writes data/normalized/pa.csv: one row per (corregimiento, code), code = 10 * P08 + P09, the
two self-identification questions (every person answers both). Labels are recovered from the
national P08 x P09 crosstab, which prints them (the AREALIST prints only codes): each P08 code's
national total must equal exactly one crosstab row total, and likewise for P09.

NO LANGUAGE QUESTION. The 2023 person questionnaire (VARLIST on the server) asks P08 "grupo
indigena" and P09 "grupo afrodescendiente" and nothing on language. taxonomy/pa2023.py reads
each group as its language with retention shares from UNICEF/MINSA MICS 2013 (sources/pa.md).

THE SOURCE is INEC's REDATAM webserver (www.inec.gob.pa/panbin, base LP2023, open, no login).

CHECKS (the script stops unless all hold):
  1. codes sum to the census population, 4,064,780, in the national crosstab's Total.
  2. per corregimiento, the P08 and P09 margins of the joint code equal separate AREABREAK
     FREQUENCY tables of P08 and of P09.
  3. the national joint code equals the national P08 x P09 crosstab cell by cell.
  4. corregimientos rebuild the provinces and comarcas (AREALIST by PROVINCIA).
"""
import html
import re
import sys
import time
import urllib.parse
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "pa"
NORM = HERE / "data" / "normalized"

HOST = "https://www.inec.gob.pa/panbin"
BASE = "LP2023"

GRP = """DEFINE PERSONA.GRP
 AS PERSONA.P08_INDIG * 10 + PERSONA.P09_AFROD
 TYPE INTEGER
 RANGE 0-999
"""
PROGRAMS = {
    # an AREALIST of ~100 code columns makes the server answer 404, so FREQUENCY by AREABREAK
    "pa_corr_grp": "RUNDEF Job\n SELECTION ALL\n" + GRP + "TABLE T\n AS FREQUENCY\n OF PERSONA.GRP\n AREABREAK CORREG\n",
    "pa_prov_grp": "RUNDEF Job\n SELECTION ALL\n" + GRP + "TABLE T\n AS FREQUENCY\n OF PERSONA.GRP\n AREABREAK PROVINCIA\n",
    "pa_break_p08": "RUNDEF Job\n SELECTION ALL\nTABLE T\n AS FREQUENCY\n OF PERSONA.P08_INDIG\n AREABREAK CORREG\n",
    "pa_break_p09": "RUNDEF Job\n SELECTION ALL\nTABLE T\n AS FREQUENCY\n OF PERSONA.P09_AFROD\n AREABREAK CORREG\n",
    "pa_nat_p08_p09": "RUNDEF Job\n SELECTION ALL\nTABLE T\n AS CROSSTABS\n OF PERSONA.P08_INDIG BY PERSONA.P09_AFROD\n",
}
CENSUS_POPULATION = 4_064_780


def fetch():
    import requests

    RAW.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0"
    for name, program in PROGRAMS.items():
        dest = RAW / f"{name}.htm"
        if dest.exists() and dest.stat().st_size > 2_000:
            print("already have", dest.name)
            continue
        print("RUN", name)
        r = s.post(f"{HOST}/RpWebStats.exe/CmdSet?", data={
            "MAIN": "WebServerMain.inl", "BASE": BASE, "LANG": "esp",
            "CODIGO": "XXUSUARIOXX", "ITEM": "PROGRED", "MODE": "RUN",
            "CMDSET": program, "Submit": "Ejecutar"}, timeout=1800)
        r.raise_for_status()
        m = re.search(r"(RpBases[^\"'&<>]*?\.htm)", r.text)
        if not m:
            raise SystemExit(f"{name}: REDATAM returned no output file.\n{r.text[:800]}")
        t = s.get(f"{HOST}/RpWebUtilities.exe/Text?LFN=" + urllib.parse.quote(m.group(1))
                  + "&TYPE=TMP", timeout=1800)
        t.raise_for_status()
        body = t.content.decode("utf-8", errors="replace")
        if "Tabla vac" in body or "<table" not in body.lower():
            raise SystemExit(f"{name}: REDATAM returned no table.\n{body[:600]}")
        (RAW / f"{name}.program.txt").write_text(program, encoding="utf-8")
        dest.write_text(body, encoding="utf-8")
        print(f"  {dest.stat().st_size:,} bytes")
        time.sleep(1)


def _rows(name):
    p = RAW / f"{name}.htm"
    if not p.exists():
        raise SystemExit(f"missing {p} -- run with --fetch")
    body = p.read_text(encoding="utf-8")
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
        raise ValueError(f"{tok!r} is not a figure")
    return int(t)


def read_arealist(name):
    rows = _rows(name)
    header = next(r for r in rows if r[0] == "Código")
    cols = header[1:]
    if cols[-1] != "Total":
        raise SystemExit(f"{name}: AREALIST header ends {cols[-1]!r}")
    data = {}
    for r in rows:
        if re.fullmatch(r"\d+", r[0].replace(" ", "")) and len(r) == len(cols) + 1:
            v = [_num(x) for x in r[1:]]
            row = dict(zip((int(c) for c in cols[:-1]), v[:-1]))
            if sum(row.values()) != v[-1]:
                raise SystemExit(f"{name}: {r[0]} sums to {sum(row.values())}, Total {v[-1]}")
            if r[0] in data:
                raise SystemExit(f"{name}: {r[0]} twice")
            data[r[0].strip()] = row
    return data


def read_break(name):
    rows = _rows(name)
    data, names = {}, {}
    cur = None
    for r in rows:
        if r[0] == "RESUMEN":
            break
        m = re.fullmatch(r"AREA # (\d+)", r[0])
        if m:
            cur = m.group(1)
            if cur in data:
                raise SystemExit(f"{name}: area {cur} twice")
            names[cur] = r[1] if len(r) > 1 else ""
            data[cur] = {}
            continue
        if cur and len(r) >= 3 and r[0] != "Total" and not r[0].startswith("No Aplica"):
            try:
                data[cur][r[0]] = _num(r[1])
            except ValueError:
                continue
    return data, names


def read_crosstab():
    rows = _rows("pa_nat_p08_p09")
    hi = next(i for i, r in enumerate(rows) if r[-1] == "Total" and len(r) > 5)
    cols = rows[hi][:-1]
    xt = {}
    for r in rows[hi + 1:]:
        if len(r) == len(cols) + 2:
            xt[r[0]] = dict(zip(cols + ["Total"], (_num(x) for x in r[1:])))
    return xt, cols


def labels_by_total(code_tot, label_tot, what):
    """code -> label, matching national totals one to one."""
    out = {}
    for c, n in code_tot.items():
        hits = [lab for lab, m in label_tot.items() if m == n]
        if len(hits) != 1:
            raise SystemExit(f"{what} code {c} total {n:,} matches {hits} crosstab labels")
        out[c] = hits[0]
    if len(set(out.values())) != len(out):
        raise SystemExit(f"{what}: two codes on one label")
    return out


def main():
    if "--fetch" in sys.argv:
        fetch()
    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'OK ' if cond else 'BAD'} {msg}")

    cor, _ = read_break("pa_corr_grp")
    cor = {c: {int(k): n for k, n in v.items()} for c, v in cor.items()}
    prov, _ = read_break("pa_prov_grp")
    prov = {c: {int(k): n for k, n in v.items()} for c, v in prov.items()}
    xt, afro_cols = read_crosstab()
    nat = {}
    for v in cor.values():
        for k, n in v.items():
            nat[k] = nat.get(k, 0) + n
    tot = sum(nat.values())
    check(tot == CENSUS_POPULATION == xt["Total"]["Total"],
          f"codes sum to {tot:,} (crosstab Total {xt['Total']['Total']:,})")
    p08_tot, p09_tot = {}, {}
    for k, n in nat.items():
        p08_tot[k // 10] = p08_tot.get(k // 10, 0) + n
        p09_tot[k % 10] = p09_tot.get(k % 10, 0) + n
    L08 = labels_by_total(p08_tot, {r: v["Total"] for r, v in xt.items() if r != "Total"}, "P08")
    L09 = labels_by_total(p09_tot, {c: xt["Total"][c] for c in afro_cols}, "P09")
    print("     P08:", L08)
    print("     P09:", L09)
    bad = [k for k, n in nat.items() if xt[L08[k // 10]][L09[k % 10]] != n]
    check(not bad, f"national joint code equals the crosstab cell by cell ({bad[:4]})")

    b08, names = read_break("pa_break_p08")
    b09, names9 = read_break("pa_break_p09")
    check(set(b08) == set(cor) == set(b09) and names == names9,
          f"{len(cor)} corregimientos, both AREABREAK runs name the same ones")
    bad = []
    for c, v in cor.items():
        m08, m09 = {}, {}
        for k, n in v.items():
            if n:
                m08[L08[k // 10]] = m08.get(L08[k // 10], 0) + n
                m09[L09[k % 10]] = m09.get(L09[k % 10], 0) + n
        if m08 != {k: n for k, n in b08[c].items() if n} or m09 != {k: n for k, n in b09[c].items() if n}:
            bad.append(c)
    check(not bad, f"P08 and P09 margins equal the AREABREAK tables on every corregimiento "
                   f"({len(bad)} failures {bad[:4]})")
    rolled = {}
    for c, v in cor.items():
        acc = rolled.setdefault(c[:2], {})
        for k, n in v.items():
            acc[k] = acc.get(k, 0) + n
    badp = [p for p in prov if {k: n for k, n in prov[p].items() if n}
            != {k: n for k, n in rolled.get(p, {}).items() if n}]
    check(set(prov) == set(rolled) and not badp,
          f"corregimientos rebuild the {len(prov)} provinces and comarcas ({badp})")
    if not ok:
        raise SystemExit("reconciliation FAILED")

    rows = []
    for c in sorted(cor):
        for k, n in sorted(cor[c].items()):
            if n:
                rows.append(dict(geo_level="corregimiento", geo_id=c, geo_name=names[c],
                                 code=k, indigenous=L08[k // 10], afro=L09[k % 10], count=n))
    df = pd.DataFrame(rows)
    df.to_csv(NORM / "pa.csv", index=False)
    pop = df.groupby(["geo_id", "geo_name"])["count"].sum().reset_index()
    pop.rename(columns={"count": "population"}).to_csv(NORM / "pa_units.csv", index=False)
    print(f"\nwrote {NORM / 'pa.csv'} ({len(df):,} rows), {len(cor)} corregimientos")


if __name__ == "__main__":
    main()
