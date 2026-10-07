"""Honduras, XVII Censo de Poblacion y VI de Vivienda 2013 (INE): self-identification and
indigenous or Afro-Honduran people, per municipio.

    python sources/hn_censo.py [--fetch]

Writes data/normalized/hn.csv: one row per (municipio, code). Every person in the census base
(7,657,684, the enumerated count; INE's published 8,303,771 adds an omission adjustment) is in
exactly one row. `code` is the derived variable GRP defined in the program below:

    1-10   P05 Indigena, AfroHondureno or Negro, and P06 pueblo: 1 Maya-Chorti, 2 Lenca,
           3 Miskito, 4 Nahua, 5 Pech, 6 Tolupan, 7 Tawahka, 8 Garifuna, 9 Negro de habla
           inglesa, 10 Otro
    14     P05 Mestizo
    15     P05 Blanco
    16     P05 Otro

NO LANGUAGE QUESTION. The 2013 questionnaire (VARLIST on the server, entity PERSONA) asks P05
"Auto-identificacion" (6 answers) and, of the first three, P06 "Pueblo indigena" (10 answers).
Nothing about language. taxonomy/hn2013.py reads each people as a language under AGENT_BRIEF
section 2's ethnicity rule; sources/hn.md has the retention calls.

THE SOURCE is INE's REDATAM webserver (181.115.7.199/binhnd, base CPVHND2013NAC, open, no
login). Programs and outputs are saved in data/raw/hn/.

CHECKS (the script stops unless all hold):
  1. 298 municipios; every person in exactly one code; codes sum to 7,657,684.
  2. per municipio and pueblo, GRP 1-10 equals a separate FREQUENCY of the raw P06 by
     AREABREAK; GRP 14-16 and the sum of 1-10 equal a separate FREQUENCY of P05.
  3. the 298 municipios rebuild the national P06 x P05 crosstab, pueblo by pueblo.
  4. the 18 departments: municipio code prefixes give 18 departments, and the AREALIST by
     department equals the municipio sums.
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
RAW = HERE / "data" / "raw" / "hn"
NORM = HERE / "data" / "normalized"

HOST = "http://181.115.7.199/binhnd"
BASE = "CPVHND2013NAC"

GRP = """DEFINE PERSONA.GRP
 AS SWITCH
  INCASE PERSONA.P05 <= 3
   ASSIGN PERSONA.P06
  INCASE PERSONA.P05 = 4
   ASSIGN 14
  INCASE PERSONA.P05 = 5
   ASSIGN 15
  INCASE PERSONA.P05 = 6
   ASSIGN 16
  DEFAULT 0
 TYPE INTEGER
 RANGE 0-16
"""

PROGRAMS = {
    "hn_mun_grp": "RUNDEF Job\n SELECTION ALL\n" + GRP + "TABLE T\n AS AREALIST\n OF MUNIC, PERSONA.GRP\n",
    "hn_dep_grp": "RUNDEF Job\n SELECTION ALL\n" + GRP + "TABLE T\n AS AREALIST\n OF DEPTO, PERSONA.GRP\n",
    "hn_break_p06": "RUNDEF Job\n SELECTION ALL\nTABLE T\n AS FREQUENCY\n OF PERSONA.P06\n AREABREAK MUNIC\n",
    "hn_break_p05": "RUNDEF Job\n SELECTION ALL\nTABLE T\n AS FREQUENCY\n OF PERSONA.P05\n AREABREAK MUNIC\n",
    "hn_nat_p06_p05": "RUNDEF Job\n SELECTION ALL\nTABLE T\n AS CROSSTABS\n OF PERSONA.P06 BY PERSONA.P05\n",
}

PUEBLOS = {1: "Maya -Chortí", 2: "Lenca", 3: "Miskito", 4: "Nahua", 5: "Pech", 6: "Tolupán",
           7: "Tawahka", 8: "Garífuna", 9: "Negro de habla inglesa", 10: "Otro"}
P05 = {1: "Indígena", 2: "AfroHondureño", 3: "Negro (a)", 4: "Mestizo (a)", 5: "Blanco (a)",
       6: "Otro"}
CATEGORY = {**{k: v for k, v in PUEBLOS.items()},
            10: "Otro pueblo (indigenous or Afro, people not listed)",
            14: "Mestizo", 15: "Blanco", 16: "Otro (self-identification)"}
CENSUS_POPULATION = 7_657_684
N_MUN, N_DEP = 298, 18


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
        body = t.content.decode("cp1252", errors="replace")
        if "Tabla vac" in body or "<table" not in body.lower():
            raise SystemExit(f"{name}: REDATAM returned no table.\n{body[:600]}")
        (RAW / f"{name}.program.txt").write_text(program, encoding="utf-8")
        dest.write_text(body, encoding="utf-8")
        print(f"  {dest.stat().st_size:,} bytes")
        time.sleep(1)


def _unmangle(c):
    if "Ã" in c or "Â" in c:
        try:
            return c.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return c
    return c


def _rows(name):
    p = RAW / f"{name}.htm"
    if not p.exists():
        raise SystemExit(f"missing {p} -- run with --fetch")
    body = p.read_text(encoding="utf-8")
    out = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S | re.I):
        cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).replace("\xa0", " ").strip()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S | re.I)]
        cells = [_unmangle(c) for c in cells if c]
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


def read_arealist(name, expected):
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
    if len(data) != expected:
        raise SystemExit(f"{name}: {len(data)} areas, expected {expected}")
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
                raise SystemExit(f"{name}: municipio {cur} twice")
            names[cur] = r[1] if len(r) > 1 else ""
            data[cur] = {}
            continue
        if cur and len(r) == 4 and r[2].endswith("%") is False and r[0] != "Total":
            pass
        if cur and len(r) >= 3 and r[0] not in ("Total",) and not r[0].startswith("No Aplica"):
            try:
                data[cur][r[0]] = _num(r[1])
            except ValueError:
                continue
    return data, names


def main():
    if "--fetch" in sys.argv:
        fetch()
    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'OK ' if cond else 'BAD'} {msg}")

    mun = read_arealist("hn_mun_grp", N_MUN)
    dep = read_arealist("hn_dep_grp", N_DEP)
    codes = sorted({k for v in mun.values() for k in v})
    check(set(codes) <= set(CATEGORY), f"GRP codes {codes} all in the plan")

    tot = sum(sum(v.values()) for v in mun.values())
    check(tot == CENSUS_POPULATION, f"codes sum to {tot:,} ({CENSUS_POPULATION:,} in the base)")

    rolled = {}
    for m, v in mun.items():
        d = m[:-2]
        acc = rolled.setdefault(d, {})
        for k, n in v.items():
            acc[k] = acc.get(k, 0) + n
    bad = [d for d in dep if {k: n for k, n in dep[d].items() if n}
           != {k: n for k, n in rolled.get(d, {}).items() if n}]
    check(set(rolled) == set(dep) and not bad,
          f"the {N_MUN} municipios rebuild the {N_DEP} departments on every code ({bad[:4]})")

    p06, names = read_break("hn_break_p06")
    p05, names5 = read_break("hn_break_p05")
    check(set(p06) == set(mun) and set(p05) == set(mun) and names == names5,
          f"both AREABREAK runs name the same {len(names)} municipios as the AREALIST")
    bad = []
    for m, v in mun.items():
        for k, lab in PUEBLOS.items():
            if p06[m].get(lab, 0) != v.get(k, 0):
                bad.append((m, lab, p06[m].get(lab, 0), v.get(k, 0)))
        ind = sum(v.get(k, 0) for k in PUEBLOS)
        if sum(p05[m].get(P05[k], 0) for k in (1, 2, 3)) != ind:
            bad.append((m, "P05 1-3"))
        for k, lab in ((14, P05[4]), (15, P05[5]), (16, P05[6])):
            if p05[m].get(lab, 0) != v.get(k, 0):
                bad.append((m, lab))
    check(not bad, f"GRP equals the raw P06 and P05 tabulations on every municipio "
                   f"({len(bad)} failures {bad[:4]})")

    xt = {}
    for r in _rows("hn_nat_p06_p05"):
        k = next((c for c, lab in PUEBLOS.items() if lab == r[0]), None)
        if k is not None and len(r) == 5:
            xt[k] = _num(r[4])
    nat = {}
    for v in mun.values():
        for k, n in v.items():
            nat[k] = nat.get(k, 0) + n
    bad = [k for k in PUEBLOS if xt.get(k) != nat.get(k, 0)]
    check(len(xt) == 10 and not bad, f"national P06 x P05 crosstab equals the municipio sums ({bad})")

    if not ok:
        raise SystemExit("reconciliation FAILED")

    rows = []
    for m in sorted(mun):
        for k, n in sorted(mun[m].items()):
            if n:
                rows.append(dict(geo_level="municipio", geo_id=m, geo_name=names[m].title(),
                                 code=k, source_category=CATEGORY[k], count=n))
    df = pd.DataFrame(rows)
    NORM.mkdir(parents=True, exist_ok=True)
    df.to_csv(NORM / "hn.csv", index=False)
    pop = df.groupby(["geo_id", "geo_name"])["count"].sum().reset_index()
    pop.rename(columns={"count": "population"}).to_csv(NORM / "hn_units.csv", index=False)
    print(f"\nwrote {NORM / 'hn.csv'} ({len(df):,} rows)\n\nnational:")
    for k in sorted(nat):
        print(f"  {k:>3} {nat[k]:>10,}  {CATEGORY[k]}")


if __name__ == "__main__":
    main()
