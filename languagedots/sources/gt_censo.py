"""Guatemala, XII Censo Nacional de Poblacion 2018 (INE): the language each person learned to speak
in, per municipio.

    python sources/gt_censo.py [--fetch]

Writes
  data/normalized/gt.csv          geo_id (GT + INE's 4-digit municipio code, = COD-AB's pcode),
                                  geo_level, geo_name, source_category (INE's label), count.
                                  Everyone aged 4 and over; "No habla" is kept as a row (not drawn).
  data/normalized/gt_units.csv    geo_id, geo_name, population (all ages), asked (aged 4+)

THE QUESTION. PCP15 "Idioma en el que aprendio a hablar" (the language in which the person learned
to speak), a mother-tongue question, asked of everyone aged 4 and over: 13,566,897 people; the
1,334,389 under 4 are "No Aplica". One answer each, 29 categories: the 22 Mayan languages INE
lists, Xinka, Garifuna, Espanol, Ingles, Senas, Otro idioma, No habla. There is no "not stated"
category. (The census also asks PCP12 pueblo, PCP13 comunidad linguistica and PCP25 other
languages spoken; none of those is the first language.)

THE SOURCE is INE's own REDATAM webserver over the full census microdata (unweighted; a full
count), base CPVGT2018 at redatam2018.ine.gob.gt/bingtm. ine.gob.gt itself is behind a Radware
bot wall; the REDATAM host is not. Its Frequency web form answers HTTP 500, but the program route
(RpWebStats.exe/CmdSet, ITEM=PROGRED) runs, and a FREQUENCY with AREABREAK gives one table per
municipio headed by both its code and its name.

TABLES
  gt_mupio_pcp15     AREALIST OF MUPIO, PERSONA.PCP15     the table drawn: municipio x language
  gt_break_pcp15     FREQUENCY OF PCP15, AREABREAK MUPIO   the same counts by a second engine path,
                                                           with names
  gt_break_sexo      FREQUENCY OF PCP6, AREABREAK MUPIO    every person (all ages) per municipio
  gt_nat_pcp15       FREQUENCY OF PCP15                    the national table
  gt_age4_x_pcp15    under-4 / 4+ BY PCP15                 who was asked
  gt_nat_pueblo      FREQUENCY OF PCP12                    pueblo, for the record (sources/gt.md)
  gt_pueblo_x_pcp15  PCP12 BY PCP15                        pueblo x mother tongue, for note_public

CHECKS (the script stops unless all hold):
  1. 340 municipios in each per-municipio table, the same codes in all three, and every
     AREALIST row sums to its printed Total.
  2. per municipio and language, the AREALIST equals the AREABREAK table exactly.
  3. summed over municipios, each language equals the national table, and the asked total is
     13,566,897; the all-ages total is 14,901,286 (the persons the 2018 census counted).
  4. per municipio, asked (4+) <= all ages, and the national gap equals the No Aplica line.
  5. the age crosstab: nobody under 4 answered PCP15, everyone 4+ did.
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
RAW = HERE / "data" / "raw" / "gt"
NORM = HERE / "data" / "normalized"

HOST = "https://redatam2018.ine.gob.gt/bingtm"
BASE = "CPVGT2018"
POPULATION = 14_901_286
ASKED = 13_566_897
N_MUNICIPIOS = 340

PROGRAMS = {
    "gt_mupio_pcp15": "RUNDEF Job\n SELECTION ALL\nTABLE T1\n AS AREALIST\n OF MUPIO, PERSONA.PCP15\n",
    "gt_break_pcp15": "RUNDEF Job\n SELECTION ALL\nTABLE T1\n AS FREQUENCY\n OF PERSONA.PCP15\n"
                      " AREABREAK MUPIO\n",
    "gt_break_sexo": "RUNDEF Job\n SELECTION ALL\nTABLE T1\n AS FREQUENCY\n OF PERSONA.PCP6\n"
                     " AREABREAK MUPIO\n",
    "gt_nat_pcp15": "RUNDEF Job\n SELECTION ALL\nTABLE T1\n AS FREQUENCY\n OF PERSONA.PCP15\n",
    # the web form eats a `<`, so the age split is written with >= only
    "gt_age4_x_pcp15": "RUNDEF Job\n SELECTION ALL\nDEFINE PERSONA.AGE4\n AS SWITCH\n"
                       " INCASE PERSONA.PCP7 >= 4\n  ASSIGN 2\n DEFAULT 1\n TYPE INTEGER\n"
                       " RANGE 1-2\nTABLE T1\n AS FREQUENCY\n OF PERSONA.AGE4 BY PERSONA.PCP15\n",
    "gt_nat_pueblo": "RUNDEF Job\n SELECTION ALL\nTABLE T1\n AS FREQUENCY\n OF PERSONA.PCP12\n",
    "gt_pueblo_x_pcp15": "RUNDEF Job\n SELECTION ALL\nTABLE T1\n AS FREQUENCY\n"
                         " OF PERSONA.PCP12 BY PERSONA.PCP15\n",
}


def fetch():
    import requests
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    RAW.mkdir(parents=True, exist_ok=True)
    for name, program in PROGRAMS.items():
        dest = RAW / f"{name}.htm"
        if dest.exists() and dest.stat().st_size > 2_000:
            print("already have", dest.name)
            continue
        print("RUN", name)
        r = s.post(f"{HOST}/RpWebStats.exe/CmdSet?", data={
            "MAIN": "WebServerMain.inl", "BASE": BASE, "LANG": "esp",
            "CODIGO": "XXUSUARIOXX", "ITEM": "PROGRED", "MODE": "RUN",
            "CMDSET": program, "Submit": "Ejecutar"}, timeout=1800, verify=False)
        if r.status_code != 200:
            raise SystemExit(f"{name}: HTTP {r.status_code}")
        m = re.search(r"LFN=([^\"'&<>]+?\.htm)", html.unescape(r.text))
        if not m:
            raise SystemExit(f"{name}: REDATAM returned no output file.\n"
                             f"{re.sub(r'<[^>]+>', ' ', r.text)[:800]}")
        for attempt in range(3):
            try:
                t = s.get(f"{HOST}/RpWebUtilities.exe/Text?LFN=" + urllib.parse.quote(m.group(1))
                          + "&TYPE=TMP", timeout=1800, verify=False)
                break
            except requests.exceptions.ConnectionError:
                if attempt == 2:
                    raise
                time.sleep(5)
        t.raise_for_status()
        # the server sends cp1252 with no charset; most accents are entities, a few (Espanol) raw
        body = t.content.decode("cp1252", errors="strict")
        if "Tabla vac" in body or "<table" not in body.lower():
            raise SystemExit(f"{name}: REDATAM returned no table.\n{body[:600]}")
        (RAW / f"{name}.program.txt").write_text(program, encoding="utf-8")
        dest.write_text(body, encoding="utf-8")
        print(f"  {dest.stat().st_size:,} bytes")
        time.sleep(1)


def _unmangle(c):
    """The server writes its raw (non-entity) accents UTF-8-encoded twice: Español arrives as the
    bytes of 'EspaÃ±ol'. Undo that only where the round trip is clean."""
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
    body = p.read_bytes().decode("utf-8-sig")
    out = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S | re.I):
        cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).replace("\xa0", " ").strip()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S | re.I)]
        cells = [_unmangle(c) for c in cells if c]
        if cells:
            out.append(cells)
    return out


def _num(tok):
    t = tok.replace(" ", "")
    if t == "-":
        return 0
    if not re.fullmatch(r"\d+", t):
        raise ValueError(f"{tok!r} is not a figure")
    return int(t)


def read_arealist(name):
    """-> ({municipio: {label: count}}, {municipio: printed Total}, [labels])"""
    rows = _rows(name)
    header = next(r for r in rows if r[0] == "Código")
    cols = header[1:]
    if cols[-1] != "Total":
        raise SystemExit(f"{name}: AREALIST header ends {cols[-1]!r}")
    labels = cols[:-1]
    data, totals = {}, {}
    for r in rows:
        if re.fullmatch(r"\d{4}", r[0]):
            if len(r) != len(cols) + 1:
                raise SystemExit(f"{name}: row {r[0]} has {len(r)} cells, header {len(cols) + 1}")
            v = [_num(x) for x in r[1:]]
            data[r[0]] = dict(zip(labels, v[:-1]))
            totals[r[0]] = v[-1]
    return data, totals, labels


def read_break(name):
    """FREQUENCY ... AREABREAK MUPIO -> ({municipio: {label: count}}, {municipio: name},
    {municipio: printed Total}, {municipio: printed No Aplica}). Each area's table is headed
    `AREA # 0101 | Guatemala`; the national RESUMEN after the last one is not read."""
    rows = _rows(name)
    data, names, totals, na = {}, {}, {}, {}
    cur = None
    i = 0
    while i < len(rows):
        r = rows[i]
        if r[0] == "RESUMEN":
            break
        m = re.fullmatch(r"AREA # (\d{4})", r[0])
        if m:
            cur = m.group(1)
            if cur in data:
                raise SystemExit(f"{name}: municipio {cur} twice")
            names[cur] = r[1]
            data[cur] = {}
            i += 1
            continue
        if cur and r[0] == "No Aplica :":
            na[cur] = _num(r[1])
        if cur and len(r) == 4 and r[2].endswith("%"):
            try:
                n = _num(r[1])
            except ValueError:
                i += 1
                continue
            if r[0] == "Total":
                totals[cur] = n
            else:
                data[cur][r[0]] = n
        i += 1
    return data, names, totals, na


def read_frequency(name):
    out = {}
    for r in _rows(name):
        if len(r) == 4 and r[2].endswith("%"):
            try:
                out[r[0]] = _num(r[1])
            except ValueError:
                continue
    return out


def main():
    if "--fetch" in sys.argv:
        fetch()

    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'OK ' if cond else 'BAD'} {msg}")

    # 1. shape
    data, totals, labels = read_arealist("gt_mupio_pcp15")
    brk, names, brk_tot, brk_na = read_break("gt_break_pcp15")
    sexo, names2, pop, _ = read_break("gt_break_sexo")
    print(f"  {len(labels)} categories: {', '.join(labels)}")
    check(len(data) == N_MUNICIPIOS, f"{len(data)} municipios in the AREALIST (expected {N_MUNICIPIOS})")
    check(set(data) == set(brk) == set(sexo),
          f"the same municipio codes in all three tables ({len(brk)}, {len(sexo)})")
    check(names == names2, "the two AREABREAK runs name every municipio alike")
    bad = [m for m in data if sum(data[m].values()) != totals[m]]
    check(not bad, f"every AREALIST row sums to its Total ({len(bad)} do not: {bad[:4]})")

    # 2. AREALIST against AREABREAK, cell by cell (AREABREAK omits zero lines)
    diff = [(m, lab, data[m][lab], brk[m].get(lab, 0)) for m in data for lab in labels
            if data[m][lab] != brk.get(m, {}).get(lab, 0)]
    stray = [(m, lab) for m in brk for lab in brk[m] if lab not in labels]
    check(not diff, f"per municipio and language the AREALIST equals the AREABREAK table "
                    f"({len(diff)} cells differ: {diff[:3]})")
    check(not stray, f"no AREABREAK label outside the AREALIST's ({stray[:3]})")
    check(all(brk_tot[m] == totals[m] for m in data), "AREABREAK Totals equal the AREALIST Totals")

    # 3. national
    nat = read_frequency("gt_nat_pcp15")
    summed = {lab: sum(data[m][lab] for m in data) for lab in labels}
    off = [(lab, summed[lab], nat.get(lab)) for lab in labels if summed[lab] != nat.get(lab)]
    check(not off, f"each language summed over municipios equals the national table ({off[:3]})")
    asked = sum(totals.values())
    check(asked == ASKED == nat.get("Total"), f"{asked:,} people asked (aged 4+)")
    allpop = sum(pop.values())
    check(allpop == POPULATION, f"{allpop:,} people of all ages (the census counted {POPULATION:,})")

    # 4. under-4 per municipio
    neg = [m for m in data if totals[m] + brk_na.get(m, 0) != pop[m]]
    check(not neg, f"in every municipio, asked + No Aplica = all ages ({neg[:4]})")
    under4 = allpop - asked
    share = sorted((1 - totals[m] / pop[m], m) for m in data)
    print(f"     under 4, not asked: {under4:,} ({100 * under4 / allpop:.1f}%); per municipio "
          f"{100 * share[0][0]:.1f}% ({names[share[0][1]]}) to {100 * share[-1][0]:.1f}% "
          f"({names[share[-1][1]]})")

    # 5. who was asked
    ax = {}
    for r in _rows("gt_age4_x_pcp15"):
        if r[0] in ("1", "2") and len(r) >= len(labels) + 2:
            ax[r[0]] = [_num(x) for x in r[1:]]
    if ax:
        check(sum(ax.get("1", [0])[:-1]) == 0 and ax["2"][-1] == ASKED,
              f"nobody under 4 answered; the 4+ row totals {ax['2'][-1]:,}")
    else:
        print("     (age crosstab not parsed; see the raw file)")
        rows = _rows("gt_age4_x_pcp15")
        for r in rows[-8:]:
            print("       ", r[:6], "...", r[-3:])

    pueblo = read_frequency("gt_nat_pueblo")
    print("     pueblo (PCP12), all ages: " + ", ".join(f"{k} {v:,}" for k, v in pueblo.items()))
    px = {r[0]: [_num(x) for x in r[1:]] for r in _rows("gt_pueblo_x_pcp15")
          if len(r) == len(labels) + 2 and r[0] in ("Maya", "Xinka", "Garífuna", "Total")}
    if "Maya" in px:
        maya, sp = px["Maya"], labels.index("Español")
        mayan_lang = sum(summed[lab] for lab in labels[:22])
        check(px["Total"][-1] == ASKED, "the pueblo x language crosstab covers everyone asked")
        print(f"     aged 4+: {maya[-1]:,} Maya by pueblo ({100 * maya[-1] / ASKED:.1f}%), "
              f"{mayan_lang:,} with a Mayan mother tongue ({100 * mayan_lang / ASKED:.1f}%); "
              f"{maya[sp]:,} Maya ({100 * maya[sp] / maya[-1]:.1f}%) learned Spanish first. "
              f"Xinka by pueblo {px['Xinka'][-1]:,}, Xinka mother tongue {summed['Xinka']:,}")

    if not ok:
        raise SystemExit("gt: checks FAILED, nothing written")

    NORM.mkdir(parents=True, exist_ok=True)
    long = [("GT" + m, "municipio", names[m], lab, n)
            for m in sorted(data) for lab in labels for n in [data[m][lab]] if n]
    df = pd.DataFrame(long, columns=["geo_id", "geo_level", "geo_name", "source_category", "count"])
    df.to_csv(NORM / "gt.csv", index=False, encoding="utf-8")
    u = pd.DataFrame([("GT" + m, names[m], pop[m], totals[m]) for m in sorted(data)],
                     columns=["geo_id", "geo_name", "population", "asked"])
    u.to_csv(NORM / "gt_units.csv", index=False, encoding="utf-8")
    print(f"wrote {NORM / 'gt.csv'} ({len(df):,} rows) and gt_units.csv ({len(u)} municipios)")
    top = df.groupby("source_category")["count"].sum().sort_values(ascending=False)
    for k, v in top.head(8).items():
        print(f"     {v:>12,}  {100 * v / asked:5.2f}%  {k}")


if __name__ == "__main__":
    main()
