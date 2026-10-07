"""Nicaragua, VIII Censo de Poblacion y IV de Vivienda 2005 (INIDE): people who speak the
language of their indigenous people or ethnic community, per municipio.

    python sources/ni_censo.py [--fetch]

Writes data/normalized/ni.csv: one row per (municipio, code). Every person counted in 2005
(5,142,098) is in exactly one row. `code` is the derived variable LNG defined in the program
below:

    1          self-identifies with a pueblo the language question was NOT asked of
               (P07 8-13 or 99: Xiu-Sutiaba, Nahoa-Nicarao, Chorotega-Nahua-Mange,
               Cacaopera-Matagalpa, Otro, No sabe, Ignorado)
    2          does not self-identify with a pueblo (P06 = 2)
    3          P06 not declared
    10*p + a   pueblo p (P07 1-7) and answer a to P08 (1 speaks the language of their people,
               2 does not, 9 no answer)

THE QUESTIONS, asked of everyone, all ages. P06 "Se considera perteneciente a un pueblo
indigena o etnia?"; if yes, P07 which one (thirteen listed, plus ignorado); and, ONLY for the
seven Caribbean-coast peoples (P07 1-7: Rama, Garifuna, Mayangna-Sumu, Miskitu, Ulwa,
Creole (Kriol), Mestizo de la costa caribe), P08 "Habla la lengua o idioma del pueblo indigena o
comunidad etnica a la que pertenece?". The Pacific and central peoples, whose own languages
went out of use generations ago, were not asked P08 at all (the national crosstab shows it:
all 172,977 of them are `No Aplica` on P08). So the census names a people, the language is
implied by it, and nobody else is asked about language.

THE SOURCE is INIDE's own REDATAM webserver (redatam.inide.gob.ni, base VIVPOB05, open, no
login), the same route religiondots/sources/ni.py uses for religion. One program defines LNG
and tabulates it per municipio (AREALIST OF MUN05) and per department; plain dictionary
variables P06 and P08 are tabulated per municipio separately, and the national P07 x P08
crosstab comes from the same server.

CHECKS (the script stops unless all hold):
  1. 153 municipios; every person in exactly one code; codes sum to 5,142,098, the 2005
     census population.
  2. the 153 municipios rebuild the 17 departments on every code.
  3. per municipio, LNG agrees with separate tabulations of the raw variables: code 2 = P06
     "No", code 3 = P06 "No declarado", code 1 + all 1x-7x = P06 "Si"; the x1 / x2 / x9 codes
     summed = P08 "Si" / "No" / "NR". Two independent tabulations of the same microdata.
  4. per pueblo, national LNG totals equal the national P07 x P08 crosstab.
  5. the printed volume (Vol. I, 2006): all 14 pueblo totals of CUADRO 10 and 24 speaker
     figures of CUADRO 11 (national, R.A.A.N., R.A.A.S., Managua) are reproduced exactly.
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
RAW = HERE / "data" / "raw" / "ni"
NORM = HERE / "data" / "normalized"

HOST = "http://redatam.inide.gob.ni/redbin"
BASE = "VIVPOB05"

LNG = """DEFINE PERS05.LNG
 AS SWITCH
  INCASE PERS05.P08 = 1 OR PERS05.P08 = 2 OR PERS05.P08 = 9
   ASSIGN PERS05.P07 * 10 + PERS05.P08
  INCASE PERS05.P06 = 1
   ASSIGN 1
  INCASE PERS05.P06 = 2
   ASSIGN 2
  DEFAULT 3
 TYPE INTEGER
 RANGE 0-99
"""

PROGRAMS = {
    "mun_lng": "RUNDEF Job\n SELECTION ALL\n" + LNG + "TABLE T\n AS AREALIST\n OF MUN05, PERS05.LNG\n",
    "dep_lng": "RUNDEF Job\n SELECTION ALL\n" + LNG + "TABLE T\n AS AREALIST\n OF DEP05, PERS05.LNG\n",
    "mun_p06": "RUNDEF Job\n SELECTION ALL\nTABLE T\n AS AREALIST\n OF MUN05, PERS05.P06\n",
    "mun_p08": "RUNDEF Job\n SELECTION ALL\nTABLE T\n AS AREALIST\n OF MUN05, PERS05.P08\n",
    "nat_p07_p08": "RUNDEF Job\n SELECTION ALL\nTABLE T\n AS CROSSTABS\n OF PERS05.P07 BY PERS05.P08\n",
    "nat_p07": "RUNDEF Job\n SELECTION ALL\nTABLE T\n AS FREQUENCY\n OF PERS05.P07\n",
    "mun_names": "RUNDEF Job\n SELECTION ALL\nTABLE T\n AS FREQUENCY\n OF MUN05.CMuni\n",
}

# P07's labels, as REDATAM prints them, codes 1-13 and 99.
PUEBLOS = {1: "Rama", 2: "Garífuna", 3: "Mayagna-Sumu", 4: "Miskitu", 5: "Ulwa",
           6: "Creole(Kriol)", 7: "Mestizo de la costa del caribe", 8: "Xiu-Sutiaba",
           9: "Nahoas-Nicarao", 10: "Chorotega-Nahua-Mange", 11: "Cacaopera-Matagalpa",
           12: "Otro", 13: "No sabe", 99: "Ignorado"}
ASKED = range(1, 8)                       # P08 was asked of these pueblos only
ANSWERS = {1: "speaks", 2: "does not speak", 9: "no answer"}
OTHER_CODES = {1: "indigenous, pueblo not asked the language question",
               2: "not indigenous", 3: "P06 not declared"}

# THE PRINTED WITNESS: INIDE, Censo 2005, Vol. I "Poblacion: Caracteristicas Generales" (2006,
# data/raw/ni/volI.pdf). Typed from the PDF, not from the query, and sharing no code path with
# it. CUADRO 10 (printed p184, PDF page 181), LA REPUBLICA, self-identified pueblo, all ages:
PUEBLOS_2005 = {1: 4_185, 2: 3_271, 3: 9_756, 4: 120_817, 5: 698, 6: 19_890, 7: 112_253,
                8: 19_949, 9: 11_113, 10: 46_002, 11: 15_240, 12: 13_740, 13: 47_473,
                99: 19_460}
# CUADRO 11 (printed p189 on, PDF pages 186-191), speakers of their people's language, by
# department: (department code, pueblo) -> printed figure; pueblo 0 is the department's total.
# 00 is LA REPUBLICA; 55 Managua, 91 R.A.A.N., 93 R.A.A.S.
SPEAKERS_PRINTED = {
    ("00", 0): 244_305, ("00", 1): 744, ("00", 3): 8_537, ("00", 4): 113_855,
    ("00", 6): 18_420,
    ("91", 0): 165_669, ("91", 1): 123, ("91", 2): 26, ("91", 3): 6_488, ("91", 4): 97_851,
    ("91", 5): 33, ("91", 6): 1_585, ("91", 7): 59_563,
    ("93", 0): 65_069, ("93", 1): 427, ("93", 2): 417,
    ("55", 0): 2_843, ("55", 1): 75, ("55", 2): 31, ("55", 3): 40, ("55", 4): 1_357,
    ("55", 5): 2, ("55", 6): 683, ("55", 7): 655,
}
CENSUS_POPULATION = 5_142_098
N_MUN, N_DEP = 153, 17


def _post(program, timeout=900):
    import requests

    r = requests.post(
        HOST + "/RpWebStats.exe/CmdSet?",
        data={"MAIN": "WebServerMain.inl", "BASE": BASE, "LANG": "esp",
              "CODIGO": "XXUSUARIOXX", "ITEM": "PROGRED", "MODE": "RUN",
              "CMDSET": program, "Submit": "Ejecutar"},
        headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    r.raise_for_status()
    tmps = sorted(set(re.findall(r"(RpBases[^\"'&<>]*?\.htm)", r.text)))
    if not tmps:
        raise SystemExit("REDATAM returned no output file. Body starts:\n" + r.text[:600])
    url = HOST + "/RpWebUtilities.exe/Text?LFN=" + urllib.parse.quote(tmps[0]) + "&TYPE=TMP"
    t = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    t.raise_for_status()
    return t.text


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, program in PROGRAMS.items():
        dest = RAW / f"ni_{name}.htm"
        if dest.exists() and dest.stat().st_size > 1_000:
            print("already have", dest.name)
            continue
        print("RUN", name)
        body = _post(program)
        # a bad program answers `Tabla vacia` with HTTP 200 (religiondots/sources/ni.md)
        if "Tabla vac" in body or "<table" not in body.lower():
            raise SystemExit(f"{name}: REDATAM returned no table\n{body[:600]}")
        dest.write_text(body, encoding="utf-8")
        (RAW / f"ni_{name}.txt").write_text(program, encoding="utf-8")
        print(f"  {dest.stat().st_size:,} bytes")
        time.sleep(1)


def _rows(name):
    path = RAW / f"ni_{name}.htm"
    if not path.exists():
        raise SystemExit(f"missing {path} -- run with --fetch first")
    body = path.read_text(encoding="utf-8")
    out = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S | re.I):
        cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).replace("\xa0", " ").strip()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S | re.I)]
        cells = [c for c in cells if c]
        if cells:
            out.append(cells)
    return out


def _int(tok):
    t = tok.replace(" ", "").replace(",", "")
    if not re.fullmatch(r"\d+", t):
        raise SystemExit(f"{tok!r} is not a figure")
    return int(t)


def _areal(name, expected):
    """AREALIST -> {area code: {column label: count}}, `Total` asserted against the row."""
    rows = _rows(name)
    header = next((r for r in rows if r[0] == "Código"), None)
    if header is None:
        raise SystemExit(f"{name}: no header row starting `Código`")
    cols = header[1:]
    if cols[-1] != "Total":
        raise SystemExit(f"{name}: last column is {cols[-1]!r}, not Total")
    out = {}
    for r in rows:
        if len(r) != len(cols) + 1 or not re.fullmatch(r"\d+", r[0].replace(" ", "")):
            continue
        vals = dict(zip(cols, (_int(x) for x in r[1:])))
        tot = vals.pop("Total")
        if sum(vals.values()) != tot:
            raise SystemExit(f"{name}: area {r[0]} categories sum to {sum(vals.values())}, "
                             f"Total {tot}")
        if r[0] in out:
            raise SystemExit(f"{name}: area {r[0]} twice")
        out[r[0].strip()] = vals
    if len(out) != expected:
        raise SystemExit(f"{name}: {len(out)} areas, expected {expected}")
    return out, cols[:-1]


def _names():
    out = {}
    for r in _rows("mun_names"):
        m = re.fullmatch(r"(\d+)\s*-\s*(.+)", r[0].strip())
        if m:
            out[m.group(1)] = m.group(2).strip()
    if len(out) != N_MUN:
        raise SystemExit(f"mun_names: {len(out)} names, expected {N_MUN}")
    return out


def _lng_codes(cols):
    codes = [int(c) for c in cols]
    allowed = set(OTHER_CODES) | {10 * p + a for p in ASKED for a in ANSWERS}
    bad = sorted(set(codes) - allowed)
    if bad:
        raise SystemExit(f"LNG codes outside the plan: {bad}")
    return codes


def main():
    if "--fetch" in sys.argv:
        fetch()
    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    mun, cols = _areal("mun_lng", N_MUN)
    _lng_codes(cols)
    dep, dcols = _areal("dep_lng", N_DEP)
    _lng_codes(dcols)
    names = _names()
    say(set(mun) <= set(names), f"{N_MUN} municipios, all named by REDATAM's own CMuni")

    # 1. every person once
    tot = sum(sum(v.values()) for v in mun.values())
    say(tot == CENSUS_POPULATION, f"codes sum to {tot:,} (census population "
        f"{CENSUS_POPULATION:,})")

    # 2. municipios rebuild departments
    rolled = {}
    for code, v in mun.items():
        acc = rolled.setdefault(code[:2], {})
        for k, n in v.items():
            acc[k] = acc.get(k, 0) + n
    bad = [(d, k) for d in dep for k in set(dep[d]) | set(rolled.get(d, {}))
           if dep[d].get(k, 0) != rolled.get(d, {}).get(k, 0)]
    say(set(rolled) == set(dep) and not bad,
        f"the {N_MUN} municipios rebuild the {N_DEP} departments on every code "
        f"({len(bad)} failures)")

    # 3. against the raw variables, per municipio
    p06, p06c = _areal("mun_p06", N_MUN)
    p08, p08c = _areal("mun_p08", N_MUN)
    print(f"      P06 columns {p06c}; P08 columns {p08c}")
    want06 = {"Si": None, "No": None, "No declarado": None}
    if set(p06c) != set(want06):
        raise SystemExit(f"P06 columns changed: {p06c}")
    lab08 = {1: p08c[0], 2: p08c[1], 9: p08c[2]}
    bad = []
    for m, v in mun.items():
        g = lambda k: v.get(str(k), 0)  # noqa: E731
        si = g(1) + sum(g(10 * p + a) for p in ASKED for a in ANSWERS)
        if (p06[m]["Si"], p06[m]["No"], p06[m]["No declarado"]) != (si, g(2), g(3)):
            bad.append((m, "P06"))
        for a, lab in lab08.items():
            if p08[m].get(lab, 0) != sum(g(10 * p + a) for p in ASKED):
                bad.append((m, "P08", lab))
    say(not bad, f"LNG agrees with separate P06 and P08 tabulations on all {N_MUN} "
        f"municipios ({len(bad)} failures) {bad[:4]}")

    # 4. national crosstab and the published pueblo totals
    nat = {}
    for v in mun.values():
        for k, n in v.items():
            nat[int(k)] = nat.get(int(k), 0) + n
    xt = {}
    for r in _rows("nat_p07_p08"):
        lab = r[0]
        p = next((c for c, l in PUEBLOS.items() if l == lab), None)
        if p is not None and len(r) == 5:
            xt[p] = [_int(x) for x in r[1:]]
    bad = [p for p in ASKED
           if xt.get(p) != [nat.get(10 * p + 1, 0), nat.get(10 * p + 2, 0),
                            nat.get(10 * p + 9, 0),
                            sum(nat.get(10 * p + a, 0) for a in ANSWERS)]]
    say(not bad and len(xt) == 7, f"per pueblo, municipio sums equal the national P07 x P08 "
        f"crosstab, speaks / does not / NR, 7 pueblos ({bad})")
    freq = {}
    for r in _rows("nat_p07"):
        p = next((c for c, l in PUEBLOS.items() if l == r[0]), None)
        if p is not None:
            freq[p] = _int(r[1])
    bad = [p for p in PUEBLOS_2005 if freq.get(p) != PUEBLOS_2005[p]]
    say(not bad, f"the 14 pueblo totals equal CUADRO 10 as printed in 2006 ({bad})")
    bad = []
    for (d, p), want in SPEAKERS_PRINTED.items():
        src = [v for k, v in dep.items() if d == "00" or k == d]
        ps = ASKED if p == 0 else [p]
        got = sum(v.get(str(10 * q + 1), 0) for v in src for q in ps)
        if got != want:
            bad.append((d, p, got, want))
    say(not bad, f"{len(SPEAKERS_PRINTED)} speaker figures equal CUADRO 11 as printed in "
        f"2006 ({bad})")
    not_asked = sum(PUEBLOS_2005[p] for p in PUEBLOS_2005 if p not in ASKED)
    say(nat.get(1) == not_asked, f"code 1 = the {not_asked:,} people of pueblos 8-13 and 99")

    if not ok:
        raise SystemExit("reconciliation FAILED")

    rows = []
    for m in sorted(mun):
        for k, n in sorted(mun[m].items(), key=lambda kv: int(kv[0])):
            k = int(k)
            if k in OTHER_CODES:
                cat = OTHER_CODES[k]
            else:
                cat = f"{PUEBLOS[k // 10]}: {ANSWERS[k % 10]}"
            if n:
                rows.append(dict(geo_level="municipio", geo_id=m, geo_name=names[m],
                                 code=k, source_category=cat, count=n))
    df = pd.DataFrame(rows)
    NORM.mkdir(parents=True, exist_ok=True)
    df.to_csv(NORM / "ni.csv", index=False)
    print(f"\nwrote {NORM / 'ni.csv'} ({len(df):,} rows)\n\nnational:")
    for k in sorted(nat):
        cat = OTHER_CODES.get(k) or f"{PUEBLOS[k // 10]}: {ANSWERS[k % 10]}"
        print(f"  {k:>3} {nat[k]:>10,}  {cat}")


if __name__ == "__main__":
    main()
