"""Colombia, Censo Nacional de Poblacion y Vivienda 2018: speakers of the native language of
their people, per municipio, split into resguardo, cabecera and the rest.

    python sources/co_cnpv.py [--fetch]

Writes
  data/normalized/co.csv     one row per (municipio, zone, code): `code` is the census's pueblo
                             code (10-999) for indigenous speakers, or one of the codes below;
                             `zone` r = the dwelling is in a resguardo indigena
                             (VIVIENDA.UVA_ESTATER = 1 and UVA1_TIPOTER = 1), u = otherwise in
                             the cabecera (Clase 1), x = otherwise (centro poblado, rural disperso).
                             Every person counted in 2018 is in exactly one row.
  data/normalized/co_pueblos.csv   the pueblo codes and DANE's labels, national figures

THE QUESTION (CNPV 2018, persons). Only people who self-identify with an ethnic group are asked:
PA1_GRP_ETNIC 1 indigenous (then PA11_COD_ETNIA, the pueblo, about 115 codes), 2 Gitano/Rrom,
3 Raizal of San Andres, 4 Palenquero of San Basilio. Then PA_HABLA_LENG "Habla la lengua nativa
de su pueblo" (1 yes, 2 no, 9 no informa) and, for everyone asked, PB_OTRAS_LENG "Habla otra(s)
lengua(s) nativa(s)" (yes/no, with a count, never the names). Nobody else is asked about language.
So the language is IMPLIED by the pueblo, and it is ability ("speaks"), not mother tongue.

THE SOURCE is DANE's own REDATAM webserver over the full census microdata (unweighted; it is a
full count), base CNPVBASE4V2 at systema59.dane.gov.co/bincol. Redatam+SP programs define a
category per person and tabulate it per municipio (AREALIST OF MUPIO), fifteen tables in all
(three zones x five windows of pueblo codes, see _program); the national crosstabs
of pueblo x speaks and group x speaks come from the same server by a second engine path
(AS FREQUENCY ... BY ...), and they are what the municipal table is checked against.

Category codes in co.csv besides the pueblo codes:
  0     everyone else: not asked, or asked and not a speaker of any native language
  1001  Rrom who speak their people's language (Romani)
  1002  Raizales who speak it (San Andres Creole)
  1003  Palenqueros who speak it (Palenquero)
  1010  asked, do NOT speak their own people's language, DO speak another native language
        (unnamed by the census)
  1011  asked, no answer on PA_HABLA_LENG

CHECKS (the script stops unless all hold):
  1. 1,122 municipios, every row's categories sum to its Total in every table, the Totals
     sum to 44,164,417 (the persons the 2018 census counted), and across the fifteen tables
     every person is counted exactly once.
  2. per pueblo, speakers summed over municipios and zones equal the national crosstab
     (pueblo x PA_HABLA_LENG) exactly, for every pueblo code.
  3. per group, Rrom / Raizal / Palenquero speakers equal the national crosstab (group x
     PA_HABLA_LENG), and indigenous speakers' sum equals its indigenous cell (838,356).
  4. 1011 equals the national "No informa" (15,354) and 1010 the speaks-no x other-yes cell
     (17,988) of the PA_HABLA_LENG x PB_OTRAS_LENG crosstab.
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
RAW = HERE / "data" / "raw" / "co"
NORM = HERE / "data" / "normalized"

HOST = "https://systema59.dane.gov.co/bincol"
BASE = "CNPVBASE4V2"
POPULATION = 44_164_417
N_MUNICIPIOS = 1122

_TER = "VIVIENDA.UVA_ESTATER = 1 AND VIVIENDA.UVA1_TIPOTER = 1"
# The server refuses a variable declared with more than about 250 categories ("Too many
# categories"; 1-250 runs, 1-499 does not), and pueblo codes run 10-999. So the municipal table
# is ten tables: each side of the resguardo line x five windows of 200 pueblo codes. In each,
#   1-200    pueblo code 200*w + value - 1, speakers of their people's language
#   201-206  the groups and remainders (window 0 only, so nobody is counted twice)
#   207      everyone this table does not count (other side, other window, other category)
# Every table's row sums to the municipio's whole population; the script checks that.
_IND = "PERSONA.PA_HABLA_LENG = 1 AND PERSONA.PA1_GRP_ETNIC = 1"
_SPECIAL = [
    ("PERSONA.PA_HABLA_LENG = 1 AND PERSONA.PA1_GRP_ETNIC = 2", 202, 1001),
    ("PERSONA.PA_HABLA_LENG = 1 AND PERSONA.PA1_GRP_ETNIC = 3", 203, 1002),
    ("PERSONA.PA_HABLA_LENG = 1 AND PERSONA.PA1_GRP_ETNIC = 4", 204, 1003),
    ("PERSONA.PA_HABLA_LENG = 2 AND PERSONA.PB_OTRAS_LENG = 1", 205, 1010),
    ("PERSONA.PA_HABLA_LENG = 9", 206, 1011),
]
EVERYONE_ELSE = 201      # -> code 0, window 0 only
NOT_HERE = 207
WINDOWS = range(5)


ZONES = {
    # zone: (conditions that send a person to another zone's table, condition every case needs)
    "r": ([], f" AND {_TER}"),                                   # in a resguardo
    "u": ([_TER, "Clase.UA_CLASE1 >= 2"], ""),                   # cabecera, outside resguardos
    "x": ([_TER, "Clase.UA_CLASE1 = 1"], ""),                    # centros poblados, rural disperso
}


def _program(zone, w):
    """This build's SWITCH has no NOT (syntax error), DEFAULT takes a constant only, and the
    web form eats a `<` (it reads as a tag), so every condition is written with `=` and `>=`
    and the INCASEs are ordered: the first that holds wins."""
    lo, hi = 200 * w, 200 * w + 199
    here = EVERYONE_ELSE if w == 0 else NOT_HERE
    away, t = ZONES[zone]
    inside = zone == "r"
    lines = ["RUNDEF Job", " SELECTION ALL", "DEFINE PERSONA.LCAT", " AS SWITCH"]
    for cond in away:
        lines += [f" INCASE {cond}", f"  ASSIGN {NOT_HERE}"]
    lines += [f" INCASE {_IND} AND PERSONA.PA11_COD_ETNIA >= {hi + 1}{t}", f"  ASSIGN {NOT_HERE}",
              f" INCASE {_IND} AND PERSONA.PA11_COD_ETNIA >= {lo}{t}",
              ("  ASSIGN PERSONA.PA11_COD_ETNIA + 1" if lo == 0 else
               f"  ASSIGN PERSONA.PA11_COD_ETNIA - {lo - 1}"),
              f" INCASE {_IND}{t}", f"  ASSIGN {NOT_HERE}"]
    for cond, val, _ in _SPECIAL:
        lines += [f" INCASE {cond}{t}", f"  ASSIGN {val if w == 0 else NOT_HERE}"]
    if inside:
        lines += [f" INCASE {_TER}", f"  ASSIGN {here}", f" DEFAULT {NOT_HERE}"]
    else:
        lines += [f" DEFAULT {here}"]
    lines += [" TYPE INTEGER", f" RANGE 1-{NOT_HERE}",
              "TABLE T1", " AS AREALIST", " OF MUPIO, PERSONA.LCAT", ""]
    return "\n".join(lines)


def _unpack(w, v):
    if v <= 200:
        return 200 * w + v - 1
    if v == EVERYONE_ELSE:
        return 0
    return {val: code for _, val, code in _SPECIAL}[v]


TABLES = [(f"co_mupio_{zone}_w{w}", zone, w) for zone in ZONES for w in WINDOWS]

PROGRAMS = {
    **{name: _program(zone, w) for name, zone, w in TABLES},
    "co_pueblo_x_habla": "RUNDEF Job\n SELECTION ALL\nTABLE T1\n AS FREQUENCY\n"
                         " OF PERSONA.PA11_COD_ETNIA BY PERSONA.PA_HABLA_LENG\n",
    "co_grupo_x_habla": "RUNDEF Job\n SELECTION ALL\nTABLE T1\n AS FREQUENCY\n"
                        " OF PERSONA.PA1_GRP_ETNIC BY PERSONA.PA_HABLA_LENG\n",
    "co_habla_x_otras": "RUNDEF Job\n SELECTION ALL\nTABLE T1\n AS FREQUENCY\n"
                        " OF PERSONA.PA_HABLA_LENG BY PERSONA.PB_OTRAS_LENG\n",
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
            raise SystemExit(f"{name}: HTTP {r.status_code}\n"
                             f"{re.sub(r'<[^>]+>', ' ', r.text)[:800]}")
        m = re.search(r"LFN=([^\"'&<>]+?\.htm)", html.unescape(r.text))
        if not m:
            raise SystemExit(f"{name}: REDATAM returned no output file.\n"
                             f"{re.sub(r'<[^>]+>', ' ', r.text)[:800]}")
        for attempt in range(3):   # the server drops a connection now and then
            try:
                t = s.get(f"{HOST}/RpWebUtilities.exe/Text?LFN=" + urllib.parse.quote(m.group(1))
                          + "&TYPE=TMP", timeout=1800, verify=False)
                break
            except requests.exceptions.ConnectionError:
                if attempt == 2:
                    raise
                time.sleep(5)
        t.raise_for_status()
        if "Tabla vac" in t.text or "<table" not in t.text.lower():
            raise SystemExit(f"{name}: REDATAM returned no table.\n{t.text[:600]}")
        (RAW / f"{name}.program.txt").write_text(program, encoding="utf-8")
        dest.write_text(t.text, encoding="utf-8")
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
    t = tok.replace(" ", "")
    if t == "-":
        return 0
    if not re.fullmatch(r"\d+", t):
        raise SystemExit(f"{tok!r} is not a figure")
    return int(t)


def read_crosstab(name, n_cols):
    """FREQUENCY ... BY ...: {row label: [cols..., total]}."""
    out = {}
    for r in _rows(name):
        if len(r) == n_cols + 2 and r[0] not in ("Si",):
            try:
                out[r[0]] = [_num(x) for x in r[1:]]
            except SystemExit:
                continue
    return out


def read_arealist(name):
    """-> {municipio: {value: count}}, and each row's printed Total."""
    rows = _rows(name)
    header = next(r for r in rows if r[0] == "Código")
    cols = header[1:]
    if cols[-1] != "Total":
        raise SystemExit(f"{name}: AREALIST header ends {cols[-1]!r}")
    vals = [int(c) for c in cols[:-1]]
    data, totals = {}, {}
    for r in rows:
        if re.fullmatch(r"\d{5}", r[0]) and len(r) == len(cols) + 1:
            v = [_num(x) for x in r[1:]]
            data[r[0]] = dict(zip(vals, v[:-1]))
            totals[r[0]] = v[-1]
    return data, totals


def main():
    if "--fetch" in sys.argv:
        fetch()

    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'OK ' if cond else 'BAD'} {msg}")

    # 1. shape and totals: every one of the fifteen tables counts every person once
    long, pop = [], None
    for name, zone, w in TABLES:
        data, totals = read_arealist(name)
        bad = [m for m in data if sum(data[m].values()) != totals[m]]
        if pop is None:
            pop = totals
            check(len(pop) == N_MUNICIPIOS, f"{len(pop):,} municipios (expected {N_MUNICIPIOS:,})")
            check(sum(pop.values()) == POPULATION,
                  f"Totals sum to {sum(pop.values()):,} (the census counted {POPULATION:,})")
        same = totals == pop
        if bad or not same:
            check(False, f"{name}: {len(bad)} rows do not sum to their Total; "
                         f"totals equal the first table's: {same}")
        for m, d in data.items():
            for v, n in d.items():
                if n and v != NOT_HERE:
                    long.append((m, zone, _unpack(w, v), n))
    df = pd.DataFrame(long, columns=["mupio", "zone", "code", "count"])
    per = df.groupby("mupio")["count"].sum()
    bad = [m for m in pop if per.get(m, 0) != pop[m]]
    check(not bad, f"across the fifteen tables every person is counted exactly once "
                   f"({len(bad)} municipios off: {bad[:4]})")
    by_code = df.groupby("code")["count"].sum()

    # 2. per pueblo, against the national crosstab
    px = read_crosstab("co_pueblo_x_habla", 3)
    pueblos = {}
    for label, v in px.items():
        m = re.match(r"(\d{3})_(.*)", label)
        if m:
            pueblos[int(m.group(1))] = (label, v)
    mism = [(c, lab, v[0], int(by_code.get(c, 0))) for c, (lab, v) in pueblos.items()
            if v[0] != int(by_code.get(c, 0))]
    stray = sorted(set(c for c in by_code.index if 10 <= c <= 999) - set(pueblos))
    check(len(pueblos) >= 100, f"{len(pueblos)} pueblo codes in the national crosstab")
    check(not mism, f"per pueblo, municipal speakers equal the national crosstab "
                    f"({len(mism)} differ: {mism[:4]})")
    check(not stray, f"no pueblo code in the municipal table missing nationally ({stray[:5]})")

    # 3. per group
    gx = read_crosstab("co_grupo_x_habla", 3)
    want = {1001: "Gitano(a) o Rrom",
            1002: "Raizal del Archipielago de San Andrés, Providencia y Santa Catalina",
            1003: "Palenquero(a) de San Basilio"}
    for c, lab in want.items():
        check(lab in gx and gx[lab][0] == int(by_code.get(c, 0)),
              f"{lab[:30]}: {int(by_code.get(c, 0)):,} speakers, crosstab "
              f"{gx.get(lab, [None])[0]}")
    ind = int(by_code[(by_code.index >= 10) & (by_code.index <= 999)].sum())
    check(gx.get("Indígena", [None])[0] == ind,
          f"indigenous speakers {ind:,} against the crosstab's {gx.get('Indígena', [None])[0]}")

    # 4. the unnamed and the unanswered
    ox = read_crosstab("co_habla_x_otras", 3)
    check(ox.get("No", [None, None, None, None])[0] == int(by_code.get(1010, 0)),
          f"speaks another native language only: {int(by_code.get(1010, 0)):,} "
          f"(crosstab {ox.get('No', [None])[0]})")
    check(ox.get("No informa", [0, 0, 0, None])[3] == int(by_code.get(1011, 0)),
          f"no answer: {int(by_code.get(1011, 0)):,} (crosstab {ox.get('No informa', [0]*4)[3]})")

    speakers = df[df["code"].between(1, 1003)]
    share = speakers.groupby("zone")["count"].sum() / speakers["count"].sum()
    print(f"  speakers {speakers['count'].sum():,}: {100 * share.get('r', 0):.1f}% in a resguardo, "
          f"{100 * share.get('u', 0):.1f}% in a cabecera outside one, "
          f"{100 * share.get('x', 0):.1f}% elsewhere")
    if not ok:
        raise SystemExit("co: checks FAILED, nothing written")

    NORM.mkdir(parents=True, exist_ok=True)
    df["geo_id"] = "CO" + df["mupio"]
    lab = {c: lab for c, (lab, _) in pueblos.items()}
    lab.update({0: "everyone else", 1001: "Rrom: Romani", 1002: "Raizal: San Andres Creole",
                1003: "Palenquero: Palenquero", 1010: "speaks another native language, unnamed",
                1011: "no answer"})
    df["source_category"] = df["code"].map(lab)
    if df["source_category"].isna().any():
        raise SystemExit(f"codes with no label: {sorted(df.loc[df['source_category'].isna(), 'code'].unique())}")
    df[["geo_id", "zone", "code", "source_category", "count"]].sort_values(
        ["geo_id", "zone", "code"]).to_csv(NORM / "co.csv", index=False)
    pd.DataFrame([(c, l, v[0], v[1], v[2], v[3]) for c, (l, v) in sorted(pueblos.items())],
                 columns=["code", "label", "speak_yes", "speak_no", "no_answer", "total"]
                 ).to_csv(NORM / "co_pueblos.csv", index=False)
    print(f"  wrote data/normalized/co.csv ({len(df):,} rows) and co_pueblos.csv")


if __name__ == "__main__":
    main()
