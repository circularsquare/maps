"""Venezuela, XIV Censo Nacional de Poblacion y Vivienda 2011: people of each indigenous pueblo,
per parroquia. NO LANGUAGE: see below and sources/ve.md.

    python sources/ve_censo.py [--fetch]

Writes
  data/normalized/ve.csv          one row per (parroquia, code): `code` is the census's pueblo code
                                  (PERSONA.CUALINDIGE, 0-999, about 80 used: INE's 52 pueblos with
                                  their spelling variants, "No declarado" and "Otro") for people
                                  who said they belong to an indigenous pueblo, or one of
                                    1001  born in Venezuela, not indigenous
                                    1002  born abroad (the pueblo question was asked only of
                                          people born in Venezuela)
                                  Every person counted in 2011 (27,227,930) is in exactly one row.
  data/normalized/ve_pueblos.csv  the codes, INE's labels, national counts, and the row of INE's
                                  published pueblo x entidad table each code falls in

WHAT THE CENSUS ASKED. Everyone born in Venezuela: "Pertenece a algun pueblo indigena o etnia?"
(INDIGENA) and which (CUALINDIGE). Indigenous people were then asked "Que idioma(s) habla?",
whether they speak their pueblo's language, Spanish, another language, and whether they read
and write the indigenous one (INE, Boletin Poblacion Indigena, Oct 2013, p. 1). INE published
the language answers only as national shares (10.2% speak only their pueblo's language, 54.1%
both, 33.1% Spanish only; POBLACION-INDIGENA-CENSO-2011.pdf p. 33) and as one speaker share per
state for eight states (p. 34). The public REDATAM base CPV2011 does not carry the language
variables; the base CPV2011SEGMENTO on the evaluation deployment (/vencgibin2/) lists them in its
dictionary (IDIOMAMULT, IDIOMAINDI, IDIOMACAST, OTROIDIOMA, CUALOTROID, ALFABETIIN) but its data
files are missing on the server ("G:\\redatam_eval\\...\\Ve110001.ptr" not found, 2026-10-04).
So this file is the pueblo, which is identity, not language. How (and whether) to turn it into a
language map is ask 007.

THE SOURCE is INE's REDATAM webserver over the full 2011 census person file, base CPV2011 at
redatam.ine.gob.ve/vencgibin (the www. host name no longer resolves; the bare host does).
Five programs, one per window of 200 pueblo codes (the server refuses a variable with more than
about 250 categories, as DANE's did for Colombia), each an AREALIST over every parroquia.

CHECKS (the script stops unless all hold):
  1. every window's rows sum to their printed Total, the windows' Totals agree, the Totals sum to
     27,227,930 (the 2011 count), and across the windows every person is counted exactly once.
  2. per pueblo code, the parroquias sum to the national frequency of CUALINDIGE (a second
     engine path); the frequency prints labels without codes, so codes and labels are paired by
     order, and the paired counts must agree for every code.
  3. per state and published pueblo, the parroquias sum to INE's own table "22.-Pueblos por
     entidad" (Tabulados_Poblacion_Indigena.xls, 2014), cell for cell (52 rows x 25 states).
  4. per state, indigenous people sum to INE's Cuadro 4 (born in Venezuela, by declaration).
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
RAW = HERE / "data" / "raw" / "ve"
NORM = HERE / "data" / "normalized"

HOST = "http://redatam.ine.gob.ve/vencgibin"
BASE = "CPV2011"
POPULATION = 27_227_930
INDIGENOUS = 724_592

NOT_INDIGENOUS = 201     # -> 1001, window 0 only
BORN_ABROAD = 202        # -> 1002, window 0 only
NOT_HERE = 207
WINDOWS = range(5)


def _program(w):
    lo, hi = 200 * w, 200 * w + 199
    ind = "PERSONA.INDIGENA = 1"
    lines = ["RUNDEF Job", " SELECTION ALL", "DEFINE PERSONA.LCAT", " AS SWITCH",
             f" INCASE {ind} AND PERSONA.CUALINDIGE >= {hi + 1}", f"  ASSIGN {NOT_HERE}",
             f" INCASE {ind} AND PERSONA.CUALINDIGE >= {lo}",
             ("  ASSIGN PERSONA.CUALINDIGE + 1" if lo == 0 else
              f"  ASSIGN PERSONA.CUALINDIGE - {lo - 1}"),
             f" INCASE {ind}", f"  ASSIGN {NOT_HERE}",
             " INCASE PERSONA.INDIGENA = 2", f"  ASSIGN {NOT_INDIGENOUS if w == 0 else NOT_HERE}",
             f" DEFAULT {BORN_ABROAD if w == 0 else NOT_HERE}",
             " TYPE INTEGER", f" RANGE 1-{NOT_HERE}",
             "TABLE T1", " AS AREALIST", " OF PARROQUI, PERSONA.LCAT", ""]
    return "\n".join(lines)


TABLES = [(f"ve_parroquia_w{w}", w) for w in WINDOWS]
PROGRAMS = {
    **{name: _program(w) for name, w in TABLES},
    "ve_cualindige": "RUNDEF Job\n SELECTION ALL\nTABLE T1\n AS FREQUENCY\n OF PERSONA.CUALINDIGE\n",
    "ve_entidad_x_indigena": "RUNDEF Job\n SELECTION ALL\nTABLE T1\n AS FREQUENCY\n"
                             " OF ENTIDAD.LBLENT BY PERSONA.INDIGENA\n",
}


def _unpack(w, v):
    if v <= 200:
        return 200 * w + v - 1
    return {NOT_INDIGENOUS: 1001, BORN_ABROAD: 1002}[v]


def fetch():
    import requests
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    RAW.mkdir(parents=True, exist_ok=True)
    for name, program in PROGRAMS.items():
        dest = RAW / f"{name}.htm"
        if dest.exists() and dest.stat().st_size > 500:
            print("already have", dest.name)
            continue
        print("RUN", name)
        r = s.post(f"{HOST}/RpWebEngine.exe/CmdSet", data={
            "MAIN": "WebServerMain.inl", "BASE": BASE, "CODIGO": "xxUsuarioxx",
            "ITEM": "PROGRED", "MODE": "RUN", "CMDSET": program, "SUBMIT": "Ejecutar"},
            timeout=1800)
        if r.status_code != 200:
            raise SystemExit(f"{name}: HTTP {r.status_code}")
        m = re.search(r"LFN=([^\"'&<>]+?\.htm)", html.unescape(r.text))
        if not m:
            raise SystemExit(f"{name}: REDATAM returned no output file.\n"
                             f"{re.sub(r'<[^>]+>', ' ', r.text)[:800]}")
        for attempt in range(3):
            try:
                t = s.get(f"{HOST}/WebUtilities.exe/Text?LFN=" + urllib.parse.quote(m.group(1))
                          + "&TYPE=TMP", timeout=1800)
                break
            except requests.exceptions.ConnectionError:
                if attempt == 2:
                    raise
                time.sleep(5)
        t.raise_for_status()
        if "<table" not in t.text.lower():
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
    t = tok.replace(".", "").replace(" ", "")
    if t == "-":
        return 0
    if not re.fullmatch(r"\d+", t):
        raise SystemExit(f"{tok!r} is not a figure")
    return int(t)


def read_arealist(name):
    rows = _rows(name)
    header = next(r for r in rows if r[0] == "Código")
    cols = header[1:]
    if cols[-1] != "Total":
        raise SystemExit(f"{name}: AREALIST header ends {cols[-1]!r}")
    vals = [int(c) for c in cols[:-1]]
    data, totals = {}, {}
    for r in rows:
        if re.fullmatch(r"\d{6}", r[0]) and len(r) == len(cols) + 1:
            v = [_num(x) for x in r[1:]]
            data[r[0]] = dict(zip(vals, v[:-1]))
            totals[r[0]] = v[-1]
    return data, totals


def read_frequency(name):
    out = []
    for r in _rows(name):
        if len(r) == 4 and r[0] not in ("Total",) and re.fullmatch(r"[\d.]+", r[1]):
            out.append((r[0], _num(r[1])))
    return out


# INE's published table 22 groups the census's spelling variants into its 52 pueblos.
PUBLISHED = {
    "Añú": "Añú", "Paraujano": "Añú", "Akawayo": "Akawayo", "Kapón": "Akawayo",
    "Eñepa": "Eñepa", "Panare": "Eñepa", "Guajibo": "Jivi", "Amorúa": "Jivi", "Sikwani": "Jivi",
    "Jiwi": "Jivi", "Hoti": "Jodi", "Jodi": "Jodi", "Mapoyo": "Mapoyo", "Wanai": "Mapoyo",
    "Ñengatú": "Yeral", "Yeral": "Yeral", "Pemón": "Pemón", "Arekuna": "Pemón",
    "Kamarakoto": "Pemón", "Taurepán": "Pemón", "Chase": "Piapoko", "Piapoko": "Piapoko",
    "Piaroa": "Piaroa", "Wótüja": "Piaroa", "Pumé": "Yaruro", "Yaruro": "Yaruro",
    "Kuiva": "Kuiva", "Cuiba": "Kuiva", "Sanema": "Sanema", "Sanüma": "Sanema",
    "Arutani": "Arutani", "Uruak": "Arutani", "Guajiro": "Wayúu", "Wayuu": "Wayúu",
    "Makiritare": "Yekwana", "Yekwana": "Yekwana", "Curripaco": "Kurripako",
    "Kurripako": "Kurripako", "Guaiquerí": "Waikerí", "Waikerí": "Waikerí",
    "Caquetío": "Kaketío", "Kaketío": "Kaketío", "Timotocuica": "Timote 3", "Timote": "Timote 3",
    "Cumanagoto": "Kumanagoto", "Kumanagoto": "Kumanagoto", "Arawako": "Arawak",
    "Lokono": "Arawak", "Guanano": "Wanano", "Píritu": "Píritu1", "Shiriana": "Shiriana2",
    "No declarado": "No declarado 4", "Otro": "Otro Pueblo 5",
}
SAME = ["Baniva", "Baré", "Barí", "Chaima", "Kariña", "Mako", "Puinave", "Sapé", "Warao",
        "Warekena", "Yanomami", "Yavarana", "Yukpa", "Japreria", "Kubeo", "Makushi", "Matako",
        "Tukano", "Wapishana", "Sáliva", "Ayaman", "Gayón", "Inga", "Kechwa", "Jirajara",
        "Tunebo"]
PUBLISHED.update({k: k for k in SAME})

# table 22's columns, in its own order, as INE entity codes
T22_COLS = {"Dto Capital": "01", "Amazonas": "02", "Anzoátegui": "03", "Apure": "04",
            "Aragua": "05", "Barinas": "06", "Bolivar": "07", "Carabobo": "08", "Cojedes": "09",
            "Delta Amacuro": "10", "Falcón": "11", "Guárico": "12", "Lara": "13", "Mérida": "14",
            "Miranda": "15", "Monagas": "16", "Nueva Esparta": "17", "Portuguesa": "18",
            "Sucre": "19", "Tachira": "20", "Trujillo": "21", "Yaracuy": "22", "Vargas": "24",
            "Zulia": "23", "Dependencias Federales": "25"}

# INE Cuadro 4 (POBLACION-INDIGENA-CENSO-2011.pdf p. 8): the eight states it names
CUADRO4 = {"02": 76_314, "10": 41_543, "23": 443_544, "07": 54_686, "04": 11_559,
           "19": 22_213, "03": 33_848, "16": 17_898, "17": 2_200, "13": 2_112}


def read_table22():
    import xlrd
    b = xlrd.open_workbook(RAW / "Tabulados_Poblacion_Indigena.xls")
    sh = b.sheet_by_name("22.-Pueblos por entidad")
    head = [str(c).strip() for c in sh.row_values(1)]
    out = {}
    for r in range(2, sh.nrows):
        v = sh.row_values(r)
        lab = str(v[0]).strip()
        if not lab or lab == "TOTAL":
            continue
        for j, col in enumerate(head[1:-1], start=1):
            x = v[j]
            out[(lab, T22_COLS[col])] = 0 if x in ("-", "") else int(round(float(x)))
    return out


def main():
    if "--fetch" in sys.argv:
        fetch()

    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'OK ' if cond else 'BAD'} {msg}")

    # 1. shape and totals
    long, pop = [], None
    for name, w in TABLES:
        data, totals = read_arealist(name)
        bad = [p for p in data if sum(data[p].values()) != totals[p]]
        if pop is None:
            pop = totals
            print(f"  {len(pop):,} parroquias")
            check(sum(pop.values()) == POPULATION,
                  f"Totals sum to {sum(pop.values()):,} (the census counted {POPULATION:,})")
        if bad or totals != pop:
            check(False, f"{name}: {len(bad)} rows off their Total; totals agree: {totals == pop}")
        for p, d in data.items():
            for v, n in d.items():
                if n and v != NOT_HERE:
                    long.append((p, _unpack(w, v), n))
    df = pd.DataFrame(long, columns=["parroquia", "code", "count"])
    per = df.groupby("parroquia")["count"].sum()
    off = [p for p in pop if per.get(p, 0) != pop[p]]
    check(not off, f"across the windows every person is counted once ({len(off)} parroquias off)")
    by_code = df[df["code"] < 1000].groupby("code")["count"].sum().sort_index()
    check(int(by_code.sum()) == INDIGENOUS, f"indigenous {int(by_code.sum()):,} (INE {INDIGENOUS:,})")

    # 2. codes against the national frequency, paired by order
    freq = read_frequency("ve_cualindige")
    pairs_ok = len(freq) == len(by_code) and all(
        n == int(c) for (_, n), c in zip(freq, by_code.values))
    check(pairs_ok, f"{len(by_code)} codes in the parroquia tables, {len(freq)} labels in the "
                    "national frequency, counts equal in order")
    labels = {int(code): lab for code, (lab, _) in zip(by_code.index, freq)}

    # 3. per state and published pueblo, against INE's table 22
    unknown = sorted(set(labels.values()) - set(PUBLISHED))
    check(not unknown, f"every census label has a table-22 row ({unknown})")
    t22 = read_table22()
    ind = df[df["code"] < 1000].copy()
    ind["state"] = ind["parroquia"].str[:2]
    ind["t22"] = ind["code"].map(labels).map(PUBLISHED)
    mine = ind.groupby(["t22", "state"])["count"].sum()
    diffs = [(k, t22[k], int(mine.get(k, 0))) for k in t22 if t22[k] != int(mine.get(k, 0))]
    stray = [k for k in mine.index if k not in t22 and mine[k]]
    check(not diffs and not stray, f"table 22: {len(t22)} cells, {len(diffs)} differ "
                                   f"{diffs[:4]}, {len(stray)} cells it lacks")

    # 4. per state, against Cuadro 4
    st = ind.groupby("state")["count"].sum()
    bad4 = {s: (n, int(st.get(s, 0))) for s, n in CUADRO4.items() if n != int(st.get(s, 0))}
    check(not bad4, f"indigenous per state equals Cuadro 4 for its ten named states ({bad4})")

    if not ok:
        raise SystemExit("ve: checks FAILED, nothing written")

    NORM.mkdir(parents=True, exist_ok=True)
    df["geo_id"] = "VE" + df["parroquia"]
    lab = dict(labels)
    lab.update({1001: "born in Venezuela, not indigenous", 1002: "born abroad, not asked"})
    df["source_category"] = df["code"].map(lab)
    df[["geo_id", "code", "source_category", "count"]].sort_values(
        ["geo_id", "code"]).to_csv(NORM / "ve.csv", index=False)
    pd.DataFrame([(c, labels[c], PUBLISHED[labels[c]], int(n)) for c, n in by_code.items()],
                 columns=["code", "label", "table22_row", "count"]
                 ).to_csv(NORM / "ve_pueblos.csv", index=False)
    print(f"  wrote data/normalized/ve.csv ({len(df):,} rows) and ve_pueblos.csv")


if __name__ == "__main__":
    main()
