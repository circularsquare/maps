"""Argentina, Censo Nacional de Poblacion, Hogares y Viviendas 2022 (INDEC): people who speak
and/or understand the language of their indigenous people, per departamento.

    python sources/ar_censo.py [--fetch]

Writes
  data/normalized/ar.csv        one row per (departamento, code): `code` is INDEC's pueblo code
                                (P23, 10-990) for people who answered yes to P24, or one of the
                                codes below. Every person counted in 2022 is in exactly one row.
  data/normalized/ar_pueblos.csv  the pueblo codes, INDEC's labels and the national crosstab
                                (speak / do not / ignorado / total) they are checked against

THE QUESTION (private dwellings only; the collective-dwelling questionnaire has no such
question). P22 "Se reconoce indigena o descendiente de pueblos indigenas u originarios?";
if yes, P23 the pueblo (written in, coded by INDEC into about 75 categories) and P24 "Habla
y/o entiende la lengua de ese pueblo indigena u originario?" (si / no / ignorado). Nobody else
is asked about language. So the language is IMPLIED by the pueblo, and it is ability (speaks
or understands), not mother tongue.

THE SOURCE is INDEC's REDATAM webserver (redatam.indec.gob.ar, base CPV2022, Redatam 7). The
pueblo questions are in a separate database, CPV2022Afro ("Pueblos originarios,
afrodescendientes e identidad de genero", private dwellings), whose geography stops at the
departamento (PROV, DPTO); the main database, which goes to radio, carries no pueblo variable.
One Redatam program defines a category per person and tabulates it per departamento
(AREALIST OF DPTO); the national crosstab of pueblo x P24 comes from the same server.
The collective-dwelling database (CPV2022col: collective dwellings and people living on the
street) gives each departamento's remaining population, who were not asked.

Category codes in ar.csv besides the pueblo codes (pueblo code = the person speaks it):
  0     private dwelling, does not self-identify as indigenous (or P22 ignorado)
  1     self-identifies as indigenous, does NOT speak/understand their people's language
  2     self-identifies as indigenous, P24 ignorado
  3     collective dwelling or living on the street: not asked

CHECKS (the script stops unless all hold):
  1. every departamento's categories sum to its Total; the Totals sum to 45,618,787, the
     private-dwelling population, and with the collective population to 45,892,285, INDEC's
     2022 definitive total.
  2. per departamento, the Afro database's total equals the main private-dwelling database's
     (CPV2022.dic) person count: two databases, one census.
  3. per pueblo code, speakers summed over departamentos equal the national crosstab
     (P23 x P24) exactly; and codes 1 and 2 equal its No and Ignorado column totals.
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
RAW = HERE / "data" / "raw" / "ar"
NORM = HERE / "data" / "normalized"

HOST = "https://redatam.indec.gob.ar"
BASE = "CPV2022"
PRIVATE = 45_618_787
TOTAL = 45_892_285

# INDEC's P23 categories, from the base's own variable list (redarg/CENSOS/CPV2022/Docs/
# CPV2022-afro.xlsx, sheet "Poblacion orig. afrod. y gen."), copied verbatim.
PUEBLOS = {
    10: "Atacama", 20: "Alakaluf", 30: "Aoniken", 40: "Avipón", 50: "Aymara", 60: "Chana",
    70: "Chané", 80: "Charrúa", 90: "Chicha", 100: "Chorote", 110: "Nivaclé",
    120: "Comechingón", 130: "Corundí", 140: "Diaguita", 141: "Diaguita",
    142: "Diaguita Amaicha", 143: "Diaguita Calchaquí", 144: "Diaguita Quechua",
    145: "Diaguita Ingamana", 146: "Diaguita Quilmes", 147: "Diaguita Tolombón",
    148: "Diaguita Colastiné", 149: "Diaguita Capayán", 150: "Diaguita Cacano",
    160: "Fiscara", 170: "Guaraní", 171: "Guaraní", 172: "Ava Guaraní", 173: "Tupí Guaraní",
    180: "Guarayo", 190: "Guaycurú", 200: "Günün A Küna", 210: "Huarpe", 220: "Iogys",
    230: "Isoceño", 240: "Kolla", 241: "Kolla", 242: "Kolla Diaguita", 243: "Kolla Quechua",
    250: "Kolla Atacameño", 260: "Lule", 270: "Lule Vilela", 280: "Mapuche", 281: "Mapuche",
    282: "Huiliche", 283: "Pehuenche", 284: "Picunche", 285: "Puelche",
    290: "Mapuche Tehuelche", 300: "Mbya Guaraní", 310: "Moqoit/Mocoví", 320: "Ocloya",
    330: "Omaguaca", 340: "Pilagá", 350: "Qom/Toba", 360: "Quechua", 370: "Querandí",
    380: "Ranquel", 390: "Sanavirón", 400: "Selk´Nam/Ona", 410: "Tapiete", 420: "Tastil",
    430: "Tehuelche", 440: "Tilián", 450: "Toara", 460: "Tonokoté", 470: "Vilela",
    480: "Weenhayek", 490: "Wichi", 500: "Yagán", 510: "Wayteca/Chono", 520: "Haush/Maneken",
    530: "Mak'a", 540: "Minuán", 550: "Ansilta", 560: "Churumata", 570: "Jujuies",
    580: "Michilingüe", 970: "No Codificables", 980: "No Corresponde", 990: "Sin información",
}
SPECIAL = {0: "No se reconoce indígena", 1: "Indígena, no habla la lengua de su pueblo",
           2: "Indígena, habla la lengua: ignorado", 3: "Vivienda colectiva o situación de calle"}

# The program's own category values: 1..N for the pueblo codes in order, then these.
_ORDER = sorted(PUEBLOS)
V_NOSPEAK, V_IGNORADO, V_OTHER = len(_ORDER) + 1, len(_ORDER) + 2, len(_ORDER) + 3


def _lcat_program():
    """The web form eats a `<` (it reads as a tag), so every condition is an `=`; the first
    INCASE that holds wins."""
    lines = ["RUNDEF Job", " SELECTION ALL", "DEFINE PERSONA.LCAT", " AS SWITCH"]
    for i, code in enumerate(_ORDER, start=1):
        lines += [f" INCASE PERSONA.LENGIND = 1 AND PERSONA.PUEBLO = {code}", f"  ASSIGN {i}"]
    lines += [" INCASE PERSONA.LENGIND = 2", f"  ASSIGN {V_NOSPEAK}",
              " INCASE PERSONA.LENGIND = 9", f"  ASSIGN {V_IGNORADO}",
              f" DEFAULT {V_OTHER}", " TYPE INTEGER", f" RANGE 1-{V_OTHER}",
              "TABLE T1", " AS AREALIST", " OF DPTO, PERSONA.LCAT", ""]
    return "\n".join(lines)


_COUNT = "RUNDEF Job\n SELECTION ALL\nTABLE T1\n AS AREALIST\n OF DPTO, PERSONA.SEXO\n"

# name: (database item, program). The ITEM picks the database: *AFRO the pueblo database,
# *PART the main private-dwelling one, *COL collective dwellings and the street.
PROGRAMS = {
    "ar_dpto_lengua": ("PROGVIVAFRO", _lcat_program()),
    "ar_pueblo_x_lengua": ("PROGVIVAFRO", "RUNDEF Job\n SELECTION ALL\nTABLE T1\n AS FREQUENCY\n"
                                          " OF PERSONA.PUEBLO BY PERSONA.LENGIND\n"),
    "ar_dpto_privadas": ("PROGVIVPART", _COUNT),
    "ar_dpto_colectivas": ("PROGVIVCOL", _COUNT),
}


def fetch():
    import requests
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    RAW.mkdir(parents=True, exist_ok=True)
    for name, (item, program) in PROGRAMS.items():
        dest = RAW / f"{name}.htm"
        if dest.exists() and dest.stat().st_size > 2_000:
            print("already have", dest.name)
            continue
        print("RUN", name)
        r = s.post(f"{HOST}/binarg/RpWebStats.exe/CmdSet?", data={
            "MAIN": "WebServerMain.inl", "BASE": BASE, "LANG": "ESP",
            "CODIGO": "XXUSUARIOXX", "ITEM": item, "MODE": "RUN", "CMDSET": program},
            timeout=1800, verify=False)
        r.encoding = "utf-8"
        if r.status_code != 200:
            raise SystemExit(f"{name}: HTTP {r.status_code}\n"
                             f"{re.sub(r'<[^>]+>', ' ', r.text)[:800]}")
        # this build answers with an iframe onto /redarg//tempo/<session>/~tmp_*.htm
        m = re.search(r"(/redarg/+tempo/[^\"'<>]+?\.htm)", html.unescape(r.text))
        if not m:
            raise SystemExit(f"{name}: REDATAM returned no output file.\n"
                             f"{re.sub(r'<[^>]+>', ' ', r.text)[:800]}")
        for attempt in range(3):
            try:
                t = s.get(HOST + m.group(1), timeout=1800, verify=False)
                break
            except requests.exceptions.ConnectionError:
                if attempt == 2:
                    raise
                time.sleep(5)
        t.raise_for_status()
        t.encoding = "utf-8"
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
        raise ValueError(f"{tok!r} is not a figure")
    return int(t)


def read_arealist(name):
    """-> ({dpto: {column header: count}}, {dpto: printed Total})."""
    rows = _rows(name)
    header = next(r for r in rows if r[0] == "Código")
    cols = header[1:]
    if cols[-1] != "Total":
        raise SystemExit(f"{name}: AREALIST header ends {cols[-1]!r}")
    data, totals = {}, {}
    for r in rows:
        if re.fullmatch(r"\d{5}", r[0]):
            if len(r) != len(cols) + 1:
                raise SystemExit(f"{name}: row {r[0]} has {len(r) - 1} cells, header {len(cols)}")
            v = [_num(x) for x in r[1:]]
            data[r[0]] = dict(zip(cols[:-1], v[:-1]))
            totals[r[0]] = v[-1]
    return data, totals


def read_crosstab(name):
    """FREQUENCY ... BY ...: {row label: [Si, No, Ignorado, Total]}, in printed order."""
    out = []
    for r in _rows(name):
        if len(r) == 5:
            try:
                out.append((r[0], [_num(x) for x in r[1:]]))
            except ValueError:
                continue
    return out


def main():
    if "--fetch" in sys.argv:
        fetch()

    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("ok    " if cond else "FAIL  ") + msg)
        ok &= bool(cond)

    lang, ltot = read_arealist("ar_dpto_lengua")
    priv, ptot = read_arealist("ar_dpto_privadas")
    col, ctot = read_arealist("ar_dpto_colectivas")

    # the AREALIST header carries the category VALUES of LCAT (1..N+3) as printed labels
    vals = {}
    for k in next(iter(lang.values())):
        if not re.fullmatch(r"\d+", k):
            raise SystemExit(f"ar_dpto_lengua: column {k!r} is not a category value")
        vals[k] = int(k)
    print(f"{len(lang)} departamentos in the Afro database, {len(priv)} in the main private "
          f"database, {len(col)} with a collective population; {len(vals)} LCAT columns")

    bad = [d for d in lang if sum(lang[d].values()) != ltot[d]]
    check(not bad, f"every departamento's categories sum to its Total ({len(bad)} do not)")
    check(sum(ltot.values()) == PRIVATE,
          f"Afro database total {sum(ltot.values()):,} == private population {PRIVATE:,}")
    check(set(lang) == set(priv), "same departamentos in the Afro and main databases")
    diff = [d for d in lang if ltot[d] != ptot.get(d)]
    check(not diff, f"per departamento, Afro total == main database's persons ({len(diff)} differ)")
    # 94028 Antartida Argentina: only collective dwellings (the bases), so not in the Afro table
    extra = sorted(set(col) - set(lang))
    check(extra == ["94028"], f"the only departamento with no private dwellings is 94028 "
                              f"Antartida Argentina ({extra[:5]}, {sum(ctot[d] for d in extra)} people)")
    check(sum(ltot.values()) + sum(ctot.values()) == TOTAL,
          f"private {sum(ltot.values()):,} + collective {sum(ctot.values()):,} == {TOTAL:,}")

    rows = []
    for d in sorted(set(lang) | set(col)):
        for k, n in lang.get(d, {}).items():
            v = vals[k]
            if n == 0:
                continue
            if v <= len(_ORDER):
                code = _ORDER[v - 1]
            else:
                code = {V_NOSPEAK: 1, V_IGNORADO: 2, V_OTHER: 0}[v]
            rows.append((d, code, n))
        if ctot.get(d, 0):
            rows.append((d, 3, ctot[d]))
    df = pd.DataFrame(rows, columns=["geo_id", "code", "count"])
    labels = {**PUEBLOS, **SPECIAL}
    df["source_category"] = df["code"].map(labels)
    df["geo_level"] = "departamento"
    check(df["count"].sum() == TOTAL, f"ar.csv sums to {df['count'].sum():,}")

    # national crosstab, pueblo x P24
    xt = read_crosstab("ar_pueblo_x_lengua")
    total_row = [v for lab, v in xt if lab == "Total"]
    xt = [(lab, v) for lab, v in xt if lab != "Total"]
    # INDEC prints labels, not codes, and two pairs of codes share a label (140/141 Diaguita,
    # 170/171 Guarani, 240/241 Kolla, 280/281 Mapuche): match them by label, summing the codes
    by_label = df[df["code"] >= 10].groupby("source_category")["count"].sum()
    xt_lab = {}
    for lab, v in xt:
        xt_lab[lab] = xt_lab.get(lab, 0) + v[0]
    xt_lab = {lab: n for lab, n in xt_lab.items() if n}
    miss = sorted(set(by_label.index) ^ set(xt_lab))
    check(not miss, f"the same pueblo labels carry speakers in both tables ({miss})")
    mism = [(lab, by_label.get(lab, 0), xt_lab.get(lab, 0)) for lab in xt_lab
            if by_label.get(lab, 0) != xt_lab[lab]]
    check(not mism, f"per pueblo, speakers over departamentos == national crosstab, "
                    f"{len(xt_lab)} labels ({mism[:5]})")
    tot = total_row[0]
    n1 = int(df.loc[df["code"] == 1, "count"].sum())
    n2 = int(df.loc[df["code"] == 2, "count"].sum())
    nsp = int(df.loc[df["code"] >= 10, "count"].sum())
    check(nsp == tot[0] and n1 == tot[1] and n2 == tot[2],
          f"speakers {nsp:,} / no {n1:,} / ignorado {n2:,} == crosstab {tot[:3]}")
    print(f"self-identified indigenous {tot[3]:,}; speak or understand their people's language "
          f"{tot[0]:,}; of whom pueblo 'Sin información' {xt_lab.get('Sin información', 0):,}")

    if not ok:
        raise SystemExit("checks failed; nothing written")
    NORM.mkdir(parents=True, exist_ok=True)
    df[["geo_level", "geo_id", "code", "source_category", "count"]].to_csv(
        NORM / "ar.csv", index=False, encoding="utf-8")
    pd.DataFrame([(lab, *v) for lab, v in xt], columns=["pueblo", "speaks", "does_not",
                                                         "ignorado", "total"]).to_csv(
        NORM / "ar_pueblos.csv", index=False, encoding="utf-8")
    print(f"wrote {NORM / 'ar.csv'} ({len(df):,} rows) and ar_pueblos.csv")


if __name__ == "__main__":
    main()
