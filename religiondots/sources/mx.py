"""Mexico — INEGI Censo de Poblacion y Vivienda 2020, religion.

Reads data/raw/mx/ and writes data/normalized/mx.csv.

Two raw files, because INEGI splits granularity between them and neither has
both halves (see sources/mx.md):

  cpv2020_b_eum_12_religion.xlsx   24 denominations, entidad federativa only
  iter_00_cpv2020_csv.zip (ITER)    4 aggregate groups, down to locality

So the normalised file carries two geo levels with two different category
depths.  Within each level the categories are a partition; ACROSS levels they
nest -- the four ITER groups are exactly the four aggregates of the 24
denominations -- so the two blocks must never be summed together.  The
reconciliation printed at the end of a run checks that nesting explicitly.

Usage:
    python sources/mx.py            normalise from data/raw/mx/
    python sources/mx.py --fetch    download the raw files first if missing
"""

import csv
import io
import os
import sys
import unicodedata
import zipfile

import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "mx")
OUT = os.path.join(ROOT, "data", "normalized", "mx.csv")

SOURCE_ID = "mx_censo_2020"
YEAR = 2020
BASIS = "self_id"

XLSX = "cpv2020_b_eum_12_religion.xlsx"
XLSX_URL = ("https://www.inegi.org.mx/contenidos/programas/ccpv/2020/"
            "tabulados/cpv2020_b_eum_12_religion.xlsx")
ITER_ZIP = "iter_00_cpv2020_csv.zip"
ITER_URL = ("https://www.inegi.org.mx/contenidos/programas/ccpv/2020/"
            "datosabiertos/iter/iter_00_cpv2020_csv.zip")
ITER_MEMBER = ("iter_00_cpv2020/conjunto_de_datos/"
               "conjunto_de_datos_iter_00CSV20.csv")
ITER_DICT = ("iter_00_cpv2020/diccionario_datos/"
             "diccionario_datos_iter_00CSV20.csv")

# ITER's four religion columns.  The category label written to the CSV is
# INEGI's own "Indicador" wording, read verbatim out of the data dictionary
# shipped inside the same zip rather than retyped here.
ITER_COLUMNS = ["PCATOLICA", "PRO_CRIEVA", "POTRAS_REL", "PSIN_RELIG"]

# Footnotes carried as superscripts on category names in the xlsx.  The digit
# is a footnote marker, not part of the name, so it is stripped from
# source_category and the footnote text goes in `note` instead.
FOOTNOTES = {
    "2": ("Incluye las denominaciones religiosas: budista, hinduista y otras "
          "de origen oriental."),
    "3": "Incluye la denominacion religiosa catolica ortodoxa.",
}

# Which ITER group each xlsx denomination rolls up into.  Used ONLY by the
# reconciliation check below -- it is not written to the CSV, and it is not a
# taxonomy mapping (spec 2.4: cross-source matching is deferred).  "No
# especificado" maps to None because ITER does not publish it at all.
ROLLUP = {
    "Catolica": "PCATOLICA",
    "Bautista": "PRO_CRIEVA",
    "Presbiteriana": "PRO_CRIEVA",
    "Iglesia del Dios Vivo, Columna y Apoyo de la Verdad, la Luz del Mundo":
        "PRO_CRIEVA",
    "Adventista del Septimo Dia": "PRO_CRIEVA",
    "Iglesia de Jesucristo de los Santos de los Ultimos Dias (Mormon)":
        "PRO_CRIEVA",
    "Testigo de Jehova": "PRO_CRIEVA",
    "Cristiana": "PRO_CRIEVA",
    "Evangelica": "PRO_CRIEVA",
    "Pentecostal": "PRO_CRIEVA",
    "Otro Protestante/cristiano evangelico": "PRO_CRIEVA",
    "Judia": "POTRAS_REL",
    "Islamica": "POTRAS_REL",
    "Origen oriental": "POTRAS_REL",
    "New Age y Escuelas esotericas": "POTRAS_REL",
    "Raices etnicas": "POTRAS_REL",
    "Raices afro": "POTRAS_REL",
    "Espiritualista": "POTRAS_REL",
    "Cultos populares": "POTRAS_REL",
    "Otras religiones o movimientos religiosos": "POTRAS_REL",
    "Ninguna religion": "PSIN_RELIG",
    "Ateos/Agnosticos": "PSIN_RELIG",
    "Sin adscripcion religiosa (creyente)": "PSIN_RELIG",
    "No especificado": None,
}

ENTIDAD_NOTE = "denominacion religiosa (detailed level)"


# Fold accents so the ROLLUP keys above can be written in ASCII; the CSV keeps
# the real accented strings.
def deaccent(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def fetch():
    import requests
    os.makedirs(RAW, exist_ok=True)
    for name, url in ((XLSX, XLSX_URL), (ITER_ZIP, ITER_URL)):
        dest = os.path.join(RAW, name)
        if os.path.exists(dest):
            print("have", name)
            continue
        print("downloading", url)
        r = requests.get(url, timeout=600,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        # INEGI's IIS answers a missing file with a 200 and a ~2KB HTML page.
        if len(r.content) < 100_000:
            raise SystemExit("suspiciously small download for %s (%d bytes) "
                             "-- INEGI serves soft 404s as HTTP 200"
                             % (name, len(r.content)))
        with open(dest, "wb") as f:
            f.write(r.content)
        print("  wrote", dest, len(r.content), "bytes")


def split_footnote(text):
    """'Origen oriental2' -> ('Origen oriental', 'Incluye ...')."""
    text = text.replace("\xa0", " ").strip()
    if text and text[-1] in FOOTNOTES:
        return text[:-1].strip(), FOOTNOTES[text[-1]]
    return text, ""


def read_entidad():
    """Sheet '02' of the xlsx: entidad federativa x 24 denominations."""
    path = os.path.join(RAW, XLSX)
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["02"]
    rows = []
    national = {}
    for raw in ws.iter_rows(values_only=True):
        ent, sexo, den, total = raw[0], raw[1], raw[2], raw[3]
        if not isinstance(ent, str) or sexo != "Total":
            continue
        if not isinstance(den, str) or not isinstance(total, int):
            continue
        category, note = split_footnote(den)
        if category == "Total":
            # The partition's parent.  Kept out of the CSV so a naive sum of
            # the file cannot double count, but held for the reconciliation.
            if ent == "Estados Unidos Mexicanos":
                national["__TOTAL__"] = total
            continue
        if ent == "Estados Unidos Mexicanos":
            national[category] = total
            continue
        code, _, name = ent.partition(" ")
        if not (len(code) == 2 and code.isdigit()):
            raise SystemExit("unexpected entidad label: %r" % (ent,))
        rows.append({
            "geo_id": code,
            "geo_level": "entidad",
            "geo_name": name.strip(),
            "source_category": category,
            "count": total,
            "basis": BASIS,
            "year": YEAR,
            "source_id": SOURCE_ID,
            "note": ENTIDAD_NOTE + ("; " + note if note else ""),
        })
    wb.close()
    return rows, national


def read_iter_labels(z):
    """INEGI's own label for each ITER column, from the shipped dictionary."""
    with z.open(ITER_DICT) as f:
        rd = csv.reader(io.TextIOWrapper(f, encoding="utf-8-sig", newline=""))
        labels = {}
        for rec in rd:
            if len(rec) >= 4 and rec[3] in ITER_COLUMNS:
                labels[rec[3]] = rec[1].strip()
    missing = [c for c in ITER_COLUMNS if c not in labels]
    if missing:
        raise SystemExit("no dictionary label for %s" % missing)
    return labels


def read_municipio():
    """ITER: municipio totals x 4 aggregate groups, plus POBTOT for checking."""
    path = os.path.join(RAW, ITER_ZIP)
    rows = []
    pobtot = {}
    national = {}
    with zipfile.ZipFile(path) as z:
        labels = read_iter_labels(z)
        with z.open(ITER_MEMBER) as f:
            rd = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig",
                                                 newline=""))
            for rec in rd:
                ent, mun, loc = rec["ENTIDAD"], rec["MUN"], rec["LOC"]
                if ent == "00" and mun == "000" and loc == "0000":
                    national = {"POBTOT": int(rec["POBTOT"])}
                    for col in ITER_COLUMNS:
                        national[col] = int(rec[col])
                    continue
                # LOC 0000 is the municipio total; 9998/9999 are ITER's
                # small-locality roll-ups and would double count.
                if mun == "000" or loc != "0000":
                    continue
                geo_id = ent + mun
                pobtot[geo_id] = int(rec["POBTOT"])
                for col in ITER_COLUMNS:
                    value = rec[col]
                    if not value.isdigit():
                        raise SystemExit(
                            "confidentiality mask %r at municipio %s -- ITER "
                            "is unmasked at this level, so this is new" %
                            (value, geo_id))
                    rows.append({
                        "geo_id": geo_id,
                        "geo_level": "municipio",
                        "geo_name": rec["NOM_MUN"],
                        "source_category": labels[col],
                        "count": int(value),
                        "basis": BASIS,
                        "year": YEAR,
                        "source_id": SOURCE_ID,
                        "note": ("grupo religioso (aggregate level); ITER "
                                 "column " + col),
                    })
    return rows, pobtot, national, labels


def reconcile(ent_rows, ent_national, mun_rows, pobtot, iter_national, labels):
    ok = True

    def check(label, got, want):
        nonlocal ok
        good = got == want
        ok = ok and good
        print("  %-58s %13s %13s  %s" %
              (label, f"{got:,}", f"{want:,}", "ok" if good else "MISMATCH"))

    print("\nreconciliation (computed vs INEGI published)")
    print("  %-58s %13s %13s" % ("", "computed", "published"))

    total = ent_national["__TOTAL__"]
    named = {k: v for k, v in ent_national.items() if k != "__TOTAL__"}
    check("entidad: 24 denominations summed -> national population",
          sum(named.values()), total)
    check("entidad: rows summed per denomination -> national",
          sum(r["count"] for r in ent_rows), total)
    check("ITER: municipio POBTOT summed -> national population",
          sum(pobtot.values()), iter_national["POBTOT"])
    check("ITER: national population", iter_national["POBTOT"], total)

    # The nesting check: the four ITER groups are the four aggregates of the
    # 24 denominations, and "No especificado" is in neither.
    roll = {col: 0 for col in ITER_COLUMNS}
    unmapped = 0
    for name, value in named.items():
        key = deaccent(name)
        if key not in ROLLUP:
            raise SystemExit("denomination not in ROLLUP: %r" % (name,))
        group = ROLLUP[key]
        if group is None:
            unmapped += value
        else:
            roll[group] += value
    print()
    for col in ITER_COLUMNS:
        check("nesting: denominations -> %s" % col, roll[col],
              iter_national[col])
    check("ITER omits 'No especificado' (residual vs POBTOT)",
          iter_national["POBTOT"] - sum(iter_national[c]
                                        for c in ITER_COLUMNS),
          unmapped)

    mun_sums = {}
    for r in mun_rows:
        mun_sums[r["source_category"]] = (
            mun_sums.get(r["source_category"], 0) + r["count"])
    print()
    for col in ITER_COLUMNS:
        check("municipio rows summed -> %s" % col, mun_sums[labels[col]],
              iter_national[col])
    return ok


def main():
    if "--fetch" in sys.argv:
        fetch()
    for name in (XLSX, ITER_ZIP):
        if not os.path.exists(os.path.join(RAW, name)):
            raise SystemExit("missing %s -- run with --fetch" %
                             os.path.join(RAW, name))

    ent_rows, ent_national = read_entidad()
    mun_rows, pobtot, iter_national, labels = read_municipio()

    entidades = len({r["geo_id"] for r in ent_rows})
    municipios = len({r["geo_id"] for r in mun_rows})
    categories = len({r["source_category"] for r in ent_rows})
    print("entidad   %d units x %d denominations = %d rows"
          % (entidades, categories, len(ent_rows)))
    print("municipio %d units x %d groups        = %d rows"
          % (municipios, len(ITER_COLUMNS), len(mun_rows)))

    ok = reconcile(ent_rows, ent_national, mun_rows, pobtot, iter_national,
                   labels)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fields = ["geo_id", "geo_level", "geo_name", "source_category", "count",
              "basis", "year", "source_id", "note"]
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in ent_rows + mun_rows:
            w.writerow(r)
    print("\nwrote", OUT, len(ent_rows) + len(mun_rows), "rows")
    if not ok:
        raise SystemExit("reconciliation failed -- see MISMATCH above")


if __name__ == "__main__":
    main()
