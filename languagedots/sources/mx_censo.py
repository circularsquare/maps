"""Mexico, Censo de Poblacion y Vivienda 2020: speakers of each indigenous language aged 3 and
over, per municipio, from INEGI's own interactive cube over the full count.

    python sources/mx_censo.py [--fetch]

Writes
  data/normalized/mx.csv          one row per (municipio, language) with speakers > 0: the
                                  cube's language labels verbatim (INALI's 68 agrupaciones, the
                                  three "insuficientemente especificado" labels, "Otras lenguas
                                  indigenas de America" and "No especificado")
  data/normalized/mx_status.csv   per municipio: pop (POBTOT, all ages, ITER), p3 (aged 3+),
                                  hli (speaks an indigenous language), no_hli, ne (did not say)

THE QUESTION (cuestionario basico, everyone aged 3+): "¿(NOMBRE) habla algun dialecto o lengua
indigena?", then "¿Que dialecto o lengua indigena habla?" (one answer, coded to INALI's
catalogue), then "¿habla tambien espanol?". Nobody is asked about Spanish as such, so a person
who does not speak an indigenous language is drawn as Spanish (spec §3.5).

THE SOURCE. INEGI's per-entity tabulados (cpv2020_b_<ent>_05_etnicidad.xlsx) give municipios
only the yes/no; the language x entidad table is national (cpv2020_b_eum_05_etnicidad.xlsx, sheet
04). The municipio x language cross is in the "Consulta interactiva de datos" OLAP cube
"Poblacion 3 anos y mas" (database PV2020_AMD_Poblacion, page
inegi.org.mx/sistemas/Olap/Proyectos/bd/censos/cpv2020/P3Mas.asp, footnote "Cuestionario
Basico"), dimension "Habla indigena y lengua (INALI)". Its CSV export takes an MDX query
(exporta.aspx, Lc_sql); _MDX below is the one used. No login, no key.

CHECKS (the script stops unless all hold):
  1. 2,469 municipios; per municipio the language columns sum to "Habla lengua indigena".
  2. per municipio, the cube's aged-3+ total and speakers equal ITER 2020's P_3YMAS and
     P3YM_HLI (religiondots' copy of iter_00_cpv2020_csv.zip, read-only).
  3. per entidad and language, the municipios' sum equals the national tabulado's sheet 04.
  4. nationally 7,364,645 speakers and 119,976,584 aged 3+, INEGI's published figures.
"""
import csv

import re
import sys
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "mx"
NORM = ROOT / "data" / "normalized"
ITER = ROOT.parent / "religiondots" / "data" / "raw" / "mx" / "iter_00_cpv2020_csv.zip"
CUBE_CSV = RAW / "cpv2020_olap_p3mas_municipio_lengua.csv"
TAB = RAW / "cpv2020_b_eum_05_etnicidad.xlsx"
TAB_URL = ("https://www.inegi.org.mx/contenidos/programas/ccpv/2020/tabulados/"
           "cpv2020_b_eum_05_etnicidad.xlsx")
EXPORT_URL = "https://www.inegi.org.mx/sistemas/olap/exporta/exporta.aspx"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

_D = "[Habla indígena y lengua INALI].[Habla indígena y lengua INALI]"
_MDX = ("select {" + _D + ".[Total], " + _D + ".[Habla indígena y lengua INALI].Members, "
        + _D + ".[Habla indígena y lengua INALI].&[Habla lengua indígena].Children} on columns, "
        "{Descendants([Entidad y municipio].[Entidad y municipio].[Total], 2)} on rows "
        "from [Poblacion 3 años y mas] where ([Measures].[Población  de 3 años y más])")
STATUS = {"Total": "p3", "Habla lengua indígena": "hli", "No habla lengua indígena": "no_hli",
          "No especificado": "ne", "No aplica": "na"}


def fetch():
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    if not TAB.exists():
        r = requests.get(TAB_URL, headers=UA, timeout=300)
        r.raise_for_status()
        if len(r.content) < 100_000:      # INEGI answers a missing file with 200 and ~2 KB
            raise SystemExit(f"{TAB_URL}: {len(r.content)} bytes, not the tabulado")
        TAB.write_bytes(r.content)
    if not CUBE_CSV.exists():
        fields = {
            "nomdimfila": "Entidad y municipio", "to_display": "",
            "cube": "Poblacion 3 años y mas", "cubeName": "Poblacion 3 años y mas",
            "nomdimColumna": "Habla indígena y lengua INALI",
            "Lc_tituloFiltro": "Consulta de: Población  de 3 años y más   Por: Entidad y "
                               "municipio   Según: Habla indígena y lengua INALI",
            "Lc_unidadmedida": "Personas", "Lc_sql": _MDX,
            "Lc_conexion": "provider=MSOLAP.8;MDX Compatibility=2;data source=W-OLAPCLPRO22;"
                           "Connect timeout=120;Initial catalog=PV2020_AMD_Poblacion",
            "Lc_titulo": "Población|", "Lc_piepagina": "FUENTE:", "Lc_salida": "0",
            "Lc_StrConexion": "1", "Lc_ValidaDimGeo": "0",
            "Lc_formato": "Texto separado por comas(.csv)",
            # exporta.aspx answers an HTML error page without these; their values do not
            # limit the export
            "Lc_encabeza": "-", "Cant_Col": "80", "Cant_Fil": "2469", "completo": "completo",
        }
        body = "&".join(requests.utils.quote(k, safe="", encoding="latin-1") + "="
                        + requests.utils.quote(v, safe="", encoding="latin-1")
                        for k, v in fields.items())
        r = requests.post(EXPORT_URL, data=body.encode("ascii"), timeout=900,
                          headers={**UA, "Content-Type": "application/x-www-form-urlencoded"})
        r.raise_for_status()
        if len(r.content) < 100_000 or b"Akateko" not in r.content:
            raise SystemExit(f"cube export: {len(r.content)} bytes, not the table")
        CUBE_CSV.write_bytes(r.content)
    print(f"  raw: {TAB.name} {TAB.stat().st_size:,} B, {CUBE_CSV.name} "
          f"{CUBE_CSV.stat().st_size:,} B")


def _num(s):
    s = s.strip().replace(",", "")
    return int(s) if s else 0


def read_cube():
    text = CUBE_CSV.read_bytes().decode("cp1252")
    rows = list(csv.reader(text.splitlines(), skipinitialspace=True))
    head = next(r for r in rows if len(r) > 3 and r[2].strip() == "Total")
    cols = [c.strip() for c in head[2:]]
    # the status members come first, then the language children; "No especificado" is both a
    # status ("did not say whether they speak one") and a language child ("speaks one, which
    # was not given"), so position decides
    n_status = cols.index("Akateko")
    out = []
    for r in rows:
        if len(r) < 3 or not re.fullmatch(r"\d\d \d\d\d", r[0].strip()):
            continue
        geo = r[0].strip().replace(" ", "")
        vals = [_num(v) for v in r[2:2 + len(cols)]]
        rec = {"geo_id": geo, "geo_name": r[1].strip()}
        for c, v in zip(cols[:n_status], vals[:n_status]):
            rec[STATUS[c]] = v
        rec["langs"] = dict(zip(cols[n_status:], vals[n_status:]))
        out.append(rec)
    return out, cols[n_status:]


def read_iter():
    z = zipfile.ZipFile(ITER)
    name = next(n for n in z.namelist() if n.endswith("conjunto_de_datos_iter_00CSV20.csv"))
    df = pd.read_csv(z.open(name), dtype=str, encoding="utf-8",
                     usecols=["ENTIDAD", "MUN", "LOC", "POBTOT", "P_3YMAS", "P3YM_HLI"])
    m = df[(df["LOC"] == "0000") & (df["MUN"] != "000")].copy()
    m["geo_id"] = m["ENTIDAD"] + m["MUN"]
    for c in ("POBTOT", "P_3YMAS", "P3YM_HLI"):
        m[c] = m[c].astype(int)
    return m.set_index("geo_id")


def _key(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\d+$", "", s.strip())             # footnote digits
    return re.sub(r"[^a-z]", "", s)


def read_tabulado():
    """Sheet 04: entidad x language, speakers aged 3+ (column 3)."""
    import openpyxl
    wb = openpyxl.load_workbook(TAB, read_only=True)
    ws = wb["04"]
    out = {}
    ent = None
    for row in ws.iter_rows(min_row=9, values_only=True):
        if row[0] is None and row[1] is None:
            continue
        if row[0]:
            ent = str(row[0]).strip()
        lang, total = row[1], row[2]
        if lang is None or not isinstance(total, (int, float)):
            continue
        code = ent[:2] if re.match(r"\d\d ", ent) else "00"
        out[(code, _key(lang))] = int(total)
    return out


def main():
    if "--fetch" in sys.argv or not CUBE_CSV.exists() or not TAB.exists():
        fetch()
    recs, langs = read_cube()
    it = read_iter()
    print(f"  cube: {len(recs):,} municipios x {len(langs)} language labels")

    # 1
    assert len(recs) == 2469, len(recs)
    assert len({r["geo_id"] for r in recs}) == 2469
    bad = [r["geo_id"] for r in recs if sum(r["langs"].values()) != r["hli"]]
    assert not bad, f"languages do not sum to speakers in {bad[:5]}"
    bad = [r["geo_id"] for r in recs if r["p3"] != r["hli"] + r["no_hli"] + r["ne"]]
    assert not bad, f"status does not sum to aged 3+ in {bad[:5]}"
    assert all(r.get("na", 0) == 0 for r in recs)
    print("  check 1: every municipio's languages sum to its speakers, status to its aged 3+")
    # 2
    miss = sorted(set(it.index) ^ {r["geo_id"] for r in recs})
    assert not miss, f"cube and ITER municipios differ: {miss[:5]}"
    bad = [r["geo_id"] for r in recs
           if r["p3"] != it.at[r["geo_id"], "P_3YMAS"] or r["hli"] != it.at[r["geo_id"], "P3YM_HLI"]]
    assert not bad, f"cube against ITER differs in {bad[:5]}"
    print("  check 2: all 2,469 municipios' aged 3+ and speakers equal ITER 2020 exactly")
    # 3
    tab = read_tabulado()
    sums = {}
    for r in recs:
        for lang, v in r["langs"].items():
            for code in (r["geo_id"][:2], "00"):
                k = (code, _key(lang))
                sums[k] = sums.get(k, 0) + v
    tab_l = {k: v for k, v in tab.items() if k[1] != "total"}
    diff = [(k, sums.get(k, 0), v) for k, v in tab_l.items() if sums.get(k, 0) != v]
    extra = [k for k, v in sums.items() if v and k not in tab_l]
    assert not diff and not extra, f"against the tabulado: {diff[:5]} {extra[:5]}"
    n_cells = sum(1 for k in tab_l if k[0] != "00")
    print(f"  check 3: {n_cells:,} entidad x language cells and {sum(1 for k in tab_l if k[0] == '00')} "
          "national ones equal the tabulado's sheet 04 exactly")
    # 4
    hli = sum(r["hli"] for r in recs)
    p3 = sum(r["p3"] for r in recs)
    assert hli == 7_364_645 and p3 == 119_976_584, (hli, p3)
    pop = int(it["POBTOT"].sum())
    print(f"  check 4: {hli:,} speakers of {p3:,} aged 3+ ({hli / p3:.2%}); population {pop:,}")

    rows = []
    for r in recs:
        for lang, v in r["langs"].items():
            if v > 0:
                rows.append({"geo_id": r["geo_id"], "geo_level": "municipio",
                             "geo_name": r["geo_name"], "source_category": lang, "count": v})
    df = pd.DataFrame(rows)
    NORM.mkdir(parents=True, exist_ok=True)
    df.to_csv(NORM / "mx.csv", index=False, encoding="utf-8")
    st = pd.DataFrame([{"geo_id": r["geo_id"], "geo_name": r["geo_name"],
                        "pop": int(it.at[r["geo_id"], "POBTOT"]), "p3": r["p3"], "hli": r["hli"],
                        "no_hli": r["no_hli"], "ne": r["ne"]} for r in recs])
    st.to_csv(NORM / "mx_status.csv", index=False, encoding="utf-8")
    nat = df.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(f"  wrote mx.csv ({len(df):,} rows) and mx_status.csv; national by label:")
    for k, v in nat.items():
        print(f"    {v:>10,}  {k}")
    print(f"  aged 3+ not saying whether they speak one: {st['ne'].sum():,}; "
          f"under 3: {pop - p3:,}")


if __name__ == "__main__":
    main()
