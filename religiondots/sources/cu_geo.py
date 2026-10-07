"""Cuba: the 16 provinces (15 and the special municipality of Isla de la Juventud), COD-AB polygons
with ONEI's own population for 31 December 2024.

Writes data/geo/cu/cu_provinces.gpkg and data/geo/cu/cu_lookup.csv, and for the placement layer
data/geo/cu/cu_municipalities.csv and .gpkg (ONEI's 2023 figure on each of COD-AB's 168
municipalities). `sources/cu.md` §4 is the record.

  * **boundaries**: COD-AB `cod-ab-cub` v01 (OCHA, from GADM; valid from 2023-10-03),
    `cub_admin1.geojson`: CU01-CU16, the provinces of the 2011 reform (Artemisa and Mayabeque).
  * **population**: ONEI, *Anuario Demográfico de Cuba 2024* (edición julio 2025), Tabla 1.5,
    *Población efectiva según provincia por zona urbana y rural*, the 2024 column: 9,748,007.
    "Población efectiva" is ONEI's measure since 2021 of the people actually living in Cuba,
    without those who have been abroad for long stays. Read from the PDF on every run and asserted
    against `ONEI_2024`.
  * **not used, and why**: COD-PS `cod-ps-cub` 2024 (UNFPA, ONEI's 2015-2050 projection from the
    2012 census) gives 11,306,203, 16.0% above ONEI's own count for the same year, because the
    projection assumed emigration would fall to zero. The office's count is the base (spec §12, "A
    COD-PS projection that agrees nationally can be wildly wrong per unit"; here it does not even
    agree nationally). COD-PS is printed per province as a witness.

THE JOIN is by name inside a fixed list of 16, folded, with ONEI's density table (Tabla 1.6, area =
2024 people / 2024 density) as the witness that neither key decides: each COD polygon's area must
be within `AREA_TOL` of ONEI's area for the same province.

Usage:
    python sources/cu_geo.py --fetch    COD-AB geojson zip, COD-PS and the yearbook into data/raw/cu/
    python sources/cu_geo.py            rebuild from data/raw/cu/
"""

import io
import os
import re
import sys
import unicodedata
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "6")

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "cu")
GEO = os.path.join(ROOT, "data", "geo", "cu")
OUT = os.path.join(GEO, "cu_provinces.gpkg")
LOOKUP = os.path.join(GEO, "cu_lookup.csv")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
_DS = "https://data.humdata.org/dataset/"
COD_AB_URL = (_DS + "32b4ba2e-2ee5-4b2b-a7e4-9e4d323ffe73/resource/b9c9175f-b974-419b-8f2b-ef2ac55221fb/"
              "download/cub_admin_boundaries.geojson.zip")
COD_AB = os.path.join(RAW, "cub_admin_boundaries.geojson.zip")
COD_PS_URL = (_DS + "b947bfba-cbde-454c-8ea8-3fd6ac075007/resource/3e055abd-7319-4821-bf8b-a620c372feb1/"
              "download/cub_admpop_2024.xlsx")
COD_PS = os.path.join(RAW, "cub_admpop_2024.xlsx")
# onei.gob.cu serves an expired certificate; the gender portal on the same office's host carries
# the same file (7,936,873 bytes, %%EOF present, fetched 2026-10-03).
ANUARIO_URL = "https://www.genero.onei.gob.cu/static/documents/informes/00-anuario-demografico-2024.pdf"
ANUARIO = os.path.join(RAW, "anuario_demografico_2024.pdf")
TABLE_PAGE = 20             # 0-based: book p.18, Tabla 1.5
DENSITY_PAGE = 21           # book p.19, Tabla 1.6

NATIONAL_2024 = 9_748_007
# ONEI Tabla 1.5, 2024 total column, by COD-AB p-code. Read 2026-10-03; asserted against the PDF.
ONEI_2024 = {
    "CU13": ("Pinar del Río", 515_208), "CU01": ("Artemisa", 452_430),
    "CU09": ("La Habana", 1_749_964), "CU12": ("Mayabeque", 330_260),
    "CU11": ("Matanzas", 619_159), "CU16": ("Villa Clara", 665_447),
    "CU04": ("Cienfuegos", 342_709), "CU14": ("Sancti Spíritus", 404_037),
    "CU03": ("Ciego de Ávila", 376_919), "CU02": ("Camagüey", 653_203),
    "CU10": ("Las Tunas", 475_343), "CU07": ("Holguín", 911_674),
    "CU05": ("Granma", 749_289), "CU15": ("Santiago de Cuba", 963_915),
    "CU06": ("Guantánamo", 465_429), "CU08": ("Isla de la Juventud", 73_021),
}
# ONEI's municipal dashboard (Tablero municipal, publicaciones/2025-05/3-tablero-municipal.xlsx,
# Wayback 20250606173952), sheet Base, rows `Municipios`, year 2023: 167 municipalities whose
# province sums equal Tabla 1.5's 2023 column exactly. Isla de la Juventud is not in it; it is one
# municipality and takes Tabla 1.5's 2023 figure. Used only to calibrate Kontur inside each
# province (sources/cu_grid.py), never as a count.
TABLERO_URL = ("https://web.archive.org/web/20250606173952id_/https://www.onei.gob.cu/sites/default/"
               "files/publicaciones/2025-05/3-tablero-municipal.xlsx")
TABLERO = os.path.join(RAW, "tablero_municipal_2025-05.xlsx")
MUNIS = os.path.join(GEO, "cu_municipalities.csv")
# Folded (province, workbook name) -> COD-AB admin2 p-code, where the names do not fold equal.
# The first four are spellings (COD's "Ciro Redodo", "Ciefuegos", "Frak País"; ONEI drops "La" in
# two Havana names). The two Mayabeque rows are joined on ONEI's own DPA code, not the name: the
# workbook labels 2407 "Güines" and 2408 "Alquizar" (Alquízar is Artemisa's 2208), while every
# other Mayabeque row follows the DPA order in which 2407 is San Nicolás and 2408 Güines; Kontur
# agrees (2407's 18,555 against San Nicolás's 24,394 Kontur people, 2408's 56,925 against Güines's
# 76,752, both at the national ratio). Measured 2026-10-03.
MUNI_PINNED = {("ciegodeavila", "ciroredondo"): "CU0305", ("cienfuegos", "cienfuegos"): "CU0403",
               ("holguin", "frankpais"): "CU0707", ("lahabana", "habanadeleste"): "CU0908",
               ("lahabana", "habanavieja"): "CU0909"}
MUNI_BY_CODE = {"2407": "CU1210", "2408": "CU1203"}
ISLA = "CU0801"
AREA_TOL = 0.09            # COD/ONEI area; measured 0.940-1.084 outside La Habana (sources/cu.md §4)
# GADM's La Habana is 819 km2 against ONEI's 728, while its neighbours Artemisa and Mayabeque are
# 125 and 70 km2 short. The mix drawn is national in every province, so a misdrawn line moves dots
# of the same colour between provinces; sources/cu_grid.py prints each province's Kontur people
# against ONEI, which is where the cost would show. Measured 2026-10-03.
AREA_PINNED = {"CU09": (1.10, 1.15)}
METRIC_AREA = "ESRI:54034"


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    return re.sub(r"[^a-z]", "", s)


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for url, dst, magic in ((COD_AB_URL, COD_AB, b"PK"), (COD_PS_URL, COD_PS, b"PK"),
                            (ANUARIO_URL, ANUARIO, b"%PDF"), (TABLERO_URL, TABLERO, b"PK")):
        if os.path.exists(dst) and os.path.getsize(dst) > 10_000:
            print(f"  have {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")
            continue
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        import ssl
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=600, context=ctx) as r:
            data = r.read()
        if not data.startswith(magic):
            raise SystemExit(f"{url} did not return {magic!r} (starts {data[:16]!r})")
        if magic == b"%PDF" and b"%%EOF" not in data[-1024:]:
            raise SystemExit(f"{url} is truncated (no %%EOF trailer)")
        with open(dst + ".part", "wb") as fh:
            fh.write(data)
        os.replace(dst + ".part", dst)
        print(f"  got  {os.path.basename(dst)} ({len(data):,} bytes)")


def _num(s):
    s = s.strip().replace(" ", " ")
    if s in ("-", ""):
        return 0.0
    return float(s.replace(" ", "").replace(",", "."))


def read_anuario():
    """{fold(name): (2024 people, 2024 density)} from Tablas 1.5 and 1.6."""
    import fitz

    doc = fitz.open(ANUARIO)
    names = {fold(n): n for n, _p in ONEI_2024.values()}

    def rows(page, width):
        lines = [x.strip() for x in doc[page].get_text().split("\n")]
        out, i = {}, 0
        while i < len(lines):
            key = fold(lines[i])
            if key == "santiagode" and i + 1 < len(lines) and fold(lines[i + 1]) == "cuba":
                key, i = "santiagodecuba", i + 1
            if key == "isladela" and i + 1 < len(lines) and fold(lines[i + 1]) == "juventud":
                key, i = "isladelajuventud", i + 1
            if (key in names or key == "cuba") and key not in out:
                out[key] = [_num(v) for v in lines[i + 1:i + 1 + width]]
                i += width
            i += 1
        return out

    t15 = rows(TABLE_PAGE, 6)          # 2023 total, urban, rural; 2024 total, urban, rural
    t16 = rows(DENSITY_PAGE, 5)        # density 2020-2024
    if "1.5 Población" not in doc[TABLE_PAGE].get_text() or "1.6 Extensión" not in doc[DENSITY_PAGE].get_text():
        raise SystemExit("the yearbook's Tablas 1.5 and 1.6 are not on the pinned pages")
    if set(t15) != set(names) | {"cuba"} or set(t16) != set(names) | {"cuba"}:
        raise SystemExit(f"Tabla 1.5 rows {sorted(t15)}; Tabla 1.6 rows {sorted(t16)}")
    if int(t15["cuba"][3]) != NATIONAL_2024:
        raise SystemExit(f"Tabla 1.5's 2024 total is {t15['cuba'][3]:,.0f}, pinned {NATIONAL_2024:,}")
    prov = {k: v for k, v in t15.items() if k != "cuba"}
    if int(sum(v[3] for v in prov.values())) != NATIONAL_2024:
        raise SystemExit("Tabla 1.5's provinces do not sum to its national row")
    for k, v in prov.items():
        if abs(v[4] + v[5] - v[3]) > 0.5:
            raise SystemExit(f"{names[k]}: urban + rural != total in 2024")
    return {k: (int(v[3]), t16[k][4], int(v[4]), int(v[5]), int(v[0])) for k, v in prov.items()}


def read_municipalities(a2, an):
    """DataFrame adm2_pcode, adm1_pcode, name, pop2023: ONEI's 2023 figure on every COD-AB admin2."""
    import warnings

    buf = io.BytesIO()
    with zipfile.ZipFile(TABLERO) as z, zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as out:
        for n in z.namelist():
            if n != "xl/styles.xml":            # openpyxl is slow on the stylesheet
                out.writestr(n, z.read(n))
    buf.seek(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = pd.read_excel(buf, sheet_name="Base", header=None, engine="openpyxl")
    hdr = next(i for i in range(30) if "Territorios" in raw.iloc[i].astype(str).tolist())
    d = raw.iloc[hdr + 1:].copy()
    d.columns = [str(c).strip() for c in raw.iloc[hdr]]
    year = d.columns[0]
    q = d[(d["Territorios"] == "Municipios") & (d[year].astype(str) == "2023")].copy()
    q["pop"] = pd.to_numeric(q["Población total"]).astype(int)
    q["prov"] = q["Provincias"].astype(str).str.split("_", n=1).str[1].str.strip()
    q["code"] = q["Municipios"].astype(str).str.split(" ", n=1).str[0]
    q["muni"] = q["Municipios"].astype(str).str.split(" ", n=1).str[1].str.strip()
    if len(q) != 167 or q["code"].nunique() != 167:
        raise SystemExit(f"the dashboard's 2023 rows: {len(q)} municipalities, {q['code'].nunique()} codes")
    for k, s in q.groupby(q["prov"].map(fold))["pop"].sum().items():
        if int(s) != an[k][4]:
            raise SystemExit(f"{k}: the dashboard's 2023 municipalities sum to {int(s):,}, Tabla 1.5 {an[k][4]:,}")

    a2 = a2.assign(kp=a2["adm1_name"].map(fold), km=a2["adm2_name"].map(fold))
    key = dict(zip(zip(a2["kp"], a2["km"]), a2["adm2_pcode"]))
    pc = []
    for p, m, c in zip(q["prov"].map(fold), q["muni"].map(fold), q["code"]):
        pc.append(MUNI_BY_CODE.get(c) or MUNI_PINNED.get((p, m)) or key.get((p, m)))
    q["adm2_pcode"] = pc
    if q["adm2_pcode"].isna().any() or q["adm2_pcode"].nunique() != 167:
        raise SystemExit(f"municipalities not joined 1:1: {q.loc[q['adm2_pcode'].isna(), ['prov', 'muni']].values.tolist()}")
    parent = dict(zip(a2["adm2_pcode"], a2["adm1_pcode"]))
    wrong = [(m, p) for m, p, prov in zip(q["muni"], q["adm2_pcode"], q["prov"])
             if fold(ONEI_2024[parent[p]][0]) != fold(prov)]
    if wrong:
        raise SystemExit(f"municipalities joined to a polygon in another province: {wrong}")
    missing = sorted(set(a2["adm2_pcode"]) - set(q["adm2_pcode"]))
    if missing != [ISLA]:
        raise SystemExit(f"COD admin2 with no dashboard row: {missing}, expected only {ISLA}")
    q = q[["adm2_pcode", "code", "muni", "pop"]]
    q = pd.concat([q, pd.DataFrame([{"adm2_pcode": ISLA, "code": "4001", "muni": "Isla de la Juventud",
                                     "pop": an["isladelajuventud"][4]}])], ignore_index=True)
    q["adm1_pcode"] = q["adm2_pcode"].map(parent)
    renamed = [(n, a) for n, a, p in zip(q["muni"], q["adm2_pcode"], q["adm2_pcode"])
               if fold(n) != fold(dict(zip(a2["adm2_pcode"], a2["adm2_name"]))[a])]
    print(f"  ONEI 2023 municipalities: 168 on COD-AB admin2, {int(q['pop'].sum()):,} people; joined "
          f"off the name: {renamed}")
    return q.rename(columns={"pop": "pop2023"})


def main():
    import geopandas as gpd

    if "--fetch" in sys.argv or not all(os.path.exists(p) for p in (COD_AB, COD_PS, ANUARIO, TABLERO)):
        fetch()

    an = read_anuario()
    for pc, (name, pop) in ONEI_2024.items():
        if an[fold(name)][0] != pop:
            raise SystemExit(f"{name}: the yearbook reads {an[fold(name)][0]:,}, pinned {pop:,}")
    print(f"ONEI 2024 (población efectiva), 16 provinces: {NATIONAL_2024:,}, every pinned figure read back")

    with zipfile.ZipFile(COD_AB) as z:
        g = gpd.read_file(io.BytesIO(z.read("cub_admin1.geojson")))
        a2 = gpd.read_file(io.BytesIO(z.read("cub_admin2.geojson")))
    if len(a2) != 168:
        raise SystemExit(f"COD-AB admin2 has {len(a2)} municipalities, expected 168")
    munis = read_municipalities(a2, an)
    if len(g) != 16 or set(g["adm1_pcode"]) != set(ONEI_2024):
        raise SystemExit(f"COD-AB admin1 is not CU01-CU16: {sorted(g['adm1_pcode'])}")
    bad = [(p, n) for p, n in zip(g["adm1_pcode"], g["adm1_name"]) if fold(n) != fold(ONEI_2024[p][0])]
    if bad:
        raise SystemExit(f"COD-AB names that are not the pinned province for their p-code: {bad}")

    ps = pd.read_excel(COD_PS, sheet_name="cub_admpop_adm1_2024")
    ps = dict(zip(ps["ADM1_PCODE"], ps["T_TL"].astype(int)))

    g["unit"] = g["adm1_pcode"]
    g["name"] = [ONEI_2024[p][0] for p in g["unit"]]
    g["pop"] = [ONEI_2024[p][1] for p in g["unit"]]
    area = g.to_crs(METRIC_AREA).area / 1e6
    print("\n  province, ONEI 2024, urban share, COD-PS 2024 / ONEI, COD km2 / ONEI km2:")
    worst = 0.0
    for (_i, r), a in sorted(zip(g.iterrows(), area), key=lambda t: t[0][1]["unit"]):
        pop, dens, urb, _rur, _p23 = an[fold(r["name"])]
        onei_km2 = pop / dens
        rel = a / onei_km2
        if r["unit"] in AREA_PINNED:
            lo, hi = AREA_PINNED[r["unit"]]
            if not lo <= rel <= hi:
                raise SystemExit(f"{r['name']}'s area ratio {rel:.3f} left its pinned band {lo}-{hi}")
        else:
            worst = max(worst, abs(rel - 1))
        print(f"      {r['unit']}  {r['name']:<20} {pop:>10,}  {urb / pop:6.1%}  "
              f"{ps[r['unit']] / pop:5.3f}  {a:>8,.0f} / {onei_km2:>8,.0f} = {rel:5.3f}")
    print(f"  COD-PS 2024 nationally: {sum(ps.values()):,}, {sum(ps.values()) / NATIONAL_2024:.3f}x ONEI")
    if worst > AREA_TOL:
        raise SystemExit(f"a province's COD area is {worst:.1%} off ONEI's; the join or the polygon is wrong")
    print(f"  area witness: every province within {worst:.1%} of ONEI's area (bar {AREA_TOL:.0%}), "
          f"La Habana inside its pinned band")

    os.makedirs(GEO, exist_ok=True)
    g[["unit", "name", "pop", "geometry"]].to_file(OUT, layer="provinces", driver="GPKG")
    lut = g[["unit", "name", "pop"]].rename(columns={"unit": "geo_id"})
    lut["unit"] = lut["geo_id"]
    lut.sort_values("geo_id").to_csv(LOOKUP, index=False, encoding="utf-8")
    munis.sort_values("adm2_pcode").to_csv(MUNIS, index=False, encoding="utf-8")
    a2[["adm2_pcode", "adm1_pcode", "adm2_name", "geometry"]].to_file(
        os.path.join(GEO, "cu_municipalities.gpkg"), layer="municipalities", driver="GPKG")
    print(f"\nwrote {OUT} and {LOOKUP} (16 provinces, {int(g['pop'].sum()):,} people), "
          f"{MUNIS} and cu_municipalities.gpkg (168)")


if __name__ == "__main__":
    main()
