"""Honduras — boundaries and populations for the 18 departments.

Writes data/geo/hn/hn_departamentos.gpkg, data/geo/hn/hn_lookup.csv and
data/geo/hn/hn_municipios.csv (the municipality names `sources/hn.py` uses to prove the LAPOP
decode). `sources/hn.md` is this country's record.

OCHA COD-AB Honduras (`cod-ab-hnd`), the shapefile bundle read with `engine="fiona"`, as
`sources/do_geo.py` does. **ADM1 here really is the 18 departments** (asserted), unlike the
Dominican Republic, where COD's ADM1 is the planning regions.

## THE POPULATION IS INE'S 2024 PROJECTION, BECAUSE NOTHING NEWER THAN 2013 WAS COUNTED

Honduras last enumerated in 2013; the Censo 2026 is in the field as this is written and has
published nothing. `cod-ps-hnd`'s 2024 table names the Instituto Nacional de Estadística as its
source, so this is the office's own projection and not a UN one, and there is no later count
to check it against per department the way `sources/do_geo.py` checked the Dominican COD-PS.
The adm0 file's national total is asserted against the sum of the 18 department rows.

## THE CODE JOIN AND THE NAME JOIN AGREE HERE, AND BOTH ARE ASSERTED

ENDESA-MICS 2019's `HH7` numbers the departments 1 to 18 in INE's official order (Atlántida,
Colón, Comayagua, Copán, Cortés, Choluteca, ...), which is not alphabetical: Choluteca comes
after Cortés. COD's pcodes follow the same official order, `HN01` Atlántida to `HN18` Yoro, so
`HN%02d % hh7` is correct. That is Costa Rica's situation (§9cp) rather than El Salvador's,
and the response is the same: join on the NAME and assert that the code join gives the same
answer for all eighteen, so a re-cut of either numbering stops the build.

`HH7` 19 and 20 are San Pedro Sula and the Distrito Central, sampled as their own domains and
**inside** Cortés and Francisco Morazán respectively; `sources/hn.py` merges them back.

Usage:
    python sources/hn_geo.py --fetch    a ~69 MB zip and two small csvs from HDX
    python sources/hn_geo.py            rebuild from data/raw/hn/
"""

import os
import sys
import unicodedata
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "hn")
SHP_DIR = os.path.join(RAW, "shp")
OUT_DIR = os.path.join(ROOT, "data", "geo", "hn")
OUT = os.path.join(OUT_DIR, "hn_departamentos.gpkg")
LOOKUP = os.path.join(OUT_DIR, "hn_lookup.csv")
MUNIS = os.path.join(OUT_DIR, "hn_municipios.csv")
HH_SAV = os.path.join(RAW, "endesa", "Bases de datos", "hh.sav")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}

N_DEPT = 18
N_MUNI = 298

DOWNLOADS = {
    # OCHA COD-AB, https://data.humdata.org/dataset/cod-ab-hnd
    "hnd_admin_boundaries.shp.zip":
        "https://data.humdata.org/dataset/bd62fb53-64d3-478f-9ca8-38e4a2de19c0/resource/"
        "4b3848a6-cbc3-406d-9640-13757d412b25/download/hnd_admin_boundaries.shp.zip",
    # OCHA COD-PS, https://data.humdata.org/dataset/cod-ps-hnd, source INE, 2024
    "hnd_admpop_adm1_2024.csv":
        "https://data.humdata.org/dataset/6203b8df-ef66-4e55-baba-6d8547438a77/resource/"
        "6fb7fbab-b923-46b2-9645-152051371b7b/download/hnd_admpop_adm1_2024.csv",
    "hnd_admpop_adm0_2024.csv":
        "https://data.humdata.org/dataset/6203b8df-ef66-4e55-baba-6d8547438a77/resource/"
        "83925f2d-174e-4fef-ac89-fc4c882270a9/download/hnd_admpop_adm0_2024.csv",
}

# ENDESA-MICS 2019 `HH7`, verbatim from hh.sav's label set (capitals, no accents). 19 and 20
# are the two city domains and are not departments. Written out so a relabelled release fails.
ENDESA_HH7 = {
    1: "ATLANTIDA", 2: "COLON", 3: "COMAYAGUA", 4: "COPAN", 5: "CORTES", 6: "CHOLUTECA",
    7: "EL PARAISO", 8: "FRANCISCO MORAZAN", 9: "GRACIAS A DIOS", 10: "INTIBUCA",
    11: "ISLAS DE LA BAHIA", 12: "LA PAZ", 13: "LEMPIRA", 14: "OCOTEPEQUE", 15: "OLANCHO",
    16: "SANTA BARBARA", 17: "VALLE", 18: "YORO",
}
CITY_DOMAINS = {19: "SAN PEDRO SULA", 20: "DISTRITO CENTRAL"}

# Display names with their accents, for the lookup. COD-AB and COD-PS both drop them.
DISPLAY = {
    1: "Atlántida", 2: "Colón", 3: "Comayagua", 4: "Copán", 5: "Cortés", 6: "Choluteca",
    7: "El Paraíso", 8: "Francisco Morazán", 9: "Gracias a Dios", 10: "Intibucá",
    11: "Islas de la Bahía", 12: "La Paz", 13: "Lempira", 14: "Ocotepeque", 15: "Olancho",
    16: "Santa Bárbara", 17: "Valle", 18: "Yoro",
}


def fold(s):
    """Accent- and case-insensitive key for a name."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return "".join(ch for ch in s.lower() if ch.isalnum())


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for name, url in DOWNLOADS.items():
        dst = os.path.join(RAW, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 256:
            print(f"  have {name} ({os.path.getsize(dst):,} bytes)")
            continue
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=900) as r, open(dst + ".part", "wb") as f:
            f.write(r.read())
        os.replace(dst + ".part", dst)
        print(f"  got  {name} ({os.path.getsize(dst):,} bytes)")


def check_sav_labels():
    """Assert ENDESA_HH7 and CITY_DOMAINS against hh.sav's own label set, when it is on disk."""
    if not os.path.exists(HH_SAV):
        print("  (hh.sav not on disk; HH7 labels not cross-checked, run sources/hn.py --fetch)")
        return
    import pyreadstat

    _, meta = pyreadstat.read_sav(HH_SAV, metadataonly=True)
    got = {int(k): v for k, v in meta.variable_value_labels.get("HH7", {}).items()}
    want = {**ENDESA_HH7, **CITY_DOMAINS}
    if got != want:
        diff = {k: (got.get(k), want.get(k)) for k in set(got) | set(want)
                if got.get(k) != want.get(k)}
        raise SystemExit(f"HH7's value labels have changed: {diff}")
    print(f"  witness 0: all {len(got)} HH7 labels match hh.sav's own label set")


def main():
    if "--fetch" in sys.argv:
        fetch()

    zpath = os.path.join(RAW, "hnd_admin_boundaries.shp.zip")
    if not os.path.exists(zpath):
        raise SystemExit(f"{zpath} missing, run with --fetch")
    os.makedirs(SHP_DIR, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(SHP_DIR)

    g = gpd.read_file(os.path.join(SHP_DIR, "hnd_admin1.shp"), engine="fiona")
    if len(g) != N_DEPT:
        raise SystemExit(f"{len(g)} ADM1 features, expected {N_DEPT}")
    if g.crs is None or g.crs.to_epsg() != 4326:
        raise SystemExit(f"unexpected CRS {g.crs}")
    g["pcode"] = g["adm1_pcode"].astype(str).str.strip()
    by_name = dict(zip(g["adm1_name"].map(fold), g["pcode"]))
    if len(by_name) != N_DEPT:
        raise SystemExit("COD's department names are not unique, the name join is unsafe")
    print(f"read hnd_admin1.shp: {len(g)} departments, {g.crs}")

    check_sav_labels()

    # ---- witness 1: every ENDESA name is a COD name
    missing = [n for n in ENDESA_HH7.values() if fold(n) not in by_name]
    spare = [n for n in g["adm1_name"] if fold(n) not in {fold(v) for v in ENDESA_HH7.values()}]
    if missing or spare:
        raise SystemExit(f"the name join FAILED: ENDESA names with no polygon {missing}, "
                         f"polygons with no ENDESA name {spare}")
    print(f"  witness 1: all {N_DEPT} ENDESA names match a COD name, no aliases")

    # ---- witness 2: the code join agrees with the name join on all eighteen
    wrong = [(c, n) for c, n in ENDESA_HH7.items() if by_name[fold(n)] != f"HN{c:02d}"]
    if wrong:
        raise SystemExit(f"the code join HN%02d disagrees with the name join on {wrong}; "
                         "one numbering has been re-cut, decide deliberately")
    print(f"  witness 2: HN%02d of HH7 gives the same pcode as the name for all {N_DEPT}")

    # ---- the population: INE's 2024 projection, via COD-PS
    ps = pd.read_csv(os.path.join(RAW, "hnd_admpop_adm1_2024.csv"), encoding="utf-8-sig")
    if len(ps) != N_DEPT or set(ps["ADM1_PCODE"]) != set(g["pcode"]):
        raise SystemExit("COD-PS's department pcodes are not COD-AB's")
    ps_name = dict(zip(ps["ADM1_PCODE"], ps["ADM1_ES"].map(fold)))
    bad = [p for p in g["pcode"] if ps_name[p] != fold(g.loc[g["pcode"] == p, "adm1_name"].iloc[0])]
    if bad:
        raise SystemExit(f"COD-PS and COD-AB name these pcodes differently: {bad}")
    pop = dict(zip(ps["ADM1_PCODE"], ps["T_TL"].astype("int64")))
    if any(int(ps.loc[ps["ADM1_PCODE"] == p, "F_TL"].iloc[0])
           + int(ps.loc[ps["ADM1_PCODE"] == p, "M_TL"].iloc[0]) != pop[p] for p in pop):
        raise SystemExit("COD-PS rows where women plus men is not the total")
    nat = pd.read_csv(os.path.join(RAW, "hnd_admpop_adm0_2024.csv"), encoding="utf-8-sig")
    total = int(nat["T_TL"].iloc[0])
    if sum(pop.values()) != total:
        raise SystemExit(f"COD-PS departments sum to {sum(pop.values()):,}, the national row "
                         f"says {total:,}")
    print(f"  COD-PS 2024 (source INE): {N_DEPT} departments summing to {total:,}, and the "
          "national row agrees")
    g["pop"] = g["pcode"].map(pop).astype("int64")

    # Cortés (San Pedro Sula) is the densest; Gracias a Dios, the Mosquitia, is the emptiest.
    # A permuted population join is what this catches.
    g["density"] = g["pop"] / g["area_sqkm"]
    lo = g.loc[g["density"].idxmin(), "pcode"]
    hi = g.loc[g["density"].idxmax(), "pcode"]
    print(f"  densest {hi} ({g['density'].max():,.0f}/km2), sparsest {lo} "
          f"({g['density'].min():,.1f}/km2)")
    if (hi, lo) != ("HN05", "HN09"):
        raise SystemExit("Cortés is not the densest or Gracias a Dios not the sparsest; the "
                         "population join is permuted")

    code_of = {by_name[fold(n)]: c for c, n in ENDESA_HH7.items()}
    g["unit"] = g["pcode"]
    g["geo_id"] = g["pcode"]
    g["name"] = g["pcode"].map(lambda p: DISPLAY[code_of[p]])

    os.makedirs(OUT_DIR, exist_ok=True)
    g[["unit", "name", "pcode", "geo_id", "pop", "geometry"]].to_file(
        OUT, layer="departamentos", driver="GPKG")
    print(f"\nwrote {OUT} ({len(g)} polygons)")

    order = sorted(g["pcode"])
    lut = pd.DataFrame({
        "geo_id": order,
        "unit": order,
        "name": [DISPLAY[code_of[p]] for p in order],
        "pop_2024": [int(pop[p]) for p in order],
        "endesa_hh7": [code_of[p] for p in order],
        "area_sqkm": [float(g.loc[g["pcode"] == p, "area_sqkm"].iloc[0]) for p in order],
    })
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows)")

    # ---- the municipality names, for sources/hn.py's LAPOP decode witness
    m = gpd.read_file(os.path.join(SHP_DIR, "hnd_admin2.shp"), engine="fiona", ignore_geometry=True)
    if len(m) != N_MUNI:
        raise SystemExit(f"{len(m)} ADM2 features, expected {N_MUNI}")
    if not set(m["adm1_pcode"]) <= set(order):
        raise SystemExit("ADM2 rows whose department is not an ADM1 pcode")
    m[["adm2_pcode", "adm2_name", "adm1_pcode", "adm1_name"]].to_csv(MUNIS, index=False,
                                                                      encoding="utf-8")
    print(f"wrote {MUNIS} ({len(m)} municipalities)")


if __name__ == "__main__":
    main()
