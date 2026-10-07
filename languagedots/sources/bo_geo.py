"""Bolivia placement layer: Kontur 400 m hexes keyed to the 339 municipalities of COD-AB
-> data/geo/bo/bo_hexes.gpkg and data/geo/bo/bo_lookup.csv (census area -> unit).

    python sources/bo_geo.py          (needs data/raw/bo/bo_munic_pop.csv from sources/bo_censo.py)

WHY NOT RELIGIONDOTS' LAYER. religiondots draws Bolivia from a survey at department level, so its
hexes are keyed to the 9 departments. The language table is per municipality, so the
municipalities come from the same COD-AB release religiondots downloaded (OCHA cod-ab-bol v02,
valid from 2024-09-16; ../religiondots/data/raw/bo/shp/bol_admin3.shp, read in place) and the
Kontur extract religiondots downloaded (../religiondots/data/raw/bo/, the .gz copied into
languagedots' data/geo/kontur/ so sources/_grid.py finds it). Nothing is written into
religiondots.

THE JOIN. COD-AB's adm3 p-codes are INE's six-digit municipality codes with "BO" in front, in
INE's order, so the join is on code: all 339 COD municipalities are census areas. Names are
compared beside the code and the differences printed (long official names, "Sopachui" against
"Sopachuy" and the like); none is a different place.

FOUR CENSUS AREAS HAVE NO POLYGON. The 2024 census counts three indigenous territories (TIOC)
and one new municipality on their own; each is folded into the municipality it was carved
from, which COD-AB's polygon still covers:
  031304 TIOC-Raqaypampa             -> 031301 Mizque (Raqaypampa was Mizque's fifth district)
  050405 San Pedro de Macha          -> 050401 Colquechaca (Macha was a canton of Colquechaca)
  051204 TIOC-Jatun Ayllu Yura       -> 051202 Tomave (Yura was a canton of Tomave)
  080901 TIOC-Territorio Indigena Multietnico -> 080501 San Ignacio de Moxos. Ley 1497 of
         2023 carved it from San Ignacio de Moxos and Santa Ana de Yacuma; no split is
         published, and religiondots (sources/bo_geo.py there) puts it in Moxos too.
"""
import os
import shutil
import sys
import unicodedata
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sources"))
from _grid import OUR_KONTUR, hex_layer  # noqa: E402

RD_RAW = ROOT.parent / "religiondots" / "data" / "raw" / "bo"
SHP = RD_RAW / "shp" / "bol_admin3.shp"
KONTUR_GZ = "kontur_population_BO_20231101.gpkg.gz"
POP = ROOT / "data" / "raw" / "bo" / "bo_munic_pop.csv"
OUT = ROOT / "data" / "geo" / "bo"
LOOKUP = OUT / "bo_lookup.csv"

EXPECTED = 339
FOLD = {"031304": "031301", "050405": "050401", "051204": "051202", "080901": "080501"}


def _n(s):
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()


def main():
    ok = True

    def report(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    g = gpd.read_file(SHP)
    g["unit"] = g["adm3_pcode"].str[2:]
    report(len(g) == EXPECTED and g["unit"].is_unique and g["unit"].str.fullmatch(r"\d{6}").all(),
           f"COD-AB adm3: {len(g)} municipalities, unique six-digit codes")

    pop = pd.read_csv(POP, dtype={"geo_id": str})
    census = set(pop["geo_id"])
    cod = set(g["unit"])
    report(cod <= census and census - cod == set(FOLD),
           f"every COD municipality is a census area, and the census's extras are exactly the "
           f"four folded ones ({sorted(census - cod)})")
    report(set(FOLD.values()) <= cod, "every fold target is a COD municipality")

    names = dict(zip(g["unit"], g["adm3_name"]))
    diff = [(c, n, names[c]) for c, n in zip(pop["geo_id"], pop["name"])
            if c in names and _n(n) != _n(names[c])]
    print(f"  {len(diff)} name differences on a matching code (printed for the record):")
    for c, a, b in diff:
        print(f"      {c}  census {a!r}  COD {b!r}")

    pop["unit"] = pop["geo_id"].map(lambda c: FOLD.get(c, c))
    OUT.mkdir(parents=True, exist_ok=True)
    pop[["geo_id", "unit", "name", "pop"]].to_csv(LOOKUP, index=False, encoding="utf-8")
    per_unit = pop.groupby("unit")["pop"].sum()
    report(int(per_unit.sum()) == 11_365_333 and len(per_unit) == EXPECTED,
           f"{len(per_unit)} units hold {int(per_unit.sum()):,} people")
    for c, into in FOLD.items():
        print(f"      fold {c} {pop.loc[pop['geo_id'] == c, 'name'].iloc[0]} "
              f"({int(pop.loc[pop['geo_id'] == c, 'pop'].iloc[0]):,} people) -> {into} "
              f"{names[into]}")

    OUR_KONTUR.mkdir(parents=True, exist_ok=True)
    gz = OUR_KONTUR / KONTUR_GZ
    if not gz.exists() and not (OUR_KONTUR / KONTUR_GZ[:-3]).exists():
        shutil.copyfile(RD_RAW / KONTUR_GZ, gz)
        print(f"  copied {RD_RAW / KONTUR_GZ} -> {gz}")

    layer = hex_layer("bo", g[["unit", "geometry"]], census=per_unit.to_dict())
    k = layer.groupby("unit")["pop"].sum()
    ratio = k.sum() / per_unit.sum()
    print("  Kontur / census, normalised, for the four fold targets:")
    for into in sorted(set(FOLD.values())):
        print(f"      {into} {names[into]}: {k.get(into, 0) / per_unit[into] / ratio:.2f}")
    if not ok:
        raise SystemExit("bo_geo: checks FAILED")


if __name__ == "__main__":
    main()
