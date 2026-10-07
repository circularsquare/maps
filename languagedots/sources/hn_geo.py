"""Honduras placement layer: Kontur 2023-11 400 m hexes keyed to the 298 municipios of the 2013
census -> data/geo/hn/hn_hexes.gpkg.

    python sources/hn_geo.py          (needs data/normalized/hn_units.csv from sources/hn_censo.py)

BOUNDARIES. OCHA COD-AB Honduras (valid_on 2016-10-05), admin2, read from religiondots' download
(religiondots/data/raw/hn/shp/, read-only; religiondots draws Honduras by department). 298
features, pcode HN + INE's four-digit municipio code.

THE JOIN IS ON NAME WITHIN DEPARTMENT, NOT ON CODE. Codes join 298 for 298, but the names show
COD numbers municipios alphabetically where INE does not: in Colon (0205-0208), Gracias a Dios
(0903-0906) and Santa Barbara (1606-1619, 1624-1625) the same code is a different municipio
(INE 0903 Juan Francisco Bulnes is COD's HN0904; INE 1606 San Jose de Colinas is COD's HN1619).
So each census code takes the COD polygon of the same folded name in the same department, with
the spellings pinned in NAME_PINNED read by hand; the match must be one-to-one, 298 for 298.
INE's code is the unit id throughout. The census names arrive with Cabanas's tilde mangled
("cabaaas"); pinned.

CHECKS: 298 units, the code and name join, the department witness, every municipio has a
populated hex, Kontur per municipio against the census (band and shuffled-join control, printed
by hex_layer).
"""
import os
import re
import sys
import unicodedata
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "sources"))
sys.path.insert(0, str(HERE))
from rdlink import RD  # noqa: E402

SHP = RD / "data" / "raw" / "hn" / "shp" / "hnd_admin2.shp"
UNITS = HERE / "data" / "normalized" / "hn_units.csv"
N = 298

# census code: COD's folded name, where the folded names differ (each looked at by hand)
NAME_PINNED = {
    "0402": "cabana",                     # Copan, Cabanas (census tilde mangled)
    "1203": "cabanas",                    # La Paz, Cabanas
    "1013": "san marcos de sierra",       # Intibuca, San Marcos de la Sierra
    "0906": "ramon villeda morales",      # Gracias a Dios
}


def fold(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def main():
    import geopandas as gpd
    from _grid import hex_layer

    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'OK ' if cond else 'BAD'} {msg}")

    g = gpd.read_file(SHP)
    check(len(g) == N, f"COD-AB admin2 has {len(g)} features")
    g["unit"] = g["adm2_pcode"].str[2:]
    check(g["unit"].is_unique, "pcodes unique")
    u = pd.read_csv(UNITS, dtype={"geo_id": str})
    cen, cod = set(u["geo_id"]), set(g["unit"])
    check(cen == cod, f"codes join both ways ({sorted(cen - cod)[:4]} census-only, "
                      f"{sorted(cod - cen)[:4]} COD-only)")
    key = {(r.adm1_pcode, fold(r.adm2_name)): r.adm2_pcode for r in g.itertuples()}
    poly = {}
    for r in u.itertuples():
        nm = NAME_PINNED.get(r.geo_id, fold(r.geo_name))
        poly[r.geo_id] = key.get(("HN" + r.geo_id[:2], nm))
    miss = sorted(c for c, p in poly.items() if p is None)
    check(not miss, f"every census municipio finds a same-name COD polygon in its department "
                    f"({miss})")
    check(len(set(poly.values())) == N, "the name join is one-to-one, 298 polygons")
    moved = sorted(c for c, p in poly.items() if p and p[2:] != c)
    print(f"     {len(moved)} census codes take a COD polygon of another code: {moved}")
    if not ok:
        raise SystemExit("hn_geo: join checks FAILED, nothing written")
    pc = {p: c for c, p in poly.items()}
    g["unit"] = g["adm2_pcode"].map(pc)

    census = dict(zip(u["geo_id"], u["population"]))
    layer = hex_layer("hn", g[["unit", "geometry"]], census=census)
    per = layer.groupby("unit")["pop"].sum()
    check(set(per.index) == cod and (per > 0).all(),
          f"every municipio has a populated hex ({int((per <= 0).sum())} do not)")
    if not ok:
        raise SystemExit("hn_geo: FAILED (the layer was written; do not use it)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
