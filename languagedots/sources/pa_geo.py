"""Panama placement layer: Kontur 2023-11 400 m hexes keyed to units built from the 2023
census's 82 distritos -> data/geo/pa/pa_hexes.gpkg and data/geo/pa/pa_units.csv.

    python sources/pa_geo.py          (needs data/normalized/pa_units.csv from sources/pa_censo.py)

WHY NOT CORREGIMIENTOS. The 2023 census has 699 corregimientos; the newest boundaries at hand,
OCHA COD-AB (valid_on 2021-10-20, religiondots' download, read-only), have 594 and 76
districts, and number provinces alphabetically (COD 02 is Chiriqui, INEC 02 Cocle), so neither
codes nor a full name join work: about 110 corregimientos and several districts were created
after COD's vintage.

THE UNITS. Each census corregimiento is looked up by folded name ("(CABECERA)" dropped) among
COD corregimientos of the same province (provinces matched by name, PROV below). A hit ties
the census corregimiento's DISTRICT to the COD corregimiento's DISTRICT. A unit is a connected
group of census districts and COD districts tied this way: mostly one to one; where a new
district was carved from an old one, both census districts join the old COD district. Every
census district and every COD district must land in some unit, and the census counts are summed
to units. The unit id is the lowest census district code in it (`pa_units.csv` maps every census
district to its unit).

CHECKS: every census district ties to at least one COD district; every COD district is in a
unit; units cover the census population exactly; Kontur against the census per unit (band and
shuffled-join control, printed by hex_layer).
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

SHP3 = RD / "data" / "raw" / "pa" / "shp" / "pan_admin3.shp"
UNITS = HERE / "data" / "normalized" / "pa_units.csv"
OUT_UNITS = HERE / "data" / "geo" / "pa" / "pa_units.csv"

# INEC province code -> COD adm1_name
PROV = {"01": "Bocas del Toro", "02": "Coclé", "03": "Colón", "04": "Chiriquí", "05": "Darién",
        "06": "Herrera", "07": "Los Santos", "08": "Panamá", "09": "Veraguas", "10": "Kuna Yala",
        "11": "Emberá", "12": "Ngöbe Buglé", "13": "Panamá Oeste"}


def fold(s):
    s = re.sub(r"\([^)]*\)", "", str(s))          # (CABECERA), and Ngabe names in brackets
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def main():
    import geopandas as gpd
    from _grid import hex_layer

    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'OK ' if cond else 'BAD'} {msg}")

    g = gpd.read_file(SHP3)
    check(set(PROV.values()) == set(g["adm1_name"]), "the 13 COD provinces match PROV by name")
    u = pd.read_csv(UNITS, dtype={"geo_id": str})
    u["dist"] = u["geo_id"].str[:4]
    look = {}
    for r in g.itertuples():
        look.setdefault((r.adm1_name, fold(r.adm3_name)), set()).add(r.adm2_pcode)
    # a name found in one COD district ties; a name found in several (San Juan, El Espino) ties
    # only if its census district has no unique tie at all
    ties, amb, hit = set(), {}, 0
    for r in u.itertuples():
        d = look.get((PROV[r.geo_id[:2]], fold(r.geo_name)))
        if d:
            hit += 1
            if len(d) == 1:
                ties.add((r.dist, next(iter(d))))
            else:
                amb.setdefault(r.dist, set()).update(d)
    for dist, ds in amb.items():
        if dist not in {a for a, _ in ties}:
            ties.update((dist, x) for x in ds)
    print(f"     {hit} of {len(u)} census corregimientos found by name in COD")
    cdist, gdist = set(u["dist"]), set(g["adm2_pcode"])
    lone_c = sorted(cdist - {a for a, _ in ties})
    lone_g = sorted(gdist - {b for _, b in ties})
    check(not lone_c, f"every census district ties to a COD district ({lone_c})")
    check(not lone_g, f"every COD district ties to a census district ({lone_g})")
    # connected components
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in ties:
        parent[find("c" + a)] = find("g" + b)
    comp = {}
    for x in list(parent):
        comp.setdefault(find(x), set()).add(x)
    units = []
    for members in comp.values():
        cs = sorted(m[1:] for m in members if m[0] == "c")
        gs = sorted(m[1:] for m in members if m[0] == "g")
        units.append((cs[0], cs, gs))
    multi = [x for x in units if len(x[1]) > 1 or len(x[2]) > 1]
    print(f"     {len(units)} units; {len(multi)} join more than one district:")
    for uid, cs, gs in sorted(multi):
        print(f"       {uid}: census {cs} <-> COD {gs}")
    if not ok:
        raise SystemExit("pa_geo: join checks FAILED, nothing written")
    dist_unit = {c: uid for uid, cs, _ in units for c in cs}
    cod_unit = {x: uid for uid, _, gs in units for x in gs}
    OUT_UNITS.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sorted(dist_unit.items()), columns=["dist", "unit"]).to_csv(OUT_UNITS, index=False)
    u["unit"] = u["dist"].map(dist_unit)
    census = u.groupby("unit")["population"].sum().to_dict()
    check(sum(census.values()) == u["population"].sum(), "units cover the census population")
    g["unit"] = g["adm2_pcode"].map(cod_unit)
    poly = g.dissolve(by="unit", as_index=False)[["unit", "geometry"]]
    layer = hex_layer("pa", poly, census=census)
    per = layer.groupby("unit")["pop"].sum()
    check(set(per.index) == set(census) and (per > 0).all(), "every unit has a populated hex")
    if not ok:
        raise SystemExit("pa_geo: FAILED (the layer was written; do not use it)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
