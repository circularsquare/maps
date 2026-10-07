"""The smallest pieces of territory the COD 2018 boundaries can give us.

COD (HDX cod-ab-kgz, valid 2018-11-19) has 9 oblasts / 53 rayons and cities /
737 third-level units, but the third level does not cover everything:
  - Bishkek and Osh have no units below the oblast level at all;
  - a city of oblast significance has only its urban-type settlements (pgt) as
    third-level units, so the city itself is a hole in the third level;
  - 277 third-level units have no name and codes ending 9xx: pasture, forest
    and reserve land outside any aiyl aimak.
An atom is a third-level unit, the remainder of a city polygon once its pgt are
cut out, or a whole capital. Every later level is a dissolve of atoms.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
import geopandas as gpd
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
COD = os.path.join(REPO, "data", "asia1m", "kyrgyzstan", "kgz_admin{}.shp")

CAPITALS = {"KG11000000000", "KG21000000000"}
# COD codes whose SOATE differs from "417" + the 11 digits
COD_TO_SOATE = {
    "KG04400000010": "41704000000010",   # Naryn city: SOATE has no 4xx segment
}


def soate(cod):
    return COD_TO_SOATE.get(cod, "417" + cod[2:])


def load_cod():
    g1 = gpd.read_file(COD.format(1)).to_crs("EPSG:4326")
    g2 = gpd.read_file(COD.format(2)).to_crs("EPSG:4326")
    g3 = gpd.read_file(COD.format(3)).to_crs("EPSG:4326")
    return g1, g2, g3


def build_atoms():
    g1, g2, g3 = load_cod()
    rows = []
    for r in g3.itertuples():
        c = r.adm3_pcode
        if c[7] == "8":
            kind = "aa"
        elif c[7] in "456":
            kind = "town"
        else:
            kind = "land"                 # 000 9xx, named or not
        rows.append(dict(atom=c, kind=kind, name=r.adm3_name, name_ru=r.adm3_name1,
                         cod_adm2=r.adm2_pcode, cod_adm1=r.adm1_pcode, geometry=r.geometry))
    # cities of oblast significance (and Kok-Zhangak): polygon minus its pgt atoms
    cities = g2[g2.adm2_pcode.str[4] == "4"]
    cities = pd.concat([cities, g2[g2.adm2_pcode == "KG03220400010"]])
    for r in cities.itertuples():
        kids = g3[g3.adm2_pcode == r.adm2_pcode]
        geom = r.geometry
        if len(kids):
            geom = geom.difference(kids.union_all())
        rows.append(dict(atom=r.adm2_pcode, kind="city", name=r.adm2_name, name_ru=r.adm2_name1,
                         cod_adm2=r.adm2_pcode, cod_adm1=r.adm1_pcode, geometry=geom))
    for r in g1[g1.adm1_pcode.isin(CAPITALS)].itertuples():
        rows.append(dict(atom=r.adm1_pcode, kind="capital", name=r.adm1_name, name_ru=r.adm1_name1,
                         cod_adm2=r.adm1_pcode, cod_adm1=r.adm1_pcode, geometry=r.geometry))
    a = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    a["soate"] = a.atom.map(soate)
    return a
