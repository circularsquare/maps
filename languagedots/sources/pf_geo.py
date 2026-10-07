"""French Polynesia placement layer: Kontur 2023-11 400 m hexes keyed to the 48 communes, weighted
inside each commune by the 2017 census population of its communes associées and islands
-> data/geo/pf/pf_hexes.gpkg (unit = ISPF commune code IDCom, 11-58).

    python sources/pf_geo.py [--fetch]

BOUNDARIES. ISPF, "Limites géographiques administratives" on data.gouv.fr (ODbL;
https://www.data.gouv.fr/fr/datasets/limites-geographiques-administratives/), shapefiles.zip
(2022-06-10): Com (48 communes), Comas (116 communes associées), Ile (119 islands), each carrying
the 2017 census population (Indivds) and population 15+ (Id15aep). RGPF (EPSG:4687), no
antimeridian crossing (-154.7 to -134.5).

WHY NOT PLAIN KONTUR. Kontur runs 1.12x the 2017 census nationally, and spreads people over
atolls nobody lives on (Moruroa 159, Pinaki 93, Tikei 88, Nihiru 63; 0 in the 2017 census) and
reads 0.4-3.4x the census on inhabited atolls of one commune (Arutua's Kaukura 3.37, Mataiva
0.39). The census counts people per commune associée and per island, so inside each commune
(the unit the language table counts in) the weights are borrowed from those (AGENT_BRIEF §4.4: a
proxy that only moves people inside the counted unit):

  1. each hex goes to the commune associée its centroid falls in, else the nearest one within
     SNAP_M (hex centroids fall just offshore of atoll rims and coasts; playbook "Hex centroids
     fall just offshore of island units"); every outside hex is within 1.9 km;
  2. each hex goes to its nearest island; a hex on an island with nobody in 2017 gets weight 0;
  3. inside each commune associée the hexes' Kontur people are scaled to its 2017 population, so
     a commune's dots fall on its communes associées in their 2017 shares and inside each by Kontur;
  4. a commune associée with people in 2017 and no weighted hex is appended as its own polygon at
     its 2017 population (none expected; asserted).

The language table is per commune, so this only decides where in a commune a dot falls. The
2017 shares are five years before the 2022 table; nothing finer than the commune is published for
2022's languages anyway.

CHECKS (stop unless they hold): 48 / 116 / 119 features; the communes associées' 2017 population
sums to each commune's; every comas' hex weights sum to its 2017 population; every commune holds
weight; no hex is further than SNAP_M from a comas. Printed: Kontur against the 2017 census per
commune before calibration, with the shuffle null.
"""
import io
import math
import os
import random
import sys
import urllib.request
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import pandas as pd  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "sources"))
RAW = HERE / "data" / "raw" / "pf"
SHP = RAW / "shp"
OUT = HERE / "data" / "geo" / "pf" / "pf_hexes.gpkg"
URL = ("https://static.data.gouv.fr/resources/limites-geographiques-administratives/"
       "20220610-202135/shapefiles.zip")
CRS_M = 3297          # RGPF / UTM zone 6S, metres
SNAP_M = 2500
N_COM, N_COMAS, N_ILE = 48, 116, 119


def fetch():
    SHP.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    data = urllib.request.urlopen(req, timeout=600).read()
    (RAW / "shapefiles.zip").write_bytes(data)
    zipfile.ZipFile(io.BytesIO(data)).extractall(SHP)
    print(f"  shapefiles.zip: {len(data):,} bytes")


def read(name, n):
    import geopandas as gpd
    g = gpd.read_file(SHP / f"{name}.shp")
    if len(g) != n:
        raise SystemExit(f"pf: {name}.shp has {len(g)} features, expected {n}")
    return g


def pear(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return num / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))


def main():
    if "--fetch" in sys.argv or not (SHP / "Com.shp").exists():
        fetch()
    import geopandas as gpd
    from _grid import kontur_path

    com = read("Com", N_COM)
    comas = read("Comas", N_COMAS)
    ile = read("Ile", N_ILE)
    if com["IDCom"].nunique() != N_COM or comas["IDComas"].nunique() != N_COMAS:
        raise SystemExit("pf: duplicated codes")
    # the communes associées tile the communes, by code and by 2017 population
    s = comas.groupby("IDCom")["Indivds"].sum()
    c = com.set_index("IDCom")["Indivds"]
    if set(s.index) != set(c.index) or (s.reindex(c.index) != c).any():
        raise SystemExit("pf: communes associées do not sum to their commune's 2017 population")
    print(f"  {N_COM} communes, {N_COMAS} communes associées, {N_ILE} islands; 2017 population "
          f"{int(c.sum()):,}, every commune = the sum of its communes associées")

    comas_m = comas.to_crs(CRS_M)
    ile_m = ile.to_crs(CRS_M)
    k = gpd.read_file(kontur_path("pf"))
    if len(k) == 0:
        raise SystemExit("pf: Kontur PF extract has ZERO features")
    pts = gpd.GeoDataFrame({"kpop": k["population"].astype(float).to_numpy()},
                           geometry=k.geometry.centroid, crs=k.crs).to_crs(CRS_M)
    j = gpd.sjoin(pts, comas_m[["IDComas", "IDCom", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(pts.index)
    out = j["IDComas"].isna()
    near = gpd.sjoin_nearest(pts[out], comas_m[["IDComas", "IDCom", "geometry"]],
                             distance_col="d")
    near = near[~near.index.duplicated(keep="first")]
    if near["d"].max() > SNAP_M:
        raise SystemExit(f"pf: a hex centroid {near['d'].max():.0f} m from every commune associée")
    j.loc[out, "IDComas"] = near["IDComas"]
    j.loc[out, "IDCom"] = near["IDCom"]
    print(f"  Kontur PF: {len(k):,} hexes, {pts['kpop'].sum():,.0f} people; {int(out.sum())} hexes "
          f"({pts.loc[out, 'kpop'].sum():,.0f} people) snapped onto the nearest commune associée "
          f"(at most {near['d'].max():,.0f} m)")

    # Kontur against the 2017 census per commune, before calibration (the playbook's band check)
    per = j.groupby("IDCom")["kpop"].sum().reindex(c.index).fillna(0)
    ratio = per.sum() / c.sum()
    norm = (per / c / ratio).sort_values()
    print(f"  Kontur / 2017 census nationally {ratio:.3f}; per commune normalised p10 "
          f"{norm.quantile(.1):.2f} median {norm.median():.2f} p90 {norm.quantile(.9):.2f}; "
          f"lowest {norm.index[0]} {norm.iloc[0]:.2f}, highest {norm.index[-1]} {norm.iloc[-1]:.2f}")
    lc, lk = [math.log(v) for v in c], [math.log(max(v, 1)) for v in per]
    r = pear(lc, lk)
    rng = random.Random(0)
    best = max(abs(pear(lc, rng.sample(lk, len(lk)))) for _ in range(500))
    print(f"  log correlation r = {r:.3f} against a best of {best:.3f} over 500 shuffles")
    if r <= best:
        raise SystemExit("pf: the commune join is not carrying information")

    # islands nobody lived on in 2017 get no weight
    ji = gpd.sjoin_nearest(pts, ile_m[["IDIle", "Ile", "Indivds", "geometry"]], distance_col="d")
    ji = ji[~ji.index.duplicated(keep="first")].reindex(pts.index)
    empty_isle = ji["Indivds"] == 0
    j["ile"] = ji["Ile"]
    j["w"] = pts["kpop"].where(~empty_isle, 0.0)
    gone = j.loc[empty_isle].groupby("ile")["kpop"].sum()
    gone = gone[gone > 0].sort_values(ascending=False)
    print(f"  weight 0 on islands with nobody in 2017: {gone.sum():,.0f} Kontur people on "
          f"{len(gone)} islands (" + ", ".join(f"{i} {v:.0f}" for i, v in gone.head(6).items())
          + ")")

    # scale each commune associée to its 2017 population
    cpop = comas.set_index("IDComas")["Indivds"].astype(float)
    wsum = j.groupby("IDComas")["w"].sum()
    j["pop"] = j["w"] * (j["IDComas"].map(cpop) / j["IDComas"].map(wsum)).fillna(0.0)
    missing = [i for i in cpop.index if cpop[i] > 0 and wsum.get(i, 0) <= 0]
    hexes = gpd.GeoDataFrame({"unit": j["IDCom"].astype(int).astype(str), "comas": j["IDComas"],
                              "pop": j["pop"]}, geometry=k.geometry.to_numpy(), crs=k.crs)
    hexes = hexes[hexes["pop"] > 0].to_crs(4326)
    if missing:
        extra = comas[comas["IDComas"].isin(missing)]
        print(f"  {len(missing)} communes associées with people and no weighted hex, drawn on "
              f"their own polygon: " + ", ".join(extra["Comas"]))
        add = gpd.GeoDataFrame({"unit": extra["IDCom"].astype(int).astype(str),
                                "comas": extra["IDComas"], "pop": extra["Indivds"].astype(float)},
                               geometry=extra.geometry.to_numpy(), crs=comas.crs).to_crs(4326)
        hexes = pd.concat([hexes, add], ignore_index=True)
    got = hexes.groupby("comas")["pop"].sum().reindex(cpop.index).fillna(0)
    if ((got - cpop).abs() > 0.01).any():
        raise SystemExit("pf: weights do not sum to each commune associée's 2017 population")
    units = set(hexes["unit"])
    if units != set(com["IDCom"].astype(str)):
        raise SystemExit(f"pf: communes without weight: {set(com['IDCom'].astype(str)) - units}")
    dens = hexes.to_crs(CRS_M)
    dens = (dens["pop"] / (dens.area / 1e6)).max()
    print(f"  calibrated densest hex {dens:,.0f} people/km2; every comas reproduces its 2017 count")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # `sub` (subdivision, 1-5) is the unit the 2012 language table counts in; `comas` lets
    # countries/pf.py weight each language inside a subdivision (sources/pf.md §3)
    hexes["comas"] = hexes["comas"].astype(int).astype(str)
    hexes["sub"] = hexes["comas"].map(dict(zip(comas["IDComas"].astype(str),
                                              comas["IDSub"].astype(int).astype(str))))
    if hexes["sub"].isna().any():
        raise SystemExit("pf: hexes without a subdivision")
    hexes[["unit", "comas", "sub", "pop", "geometry"]].to_file(OUT, layer="hexes", driver="GPKG")
    print(f"  wrote {OUT} ({len(hexes):,} features, {hexes['unit'].nunique()} communes)")


if __name__ == "__main__":
    main()
