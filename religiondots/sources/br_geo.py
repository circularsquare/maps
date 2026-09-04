"""Brazil — malha municipal 2010, downloaded per state and merged into one layer.

IBGE publishes the 2010 municipal mesh as 27 per-state shapefiles and no national file, so
this fetches all 27 and concatenates them.

VINTAGE (spec §8.1).  The 2010 mesh, not the current one, because `br.csv`'s detailed half is
the 2010 census: 5,565 municipios then against 5,570 now.  The five created since -- Pescaria
Brava and Balneario Rincao (SC), Mojui dos Campos (PA), Paraiso das Aguas (MS) and Pinto
Bandeira (RS) -- were all split OFF existing municipios, so every 2010 code still exists in a
current mesh and a join to it would succeed silently while five parents quietly lost the
territory that became a child.  That is §8.1's failure mode exactly: the codes match and the
answer is wrong.

TWO TRANSPORT GOTCHAS, both worth carrying forward:

  * `geoftp.ibge.gov.br` serves an INCOMPLETE TLS CHAIN.  curl and Python both fail with
    "unable to get local issuer certificate", and certifi does not help because the missing
    piece is an intermediate the server should be sending and does not.  Browsers paper over
    it by fetching the intermediate themselves.  Plain `http://` works and is what this uses;
    the files are public and unsigned either way, so nothing is being protected by the TLS
    that is not working.
  * The shapefile inside each zip is named for the NUMERIC state code, not the two-letter
    one -- `12MUE250GC_SIR.shp` for Acre -- so the member name cannot be predicted from the
    URL.  Globbed, not constructed.

The IBGE malhas API (`servicodados.../api/v3/malhas/`) is NOT an alternative here: it serves
only the current mesh, `periodo=2010` returns HTTP 500, and its output is generalised.

Writes:
    data/geo/br/br_municipios_2010.gpkg   one layer, `kod` + `nome`

Usage:
    python sources/br_geo.py            download what is missing, then merge
    python sources/br_geo.py --merge    merge only, from what is already on disk
"""

import glob
import os
import sys
import urllib.request
import zipfile

import geopandas as gpd
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GEO = os.path.join(ROOT, "data", "geo", "br")
OUT = os.path.join(GEO, "br_municipios_2010.gpkg")

# http, not https -- see the module docstring.
BASE = ("http://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/"
        "malhas_municipais/municipio_2010")

UFS = ["ac", "al", "am", "ap", "ba", "ce", "df", "es", "go", "ma", "mg", "ms", "mt",
       "pa", "pb", "pe", "pi", "pr", "rj", "rn", "ro", "rr", "rs", "sc", "se", "sp", "to"]

EXPECTED = 5565   # municipios at the 2010 census

# IBGE ships the two big Rio Grande do Sul coastal lagoons in the municipal mesh as
# pseudo-municipios with codes of their own, so the raw merge is 5,567 rather than 5,565.
# They carry no census rows, so nothing would be placed in them either way, but a polygon
# that is a lake has no business in a layer of populated units. Dropped by code.
LAGOONS = {"4300001": "Lagoa Mirim", "4300002": "Lagoa dos Patos"}


def fetch():
    os.makedirs(GEO, exist_ok=True)
    for uf in UFS:
        zpath = os.path.join(GEO, f"{uf}_municipios.zip")
        dest = os.path.join(GEO, uf)
        if glob.glob(os.path.join(dest, "*.shp")):
            continue
        if not (os.path.exists(zpath) and os.path.getsize(zpath) > 10000):
            url = f"{BASE}/{uf}/{uf}_municipios.zip"
            print(f"  {uf}: downloading")
            urllib.request.urlretrieve(url, zpath)
            size = os.path.getsize(zpath)
            # sources.md §5a -- a 200 is not a download.
            if size < 10000:
                raise SystemExit(f"{url} returned {size} bytes")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(dest)
        print(f"  {uf}: {os.path.getsize(zpath):,} bytes")


def merge():
    frames = []
    for uf in UFS:
        shps = glob.glob(os.path.join(GEO, uf, "*.shp"))
        if not shps:
            raise SystemExit(f"no shapefile for {uf} — run without --merge first")
        if len(shps) > 1:
            raise SystemExit(f"{uf}: expected one shapefile, found {len(shps)}")
        g = gpd.read_file(shps[0])
        if "CD_GEOCODM" not in g.columns:
            raise SystemExit(f"{uf}: no CD_GEOCODM column, got {list(g.columns)}")
        g = g[["CD_GEOCODM", "NM_MUNICIP", "geometry"]].rename(
            columns={"CD_GEOCODM": "kod", "NM_MUNICIP": "nome"})
        g["kod"] = g["kod"].astype(str).str.strip()
        g["uf"] = uf
        frames.append(g)
        print(f"  {uf}: {len(g):>4} municipios  ({g.crs})")

    crss = {str(f.crs) for f in frames}
    if len(crss) > 1:
        raise SystemExit(f"states disagree about CRS: {crss}")

    br = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True),
                          geometry="geometry", crs=frames[0].crs)

    lagoons = br[br["kod"].isin(LAGOONS)]
    if len(lagoons):
        print("\ndropping IBGE's lagoon pseudo-municipios: "
              + ", ".join(f"{r.kod} {r.nome}" for r in lagoons.itertuples()))
        br = br[~br["kod"].isin(LAGOONS)].reset_index(drop=True)

    print(f"\n{len(br):,} municipios (expected {EXPECTED:,})")
    if len(br) != EXPECTED:
        print(f"  !! off by {len(br) - EXPECTED:+,}")
    dup = br["kod"].duplicated().sum()
    if dup:
        raise SystemExit(f"{dup} duplicate municipality codes across states")
    bad = (br["kod"].str.len() != 7).sum()
    if bad:
        raise SystemExit(f"{bad} codes are not 7 digits")

    empty = br.geometry.isna() | br.geometry.is_empty
    if empty.any():
        print(f"  !! {int(empty.sum())} empty geometries")

    # Check against the counts before writing, so a vintage mistake is caught here rather
    # than as a silent shortfall in scatter.py.
    csv = os.path.join(ROOT, "data", "normalized", "br.csv")
    if os.path.exists(csv):
        d = pd.read_csv(csv, usecols=["geo_id", "year"], dtype={"geo_id": str})
        for year in (2010, 2022):
            want = set(d.loc[d["year"] == year, "geo_id"])
            have = set(br["kod"])
            print(f"  {year} data: {len(want):,} municipios, "
                  f"{len(want - have):,} with no polygon, "
                  f"{len(have - want):,} polygons with no data")

    br.to_file(OUT, layer="br_municipios_2010", driver="GPKG")
    print(f"\nwrote {OUT}")


def main():
    if "--merge" not in sys.argv:
        fetch()
    merge()


if __name__ == "__main__":
    main()
