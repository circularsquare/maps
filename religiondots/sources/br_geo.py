"""Brazil — malha municipal, 2010 or 2022, merged into one layer per vintage.

    python sources/br_geo.py                 # 2010: 27 per-state shapefiles, 5,565 units
    python sources/br_geo.py --year 2022     # 2022: one 194MB national file, 5,570 units

VINTAGE (spec §8.1), AND IT CHANGED ON 2026-09-04.  The rule is that the boundaries must be
the vintage the DATA is published on, and Brazil's drawn data moved: `br_rescale.py` now
puts 2022 municipal totals on 2010 municipal structure (§3.4), so what the map places is
keyed to the **2022** mesh and `countries.py` reads `br_municipios_2022.gpkg`.

The 2010 mesh is kept, still built by the default invocation, and is still the right answer
for anything drawing the 2010 census as it stands.  Its original argument is unchanged and
is worth keeping because it is the general form of the trap: 5,565 municipios in 2010
against 5,570 now, and the five created since -- Pescaria Brava and Balneario Rincao (SC),
Mojui dos Campos (PA), Paraiso das Aguas (MS) and Pinto Bandeira (RS) -- were all split OFF
existing municipios, so **every 2010 code still exists in a current mesh**.  A join from
2010 data to a current mesh therefore succeeds silently while five parents quietly lose the
territory that became a child.  Running it the other way -- 2022 data on the 2010 mesh --
fails in the mirror image: the five children have no polygon at all and their 49,483 people
are dropped, while their parents are drawn with territory that is no longer theirs.  Neither
is loud.  Match the vintage to the data and the question does not arise.

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


# ---------------------------------------------------------------------------- 2022 mesh
# Unlike 2010, IBGE ships 2022 as ONE national file. Same host, same missing TLS
# intermediate, so the same plain http.
BASE_2022 = ("http://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/"
             "malhas_municipais/municipio_2022/Brasil/BR")
ZIP_2022 = os.path.join(GEO, "BR_Municipios_2022.zip")
OUT_2022 = os.path.join(GEO, "br_municipios_2022.gpkg")
EXPECTED_2022 = 5570


def fetch_2022():
    os.makedirs(GEO, exist_ok=True)
    if os.path.exists(ZIP_2022) and os.path.getsize(ZIP_2022) > 150_000_000:
        print("already have", ZIP_2022)
        return
    url = f"{BASE_2022}/BR_Municipios_2022.zip"
    print(f"downloading {url} (~194 MB)")
    urllib.request.urlretrieve(url, ZIP_2022)
    size = os.path.getsize(ZIP_2022)
    if size < 150_000_000 or not zipfile.is_zipfile(ZIP_2022):
        raise SystemExit(f"{url} returned {size:,} bytes and is not a usable zip")
    with zipfile.ZipFile(ZIP_2022) as z:
        shps = [n for n in z.namelist() if n.lower().endswith(".shp")]
    if not shps:
        raise SystemExit(f"no .shp inside {ZIP_2022}")
    print(f"  {size:,} bytes, member {shps[0]}")


def merge_2022():
    if not os.path.exists(ZIP_2022):
        raise SystemExit(f"missing {ZIP_2022} -- run with --year 2022 and no --merge")
    with zipfile.ZipFile(ZIP_2022) as z:
        shps = [n for n in z.namelist() if n.lower().endswith(".shp")]
    br = gpd.read_file(f"zip://{ZIP_2022}!{shps[0]}")
    # A read that succeeds is not a read that returned data (§12, found with Chile).
    if len(br) == 0:
        raise SystemExit(f"{shps[0]} read cleanly and returned ZERO features")
    print(f"read {shps[0]}: {len(br):,} features, crs={br.crs}")

    cols = {c.upper(): c for c in br.columns}
    kod = cols.get("CD_MUN") or cols.get("CD_GEOCODM")
    nome = cols.get("NM_MUN") or cols.get("NM_MUNICIP")
    if not kod or not nome:
        raise SystemExit(f"unexpected columns: {list(br.columns)}")
    br = br[[kod, nome, "geometry"]].rename(columns={kod: "kod", nome: "nome"})
    br["kod"] = br["kod"].astype(str).str.strip()

    lagoons = br[br["kod"].isin(LAGOONS)]
    if len(lagoons):
        print("dropping IBGE's lagoon pseudo-municipios: "
              + ", ".join(f"{r.kod} {r.nome}" for r in lagoons.itertuples()))
        br = br[~br["kod"].isin(LAGOONS)].reset_index(drop=True)

    print(f"{len(br):,} municipios (expected {EXPECTED_2022:,})")
    if len(br) != EXPECTED_2022:
        raise SystemExit(f"off by {len(br) - EXPECTED_2022:+,}")
    if br["kod"].duplicated().any():
        raise SystemExit("duplicate municipality codes")
    if (br["kod"].str.len() != 7).any():
        raise SystemExit("some codes are not 7 digits")
    empty = br.geometry.isna() | br.geometry.is_empty
    if empty.any():
        raise SystemExit(f"{int(empty.sum())} empty geometries")

    # The join that matters, against what countries.py actually reads (§8.1).
    resc = os.path.join(ROOT, "data", "normalized", "br_municipio_rescaled.csv")
    if os.path.exists(resc):
        d = pd.read_csv(resc, usecols=["geo_id"], dtype={"geo_id": str})
        want, have = set(d["geo_id"]), set(br["kod"])
        print(f"  rescaled data: {len(want):,} municipios, {len(want - have):,} with no "
              f"polygon, {len(have - want):,} polygons with no data")
        if want - have:
            raise SystemExit(f"{len(want - have)} municipios have no polygon: "
                             f"{sorted(want - have)[:10]}")
    else:
        print(f"  (no {resc} yet -- run br_rescale.py to check the join)")

    br.to_file(OUT_2022, layer="br_municipios_2022", driver="GPKG")
    print(f"\nwrote {OUT_2022}")


def main():
    year = "2010"
    if "--year" in sys.argv:
        year = sys.argv[sys.argv.index("--year") + 1]
    if year not in ("2010", "2022"):
        raise SystemExit("--year must be 2010 or 2022")
    if year == "2022":
        if "--merge" not in sys.argv:
            fetch_2022()
        merge_2022()
        return
    if "--merge" not in sys.argv:
        fetch()
    merge()


if __name__ == "__main__":
    main()
