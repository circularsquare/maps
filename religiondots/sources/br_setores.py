"""Brazil — a placement layer of census setores, weighted by their own population.

Writes data/geo/br/br_setores_2022.gpkg — `setor`, `unit` (the município code), `pop`.

WHAT THIS FIXES. Brazil's religion counts are published per município and nothing finer
exists (sources/br.md), so the counts stay municipal. But §8.2's assumption — that a
country's units are small enough that spreading dots evenly inside one is harmless — does
not survive Brazil. **São Paulo is one polygon holding 11.5M people**, and drawing its
11,500 dots uniformly across it puts as many in the Serra da Cantareira as on Avenida
Paulista. Rio, Manaus and Brasília are the same. That is the failure India hit (§8.2a) and
Germany answered with a grid (§8.2b); Brazil answers it with setores.

§14.4 permits exactly this and no more: **refine placement, never invent magnitude.** Every
município's total stays precisely what IBGE published. What changes is only where inside the
município the dots land — and it is a POPULATION weight, not a religion one, because nothing
measures where a given church's members live inside a município. A Catholic dot and an
Assembleia de Deus dot are spread identically. Read a cluster as "this município, drawn
where its people are", never as a neighbourhood.

WHY SETORES AND NOT THE 1KM GRID. IBGE publishes both. The grid needs a spatial join and a
clip against 5,570 municípios, and its cells straddle boundaries, which is the work
`de_grid.py` had to do for Germany. **Setores nest by CODE** — `CD_SETOR[:7]` is exactly
`CD_MUN`, verified below on every row — so the assignment is a string slice with no
geometry, no ambiguity and no slivers. They are also the units the census was collected in.

TWO THINGS THAT WOULD GO WRONG QUIETLY.

- **Equal shares per setor is NOT good enough here, unlike US tracts.** COMMANDS.txt argues
  that US tracts are built to ~4,000 people each so an equal split is already a population
  weighting. Brazilian setores are built to roughly 300 households in cities and fewer in
  the country, and rural ones cover enormous areas, so equal shares would systematically
  over-weight the countryside. The real `v0001` population is joined and used.
- **`CD_SIT = 9` setores are bodies of water.** IBGE gives open water its own setor codes.
  They carry no population, so a population weight already ignores them, but they are
  dropped outright so they cannot take a dot through the zero-population fallback either.

Usage:
    python sources/br_setores.py --fetch    27 per-state gpkg (~1.5 GB) + a 15 MB csv
    python sources/br_setores.py            build from what is on disk
"""

import glob
import io
import os
import sys
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "br_setores")
GEO = os.path.join(ROOT, "data", "geo", "br")
OUT = os.path.join(GEO, "br_setores_2022.gpkg")
RESCALED = os.path.join(ROOT, "data", "normalized", "br_municipio_rescaled.csv")

# http, not https — geoftp/ftp.ibge.gov.br serve an incomplete TLS chain (sources/br_geo.py).
MESH = ("http://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/"
        "malhas_de_setores_censitarios__divisoes_intramunicipais/censo_2022/setores/gpkg/UF")
POP = ("http://ftp.ibge.gov.br/Censos/Censo_Demografico_2022/"
       "Agregados_por_Setores_Censitarios/Agregados_por_Setor_csv/"
       "Agregados_por_setores_basico_BR_20260520.zip")
POP_ZIP = os.path.join(RAW, "Agregados_por_setores_basico_BR.zip")

UFS = ["AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT",
       "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO"]

WATER_SIT = "9"          # "Massas de água", per Dicionario_de_dados_malha_agregados
POP_COL = "v0001"        # população residente, per the agregados dictionary
SIMPLIFY = 0.0002        # ~22 m. A dot only has to land in the right setor.
EXPECTED_MUNICIPIOS = 5570


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for i, uf in enumerate(UFS, 1):
        dest = os.path.join(RAW, f"{uf}_setores_CD2022.gpkg")
        if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
            print(f"  [{i:2}/27] {uf}: have it")
            continue
        url = f"{MESH}/{uf}/{uf}_setores_CD2022.gpkg"
        print(f"  [{i:2}/27] {uf}: downloading", flush=True)
        urllib.request.urlretrieve(url, dest)
        size = os.path.getsize(dest)
        if size < 1_000_000:                       # §5a: a 200 is not a download
            raise SystemExit(f"{url} returned {size:,} bytes")
        print(f"           {size:,} bytes", flush=True)

    if not (os.path.exists(POP_ZIP) and os.path.getsize(POP_ZIP) > 1_000_000):
        print("  population: downloading", flush=True)
        urllib.request.urlretrieve(POP, POP_ZIP)
    if not zipfile.is_zipfile(POP_ZIP):
        raise SystemExit(f"{POP_ZIP} is not a zip")
    print(f"  population: {os.path.getsize(POP_ZIP):,} bytes")


def read_pop():
    import pandas as pd

    with zipfile.ZipFile(POP_ZIP) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise SystemExit(f"no csv in {POP_ZIP}: {z.namelist()[:5]}")
        with z.open(names[0]) as fh:
            raw = fh.read()
    # IBGE ships these as latin-1, semicolon-separated.
    df = pd.read_csv(io.BytesIO(raw), sep=";", dtype=str, encoding="latin-1",
                     low_memory=False)
    cols = {c.lower(): c for c in df.columns}
    setor = cols.get("cd_setor")
    pop = cols.get(POP_COL)
    if not setor or not pop:
        raise SystemExit(f"expected CD_SETOR and {POP_COL}; got {list(df.columns)[:15]}")
    out = df[[setor, pop]].copy()
    out.columns = ["setor", "pop"]
    out["setor"] = out["setor"].astype(str).str.strip()
    # Suppressed / blank cells become 0 rather than NaN: a setor with no published
    # population must not take dots, and must not poison its município's weights either.
    out["pop"] = pd.to_numeric(out["pop"], errors="coerce").fillna(0.0)
    print(f"  population rows: {len(out):,}, total {out['pop'].sum():,.0f}")
    return dict(zip(out["setor"], out["pop"]))


def main():
    import geopandas as gpd
    import pandas as pd

    have = sorted(glob.glob(os.path.join(RAW, "*_setores_CD2022.gpkg")))
    if len(have) != 27:
        raise SystemExit(f"{len(have)} of 27 per-state files -- run with --fetch first")
    if not os.path.exists(POP_ZIP):
        raise SystemExit(f"missing {POP_ZIP} -- run with --fetch first")

    pop = read_pop()

    # WRITTEN ONE STATE AT A TIME, not concatenated. 452,000 simplified polygons in a
    # single GeoDataFrame plus the copy `pd.concat` makes is several GB of RAM for no
    # benefit; appending keeps the peak at one state (São Paulo, ~70,000 setores). Only the
    # three non-geometry columns are accumulated, and the checks run on those.
    if os.path.exists(OUT):
        os.remove(OUT)
    os.makedirs(GEO, exist_ok=True)

    attrs, n_water, n_raw, first, n_parts = [], 0, 0, True, {}
    for i, uf in enumerate(UFS, 1):
        p = os.path.join(RAW, f"{uf}_setores_CD2022.gpkg")
        g = gpd.read_file(p)
        if len(g) == 0:          # §12: a read that succeeds is not a read that returned data
            raise SystemExit(f"{p} read cleanly and returned ZERO features")
        n_raw += len(g)
        sit = "CD_SIT" if "CD_SIT" in g.columns else "CD_SITUACAO"
        water = g[sit].astype(str).str.strip() == WATER_SIT

        # IBGE'S TWO LAGOON PSEUDO-MUNICÍPIOS TURN UP HERE TOO, IN A THIRD DISGUISE.
        # `br_geo.py` drops Lagoa Mirim (4300001) and Lagoa dos Patos (4300002) from the
        # municipal mesh by code. In the setor mesh they appear as setores
        # `430000100000000` and `430000200000000` — 2,884 km² and 10,202 km² of open water
        # in Rio Grande do Sul — with **CD_MUN null and CD_SIT null**, so neither the water
        # filter nor a code test catches them, and the nesting assertion below fires
        # instead. A setor with no município cannot be placed in one, so a null município is
        # the honest test and it is the general form of the rule.
        orphan = g["CD_MUN"].isna() | (g["CD_MUN"].astype(str).str.strip() == "")
        if orphan.any():
            codes = g.loc[orphan, "CD_SETOR"].astype(str).tolist()
            print(f"           {uf}: dropping {int(orphan.sum())} setor(es) with no "
                  f"município: {codes}", flush=True)

        n_water += int(water.sum())
        g = g[~water & ~orphan]
        g = g[["CD_SETOR", "CD_MUN", "geometry"]].rename(
            columns={"CD_SETOR": "setor", "CD_MUN": "unit"})
        g["setor"] = g["setor"].astype(str).str.strip()
        g["unit"] = g["unit"].astype(str).str.strip()

        # THE WHOLE REASON SETORES WERE CHOSEN — asserted, not assumed.
        bad = g["setor"].str[:7] != g["unit"]
        if bad.any():
            raise SystemExit(f"{uf}: {int(bad.sum())} setores whose code does not start "
                             "with their município -- the nesting assumption is broken")

        g["geometry"] = g.geometry.simplify(SIMPLIFY, preserve_topology=True)
        empty = g.geometry.isna() | g.geometry.is_empty
        if empty.any():
            g = g[~empty]

        # A SETOR CAN ARRIVE AS SEVERAL ROWS, AND THE POPULATION JOIN MUST NOT SEE THEM.
        # IBGE ships 914 setores nationally as multiple separate polygon features — one per
        # disjoint part, overwhelmingly river islands in Pará and Amazonas plus coastal
        # fragments in Rio and São Paulo. `pop` is keyed on the setor CODE, so mapping it
        # onto the parts as delivered gives a five-part setor five times its population and
        # five times its pull on the dots. Dissolving first makes the layer one row per
        # setor, which is also exactly the 468,099 rows the population file has.
        dup = g["setor"].duplicated(keep=False)
        if dup.any():
            merged = g[dup].dissolve(by="setor", as_index=False)
            g = pd.concat([g[~dup], merged], ignore_index=True)
            g = gpd.GeoDataFrame(g, geometry="geometry", crs=merged.crs)
            n_parts[uf] = (int(dup.sum()), int(len(merged)))

        g["pop"] = g["setor"].map(pop).fillna(0.0)
        g = g[["setor", "unit", "pop", "geometry"]]
        g.to_file(OUT, layer="br_setores_2022", driver="GPKG",
                  mode="w" if first else "a")
        first = False
        attrs.append(g.drop(columns="geometry"))
        print(f"  [{i:2}/27] {uf}: {len(g):>7,} setores, "
              f"{g['pop'].sum():>12,.0f} people", flush=True)

    br = pd.concat(attrs, ignore_index=True)
    multi_rows = sum(v[0] for v in n_parts.values())
    multi_setores = sum(v[1] for v in n_parts.values())
    print(f"\n{len(br):,} setores kept of {n_raw:,} rows delivered "
          f"({n_water:,} water setores dropped)")
    if n_parts:
        top = ", ".join(f"{k} {v[1]}" for k, v in
                        sorted(n_parts.items(), key=lambda x: -x[1][1])[:5])
        print(f"  multi-part setores dissolved: {multi_setores:,} setores that arrived as "
              f"{multi_rows:,} rows ({top})")
    if br["setor"].duplicated().any():
        raise SystemExit("duplicate setor codes")

    matched = br["pop"].gt(0).sum()
    print(f"  population joined: {matched:,} setores with pop>0, "
          f"{br['pop'].sum():,.0f} people total")
    missing = br[~br["setor"].isin(pop)]
    print(f"  setores with no row in the population file: {len(missing):,}")

    # ---- the check that matters: can every município still be drawn? ----
    if os.path.exists(RESCALED):
        d = pd.read_csv(RESCALED, usecols=["geo_id"], dtype={"geo_id": str})
        want = set(d["geo_id"])
        have_any = set(br["unit"])
        have_pop = set(br.loc[br["pop"] > 0, "unit"])
        print(f"\n  municípios in the drawn data: {len(want):,}")
        print(f"    with at least one setor      {len(want & have_any):,}")
        print(f"    with a POPULATED setor       {len(want & have_pop):,}")
        gap = sorted(want - have_any)
        if gap:
            raise SystemExit(f"{len(gap)} municípios have no setor at all: {gap[:10]}")
        nopop = sorted(want - have_pop)
        if nopop:
            print(f"    !! {len(nopop)} have setores but none populated: {nopop[:10]}\n"
                  "       scatter.py falls back to equal shares inside those (§8.2).")
        if len(have_any) != EXPECTED_MUNICIPIOS:
            print(f"    note: the mesh covers {len(have_any):,} municípios, expected "
                  f"{EXPECTED_MUNICIPIOS:,}")
    else:
        print(f"\n  (no {RESCALED} -- run br_rescale.py to check the join)")

    print(f"\nwrote {OUT} ({len(br):,} polygons, "
          f"{os.path.getsize(OUT) / 1e6:.0f} MB)")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    main()
