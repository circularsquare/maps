"""US pipeline: which direction are the rich neighbourhoods, per metro.

Inputs (see fetch.py): block group population-weighted centroids + population,
ACS median household income, county->CBSA crosswalk. The estimator itself lives
in core.py and is shared with the global RWI pipeline.

Ranks rather than dollars, because ACS top-codes median household income at
$250,001 and the distribution is right-skewed -- in raw dollars a handful of
block groups would dominate every fit.

Outputs
-------
  out/metros.csv   one row per metro: bearing, strength, R^2, radial term
  out/field.csv    the per-block-group wealth field, for plotting
"""
import numpy as np
import pandas as pd
from pathlib import Path

from core import KM_PER_DEG_LAT, fit_wealth_field, to_local_xy, weighted_quantile

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"

MIN_METRO_POP = 500_000   # smaller metros give noisy, unstable fits
MAX_RADIUS_KM = 50.0      # CBSAs include huge empty counties (San Bernardino!)
TRIM_PCTL = 97.5          # pop-weighted distance percentile to trim to


def load_block_groups():
    cen = pd.read_csv(
        DATA / "CenPop2020_Mean_BG.txt",
        dtype={"STATEFP": str, "COUNTYFP": str, "TRACTCE": str, "BLKGRPCE": str},
        encoding="utf-8-sig",
    )
    cen["geoid"] = cen.STATEFP + cen.COUNTYFP + cen.TRACTCE + cen.BLKGRPCE
    cen["county"] = cen.STATEFP + cen.COUNTYFP
    cen = cen.rename(columns={"POPULATION": "pop", "LATITUDE": "lat", "LONGITUDE": "lon"})

    acs = pd.read_csv(DATA / "acsdt5y2023-b19013.dat", sep="|", dtype=str)
    acs = acs[acs.GEO_ID.str.startswith("1500000US")].copy()
    acs["geoid"] = acs.GEO_ID.str[9:]
    acs["income"] = pd.to_numeric(acs.B19013_E001, errors="coerce")
    acs.loc[acs.income < 0, "income"] = np.nan   # -666666666 = suppressed

    bg = cen[["geoid", "county", "pop", "lat", "lon"]].merge(
        acs[["geoid", "income"]], on="geoid", how="inner")
    return bg[(bg["pop"] > 0) & bg.income.notna()]


def load_metros():
    """County -> CBSA, metropolitan statistical areas only."""
    x = pd.read_excel(DATA / "list1_2023.xlsx", header=2)
    x = x[x["Metropolitan/Micropolitan Statistical Area"] == "Metropolitan Statistical Area"]
    x = x.dropna(subset=["FIPS State Code", "FIPS County Code"])
    x["county"] = (x["FIPS State Code"].astype(int).map("{:02d}".format)
                   + x["FIPS County Code"].astype(int).map("{:03d}".format))
    x["cbsa"] = x["CBSA Code"].astype(int).astype(str)
    return x[["county", "cbsa", "CBSA Title"]].rename(columns={"CBSA Title": "name"})


def analyse(g):
    pop = g["pop"].to_numpy(float)
    lat0 = np.average(g.lat.to_numpy(), weights=pop)
    lon0 = np.average(g.lon.to_numpy(), weights=pop)
    x, y = to_local_xy(g.lat.to_numpy(), g.lon.to_numpy(), lat0, lon0)

    # Trim the rural tail: CBSAs are county-based and some sprawl for 200 km.
    dist = np.hypot(x, y)
    cut = min(MAX_RADIUS_KM, weighted_quantile(dist, pop, TRIM_PCTL / 100))
    keep = dist <= cut
    if keep.sum() < 50:
        return None
    g, pop, x, y = g[keep], pop[keep], x[keep], y[keep]

    fit = fit_wealth_field(x, y, g.income.to_numpy(float), pop)
    if fit is None:
        return None
    rank = fit.pop("rank")
    fit.pop("z")

    return dict(
        n_bg=len(g), pop=pop.sum(), radius_km=cut,
        cent_lat=lat0, cent_lon=lon0,
        center_lat=lat0 + fit["peak_dy"] / KM_PER_DEG_LAT,
        center_lon=lon0 + fit["peak_dx"] / (KM_PER_DEG_LAT * np.cos(np.radians(lat0))),
        **fit,
        field=pd.DataFrame(dict(geoid=g.geoid.to_numpy(), lat=g.lat.to_numpy(),
                                lon=g.lon.to_numpy(), pop=pop, x=x, y=y,
                                rank=rank, income=g.income.to_numpy())),
    )


def main():
    OUT.mkdir(exist_ok=True)
    bg = load_block_groups().merge(load_metros(), on="county", how="inner")
    print(f"{len(bg):,} block groups in {bg.cbsa.nunique():,} metros")

    rows, fields = [], []
    for (cbsa, name), g in bg.groupby(["cbsa", "name"]):
        if g["pop"].sum() < MIN_METRO_POP:
            continue
        res = analyse(g)
        if res is None:
            continue
        f = res.pop("field")
        f.insert(0, "cbsa", cbsa)
        fields.append(f)
        rows.append(dict(cbsa=cbsa, name=name, **res))

    out = pd.DataFrame(rows).sort_values("pop", ascending=False)
    out["core"] = np.where(out.radial < 0, "rich core", "rich suburbs")
    out.to_csv(OUT / "metros.csv", index=False)
    pd.concat(fields).to_csv(OUT / "field.csv", index=False)

    print(f"\n{len(out)} metros -> out/metros.csv, out/field.csv\n")
    show = out.head(25)[["name", "compass", "bearing", "strength", "r2", "radial", "pop"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))


if __name__ == "__main__":
    main()
