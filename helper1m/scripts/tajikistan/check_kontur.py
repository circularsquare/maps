"""Independent check: Kontur population (400 m H3 hexagons, release 2023-11-01)
summed inside each city/district polygon, against the bulletin's 1 Jan 2024
figure. Kontur is scaled to a different national total, so units are compared
as shares of the national total (ratio = Kontur share / official share).
Read-only on everything; prints a table."""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parents[2] / "data" / "tajikistan"
YEAR = 2024

adm2 = gpd.read_file(D / "boundaries" / "adm2.gpkg")
k = gpd.read_file(D / "raw" / "kontur_population_TJ_20231101.gpkg")
k["geometry"] = k.geometry.centroid.to_crs(4326)  # hex centres; Kontur is EPSG:3857
j = gpd.sjoin(k[["population", "geometry"]], adm2[["code", "geometry"]], how="left")
ks = j.groupby("code")["population"].sum()
print(f"Kontur total {k.population.sum():,.0f}; outside every polygon {j[j.code.isna()].population.sum():,.0f}")

pop = pd.read_csv(D / "population.csv", dtype={"code": str})
off = pop[(pop.level == 2) & (pop.year == YEAR)].set_index("code")["pop"]
df = pd.DataFrame({"name": adm2.set_index("code")["name"], "official": off, "kontur": ks}).fillna(0)
df["ratio"] = (df.kontur / df.kontur.sum()) / (df.official / df.official.sum())
# Same comparison inside each region, which takes out Kontur's regional bias
# and leaves what a misdrawn polygon would show.
df["region"] = df.index.str[:5]
g = df.groupby("region")[["official", "kontur"]].transform("sum")
df["in_region"] = (df.kontur / g.kontur) / (df.official / g.official)
reg = df.groupby("region")[["official", "kontur"]].sum()
reg["ratio"] = (reg.kontur / reg.kontur.sum()) / (reg.official / reg.official.sum())
print("Region shares, Kontur / official:")
print(reg.to_string(formatters={"official": "{:,.0f}".format, "kontur": "{:,.0f}".format,
                                "ratio": "{:.2f}".format}))
df = df.sort_values("in_region")
for col, lab in (("ratio", "national share"), ("in_region", "share of own region")):
    within = ((df[col] - 1).abs() <= 0.10).sum()
    w20 = ((df[col] - 1).abs() <= 0.20).sum()
    print(f"{lab}: {within}/{len(df)} units within 10%, {w20}/{len(df)} within 20%")
print(df.drop(columns="region").to_string(formatters={
    "official": "{:,.0f}".format, "kontur": "{:,.0f}".format,
    "ratio": "{:.2f}".format, "in_region": "{:.2f}".format}))
