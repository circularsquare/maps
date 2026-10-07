"""Ecuador: 221 cantons and their placement layer.

    python sources/ec_geo.py

Writes data/geo/ec/ec_lookup.csv (INEC province + canton name -> COD-AB pcode) and
data/geo/ec/ec_hexes.gpkg (Kontur 400 m hexes keyed to canton pcode, with `pop`).

BOUNDARIES: COD-AB Ecuador 2024 (OCHA / INEC), ADM2, read in place from religiondots'
data/raw/ec/ecu_adm_2024.zip (read-only). 221 cantons, the same count as the census tabulado,
and since the 2015-16 referendums there are no undelimited zones left in either.

THE JOIN is by name inside the province: the tabulado prints names only. Names are folded
(case, accents, punctuation, a parenthetical); all 221 match with no alias (2026-10-05), since
COD-AB 2024 is INEC's own DPA. ALIAS is there for a later release that renames one.
Asserted both ways (every census canton finds one polygon and every polygon one canton), and
witnessed twice by things the names do not decide:
  1. ORDER. INEC lists cantons inside a province in its own DPA code order, and COD's pcode is
     EC + that code; so, read down the sheet, the joined pcodes must rise inside every province.
     A wrong twin or a mis-pinned alias breaks the run.
  2. KONTUR. Kontur 2023 people per canton against the census count, normalised by the national
     ratio, printed at both ends, with the log correlation against 500 shuffled pairings.

PLACEMENT: religiondots' ec_hexes.gpkg is plain Kontur EC (2023-11-01), keyed to province by
hex centroid with the hexes outside every province already dropped (../religiondots/sources/
ec_grid.py). It is re-keyed here to canton by the same rule, centroid taken in Kontur's own
EPSG:3857. A hex whose centroid falls in no canton (none expected: provinces and cantons are one
release) goes to the nearest canton of its own province, and the count is printed.
"""
import os
import re
import sys
import unicodedata

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RD = os.path.join(ROOT, "..", "religiondots")
COD_ZIP = os.path.join(RD, "data", "raw", "ec", "ecu_adm_2024.zip")
RD_HEX = os.path.join(RD, "data", "geo", "ec", "ec_hexes.gpkg")
XLSX = os.path.join(ROOT, "data", "raw", "ec", "2022_CPV_Autoidentificacion_Cultura.xlsx")
GEO = os.path.join(ROOT, "data", "geo", "ec")
LOOKUP = os.path.join(GEO, "ec_lookup.csv")
HEXES = os.path.join(GEO, "ec_hexes.gpkg")

EXPECTED = 221
CENSUS_POPULATION = 16_938_986

# (province, INEC's canton) folded -> COD-AB's canton folded. Empty: none needed. The order
# witness below would check any added.
ALIAS = {}


def fold(s):
    s = re.sub(r"\(.*?\)", " ", str(s))
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def census_cantons():
    """[(province, canton, population)] in the sheet's order, from table 1.1."""
    import openpyxl
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    out = []
    for r in wb["1.1"].iter_rows(min_row=11, values_only=True):
        p, c = r[1], r[2]
        if p is None or c is None or str(p).startswith(("Nota", "Total Nacional")):
            continue
        p, c = str(p).strip(), str(c).strip()
        if c == f"Total {p}":
            continue
        if str(r[3]).strip() == f"Total {c}" and str(r[4]).strip() == f"Total {c}":
            out.append((p, c, int(r[5])))
    return out


def main():
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    cens = census_cantons()
    assert len(cens) == EXPECTED, len(cens)
    assert sum(n for _, _, n in cens) == CENSUS_POPULATION

    cod = gpd.read_file(f"/vsizip/{os.path.abspath(COD_ZIP)}/ecu_adm_adm2_2024.shp")
    assert len(cod) == EXPECTED, len(cod)
    assert cod["ADM2_PCODE"].is_unique
    cod["kp"] = cod["ADM1_ES"].map(fold)
    cod["kc"] = cod["ADM2_ES"].map(fold)
    assert not cod.duplicated(["kp", "kc"]).any(), "two COD cantons fold to one name"
    idx = {(r.kp, r.kc): r.ADM2_PCODE for r in cod.itertuples()}

    rows, miss = [], []
    for p, c, n in cens:
        kp, kc = fold(p), fold(c)
        kc = ALIAS.get((kp, kc), kc)
        code = idx.get((kp, kc))
        if code is None:
            miss.append((p, c))
        rows.append(dict(province=p, canton=c, unit=code, census_pop=n))
    if miss:
        used = {r["unit"] for r in rows}
        left = cod.loc[~cod["ADM2_PCODE"].isin(used), ["ADM1_ES", "ADM2_ES", "ADM2_PCODE"]]
        print("unmatched census cantons:", miss)
        print("unmatched COD cantons:\n" + left.to_string())
        raise SystemExit("add ALIAS entries")
    lut = pd.DataFrame(rows)
    assert lut["unit"].is_unique and set(lut["unit"]) == set(cod["ADM2_PCODE"]), "not a bijection"
    print(f"  name join: {EXPECTED} census cantons <-> {EXPECTED} COD polygons, one to one "
          f"({len(ALIAS)} pinned aliases)")

    # witness 1: pcodes rise down the sheet inside each province; and the province part agrees
    bad = []
    for p, g in lut.groupby("province", sort=False):
        codes = list(g["unit"])
        if codes != sorted(codes):
            bad.append(p)
        if len({c[:4] for c in codes}) != 1:
            bad.append(p + " (spans two provinces)")
    if bad:
        raise SystemExit(f"code order broken in {bad}: a wrong twin or a wrong alias")
    print("  order witness: pcodes rise in the sheet's order in all 24 provinces")

    # placement: re-key religiondots' province hexes to cantons
    hx = gpd.read_file(RD_HEX)
    if len(hx) == 0:
        raise SystemExit("religiondots' ec_hexes.gpkg has ZERO features")
    pts = gpd.GeoDataFrame({"pop": hx["pop"].to_numpy(dtype=float), "prov": hx["unit"].to_numpy()},
                           geometry=hx.geometry.to_crs(3857).centroid, crs=3857)
    units = cod[["ADM2_PCODE", "ADM1_PCODE", "geometry"]].rename(columns={"ADM2_PCODE": "unit"})
    j = gpd.sjoin(pts, units.to_crs(3857)[["unit", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(pts.index)
    out = j["unit"].isna()
    print(f"  {len(hx):,} hexes, {pts['pop'].sum():,.0f} Kontur people; "
          f"{int(out.sum())} hexes ({pts.loc[out, 'pop'].sum():,.0f} people) in no canton")
    if out.any():
        u3 = units.to_crs(32717)
        for i in np.flatnonzero(out.to_numpy()):
            prov = pts["prov"].iat[i]
            cand = u3[u3["ADM1_PCODE"] == prov]
            d = cand.distance(pts.geometry.iloc[[i]].to_crs(32717).iat[0])
            j.iloc[i, j.columns.get_loc("unit")] = cand.loc[d.idxmin(), "unit"]
            if d.min() > 2000:
                print(f"    !! snapped {d.min():,.0f} m to {cand.loc[d.idxmin(), 'unit']}")
    # province of the canton must be the hex's province
    wrong = (j["unit"].str[:4] != pts["prov"]).sum()
    print(f"  hexes whose canton is not in their religiondots province: {wrong}")
    assert wrong < 0.001 * len(hx), wrong
    layer = gpd.GeoDataFrame({"unit": j["unit"].to_numpy(), "pop": pts["pop"].to_numpy()},
                             geometry=hx.geometry.to_numpy(), crs=hx.crs)

    per = layer.groupby("unit")["pop"].sum()
    empty = sorted(set(lut["unit"]) - set(per.index[per > 0]))
    assert not empty, f"cantons with no populated hex: {empty}"
    print(f"  every canton has populated hexes ({layer.groupby('unit').size().min()} at least)")

    # witness 2: Kontur against the census per canton
    c = lut.set_index("unit")["census_pop"].astype(float)
    k = per.reindex(c.index).astype(float)
    ratio = k.sum() / c.sum()
    norm = (k / c / ratio).sort_values()
    print(f"  Kontur / census nationally {ratio:.3f}; per canton normalised p10 "
          f"{norm.quantile(.1):.2f} median {norm.median():.2f} p90 {norm.quantile(.9):.2f}")
    names = lut.set_index("unit")["canton"]
    print("  lowest: " + ", ".join(f"{names[u]} {v:.2f}" for u, v in norm.head(5).items()))
    print("  highest: " + ", ".join(f"{names[u]} {v:.2f}" for u, v in norm.tail(5).items()))
    lc, lk = np.log(c.to_numpy()), np.log(k.to_numpy())
    r = np.corrcoef(lc, lk)[0, 1]
    rng = np.random.default_rng(0)
    best = max(abs(np.corrcoef(lc, rng.permutation(lk))[0, 1]) for _ in range(500))
    print(f"  log correlation r = {r:.3f}; best of 500 shuffles {best:.3f}")
    assert r > best, "the join is not carrying information"
    outside = int(((norm < 1 / 3) | (norm > 3)).sum())
    print(f"  {outside} cantons outside a factor of 3")

    os.makedirs(GEO, exist_ok=True)
    lut.to_csv(LOOKUP + ".part", index=False, encoding="utf-8")
    os.replace(LOOKUP + ".part", LOOKUP)
    tmp = HEXES.replace(".gpkg", ".part.gpkg")
    layer.to_file(tmp, layer="hexes", driver="GPKG")
    os.replace(tmp, HEXES)
    print(f"  wrote {LOOKUP} and {HEXES} ({len(layer):,} hexes)")


if __name__ == "__main__":
    main()
