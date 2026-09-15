"""Palestine — the sixteen governorates of the 2017 census and the placement grid.

Writes:
    data/geo/ps/ps_governorates.gpkg   the 16 counted units
    data/geo/ps/ps_hexes.gpkg          Kontur 400 m hexes with `unit` and `pop` (`place`),
                                       Israeli settlement population taken out
    data/geo/ps/ps_lookup.csv          unit -> pcode, census populations, Kontur before/after

Usage:
    python sources/ps_geo.py --fetch    COD-AB geojson zip (1.6 MB), OCHA communities (74 KB),
                                        Kontur PS (0.4 MB); reuses the Israel build's files
    python sources/ps_geo.py            rebuild from what is on disk

## THE BOUNDARIES ARE OCHA COD-AB `cod-ab-pse` v01 ADMIN 2

The sixteen governorates (Palestinian Authority Ministry of Planning, valid 2023-10-19, reviewed
2024-12-24, CC BY-IGO), the tier Table 3 prints. It is the same dataset whose admin 0 the Israel
build cut on (`sources/il_geo.py`, `pse_admin0.geojson`), so the two countries meet on one line
and nothing is drawn twice or left between them. COD's Jerusalem governorate includes the part
Israel annexed (J1): every OCHA community flagged `EJ` falls inside it (witness 3).

## THE JOIN IS AN AUTHORED TABLE, CHECKED THREE WAYS

PCBS and COD spell six of sixteen differently (`Tubas and the Northern Valleys`/`Tubas`,
`Qalqiliya`/`Qalqilya`, `Ramallah & Al-Bireh`/`Ramallah`, `Jericho & Al-Aghwar`/`Jericho`,
`Dier Al-Balah`/`Deir Al-Balah`, `Khan Yunis`/`Khan Younis`), so the names are not the key.

  1. COD's pcodes run in PCBS's own table order, north to south, West Bank then Gaza, and
     COD's admin 1 agrees with the territory each row sits under in Table 3.
  2. OCHA's 893 community points (pop2017 from PCBS's locality table), joined to COD's polygons
     by location and summed, against Table 2 per governorate. A third gazetteer, positions
     rather than names; asserted as a band and by a permutation test.
  3. every community OCHA flags as East Jerusalem is inside COD's Jerusalem polygon.

## KONTUR: TWO EXTRACTS, BECAUSE THE PS ONE STOPS AT ISRAEL'S JERUSALEM LINE

Kontur's `PS` extract has no hexes in the annexed part of Jerusalem; the `IL` extract has them.
Inside COD's Jerusalem governorate the PS extract holds 375,091 people and the IL extract adds
350,549. So both are read, de-duplicated on `h3` (410 hexes are in both, with identical
population, asserted), and joined on hex centroids to the governorates.

## ISRAELI SETTLEMENTS ARE TAKEN OUT OF THE WEIGHTS

Kontur models everyone, and the West Bank's weights include the Israeli settlements, which the
Palestinian census does not count: Kontur reads Ramallah & Al-Bireh at 1.85x Table 2 and
Salfit at 1.55x. Left in, Palestinian dots (Christians among them) would be placed in Modi'in
Illit and Ariel.

The Israel build already has what is needed. CBS's 2022 census units that `il_geo.py` dropped as
beyond the Green Line (267) are rebuilt with `il_geo.build_units()`, and each unit's non-Arab
count (Jews plus the register's `Others`) comes from `data/normalized/il.csv`: 723,899 people.
Each unit's count is taken off the hexes its polygon overlaps, in proportion to hex population
times the overlapping share, and no hex goes below zero. East Jerusalem's Palestinian
neighbourhoods are CBS units too, and their Arab majority means almost nothing is taken off them.

**CBS draws its small localities as 0.008 km2 placeholders** (Talmon, Shilo, Kiryat Arba, Beit El;
read from `statareas2022.geojson` itself), which overlap almost no population. Those under
`PLACEHOLDER_KM2` are replaced by a disc at the placeholder, sized to the locality's count at
`SETTLEMENT_DENSITY`. Without that, the overlap took 96 people out of Hebron governorate against
CBS's 21,950.

What is left is printed per governorate and witnessed by the OCHA community points: the share of
the kept weight within `NEAR_M` of a Palestinian community must rise where anything was removed.
"""

import contextlib
import csv
import gzip
import io
import json
import os
import shutil
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("OMP_NUM_THREADS", "6")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ps")
GEO = os.path.join(ROOT, "data", "geo", "ps")
KONTUR = os.path.join(ROOT, "data", "geo", "kontur")
NORM = os.path.join(ROOT, "data", "normalized", "ps.csv")
IL_CSV = os.path.join(ROOT, "data", "normalized", "il.csv")
IL_DROPPED = os.path.join(ROOT, "data", "geo", "il", "dropped_units.json")
sys.path.insert(0, HERE)

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}

COD_NAME = "pse_admin_boundaries.geojson.zip"
COD_URL = ("https://data.humdata.org/dataset/2caf8373-816f-458c-9913-71bddb9cab7c/resource/"
           "ca372385-4c79-4378-abf1-cb506fb98023/download/pse_admin_boundaries.geojson.zip")
COMM_NAME = "palestiniancommunities_wb_gs.zip"
COMM_URL = ("https://data.humdata.org/dataset/c7e2f4b3-6a74-4b98-b064-1e9c2d066242/resource/"
            "1936b09f-74a8-4d4d-b92b-2ee75abb21f1/download/palestiniancommunities_wb_gs.zip")
KONTUR_BASE = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
               "kontur_population_{}_20231101.gpkg.gz")

OUT_UNITS = os.path.join(GEO, "ps_governorates.gpkg")
OUT_HEXES = os.path.join(GEO, "ps_hexes.gpkg")
OUT_LOOKUP = os.path.join(GEO, "ps_lookup.csv")

# (PCBS Table 3 name, COD adm2_pcode, COD adm2_name, COD adm1_pcode), in Table 3's order.
GOVERNORATES = [
    ("Jenin", "PS0101", "Jenin", "PS01"),
    ("Tubas and the Northern Valleys", "PS0105", "Tubas", "PS01"),
    ("Tulkarm", "PS0110", "Tulkarm", "PS01"),
    ("Nablus", "PS0115", "Nablus", "PS01"),
    ("Qalqiliya", "PS0120", "Qalqilya", "PS01"),
    ("Salfit", "PS0125", "Salfit", "PS01"),
    ("Ramallah & Al-Bireh", "PS0130", "Ramallah", "PS01"),
    ("Jericho & Al-Aghwar", "PS0135", "Jericho", "PS01"),
    ("Jerusalem", "PS0140", "Jerusalem", "PS01"),
    ("Bethlehem", "PS0145", "Bethlehem", "PS01"),
    ("Hebron", "PS0150", "Hebron", "PS01"),
    ("North Gaza", "PS0255", "North Gaza", "PS02"),
    ("Gaza", "PS0260", "Gaza", "PS02"),
    ("Dier Al-Balah", "PS0265", "Deir Al-Balah", "PS02"),
    ("Khan Yunis", "PS0270", "Khan Younis", "PS02"),
    ("Rafah", "PS0275", "Rafah", "PS02"),
]
UNITS = 16
UTM = 32636                      # UTM 36N; Gaza and the West Bank lie in 34.2-35.6E

PLACEHOLDER_KM2 = 0.05           # CBS polygons smaller than this are placeholders
SETTLEMENT_DENSITY = 4_000.0     # people per km2 used to size a placeholder's disc
MIN_DISC_M = 300.0
NEAR_M = 1_500.0                 # witness: "within reach of a Palestinian community"

# Kontur 2023-11 against the 2017 count; PCBS's own mid-2023 projection is about 1.13x its
# 2017 total. Per-unit ratios after the settlement subtraction ran 0.59 (Tubas) to 1.81
# (Jericho) when this was written; the band is for a broken download, not for shape.
KONTUR_UNIT_MIN, KONTUR_UNIT_MAX = 0.45, 2.3
COMM_BAND = 0.25                 # witness 2: communities' sum within 25% of Table 2
# Named, not a looser band. OCHA's file gives Dier Al-Balah 193,188 against Table 2's 269,830
# (0.716x on 2026-09-15) because it has eight of the governorate's eleven Table 25 localities:
# An Nuseirat (54,851), Al Bureij (15,491) and Al Maghazi (9,670) are missing while their camps
# are present, and those three are exactly the 80,012 between OCHA's sum and Table 25's
# 273,200. The permutation test, which is what pins the join, passes with nothing close to it.
COMM_BAND_EXCEPT = {"Dier Al-Balah"}

# Witness 3's two known exceptions: OCHA flags them East Jerusalem, both have no population
# figure, neither is among the 21 J1 localities PCBS lists under Table 25, and both sit just
# south of Beit Safafa inside COD's Bethlehem polygon. A third would mean the line has moved.
EJ_OUTSIDE_KNOWN = {"khirbet Khamis", "Umm al-'Asafir"}


def _get(url, dst, magic, min_size):
    import requests

    if os.path.exists(dst) and os.path.getsize(dst) > min_size:
        print(f"  have {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")
        return
    print("GET", url)
    r = requests.get(url, timeout=1800, stream=True, headers=UA)
    r.raise_for_status()
    with open(dst + ".part", "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    with open(dst + ".part", "rb") as fh:
        head = fh.read(len(magic))
    if head != magic:                                   # §5a: a 200 is not a download.
        raise SystemExit(f"{dst}: starts {head!r}, expected {magic!r}")
    os.replace(dst + ".part", dst)
    print(f"  got {os.path.getsize(dst):,} bytes")


def _unpack(cc):
    gz = os.path.join(KONTUR, f"kontur_population_{cc}_20231101.gpkg.gz")
    gpkg = gz[:-3]
    if not (os.path.exists(gpkg) and os.path.getsize(gpkg) > 200_000):
        with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(gpkg + ".part", gpkg)
    with open(gpkg, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")
    return gpkg


def fetch():
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(KONTUR, exist_ok=True)
    _get(COD_URL, os.path.join(RAW, COD_NAME), b"PK", 1_000_000)
    _get(COMM_URL, os.path.join(RAW, COMM_NAME), b"PK", 50_000)
    for cc, size in (("PS", 300_000), ("IL", 600_000)):
        _get(KONTUR_BASE.format(cc), os.path.join(KONTUR, f"kontur_population_{cc}_20231101.gpkg.gz"),
             b"\x1f\x8b", size)
        _unpack(cc)


def read_cod():
    import geopandas as gpd

    with zipfile.ZipFile(os.path.join(RAW, COD_NAME)) as z:
        g = gpd.read_file(io.BytesIO(z.read("pse_admin2.geojson")))
    if len(g) != UNITS:
        raise SystemExit(f"COD-AB pse admin2 has {len(g)} features, expected {UNITS}")
    return g


def read_communities():
    import geopandas as gpd
    import pandas as pd

    d = os.path.join(RAW, "communities")
    with zipfile.ZipFile(os.path.join(RAW, COMM_NAME)) as z:
        z.extractall(d)
    g = gpd.read_file(os.path.join(d, "PalestinianCommunities_WB_GS.shp"), engine="fiona")
    if len(g) == 0:
        raise SystemExit("OCHA communities read as ZERO features")
    # 164 rows carry `NA`, an empty cell, or `with <neighbour>` (counted with another community).
    g["pop"] = pd.to_numeric(g["pop2017"], errors="coerce").fillna(0.0)
    return g.to_crs(4326)


def census_totals():
    import pandas as pd

    df = pd.read_csv(NORM, keep_default_na=False, na_values=[""])
    tot = df[df["source_category"] == "Total"].set_index("geo_id")["count"].astype(int)
    note = df[df["source_category"] == "Total"].set_index("geo_id")["note"]
    t2 = note.str.extract(r"Table 2 counted pop=(\d+)")[0].astype(int)
    if len(tot) != UNITS:
        raise SystemExit(f"{len(tot)} Total rows in ps.csv, expected {UNITS}")
    return tot, t2


def settlement_units():
    """CBS 2022 census units beyond the Green Line, with their non-Arab counts, in UTM."""
    import il_geo
    import pandas as pd

    for p in (IL_CSV, IL_DROPPED):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- the Israel build (sources/il.py, il_geo.py) "
                             "must be on disk; it supplies the settlement counts")
    with contextlib.redirect_stdout(io.StringIO()):
        units = il_geo.build_units()
    dropped = set(json.load(open(IL_DROPPED, encoding="utf-8")))
    du = units[units["unit"].isin(dropped)].copy()
    if len(du) != len(dropped):
        raise SystemExit(f"rebuilt {len(du)} of {len(dropped)} dropped CBS units")

    df = pd.read_csv(IL_CSV, dtype={"geo_id": str}, keep_default_na=False, na_values=[""],
                     low_memory=False)
    df = df[df["geo_id"].isin(dropped)].copy()
    df["grp"] = df["source_category"].str.split(" [", regex=False).str[0]
    groups = set(df["grp"])
    if not groups <= {"Jews", "Muslims", "Christians", "Druze", "Others"}:
        raise SystemExit(f"unexpected il.csv groups beyond the line: {sorted(groups)}")
    t = df.pivot_table(index="geo_id", columns="grp", values="count", aggfunc="sum", fill_value=0)
    t["total"] = t.sum(axis=1)
    t["nonarab"] = t.get("Jews", 0) + t.get("Others", 0)
    du = du.merge(t[["total", "nonarab"]], left_on="unit", right_index=True, how="left")
    if du["nonarab"].isna().any():
        raise SystemExit(f"dropped CBS units with no il.csv rows: "
                         f"{du.loc[du['nonarab'].isna(), 'unit'].tolist()[:10]}")
    du = du.to_crs(UTM)
    du["km2"] = du.geometry.area / 1e6
    ph = du["km2"] < PLACEHOLDER_KM2
    radius = ((du["nonarab"] / SETTLEMENT_DENSITY) * 1e6 / 3.141592653589793) ** 0.5
    radius = radius.clip(lower=MIN_DISC_M)
    du.loc[ph, "geometry"] = du.loc[ph].geometry.centroid.buffer(radius[ph])
    print(f"  CBS 2022 units beyond the Green Line: {len(du)}, {du['total'].sum():,.0f} people, "
          f"{du['nonarab'].sum():,.0f} Jews and Others")
    print(f"  {int(ph.sum())} are placeholders under {PLACEHOLDER_KM2} km2 "
          f"({du.loc[ph, 'nonarab'].sum():,.0f} people), replaced by discs at "
          f"{SETTLEMENT_DENSITY:,.0f}/km2")
    return du


def main():
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    from scipy import stats

    if "--fetch" in sys.argv:
        fetch()
    for p in (os.path.join(RAW, COD_NAME), os.path.join(RAW, COMM_NAME), NORM):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- run sources/ps.py --fetch and "
                             "sources/ps_geo.py --fetch")
    gp_ps, gp_il = _unpack("PS"), _unpack("IL")
    os.makedirs(GEO, exist_ok=True)

    tot, t2 = census_totals()
    names = [g[0] for g in GOVERNORATES]
    if sorted(tot.index) != sorted(names):
        raise SystemExit(f"ps.csv governorates and GOVERNORATES differ: "
                         f"{sorted(set(tot.index) ^ set(names))}")

    # ---- 1. COD-AB and the authored join
    cod = read_cod()
    print(f"COD-AB pse admin2: {len(cod)} features, version {sorted(set(cod['version']))}, "
          f"valid_on {sorted(set(map(str, cod['valid_on'])))}")
    by_code = {r["adm2_pcode"]: r for _i, r in cod.iterrows()}
    for pcbs, code, cname, adm1 in GOVERNORATES:
        r = by_code.get(code)
        if r is None or r["adm2_name"] != cname or r["adm1_pcode"] != adm1:
            raise SystemExit(f"{pcbs}: COD has {None if r is None else (r['adm2_name'], r['adm1_pcode'])}, "
                             f"expected ({cname}, {adm1})")
    if [g[1] for g in GOVERNORATES] != sorted(g[1] for g in GOVERNORATES):
        raise SystemExit("COD's pcodes are not in Table 3's order")
    print("  witness 1: all 16 pcodes present with the expected names; pcodes run in Table 3's "
          "order; admin 1 matches each row's territory")

    units = cod.copy()
    units["unit"] = units["adm2_pcode"].map({g[1]: g[0] for g in GOVERNORATES})
    units["census_pop"] = units["unit"].map(tot).astype(int)
    units["counted_pop"] = units["unit"].map(t2).astype(int)
    units = units[["unit", "adm2_pcode", "adm2_name", "adm1_pcode", "census_pop", "counted_pop",
                   "geometry"]].to_crs(4326)

    # ---- witness 2: OCHA communities by location against Table 2
    com = read_communities()
    cj = gpd.sjoin(com, units[["unit", "geometry"]], how="left", predicate="within")
    cj = cj[~cj.index.duplicated(keep="first")]
    unplaced = cj["unit"].isna()
    print(f"\n  witness 2: {len(com)} OCHA community points, {int(unplaced.sum())} outside every "
          f"governorate ({cj.loc[unplaced, 'pop'].sum():,.0f} people)")
    csum = cj.groupby("unit")["pop"].sum().reindex(names).fillna(0)
    nofig = cj[cj["pop"] <= 0].groupby("unit").size().reindex(names).fillna(0).astype(int)
    ratio = csum / t2.reindex(names)
    for n in names:
        print(f"      {n:<31} communities {csum[n]:>9,.0f}  Table 2 {t2[n]:>9,}  {ratio[n]:.3f}"
              f"   ({nofig[n]} with no pop2017)")
    rho = stats.spearmanr(csum.to_numpy(), t2.reindex(names).to_numpy()).statistic
    rng = np.random.default_rng(0)
    a = csum.to_numpy()
    b = t2.reindex(names).to_numpy()
    err = np.abs(np.log(np.maximum(a, 1) / b)).sum()
    perm = np.array([np.abs(np.log(np.maximum(a, 1) / rng.permutation(b))).sum()
                     for _ in range(20_000)])
    beaten = int((perm <= err).sum())
    print(f"      rho {rho:+.3f}; summed |log ratio| {err:.3f}, and {beaten} of 20,000 random "
          f"pairings do as well (best {perm.min():.3f})")
    off = ratio.drop(index=list(COMM_BAND_EXCEPT))
    if beaten or (off - 1).abs().max() > COMM_BAND:
        raise SystemExit("the communities do not confirm the governorate join -- STOP")
    print(f"      the other {len(off)} within {COMM_BAND:.0%}; "
          f"{', '.join(sorted(COMM_BAND_EXCEPT))} excepted by name (see COMM_BAND_EXCEPT)")

    # ---- witness 3: East Jerusalem communities are inside COD's Jerusalem
    ej = cj[pd.to_numeric(cj["EJ"], errors="coerce").fillna(0) == 1]
    where = ej["unit"].value_counts(dropna=False).to_dict()
    outside = set(ej.loc[ej["unit"] != "Jerusalem", "PCBS_NAME"])
    inside_pop = ej.loc[ej["unit"] == "Jerusalem", "pop"].sum()
    print(f"\n  witness 3: {len(ej)} communities flagged EJ, by governorate {where}; "
          f"{inside_pop:,.0f} people in those inside Jerusalem; outside: {sorted(outside)}")
    if len(ej) == 0 or outside != EJ_OUTSIDE_KNOWN or ej.loc[ej["unit"] != "Jerusalem", "pop"].sum():
        raise SystemExit("OCHA's East Jerusalem communities are not inside COD's Jerusalem "
                         f"governorate beyond the two known points {sorted(EJ_OUTSIDE_KNOWN)}")

    units.to_file(OUT_UNITS, layer="governorates", driver="GPKG")
    print(f"\nwrote {OUT_UNITS} ({len(units)} governorates)")

    # ---- 2. Kontur, both extracts
    ps = gpd.read_file(gp_ps)
    il = gpd.read_file(gp_il)
    if len(ps) == 0 or len(il) == 0:
        raise SystemExit("a Kontur extract read as ZERO features")
    both = pd.concat([ps, il], ignore_index=True)
    dupmask = both.duplicated("h3", keep=False)
    same = both[dupmask].groupby("h3")["population"].nunique()
    if not (same == 1).all():
        raise SystemExit(f"{int((same > 1).sum())} hexes are in both extracts with different "
                         "populations")
    hx = gpd.GeoDataFrame(both[~both.duplicated("h3")].reset_index(drop=True), crs=ps.crs)
    in_ps = set(ps["h3"])
    print(f"\nKontur PS {len(ps):,} hexes + IL {len(il):,}, {int(same.size)} in both "
          f"(identical); {len(hx):,} distinct")

    cent = gpd.GeoDataFrame({"h3": hx["h3"]}, geometry=hx.geometry.centroid,
                            crs=hx.crs).to_crs(4326)
    j = gpd.sjoin(cent, units[["unit", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(cent.index)
    hx["unit"] = j["unit"].to_numpy()
    hx = hx[hx["unit"].notna()].to_crs(UTM).reset_index(drop=True)
    hx["src"] = np.where(hx["h3"].isin(in_ps), "PS", "IL")
    print("  inside the governorates, by extract:")
    print("    " + hx.groupby(["unit", "src"])["population"].sum().unstack(fill_value=0)
          .reindex(names).round(0).to_string().replace("\n", "\n    "))

    # ---- 3. take the settlements out
    du = settlement_units()
    hx["hid"] = np.arange(len(hx))
    hx["hex_area"] = hx.geometry.area
    ov = gpd.overlay(hx[["hid", "hex_area", "population", "geometry"]],
                     du[["unit", "nonarab", "geometry"]].rename(columns={"unit": "cbs"}),
                     how="intersection", keep_geom_type=True)
    ov["w"] = ov["population"] * ov.geometry.area / ov["hex_area"]
    wsum = ov.groupby("cbs")["w"].transform("sum")
    ov["take"] = np.where(wsum > 0, ov["nonarab"] * ov["w"] / wsum, 0.0)
    take = ov.groupby("hid")["take"].sum()
    hx["take"] = hx["hid"].map(take).fillna(0.0)
    hx["pop"] = (hx["population"] - hx["take"]).clip(lower=0.0)
    hx["removed"] = hx["population"] - hx["pop"]
    no_overlap = du.loc[~du["unit"].isin(set(ov["cbs"])), "nonarab"].sum()
    print(f"  asked to remove {hx['take'].sum():,.0f}; removed {hx['removed'].sum():,.0f} "
          f"(hexes floor at zero); CBS units overlapping no populated hex hold {no_overlap:,.0f}")

    # ---- witness 4: the kept weight moves towards Palestinian communities
    cm = com[com["pop"] > 0].to_crs(UTM)
    near = gpd.sjoin_nearest(gpd.GeoDataFrame(hx[["hid"]], geometry=hx.geometry.centroid,
                                              crs=UTM),
                             cm[["geometry"]], how="left", max_distance=NEAR_M,
                             distance_col="d")
    near = near[~near.index.duplicated(keep="first")]
    hx["near"] = near["d"].notna().to_numpy()
    per = hx.groupby("unit").apply(lambda d: pd.Series({
        "kontur": d["population"].sum(), "removed": d["removed"].sum(), "kept": d["pop"].sum(),
        "near_before": (d["population"] * d["near"]).sum() / d["population"].sum(),
        "near_after": (d["pop"] * d["near"]).sum() / max(d["pop"].sum(), 1.0),
    })).reindex(names)
    per["t2"] = t2.reindex(names)
    per["before"] = per["kontur"] / per["t2"]
    per["after"] = per["kept"] / per["t2"]
    print(f"\n  witness 4: share of weight within {NEAR_M:,.0f} m of an OCHA community, and "
          "Kontur over Table 2")
    print(f"    {'governorate':<31}{'removed':>9}{'near before':>13}{'after':>7}"
          f"{'ratio before':>14}{'after':>7}")
    for n, r in per.iterrows():
        print(f"    {n:<31}{r['removed']:>9,.0f}{r['near_before']:>13.3f}{r['near_after']:>7.3f}"
              f"{r['before']:>14.2f}{r['after']:>7.2f}")
    worse = per[(per["removed"] > 1_000) & (per["near_after"] < per["near_before"] - 1e-9)]
    if len(worse):
        raise SystemExit(f"removing settlements moved weight AWAY from Palestinian communities "
                         f"in {worse.index.tolist()} -- STOP")
    bad = per[(per["after"] < KONTUR_UNIT_MIN) | (per["after"] > KONTUR_UNIT_MAX)]
    if len(bad):
        raise SystemExit(f"Kontur/Table 2 outside [{KONTUR_UNIT_MIN}, {KONTUR_UNIT_MAX}] for "
                         f"{bad.index.tolist()}")
    missing = sorted(set(names) - set(hx.loc[hx["pop"] > 0, "unit"]))
    if missing:
        raise SystemExit(f"governorates with no populated hex: {missing}")

    out = hx.loc[hx["pop"] > 0, ["unit", "pop", "geometry"]].to_crs(4326)
    out.to_file(OUT_HEXES, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT_HEXES} ({len(out):,} hexes, {out['pop'].sum():,.0f} people of weight)")

    with open(OUT_LOOKUP, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "adm2_pcode", "palestinians_t3", "counted_t2", "kontur_2023",
                    "settlements_removed", "kontur_kept", "kept_over_t2", "communities_pop2017"])
        for pcbs, code, _c, _a in GOVERNORATES:
            r = per.loc[pcbs]
            w.writerow([pcbs, code, int(tot[pcbs]), int(t2[pcbs]), round(r["kontur"]),
                        round(r["removed"]), round(r["kept"]), round(r["after"], 3),
                        round(csum[pcbs])])
    print(f"wrote {OUT_LOOKUP}")


if __name__ == "__main__":
    main()
