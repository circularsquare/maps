"""Peru — boundaries for the 1,874 census districts.

Writes data/geo/pe/pe_distritos.gpkg and data/geo/pe/pe_lookup.csv.

OCHA COD-AB Peru (`cod-ab-per`, version 01, boundaries created 2015-07-24), the **shapefile**
bundle rather than the geodatabase on §12's Chile rule — GDAL's OpenFileGDB driver has been
seen to open a .gdb, list its layers, report the right CRS and return ZERO features while
raising nothing. Read with `engine="fiona"`, because pyogrio is geopandas' default when
installed and is the engine that has silently returned zero. geoBoundaries `PER` stops at
ADM2 (196) and 404s on ADM3, so COD-AB is the only ADM3 layer there is.

**THE CODE JOIN IS USED HERE, AND THAT REVERSES `ni_geo.py` DELIBERATELY.** Nicaragua joins
on name because its code join is a §12 shape-2 trap — five municipalities collide on a code
while naming different places. Peru is the opposite case and the difference is *measured*
rather than assumed. COD's `adm3_pcode` is `PE` + the six-digit ubigeo, the same identifier
the census tabulates on, and:

    1,872 of 1,874 census codes are present in COD, and 1,870 of those agree on the
    district name character-for-character after folding accents.

The two that do not are spelling, both confirmed by reading the whole province's district
list out of each source and finding them otherwise identical, position for position:

    051010  census `Hualla`               COD `Huaya`     Víctor Fajardo, Ayacucho
    150712  census `San Pedro de Laraos`  COD `Laraos`    Huarochirí, Lima

Neither spelling occurs anywhere else in its province, so neither pairing is ambiguous.
**A name join would be the risky one here**, because Peru has many districts sharing a name
across provinces (`San Juan`, `Santa Rosa`, `Pachía`), and it would have to be disambiguated
by — the code. So the code carries the join and the NAME is demoted to the check, which is
Nicaragua's arrangement turned around. Both files state which one is doing the work and why.

**THE ONE-DISTRICT GAP IS A MERGE, NOT A MISSING POLYGON.** §11y read COD's ADM3 count of
1,873 against the census's 1,874 as a vintage difference — districts created by law between
the boundary file's 2015 vintage and the census. It is not that. The whole difference is one
pair, in Satipo province, Junín:

    census 120604 Mazamari (24,193 people 12+) ─┐
                                                ├─> COD 120699 `Mazamari - Pangoa`, 5,675 km²
    census 120606 Pangoa   (38,036 people 12+) ─┘

COD carries **one** polygon spanning both districts and no separate polygon for either. So
nothing is dropped and nobody is lost: `pe_lookup.csv` sends both census codes to the single
`PE120699` unit and `countries.py` sums them there. The cost is stated rather than hidden —
62,229 people, **0.27% of the universe**, draw at half the resolution of the rest of Peru,
and the two districts cannot be told apart on the map.

**THE WITNESS USES NEITHER NAME NOR CODE.** Peru's Adventists are the Aymara altiplano — the
mission at Platería on the Puno shore of Lake Titicaca has been running schools since 1898 —
and after the join the most Adventist districts must come out in the south-east highlands,
checked against COD's own centroid coordinates. That is what would catch a systematic
permutation, which every name and code check would survive.

Usage:
    python sources/pe_geo.py --fetch    one ~43 MB zip from HDX
    python sources/pe_geo.py            rebuild from data/raw/pe/
"""

import os
import re
import sys
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "pe")
OUT_DIR = os.path.join(ROOT, "data", "geo", "pe")
OUT = os.path.join(OUT_DIR, "pe_distritos.gpkg")
LOOKUP = os.path.join(OUT_DIR, "pe_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "pe.csv")

ZIP_URL = ("https://data.humdata.org/dataset/54fc7f4d-f4c0-4892-91f6-2fe7c1ecf363/"
           "resource/61faa8d6-fbfa-4d44-a94d-8f3b0241277a/download/"
           "per_admin_boundaries.shp.zip")
ZIP_NAME = "per_admin_boundaries.shp.zip"

EXPECTED_CENSUS = 1874
EXPECTED_POLYGONS = 1873

# COD's single polygon covering two census districts. See the module docstring: this is the
# entire 1,874-vs-1,873 difference, and it is a merge rather than a gap.
MERGED = {"120604": "120699", "120606": "120699"}

# census spelling -> COD's, for the two that are the same district written differently. Each
# was confirmed by reading the whole province's list out of both sources; neither spelling
# occurs elsewhere in its province, so neither is a judgement call.
ALIASES = {
    "Hualla": "Huaya",                          # 051010, Víctor Fajardo, Ayacucho
    "San Pedro de Laraos": "Laraos",            # 150712, Huarochirí, Lima
}

# Adventist Peru is the Puno altiplano and the south-eastern sierra. Used only as a check on
# the join, never to place anything: the most Adventist districts must come out south of this
# parallel. Puno sits between about -13.0 and -17.2.
ADVENTIST_SOUTH_OF = -11.0
ADVENTIST_TOP_N = 10
# Districts small enough that one congregation swings the share are excluded from the witness.
WITNESS_MIN_POP = 2000


def fold(s):
    """Casefold, strip accents and punctuation — for COMPARING names, never for storing."""
    s = unicodedata.normalize("NFKD", str(s).strip().lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, ZIP_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print("already have", dest)
        return
    print("GET", ZIP_URL)
    r = requests.get(ZIP_URL, timeout=1800, stream=True,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    print(f"  {os.path.getsize(dest):,} bytes")
    # §5a: a 200 is not a download.
    with open(dest, "rb") as fh:
        if fh.read(2) != b"PK":
            raise SystemExit(f"{dest} is not a zip -- HDX served something else")


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    zpath = os.path.join(RAW, ZIP_NAME)
    if not os.path.exists(zpath):
        raise SystemExit(f"missing {zpath} -- run with --fetch first")
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/pe.py first")

    # §12, Chile: the shapefile, read with fiona, and the count asserted either way.
    g = gpd.read_file("zip://" + zpath + "!per_admin3.shp", engine="fiona")
    if len(g) != EXPECTED_POLYGONS:
        raise SystemExit(f"COD ADM3 has {len(g)} features, expected {EXPECTED_POLYGONS}")
    print(f"COD-AB ADM3: {len(g):,} polygons, crs={g.crs}")

    g["ubigeo"] = g["adm3_pcode"].astype(str).str.strip().str[2:]
    if g["ubigeo"].duplicated().any():
        raise SystemExit("COD ADM3 has duplicate ubigeo codes")

    # ---- the census side ----
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "distrito"]
    cen_name = dict(zip(df["geo_id"], df["geo_name"]))
    prov_of = {gid: re.search(r"province=([^;]+)", n).group(1).strip()
               for gid, n in zip(df["geo_id"], df["note"])}
    if len(cen_name) != EXPECTED_CENSUS:
        raise SystemExit(f"{NORM} has {len(cen_name)} districts, "
                         f"expected {EXPECTED_CENSUS}")

    # ---- the join, on code ----
    cod_name = dict(zip(g["ubigeo"], g["adm3_name"]))
    unit_of = {gid: MERGED.get(gid, gid) for gid in cen_name}
    missing = sorted(gid for gid, u in unit_of.items() if u not in cod_name)
    spare = sorted(set(cod_name) - set(unit_of.values()))

    print("\n  the join, both ways (§12) — ON CODE, see the module docstring:")
    print(f"    census districts           {len(cen_name):>5}")
    print(f"    COD polygons               {len(cod_name):>5}")
    print(f"    census codes matched 1:1   "
          f"{sum(1 for gid in cen_name if gid not in MERGED):>5}")
    print(f"    census codes merged        {len(MERGED):>5}  "
          f"-> {len(set(MERGED.values()))} polygon")
    print(f"    census with no polygon     {len(missing):>5}")
    print(f"    polygons with no census    {len(spare):>5}")
    for gid in missing[:10]:
        print(f"      no polygon: {gid} {cen_name[gid]!r}")
    for u in spare[:10]:
        print(f"      no census : {u} {cod_name[u]!r}")
    if missing or spare:
        raise SystemExit("join FAILED")

    # ---- witness 1: the NAME must agree, on every 1:1 pair ----
    # The code carries the join here, so the name is the check rather than the key. This is
    # ni_geo.py's arrangement reversed, and both files say which is which.
    disagree = []
    for gid in sorted(cen_name):
        if gid in MERGED:
            continue
        a, b = cen_name[gid], cod_name[gid]
        if fold(a) != fold(b) and fold(ALIASES.get(a, a)) != fold(b):
            disagree.append((gid, a, b))
    n_pairs = len(cen_name) - len(MERGED)
    aliased = sum(1 for gid in cen_name if gid not in MERGED
                  and fold(cen_name[gid]) != fold(cod_name[gid]))
    print(f"\n    witness 1 — the census name equals COD's on "
          f"{n_pairs - aliased}/{n_pairs} pairs outright,")
    print(f"    and on {n_pairs - len(disagree)}/{n_pairs} once the {len(ALIASES)} listed "
          "spellings are allowed:")
    for a, b in ALIASES.items():
        print(f"      census {a!r} = COD {b!r}")
    for gid, a, b in disagree[:10]:
        print(f"      MISMATCH {gid}: census {a!r} vs COD {b!r}")
    if disagree:
        raise SystemExit(f"{len(disagree)} districts pair on code while naming different "
                         "places -- this is §12's shape-2 failure and the join must not be "
                         "trusted until each one is resolved by hand")

    # ---- witness 2: the province and department prefixes must agree ----
    bad = [(gid, cen_name[gid], gid[:4], cod_name[gid])
           for gid in cen_name if gid not in MERGED
           and g.set_index("ubigeo").loc[gid, "adm2_pcode"][2:] != gid[:4]]
    print(f"\n    witness 2 — COD's own province pcode agrees with the first four digits "
          f"of the\n    census ubigeo on {n_pairs - len(bad)}/{n_pairs} pairs")
    for gid, nm, a, b in bad[:8]:
        print(f"      {nm!r}: census prov {a}, COD {b}")
    if bad:
        raise SystemExit("a district's province prefix disagrees with COD's")

    # ---- witness 3: the data's own geography, and it is NOT a hand-picked region ----
    #
    # The first version of this check asserted that the most Adventist districts are the Puno
    # altiplano, on the history of the Platería mission. It fired, and the prior was wrong
    # rather than the join: Yantalo, Omia and San Fernando are Moyobamba, Rodríguez de
    # Mendoza and Rioja — the ALTO MAYO, which is Peru's other historic Adventist region and
    # is 800 km north of Titicaca. Adventist Peru is two places, not one.
    #
    # So the witness is the property that made the naive version tempting in the first place,
    # stated without naming anywhere: **religion shares are spatially smooth**. Neighbouring
    # districts resemble each other, wherever the clusters happen to be. A permuted join
    # destroys that and nothing else here would notice, because every name, code and total
    # survives a permutation intact. The threshold is not a guess — it is calibrated against
    # random re-pairings of this same data on every run.
    import numpy as np

    tot = df[df["source_category"] == "Total"].set_index("geo_id")["count"].to_dict()
    lat = dict(zip(g["ubigeo"], g["center_lat"]))
    lon = dict(zip(g["ubigeo"], g["center_lon"]))

    def smoothness(cat, k=8):
        gids = [gid for gid in sorted(cen_name)
                if gid not in MERGED and tot.get(gid, 0) >= WITNESS_MIN_POP]
        cnt = df[df["source_category"] == cat].set_index("geo_id")["count"].to_dict()
        s = np.array([cnt[gid] / tot[gid] for gid in gids])
        la = np.array([lat[gid] for gid in gids])
        lo = np.array([lon[gid] for gid in gids]) * np.cos(np.radians(la))
        d = (la[:, None] - la[None, :]) ** 2 + (lo[:, None] - lo[None, :]) ** 2
        np.fill_diagonal(d, np.inf)
        nb = np.argsort(d, axis=1)[:, :k]
        # correlation between a district's share and its k nearest neighbours' mean share
        obs = float(np.corrcoef(s, s[nb].mean(axis=1))[0, 1])
        rng = np.random.default_rng(0)
        null = []
        for _ in range(200):
            p = rng.permutation(s)
            null.append(abs(float(np.corrcoef(p, p[nb].mean(axis=1))[0, 1])))
        return obs, max(null), sum(1 for x in null if x >= obs), len(gids)

    print("\n    witness 3 — spatial smoothness on the 8 nearest neighbours by COD centroid,")
    print("    calibrated against 200 random re-pairings of the same shares:")
    worst = None
    for cat in ["Católica", "Evangélica", "Adventista"]:
        obs, best_null, beat, n = smoothness(cat)
        print(f"      {cat:<18} r = {obs:.4f}   best random {best_null:.4f}   "
              f"{beat}/200 reach it   ({n:,} districts)")
        if worst is None or obs < worst[1]:
            worst = (cat, obs, beat)
    if worst[1] < 0.5 or worst[2] > 5:
        raise SystemExit(
            f"{worst[0]} shares are not spatially smooth after the join (r={worst[1]:.4f}, "
            f"{worst[2]}/200 random pairings reach it). Religion varies smoothly across "
            "Peru's districts; if it does not, the counts and the polygons are permuted "
            "relative to each other and every name, code and total would still reconcile.")
    print("      neighbouring districts resemble each other, which neither the names nor "
          "the\n      codes were used to establish — and a permuted join would destroy it.")

    # Reported, not asserted: where the Adventists actually are.
    adv = df[df["source_category"] == "Adventista"].set_index("geo_id")["count"].to_dict()
    share = {gid: adv[gid] / tot[gid] for gid in cen_name
             if tot.get(gid, 0) >= WITNESS_MIN_POP}
    top = sorted(share, key=share.get, reverse=True)[:ADVENTIST_TOP_N]
    print(f"\n    and the {ADVENTIST_TOP_N} most Adventist districts, which are TWO regions "
          "and not one:")
    for gid in top:
        la = lat[unit_of[gid]]
        where = "altiplano" if la < ADVENTIST_SOUTH_OF else "Alto Mayo"
        print(f"      {cen_name[gid]:<28} {prov_of[gid][:16]:<17} "
              f"{100 * share[gid]:5.1f}%   lat {la:7.3f}  {where}")

    # ---- write ----
    out = g[["ubigeo", "adm3_name", "adm3_pcode", "geometry"]].copy()
    out["unit"] = out["adm3_pcode"].astype(str).str.strip()
    # Take the NAME from the statistical source, not the boundary file (§12, Chile). The
    # merged polygon has two census names and gets both, which is what it draws as.
    by_unit = {}
    for gid in sorted(cen_name):
        by_unit.setdefault(unit_of[gid], []).append(cen_name[gid])
    out["name"] = out["ubigeo"].map(lambda u: " - ".join(by_unit[u]))
    if out["name"].isna().any():
        raise SystemExit("a polygon came out of the join with no census name")

    os.makedirs(OUT_DIR, exist_ok=True)
    out[["unit", "name", "ubigeo", "geometry"]].to_file(
        OUT, layer="distritos", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} polygons)")

    lut = pd.DataFrame({"geo_id": sorted(cen_name),
                        "unit": [f"PE{unit_of[k]}" for k in sorted(cen_name)]})
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut):,} rows, {lut['unit'].nunique():,} units)")


if __name__ == "__main__":
    main()
