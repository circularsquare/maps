"""Botswana — the 519 ADM3 localities, the name join, and the Kontur placement layer.

Writes data/geo/bw/bw_localities.gpkg, data/geo/bw/bw_hexes.gpkg and
data/geo/bw/bw_lookup.csv.

**THE UNIT IS THE ADM3 LOCALITY AND THAT IS THE WHOLE POINT OF THE COUNTRY.** Statistics
Botswana published the 2011 religion question at named-village level, one booklet per census
district, and COD-AB's Botswana bundle carries a 519-polygon ADM3 layer that tiles the
country exactly (its area is 1.0000 of ADM0). So the counts and the polygons are at the same
grain and nothing is spread across a province.

**THE JOIN IS ON NAME AND THAT IS THE RISK.** There is no code on the census side: the
booklets print a village name and nine numbers. Names are matched WITHIN THE DISTRICT, never
across the country, and that is load-bearing rather than tidy — eight ADM3 names occur twice
in Botswana (TULI three times, in Bobonong, Mahalapye and Serowe Palapye; also MAKALAMABEDI,
BOROTSI, CHADIBE, OTSE, PHUDUHUDU, SESUNG and TOTENG), and a country-wide name lookup would
pair some of them with the wrong district's polygon while every total still reconciled.
[[reference_name_join_wrong_neighbour]] is the standing warning and this is its shape.

**WHAT ACTUALLY TESTS THE JOIN IS KONTUR, NOT THE NAMES.** Kontur's modelled population per
ADM3 is built from building footprints and knows nothing about the census, so the log-log
correlation between it and the census locality population is a quantity the join does not
determine. It is measured here against 500 random pairings rather than asserted, because a
band alone would pass a scrambled join ([[reference_check_needs_power]], and `bj_grid.py`
measured the same thing for Benin).

**EVERY DISTRICT'S BOOKLET CARRIES AN `Other` RESIDUAL** for the localities it does not name.
Those people are real and counted, and they are spread over the ADM3 polygons in that
district which no named locality claimed, weighted by Kontur population. That is the only
part of the geography that is not one-to-one, and it is why `bw_lookup.csv` has a `weight`
column at all.

Usage:
    python sources/bw_geo.py --fetch    the COD-AB shapefile bundle and the Kontur grid
    python sources/bw_geo.py            rebuild from data/raw/bw/
"""

import gzip
import os
import re
import shutil
import sys
import unicodedata
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "6")     # [[reference_scipy_eats_all_cores]]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "bw")
GEO = os.path.join(ROOT, "data", "geo", "bw")
NORM = os.path.join(ROOT, "data", "normalized", "bw.csv")
LOCALITIES = os.path.join(GEO, "bw_localities.gpkg")
HEXES = os.path.join(GEO, "bw_hexes.gpkg")
LOOKUP = os.path.join(GEO, "bw_lookup.csv")

COD_URL = ("https://data.humdata.org/dataset/74d579a7-e30b-445a-9ff3-1d673da53d3b/"
           "resource/61ec818b-57eb-4043-8a13-f5d59c7a192b/download/bwa_adm_2011_shp.zip")
COD_ZIP = "bwa_adm_2011_shp.zip"
KONTUR_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
              "kontur_population_BW_20231101.gpkg.gz")
KONTUR_GZ = "kontur_population_BW_20231101.gpkg.gz"
KONTUR_GPKG = "kontur_population_BW_20231101.gpkg"

EXPECTED_ADM3 = 519
EXPECTED_ADM2 = 28
UNIT_BAND = 8.0        # localities are small and Kontur is a 2023 model of a 2011 count
MIN_R = 0.55

# Census spelling -> COD ADM3 spelling, where folding is not enough.  Each one was read off
# the join report and checked against the district it sits in; none crosses a district.
# Census spelling -> COD ADM3 spelling, keyed on the folded census name. Every one of these
# was read off the join report, has exactly one plausible candidate INSIDE ITS OWN DISTRICT,
# and was checked against the 2022 locality report's spelling of the same village. None
# crosses a district boundary, which is the property that makes them safe to add at all.
ALIAS = {
    "selebiphikwe": "Selibe Phikwe",          # the town: booklets use e, COD uses i
    "ckgr": "Central Kgalagadi Game Reserve",  # the Ghanzi booklet abbreviates it
    "metsimotlhaba": "Metsimotlhabe",
    "ramotswataung": "Ramotswa Station/Taung",
    "artesia": "Artisia",
    "lesenepole": "Lesenepole/Matolwane",
    "jamakata": "Jamataka",
    "jackalasno2": "Jackalas 2",
    "matsaudisekapane": "Matsaudi/Sakapnae",
    "muchingemabele": "Muchinje/Mabele",
    "karakubis": "Karakobis",
    "khawa": "Khwawa",
    "samane": "Semane",
}


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for url, name in ((COD_URL, COD_ZIP), (KONTUR_URL, KONTUR_GZ)):
        dest = os.path.join(RAW, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
            print("already have", name)
            continue
        print("GET", url)
        r = requests.get(url, timeout=1800, stream=True,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        print(f"  {os.path.getsize(dest):,} bytes")
    if not zipfile.is_zipfile(os.path.join(RAW, COD_ZIP)):
        raise SystemExit(f"{COD_ZIP} is not a zip -- HDX answered the un-redirected URL")
    gp = os.path.join(RAW, KONTUR_GPKG)
    if not os.path.exists(gp):
        with gzip.open(os.path.join(RAW, KONTUR_GZ), "rb") as f, open(gp, "wb") as o:
            shutil.copyfileobj(f, o, 1 << 22)
        print(f"  unpacked {os.path.getsize(gp):,} bytes")


def adm2_of(note):
    """The district p-code `sources/bw.py` writes into the note as `adm2=BW1201`.

    The normalised schema is nine fixed columns (§3.9), so the district travels in the note
    the way Laos carries its province and district codes, rather than as a tenth column.
    """
    m = re.search(r"adm2=([A-Z0-9]+)", str(note))
    if not m:
        raise SystemExit(f"no adm2= in note {note!r} -- bw.csv is from an older bw.py")
    return m.group(1)


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"^\s*village\s+", "", s.strip(), flags=re.I)
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _read_cod():
    import geopandas as gpd

    src = os.path.join(RAW, COD_ZIP)
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} -- run with --fetch first")
    # §12's Chile rule: read the SHAPEFILE with fiona.  pyogrio is geopandas' default when
    # installed and is the engine that has been seen to return zero features in silence.
    g3 = gpd.read_file(f"zip://{src}!bwa_admbnda_adm3_2011.shp", engine="fiona")
    if len(g3) != EXPECTED_ADM3:
        raise SystemExit(f"ADM3 has {len(g3)} features, expected {EXPECTED_ADM3}")
    if g3["ADM2_PCODE"].nunique() != EXPECTED_ADM2:
        raise SystemExit(f"{g3['ADM2_PCODE'].nunique()} districts in ADM3, "
                         f"expected {EXPECTED_ADM2}")
    if g3.crs is None or g3.crs.to_epsg() != 4326:
        g3 = g3.to_crs(4326)
    return g3


def main():
    import geopandas as gpd
    import math
    import pandas as pd
    import random

    if "--fetch" in sys.argv:
        fetch()

    g3 = _read_cod()
    print(f"COD-AB ADM3: {len(g3)} polygons over {g3['ADM2_PCODE'].nunique()} districts")

    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/bw.py first")
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df["adm2"] = df["note"].map(adm2_of)
    # A district's unnamed remainder is a `locality` row like any other so that every check
    # in the project sees it (bw.py says why); it is marked in the note instead.
    df["residual"] = df["note"].str.contains("residual=yes")
    cen = (df[df["geo_level"] == "locality"]
           .drop_duplicates("geo_id")[["geo_id", "residual", "geo_name", "adm2"]])
    pop = (df[df["geo_level"] == "locality"]
           .groupby("geo_id")["count"].sum().to_dict())
    named = cen[~cen["residual"]]
    resid = cen[cen["residual"]]
    print(f"census rows: {len(named)} named localities, {len(resid)} district residuals")

    # ---- the name join, WITHIN the district ----
    poly = {}
    for nm, a2, pc in zip(g3["ADM3_EN"], g3["ADM2_PCODE"], g3["ADM3_PCODE"]):
        poly.setdefault((a2, fold(nm)), []).append((pc, nm))

    pairs, missing, ambiguous = {}, [], []
    for gid, nm, a2 in zip(named["geo_id"], named["geo_name"], named["adm2"]):
        key = (a2, fold(ALIAS.get(fold(nm), nm)))
        cand = poly.get(key, [])
        if len(cand) == 1:
            pairs[gid] = cand[0][0]
        elif len(cand) > 1:
            ambiguous.append((gid, nm, a2, [c[0] for c in cand]))
        else:
            missing.append((gid, nm, a2))

    used = set(pairs.values())
    dup = [u for u in used if list(pairs.values()).count(u) > 1]
    print("\n  the join, both ways (§12):")
    print(f"    census named localities   {len(named):>5}")
    print(f"    ADM3 polygons             {len(g3):>5}")
    print(f"    matched one-to-one        {len(pairs):>5}")
    print(f"    census name not in ADM3   {len(missing):>5}")
    print(f"    ambiguous inside district {len(ambiguous):>5}")
    print(f"    ADM3 claimed by no name   {len(g3) - len(used):>5}")
    for gid, nm, a2, c in ambiguous:
        print(f"      AMBIGUOUS: {nm!r} in {a2} -> {c}")
    if dup:
        raise SystemExit(f"one ADM3 polygon claimed by two localities: {sorted(set(dup))}")
    if ambiguous:
        raise SystemExit("a census name matches two polygons inside one district -- add an "
                         "ALIAS, do not guess")
    lostpop = sum(pop.get(g, 0) for g, _, _ in missing)
    tot = sum(pop.values())
    print(f"    unmatched names carry {lostpop:,} of {tot:,} people "
          f"({100.0 * lostpop / tot:.2f}%); they fall into their district's residual")
    for gid, nm, a2 in sorted(missing, key=lambda m: -pop.get(m[0], 0))[:15]:
        print(f"      no polygon: {nm!r} ({a2}, {pop.get(gid, 0):,})")

    # ---- Kontur, as the placement layer and as the test of the join ----
    kp = os.path.join(RAW, KONTUR_GPKG)
    if not os.path.exists(kp):
        raise SystemExit(f"missing {kp} -- run with --fetch first")
    hexes = gpd.read_file(kp)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"\nKontur: {len(hexes):,} hexes, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    # Take the centroid in the CRS the hexes were tiled in, then reproject the POINTS.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(g3.crs)
    hx = hexes.to_crs(g3.crs)
    j = gpd.sjoin(pts, g3[["ADM3_PCODE", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(pts.index)
    outside = j["ADM3_PCODE"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"  hexes outside every ADM3: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%) -- dropped")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": j.loc[keep, "ADM3_PCODE"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hx.geometry[keep.to_numpy()].to_numpy(), crs=g3.crs)
    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    empty = sorted(set(g3["ADM3_PCODE"]) - set(per.index))
    if empty:
        print(f"  ADM3 with no populated hex: {len(empty)} -- {empty[:6]}")

    # ---- the discriminating check: does the pairing carry information? ----
    rows = [(u, pop[g], float(per["sum"].get(u, 0.0)))
            for g, u in pairs.items() if per["sum"].get(u, 0.0) > 0 and pop.get(g, 0) > 0]
    lc = [math.log(r[1]) for r in rows]
    lk = [math.log(r[2]) for r in rows]

    def pearson(a, b):
        n = len(a)
        ma, mb = sum(a) / n, sum(b) / n
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
        return num / den

    r_true = pearson(lc, lk)
    rng = random.Random(0)
    perm = []
    for _ in range(500):
        s = list(lk)
        rng.shuffle(s)
        perm.append(abs(pearson(lc, s)))
    perm.sort()
    print(f"\n  the join carries information, measured rather than asserted: log-log "
          f"r = {r_true:.4f}\n  over {len(rows)} paired localities, against "
          f"{perm[-1]:.4f} for the best of 500 random pairings\n  "
          f"(median {perm[len(perm) // 2]:.4f}).")
    if r_true <= perm[-1] or r_true < MIN_R:
        raise SystemExit(f"census and Kontur populations correlate at r={r_true:.4f}, which "
                         f"random pairings reach ({perm[-1]:.4f}) -- the name join is not "
                         "carrying information")

    # ---- the lookup: named localities 1:1, residuals spread over the unclaimed ----
    # `tier` is spec §7 and is decided HERE, because this is where the distinction is made:
    # a row that matched its own polygon by name is drawn where the census counted it, and a
    # row that was spread across a district is not. It rides in the lookup so `countries.py`
    # does not have to re-derive it from the weights, which would call a residual that landed
    # on a single polygon `measured` and be wrong.
    lut = [{"geo_id": g, "unit": u, "weight": 1.0, "tier": "measured"}
           for g, u in pairs.items()]
    free = {}
    for a2, sub in g3.groupby("ADM2_PCODE"):
        free[a2] = [p for p in sub["ADM3_PCODE"] if p not in used]

    # THE SPREAD IS BOTH KINDS OF ROW AND NOT JUST THE RESIDUALS. A named locality whose
    # name is not in ADM3 still has people in it, and they belong to its district; sending
    # only the `Other` rows here would silently drop every one of them, which is the failure
    # this file exists to prevent rather than commit.
    spread = ([(g, n, a) for g, n, a in zip(resid["geo_id"], resid["geo_name"],
                                            resid["adm2"])]
              + [(g, n, a) for g, n, a in missing])
    for gid, nm, a2 in spread:
        cand = free.get(a2, [])
        w = {p: float(per["sum"].get(p, 0.0)) for p in cand}
        if not cand or sum(w.values()) <= 0:
            # nothing unclaimed in this district: spread over ALL of its polygons instead
            cand = list(g3.loc[g3["ADM2_PCODE"] == a2, "ADM3_PCODE"])
            w = {p: float(per["sum"].get(p, 0.0)) for p in cand}
        s = sum(w.values())
        if s <= 0:
            raise SystemExit(f"residual for {a2} has nowhere to go")
        for p, v in w.items():
            if v > 0:
                lut.append({"geo_id": gid, "unit": p, "weight": v / s,
                            "tier": "derived"})
    lutdf = pd.DataFrame(lut)
    bad = lutdf.groupby("geo_id")["weight"].sum()
    off = bad[(bad - 1.0).abs() > 1e-9]
    if len(off):
        raise SystemExit(f"weights do not sum to 1 for {list(off.index)[:5]}")

    os.makedirs(GEO, exist_ok=True)
    g3out = g3[["ADM3_PCODE", "ADM3_EN", "ADM2_PCODE", "ADM2_EN", "geometry"]].rename(
        columns={"ADM3_PCODE": "unit", "ADM3_EN": "name",
                 "ADM2_PCODE": "adm2", "ADM2_EN": "district"})
    g3out.to_file(LOCALITIES, layer="localities", driver="GPKG")
    out.to_file(HEXES, layer="hexes", driver="GPKG")
    lutdf.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"\nwrote {LOCALITIES} ({len(g3out)} polygons)")
    print(f"wrote {HEXES} ({len(out):,} hexes)")
    print(f"wrote {LOOKUP} ({len(lutdf):,} rows, "
          f"{lutdf['geo_id'].nunique()} census units)")


if __name__ == "__main__":
    main()
