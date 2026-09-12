"""Germany — Gemeinde boundaries for the Zensus 2022 religion data.

Writes data/geo/de/de_gemeinden.gpkg, the single layer countries.py reads, and prints
the join against data/normalized/de.csv both ways.

SOURCE: BKG VG250, Verwaltungsgebiete 1:250 000, **Gebietsstand 31.12.2022**, licensed
dl-de/by-2-0.  Free, no registration, no form.

THE VINTAGE WAS DETERMINED BY TRYING THEM, not by reasoning about it (spec §8.1), and it
is the whole reason this join is clean.  BKG publishes both a 01.01 and a 31.12 edition
of every year, and destatis does not say anywhere in the Sonderauswertung which
Gebietsstand it used:

    VG250 01.01.2022     10,993 units    2 census rows unmatched
    VG250 01.01.2023     10,981 units   10 census rows unmatched
    VG250 31.12.2022     10,990 units    0 census rows unmatched   <- this one

So the census was published on the Gebietsstand 31.12.2022 and the 31.12 edition is the
answer.  Both near misses are instructive and both would have been survivable-but-wrong:

  * Against 01.01.2022 the two unmatched rows are Schwedt/Oder and Pinnow in the
    Uckermark, and they are NOT a merger — the Gemeinde number is unchanged and only the
    **Verbandsschlüssel** moved (Schwedt from amtsfrei 0532 into Amt 5051, Pinnow from
    Amt 5310 into 5051).  Joining on the 8-digit AGS instead of the 12-digit ARS hides
    that completely, and would then have left Passow, Berkholz-Meyenburg and Mark Landin
    as orphan polygons whose people are counted inside Schwedt and Pinnow but whose
    territory belongs to nothing — about 3,000 people placed in the wrong villages, with
    every count still reconciling.  The ARS carries the Verband and the AGS does not, so
    the LONGER key is the safer one here; that is the opposite of Poland (§pl_geo).
  * Against 01.01.2023 the ten unmatched rows (Bromskirchen, Ostrau, Dünwald, Menteroda,
    Anrode …) are Gemeinden dissolved during 2023, i.e. after the census was published.

JOIN: the census `geo_id` IS the ARS, verbatim, with no derived key and no slicing —
10,786 of 10,786 both ways.  Every one of the 204 VG250 polygons with no census row
carries `BEZ == 'Gemeindefreies Gebiet'`: unincorporated forest, lake and military
areas, which have no inhabitants and so no religion row.  That is asserted rather than
assumed, because "204 leftovers" and "204 leftovers that are all uninhabited" are very
different findings.

The independent evidence that the key is right rather than coincidentally unique is the
names: 10,697 of 10,786 agree after folding.  The 89 that differ are all cosmetic and
several are interesting — destatis carries the official bilingual Sorbian names
(`Cottbus/Chóśebuz`, `Märkische Heide/Markojska Góla`) where BKG carries the German
alone.

PLACEMENT (spec §8.2), AND GERMANY IS THE WORST CASE ON THE MAP.  §8.2's trick — a
statistical agency designs its fine unit to a population target, so equal dots per
polygon is already a population weighting — does not hold for Gemeinden at all.  They
are historical administrative units and their populations span six orders of magnitude,
from Dierfeld (12 people) to Berlin (3.6 million in ONE polygon, 4.4% of the country).
That is worse than Warsaw (4.7% but a fifth the people), worse than Prague, and worse
than Bucharest.  The concentration is measured and printed at the end of this script.

The fix exists and is unusually good: destatis publishes THE SAME THREE CATEGORIES on a
100m INSPIRE grid, 3,088,036 populated cells, geometry derivable from the cell id with
no boundary file at all.  Germany is therefore the one country where placement could be
MEASURED rather than modelled, and §8.2's approximation dropped entirely.  Not built
here — this file is the Gemeinde layer — but the reason to build it is Berlin.

Usage:
    python sources/de_geo.py            # needs the VG250 zip, see sources/de_geo.md
    python sources/de_geo.py --fetch    # download it first (69MB)
"""

import csv
import os
import shutil
import sys
import unicodedata
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GEO_DIR = os.path.join(ROOT, "data", "geo", "de")
ZIP_NAME = "vg250_1231_2022.zip"
SHP_DIR = os.path.join(GEO_DIR, "vg250_1231_2022_shp")
OUT = os.path.join(GEO_DIR, "de_gemeinden.gpkg")
CENSUS = os.path.join(ROOT, "data", "normalized", "de.csv")

ZIP_URL = ("https://daten.gdz.bkg.bund.de/produkte/vg/vg250_ebenen_1231/2022/"
           "vg250_12-31.utm32s.shape.ebenen.zip")
MIN_BYTES = 60_000_000

# GF is the Geofaktor: 4 is "mit Struktur Land", the land polygons.  1-3 are the water
# bodies (Bodensee, the Küstenmeer) which carry the same ARS as their neighbours and
# would duplicate keys if left in.
GF_LAND = 4
GEMEINDEFREI = "Gemeindefreies Gebiet"
N_GEMEINDEN = 10_786


def fetch():
    os.makedirs(GEO_DIR, exist_ok=True)
    dest = os.path.join(GEO_DIR, ZIP_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) >= MIN_BYTES:
        print("already have", dest)
        return
    print("downloading", ZIP_URL)
    urllib.request.urlretrieve(ZIP_URL, dest)
    size = os.path.getsize(dest)
    if size < MIN_BYTES or not zipfile.is_zipfile(dest):
        raise SystemExit(f"{dest} is {size:,} bytes and "
                         f"{'is' if zipfile.is_zipfile(dest) else 'is NOT'} a zip -- "
                         f"expected >= {MIN_BYTES:,} bytes of zip (spec §12)")
    print(f"  {size:,} bytes")


def extract():
    """Pull just the Gemeinde layer out of the 69MB zip, into its own directory."""
    src = os.path.join(GEO_DIR, ZIP_NAME)
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} -- run with --fetch first")
    shp = os.path.join(SHP_DIR, "VG250_GEM.shp")
    if os.path.exists(shp):
        return shp
    os.makedirs(SHP_DIR, exist_ok=True)
    with zipfile.ZipFile(src) as z:
        members = [n for n in z.namelist() if "VG250_GEM." in n]
        if not members:
            raise SystemExit(f"{src} has no VG250_GEM layer -- contents changed?")
        for n in members:
            with z.open(n) as fh, open(
                    os.path.join(SHP_DIR, os.path.basename(n)), "wb") as out:
                shutil.copyfileobj(fh, out)
    print(f"  extracted {len(members)} files to {SHP_DIR}")
    return shp


def fold(s):
    """Fold a name for comparison only.  Never used as a key."""
    s = str(s).replace("ß", "ss")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s.lower() if c.isalnum() and not unicodedata.combining(c))


def main():
    if "--fetch" in sys.argv:
        fetch()
    shp = extract()

    import geopandas as gpd

    g = gpd.read_file(shp)
    print(f"\n  VG250 31.12.2022: {len(g):,} rows, CRS {g.crs}")
    by_gf = g["GF"].value_counts().to_dict()
    g = g[g["GF"] == GF_LAND].copy()
    print(f"  GF {by_gf} -> keeping GF=={GF_LAND} (land), {len(g):,} rows")

    g["ars"] = g["ARS"].astype(str)
    bad = g.loc[~g["ars"].str.fullmatch(r"\d{12}"), "ars"].tolist()
    if bad:
        raise SystemExit(f"ARS is not 12 digits for {len(bad)} rows: {bad[:5]}")
    if g["ars"].duplicated().any():
        dup = g.loc[g["ars"].duplicated(), "ars"].tolist()
        raise SystemExit(f"duplicate ARS after the GF filter: {dup[:5]}")

    # census side
    cen = {}
    with open(CENSUS, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["geo_level"] == "gemeinde" and r["source_category"] == "Einwohnerzahl":
                cen[r["geo_id"]] = (r["geo_name"], int(r["count"]))
    if len(cen) != N_GEMEINDEN:
        raise SystemExit(f"{CENSUS} has {len(cen):,} Gemeinden, expected {N_GEMEINDEN:,}")

    geo_keys, cen_keys = set(g["ars"]), set(cen)
    missing_geo = cen_keys - geo_keys
    extra_geo = geo_keys - cen_keys
    print(f"\n  census Gemeinden {len(cen_keys):,}  |  polygons {len(geo_keys):,}")
    print(f"  {'OK ' if not missing_geo else 'BAD'} census units with no polygon: "
          f"{len(missing_geo)}")
    for k in sorted(missing_geo)[:10]:
        print(f"      no polygon: {k}  {cen[k][0]!r}")
    if missing_geo:
        raise SystemExit("join FAILED -- every census Gemeinde must have a polygon")

    # The polygons with no census row must ALL be gemeindefreie Gebiete.  Asserted,
    # because a populated Gemeinde landing in here would be a silent hole in the map.
    ex = g[g["ars"].isin(extra_geo)]
    by_bez = ex["BEZ"].value_counts().to_dict()
    good = set(by_bez) <= {GEMEINDEFREI}
    print(f"  {'OK ' if good else 'BAD'} polygons with no census row: {len(extra_geo)}, "
          f"by BEZ {by_bez}")
    if not good:
        for _, r in ex[ex["BEZ"] != GEMEINDEFREI].head(10).iterrows():
            print(f"      unexpected: {r['ars']}  {r['GEN']!r}  {r['BEZ']!r}")
        raise SystemExit("join FAILED -- a populated polygon has no census row")

    g = g[g["ars"].isin(cen_keys)].copy()
    g["name"] = g["ars"].map(lambda k: cen[k][0])
    g["pop"] = g["ars"].map(lambda k: cen[k][1])

    # Independent evidence the key is right: the names.  The census appends the
    # Bezeichnung ("Flensburg, Stadt") where BKG keeps it in a separate column, so the
    # comparison is on the part before the comma.
    agree = g.apply(lambda r: fold(str(r["name"]).split(",")[0]) == fold(r["GEN"]),
                    axis=1)
    print(f"\n  name agreement on joined pairs: {agree.sum():,} of {len(g):,}")
    for _, r in g[~agree].head(8).iterrows():
        print(f"      {r['ars']}  census={r['name']!r:42s} BKG={r['GEN']!r}")
    if len(g) - agree.sum() > 200:
        raise SystemExit("name agreement collapsed -- the key is probably wrong")

    out = g[["ars", "name", "pop", "geometry"]].to_crs(4326)
    os.makedirs(GEO_DIR, exist_ok=True)
    out.to_file(OUT, layer="gemeinden", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} polygons, EPSG:4326)")

    # spec §8.2: how bad is equal-dots-per-polygon here?  Printed rather than asserted,
    # because it is a property of German local government and not something to fix.
    pops = sorted(cen[k][1] for k in cen)
    total = sum(pops)
    print(f"\n  Gemeinde population: min {pops[0]:,}  median {pops[len(pops)//2]:,}  "
          f"max {pops[-1]:,}")
    big = sorted(cen.items(), key=lambda kv: -kv[1][1])[:8]
    print(f"  largest single polygons (the spec §8.2 caveat for Germany):")
    for k, (nm, p) in big:
        print(f"      {p:>10,}  {100.0*p/total:4.1f}%  {nm}")
    share = sum(p for _, (_, p) in big) / total
    print(f"  those 8 polygons hold {share*100:.1f}% of Germany")
    over = [p for p in pops if p >= 100_000]
    print(f"  {len(over)} Gemeinden of {len(pops):,} hold >=100,000 people, "
          f"{sum(over)/total*100:.1f}% of the country between them")


if __name__ == "__main__":
    main()
