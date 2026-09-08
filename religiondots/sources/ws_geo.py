"""Samoa — boundaries, and the reason they are coarser than the data.

Writes data/geo/ws/ws_districts.gpkg and data/geo/ws/ws_lookup.csv.

**SAMOA IS THE ONE PACIFIC COUNTRY WITH NO COD-AB.** Tonga, Fiji, Solomon Islands, PNG,
Vanuatu, Kiribati, FSM and the Marshall Islands all have one on HDX; Samoa has none, and this
was re-checked on 2026-09-08. What exists instead:

- **geoBoundaries WSM ADM2, 43 polygons**, sourced from the Pacific Data Hub. **GADM 4.1's
  level 2 is the same 43 with the same names**, including the same `(PART)` artefacts, so it
  is one file and not two independent sources.
- OSM has 11 admin relations for the whole country, so there is no boundary layer to build.
- `pacificdata.org` sits behind a Cloudflare challenge and was not fought.
- SBS itself publishes no geography: its media library is 849 files with no shapefile, and its
  own census dashboard is a Looker Studio embed.

**AND THE 43 DO NOT MATCH THE CENSUS'S 51.** They are two different cuts of the same country
made at different times. The census numbers its districts (`Vaimauga 1` … `Vaimauga 4`); the
polygon layer names them by compass point (`Vaimauga East`, `Vaimauga West`). Four census
districts against two polygons, and neither nests inside the other. The stem alone settles only
18 of the 51 districts, **23.1% of the population**.

**WHAT DOES WORK IS THAT BOTH ARE CUTS OF THE SAME TRADITIONAL DISTRICTS.** Every census
district and every polygon carries the name of one of **25 traditional districts** as its stem,
so both aggregate into those 25 exactly, by name, with nothing assumed and nothing geocoded.
That is what this file builds: the 43 polygons dissolved by stem, and the 51 census districts
mapped onto the same 25. **8,222 people per unit.**

**THE ROUTE TO 43 WAS TRIED AND IS NOT GOOD ENOUGH YET.** The census's 339 villages could be
assigned to the 43 by locating each one, and OSM has 554 Samoan villages. Matching them by
name, disambiguated by requiring the census district's stem to agree with the polygon's,
placed **285 of 339 villages, 85.2% of the population**. The rest fail for reasons that are
fixable but not yet fixed: the census anglicises (`Lalovaea East` is OSM's `Lalovaea Sasa'e`,
`Samata Uta` is `Samata-i-Uta`) and qualifies repeated names with a district (`Vailoa Faleata`,
`Siufaga Faasaleleaga`), while Solosolo, Falefa, Faleseela, Falevao and Tuanimato are absent
from OSM altogether. **A wrong village assignment would move people between districts and every
total would still balance** ([[reference_name_join_wrong_neighbour]]), so 85% is not a basis to
draw on. Recorded here rather than in a comment because it is the next improvement.

Usage:
    python sources/ws_geo.py --fetch    one ~0.6 MB geojson from geoBoundaries
    python sources/ws_geo.py            rebuild from data/raw/ws/
"""

import csv
import os
import re
import sys
import unicodedata

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ws")
OUT_DIR = os.path.join(ROOT, "data", "geo", "ws")
OUT = os.path.join(OUT_DIR, "ws_districts.gpkg")
LOOKUP = os.path.join(OUT_DIR, "ws_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "ws.csv")

GEOJSON_URL = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/"
               "WSM/ADM2/geoBoundaries-WSM-ADM2.geojson")
GEOJSON_NAME = "geoBoundaries-WSM-ADM2.geojson"

EXPECTED_POLYGONS = 43
EXPECTED_CENSUS_DISTRICTS = 51
EXPECTED_UNITS = 25

# Samoa sits at 172 W, 800 km from the antimeridian, and does not cross it.
# [[reference_antimeridian]]
MAX_SPAN_DEG = 6.0
EXPECTED_BBOX = (-173.5, -14.6, -170.8, -13.2)

# Spelling differences between the census's district names and the polygon layer's, applied
# after the stem is taken. Every one was read off the two lists side by side.
STEM_ALIAS = {
    "lefaga faleaseela": "lefaga faleseela",   # census `Faleaseela`, polygon `Faleseela`
    "alataua i sisifo": "alataua",             # census keeps the Samoan `i Sisifo` (west)
    "satupaitea": "satuipaitea",               # the polygon layer transposes the vowels
    "faasalelelaga": "faasaleleaga",           # polygon `Faasalelelaga IV`, an extra `le`
}

# The 25 units, in the census's own spelling. Explicit rather than derived, because these are
# the labels a reader sees and a regex should not be choosing them.
DISPLAY = {
    "aana alofi": "Aana Alofi", "aiga i le tai": "Aiga i le Tai", "alataua": "Alataua",
    "aleipata itupa i lalo": "Aleipata Itupa i Lalo",
    "aleipata itupa i luga": "Aleipata Itupa i Luga",
    "anoamaa": "Anoamaa", "faasaleleaga": "Faasaleleaga", "falealili": "Falealili",
    "falealupo": "Falealupo", "faleata": "Faleata",
    "falelatai samatau": "Falelatai & Samatau", "gagaemauga": "Gagaemauga",
    "gagaifomauga": "Gagaifomauga", "lefaga faleseela": "Lefaga & Faleaseela",
    "lepa": "Lepa", "lotofaga": "Lotofaga", "palauli": "Palauli", "safata": "Safata",
    "sagaga": "Sagaga", "salega": "Salega", "satuipaitea": "Satupaitea", "siumu": "Siumu",
    "vaa o fonoti": "Vaa o Fonoti", "vaimauga": "Vaimauga", "vaisigano": "Vaisigano",
}


def fold(s):
    """Casefold, strip accents and punctuation — for COMPARING names, never for storing."""
    s = unicodedata.normalize("NFKD", str(s).strip().lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("‘", "'").replace("’", "'").replace("`", "'")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def stem(name):
    """A district name -> its traditional-district key.

    Strips exactly the things that distinguish one cut from the other: the census's trailing
    number (`Vaimauga 3`), the polygon layer's compass word or Roman numeral (`Vaimauga East`,
    `Faasaleleaga III`), its `le Falefa` / `le Usoga` sub-names, and its `(PART)` marker for a
    district split across the two main islands.
    """
    s = fold(name)
    s = re.sub(r"\s*\(?\s*part\s*\)?$", "", s)
    s = re.sub(r"\s+\d+$", "", s)
    s = re.sub(r"\s+le (falefa|usoga)$", "", s)
    s = re.sub(r"\s+(east|west|sasa e|sisifo)$", "", s)
    s = re.sub(r"\s+(i|ii|iii|iv|v)$", "", s)
    s = s.strip()
    return STEM_ALIAS.get(s, s)


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, GEOJSON_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
        print("already have", dest)
        return
    print("GET", GEOJSON_URL)
    r = requests.get(GEOJSON_URL, timeout=900, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    tmp = dest + ".part"                                  # [[reference_wb_truncates]]
    with open(tmp, "wb") as fh:
        fh.write(r.content)
    os.replace(tmp, dest)
    print(f"  {os.path.getsize(dest):,} bytes")


def build():
    import geopandas as gpd
    import pandas as pd

    src = os.path.join(RAW, GEOJSON_NAME)
    if not os.path.exists(src):
        raise SystemExit(f"{src} is missing — run `python sources/ws_geo.py --fetch`")
    g = gpd.read_file(src)
    if len(g) != EXPECTED_POLYGONS:
        raise SystemExit(f"{len(g)} ADM2 polygons, expected {EXPECTED_POLYGONS} — "
                         "geoBoundaries reissued the layer")
    g = g.rename(columns={"shapeName": "adm2_name"}).to_crs(4326)
    minx, miny, maxx, maxy = g.total_bounds
    if max(maxx - minx, maxy - miny) > MAX_SPAN_DEG:
        raise SystemExit(f"bbox spans {maxx - minx:.1f} x {maxy - miny:.1f} degrees — torn")
    if not (EXPECTED_BBOX[0] <= minx and maxx <= EXPECTED_BBOX[2]
            and EXPECTED_BBOX[1] <= miny and maxy <= EXPECTED_BBOX[3]):
        raise SystemExit(f"bbox {minx:.2f},{miny:.2f},{maxx:.2f},{maxy:.2f} is not Samoa")

    g["unit"] = g["adm2_name"].map(stem)

    # --- the census's 51 districts, off the normalised file.
    if not os.path.exists(NORM):
        raise SystemExit(f"{NORM} is missing — run `python sources/ws.py` first")
    df = pd.read_csv(NORM, dtype=str, keep_default_na=False)
    df["count"] = df["count"].astype(int)
    df[["region", "district"]] = df["note"].str.split("|", expand=True)
    districts = sorted(df["district"].unique())
    if len(districts) != EXPECTED_CENSUS_DISTRICTS:
        raise SystemExit(f"{len(districts)} census districts, "
                         f"expected {EXPECTED_CENSUS_DISTRICTS}")
    df["unit"] = df["district"].map(stem)

    # --- the two stem sets must be the same 25. This is the whole argument of the file, so it
    #     fails rather than warning: a stem on one side only means a district went undrawn or
    #     a polygon got no people.
    pol = set(g["unit"])
    cen = set(df["unit"])
    if pol != cen:
        raise SystemExit("the polygon and census stems disagree:\n"
                         f"   polygons only: {sorted(pol - cen)}\n"
                         f"   census only:   {sorted(cen - pol)}")
    if len(pol) != EXPECTED_UNITS:
        raise SystemExit(f"{len(pol)} stems, expected {EXPECTED_UNITS}: {sorted(pol)}")
    missing_label = sorted(pol - set(DISPLAY))
    if missing_label:
        raise SystemExit(f"no DISPLAY name for {missing_label}")

    diss = g.dissolve(by="unit", as_index=False)[["unit", "geometry"]]
    diss["name"] = diss["unit"].map(DISPLAY)
    if len(diss) != EXPECTED_UNITS:
        raise SystemExit(f"dissolve produced {len(diss)} units")

    print(f"  {EXPECTED_POLYGONS} polygons and {EXPECTED_CENSUS_DISTRICTS} census districts "
          f"both fold to the same {EXPECTED_UNITS} traditional districts")
    per = df.groupby("unit")["count"].sum()
    npol = g.groupby("unit").size()
    ndis = df.groupby("unit")["district"].nunique()
    nvil = df.groupby("unit")["geo_id"].nunique()
    print(f"\n    {'unit':<24} {'people':>8} {'polys':>6} {'cens.dist':>10} {'villages':>9}")
    for u in sorted(pol, key=lambda u: -per[u]):
        print(f"    {DISPLAY[u]:<24} {per[u]:>8,} {npol[u]:>6} {ndis[u]:>10} {nvil[u]:>9}")
    total = int(per.sum())
    print(f"    {'':<24} {total:>8,}")
    if total != 205_557:
        raise SystemExit(f"units hold {total:,} people, the census says 205,557")

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = OUT[:-5] + ".part.gpkg"
    if os.path.exists(tmp):
        os.remove(tmp)
    diss[["unit", "name", "geometry"]].to_file(tmp, driver="GPKG", layer="districts")
    os.replace(tmp, OUT)

    lut = df[["geo_id", "unit", "geo_name", "district", "region"]].drop_duplicates("geo_id")
    with open(LOOKUP + ".part", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_id", "unit", "village", "census_district", "region"])
        for _, r in lut.iterrows():
            w.writerow([r["geo_id"], r["unit"], r["geo_name"], r["district"], r["region"]])
    os.replace(LOOKUP + ".part", LOOKUP)

    print(f"\nwrote {OUT} ({len(diss)} units)")
    print(f"wrote {LOOKUP} ({len(lut)} villages)")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    else:
        build()
