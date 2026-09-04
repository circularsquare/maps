"""Estonia — the finest complete cover Statistics Estonia publishes religion for.

Writes data/geo/ee/ee_finest.gpkg — 78 municipalities + the 8 Tallinn city districts that
replace the 79th — and data/geo/ee/ee_replaced.csv naming what was replaced.

WHY NOT GISCO. Estonia is in the LAU 2021 file that Poland and Romania use, with 79
features that join on the EHAK code with no decoding at all. It is not used, because
**Tallinn is 33.05% of Estonia in a single 159 km² polygon** — three times worse than
Bucharest, and far and away the worst capital-in-one-unit case the project has met. A
third of the Estonian map would have been one uniform smear.

Statistics Estonia publishes RL21452 for Tallinn's 8 linnaosad as well as for the 79
municipalities, so the split is MEASURED and not an allocation — the same situation as
Czechia, where ČSÚ publishes the 142 city districts, and the opposite of Poland and
Romania, where GUS and INS publish nothing below the capital and it had to stand.

SOURCES — Maa-amet (Estonian Land Board), Haldus- ja asustusjaotus, EPSG:3301:

    https://geoportaal.maaamet.ee/docs/haldus_asustus/omavalitsus_shp.zip     79 municipalities
    https://geoportaal.maaamet.ee/docs/haldus_asustus/asustusyksus_shp.zip    4,714 settlement units

`linnaosa_shp.zip` looks like it should be the third download and IS NOT — that path
returns **HTTP 200 with a 282-byte PNG**, an image of an error message rather than an
error status, which is §5a's rule in its most literal form. The city districts are inside
`asustusyksus_shp.zip` instead, as the rows with `TYYP == 6`.

THE KEYS. The PxWeb place code is a 14-character concatenation of EHAK codes:

    00370141000001   county 0037 + municipality 0141 + filler        -> OKOOD = code[4:8]
    003707840000L4   county 0037 + municipality 0784 (Tallinn)       -> OKOOD = code[4:8]
    003707840176L6   county 0037 + Tallinn 0784 + district 0176      -> AKOOD = code[8:12]

so a municipality is keyed on `code[4:8]` and a city district on `code[8:12]`. EHAK codes
are unique across both kinds, which this script checks rather than assumes.

VINTAGE. The Maa-amet files are stamped 2024-12-01 and the census is 2021, which is
normally the §8.1 hazard. It is safe here for a specific reason: Estonia's 2017
haldusreform cut 213 municipalities to 79 and nothing has merged since, so the 2024 set
and the 2021 set are the same 79 units. The join proves it rather than the argument.

Usage:
    python sources/ee_geo.py --fetch
"""

import os
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

GEO = os.path.join(ROOT, "data", "geo", "ee")
OUT = os.path.join(GEO, "ee_finest.gpkg")
REPLACED = os.path.join(GEO, "ee_replaced.csv")
NORMALIZED = os.path.join(ROOT, "data", "normalized", "ee.csv")

FILES = {
    "omavalitsus": ("https://geoportaal.maaamet.ee/docs/haldus_asustus/"
                    "omavalitsus_shp.zip", 5_000_000),
    "asustusyksus": ("https://geoportaal.maaamet.ee/docs/haldus_asustus/"
                     "asustusyksus_shp.zip", 10_000_000),
}

TALLINN = "0784"
DISTRICT_TYYP = "6"
EXPECTED_MUNICIPALITIES = 79
EXPECTED_DISTRICTS = 8


def fetch():
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    os.makedirs(GEO, exist_ok=True)
    for name, (url, minb) in FILES.items():
        p = os.path.join(GEO, name + ".zip")
        if os.path.exists(p) and os.path.getsize(p) >= minb:
            print("already have", name)
            continue
        print("downloading", url)
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=900,
                         verify=False)
        r.raise_for_status()
        with open(p, "wb") as fh:
            fh.write(r.content)
        size = os.path.getsize(p)
        # A 282-byte PNG is what this server sends instead of a 404 (see the header).
        if size < minb or not zipfile.is_zipfile(p):
            raise SystemExit(f"{p} is {size:,} bytes and "
                             f"{'not ' if not zipfile.is_zipfile(p) else ''}a zip -- "
                             "Maa-amet serves an image of an error message with HTTP 200")
        print(f"  {size:,} bytes")


def bare(name):
    """Strip the unit-type words so the two languages can be compared.

    Statistics Estonia writes the English type ("Antsla rural municipality",
    "Narva-Jõesuu city") and Maa-amet writes the Estonian one ("Antsla vald",
    "Narva-Jõesuu linn"), so neither the whole string nor a shared suffix matches.
    """
    s = " ".join(str(name).split()).casefold()
    for word in ("rural municipality", "city district", "municipality", "city",
                 "linnaosa", "vald", "linn"):
        if s.endswith(" " + word):
            s = s[: -len(word) - 1]
    return s.strip()


def _layer(name):
    import geopandas as gpd
    p = os.path.join(GEO, name + ".zip")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run with --fetch first")
    ex = os.path.join(GEO, name)
    if not os.path.isdir(ex):
        with zipfile.ZipFile(p) as z:
            z.extractall(ex)
    shp = [f for f in os.listdir(ex) if f.lower().endswith(".shp")][0]
    return gpd.read_file(os.path.join(ex, shp))


def main():
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    oma = _layer("omavalitsus")
    print(f"omavalitsus: {len(oma):,} features, crs {oma.crs}")
    if len(oma) != EXPECTED_MUNICIPALITIES:
        raise SystemExit(f"{len(oma)} municipalities, expected "
                         f"{EXPECTED_MUNICIPALITIES} -- Estonia has reorganised")

    asu = _layer("asustusyksus")
    lin = asu[(asu["OKOOD"].astype(str) == TALLINN)
              & (asu["TYYP"].astype(str) == DISTRICT_TYYP)].copy()
    print(f"asustusyksus: {len(asu):,} features, of which {len(lin)} Tallinn linnaosad")
    if len(lin) != EXPECTED_DISTRICTS:
        raise SystemExit(f"{len(lin)} Tallinn city districts, expected "
                         f"{EXPECTED_DISTRICTS}")

    keep = oma[oma["OKOOD"].astype(str) != TALLINN].copy()
    keep["kod"] = keep["OKOOD"].astype(str)
    keep["name"] = keep["ONIMI"]
    keep["kind"] = "municipality"

    lin["kod"] = lin["AKOOD"].astype(str)
    lin["name"] = lin["ANIMI"]
    lin["kind"] = "city_district"

    overlap = set(keep["kod"]) & set(lin["kod"])
    if overlap:
        raise SystemExit(f"EHAK code collision between municipalities and districts: "
                         f"{sorted(overlap)}")

    import geopandas as gpd
    out = gpd.GeoDataFrame(
        pd.concat([keep[["kod", "name", "kind", "geometry"]],
                   lin[["kod", "name", "kind", "geometry"]]], ignore_index=True),
        crs=oma.crs).to_crs("EPSG:4326")
    print(f"\n  merged layer: {len(out):,} units "
          f"({(out['kind'] == 'municipality').sum()} municipalities + "
          f"{(out['kind'] == 'city_district').sum()} city districts)")

    # ---- the join, both ways
    df = pd.read_csv(NORMALIZED, dtype={"geo_id": str}, low_memory=False)
    cen = df[df["geo_level"].isin(("municipality", "city_district"))][
        ["geo_id", "geo_level", "geo_name"]].drop_duplicates("geo_id").copy()
    cen["kod"] = [c[4:8] if lv == "municipality" else c[8:12]
                  for c, lv in zip(cen["geo_id"], cen["geo_level"])]
    # Tallinn as a whole is replaced by its districts and must not also be drawn.
    cen = cen[~((cen["geo_level"] == "municipality") & (cen["kod"] == TALLINN))]

    geo_keys, cen_keys = set(out["kod"]), set(cen["kod"])
    missing, extra = cen_keys - geo_keys, geo_keys - cen_keys
    print(f"  census units (Tallinn replaced): {len(cen_keys):,}  |  polygons "
          f"{len(geo_keys):,}")
    print(f"  joined on EHAK code: {len(cen_keys & geo_keys)} of {len(cen_keys)}")

    # FOUR EHAK CODES WERE RETIRED BETWEEN THE CENSUS AND THIS BOUNDARY RELEASE.
    # Statistics Estonia keys the 2021 table on the code the municipality had on census
    # day; Maa-amet's 2024-12-01 file uses its current code. The old codes appear NOWHERE
    # in the current file -- not as a municipality, not as a settlement -- so this is a
    # real §8.1 vintage difference and not an encoding quirk:
    #
    #     0142 -> 0145  Antsla vald          0514 -> 0515  Narva-Jõesuu linn
    #     0735 -> 0736  Sillamäe linn        0855 -> 0857  Valga vald
    #
    # They are re-joined by name, and only where exactly one candidate remains on each
    # side, so nothing is guessed. The alias map is NOT hard-coded, because the next
    # release may retire different codes and a frozen list would go stale silently.
    if missing:
        cen_left = {k: bare(cen[cen["kod"] == k]["geo_name"].iloc[0]) for k in missing}
        geo_left = {k: bare(out[out["kod"] == k]["name"].iloc[0]) for k in extra}
        remap = {}
        for ck, cname in cen_left.items():
            hits = [gk for gk, gname in geo_left.items() if gname == cname]
            if len(hits) == 1:
                remap[ck] = hits[0]
        print(f"  re-joined by name after an EHAK code change: {len(remap)}")
        for ck, gk in sorted(remap.items()):
            print(f"      census {ck} -> polygon {gk}  ({cen_left[ck]})")
        # rewrite the polygon key to the census's code so `kod` is one namespace
        back = {v: k for k, v in remap.items()}
        out["kod"] = out["kod"].map(lambda k: back.get(k, k))
        geo_keys = set(out["kod"])
        missing, extra = cen_keys - geo_keys, geo_keys - cen_keys

    print(f"  {'OK ' if not missing else 'BAD'} census units with no polygon: "
          f"{len(missing)}")
    print(f"  {'OK ' if not extra else 'BAD'} polygons with no census unit: {len(extra)}")
    for k in sorted(missing)[:10]:
        print("      no polygon:", k,
              cen[cen["kod"] == k]["geo_name"].iloc[0])
    for k in sorted(extra)[:10]:
        print("      no data   :", k, out[out["kod"] == k]["name"].iloc[0])
    if missing or extra:
        raise SystemExit("join FAILED")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="finest", driver="GPKG")
    print("\nwrote", OUT, f"({len(out):,} polygons)")

    pd.DataFrame({"kod": [TALLINN], "name": ["Tallinn"],
                  "replaced_by": ["8 linnaosad"]}).to_csv(REPLACED, index=False)
    print("wrote", REPLACED)

    # what the split bought
    df2 = df[df["geo_level"].isin(("municipality", "city_district"))]
    tot = df2[(df2["geo_level"] == "municipality")
              & (df2["source_category"] == "Religion total")]["count"].sum()
    tln = df2[(df2["geo_level"] == "municipality")
              & (df2["source_category"] == "Religion total")
              & (df2["geo_id"].str[4:8] == TALLINN)]["count"].sum()
    big = out.to_crs(3301)
    big["area_km2"] = big.geometry.area / 1e6
    print(f"\n  Tallinn is {100 * tln / tot:.2f}% of the 15+ population and is now "
          f"{EXPECTED_DISTRICTS} units instead of 1.")
    print("  largest remaining polygons by area:")
    for _, r in big.nlargest(4, "area_km2").iterrows():
        print(f"    {r['name']:<26} {r['area_km2']:>8.0f} km2  ({r['kind']})")


if __name__ == "__main__":
    main()
