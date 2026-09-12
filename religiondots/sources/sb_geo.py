"""Solomon Islands — boundaries for the 183 wards.

Writes data/geo/sb/sb_wards.gpkg and data/geo/sb/sb_lookup.csv.

OCHA COD-AB Solomon Islands (`cod-ab-slb`, reviewed October 2024), the ADM3 **shapefile** on
§12's Chile rule. HDX's own description says where it comes from: *"Sourced from Solomon
Islands National Statistics Office (SINSO), 2009 Census of Population and Housing"* — so ADM3
is the census's own ward layer, and it carries **`SINSO_WID`, the office's ward id**, beside
the OCHA pcode. Fiji's situation (§9bd §6) rather than Vanuatu's.

**THE JOIN IS ON THAT ID AND IT IS 183/183.** The census prints a ward number that restarts at
01 inside each province; SINSO_WID is the province number followed by the ward number padded to
two digits, so Choiseul ward 1 is `101` and Honiara ward 1 is `1001`. Nothing spare either way.

**AND JOINING ON THE NAME WOULD HAVE BEEN WRONG.** A folded-name join matches only **154 of
183**. Twenty-eight of the twenty-nine misses are orthography — Solomon Islands English writes
prenasalised stops both ways, so the census's `Mbilua`, `Ndovele`, `Mbuini Tusu` and `Banika`
are COD's `Bilua`, `Dovele`, `Buini Tusu` and `Mbanika`, and apostrophes and separators account
for most of the rest.

**The twenty-ninth is not a spelling at all: Isabel ward 02 is `Baolo` in the census and
`Havulei` in COD.** Neither name appears anywhere on the other side, and every other Isabel
ward pairs on both id and name, so this is a renamed ward and not a mispairing — but it is
exactly the case a name join cannot see, and it is why this file joins on the id and treats
the names as evidence rather than as the key.

**THE WITNESS IS THE PROVINCE.** COD files each ADM3 under an ADM1 independently of SINSO's
numbering; the census files each ward under the province whose block it prints in. Two
organisations, and they agree on all 183.

Usage:
    python sources/sb_geo.py --fetch    one ~2.4 MB zip from HDX
    python sources/sb_geo.py            rebuild from data/raw/sb/
"""

import csv
import os
import re
import sys
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "sb")
OUT_DIR = os.path.join(ROOT, "data", "geo", "sb")
OUT = os.path.join(OUT_DIR, "sb_wards.gpkg")
LOOKUP = os.path.join(OUT_DIR, "sb_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "sb.csv")

ZIP_URL = ("https://data.humdata.org/dataset/776dd627-a97a-4f86-9b0d-52ce64525120/"
           "resource/5b69c85c-392f-42da-81dc-ebb0a5659fa2/download/slb_admbnda_adm3.zip")
ZIP_NAME = "slb_admbnda_adm3.zip"
SHP = "slb_admbnda_adm3.shp"
EXPECTED = 183

# The Solomons run 155.5E to 170E. Nowhere near 180, but a torn polygon is silent (§9bd §7).
MAX_SPAN_DEG = 25.0

# COD's ADM1 spelling against the census's province name, for the witness.
PROV_ALIAS = {"rennell bell": "rennell bellona",
              "honiara": "honiara city council"}


def fold(s):
    """Casefold, strip accents and punctuation — for COMPARING names, never for storing."""
    s = unicodedata.normalize("NFKD", str(s).strip().lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, ZIP_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 500_000:
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
        raise SystemExit(f"missing {NORM} -- run sources/sb.py first")

    g = gpd.read_file("zip://" + zpath + "!" + SHP, engine="fiona")
    if len(g) != EXPECTED:
        raise SystemExit(f"COD ADM3 has {len(g)} features, expected {EXPECTED}")
    g = g.to_crs("EPSG:4326")
    b = g.total_bounds
    print(f"COD-AB ADM3: {len(g)} wards, crs={g.crs}")
    print(f"  bounds lon {b[0]:.3f}..{b[2]:.3f}, lat {b[1]:.3f}..{b[3]:.3f} "
          f"— {b[2] - b[0]:.2f}° wide")
    if b[2] - b[0] > MAX_SPAN_DEG:
        raise SystemExit(f"the country came out {b[2] - b[0]:.1f}° wide; a polygon is torn")
    widest = max(r.geometry.bounds[2] - r.geometry.bounds[0] for r in g.itertuples())
    print(f"  and no single polygon is wider than {widest:.2f}°: nothing is torn")

    g["wid"] = g["SINSO_WID"].astype(str).str.strip()
    if g["wid"].duplicated().any():
        raise SystemExit("COD ADM3 has duplicate SINSO_WID values")

    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "ward"]
    cen_name = dict(zip(df["geo_id"], df["geo_name"]))
    cen_prov = {}
    for gid, note in zip(df["geo_id"], df["note"]):
        m = re.search(r"province=([^;]+)", str(note))
        if m:
            cen_prov[gid] = m.group(1).strip()
    if len(cen_name) != EXPECTED:
        raise SystemExit(f"{NORM} has {len(cen_name)} wards, expected {EXPECTED}")

    cod_name = dict(zip(g["wid"], g["ADM3_NAME"].astype(str).str.strip()))
    cod_prov = dict(zip(g["wid"], g["ADM1_NAME"].astype(str).str.strip()))

    missing = sorted(set(cen_name) - set(cod_name))
    spare = sorted(set(cod_name) - set(cen_name))
    print("\n  the join, both ways (§12) — ON SINSO's OWN WARD ID:")
    print(f"    census wards               {len(cen_name):>4}")
    print(f"    COD polygons               {len(cod_name):>4}")
    print(f"    matched                    {len(set(cen_name) & set(cod_name)):>4}")
    print(f"    census with no polygon     {len(missing):>4}")
    print(f"    polygons with no census    {len(spare):>4}")
    for k in missing[:8]:
        print(f"      no polygon: {k} {cen_name[k]!r}")
    for k in spare[:8]:
        print(f"      no census : {k} {cod_name[k]!r}")
    if missing or spare:
        raise SystemExit("join FAILED")

    # ---- witness 1: the province, from two organisations ----
    disagree = []
    for wid in cen_name:
        a = fold(cen_prov.get(wid, ""))
        b_ = fold(cod_prov[wid])
        if a != b_ and PROV_ALIAS.get(b_) != a:
            disagree.append((wid, cen_name[wid], cen_prov.get(wid), cod_prov[wid]))
    print(f"\n    witness 1 — COD's ADM1 against the province the census prints each ward "
          f"under:\n      agree on {EXPECTED - len(disagree)}/{EXPECTED} wards")
    for wid, nm, a, b_ in disagree[:8]:
        print(f"        MISMATCH {wid} {nm!r}: census {a!r} vs COD {b_!r}")
    if disagree:
        raise SystemExit(f"{len(disagree)} wards pair on id while the two sources file them "
                         "under different provinces -- §12's shape-2 failure")

    # ---- witness 2: the name, reported and NOT asserted ----
    same = [w for w in cen_name if fold(cen_name[w]) == fold(cod_name[w])]
    print(f"\n    witness 2 — the NAME agrees on {len(same)}/{EXPECTED}, and the rest are "
          "reported\n      rather than asserted, because they are real differences in "
          "spelling:")
    for wid in sorted(set(cen_name) - set(same), key=lambda k: (len(k), k)):
        print(f"        {wid:>5}  census {cen_name[wid]!r:<26} COD {cod_name[wid]!r}")
    print("      Prenasalised stops are written both ways in Solomon Islands English "
          "(Mbilua/Bilua,\n      Ndovele/Dovele), which is most of these. **`Baolo` vs "
          "`Havulei` on 302 is NOT a\n      spelling**: neither name appears on the other "
          "side, and every other Isabel ward\n      pairs on both id and name, so it is a "
          "renamed ward. See the module docstring.")

    out = g[["wid", "ADM3_NAME", "ADM3_PCODE", "ADM1_NAME", "geometry"]].copy()
    out["unit"] = out["ADM3_PCODE"].astype(str).str.strip()
    # Take the NAME from the statistical source, not the boundary file (§12, Chile).
    out["name"] = out["wid"].map(cen_name)
    if out["name"].isna().any():
        raise SystemExit("a polygon came out of the join with no census name")
    if out["unit"].duplicated().any():
        raise SystemExit("COD ADM3 has duplicate pcodes")

    os.makedirs(OUT_DIR, exist_ok=True)
    out[["unit", "name", "wid", "ADM1_NAME", "geometry"]].to_file(
        OUT, layer="wards", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    pairs = sorted(zip(out["wid"], out["unit"]), key=lambda t: t[1])
    with open(LOOKUP, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_id", "unit"])
        w.writerows(pairs)
    print(f"wrote {LOOKUP} ({len(pairs)} rows)")


if __name__ == "__main__":
    main()
