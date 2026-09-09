"""São Tomé and Príncipe — boundaries for the seven districts.

Writes data/geo/st/st_districts.gpkg and data/geo/st/st_lookup.csv.

OCHA COD-AB São Tomé and Príncipe (`cod-ab-stp`), the **shapefile** bundle rather than the
geodatabase on §12's Chile rule. The tier is the *distrito*: six on São Tomé island plus the
Região Autónoma do Príncipe, which is the whole of the northern island and is a region in law
and a district in every census table.

**COD-AB AND COD-PS DISAGREE ABOUT WHAT LEVEL THIS TIER IS, AND THEIR PCODES DO NOT JOIN.**
The 2026 boundary bundle publishes the seven districts as **ADM1** with pcodes `ST11` and
`ST21`-`ST26`; the 2022 population bundle publishes the same seven as **ADM2**, under an
ADM1 of two provinces, with pcodes `ST0101` and `ST0201`-`ST0206`. Neither file mentions the
other's scheme, so `ST21` and `ST0201` are the same district and share no key. Anything that
joins the two COD files on a pcode gets an empty frame; anything that joins them on
`adm1_pcode` alone silently pairs a district with a province. This file uses COD-AB's ADM1
pcodes as `unit` throughout and never reads COD-PS at all, for the reason below.

## THE NAMES ARE DECISIVE HERE, AND THE OTHER TWO WITNESSES ARE CONFIRMATION

Unlike Cabo Verde (§9ci, three pairs of concelhos sharing a name), **São Tomé has no twins**:
seven districts, seven distinct names, six of which fold to COD's spelling exactly, and the
seventh is COD's English gloss `Príncipe (Autonomous Region)` for INE's `Região Autónoma do
Príncipe`. There is nothing for [[reference_name_join_wrong_neighbour]] to bite on, so the
name join is the join and the other two witnesses are checking that the polygons are the
right shapes rather than resolving an ambiguity.

**Witness 2 is area, and it is worth reading for what it CANNOT do.** The V RGPH 2024 prints
each district's area in km² beside its population (Tabela 1.3), and measuring each COD-AB
polygon in an equal-area projection reproduces all seven to within 8.8%. That sounds like a
geometric proof of the pairing and it is not quite one: Cantagalo and Mé-Zóchi are 128.9 and
120.8 km² measured against 118.5 and 122.0 printed, so **swapping those two fits the printed
areas very slightly better than the truth does**: exactly 1 of the 5,039 other permutations
beats the identity, and it is that swap. So area pins five of the seven
outright, leaves one pair inside its own error, and `check_areas` asserts exactly that rather
than pretending to more. In a country where the names *were* ambiguous this witness would not
have been enough on its own, which is the thing worth carrying forward.

**Witness 3 is the same office counting the same districts twelve years apart**: every
district's 2024 population is 1.13 to 1.35 times its 2012 population, national 1.17. It
separates Cantagalo from Mé-Zóchi by a factor of three, which is what closes the pair area
leaves open.

COD-PS 2022 is deliberately not used. It is a projection, its pcodes do not join COD-AB's,
and the 2024 census is a better witness than a projection of the 2012 one.

Usage:
    python sources/st_geo.py --fetch    one ~110 KB zip from HDX
    python sources/st_geo.py            rebuild from data/raw/st/
"""

import math
import os
import sys
import unicodedata
import urllib.request
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

RAW = os.path.join(ROOT, "data", "raw", "st")
SHP_DIR = os.path.join(RAW, "shp")
OUT_DIR = os.path.join(ROOT, "data", "geo", "st")
OUT = os.path.join(OUT_DIR, "st_districts.gpkg")
LOOKUP = os.path.join(OUT_DIR, "st_lookup.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}

DOWNLOADS = {
    "stp_admin_boundaries.shp.zip":
        "https://data.humdata.org/dataset/9746c0f7-9ed3-4a3f-890e-88bc19166770/resource/"
        "8a409429-f4a1-4132-8747-9a929df0c376/download/stp_admin_boundaries.shp.zip",
}

EXPECTED_UNITS = 7

# COD-AB's own spelling, so a rename stops the file rather than moving the ambiguity.
# Six of the seven fold to INE's spelling; the seventh is COD's English gloss of the
# autonomous region, which is the only alias this country needs.
COD_NAME = {
    "ST11": "Príncipe (Autonomous Region)",
    "ST21": "Água Grande",
    "ST22": "Cantagalo",
    "ST23": "Caué",
    "ST24": "Lembá",
    "ST25": "Lobata",
    "ST26": "Mé-Zóchi",
}
ALIAS = {"ST11": "principe autonomous region"}

# Which island each district is on, for the record and for `st_grid.py`'s stray report.
ISLAND = {"ST11": "Príncipe", "ST21": "São Tomé", "ST22": "São Tomé", "ST23": "São Tomé",
          "ST24": "São Tomé", "ST25": "São Tomé", "ST26": "São Tomé"}

# EPSG:32632 is UTM 32N, which covers the whole country; both islands sit within a degree
# of the equator and of 6.5-7.5°E, so a single UTM zone measures area honestly.
AREA_CRS = "EPSG:32632"
AREA_TOLERANCE = 0.12          # |polygon / census printed area - 1|, per district
# Cantagalo (128.9 km² measured, 118.5 printed) and Mé-Zóchi (120.8, 122.0) are inside each
# other's error, so the printed areas cannot tell them apart. Nothing else is.
AREA_TIED = {"ST22", "ST26"}
GROWTH_BAND = (1.05, 1.50)     # 2024 population / 2012 population, per district


def fold(s):
    s = unicodedata.normalize("NFKD", str(s).strip().lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join("".join(ch if ch.isalnum() else " " for ch in s).split())


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for name, url in DOWNLOADS.items():
        dst = os.path.join(RAW, name)
        if os.path.exists(dst):
            print(f"  have {name} ({os.path.getsize(dst):,} bytes)")
            continue
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=300) as r, open(dst + ".part", "wb") as f:
            f.write(r.read())
        os.replace(dst + ".part", dst)
        print(f"  got  {name} ({os.path.getsize(dst):,} bytes)")


def check_names(g, ine):
    """Witness 1 — COD's names against INE's, and the one gloss that is not a fold."""
    print(f"\n  witness 1 — names")
    unmatched = []
    for pcode, name in COD_NAME.items():
        got = g.loc[g["pcode"] == pcode, "adm1_name"].iloc[0]
        if fold(got) != fold(name):
            raise SystemExit(f"{pcode}: COD now calls it {got!r}, not {name!r} — the "
                             "bundle has been reissued and this file's ALIAS may have moved")
        if fold(got) == fold(ine[pcode]):
            print(f"      {pcode}  {ine[pcode]:<30} folds to COD's {got!r}")
        else:
            if ALIAS.get(pcode) != fold(got):
                raise SystemExit(f"{pcode}: INE says {ine[pcode]!r}, COD says {got!r}, and "
                                 "that pairing is not in ALIAS")
            unmatched.append(pcode)
            print(f"      {pcode}  {ine[pcode]:<30} does NOT fold to COD's {got!r}; "
                  "paired by ALIAS")
    if len(unmatched) != len(ALIAS):
        raise SystemExit(f"{len(unmatched)} names needed an alias, ALIAS has {len(ALIAS)}")


def check_areas(g, census2024, ine):
    """Witness 2 — each polygon measured, against the area the 2024 census prints for it.

    This is what the join rests on. It uses no population figure at all, so it is
    independent of witness 3 as well as of COD-PS, and the sixteenfold spread between
    Água-Grande and Caué means a permutation cannot survive it.
    """
    print(f"\n  witness 2 — polygon area against the V RGPH 2024's own Tabela 1.3:")
    m = g.to_crs(AREA_CRS)
    measured = dict(zip(m["pcode"], m.geometry.area / 1e6))
    worst = 0.0
    for pcode in sorted(COD_NAME, key=lambda p: -measured[p]):
        printed = census2024[pcode][1]
        err = measured[pcode] / printed - 1.0
        worst = max(worst, abs(err))
        print(f"      {pcode}  {ine[pcode]:<30}{measured[pcode]:>8.1f} km² vs "
              f"{printed:>6.1f} printed  ({err:+.1%})")
        if abs(err) > AREA_TOLERANCE:
            raise SystemExit(f"{pcode} ({ine[pcode]}) is {err:+.0%} off the area the census "
                             f"prints for it — the polygon and the name are not paired")
    print(f"      worst residual {worst:.1%}, inside {AREA_TOLERANCE:.0%}")

    # How much of the pairing area actually pins. Every alternative assignment of the seven
    # printed areas to the seven polygons is scored, and the ones that beat the truth are
    # allowed only if they move nothing outside AREA_TIED — the pair the country's own
    # geography leaves indistinguishable. If a THIRD district ever joins that set, area has
    # stopped being a witness and this says so rather than passing quietly.
    import itertools
    codes = sorted(COD_NAME)

    def cost(perm):
        return sum(abs(math.log(measured[a] / census2024[b][1]))
                   for a, b in zip(codes, perm))

    true = cost(codes)
    beat, moved = 0, set()
    for p in itertools.permutations(codes):
        if list(p) == codes or cost(p) >= true:
            continue
        beat += 1
        moved |= {a for a, b in zip(codes, p) if a != b}
    print(f"      the identity pairing costs {true:.3f}; {beat} of the other "
          f"{math.factorial(len(codes)) - 1:,} permutations beat it, and every one of them "
          f"moves only {sorted(moved) or 'nothing'}")
    if not moved <= AREA_TIED:
        raise SystemExit(f"area cannot separate {sorted(moved)}, which is more than the "
                         f"{sorted(AREA_TIED)} this file expects — the polygons have moved "
                         "and this witness needs re-reading, not relaxing")
    if moved:
        print(f"      so area pins {len(codes) - len(moved)} of {len(codes)} outright and "
              f"leaves {sorted(moved)} to witness 3, which separates them by a factor of 3")


def check_growth(census2012, census2024, ine):
    """Witness 3 — the same office counting the same districts twelve years apart."""
    print(f"\n  witness 3 — 2012 against 2024, both INE's own counts:")
    lo, hi = GROWTH_BAND
    nat12 = sum(census2012.values())
    nat24 = sum(p for p, _ in census2024.values())
    for pcode in sorted(census2012, key=lambda p: -census2012[p]):
        r = census2024[pcode][0] / census2012[pcode]
        print(f"      {pcode}  {ine[pcode]:<30}{census2012[pcode]:>8,} -> "
              f"{census2024[pcode][0]:>8,}  ({r:.3f}x)")
        if not lo <= r <= hi:
            raise SystemExit(f"{pcode} ({ine[pcode]}) grew {r:.2f}x between the censuses, "
                             f"outside {lo}-{hi} — check the pairing")
    print(f"      national {nat12:,} -> {nat24:,} ({nat24 / nat12:.3f}x); every district "
          f"inside {lo}-{hi}")

    # And it has to close the pair witness 2 left open, or the join rests on names alone.
    a, b = sorted(AREA_TIED)
    swapped = [census2024[b][0] / census2012[a], census2024[a][0] / census2012[b]]
    print(f"      swapping {a} and {b}, the pair area cannot separate, gives "
          f"{swapped[0]:.2f}x and {swapped[1]:.2f}x")
    if lo <= min(swapped) and max(swapped) <= hi:
        raise SystemExit(f"swapping {a} and {b} also fits the growth band — neither witness "
                         "separates them and the join rests on the names alone")


def main():
    if "--fetch" in sys.argv:
        fetch()

    import st as stmod

    zpath = os.path.join(RAW, "stp_admin_boundaries.shp.zip")
    if not os.path.exists(zpath):
        raise SystemExit(f"{zpath} missing — run with --fetch")
    os.makedirs(SHP_DIR, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(SHP_DIR)

    shp = os.path.join(SHP_DIR, "stp_admin1.shp")
    g = gpd.read_file(shp, engine="fiona")
    if len(g) != EXPECTED_UNITS:
        raise SystemExit(f"{len(g)} ADM1 features, expected {EXPECTED_UNITS} — COD has "
                         "re-levelled São Tomé, and the districts may now be ADM2")
    if g.crs is None or g.crs.to_epsg() != 4326:
        raise SystemExit(f"unexpected CRS {g.crs}")
    minx, miny, maxx, maxy = g.total_bounds
    if not (6.0 < minx and maxx < 8.0 and -0.5 < miny and maxy < 2.0):
        raise SystemExit(f"the layer's bbox {g.total_bounds} is not São Tomé and Príncipe")
    print(f"read {shp}: {len(g)} districts, {g.crs}")

    g["pcode"] = g["adm1_pcode"].astype(str).str.strip()
    if sorted(g["pcode"]) != sorted(COD_NAME):
        raise SystemExit(f"COD's pcodes are {sorted(g['pcode'])}, expected "
                         f"{sorted(COD_NAME)}")

    districts = stmod.read()
    ine = {p: n for p, (n, _, _) in districts.items()}
    census2012 = {p: t for p, (_, t, _) in districts.items()}
    census2024 = stmod.read_2024()
    if sorted(ine) != sorted(COD_NAME):
        raise SystemExit("st.py's DISTRICTS and COD-AB do not cover the same pcodes")

    check_names(g, ine)
    check_areas(g, census2024, ine)
    check_growth(census2012, census2024, ine)

    g["unit"] = g["pcode"]
    g["geo_id"] = g["pcode"]
    g["name"] = g["pcode"].map(ine)                 # INE's spelling, not COD's gloss
    g["cod_name"] = g["adm1_name"]
    g["island"] = g["pcode"].map(ISLAND)
    g["pop"] = g["pcode"].map(census2012).astype("int64")
    g["pop2024"] = g["pcode"].map({p: v[0] for p, v in census2024.items()}).astype("int64")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = g[["unit", "name", "cod_name", "island", "pcode", "geo_id", "pop", "pop2024",
             "geometry"]]
    tmp = OUT[:-5] + ".part.gpkg"
    if os.path.exists(tmp):
        os.remove(tmp)
    out.to_file(tmp, layer="districts", driver="GPKG")
    os.replace(tmp, OUT)
    print(f"\nwrote {OUT} ({len(out)} polygons, {out['pop'].sum():,} people in 2012)")

    lut = out.drop(columns="geometry").sort_values("pcode")
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows)")


if __name__ == "__main__":
    main()
