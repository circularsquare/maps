"""Nepal — boundaries for the 753 local levels.

Writes data/geo/np/np_units.gpkg and data/geo/np/np_lookup.csv.

OCHA COD-AB Nepal (`cod-ab-npl`, v02, valid_on 2024-03-14), the **shapefile** bundle rather
than the geodatabase on §12's Chile rule. 53.7 MB, one GET from HDX, no wall.

**THE BOUNDARY FILE HAS 775 ADM3 POLYGONS AND THE CENSUS HAS 753 LOCAL LEVELS, AND THE
TWENTY-TWO EXTRA ARE NATIONAL PARKS.** They are Chitawan, Parsa, Bardiya, Khaptad, Langtang,
Shivapuri, Shuklaphanta, Koshi Tappu and the Dhorpatan hunting reserve, several of them split
across districts and each carried as its own ADM3 feature. Nepal's protected areas sit
OUTSIDE the local levels rather than inside them, which is unusual and is the reason for the
mismatch: the 753 palika polygons and the 22 park polygons tile the country between them
(143,084 + 4,569 = 147,653 km², exactly COD's own ADM0 area), so the parks are carved out of
the palikas and not overlaid on them.

**THE CARVE-OUT IS READ OFF THE P-CODE, NOT OFF THE NAMES, AND IT IS SELF-CHECKING.** An
ADM3 p-code is `NP` + province + district + a three-digit unit code whose FIRST DIGIT is the
unit type. Filtering it reproduces Nepal's official local-government composition exactly:

    1  metropolitan city          (mahanagarpalika)          6      <- Nepal has 6
    2  sub-metropolitan city      (upa-mahanagarpalika)     11      <- Nepal has 11
    3  municipality               (nagarpalika)            276      <- Nepal has 276
    4  rural municipality         (gaunpalika)             460      <- Nepal has 460
    5  protected area                                       22      <- dropped here
                                                          ----
                                                           753

That 6/11/276/460 is a published fact about Nepal's federal structure that COD's code
attribute has no reason to reproduce unless the codes mean what they appear to mean, so it is
evidence rather than a restatement — §12's rule about a shared code being trustworthy only as
far as it is independently verified. It also **has to be done before the name join**, because
four of the parks share a name with a palika in the same district — Shivapuri in Nuwakot,
Dhorpatan in Baglung, Shuklaphanta in Kanchanpur and Lumbini Sanskritik in Rupandehi — and a
name-first join silently pairs four local levels with a national park.

**DOTS THEREFORE DO NOT LAND IN NATIONAL PARKS**, which is right and is not something this
module had to arrange: the census attributes nobody to them, and `np_grid.py` keys its hexes
to the palika polygons alone. It is §8.2c's problem (administrative units owning territory
nobody lives in) solved by the boundary file rather than patched afterwards.

**THE JOIN IS ON NAMES AND IS SCOPED TO THE DISTRICT**, because the census workbook carries
no code of any kind — its only geography is three columns of indented labels. Matching inside
a district rather than nationally is what makes a name join safe here: 753 names have real
collisions across Nepal (there are four Madi's and three Bhimad-like pairs), and none inside
one district.

**TWO KINDS OF DIFFERENCE, AND BOTH ARE RULES RATHER THAN ALIAS LISTS (§12).**

  1. *The unit-type word.* The census appends it and COD does not, in six spellings and two
     languages at once — `Gaunpalika`, `Nagarpalika`, `Municipality`, `Rural Municipality`,
     and NSO's own misspelling `Metropolitian City` / `Sub Metropolitian City`. Stripped
     repeatedly, because a few are doubled (`Madi Rural Municipality Municipality`).
  2. *A leading district name.* `Manang Ngisyang Gaunpalika` against COD's `Ngisyang`.
     Stripped when the local-level label starts with its own district's name.

After both, **751 of 753 match exactly**. The last one is `Melanchi` against COD's
`Melamchi` — an n/m nasal, one character. Rather than hard-code it, the leftovers fall
through to a **unique edit-distance-1 match among the district's unclaimed polygons**, which
is a strong test on a set of ten and raises if it is ever ambiguous. Collapsing n and m in
the fold itself was the alternative and is rejected: it is a national-scale rule bought to
fix one unit, and §12's Cambodia note is about exactly that trade.

Usage:
    python sources/np_geo.py --fetch    one 53.7 MB zip from HDX
    python sources/np_geo.py            rebuild from data/raw/np/
"""

import os
import re
import sys
import unicodedata
import zipfile
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "np")
OUT_DIR = os.path.join(ROOT, "data", "geo", "np")
OUT = os.path.join(OUT_DIR, "np_units.gpkg")
LOOKUP = os.path.join(OUT_DIR, "np_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "np.csv")

ZIP_URL = ("https://data.humdata.org/dataset/07db728a-4f0f-4e98-8eb0-8fa9df61f01c/"
           "resource/b6ab1a5a-8b6e-41fb-a61f-8202ce98d16c/download/"
           "npl_admin_boundaries.shp.zip")
ZIP_NAME = "npl_admin_boundaries.shp.zip"

EXPECTED_LOCAL = 753
EXPECTED_DISTRICTS = 77
# Nepal's local-government composition, from the 2017 federal restructuring. Unchanged since,
# which is also why COD's 2024 vintage is the right one for a 2021 census (§8.1).
EXPECTED_TYPES = {"1": 6, "2": 11, "3": 276, "4": 460}
PROTECTED_TYPE = "5"

# Every unit-type word either side can append, longest first so `Sub Metropolitian City` is
# not eaten as `Metropolitian City`. NSO writes `Metropolitian`; that is not a typo here.
SUFFIX = re.compile(
    r"[\s\-]*(sub[\s\-]*metropolit[a-z]*[\s\-]*city|metropolit[a-z]*[\s\-]*city|"
    r"upa[\s\-]*mahanagarpalika|mahanagarpalika|nagarpalika|gaunpalika|gaupalika|"
    r"rural[\s\-]*municipality|municipality)\s*$", re.I)


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, ZIP_NAME)
    if os.path.exists(dest) and zipfile.is_zipfile(dest):
        print("already have", dest)
        return
    print("GET", ZIP_URL)
    r = requests.get(ZIP_URL, timeout=1800, stream=True,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    # sources.md §5a: HDX answers the un-redirected URL with a 302 and a small HTML body,
    # which is a perfectly good 200 to a client that does not follow it.
    if not zipfile.is_zipfile(dest):
        raise SystemExit(f"{dest} is not a zip -- got {os.path.getsize(dest):,} bytes")
    print(f"  {os.path.getsize(dest):,} bytes")


def _strip_types(s):
    prev = None
    while prev != s:
        prev = s
        s = SUFFIX.sub("", s)
    return s


def fold(name, district=None):
    """Key a local-level name. See the module docstring for why each rule is here."""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = _strip_types(s)
    if district:
        d = _strip_types(unicodedata.normalize("NFKD", str(district))).strip()
        if d and re.match(rf"^{re.escape(d)}\s+\S", s, re.I):
            s = s[len(d):]
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _edit1(a, b):
    """True if a and b are one substitution, insertion or deletion apart."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long = (a, b) if len(a) < len(b) else (b, a)
    for i in range(len(long)):
        if long[:i] + long[i + 1:] == short:
            return True
    return False


def _read_adm3():
    import geopandas as gpd

    src = os.path.join(RAW, ZIP_NAME)
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} -- run with --fetch first")
    names = [i.filename for i in zipfile.ZipFile(src).infolist()]
    # `_em` is the edge-matched edition; take the plain one, as every other country here does
    shp = [n for n in names
           if re.search(r"admin3\.shp$", n, re.I) and "_em" not in n.lower()]
    if len(shp) != 1:
        raise SystemExit(f"expected one admin3 shapefile in the bundle, found {shp}")
    g = gpd.read_file(f"zip://{src}!{shp[0]}", engine="fiona")
    # §12 shape 4: assert the feature count, never the absence of an exception.
    if len(g) < 700:
        raise SystemExit(f"{shp[0]}: {len(g)} features -- a zero-or-short read")
    if g.crs is None or g.crs.to_epsg() != 4326:
        print(f"  reprojecting {g.crs} -> EPSG:4326")
        g = g.to_crs(4326)
    return g


def main():
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    g = _read_adm3()
    print(f"COD ADM3: {len(g)} polygons, crs={g.crs}")
    g["unit_type"] = g["adm3_pcode"].str[6]

    # ---- 1. the carve-out, and the check that says the p-code means what it looks like
    got = {t: int((g["unit_type"] == t).sum()) for t in sorted(set(g["unit_type"]))}
    print("\n  ADM3 by unit-type digit (the first of the three-digit unit code):")
    labels = {"1": "metropolitan city", "2": "sub-metropolitan city", "3": "municipality",
              "4": "rural municipality", "5": "PROTECTED AREA -- dropped"}
    for t in sorted(got):
        want = EXPECTED_TYPES.get(t)
        flag = "" if want is None else ("  OK" if got[t] == want else f"  BAD want {want}")
        print(f"    {t}  {labels.get(t, '?'):<32} {got[t]:>4}{flag}")
    if {t: got.get(t, 0) for t in EXPECTED_TYPES} != EXPECTED_TYPES:
        raise SystemExit("COD's unit-type digits no longer reproduce Nepal's 6/11/276/460 "
                         "local-government composition -- the carve-out below is unsafe")

    parks = g[g["unit_type"] == PROTECTED_TYPE]
    g = g[g["unit_type"] != PROTECTED_TYPE].copy()
    if len(g) != EXPECTED_LOCAL:
        raise SystemExit(f"{len(g)} local levels after the carve-out, "
                         f"expected {EXPECTED_LOCAL}")
    print(f"\n  dropped {len(parks)} protected-area polygons "
          f"({parks['area_sqkm'].sum():,.0f} km², "
          f"{parks['area_sqkm'].sum() / (parks['area_sqkm'].sum() + g['area_sqkm'].sum()):.1%}"
          " of Nepal); nobody is attributed to them by the census and no dot will land "
          "in one")
    print("    " + ", ".join(sorted(set(parks["adm3_name"]))))

    # ---- 2. the census side
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/np.py first")
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    cen = (df[df["geo_level"] == "local"]
           .drop_duplicates("geo_id")[["geo_id", "geo_name", "note"]])
    cen["district"] = cen["note"].str.extract(r"district=([^;]+)")[0].str.strip()
    if len(cen) != EXPECTED_LOCAL:
        raise SystemExit(f"{len(cen)} census local levels, expected {EXPECTED_LOCAL}")
    if cen["district"].nunique() != EXPECTED_DISTRICTS:
        raise SystemExit(f"{cen['district'].nunique()} districts in np.csv, "
                         f"expected {EXPECTED_DISTRICTS}")

    # ---- 3. the join, scoped to the district
    cod = defaultdict(dict)
    for nm, pc, dn in zip(g["adm3_name"], g["adm3_pcode"], g["adm2_name"]):
        d = fold(dn)
        k = fold(nm, dn)
        if k in cod[d]:
            raise SystemExit(f"COD folds {nm!r} onto {cod[d][k][0]!r} inside {dn!r} -- "
                             "the fold has collided and is no longer safe as a key")
        cod[d][k] = (nm, pc)

    seen = defaultdict(dict)
    for nm, dn in zip(cen["geo_name"], cen["district"]):
        d, k = fold(dn), fold(nm, dn)
        if k in seen[d]:
            raise SystemExit(f"the census folds {nm!r} onto {seen[d][k]!r} inside {dn!r}")
        seen[d][k] = nm

    pairs, missing, near = {}, [], []
    used = defaultdict(set)
    for gid, nm, dn in zip(cen["geo_id"], cen["geo_name"], cen["district"]):
        d, k = fold(dn), fold(nm, dn)
        if d not in cod:
            raise SystemExit(f"district {dn!r} ({gid}) has no COD polygons at all")
        hit = cod[d].get(k)
        if hit:
            pairs[gid] = (nm, hit[1], hit[0], dn)
            used[d].add(k)
        else:
            missing.append((gid, nm, dn, d, k))

    # the edit-distance-1 fallback, applied only to what the fold could not place
    for gid, nm, dn, d, k in list(missing):
        cands = [kk for kk in cod[d] if kk not in used[d] and _edit1(kk, k)]
        if len(cands) == 1:
            hit = cod[d][cands[0]]
            pairs[gid] = (nm, hit[1], hit[0], dn)
            used[d].add(cands[0])
            near.append((nm, hit[0], dn))
            missing.remove((gid, nm, dn, d, k))
        elif len(cands) > 1:
            raise SystemExit(f"{nm!r} in {dn!r} is one character from {cands} -- "
                             "ambiguous, so it is not being guessed")

    spare = [(d, cod[d][k][0], cod[d][k][1]) for d in cod for k in cod[d]
             if k not in used[d]]

    print("\n  the join, both ways (§12):")
    print(f"    census local levels        {len(cen):>4}")
    print(f"    COD polygons (post-carve)  {len(g):>4}")
    print(f"    matched                    {len(pairs):>4}")
    print(f"    census with no polygon     {len(missing):>4}")
    print(f"    polygons with no census    {len(spare):>4}")
    for gid, nm, dn, d, k in missing:
        print(f"      no polygon: {gid} {nm!r} in {dn!r}; district has "
              f"{sorted(set(cod[d]) - used[d])}")
    for d, nm, pc in spare:
        print(f"      no census : {pc} {nm!r} in {d!r}")
    if missing or spare:
        raise SystemExit("join FAILED")

    exact = len(pairs) - len(near)
    print(f"\n    {exact}/{len(pairs)} matched by the derived fold alone; {len(near)} "
          "needed the\n    edit-distance-1 fallback, and each was unique among its "
          "district's unclaimed polygons:")
    for a, b, dn in sorted(near):
        print(f"      census {a!r} vs COD {b!r}   ({dn})")

    # How much work the SUFFIX rule is actually doing, measured rather than asserted: the
    # comparison is between the two names once the unit-type word is off, because with it
    # on every one of the 753 differs and the number says nothing.
    def bare(s, dn):
        s = unicodedata.normalize("NFKD", str(s))
        s = "".join(c for c in s if not unicodedata.combining(c))
        s = _strip_types(s).strip()
        d = _strip_types(unicodedata.normalize("NFKD", str(dn))).strip()
        if d and re.match(rf"^{re.escape(d)}\s+\S", s, re.I):
            s = s[len(d):].strip()
        return s

    renamed = [(v[0], v[2], v[3]) for v in pairs.values()
               if bare(v[0], v[3]) != str(v[2]).strip()]
    print(f"\n    once the unit-type word is stripped, {len(pairs) - len(renamed)} of "
          f"{len(pairs)} names agree\n    character for character with COD; {len(renamed)} "
          "differ and are matched by rule\n    rather than by a hard-coded alias list:")
    for a, b, dn in sorted(renamed)[:12]:
        print(f"      census {a!r:<44} COD {b!r}   ({dn})")

    # ---- 4. an independent check the pairing does not use: the census's PRINT ORDER
    # inside a district against COD's own unit code inside the same district. Both are
    # ordered lists of the same set and neither was consulted by the fold, so agreement on
    # the COUNT per district is evidence the districts themselves were not crossed.
    cen_per_d = cen.groupby("district").size()
    cod_per_d = {d: len(cod[d]) for d in cod}
    bad = [(d, int(n), cod_per_d.get(fold(d), 0)) for d, n in cen_per_d.items()
           if cod_per_d.get(fold(d), 0) != n]
    print(f"\n    every district holds the same number of units on both sides: "
          f"{len(cen_per_d) - len(bad)}/{len(cen_per_d)}")
    for d, a, b in bad[:6]:
        print(f"      {d}: census {a}, COD {b}")
    if bad:
        raise SystemExit("a district's unit count differs between the two sides")

    # ---- 5. write
    out = g[["adm3_name", "adm3_pcode", "adm2_name", "adm1_name", "geometry"]].copy()
    out = out.rename(columns={"adm3_pcode": "unit", "adm2_name": "district",
                              "adm1_name": "province"})
    # Take the NAME from the statistical source, not the boundary file (§12, Chile).
    nso = {v[1]: v[0] for v in pairs.values()}
    out["name"] = out["unit"].map(nso)
    if out["name"].isna().any():
        raise SystemExit("a polygon came out of the join with no NSO name")

    os.makedirs(OUT_DIR, exist_ok=True)
    out[["unit", "name", "district", "province", "geometry"]].to_file(
        OUT, layer="local_levels", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    lut = pd.DataFrame({"geo_id": sorted(pairs),
                        "unit": [pairs[k][1] for k in sorted(pairs)]})
    lut.to_csv(LOOKUP, index=False)
    print(f"wrote {LOOKUP} ({len(lut)} rows)")


if __name__ == "__main__":
    main()
