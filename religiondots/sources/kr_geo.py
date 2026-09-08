"""South Korea — boundaries and placement grid for the 229 drawn sigungu.

Writes:
    data/geo/kr/kr_sigungu.gpkg           the matched units (`units`)
    data/geo/kr/kr_grid_400m.gpkg         Kontur H3 r8 hexes with `unit` and `pop` (`place`)
    data/geo/kr/kr_lookup.csv             unit -> name, romanisation, province, populations

Usage:
    python sources/kr_geo.py --fetch      three downloads, ~12 MB
    python sources/kr_geo.py              rebuild from data/raw/kr/ and data/geo/kontur/

THE JOIN IS TWO STEPS AND THE FIRST ONE IS FREE. geoBoundaries KOR **ADM1** carries
`shapeISO` = ISO 3166-2:KR (`KR-11` Seoul … `KR-49` Jeju), which is Guyana's trick again, so
the seventeen provinces bridge to KOSIS's seventeen with no name matching at all. **ADM2
carries no code — `shapeISO` is empty on all 228 features** — so the second step, sigungu
inside a province, has to be by name.

AND THE TWO SIDES ARE IN DIFFERENT SCRIPTS. KOSIS writes 종로구; geoBoundaries writes
`Jongno-gu`. There is no shared key, so this file romanises the Korean and matches on a fold.
**Crude romanisation is deliberate**: Revised Romanization's inter-syllable assimilation
(종로 -> Jongno, not Jongro) is the hard half and implementing it wrong is worse than not
implementing it, so `romanise()` transliterates jamo only and the fold absorbs the
difference. That is safe ONLY because of where it is applied.

**THE FOLD IS APPLIED INSIDE A PROVINCE AND NOWHERE ELSE**, which is spec §12's Sri Lanka
rule verbatim: a fold aggressive enough to bridge Jongro/Jongno is far too aggressive to be a
national key. Nationally 중구 appears six times, 동구 six, 서구 five and 남구 five — Korea has
more colliding district names than any country on this map — so a global match would pair one
metropolis's Jung-gu with another's, and every provincial and national total would still
reconcile (§9n's Ghana `TMA` failure exactly). Within a province the names are unique, which
`sources/kr.py` asserts, and every match here is required to be 1:1 or it is reported.

229 KOSIS UNITS AGAINST 228 POLYGONS, AND THE MISSING ONE IS A WHOLE COUNTY. **영광군,
Yeonggwang-gun in South Jeolla, 53,984 people, is simply absent from ADM2** — no polygon of
that name and nothing covering that ground. It is rebuilt from the eleven ADM3 eup and myeon
that fall inside no ADM2 polygon, which come to 481 km² against a published 475; see
`patch_hole()`. Spec §12's Philippines rule, and the hole is patched BEFORE the join runs so
that every count downstream is 229 against 229.

**The missing unit was NOT the one that looked obvious.** Sejong was the expectation — it is a
special self-governing city that is both a province and its own single sigungu, so it is the
natural candidate for a tier that does not exist — and it turned out to have an ADM2 polygon
like everything else. Guessing which unit is missing wastes time; the join reports it.

AND ONE MISSING UNIT CASCADED INTO A SECOND, WRONG-LOOKING FAILURE. With Yeonggwang absent,
South Jeolla was one polygon short, so the count-constrained assignment below handed it
Gwangju's `Buk-gu` — and the visible symptom was **Gwangju** coming up short, 500 km from the
actual defect. Two units unplaced, one real cause. Fix the hole and both resolve.

INDEPENDENT CHECK: Kontur against the census, per unit. The name fold decides which polygon
is which district; it does not decide how many people a modelled surface puts there. A
mispairing inside a province — Jung-gu for Dong-gu in the same metropolis — would show up as
two units with wildly wrong ratios, because Korean city districts differ in population by an
order of magnitude. That is the only check available: there is no population column in the
boundary file and no code on the census side.
"""

import csv
import gzip
import os
import re
import shutil
import sys
import unicodedata
from difflib import SequenceMatcher

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "kr")
OUT_DIR = os.path.join(ROOT, "data", "geo", "kr")
KONTUR_DIR = os.path.join(ROOT, "data", "geo", "kontur")

UNITS_OUT = os.path.join(OUT_DIR, "kr_sigungu.gpkg")
GRID_OUT = os.path.join(OUT_DIR, "kr_grid_400m.gpkg")
LOOKUP = os.path.join(OUT_DIR, "kr_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "kr.csv")

GB = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/KOR/"
      "{lvl}/geoBoundaries-KOR-{lvl}.geojson")
ADM1 = os.path.join(RAW, "geoBoundaries-KOR-ADM1.geojson")
ADM2 = os.path.join(RAW, "geoBoundaries-KOR-ADM2.geojson")
ADM3 = os.path.join(RAW, "geoBoundaries-KOR-ADM3.geojson")

# THE BOUNDARY FILE HAS A HOLE, AND IT IS A WHOLE COUNTY.
# geoBoundaries KOR ADM2 is 228 polygons against the census's 229 sigungu, and the missing
# one is **영광군, Yeonggwang-gun in South Jeolla** — 53,984 people, simply absent. Not a
# rename, not a merge, not a vintage split: there is no polygon of that name and no polygon
# covering that ground at ADM2 at all.
#
# It is reconstructable, which is spec §12's Philippines rule ("a unit with no polygon
# anywhere may still be reconstructable from its parts"). ADM3 carries all eleven of its
# eup and myeon, and they are exactly the eleven that fall inside NO ADM2 polygon. The list
# below is the county's own administrative composition, and it is used as a CHECK on a
# geometric selection rather than as the selection itself: the script takes the loose ADM3
# units, requires these eleven to be among them, and asserts the dissolved area against the
# published one. Three independent conditions, so a future release that fills the hole or
# renames a myeon fails loudly instead of drawing a wrong county.
YEONGGWANG = "영광군"
YEONGGWANG_PARTS = [
    "Yeonggwang-eup", "Baeksu-eup", "Hongnong-eup", "Daema-myeon", "Myoryang-myeon",
    "Bulgap-myeon", "Gunseo-myeon", "Gunnam-myeon", "Yeomsan-myeon", "Beopseong-myeon",
    "Nagwol-myeon",
]
YEONGGWANG_KM2 = 475.0          # published area; Nagwol-myeon is islands, so tolerance is wide

KONTUR_URL = ("https://geodata-eu-central-1-kontur-public.s3.eu-central-1.amazonaws.com/"
              "kontur_datasets/kontur_population_KR_20231101.gpkg.gz")
KONTUR_GZ = os.path.join(KONTUR_DIR, "kontur_population_KR_20231101.gpkg.gz")
KONTUR = os.path.join(KONTUR_DIR, "kontur_population_KR_20231101.gpkg")

EXPECTED_ADM1 = 17
EXPECTED_ADM2 = 228             # before the Yeonggwang hole is patched
EXPECTED_UNITS = 229

# ISO 3166-2:KR. The province bridge, and a published standard rather than a name match.
ISO_PROVINCE = {
    "KR-11": "서울특별시", "KR-26": "부산광역시", "KR-27": "대구광역시",
    "KR-28": "인천광역시", "KR-29": "광주광역시", "KR-30": "대전광역시",
    "KR-31": "울산광역시", "KR-50": "세종특별자치시", "KR-41": "경기도",
    "KR-42": "강원도", "KR-43": "충청북도", "KR-44": "충청남도",
    "KR-45": "전라북도", "KR-46": "전라남도", "KR-47": "경상북도",
    "KR-48": "경상남도", "KR-49": "제주특별자치도",
}

RATIO_LO, RATIO_HI = 0.30, 3.5

# ---- Revised Romanization, jamo only -------------------------------------------------
INITIAL = ["g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj",
           "ch", "k", "t", "p", "h"]
MEDIAL = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo",
          "u", "wo", "we", "wi", "yu", "eu", "ui", "i"]
FINAL = ["", "k", "k", "ks", "n", "nj", "nh", "t", "l", "lk", "lm", "lb", "ls", "lt",
         "lp", "lh", "m", "p", "ps", "t", "t", "ng", "t", "t", "k", "t", "p", "t"]


def romanise(s):
    """Hangul -> latin, jamo by jamo. No assimilation; the fold covers the difference."""
    out = []
    for ch in s:
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7A3:
            c = o - 0xAC00
            out.append(INITIAL[c // 588] + MEDIAL[(c % 588) // 28] + FINAL[c % 28])
        else:
            out.append(ch)
    return "".join(out)


# Administrative suffixes, stripped from both sides before comparison.
SUFFIX = re.compile(r"(si|gun|gu|city|county|district)$")


def _fold1(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z]+", "", s)
    for _ in range(2):                          # "-si" then a trailing "-gu"
        s = SUFFIX.sub("", s)
    # ㄹ AND ㄴ ARE ONE LETTER AS FAR AS THIS FOLD IS CONCERNED. Revised Romanization
    # applies inter-syllable assimilation and romanise() does not, and almost every case
    # that produces is an l/r/n alternation: 종로 is romanised Jongno and transliterates
    # Jongro, 중랑구 is Jungnang-gu and transliterates jungrang. Collapsing all three to one
    # symbol absorbs the whole family without implementing the rules. It is a very
    # aggressive fold and it is why this is only ever applied inside one province.
    s = re.sub(r"[lr]", "n", s)
    s = re.sub(r"(.)\1+", r"\1", s)             # doubled letters from jamo boundaries
    return s


def fold(s):
    """Aggressive, and legal ONLY inside a province (spec §12)."""
    return _fold1(re.sub(r"\[[^\]]*\]", " ", str(s)))


def folds(s):
    """Every form a geoBoundaries label might be matched on.

    THE BRACKET IS SOMETIMES A RENAME AND SOMETIMES A TRANSLATION, and both matter.
    `Jung-gu [Central District]` glosses the name in English, so the bracket is noise; but
    **`Michuhol-gu [Nam-gu]` carries the name the census uses** — Incheon's 남구 became
    미추홀구 in 2018, three years after this census, so the 2015 table says 남구 and the 2020
    boundary file says Michuhol. Dropping the bracket loses the only string the two sides
    share. Both readings are offered and the 1:1 requirement decides which is real.
    """
    s = str(s)
    out = {fold(s)}
    for b in re.findall(r"\[([^\]]*)\]", s):
        out.add(_fold1(b))
    return {f for f in out if f}


def sim(a, b):
    return SequenceMatcher(None, a, b).ratio()


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    os.makedirs(KONTUR_DIR, exist_ok=True)
    for lvl, path in (("ADM1", ADM1), ("ADM2", ADM2), ("ADM3", ADM3)):
        if os.path.exists(path) and os.path.getsize(path) > 50_000:
            print("already have", os.path.basename(path))
            continue
        url = GB.format(lvl=lvl)
        print("GET", url)
        r = requests.get(url, timeout=600, headers={"User-Agent": "religiondots/1.0"})
        r.raise_for_status()
        if b'"FeatureCollection"' not in r.content[:400]:
            raise SystemExit(f"not GeoJSON -- starts {r.content[:80]!r}")
        with open(path, "wb") as fh:
            fh.write(r.content)
        print(f"  {os.path.getsize(path):,} bytes")

    if os.path.exists(KONTUR) and os.path.getsize(KONTUR) > 1_000_000:
        print("already have", os.path.basename(KONTUR))
        return
    if not os.path.exists(KONTUR_GZ) or os.path.getsize(KONTUR_GZ) < 1_000_000:
        print("GET", KONTUR_URL)
        r = requests.get(KONTUR_URL, timeout=900, headers={"User-Agent": "religiondots/1.0"})
        r.raise_for_status()
        if r.content[:2] != b"\x1f\x8b":
            raise SystemExit(f"not gzip: {r.content[:40]!r}")
        with open(KONTUR_GZ, "wb") as fh:
            fh.write(r.content)
        print(f"  {os.path.getsize(KONTUR_GZ):,} bytes")
    with gzip.open(KONTUR_GZ, "rb") as src, open(KONTUR, "wb") as dst:
        shutil.copyfileobj(src, dst)
    print(f"  decompressed to {KONTUR}  {os.path.getsize(KONTUR):,} bytes")


def census():
    """kr.csv -> [(unit_id, name, province, population)] for the 229 drawn sigungu."""
    rows, pop = {}, {}
    with open(NORM, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["geo_level"] != "sigungu":
                continue
            gid = r["geo_id"]
            if gid not in rows:
                prov = re.search(r"parent=([^;]+)", r["note"]).group(1).strip()
                rows[gid] = (r["geo_name"], prov)
            if r["source_category"] == "계":
                pop[gid] = int(r["count"])
    out = [(g, rows[g][0], rows[g][1], pop[g]) for g in sorted(rows)]
    if len(out) != EXPECTED_UNITS:
        raise SystemExit(f"{len(out)} sigungu in {NORM}, expected {EXPECTED_UNITS}")
    return out


def patch_hole(a2):
    """Rebuild Yeonggwang-gun from its ADM3 eup and myeon and append it to ADM2."""
    import geopandas as gpd
    import pandas as pd

    a3 = gpd.read_file(ADM3).to_crs(4326)
    pts = gpd.GeoDataFrame(geometry=a3.geometry.representative_point(), crs=4326)
    hit = gpd.sjoin(pts, a2[["shapeName", "geometry"]], how="left", predicate="within")
    hit = hit[~hit.index.duplicated(keep="first")]
    loose = a3[hit["shapeName"].isna().to_numpy()]

    have = set(loose["shapeName"])
    missing = [p for p in YEONGGWANG_PARTS if p not in have]
    if missing:
        raise SystemExit(
            f"cannot rebuild {YEONGGWANG}: {len(missing)} of its "
            f"{len(YEONGGWANG_PARTS)} eup/myeon are not among the {len(loose)} ADM3 units "
            f"outside every ADM2 polygon -- {missing}. Either geoBoundaries has filled the "
            "hole (check ADM2 for Yeonggwang-gun and delete this function) or the county "
            "has been redistricted.")

    part = loose[loose["shapeName"].isin(YEONGGWANG_PARTS)]
    poly = part.dissolve().geometry.iloc[0]
    km2 = gpd.GeoSeries([poly], crs=4326).to_crs(5179).area.iloc[0] / 1e6
    off = abs(km2 - YEONGGWANG_KM2) / YEONGGWANG_KM2
    if off > 0.15:
        raise SystemExit(f"rebuilt {YEONGGWANG} is {km2:,.0f} km2 against a published "
                         f"{YEONGGWANG_KM2:,.0f} -- {100 * off:.0f}% out, too far to accept")
    print(f"  {YEONGGWANG} is ABSENT from ADM2 and was rebuilt from its "
          f"{len(part)} ADM3 eup/myeon: {km2:,.0f} km2 against a published "
          f"{YEONGGWANG_KM2:,.0f} ({100 * off:.0f}% out)")

    row = gpd.GeoDataFrame([{"shapeName": "Yeonggwang-gun", "shapeISO": "",
                             "geometry": poly}], crs=4326)
    return pd.concat([a2[["shapeName", "shapeISO", "geometry"]], row],
                     ignore_index=True)


def build_units():
    import geopandas as gpd
    import pandas as pd

    a1 = gpd.read_file(ADM1)
    a2 = gpd.read_file(ADM2)
    if len(a1) != EXPECTED_ADM1:
        raise SystemExit(f"{len(a1)} ADM1 features, expected {EXPECTED_ADM1}")
    if len(a2) != EXPECTED_ADM2:
        raise SystemExit(f"{len(a2)} ADM2 features, expected {EXPECTED_ADM2}")
    print(f"  geoBoundaries KOR: {len(a1)} ADM1, {len(a2)} ADM2")

    got = set(a1["shapeISO"])
    if got != set(ISO_PROVINCE):
        raise SystemExit(f"ADM1 shapeISO mismatch.\n  file: {sorted(got)}\n"
                         f"  want: {sorted(ISO_PROVINCE)}")
    a1 = a1.to_crs(4326)
    a1["province"] = a1["shapeISO"].map(ISO_PROVINCE)
    a2 = a2.to_crs(4326)

    # ADM2 -> province by GREATEST OVERLAP, not by point-in-polygon.
    #
    # THE ADM1 POLYGONS OVERLAP EACH OTHER, which point-in-polygon cannot survive. Six of
    # Korea's metropolitan cities are enclaves carved out of the province around them —
    # Gwangju out of South Jeolla, Daegu out of North Gyeongsang, Busan out of South
    # Gyeongsang — and geoBoundaries draws the surrounding province WITHOUT cutting the city
    # out of it. So a point in Gwangju's Dong-gu is inside both `Gwangju` and `South
    # Jeolla`, `sjoin` returns two rows, and taking the first hands whole metropolitan
    # cities to the wrong province: 14 units landed in a neighbour, every one of them in an
    # enclave city or on the Incheon islands.
    #
    # GREATEST OVERLAP DOES NOT FIX IT EITHER, which is the finding. The two layers are
    # different vintages and genuinely misaligned — 85 of 228 districts sit less than 90%
    # inside their best province and Ganghwa-gun, an Incheon island, comes out 77% inside
    # GYEONGGI. A geometric assignment is only as good as the geometry, and this geometry
    # is not good enough to be trusted with the thing the whole join rests on.
    #
    # **So ADM1 is not used for the join at all.** The province assignment is derived from
    # the NAMES instead, in two passes:
    #
    #   1. A fold that is unique on BOTH sides — once in ADM2, once across the whole census
    #      — can be matched globally with no province at all, because there is exactly one
    #      candidate and no collision is possible. ~200 of 229 anchor this way, and each one
    #      TELLS us its polygon's province.
    #   2. What is left is precisely the colliding gu names (Jung-gu, Dong-gu, Nam-gu,
    #      Seo-gu, Buk-gu, Gangseo-gu, Goseong-gun). Each remaining polygon takes the
    #      province of the NEAREST anchored polygon — a Jung-gu is surrounded by its own
    #      city's other districts — and is then matched 1:1 inside it.
    #
    # Geometry is used only where the names are ambiguous, and names only where they are
    # unique. Both halves are checkable and the 1:1 requirement means a wrong pairing is
    # reported rather than resolved (spec §12's Sri Lanka rule).
    a2 = a2.to_crs(4326)
    a2 = patch_hole(a2)

    cen = census()
    by_prov = {}
    for gid, name, prov, pop in cen:
        by_prov.setdefault(prov, []).append((gid, name, pop))

    # ---- pass 1: globally unique folds anchor the provinces --------------------------
    import collections
    cen_fold = {g: fold(romanise(n)) for g, n, p, _ in cen}
    cen_count = collections.Counter(cen_fold.values())
    a2_folds = {i: folds(a2.loc[i, "shapeName"]) for i in a2.index}
    a2_count = collections.Counter(f for fs in a2_folds.values() for f in fs)

    prov_of = {}
    anchors = {}
    for gid, name, prov, pop in cen:
        f = cen_fold[gid]
        if cen_count[f] != 1 or a2_count[f] != 1:
            continue
        hits = [i for i in a2.index if f in a2_folds[i]]
        if len(hits) == 1:
            anchors[gid] = hits[0]
            prov_of[hits[0]] = prov
    print(f"  pass 1: {len(anchors)} units matched on a fold unique on both sides; "
          f"that fixes the province of {len(prov_of)} polygons")

    # ---- pass 2: the rest, by proximity but CONSTRAINED BY THE KNOWN COUNTS -----------
    #
    # Nearest-anchor alone is not enough and fails in exactly the place this country is
    # awkward: Gwangju's Dong-gu and Buk-gu are nearer to South Jeolla's districts, which
    # ring the city, than to Gwangju's own — so both were handed to the province and Gwangju
    # came up two short. The census already says how many units each province has, which
    # turns a guess into an assignment problem: fill the provinces that are SHORT, cheapest
    # pairing first, and a province that is already full cannot steal anyone else's.
    proj = a2.to_crs(5179)
    cent = proj.geometry.centroid
    anchored = [i for i in a2.index if i in prov_of]
    rest = [i for i in a2.index if i not in prov_of]

    need = {p: len(v) for p, v in by_prov.items()}
    for i in anchored:
        need[prov_of[i]] -= 1
    short = {p: n for p, n in need.items() if n > 0}
    # 229 census units against 228 polygons, so the provinces are short by one more than
    # there are loose polygons to give them. That is not an error to raise on -- it is the
    # one genuinely missing unit, and which one it is falls out of the matching below and
    # is reported there rather than guessed at here.
    if sum(short.values()) < len(rest):
        raise SystemExit(f"{len(rest)} unassigned polygons but provinces are short by only "
                         f"{sum(short.values())} -- a polygon has nowhere to go")

    # distance from each loose polygon to the nearest anchored polygon of each short province
    cost = []
    for i in rest:
        for p in short:
            near = [cent[i].distance(cent[j]) for j in anchored if prov_of[j] == p]
            if near:
                cost.append((min(near), i, p))
    cost.sort()
    left = dict(short)
    for _, i, p in cost:
        if i in prov_of or left.get(p, 0) <= 0:
            continue
        prov_of[i] = p
        left[p] -= 1
    stuck = [i for i in rest if i not in prov_of]
    if stuck:
        raise SystemExit(f"{len(stuck)} polygons could not be assigned a province: "
                         f"{[a2.loc[i, 'shapeName'] for i in stuck]}")
    print(f"  pass 2: {len(rest)} polygons with a colliding name assigned by proximity, "
          f"constrained to the {len(short)} provinces that were short")
    a2["province"] = [prov_of[i] for i in a2.index]

    got = collections.Counter(a2["province"])
    want = {p: len(v) for p, v in by_prov.items()}
    diff = {p: (got.get(p, 0), want.get(p, 0)) for p in set(got) | set(want)
            if got.get(p, 0) != want.get(p, 0)}
    print(f"  polygon count per province vs census: "
          f"{'all 17 agree' if not diff else diff}")

    matched, unmatched_census, spare = {}, [], []
    for prov, group in a2.groupby("province"):
        want = by_prov.get(prov, [])
        cand = list(group.index)
        cfold = {i: folds(group.loc[i, "shapeName"]) for i in cand}

        # exact fold first, then best fuzzy, always 1:1
        taken = set()
        pending = []
        for gid, name, pop in want:
            f = fold(romanise(name))
            hits = [i for i in cand if i not in taken and f in cfold[i]]
            if len(hits) == 1:
                matched[gid] = hits[0]
                taken.add(hits[0])
            elif len(hits) > 1:
                raise SystemExit(f"{prov}: {name!r} folds onto {len(hits)} polygons -- "
                                 "the fold is too aggressive for this province")
            else:
                pending.append((gid, name, pop, f))

        # BEST-FIRST, not first-come. Taking each pending unit's best free polygon in list
        # order lets an early mediocre match consume the polygon a later unit needs
        # exactly — that is how 영광군 lost Yeonggwang-gun to a 0.8-scoring neighbour and
        # was then left choosing between Dong-gu and Buk-gu. Scoring every pair first and
        # assigning the most confident one at a time removes the ordering entirely.
        pairs = sorted(((max(sim(f, c) for c in cfold[i]), gid, i)
                        for gid, name, pop, f in pending for i in cand),
                       reverse=True)
        done = set()
        for s, gid, i in pairs:
            if gid in done or i in taken or s < 0.80:
                continue
            matched[gid] = i
            taken.add(i)
            done.add(gid)
        for gid, name, pop, f in pending:
            if gid in done:
                continue
            scored = sorted(((max(sim(f, c) for c in cfold[i]), i)
                             for i in cand if i not in taken), reverse=True)
            near = [(round(s, 2), group.loc[i, "shapeName"]) for s, i in scored[:3]]
            unmatched_census.append((prov, gid, name, romanise(name), near))

        spare += [(prov, group.loc[i, "shapeName"]) for i in cand if i not in taken]

    return a1, a2, cen, matched, unmatched_census, spare



def main():
    if "--fetch" in sys.argv:
        fetch()
    os.makedirs(OUT_DIR, exist_ok=True)

    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import shapely

    a1, a2, cen, matched, unmatched_census, spare = build_units()

    print(f"\n  matched {len(matched)} of {len(cen)} census units by name inside their "
          f"province")
    ok = not unmatched_census and not spare
    print(f"  {'OK ' if ok else 'BAD'} {len(unmatched_census)} census units unplaced, "
          f"{len(spare)} polygons spare")
    for prov, gid, name, rom, near in unmatched_census[:10]:
        print(f"      {prov} {name} ({rom}) -> nearest {near}")
    for prov, nm in spare[:10]:
        print(f"      spare polygon {prov} {nm}")
    if not ok:
        raise SystemExit("the join is incomplete -- fix it before drawing")

    info = {g: (n, p, pop) for g, n, p, pop in cen}
    recs = []
    for gid, idx in matched.items():
        n, p, pop = info[gid]
        recs.append(dict(unit=gid, name=n, province=p, census=pop,
                         gb_name=a2.loc[idx, "shapeName"], geometry=a2.loc[idx, "geometry"]))
    units = gpd.GeoDataFrame(recs, crs=4326).sort_values("unit").reset_index(drop=True)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{len(units)} units built, expected {EXPECTED_UNITS}")

    # ---- placement -------------------------------------------------------------------
    import fiona
    layers = list(fiona.listlayers(KONTUR))
    layer = "population" if "population" in layers else layers[0]
    print(f"\n  reading Kontur r8 hexes from layer {layer!r}…")
    hexes = gpd.read_file(KONTUR, layer=layer)
    if len(hexes) == 0:
        raise SystemExit("Kontur read returned ZERO features -- §12's pyogrio/fiona trap")
    print(f"    {len(hexes):,} hexes, {hexes['population'].sum():,.0f} people")
    hexes = hexes.to_crs(4326)

    centres = gpd.GeoDataFrame(geometry=hexes.geometry.representative_point(), crs=4326)
    hit = gpd.sjoin(centres, units[["unit", "geometry"]], how="left", predicate="within")
    hit = hit[~hit.index.duplicated(keep="first")]
    hexes["unit"] = hit["unit"].to_numpy()
    outside = hexes["unit"].isna()
    print(f"    {outside.sum():,} hexes ({100.0 * outside.mean():.2f}%) outside every unit, "
          f"{hexes.loc[outside, 'population'].sum():,.0f} people — the coastal overrun")
    hexes = hexes[~outside].copy()

    print(f"  clipping {len(hexes):,} hexes to their unit…")
    poly = units.set_index("unit")["geometry"]
    geom = hexes.geometry.to_numpy()
    who = hexes["unit"].to_numpy()
    out = np.empty(len(hexes), dtype=object)
    n_clip = 0
    for unit, parent in poly.items():
        idx = np.flatnonzero(who == unit)
        if idx.size == 0:
            continue
        shapely.prepare(parent)
        inside = shapely.contains_properly(parent, geom[idx])
        out[idx[inside]] = geom[idx[inside]]
        edge = idx[~inside]
        if edge.size:
            out[edge] = shapely.intersection(parent, geom[edge])
            n_clip += edge.size
    print(f"    {n_clip:,} boundary hexes clipped, {len(hexes) - n_clip:,} left whole")
    hexes["geometry"] = gpd.GeoSeries(out, crs=4326, index=hexes.index)
    empty = hexes.geometry.is_empty | hexes.geometry.isna()
    if empty.any():
        print(f"    dropped {empty.sum():,} hexes whose clip came out empty")
        hexes = hexes[~empty].copy()
    hexes = hexes.rename(columns={"population": "pop"})[["unit", "pop", "geometry"]]

    missing = sorted(set(units["unit"]) - set(hexes["unit"]))
    if missing:
        raise SystemExit(f"units with no hex centre: "
                         f"{[info[m][0] for m in missing]} — at r8 over Korean districts "
                         "that means the join is wrong, not that the grid is coarse")

    # ---- the independent check -------------------------------------------------------
    k = hexes.groupby("unit")["pop"].sum()
    units["kontur"] = units["unit"].map(k).astype(float)
    units["ratio"] = units["kontur"] / units["census"]
    tot = units["kontur"].sum() / units["census"].sum()
    bad = units[(units["ratio"] < RATIO_LO) | (units["ratio"] > RATIO_HI)]
    print(f"\n  national Kontur/census ratio {tot:.3f} "
          f"({units['kontur'].sum():,.0f} vs {units['census'].sum():,})")
    print(f"  {'OK ' if bad.empty else 'BAD'} every unit's ratio inside "
          f"[{RATIO_LO}, {RATIO_HI}] ({len(bad)} outside)")
    for _, r in bad.head(12).iterrows():
        print(f"      {r['province']} {r['name']} ({r['gb_name']}) "
              f"census {r['census']:,} kontur {r['kontur']:,.0f} ratio {r['ratio']:.2f}")
    if not bad.empty:
        raise SystemExit("ratio check FAILED — likely a mispaired district")

    worst = units.reindex(units["ratio"].sub(tot).abs().sort_values(ascending=False).index)
    print("  furthest from the national ratio, i.e. weakest placement WITHIN the unit:")
    for _, r in worst.head(4).iterrows():
        print(f"      {r['province']} {r['name']} — {r['ratio']:.2f}x")

    units.to_file(UNITS_OUT, driver="GPKG", layer="sigungu")
    print(f"\nwrote {UNITS_OUT} ({len(units)} units)")
    hexes.to_file(GRID_OUT, driver="GPKG", layer="grid")
    print(f"wrote {GRID_OUT} ({len(hexes):,} hexes)")
    units.drop(columns="geometry").to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP}")


if __name__ == "__main__":
    main()
