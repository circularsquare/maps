"""Nigeria — boundaries and populations for the 36 states and the Federal Capital Territory.

Writes data/geo/ng/ng_states.gpkg and data/geo/ng/ng_lookup.csv.

  * **boundaries** — OCHA COD-AB Nigeria (`cod-ab-nga`), the **shapefile** bundle rather than
    the geodatabase on §12's Chile rule, read with `engine="fiona"`. 37 ADM1 features,
    `NG001`..`NG037`, alphabetical by name, `valid_on` 2019-04-17. The same bundle carries
    ADM2 (774 LGAs) and ADM3 (wards); neither is used, because nothing measures religion at
    either.
  * **populations** — OCHA COD-PS Nigeria 2022 (`cod-ps-nga`), `nga_admpop_adm1_2022.csv`,
    216,798,930 people. Same pcodes, so the join is on the code and is asserted to be a
    bijection rather than eyeballed.

## THE POPULATION IS A PROJECTION OFF A DISPUTED CENSUS, AND THERE IS NOTHING BETTER

**Nigeria last counted between 21 and 27 March 2006, and the state totals were disputed.** The
final results were gazetted on 2 February 2009 at 140,431,790 people (Extraordinary Official
Gazette No. 2, Vol. 96), after a 2007 provisional gazette the notice says they revise by
428,248 people without changing any state's rank. Lagos State ran a parallel count and put
itself at roughly twice its gazetted 9,113,605, and the north/south balance of the totals was
argued over more broadly. A 2023 census was announced and postponed and has not been held. So
every current Nigerian population number is a projection off a 2006 base that part of the
country does not accept, and COD-PS is that projection: NPC's own, carried by UNFPA, 2022
vintage.

**It is used anyway, and the reason is that the alternative is worse rather than that the
objection is wrong.** The disputed part of 2006 is the north/south balance of the totals, which
is the same axis this map's subject runs along, so it cannot be waved away; `sources/ng.md`
records it and `countries.py`'s `note_public` says it to the reader. What there is no case for
is substituting the 2006 raw count, which is the disputed number itself and sixteen years older,
or a rival projection nobody has published at state level. Lagos's own figure is a claim by one
state about itself and there is no equivalent for the other thirty-six, so adopting it would
re-weight the country on one unit's say-so.

The comparison against 2006 is printed on every run so the size of the extrapolation is on the
record rather than assumed. It is large: 216.8 million against 140.4 million is +54% in sixteen
years, and the states that gain most are not the ones a reader would guess.

## THE STATE NAMES ARE STABLE AND THE JOIN IS A CODE JOIN

Nigeria has had 36 states and the FCT since 1 October 1996 and has created none since, so
unlike most of this map there is no vintage question in the geography at all. COD-AB and COD-PS
are both OCHA products off the same pcode list, so the join is `adm1_pcode` and every name
agrees; `check_join()` asserts both directions and asserts the names agree too, because two
OCHA files sharing a code list is a reason to expect a clean join and not a reason to skip
testing it ([[reference_name_join_wrong_neighbour]]).

The Afrobarometer's own spellings are a different problem and are handled in `sources/ng.py`,
which is where the label set that actually varies lives.

Usage:
    python sources/ng_geo.py --fetch    a ~10 MB zip and a 15 KB csv from HDX
    python sources/ng_geo.py            rebuild from data/raw/ng/
"""

import os
import sys
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ng")
SHP_DIR = os.path.join(RAW, "shp")
OUT_DIR = os.path.join(ROOT, "data", "geo", "ng")
OUT = os.path.join(OUT_DIR, "ng_states.gpkg")
LOOKUP = os.path.join(OUT_DIR, "ng_lookup.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}

N_STATES = 37          # 36 states plus the Federal Capital Territory, unchanged since 1996

DOWNLOADS = {
    # OCHA COD-AB, https://data.humdata.org/dataset/cod-ab-nga
    "nga_admin_boundaries.shp.zip":
        "https://data.humdata.org/dataset/81ac1d38-f603-4a98-804d-325c658599a3/resource/"
        "01c65fd9-bd0c-4608-aa1e-2e86bbccf3e5/download/nga_admin_boundaries.shp.zip",
    # OCHA COD-PS 2022, https://data.humdata.org/dataset/cod-ps-nga
    "nga_admpop_adm1_2022.csv":
        "https://data.humdata.org/dataset/a7c3de5e-ff27-4746-99cd-05f2ad9b1066/resource/"
        "d9fc551a-b5e4-4bed-9d0d-b047b6961817/download/nga_admpop_adm1_2022.csv",
}

CODPS_TOTAL = 216_798_930      # asserted, so a re-release is a failure here rather than a drift

# Federal Republic of Nigeria, EXTRAORDINARY Official Gazette No. 2, Vol. 96, 2 February 2009,
# S.I. No. 1 of 2009, "Legal Notice on Publication of 2006 Census Final Results", pages B2-B3.
# The FINAL results, which the same notice says differ from the 2007 provisional gazette by
# 428,248 people nationally (+0.3%) and change no state's rank. Transcribed from the gazette
# PDF and keyed on the gazette's OWN uppercase names, not on rank order: the notice numbers the
# states 1..37 alphabetically with FCT last, and COD-AB numbers them alphabetically with
# `Federal Capital Territory` fifteenth, so a positional join is off by one for twenty-two
# states and every national total still adds up ([[reference_name_join_wrong_neighbour]]).
#
# Used ONLY to print how far the projection has moved; nothing is drawn from it.
CENSUS_2006 = {
    "ABIA": 2_845_380,        "ADAMAWA": 3_178_950,     "AKWA IBOM": 3_902_051,
    "ANAMBRA": 4_177_828,     "BAUCHI": 4_653_066,      "BAYELSA": 1_704_515,
    "BENUE": 4_253_641,       "BORNO": 4_171_104,       "CROSS RIVER": 2_892_988,
    "DELTA": 4_112_445,       "EBONYI": 2_176_947,      "EDO": 3_233_366,
    "EKITI": 2_398_957,       "ENUGU": 3_267_837,       "GOMBE": 2_365_040,
    "IMO": 3_927_563,         "JIGAWA": 4_361_002,      "KADUNA": 6_113_503,
    "KANO": 9_401_288,        "KATSINA": 5_801_584,     "KEBBI": 3_256_541,
    "KOGI": 3_314_043,        "KWARA": 2_365_353,       "LAGOS": 9_113_605,
    "NASARAWA": 1_869_377,    "NIGER": 3_954_772,       "OGUN": 3_751_140,
    "ONDO": 3_460_877,        "OSUN": 3_416_959,        "OYO": 5_580_894,
    "PLATEAU": 3_206_531,     "RIVERS": 5_198_716,      "SOKOTO": 3_702_676,
    "TARABA": 2_294_800,      "YOBE": 2_321_339,        "ZAMFARA": 3_278_873,
    "FCT ABUJA": 1_406_239,
}
CENSUS_2006_TOTAL = 140_431_790

# The gazette's name -> COD's, where they differ. Exactly one, and it is the capital territory.
GAZETTE_TO_COD = {"FCT ABUJA": "Federal Capital Territory"}


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for name, url in DOWNLOADS.items():
        dst = os.path.join(RAW, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 500:
            print(f"  have {name} ({os.path.getsize(dst):,} bytes)")
            continue
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=900) as r:
            data = r.read()
        # §5a: a 200 is not a download.
        head = data[:2]
        if name.endswith(".zip") and head != b"PK":
            raise SystemExit(f"{name} is not a zip — starts {data[:16]!r}")
        if name.endswith(".csv") and not data[:4].isascii():
            raise SystemExit(f"{name} is not a csv — starts {data[:16]!r}")
        with open(dst + ".part", "wb") as f:
            f.write(data)
        os.replace(dst + ".part", dst)
        print(f"  got  {name} ({os.path.getsize(dst):,} bytes)")
    with zipfile.ZipFile(os.path.join(RAW, "nga_admin_boundaries.shp.zip")) as z:
        z.extractall(SHP_DIR)


def check_join(g, ps):
    """Both directions, on the code AND on the name.

    Two OCHA products off one pcode list is a reason to expect this to pass and not a reason to
    skip it. The failure mode a code join has here is not a mismatch, which raises; it is a
    SILENT re-cut, where a later COD-PS keeps the codes and moves what they mean. Asserting the
    names as well catches that, which the codes alone cannot.
    """
    gcodes, pcodes = set(g["unit"]), set(ps["ADM1_PCODE"])
    if gcodes != pcodes:
        raise SystemExit(f"COD-AB and COD-PS do not cover the same states: "
                         f"AB only {sorted(gcodes - pcodes)}, PS only {sorted(pcodes - gcodes)}")
    names_ab = dict(zip(g["unit"], g["name"]))
    names_ps = dict(zip(ps["ADM1_PCODE"], ps["ADM1_EN"]))
    differ = {c: (names_ab[c], names_ps[c]) for c in gcodes if names_ab[c] != names_ps[c]}
    if differ:
        raise SystemExit(f"the two files give a pcode different names: {differ} — one of them "
                         "has been re-cut and the code join is now joining two things")
    print(f"  COD-AB x COD-PS: {len(gcodes)} pcodes, every name identical")


def main():
    if "--fetch" in sys.argv:
        fetch()

    shp = os.path.join(SHP_DIR, "nga_admin1.shp")
    if not os.path.exists(shp):
        raise SystemExit(f"missing {shp} — run with --fetch first")

    # engine="fiona": pyogrio is the engine that has silently returned zero features here.
    g = gpd.read_file(shp, engine="fiona")
    if len(g) != N_STATES:
        raise SystemExit(f"{shp} has {len(g)} features, expected {N_STATES}")
    print(f"COD-AB Nigeria admin1: {len(g)} states, crs={g.crs}, "
          f"valid_on {sorted(set(g['valid_on']))}")

    g = g.rename(columns={"adm1_pcode": "unit", "adm1_name": "name"})
    g["geo_id"] = g["unit"]

    ps = pd.read_csv(os.path.join(RAW, "nga_admpop_adm1_2022.csv"))
    if len(ps) != N_STATES:
        raise SystemExit(f"COD-PS has {len(ps)} rows, expected {N_STATES}")
    if int(ps["T_TL"].sum()) != CODPS_TOTAL:
        raise SystemExit(f"COD-PS sums to {int(ps['T_TL'].sum()):,}, not the {CODPS_TOTAL:,} "
                         "this file was written against — HDX has re-released it, so re-read "
                         "the vintage before anything downstream is trusted")
    check_join(g, ps)

    g["pop"] = g["unit"].map(dict(zip(ps["ADM1_PCODE"], ps["T_TL"]))).astype("int64")
    if int(g["pop"].sum()) != CODPS_TOTAL:
        raise SystemExit("the joined populations do not sum to the COD-PS total")

    # ---- how far the projection has moved, printed and never used ----
    if sum(CENSUS_2006.values()) != CENSUS_2006_TOTAL:
        raise SystemExit(f"the 2006 transcription sums to {sum(CENSUS_2006.values()):,}, not "
                         f"the {CENSUS_2006_TOTAL:,} the gazette states in words — a digit is "
                         "wrong somewhere in it")
    by_cod = {GAZETTE_TO_COD.get(k, k).casefold(): v for k, v in CENSUS_2006.items()}
    if len(by_cod) != N_STATES:
        raise SystemExit("two gazette names fold to the same COD name")
    unmatched = sorted(n for n in g["name"] if n.casefold() not in by_cod)
    if unmatched:
        raise SystemExit(f"COD states with no 2006 row: {unmatched} — add the spelling to "
                         "GAZETTE_TO_COD rather than dropping the state")
    g["pop2006"] = g["name"].str.casefold().map(by_cod).astype("int64")
    g["growth"] = g["pop"] / g["pop2006"]
    g["share_now"] = g["pop"] / g["pop"].sum()
    g["share_2006"] = g["pop2006"] / CENSUS_2006_TOTAL

    print(f"\n  COD-PS 2022 {CODPS_TOTAL:,} against the 2006 census {CENSUS_2006_TOTAL:,} — "
          f"+{100 * (CODPS_TOTAL / CENSUS_2006_TOTAL - 1):.1f}% in sixteen years")
    moved = g.assign(d=(g["share_now"] - g["share_2006"]) * 100).sort_values("d")
    print("    states whose SHARE of Nigeria the projection moves most (percentage points):")
    for _i, r in pd.concat([moved.head(4), moved.tail(4)]).iterrows():
        print(f"      {r['name']:<28}{r['share_2006'] * 100:6.2f}% -> "
              f"{r['share_now'] * 100:5.2f}%   {r['d']:+5.2f}pp   x{r['growth']:.2f}")
    lag = g.loc[g["unit"] == "NG025"].iloc[0]
    print(f"    and the one state that ran its own count: Lagos, gazetted at "
          f"{int(lag['pop2006']):,} in 2006 and claiming roughly twice that; COD-PS 2022 has "
          f"it at {int(lag['pop']):,}, which is still below the state's own 2006 claim")

    print(f"\n  {len(g)} states, {int(g['pop'].sum()):,} people (COD-PS 2022):")
    for _i, r in g.sort_values("pop", ascending=False).iterrows():
        print(f"    {r['unit']}  {r['name']:<28}{int(r['pop']):>12,}  "
              f"{r['area_sqkm']:>10,.0f} km²")

    os.makedirs(OUT_DIR, exist_ok=True)
    keep = g[["geo_id", "unit", "name", "pop", "pop2006", "area_sqkm", "geometry"]]
    keep.to_file(OUT, layer="states", driver="GPKG")
    pd.DataFrame(keep.drop(columns="geometry")).to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"\nwrote {OUT}")
    print(f"wrote {LOOKUP}")


if __name__ == "__main__":
    main()
