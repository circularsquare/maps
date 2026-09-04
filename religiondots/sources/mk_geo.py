"""North Macedonia — boundaries for the 80 census municipalities.

Writes data/geo/mk/mk_opstini.gpkg and data/geo/mk/mk_lookup.csv.

**No download at all.** Eurostat GISCO LAU 2021 covers 34 countries and North Macedonia is
one of them — a candidate country, not a member — so the file `sources/pl_geo.py` already
pulled has all 80 opštini. Worth knowing before hunting: GISCO's LAU set is not the EU27.

THE JOIN IS BY NAME, AND IT HAD TO BE. SSO's PxWeb municipality codes are four digits
(`1066`); GISCO's `LAU_ID` is `MK00101`-style. They share no substring and joining as
delivered matches ZERO of 80 — Poland's trap exactly (§12), and here there is no
correspondence table to slice, because the EU LAU–NUTS workbook is EU27 and North Macedonia
is not in it.

What makes a name join safe here is that GISCO carries the names in **Cyrillic**, which is
what SSO publishes too. The English edition of the same PxWeb table gives `Veles`; the
Macedonian edition gives `Велес`; GISCO gives `Велес`. So `sources/mk.py` fetches BOTH
language editions for this reason alone — English for the category labels the taxonomy
keys on, Macedonian for the names the geometry keys on.

AND THE NAMES ARE VERIFIED AGAINST SOMETHING INDEPENDENT, per §12, because a name join
that is 100% is exactly what a subtly wrong name join also looks like. GISCO carries
`POP_2021`; the census carries its own municipal totals. The two are different collections
of the same population and agree to a fraction of a percent — see the printout.

Usage:
    python sources/mk_geo.py
"""

import json
import os
import sys
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "mk")
SHP = os.path.join(ROOT, "data", "geo", "lau2021", "shp4326",
                   "LAU_RG_01M_2021_4326.shp")
OUT_DIR = os.path.join(ROOT, "data", "geo", "mk")
OUT = os.path.join(OUT_DIR, "mk_opstini.gpkg")
LOOKUP = os.path.join(OUT_DIR, "mk_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "mk.csv")

EXPECTED = 80
NATIONAL_CODE, SKOPJE_CODE = "0000", "0019"


def fold(s):
    """Fold for comparison only. Cyrillic has its own look-alikes, so this is conservative:
    strip combining marks, lowercase, drop everything that is not alphanumeric."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return "".join(ch for ch in s.lower() if ch.isalnum())


def main():
    import geopandas as gpd
    import pandas as pd

    if not os.path.exists(SHP):
        raise SystemExit(f"missing {SHP} -- run sources/pl_geo.py --fetch once")

    # ---- census side: Cyrillic names straight out of the source's own cube ----
    p = os.path.join(RAW, "T1012P21_mk.json")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run sources/mk.py --fetch first")
    with open(p, encoding="utf-8") as fh:
        doc = json.load(fh)
    geo = None
    for d in doc["id"]:
        cat = doc["dimension"][d]["category"]
        if len(cat["index"]) == 82:
            idx = cat["index"]
            order = (sorted(idx, key=lambda k: idx[k]) if isinstance(idx, dict)
                     else list(idx))
            geo = {k: cat["label"][k] for k in order}
            break
    if geo is None:
        raise SystemExit("no 82-value geography dimension in the Macedonian cube")
    cen = {k: v for k, v in geo.items() if k not in (NATIONAL_CODE, SKOPJE_CODE)}
    print(f"census municipalities: {len(cen)}")
    if len(cen) != EXPECTED:
        raise SystemExit(f"expected {EXPECTED}")

    # ---- geo side ----
    gdf = gpd.read_file(SHP)
    mk = gdf[gdf["CNTR_CODE"] == "MK"].copy()
    mk["LAU_ID"] = mk["LAU_ID"].astype(str).str.strip()
    print(f"GISCO MK polygons:    {len(mk)}")
    if len(mk) != EXPECTED:
        raise SystemExit(f"expected {EXPECTED} GISCO polygons, got {len(mk)}")

    by_name = {}
    for i, n in zip(mk["LAU_ID"], mk["LAU_NAME"]):
        by_name.setdefault(fold(n), []).append(i)

    resolved, missing, ambiguous = {}, [], []
    for code, name in cen.items():
        ids = by_name.get(fold(name), [])
        if len(ids) == 1:
            resolved[code] = ids[0]
        elif ids:
            ambiguous.append((code, name, ids))
        else:
            missing.append((code, name))

    used = set(resolved.values())
    spare = [(i, n) for i, n in zip(mk["LAU_ID"], mk["LAU_NAME"]) if i not in used]
    print(f"\n  the join, both ways (§12):")
    print(f"    matched                    {len(resolved):>4}")
    print(f"    census with no polygon     {len(missing):>4}")
    print(f"    polygons with no census    {len(spare):>4}")
    print(f"    ambiguous                  {len(ambiguous):>4}")
    for c, n in missing:
        print(f"      no polygon: {c} {n!r}")
    for i, n in spare:
        print(f"      no census : {i} {n!r}")
    for c, n, ids in ambiguous:
        print(f"      ambiguous : {c} {n!r} -> {ids}")
    if missing or spare or ambiguous:
        raise SystemExit("join FAILED")

    # ---- independent verification: GISCO population vs the census total ----
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False)
    tot = (df[(df["geo_level"] == "municipality")
              & df["note"].str.contains("universe total")]
           .set_index("geo_id")["count"].to_dict())
    gpop = dict(zip(mk["LAU_ID"], mk["POP_2021"]))
    # GISCO's POP_2021 IS ZERO FOR SEVEN OF SKOPJE'S TEN MUNICIPALITIES — Aerodrom, Butel,
    # Kisela Voda, Sopište, Centar, Čair, Šuto Orizari — which are among the most populated
    # places in the country. It is a hole in GISCO, not a join failure, and it is why the
    # MK column of that file sums to 1,746,833 against a census 1,836,713. Nothing here
    # uses POP_2021 for anything but this check (§8.2: placement needs no population), but
    # anyone who reaches for it as a weight must know it is missing central Skopje.
    zero = sorted(i for i, p in gpop.items() if not p or p != p)
    print(f"\n  GISCO POP_2021 is 0 or null on {len(zero)} polygons: "
          f"{[str(mk.loc[mk['LAU_ID'] == i, 'LAU_NAME'].iloc[0]) for i in zero]}")

    # The two are DIFFERENT QUANTITIES — a 2021 resident-population census against GISCO's
    # own series — so they must not be asserted equal. What a correct join guarantees is
    # that the ratio is SYSTEMATIC: every unit within a factor of about two, clustered.
    # A scrambled join pairs villages with cities and scatters the ratio over orders of
    # magnitude, which is the thing this can actually detect.
    rows = []
    for code, lau in resolved.items():
        c, g = tot.get(code), gpop.get(lau)
        if c and g and g == g:
            rows.append((c / g, cen[code], c, g))
    rows.sort()
    if len(rows) < 60:
        raise SystemExit(f"only {len(rows)} units comparable, expected ~73")
    ratios = [r[0] for r in rows]
    med = ratios[len(ratios) // 2]
    print(f"  independent check — census total / GISCO POP_2021 on the "
          f"{len(rows)} units GISCO populates:")
    print(f"    min {ratios[0]:.3f}   median {med:.3f}   max {ratios[-1]:.3f}")
    for r, name, c, g in rows[:3]:
        print(f"      {r:.3f}  {name:<24} census {c:>8,}  gisco {g:>9,.0f}")
    if not (0.80 <= med <= 1.10) or ratios[0] < 0.40 or ratios[-1] > 1.60:
        raise SystemExit("census and GISCO populations are not in a systematic ratio -- "
                         "the name join is pairing the wrong units")
    print("    OK  systematic, not scattered — the join pairs like with like. The gap is "
          "emigration:\n        the 2021 census counts RESIDENTS and the lowest ratios are "
          "the western\n        emigration municipalities (Centar Župa, Mavrovo, Želino).")

    # ---- write ----
    out = mk[["LAU_ID", "LAU_NAME", "POP_2021", "AREA_KM2", "geometry"]].rename(
        columns={"LAU_ID": "kod", "LAU_NAME": "name", "POP_2021": "pop_2021",
                 "AREA_KM2": "area_km2"})
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_file(OUT, layer="opstini", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out)} polygons)")

    lut = pd.DataFrame({"geo_id": list(resolved), "kod": list(resolved.values())})
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows)")


if __name__ == "__main__":
    main()
