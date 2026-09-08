"""Sri Lanka — boundaries for the 14,003 census GN divisions.

Writes data/geo/lk/lk_gnd.gpkg and data/geo/lk/lk_lookup.csv.

Source: OCHA's Common Operational Dataset, `cod-ab-lka` v03, admin4 = Grama Niladhari
division, 14,043 polygons. It is the only GN-level boundary set that exists publicly —
geoBoundaries stops at ADM2, Kontur is a different geography, and DCS publishes no digital
GN boundary of its own.

**THE PCODE JOIN LOOKS RIGHT AND IS WRONG, AND THIS IS THE WHOLE FILE.**

The census gives every GN division a district / DS / GN code triple, and COD's `adm4_pcode`
is exactly `LK` + those three numbers zero-padded — `LK1103005` is district 11, DS 03, GN
005, Sammanthranapura, first row of both files. Joining on that string matches 13,472 of
14,003, which reads like a good join with a vintage gap.

It is not. **Thirteen DS divisions carry different codes in the two files**, because COD is
valid_on 2022-08-16 and the census is 2024, and Sri Lanka renumbered. The census's `2309` is
Walapane; COD's `LK2309` is Nildandahinna. Eravur Pattu and Eravur Town are swapped. So the
pcode join does not merely MISS those units, it silently pairs each of them with a polygon
somewhere else in the same district — **762,824 people placed in the wrong valley**, with no
symptom except a slightly low match rate that the vintage gap already explains.

The fix is to stop trusting the middle of the code. **Align DS divisions by NAME within each
district first, then match GN codes only within an aligned pair.** Names survive
renumbering; codes do not.

Which needs a transliteration-tolerant comparison, because DCS and COD romanise Sinhala and
Tamil differently and disagree on 81 of 340 DS names — Mathugama/Matugama, Thumpane/Tumpane,
Dickwella/Dikwella, Vadamaradchi/Vadamaradchchi. `fold()` below collapses the aspirates,
w/v, ee/i, oo/u and doubled letters. Four pairs still need `DS_ALIAS`, by hand, and one of
those four is not a spelling difference at all — see the note there.

WHAT IS LEFT AFTER ALL THAT: 24 census GN divisions, 42,843 people, 0.20%, that exist in the
2024 census and in no 2022 polygon. They are placed in their DS DIVISION instead, which is
coarser for those alone and is marked `derived` in countries.py so the reader is not told a
neighbourhood we do not have. The 85 polygons with no census row are dropped.

Usage:
    python sources/lk_geo.py --fetch    118 MB shapefile bundle from HDX
    python sources/lk_geo.py
"""

import csv
import difflib
import os
import re
import sys
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "lk")
ZIP = os.path.join(RAW, "lka_admin_boundaries.shp.zip")
OUT_DIR = os.path.join(ROOT, "data", "geo", "lk")
OUT = os.path.join(OUT_DIR, "lk_gnd.gpkg")
LOOKUP = os.path.join(OUT_DIR, "lk_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "lk.csv")

URL = ("https://data.humdata.org/dataset/0bedcaf3-88cd-4591-b9d5-5d3220e26abf/resource/"
       "39b80702-0c9c-4525-8490-ba4b13080459/download/lka_admin_boundaries.shp.zip")

EXPECTED_GND = 14_003
EXPECTED_DSD = 340
COD_ADMIN4 = 14_043
COD_ADMIN3 = 339

# Four census DS divisions `fold()` cannot reach. Three are spelling; the fourth is real.
DS_ALIAS = {
    "2130": "LK2130",   # 'Kandy Four Gravets & Gangawata Korale' -> 'Gangawata Korale'
    "2224": "LK2224",   # 'Laggala-Pallegama' -> 'Laggala'
    "5315": "LK5315",   # 'Trincomalee Town and Gravets' -> 'Town & Gravets'
    # NOT a spelling difference. The census splits Kalmunai into 'Kalmunai' (5224) and
    # 'Kalmunai North Sub' (5221) — the Tamil division — where COD has one polygon. Both
    # census DS divisions therefore point at LK5221 and their GN divisions are matched
    # against its polygons as one pool. This mapping is deliberately NOT injective.
    "5221": "LK5221",
    "5224": "LK5221",
}

FUZZY_CUTOFF = 0.72


def fetch():
    import requests
    import urllib3
    import zipfile
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(ZIP) and os.path.getsize(ZIP) > 100_000_000:
        print("already have", ZIP)
        return
    print("GET", URL)
    with requests.get(URL, timeout=1800, verify=False, stream=True,
                      headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        with open(ZIP, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    # §5a: assert type and size, not status.
    if not zipfile.is_zipfile(ZIP):
        raise SystemExit(f"{ZIP} is not a zip")
    with zipfile.ZipFile(ZIP) as z:
        names = z.namelist()
    if "lka_admin4.shp" not in names:
        raise SystemExit(f"no lka_admin4.shp in the bundle: {names[:10]}")
    print(f"  {os.path.getsize(ZIP):,} bytes -> {ZIP}")


def fold(s):
    """Fold two romanisations of the same Sinhala or Tamil name onto one key.

    Deliberately aggressive — it drops `h` entirely and collapses doubled letters — which is
    safe ONLY because it is applied within one district (at most ~30 DS names) or within one
    DS division (at most ~120 GN names), never across the country. Every use below also
    requires the match to be 1:1, so a collision is reported rather than resolved.
    """
    s = unicodedata.normalize("NFKD", str(s)).lower()
    s = re.sub(r"\(.*?\)", " ", s)              # 'Island North (Kayts)' -> 'Island North'
    s = re.sub(r"[^a-z0-9]+", "", s)
    for a, b in (("th", "t"), ("dh", "d"), ("bh", "b"), ("gh", "g"), ("kh", "k"),
                 ("ph", "p"), ("ch", "c"), ("sh", "s"), ("w", "v"), ("ee", "i"),
                 ("oo", "u"), ("aa", "a")):
        s = s.replace(a, b)
    s = re.sub(r"(.)\1+", r"\1", s)
    return s.replace("h", "")


def pair(a_items, b_items, cutoff=FUZZY_CUTOFF):
    """Pair (key, name) lists 1:1 — exact fold first, then best-ratio fuzzy above cutoff.

    Returns (mapping, unpaired_a_keys, unpaired_b_keys).
    """
    out, used, bmap = {}, set(), {}
    for k, n in b_items:
        bmap.setdefault(fold(n), []).append(k)
    left = []
    for k, n in a_items:
        cand = [x for x in bmap.get(fold(n), []) if x not in used]
        if len(cand) == 1:
            out[k] = cand[0]
            used.add(cand[0])
        else:
            left.append((k, n))
    rem = [(k, n) for k, n in b_items if k not in used]
    for k, n in left:
        best, score = None, 0.0
        for bk, bn in rem:
            s = difflib.SequenceMatcher(None, fold(n), fold(bn)).ratio()
            if s > score:
                best, score = bk, s
        if best is not None and score >= cutoff:
            out[k] = best
            rem = [(x, y) for x, y in rem if x != best]
    return out, [k for k, _ in a_items if k not in out], [k for k, _ in rem]


def main():
    import geopandas as gpd
    import pandas as pd

    if not os.path.exists(ZIP):
        raise SystemExit(f"missing {ZIP} -- run with --fetch first")
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/lk.py first")

    # ---- census side: one row per GN division, out of the normalised file ----
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False)
    tot = df[df["source_category"] == "Total"].copy()
    tot["dsd_id"] = tot["geo_id"].str[:4]
    tot["dsd_name"] = tot["note"].str.extract(r"dsd=([^;]*)")[0].str.strip()
    tot["d_id"] = tot["geo_id"].str[:2]
    tot["d_name"] = tot["note"].str.extract(r"district=([^;]*)")[0].str.strip()
    if len(tot) != EXPECTED_GND:
        raise SystemExit(f"{len(tot)} census GN divisions, expected {EXPECTED_GND}")
    print(f"census GN divisions: {len(tot):,} in {tot['dsd_id'].nunique()} DS divisions")

    # ---- geo side ----
    print("reading lka_admin4 from the bundle (112 MB shapefile, ~1 min)…")
    cod = gpd.read_file(f"zip://{ZIP}!lka_admin4.shp")
    print(f"COD admin4 polygons: {len(cod):,}  crs={cod.crs}")
    if len(cod) != COD_ADMIN4:
        raise SystemExit(f"expected {COD_ADMIN4} COD polygons")
    cod["adm4_pcode"] = cod["adm4_pcode"].astype(str).str.strip()
    cod["adm3_pcode"] = cod["adm3_pcode"].astype(str).str.strip()
    cod["d_id"] = cod["adm2_pcode"].astype(str).str[2:4]
    if cod["adm4_pcode"].duplicated().any():
        raise SystemExit("duplicate adm4_pcode in COD")

    # ---- stage 1: DS divisions, by NAME within district (never by code) ----
    cen_ds = (tot.groupby("dsd_id")
                 .agg(d_id=("d_id", "first"), name=("dsd_name", "first"))
                 .reset_index())
    cod_ds = (cod.drop_duplicates("adm3_pcode")[["adm3_pcode", "adm3_name", "d_id"]]
                 .reset_index(drop=True))
    if len(cod_ds) != COD_ADMIN3:
        raise SystemExit(f"expected {COD_ADMIN3} COD DS divisions, got {len(cod_ds)}")

    ds_map, ds_un_cen, ds_un_cod = {}, [], []
    for d in sorted(set(cen_ds["d_id"])):
        a = [(r.dsd_id, r.name) for r in
             cen_ds[(cen_ds.d_id == d) & (~cen_ds.dsd_id.isin(DS_ALIAS))].itertuples()]
        taken = {v for k, v in DS_ALIAS.items() if k.startswith(d)}
        b = [(r.adm3_pcode, r.adm3_name) for r in cod_ds[cod_ds.d_id == d].itertuples()
             if r.adm3_pcode not in taken]
        m, au, bu = pair(a, b)
        ds_map.update(m)
        ds_un_cen += au
        ds_un_cod += bu
    ds_map.update(DS_ALIAS)

    print(f"\n  stage 1 — DS divisions aligned by name within district:")
    print(f"    paired                     {len(ds_map):>5} of {EXPECTED_DSD}"
          f"   ({len(DS_ALIAS)} by hand alias)")
    print(f"    census DS with no polygon  {len(ds_un_cen):>5}  {ds_un_cen}")
    print(f"    COD DS with no census row  {len(ds_un_cod):>5}  {ds_un_cod}")
    if len(ds_map) != EXPECTED_DSD or ds_un_cen:
        raise SystemExit("DS alignment FAILED -- every census DS division must be placed")

    # The finding this file exists for: how many aligned pairs disagree about the CODE.
    shifted = {k: v for k, v in ds_map.items() if f"LK{k}" != v}
    pop_shift = int(tot[tot["dsd_id"].isin(shifted)]["count"].sum())
    print(f"\n    DS divisions whose CODE DIFFERS between census and COD: {len(shifted)}")
    for k, v in sorted(shifted.items()):
        nm = cen_ds.loc[cen_ds.dsd_id == k, "name"].iloc[0]
        print(f"      census {k} -> COD {v}   {nm}")
    print(f"    {pop_shift:,} people ({100.0 * pop_shift / tot['count'].sum():.2f}%) live "
          "in them.\n      A pcode join places every one of them in the wrong DS division, "
          "silently.")

    # ---- stage 2: GN divisions, by code then name, WITHIN an aligned DS pair ----
    cod_by_ds = {k: v for k, v in cod.groupby("adm3_pcode")}
    pooled = {}
    for dsd, adm3 in ds_map.items():
        pooled.setdefault(adm3, []).append(dsd)

    gn_map, gn_un_cen, by_code, by_name = {}, [], 0, 0
    for adm3, dsds in pooled.items():
        cand = cod_by_ds.get(adm3)
        rows = tot[tot["dsd_id"].isin(dsds)]
        if cand is None:
            gn_un_cen += list(rows["geo_id"])
            continue
        # WHERE TWO CENSUS DS DIVISIONS SHARE ONE COD POLYGON SET, THE CODES ARE NOT
        # COMPARABLE AND THE CODE STAGE MUST BE SKIPPED. Kalmunai is the case: COD holds it
        # as one DS of 58 GN divisions numbered 005..290, while the census splits it into
        # 'Kalmunai' and 'Kalmunai North Sub' of 29 each, EACH RESTARTING AT 005. Matching
        # on the code gives all 29 low numbers to whichever half is seen first and orphans
        # the other half entirely — 52,798 people, the whole Muslim division. Names are
        # unique across the pooled 58 (Kalmunai 01, Kalmunaikudi 01, Chenaikudiyiruppu 01),
        # so name matching is both available and the only correct option here.
        bcode = {}
        if len(dsds) == 1:
            for r in cand.itertuples():
                bcode[str(r.adm4_pcode)[6:]] = r.adm4_pcode
        used, left = set(), []
        for r in rows.itertuples():
            t = bcode.get(r.geo_id[4:])
            if t is not None and t not in used:
                gn_map[r.geo_id] = t
                used.add(t)
                by_code += 1
            else:
                left.append((r.geo_id, r.geo_name))
        rem = [(str(r.adm4_pcode), r.adm4_name) for r in cand.itertuples()
               if r.adm4_pcode not in used]
        if left and rem:
            m, au, _ = pair(left, rem)
            gn_map.update(m)
            by_name += len(m)
            gn_un_cen += au
        else:
            gn_un_cen += [k for k, _ in left]

    orphan = sorted(set(gn_un_cen))
    orph_pop = int(tot[tot["geo_id"].isin(orphan)]["count"].sum())
    print(f"\n  stage 2 — GN divisions, within an aligned DS pair:")
    print(f"    matched by code            {by_code:>6,}")
    print(f"    matched by name            {by_name:>6,}")
    print(f"    total matched              {len(gn_map):>6,} of {EXPECTED_GND:,}"
          f"  ({100.0 * len(gn_map) / EXPECTED_GND:.2f}%)")
    print(f"    census GN with no polygon  {len(orphan):>6,}  "
          f"({orph_pop:,} people, {100.0 * orph_pop / tot['count'].sum():.2f}%)")
    print(f"    COD polygons unused        {len(cod) - len(set(gn_map.values())):>6,}")

    # ---- independent check: does the join keep districts intact? ----
    # Name agreement is not independent of a name join, so this asks something else: every
    # census GN division must land on a polygon in the SAME DISTRICT. A scrambled join
    # crosses district lines; a correct one cannot.
    cod_d = dict(zip(cod["adm4_pcode"], cod["d_id"]))
    crossed = [(g, p) for g, p in gn_map.items() if cod_d.get(p) != g[:2]]
    print(f"\n  independent check — GN divisions landing outside their own district: "
          f"{len(crossed)}")
    if crossed:
        for g, p in crossed[:10]:
            print(f"      census {g} (district {g[:2]}) -> {p} (district {cod_d.get(p)})")
        raise SystemExit("join FAILED -- the district must be preserved")

    # And a second one that IS independent of the names: the census's own GN-division
    # population against the polygon's area. Not a tight test, but a join that pairs a
    # Colombo ward with a Vavuniya jungle GND shows up as an absurd density.
    area = dict(zip(cod["adm4_pcode"], cod["area_sqkm"]))
    dens = []
    for r in tot.itertuples():
        p = gn_map.get(r.geo_id)
        a = area.get(p)
        if p and a and a > 0:
            dens.append(r.count / a)
    dens.sort()
    med = dens[len(dens) // 2]
    print(f"    density on the {len(dens):,} matched units: median {med:,.0f} people/km2, "
          f"p1 {dens[len(dens)//100]:,.0f}, p99 {dens[-len(dens)//100]:,.0f}")

    # ---- names, reported not enforced ----
    cod_name = dict(zip(cod["adm4_pcode"], cod["adm4_name"]))
    agree = sum(1 for r in tot.itertuples()
                if r.geo_id in gn_map
                and fold(r.geo_name) == fold(cod_name.get(gn_map[r.geo_id]) or ""))
    print(f"    GN names agree after folding on {agree:,} of {len(gn_map):,} "
          f"({100.0 * agree / len(gn_map):.1f}%) — the rest are romanisation and "
          "renaming,\n      which is what stage 1 exists to tolerate.")

    # ---- build the output layer ----
    # One row per CENSUS GN division, carrying the census id, so scatter.py never sees a
    # COD pcode and cannot accidentally join on one.
    geom = cod.set_index("adm4_pcode")["geometry"]

    # AN ORPHAN'S FALLBACK IS THE UNMATCHED REMAINDER OF ITS DS DIVISION, NOT THE WHOLE
    # DIVISION. Every polygon that did match belongs to some other GN division, so whatever
    # is left is where the unmatched people must be — a tighter and strictly more honest
    # area than the DS division, for free.
    #
    # Kalmunai is the case that forces it and shows the size of the difference. COD names
    # only the 29 polygons of the Tamil division and leaves the Muslim division's 29
    # completely unnamed, so no name can reach them; but they are precisely the remainder,
    # and Kalmunai is where the Muslim/Tamil boundary is the sharpest religious line in the
    # country. Falling back to the whole DS division would smear 52,798 people across both
    # halves and erase exactly the thing worth seeing.
    used_poly = set(gn_map.values())
    rest = cod[~cod["adm4_pcode"].isin(used_poly)]
    rest_geom = rest.dissolve(by="adm3_pcode")["geometry"] if len(rest) else {}
    ds_geom = cod.dissolve(by="adm3_pcode")["geometry"]

    recs, tight, wide = [], 0, 0
    for r in tot.itertuples():
        p = gn_map.get(r.geo_id)
        if p is not None:
            recs.append({"gnd": r.geo_id, "name": r.geo_name, "level": "gnd",
                         "geometry": geom.loc[p]})
            continue
        adm3 = ds_map[r.geo_id[:4]]
        if len(rest) and adm3 in rest_geom.index:
            recs.append({"gnd": r.geo_id, "name": r.geo_name, "level": "dsd_rest",
                         "geometry": rest_geom.loc[adm3]})
            tight += 1
        else:
            recs.append({"gnd": r.geo_id, "name": r.geo_name, "level": "dsd",
                         "geometry": ds_geom.loc[adm3]})
            wide += 1
    print(f"\n  the {len(orphan)} orphans: {tight} placed in the unmatched remainder of "
          f"their DS division, {wide} in the whole division")
    out = gpd.GeoDataFrame(recs, geometry="geometry", crs=cod.crs)
    if len(out) != EXPECTED_GND:
        raise SystemExit(f"{len(out)} output rows, expected {EXPECTED_GND}")
    bad = out.geometry.isna() | out.geometry.is_empty
    if bad.any():
        raise SystemExit(f"{int(bad.sum())} empty geometries")

    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_file(OUT, layer="gnd", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} polygons, "
          f"{int((out['level'] == 'dsd').sum())} of them DS-level fallbacks)")

    with open(LOOKUP, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_id", "adm4_pcode", "adm3_pcode", "level"])
        for r in tot.itertuples():
            p = gn_map.get(r.geo_id)
            w.writerow([r.geo_id, p or "", ds_map[r.geo_id[:4]],
                        "gnd" if p else "dsd"])
    print(f"wrote {LOOKUP} ({len(tot):,} rows)")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    main()
