"""Uganda — the 56 districts of 2002, rebuilt from the 135 of 2020, and the rebuild proved.

Writes data/geo/ug/ug_districts.gpkg and data/geo/ug/ug_lookup.csv.

**THERE IS NO 2002-VINTAGE BOUNDARY FILE FOR UGANDA and spec §8.1 still applies**: the
counts are published on the 56 districts of 2002, so that is the geography that must be
drawn. Uganda has split districts continuously since — 56 in 2002, 112 at the 2014
census, 135 in COD-AB's 2020 edition, 146 by the 2024 census — and HDX carries only the
current one. COD-AB is 2020, geoBoundaries is current, and neither ships an older
vintage. UBOS's own GIS portal, `ubosgis.ubos.org`, answers 502 to every path.

**SO THE 2002 DISTRICTS ARE THE UNION OF THE 2020 DISTRICTS INSIDE THEM, and which is
inside which comes out of the census itself.** Table C1 of the 2002 annex series
(`Total Population by Sub-county`) prints the whole 2002 hierarchy — district, then
county, then sub-county — for 995 named places. Every one of those names is a vote for
the 2002 district it is printed under. COD-AB's county layer (208 units) and sub-county
layer (1,520) are matched against those names, and each of the 135 current districts
takes the 2002 district its places vote for.

**AND THE RESULT IS PROVED ON POPULATION, NOT EYEBALLED.** Table B1 of the same annex
series gives the 1991 census redistributed onto the 56 districts of 2002. Table A3 of the
2014 census Main Report gives the 1991 census redistributed onto the 112 districts of
2014 — a different publication, twelve years later, from a different census. Both national
totals are 16,671,705, so they are the same universe. Grouping A3's 112 figures by this
concordance reproduces **all 56 of B1's figures exactly**, on 56 distinct values. A group
that swallowed a district it should not have, or dropped one it should have kept, would
fail; [[reference_name_join_wrong_neighbour]] has nowhere to hide in an exact equality on
distinct values.

**WHAT THE PROOF ALSO FOUND.** Run on the 2002 column instead of the 1991 one, 55 of the
56 still agree to the person and **Kotido does not**: 591,889 in the 2002 tabulations
against 377,102 in the 2014 report, a cut of 214,787 people, 36% of the district. Kotido's
1991 figure agrees, so it is a revision of one count and not a boundary artefact.
countries.py decides what to do about it; `sources/ug.md` has the record.

Two things the vote had to survive, both of them [[reference_name_join_wrong_neighbour]]
in miniature:

  * **A sub-county can carry a distant district's name.** 22 of the 135 have one stray
    vote each — Agago's places vote 11 for Pader and once for Nakasongola, because
    Nakasongola district contains a sub-county called Wabinyonyi that Agago also has a
    place matching. A plurality settles them and the population proof is what says the
    plurality was right.
  * **C1 spells Ssembabule with two esses and prints a sub-county called `Sembabule
    T.C.` at the district indent.** District headers are therefore matched by NAME
    against the 56 Table B7 prints, first occurrence wins, rather than by indent alone.
    Trusting the indent attributes a whole district's places to its predecessor: on the
    first attempt Bugiri's header was missed and all of Bugiri's sub-counties voted for
    Wakiso, which the 1991 proof caught.

Usage:
    python sources/ug_geo.py --fetch    one ~33 MB shapefile bundle from HDX
    python sources/ug_geo.py            rebuild from data/raw/ug/
"""

import importlib.util
import os
import re
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")     # [[reference_scipy_eats_all_cores]]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ug")
GEO = os.path.join(ROOT, "data", "geo", "ug")
DISTRICTS = os.path.join(GEO, "ug_districts.gpkg")
LOOKUP = os.path.join(GEO, "ug_lookup.csv")

COD_URL = ("https://data.humdata.org/dataset/6d6d1495-196b-49d0-86b9-dc9022cde8e7/"
           "resource/7984457f-d12e-438f-8d6d-7e1a75694864/download/"
           "uga_admin_boundaries.shp.zip")
COD_ZIP = "uga_admin_boundaries.shp.zip"

EXPECTED_2002 = 56
EXPECTED_2020 = 135
EXPECTED_COUNTIES = 208
EXPECTED_SUBCOUNTIES = 1520
POP_1991 = 16_671_705

NUM = re.compile(r"^[\d,]+$")
DEC = re.compile(r"^\d+\.\d$")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")

# The 2014 Main Report prints Kyankwanzi as `Kyakwanzi` in Table A3 and nowhere else.
A3_ALIAS = {"kyakwanzi": "kyankwanzi"}


def _ug():
    spec = importlib.util.spec_from_file_location("ug_src", os.path.join(HERE, "ug.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def norm(s):
    """Uganda's place names carry a lot of administrative furniture and one spelling
    habit (Ss- for S-). Strip both before comparing."""
    s = s.lower().replace("’", "'").replace("`", "'")
    s = re.sub(r"\b(t\.?\s*c\.?|town council|municipality|municipal council|county|"
               r"city council|city|division|island|islands)\b", " ", s)
    s = re.sub(r"[^a-z]", "", s)
    if s.startswith("ss"):
        s = s[1:]
    return s


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, COD_ZIP)
    if os.path.exists(path) and os.path.getsize(path) > 20_000_000:
        print("already have", path)
        return
    print("GET", COD_URL)
    r = requests.get(COD_URL, timeout=1800, stream=True, headers={"User-Agent": UA})
    r.raise_for_status()
    with open(path + ".part", "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    os.replace(path + ".part", path)
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        if fh.read(2) != b"PK":
            raise SystemExit(f"{path} is not a zip")
    print(f"  {size:,} bytes")


def c1_votes(path, names):
    """Table C1 -> {normalised place name: {2002 district}}. District headers are found
    by name against `names`, first occurrence winning, never by indent alone."""
    import fitz

    doc = fitz.open(path)
    want = [norm(n) for n in names]
    seen, cur = [], None
    votes = {}
    for pi in range(doc.page_count):
        lines = {}
        for x0, y0, x1, y1, w, *_ in doc[pi].get_text("words"):
            lines.setdefault(round(y0 / 3.0), []).append((x0, w))
        for key in sorted(lines):
            ws = sorted(lines[key])
            namew = [(x, t) for x, t in ws if not (NUM.match(t) or DEC.match(t))]
            if not namew:
                continue
            name = " ".join(t for _, t in namew).strip()
            nn = norm(name)
            if namew[0][0] < 95 and nn in want and names[want.index(nn)] not in seen:
                cur = names[want.index(nn)]
                seen.append(cur)
                continue
            if cur and nn and nn != norm(cur):
                votes.setdefault(nn, set()).add(cur)
    if len(seen) != len(names):
        missing = [n for n in names if n not in seen]
        raise SystemExit(f"Table C1: found {len(seen)} of {len(names)} district headers, "
                         f"missing {missing}")
    return votes


def parse_a3(path):
    """Table A3 of the 2014 Main Report -> [(2014 district, [1991 m,f,t, 2002 m,f,t,
    2014 m,f,t])]."""
    import fitz

    doc = fitz.open(path)
    out = []
    for pi in range(doc.page_count):
        if "Table A3" not in doc[pi].get_text():
            continue
        lines = {}
        for x0, y0, x1, y1, w, *_ in doc[pi].get_text("words"):
            lines.setdefault(round(y0 / 3.0), []).append((x0, w))
        for key in sorted(lines):
            ws = sorted(lines[key])
            nums = [t for _, t in ws if NUM.match(t)]
            name = " ".join(t for _, t in ws if not NUM.match(t)).strip()
            if len(nums) == 9 and name and name.lower() != "total":
                out.append((name, [int(t.replace(",", "")) for t in nums]))
    return out


def parse_b1(path):
    """Table B1 -> {2002 district: [1980, 1991, 2002]}."""
    import fitz

    doc = fitz.open(path)
    out = {}
    for pi in range(doc.page_count):
        lines = {}
        for x0, y0, x1, y1, w, *_ in doc[pi].get_text("words"):
            lines.setdefault(round(y0 / 3.0), []).append((x0, w))
        for key in sorted(lines):
            ws = sorted(lines[key])
            nums = [t for _, t in ws if NUM.match(t)]
            name = " ".join(t for _, t in ws if not NUM.match(t)).strip()
            if len(nums) != 3 or not name:
                continue
            if name in ("District", "UGANDA") or "Total" in name or "Table" in name:
                continue
            out[norm(name)] = (name, [int(t.replace(",", "")) for t in nums])
    return out


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    zip_path = os.path.join(RAW, COD_ZIP)
    for p in (zip_path, os.path.join(RAW, "centableC1.pdf"),
              os.path.join(RAW, "centableB1.pdf"),
              os.path.join(RAW, "2014_NPHC_Main_Report.pdf")):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} — run sources/ug.py --fetch and "
                             f"sources/ug_geo.py --fetch first")

    ug = _ug()
    districts, _ = ug.parse_b7(os.path.join(RAW, "centableB7.pdf"))
    names = [n for n, _, _ in districts]
    if len(names) != EXPECTED_2002:
        raise SystemExit(f"Table B7 gave {len(names)} districts, expected {EXPECTED_2002}")
    gid = {n: "UG%02d" % (i + 1) for i, n in enumerate(names)}

    votes = c1_votes(os.path.join(RAW, "centableC1.pdf"), names)
    amb = sum(1 for v in votes.values() if len(v) > 1)
    print(f"Table C1: {len(votes)} place names under the {EXPECTED_2002} districts of 2002 "
          f"({amb} used in more than one district and therefore ignored)")

    src = "zip://" + zip_path.replace(os.sep, "/")
    a2 = gpd.read_file(src, layer="uga_admin2")
    a3g = gpd.read_file(src, layer="uga_admin3")
    a4g = gpd.read_file(src, layer="uga_admin4")
    for got, wanted, what in ((len(a2), EXPECTED_2020, "districts"),
                              (len(a3g), EXPECTED_COUNTIES, "counties"),
                              (len(a4g), EXPECTED_SUBCOUNTIES, "sub-counties")):
        if got != wanted:
            raise SystemExit(f"COD-AB has {got} {what}, expected {wanted} — the vintage "
                             "has moved and the concordance must be re-proved")
    print(f"COD-AB 2020-08-24: {len(a2)} districts, {len(a3g)} counties, "
          f"{len(a4g)} sub-counties")

    tally = {}
    for layer, col in ((a3g, "adm3_name"), (a4g, "adm4_name")):
        for _, r in layer.iterrows():
            tgt = votes.get(norm(r[col]))
            if tgt and len(tgt) == 1:
                d = tally.setdefault(r["adm2_name"], {})
                d[next(iter(tgt))] = d.get(next(iter(tgt)), 0) + 1
    # A current district that still carries a 2002 district's name IS that district; that
    # is worth more than any number of sub-county votes and is weighted to say so.
    for d in a2["adm2_name"]:
        if norm(d) in [norm(n) for n in names]:
            nm = names[[norm(n) for n in names].index(norm(d))]
            t = tally.setdefault(d, {})
            t[nm] = t.get(nm, 0) + 1000

    no_vote = sorted(set(a2["adm2_name"]) - set(tally))
    if no_vote:
        raise SystemExit(f"current districts no 2002 place name reaches: {no_vote}")
    conc = {d: max(v.items(), key=lambda kv: kv[1])[0] for d, v in tally.items()}
    split = [(d, sorted(v.items(), key=lambda kv: -kv[1]))
             for d, v in tally.items() if len(v) > 1]
    uncovered = sorted(set(names) - set(conc.values()))
    if uncovered:
        raise SystemExit(f"2002 districts nothing maps to: {uncovered}")
    print(f"  all {len(conc)} current districts placed; all {EXPECTED_2002} of 2002 "
          f"covered; {len(split)} settled on a plurality")

    # --- the proof ---------------------------------------------------------------
    b1 = parse_b1(os.path.join(RAW, "centableB1.pdf"))
    if len(b1) != EXPECTED_2002:
        raise SystemExit(f"Table B1 gave {len(b1)} districts, expected {EXPECTED_2002}")
    if sum(v[1][1] for v in b1.values()) != POP_1991:
        raise SystemExit(f"Table B1's 1991 column sums to "
                         f"{sum(v[1][1] for v in b1.values()):,}, expected {POP_1991:,}")
    a3 = parse_a3(os.path.join(RAW, "2014_NPHC_Main_Report.pdf"))
    if sum(v[2] for _, v in a3) != POP_1991:
        raise SystemExit(f"Table A3's 1991 column sums to {sum(v[2] for _, v in a3):,}, "
                         f"expected {POP_1991:,} — not the same universe as B1")
    by_norm = {norm(k): k for k in conc}
    agg91, agg02 = {}, {}
    for name, v in a3:
        key = by_norm.get(A3_ALIAS.get(norm(name), norm(name)))
        if key is None:
            raise SystemExit(f"Table A3's `{name}` matches no COD-AB district")
        d = conc[key]
        agg91[d] = agg91.get(d, 0) + v[2]
        agg02[d] = agg02.get(d, 0) + v[5]
    wrong = [(d, agg91.get(d), b1[norm(d)][1][1]) for d in names
             if agg91.get(d) != b1[norm(d)][1][1]]
    if wrong:
        for d, got, want in wrong:
            print(f"    {d}: grouped 1991 {got:,} against Table B1's {want:,}")
        raise SystemExit(f"{len(wrong)} of {EXPECTED_2002} districts fail the 1991 proof")
    distinct = len({v[1][1] for v in b1.values()})
    print(f"  PROVED: grouping Table A3's {len(a3)} districts by this concordance "
          f"reproduces all\n  {EXPECTED_2002} of Table B1's 1991 figures exactly, on "
          f"{distinct} distinct values")

    off = [(d, agg02[d], b1[norm(d)][1][2]) for d in names
           if agg02[d] != b1[norm(d)][1][2]]
    print(f"  on the 2002 column instead, {EXPECTED_2002 - len(off)} of {EXPECTED_2002} "
          f"still agree to the person:")
    for d, got, want in off:
        print(f"    {d}: the 2014 report says {got:,} where the 2002 tabulations say "
              f"{want:,} ({100.0 * (got - want) / want:+.1f}%)")

    # --- dissolve ----------------------------------------------------------------
    a2 = a2.copy()
    a2["d2002"] = a2["adm2_name"].map(conc)
    out = a2.dissolve(by="d2002", as_index=False)[["d2002", "geometry"]]
    out["unit"] = out["d2002"].map(gid)
    out["name"] = out["d2002"]
    out = out[["unit", "name", "geometry"]].sort_values("unit").reset_index(drop=True)
    if len(out) != EXPECTED_2002:
        raise SystemExit(f"dissolve produced {len(out)} polygons, expected {EXPECTED_2002}")
    if out["geometry"].isna().any() or not out["geometry"].is_valid.all():
        out["geometry"] = out["geometry"].buffer(0)
    km2 = out.to_crs(3857).area.sum() / 1e6
    print(f"\n  dissolved {len(a2)} 2020 districts into {len(out)} 2002 districts "
          f"({km2:,.0f} km2 in web mercator, area not meaningful at this latitude)")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(DISTRICTS, layer="districts", driver="GPKG")
    # `geo_id` and `unit` are the same string here — data/normalized/ug.csv is keyed on
    # Table B7's own print order, which is what gid built — but both columns are written
    # because countries.py's other entries all join through a lookup and one that quietly
    # lacked the column would be a surprise.
    lut = pd.DataFrame({"geo_id": out["unit"], "unit": out["unit"], "name": out["name"]})
    lut["members"] = lut["name"].map(
        lambda n: "|".join(sorted(d for d, v in conc.items() if v == n)))
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {DISTRICTS} ({len(out)} districts)")
    print(f"wrote {LOOKUP}")

    per = lut["members"].str.count(r"\|").add(1)
    print(f"\n  2002 districts by how many of 2020's they hold: "
          f"{per.min()} to {per.max()}, median {int(per.median())}")
    big = lut.assign(n=per).sort_values("n", ascending=False).head(4)
    for _, r in big.iterrows():
        print(f"    {r['name']:<14} {r['n']:>2} -> {r['members']}")


if __name__ == "__main__":
    main()
