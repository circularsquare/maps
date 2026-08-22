"""prep_fua.py -- build fua2025.json: a metropolitan-level replacement series for the
WUP urban centres that sit inside a functional urban area.

WHY. WUP 2025 measures the Degree-of-Urbanisation "urban centre" -- contiguous cells above
1,500/km2. That rule works where cities are dense and fails where they are not, and the
United States is the worst case on the map: American suburbia sits nearer 1,000/km2, so it
fails the density test outright and the metro shatters into disconnected cores. WUP 2025 has
only 357 US centres, 25 of them >=1M, and it puts Chicago at 3.74M (metro 9.5M), Dallas at
1.52M (metro 8.1M), Boston at 1.64M (metro 4.9M). Grafting that onto populstat's metro-wide
American figures produced a systematically DOWNWARD seam: for US cities peaking >=100k the
median switch step is 0.80x and a quarter of them lose more than half their population in a
single year. Worldwide the same numbers are 1.08x and 8.5%.

The fix is not a different population raster, it is a different DELINEATION. GHS-FUA (a.k.a.
eFUA, JRC + OECD, R2019A) draws 9,031 functional urban areas worldwide -- an urban centre
plus the commuting zone that sends >=15% of its residents into it, which is the same concept
as a US metropolitan statistical area. Its 2015 populations land within a few percent of the
official MSAs where those exist (New York 19.5M, Philadelphia 6.11M, Atlanta 5.59M, Chicago
8.80M, Seattle 3.87M).

WHAT THIS PRODUCES. eFUA ships one epoch, FUA_p_2015, so a series has to be constructed. For
each FUA we take the WUP centres whose population-weighted centroid falls inside the polygon,
sum their annual series, and scale the whole sum by

    k = FUA_p_2015 / (that sum in 2015)

so the curve passes through the FUA's own measured 2015 population. k is the commuting-zone
uplift: what the low-density ring adds on top of the dense cores. Computing the ratio against
the summed WUP value rather than against eFUA's own UC_p_2015 keeps everything in WUP's units
-- eFUA's populations come from GHS-POP R2019A and WUP's are UN-adjusted, and they differ by a
few percent for reasons that have nothing to do with the commuting zone.

The result is one definition across the entire 1975-2025 range. It is not spliced onto the WUP
series and never mixes with it: build.py swaps the whole population dict of the FUA's principal
centre, so a city either runs on FUA figures throughout or on WUP figures throughout.

WHO GETS IT. Only the FUA's PRINCIPAL centre -- the largest member by peak population. The
other members keep their own urban-centre series. Anything else double-counts: New York's FUA
contains 6 WUP centres, London's 12, Cairo's 17, and handing 19.5M to each of New York's six
would draw the same 19.5M people six times over. So Dallas gets the 7.08M Dallas-Fort Worth
FUA and Fort Worth keeps its own centre, which is the same choice a metro table makes.

MEMBER CHURN. A WUP centre enters the dataset the year it crosses 50k, so a naive year-by-year
sum steps up whenever a member appears. Each member's series is therefore clamped -- held flat
at its first value backwards and its last value forwards -- across the union range, so the
member set is constant and the only movement in the sum is real. The error is bounded by the
50k threshold and lands on cities of millions.

Inputs:  data/efua.gpkg              GHS_FUA_UCDB2015_GLOBE_R2019A (EC reuse licence)
         data/stadester/wup2025.json output of prep_wup.py
Output:  data/stadester/fua2025.json keyed by the principal's WUP City_Code

Source: https://human-settlement.emergency.copernicus.eu/ghs_fua.php
"""
import json, os, sys
from collections import defaultdict

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FUA_GPKG = "data/efua.gpkg"
WUP      = "data/stadester/wup2025.json"
OUT      = "data/stadester/fua2025.json"

REF_YEAR = 2015        # the epoch eFUA measures, and so the year k is calibrated on
COV_MIN  = 0.22        # minimum sum(WUP members) / eFUA's own UC_p_2015 -- see build_fua()
COV_NONAME = 0.6       # ...raised to this when no member is named after the FUA at all
NAME_MIN_FRAC = 0.5    # a name match only outranks size down to this fraction of the largest
                       # member -- see pick_principal()


def load_wup():
    with open(WUP, encoding="utf-8") as f:
        g = json.load(f)
    rows = []
    for code, c in g.items():
        co = c.get("coords")
        pop = {int(y): v for y, v in (c.get("population") or {}).items() if v and v > 0}
        if not co or len(co) != 2 or not pop:
            continue
        rows.append({"code": code, "name": c.get("name", ""), "iso3": c.get("iso3", ""),
                     "lat": co[0], "lon": co[1], "pop": pop, "peak": max(pop.values())})
    return rows


def norm(s):
    import unicodedata
    return "".join(ch for ch in unicodedata.normalize("NFKD", s or "")
                   if not unicodedata.combining(ch)).lower().strip()


def pick_principal(codes, by_code, fua_name):
    """Which member carries the FUA's population. Returns (code, matched_by_name).

    Prefers the member eFUA named the FUA after, then the largest. The name preference is
    needed because DEGURBA fragments some cities so completely that the piece carrying the
    city's own name is not the biggest piece: Tangail's own centre is 444k against a fragment
    called Elenga at 460k. match_centre() joins a city record to a centre by NAME and distance,
    so the FUA has to end up on the centre the city itself will claim or the override never
    reaches it. Same reasoning as GRAFT_PRINCIPAL_WINS in build.py, arrived at from the other
    end.

    BOUNDED BY SIZE, because unbounded it hands a conurbation to a junior partner. eFUA calls
    the western Ruhr "Dortmund" (5.84M) and Dortmund's own centre is 753k inside it -- Essen's
    is 2.85M, nearly four times larger. On the name alone Dortmund took the whole Ruhr and the
    map drew a 6.3M dot on it, which is GRAFT_DENY's Essen problem rebuilt in a new place. So
    the named member has to be within NAME_MIN_FRAC of the largest to outrank it; below that
    the FUA is a conurbation of peers rather than one city's metro, and size decides. (For the
    Ruhr that hands it to Essen, which GRAFT_DENY already refuses -- so it stays unbuilt, which
    is the right answer.)"""
    fn = norm(fua_name)
    # eFUA writes conurbations as "Osaka [Kyoto]" and "Delhi [New Delhi]"; the lead name is the
    # one the FUA is really about, so match on it rather than on the whole string.
    if "[" in fn:
        fn = fn[:fn.index("[")].strip()
    if not codes:
        return None, False
    biggest = max(by_code[c]["peak"] for c in codes)
    exact, partial = [], []
    for c in codes:
        cn = norm(by_code[c]["name"])
        if not cn or not fn:
            continue
        if cn == fn:
            exact.append(c)
        elif min(len(cn), len(fn)) >= 5 and (cn in fn or fn in cn):
            partial.append(c)
    named = bool(exact or partial)
    for pool in (exact, partial):
        if not pool:
            continue
        best = max(pool, key=lambda c: by_code[c]["peak"])
        if by_code[best]["peak"] >= NAME_MIN_FRAC * biggest:
            return best, True
    return max(codes, key=lambda c: by_code[c]["peak"]), named


def clamped(pop, years):
    """Series held flat outside its own range, so a member that crosses WUP's 50k threshold
    mid-record does not step the FUA sum. See MEMBER CHURN above."""
    ys = sorted(pop)
    lo, hi = ys[0], ys[-1]
    out = {}
    for y in years:
        if y in pop:
            out[y] = pop[y]
        elif y < lo:
            out[y] = pop[lo]
        elif y > hi:
            out[y] = pop[hi]
        else:                                   # interior hole: log-linear between neighbours
            prev = max(v for v in ys if v < y)
            nxt = min(v for v in ys if v > y)
            f = (y - prev) / (nxt - prev)
            out[y] = pop[prev] + (pop[nxt] - pop[prev]) * f
    return out


def build_fua():
    fua = gpd.read_file(FUA_GPKG)
    print(f"eFUA polygons: {len(fua):,}")
    fua = fua.to_crs("EPSG:4326")

    wup = load_wup()
    print(f"WUP centres:   {len(wup):,}")
    pts = gpd.GeoDataFrame(
        pd.DataFrame([{"code": r["code"], "peak": r["peak"]} for r in wup]),
        geometry=[Point(r["lon"], r["lat"]) for r in wup], crs="EPSG:4326")

    joined = gpd.sjoin(pts, fua[["eFUA_ID", "eFUA_name", "Cntry_ISO", "FUA_p_2015",
                                 "UC_p_2015", "UC_num", "geometry"]],
                       how="inner", predicate="within")
    print(f"WUP centres inside an FUA: {len(joined):,} "
          f"({len(joined)/len(wup):.0%}) in {joined['eFUA_ID'].nunique():,} FUAs")

    by_code = {r["code"]: r for r in wup}
    members = defaultdict(list)
    meta = {}
    for _, row in joined.iterrows():
        members[row["eFUA_ID"]].append(row["code"])
        meta[row["eFUA_ID"]] = (row["eFUA_name"], row["Cntry_ISO"],
                                float(row["FUA_p_2015"]), float(row["UC_p_2015"]),
                                int(row["UC_num"]))

    out, skipped = {}, defaultdict(int)
    for fid, codes in members.items():
        name, iso, fua_p, uc_p, uc_num = meta[fid]
        principal, named = pick_principal(codes, by_code, name)
        if principal is None:
            skipped["no principal"] += 1
            continue

        years = sorted({y for c in codes for y in by_code[c]["pop"]})
        if not years:
            skipped["no years"] += 1
            continue
        ref = min(years, key=lambda y: abs(y - REF_YEAR))   # 2015 unless the FUA is short-lived
        series = [clamped(by_code[c]["pop"], years) for c in codes]
        total = {y: sum(s[y] for s in series) for y in years}

        if total[ref] <= 0:
            skipped["empty at ref year"] += 1
            continue

        # COVERAGE: did the spatial join actually find this FUA's cores? eFUA states its own
        # urban-centre total (UC_p_2015), so we can check rather than assume. Where the WUP
        # centroids inside the polygon sum to a small fraction of it, the polygon is a
        # duplicate or satellite whose real core sits in a NEIGHBOURING polygon -- there is a
        # second "Jakarta" FUA holding a single 58k centre against a stated core of 1.46M, and
        # three spare "Guangzhou" polygons of the same kind. Scaling a 58k centre up to the
        # FUA's 1.65M would invent a city.
        #
        # This is the gate that a threshold on k cannot be: high k is ALSO the signature of the
        # thing we are here to fix. Charlotte's k is 10.2 and it is entirely real (FUA 2.02M,
        # MSA 2.7M) -- DEGURBA simply leaves 198k of it above 1,500/km2. What separates the two
        # is whether eFUA's own core figure agrees with WUP's, and it separates cleanly: every
        # bad join is at or below 0.20, and the lowest genuine US case (Davenport) is 0.23.
        #
        # The bar is higher when NO member is named after the FUA (COV_NONAME). Those two
        # signals fail independently -- a satellite polygon can clear COV_MIN on a single
        # unrelated town -- and together they are decisive: eFUA has a second "Jakarta" FUA of
        # 2.9M whose only members are Purwakarta and a village, coverage 0.24 and not a Jakarta
        # among them, and on coverage alone Purwakarta was handed 3.8M.
        cov = total[ref] / uc_p if uc_p > 0 else 0.0
        if cov < (COV_MIN if named else COV_NONAME):
            skipped["core not found by join (low coverage)"] += 1
            continue

        # k is the commuting-zone uplift: what the low-density ring adds on top of the cores.
        #
        # CLAMPED AT 1, and the clamp is load-bearing rather than defensive. eFUA is built on
        # GHSL R2019A with a 2015 delineation; WUP 2025 is built on R2023A with a fixed 2025
        # one, and in ten years a lot of ring became core. So for 1,081 FUAs the WUP centres
        # inside the polygon already sum to MORE than the whole polygon held in 2015 -- WUP's
        # single Jakarta centre is 36.1M against an entire 2015 Jakarta FUA of 29.8M. That is
        # not a bad join, it is the ring being absorbed, and scaling by a ratio below 1 would
        # shrink a correct modern measurement to match a stale one. An FUA is a superset of
        # its urban centres by construction, so where the cores have caught up the honest
        # uplift is none: keep the member sum as it stands.
        k = max(1.0, fua_p / total[ref])

        out[principal] = {
            "population": {str(y): round(total[y] * k) for y in years},
            "fua_id": int(fid),
            "fua_name": name,
            "iso3": iso,
            "k": round(k, 4),
            "fua_p_2015": round(fua_p),
            "members": sorted(codes, key=lambda c: -by_code[c]["peak"]),
            "uc_num": uc_num,
            "ref": ref,
            "cov": round(cov, 3),
        }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print(f"\nwrote {OUT}: {len(out):,} FUA series "
          f"({os.path.getsize(OUT)/1e6:.1f} MB)")
    for reason, n in sorted(skipped.items(), key=lambda x: -x[1]):
        print(f"  skipped {n:,}: {reason}")

    ks = sorted(v["k"] for v in out.values())
    multi = sum(1 for v in out.values() if len(v["members"]) > 1)
    print(f"  uplift k: median {ks[len(ks)//2]:.2f}, "
          f"p10 {ks[len(ks)//10]:.2f}, p90 {ks[9*len(ks)//10]:.2f}")
    print(f"  {multi:,} FUAs absorb more than one WUP centre "
          f"({sum(len(v['members']) for v in out.values()) - len(out):,} centres demoted)")

    print("\nspot check (principal WUP 2025 -> FUA 2025):")
    for want in ["New York", "Chicago", "Dallas", "Atlanta", "Boston", "Philadelphia",
                 "Houston", "Seattle", "Detroit", "London", "Paris", "Milan", "Tokyo",
                 "Toronto", "Sydney", "Sao Paulo", "Lagos", "Mumbai"]:
        hit = None
        for code, v in out.items():
            if norm(want) in norm(by_code[code]["name"]) or norm(want) in norm(v["fua_name"]):
                if hit is None or v["fua_p_2015"] > hit[1]["fua_p_2015"]:
                    hit = (code, v)
        if not hit:
            print(f"  {want:14} -- no FUA --")
            continue
        code, v = hit
        w25 = by_code[code]["pop"].get(2025)
        f25 = v["population"].get("2025")
        print(f"  {by_code[code]['name']:16} {w25:>11,} -> {f25:>11,}  "
              f"(k={v['k']:.2f}, {len(v['members'])} centres)  [{v['fua_name']}]")


if __name__ == "__main__":
    if not os.path.exists(FUA_GPKG):
        sys.exit(f"missing {FUA_GPKG} -- download GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.zip")
    build_fua()
