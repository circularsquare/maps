"""Build Pakistan's three boundary levels for helper1m, on the 2023 census's own units.

Reads
  helper1m/data/pakistan/census_units.csv       (fetch.py, from PBS Table 1)
  helper1m/data/pakistan/raw/osm_tehsils_*.json (OSM admin 6/7, fetched by osm.py)
  data/asia1m/pakistan/pak_admin{1,2,3}.shp     (OCHA COD-AB v01, 2022)
  religiondots/data/geo/pk2023/pk_districts.gpkg (the census's 136 districts, as religiondots
                                                  built them; used only to say which census
                                                  district each OSM tehsil belongs to)
Writes
  helper1m/data/pakistan/boundaries/adm{1,2,3}.gpkg   columns code, name, parent, group
  helper1m/data/pakistan/crosswalk.csv                census unit -> adm3/adm2/adm1 code

Method (scripts/pakistan/README.md has the long version):
  * four provinces + Islamabad: OSM admin_level=7 tehsils, paired with census units by
    crosswalk.py, holes filled from COD, clipped to COD's outline of those five units;
  * Azad Kashmir and Gilgit-Baltistan: COD tehsils (AJK) and districts (GB) as they are;
  * districts and provinces are dissolved from the tehsil level, so the three levels nest.
"""

import os
import re
import sys
import difflib

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
import pandas as pd
import shapely
from shapely.ops import unary_union

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ajk_gb
import crosswalk as cw
import osm

REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DATA = os.path.join(REPO, "helper1m", "data", "pakistan")
RAW = os.path.join(DATA, "raw")
OUT = os.path.join(DATA, "boundaries")
COD = os.path.join(REPO, "data", "asia1m", "pakistan")
RD_DISTRICTS = os.path.join(REPO, "religiondots", "data", "geo", "pk2023", "pk_districts.gpkg")

EQ = "ESRI:54009"          # Mollweide, for areas

PROV_CODE = {"Azad Kashmir": "PK1", "Balochistan": "PK2", "Gilgit-Baltistan": "PK3",
             "Islamabad": "PK4", "Khyber Pakhtunkhwa": "PK5", "Punjab": "PK6", "Sindh": "PK7"}
COD_PROV = {"Azad Kashmir": "Azad Kashmir", "Balochistan": "Balochistan",
            "Gilgit-Baltistan": "Gilgit Baltistan", "Islamabad": "Islamabad",
            "Khyber Pakhtunkhwa": "Khyber Pakhtunkhwa", "Punjab": "Punjab", "Sindh": "Sindh"}

_SUFFIX = re.compile(r"\b(SUB-DIVISION|SUB- DIVISION|SUB DIVISION|SUB-TEHSIL|TEHSIL|TALUKA|TOWN|DIVISION|SUB)\b", re.I)


def fold(s):
    return re.sub(r"[^a-z]", "", str(s).lower())


def fold_unit(s):
    return fold(_SUFFIX.sub(" ", str(s)))


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def pretty(s):
    t = " ".join(str(s).split()).title()
    return t.replace("Sub- Division", "Sub-Division").replace("Sub-Tehsil", "Sub-tehsil") \
            .replace("Sub-Division", "Sub-division").replace("(S.F.Rahu)", "(S.F. Rahu)")


def district_base(name):
    return name[:-len(" DISTRICT")] if name.endswith(" DISTRICT") else name


def load_osm_tehsils(cod1):
    g = osm.polygons(RAW)
    g = g[(g.level == 7) & g.geometry.notna()].copy()
    g["geometry"] = shapely.make_valid(g.geometry.values)
    g = g[~g.geometry.is_empty]
    if g["osm_id"].duplicated().any():
        raise SystemExit("duplicate OSM ids")
    # drop the ones in Azad Kashmir / Gilgit-Baltistan (COD is used there)
    north = cod1[cod1.adm1_name.isin(["Azad Kashmir", "Gilgit Baltistan"])].to_crs(EQ).union_all()
    ge = g.to_crs(EQ)
    share_n = ge.geometry.intersection(north).area / ge.area
    g = g[(share_n < 0.5).to_numpy()].copy()
    return g


def assign_districts(o7, t1_districts):
    rd = gpd.read_file(RD_DISTRICTS)[["geo_name", "geometry"]].to_crs(EQ)
    rd["district"] = rd.geo_name.map(lambda n: cw.DIST_T9_TO_T1.get(n, n))
    missing = set(t1_districts) - set(rd.district)
    if missing:
        raise SystemExit(f"census districts with no religiondots polygon: {sorted(missing)}")
    ov = gpd.overlay(o7[["osm_id", "geometry"]].to_crs(EQ), rd[["district", "geometry"]],
                     how="intersection", keep_geom_type=False)
    ov["a"] = ov.area
    best = ov.sort_values("a").groupby("osm_id").tail(1).set_index("osm_id")["district"]
    o7 = o7.copy()
    o7["district"] = o7.osm_id.map(best)
    for name, d in cw.OSM_DISTRICT_OVERRIDE.items():
        sel = o7.name == name
        if sel.sum() != 1:
            raise SystemExit(f"OSM_DISTRICT_OVERRIDE: {sel.sum()} OSM tehsils named {name}")
        o7.loc[sel, "district"] = d
    lost = o7[o7.district.isna()]
    if len(lost):
        raise SystemExit(f"OSM tehsils in no census district: {lost.name.tolist()}")
    return o7


def auto_pairs(cunits, onames):
    """1:1 name pairing, exact fold first, then difflib >= 0.75, best pairs first."""
    pairs, cu, on = [], list(cunits), list(onames)
    fo = {o: fold_unit(o) for o in on}
    for c in list(cu):
        k = fold_unit(c)
        hit = [o for o in on if fo[o] == k]
        if len(hit) == 1:
            pairs.append((c, hit[0]))
            cu.remove(c)
            on.remove(hit[0])
    cand = sorted(((difflib.SequenceMatcher(None, fold_unit(c), fo[o]).ratio(), c, o)
                   for c in cu for o in on), reverse=True)
    for r, c, o in cand:
        if r < 0.75:
            break
        if c in cu and o in on:
            pairs.append((c, o))
            cu.remove(c)
            on.remove(o)
    return pairs, cu, on


def build_units(units, o7):
    """List of dicts: code, name, district, province, members (census names), terms, reason."""
    out = []
    by_d = {}
    for u in units.itertuples():
        if u.level == "tehsil":
            by_d.setdefault((u.province, u.district), []).append(u.name)
    problems = []
    for (prov, dist), cnames in sorted(by_d.items()):
        osm_here = o7[o7.district == dist]
        onames = osm_here.name.tolist()
        if len(set(onames)) != len(onames):
            problems.append(f"{dist}: duplicate OSM names {onames}")
        d_code = f"{PROV_CODE[prov]}-{slug(district_base(dist))}"
        groups = []
        if dist in cw.WHOLE_DISTRICT:
            groups.append((cnames, onames, cw.WHOLE_DISTRICT[dist]))
            cleft, oleft = [], []
        else:
            cleft, oleft = list(cnames), list(onames)
            for cs, terms, why in cw.GROUPS.get(dist, []):
                for c in cs:
                    if c not in cleft:
                        problems.append(f"{dist}: GROUPS names census unit {c!r}, not left in district")
                    else:
                        cleft.remove(c)
                for t in terms:
                    n = t if isinstance(t, str) else (t[1] if t[0] != "cod-osm" else None)
                    if n is None:
                        continue
                    if n not in onames:
                        problems.append(f"{dist}: GROUPS names OSM tehsil {n!r}, not in district "
                                        f"(have {onames})")
                    elif n in oleft:
                        oleft.remove(n)
                groups.append((cs, terms, why))
            pairs, cleft, oleft = auto_pairs(cleft, oleft)
            for c, o in pairs:
                groups.append(([c], [o], "name"))
        if cleft or oleft:
            problems.append(f"{dist}: unpaired census {cleft}, unpaired OSM {oleft}")
        for cs, terms, why in groups:
            if dist in cw.WHOLE_DISTRICT:
                code = f"{d_code}-all"
                name = f"{pretty(district_base(dist))} (whole district, {len(cs)} census units)"
            else:
                code = f"{d_code}-{slug(fold_unit(cs[0]) or cs[0])}"
                name = " + ".join(pretty(c) for c in cs)
            out.append({"code": code, "name": name, "district": dist, "province": prov,
                        "members": cs, "terms": terms, "reason": why,
                        "osm_names": [t if isinstance(t, str) else t[1] for t in terms
                                      if isinstance(t, str) or t[0] != "cod-osm"]})
    if problems:
        for p in problems:
            print("  PROBLEM", p)
        raise SystemExit(f"{len(problems)} crosswalk problems")
    codes = [u["code"] for u in out]
    dup = {c for c in codes if codes.count(c) > 1}
    if dup:
        raise SystemExit(f"duplicate unit codes {dup}")
    return out


def geometry_of(u, o7, cod3, osm_all_union):
    here = o7[o7.district == u["district"]].set_index("name")
    parts = []
    for t in u["terms"]:
        if isinstance(t, str):
            parts.append(here.loc[t, "geometry"])
        elif t[0] == "osm&cod":
            parts.append(here.loc[t[1], "geometry"].intersection(cod3.loc[t[2], "geometry"]))
        elif t[0] == "osm-cod":
            parts.append(here.loc[t[1], "geometry"].difference(cod3.loc[t[2], "geometry"]))
        elif t[0] == "cod-osm":
            parts.append(cod3.loc[t[1], "geometry"].difference(osm_all_union))
        else:
            raise SystemExit(f"unknown term {t}")
    return unary_union(parts)


def drop_pinholes(geom, max_deg2=1e-4):
    """Remove interior rings under ~1 km2: slivers left where two sources' lines meet."""
    from shapely.geometry import MultiPolygon, Polygon

    def one(p):
        return Polygon(p.exterior, [r for r in p.interiors if Polygon(r).area > max_deg2])
    if geom.geom_type == "Polygon":
        return one(geom)
    if geom.geom_type == "MultiPolygon":
        return MultiPolygon([one(p) for p in geom.geoms])
    return geom


def fill_gaps(gdf, domain):
    """Give every piece of `domain` no unit covers to the unit it touches most."""
    covered = gdf.union_all()
    gaps = gpd.GeoDataFrame(geometry=[domain.difference(covered)], crs=4326).explode(index_parts=False)
    gaps = gaps[~gaps.geometry.is_empty]
    gaps["km2"] = gaps.to_crs(EQ).area / 1e6
    total = gaps.km2.sum()
    big = gaps[gaps.km2 > 0]
    sidx = gdf.sindex
    add = {}
    for g in big.geometry:
        gb = g.buffer(0.0005)
        cand = list(sidx.query(gb, predicate="intersects"))
        if not cand:
            continue
        best = max(cand, key=lambda i: gb.intersection(gdf.geometry.iloc[i]).area)
        add.setdefault(best, []).append(g)
    for g, km2 in sorted(zip(big.geometry, big.km2), key=lambda x: -x[1])[:8]:
        gb = g.buffer(0.0005)
        cand = list(sidx.query(gb, predicate="intersects"))
        to = max(cand, key=lambda i: gb.intersection(gdf.geometry.iloc[i]).area) if cand else None
        c = g.representative_point()
        print(f"      largest: {km2:8.1f} km2 at {c.y:.3f}N {c.x:.3f}E -> "
              f"{gdf.code.iloc[to] if to is not None else 'nothing'}")
    geoms = list(gdf.geometry)
    for i, gs in add.items():
        geoms[i] = unary_union([geoms[i]] + gs)
    gdf = gdf.copy()
    gdf["geometry"] = geoms
    print(f"  gaps between units inside the census area: {total:,.1f} km2 in {len(gaps)} pieces; "
          f"{big.km2.sum():,.1f} km2 in {len(big)} pieces given to the unit they touch most")
    return gdf


def main():
    units = pd.read_csv(os.path.join(DATA, "census_units.csv"))
    cod1 = gpd.read_file(os.path.join(COD, "pak_admin1.shp")).to_crs(4326)
    cod2 = gpd.read_file(os.path.join(COD, "pak_admin2.shp")).to_crs(4326)
    cod3 = gpd.read_file(os.path.join(COD, "pak_admin3.shp")).to_crs(4326)
    cod3["geometry"] = shapely.make_valid(cod3.geometry.values)
    cod3i = cod3.set_index("adm3_pcode")

    o7 = load_osm_tehsils(cod1)
    t1d = sorted(units.loc[units.level == "district", "name"])
    o7 = assign_districts(o7, t1d + ["ISLAMABAD DISTRICT"])
    print(f"OSM tehsils outside AJK/GB: {len(o7)}; census districts: {len(t1d)} + Islamabad")

    ulist = build_units(units, o7)
    print(f"tehsil-level units for the census area: {len(ulist)} "
          f"(from {int((units.level == 'tehsil').sum())} census units)")

    osm_all = o7.union_all()
    mainland = cod1[cod1.adm1_name.isin(["Balochistan", "Islamabad", "Khyber Pakhtunkhwa",
                                          "Punjab", "Sindh"])].union_all()
    geoms = [geometry_of(u, o7, cod3i, osm_all).intersection(mainland) for u in ulist]
    g3 = gpd.GeoDataFrame({
        "code": [u["code"] for u in ulist], "name": [u["name"] for u in ulist],
        "district": [u["district"] for u in ulist], "province": [u["province"] for u in ulist],
    }, geometry=geoms, crs=4326)
    empty = g3[g3.geometry.is_empty]
    if len(empty):
        raise SystemExit(f"empty unit geometries: {empty.code.tolist()}")
    g3 = fill_gaps(g3, mainland)

    # overlaps (should be none: OSM tiles, and the COD pieces are differences)
    ge = g3.to_crs(EQ)
    ov = gpd.overlay(ge[["code", "geometry"]], ge[["code", "geometry"]], how="intersection",
                     keep_geom_type=False)
    ov = ov[ov.code_1 < ov.code_2]
    ov_km2 = ov.area.sum() / 1e6
    print(f"  overlap between units: {ov_km2:,.2f} km2")

    # ---- Azad Kashmir: COD tehsils, 1:1 with the yearbook's tehsils
    ajk3 = cod3[cod3.adm1_name == "Azad Kashmir"]
    rows3 = []
    for name, d, cname, p23, p17 in ajk_gb.AJK_TEHSILS:
        hit = ajk3[(ajk3.adm2_name == d) & (ajk3.adm3_name == cname)]
        if len(hit) != 1:
            raise SystemExit(f"AJK: COD has {len(hit)} tehsils {cname} in {d}")
        r = hit.iloc[0]
        rows3.append({"code": r.adm3_pcode, "name": name if name == cname else f"{name} ({cname})",
                      "district": f"AJK:{d}", "province": "Azad Kashmir", "geometry": r.geometry})
    if len(rows3) != len(ajk3):
        raise SystemExit(f"AJK: {len(ajk3)} COD tehsils, {len(rows3)} paired")
    # ---- Gilgit-Baltistan: COD districts, merged to the census's ten
    gb2 = cod2[cod2.adm1_name == "Gilgit Baltistan"]
    used = []
    for name, cods, p23, p17 in ajk_gb.GB_DISTRICTS:
        hit = gb2[gb2.adm2_name.isin(cods)]
        if len(hit) != len(cods):
            raise SystemExit(f"GB: {name} wants COD {cods}, found {hit.adm2_name.tolist()}")
        used += cods
        code = hit[hit.adm2_name == cods[0]].adm2_pcode.iloc[0]
        rows3.append({"code": code, "name": name, "district": f"GB:{name}",
                      "province": "Gilgit-Baltistan", "geometry": hit.union_all()})
    if sorted(used) != sorted(gb2.adm2_name):
        raise SystemExit(f"GB: COD districts unused {set(gb2.adm2_name) - set(used)}")
    g3 = pd.concat([g3, gpd.GeoDataFrame(rows3, crs=4326)], ignore_index=True)

    # ---- district and province codes
    def d_code(r):
        if r.province == "Azad Kashmir":
            return cod2[(cod2.adm1_name == "Azad Kashmir") & (cod2.adm2_name == r.district[4:])].adm2_pcode.iloc[0]
        if r.province == "Gilgit-Baltistan":
            return r.code
        return f"{PROV_CODE[r.province]}-{slug(district_base(r.district))}"

    def d_name(r):
        if ":" in r.district:
            return r.district.split(":", 1)[1]
        return pretty(district_base(r.district))

    g3["parent"] = [d_code(r) for r in g3.itertuples()]
    g3["district_name"] = [d_name(r) for r in g3.itertuples()]
    g3["group"] = g3.province.map(PROV_CODE)
    g3["province_name"] = g3.province

    os.makedirs(OUT, exist_ok=True)
    g3["geometry"] = [drop_pinholes(g) for g in g3.geometry]
    g3[["code", "name", "parent", "group", "geometry"]].to_file(os.path.join(OUT, "adm3.gpkg"), driver="GPKG")
    g2 = g3.dissolve(by="parent", aggfunc={"district_name": "first", "group": "first"}).reset_index()
    g2["geometry"] = [drop_pinholes(g) for g in g2.geometry]
    g2 = g2.rename(columns={"parent": "code", "district_name": "name"})
    g2["parent"] = g2["group"]
    g2[["code", "name", "parent", "group", "geometry"]].to_file(os.path.join(OUT, "adm2.gpkg"), driver="GPKG")
    g1 = g3.dissolve(by="group", aggfunc={"province_name": "first"}).reset_index()
    g1 = g1.rename(columns={"province_name": "name"})
    g1["geometry"] = [drop_pinholes(g) for g in g1.geometry]
    g1["code"] = g1["group"]
    g1[["code", "name", "group", "geometry"]].to_file(os.path.join(OUT, "adm1.gpkg"), driver="GPKG")
    print(f"wrote {OUT}: adm1 {len(g1)}, adm2 {len(g2)}, adm3 {len(g3)}")

    # ---- crosswalk: census unit -> codes
    cwrows = []
    for u in ulist:
        for m in u["members"]:
            cwrows.append({"source": "pbs_table1", "province": u["province"], "district": u["district"],
                           "unit": m, "adm3": u["code"], "reason": u["reason"],
                           "polygons": "; ".join(t if isinstance(t, str) else ":".join(t) for t in u["terms"])})
    for name, d, cname, p23, p17 in ajk_gb.AJK_TEHSILS:
        code = cod3[(cod3.adm1_name == "Azad Kashmir") & (cod3.adm2_name == d) & (cod3.adm3_name == cname)].adm3_pcode.iloc[0]
        cwrows.append({"source": "ajk_yearbook", "province": "Azad Kashmir", "district": d, "unit": name,
                       "adm3": code, "reason": "name" if name == cname else f"COD calls it {cname}",
                       "polygons": f"cod:{code}"})
    for name, cods, p23, p17 in ajk_gb.GB_DISTRICTS:
        code = g3[(g3.province == "Gilgit-Baltistan") & (g3.name == name)].code.iloc[0]
        cwrows.append({"source": "gb_at_a_glance", "province": "Gilgit-Baltistan", "district": name,
                       "unit": name, "adm3": code, "reason": "district only; COD " + ", ".join(cods),
                       "polygons": "cod:" + ",".join(cods)})
    cwdf = pd.DataFrame(cwrows)
    par = dict(zip(g3.code, g3.parent))
    grp = dict(zip(g3.code, g3.group))
    cwdf["adm2"] = cwdf.adm3.map(par)
    cwdf["adm1"] = cwdf.adm3.map(grp)
    cwdf.to_csv(os.path.join(DATA, "crosswalk.csv"), index=False)
    print(f"wrote crosswalk.csv ({len(cwdf)} census units)")


if __name__ == "__main__":
    main()
