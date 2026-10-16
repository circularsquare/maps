"""T-457: South Korea's inputs for a tier 0 city (the Seoul Capital Area first), through tier0's hooks.

- **Outline and districts**: OSM boundary=administrative relations from the Geofabrik extract at
  the finest level the population table has: autonomous gu, gun and si, and the non-autonomous gu
  inside the large si (Suwon's four, say). Matched by Korean name inside their province (admin_level
  4), so Seoul's and Incheon's Jung-gu stay apart; clipped to land by the OSM coastline polygons;
  cached as a small GeoJSON (the .pbf is deleted after routes.py).
- **Population**: the Ministry of the Interior and Safety's resident registration count by si-gun-gu
  (jumin.mois.go.kr, monthly CSV, open, no key) for the census month, November 2020, spread inside
  each district by GHSL. Registered foreigners are not in it (about 4% of the area's people);
  KOSIS's 2020 census table by si-gun-gu needs a key or the interactive site.
- **Jobs**: tier 0's proxy (the Census on Establishments by si-gun-gu is on KOSIS only).

    python pipeline/korea.py fetch seo     (the registration CSVs)
    python pipeline/korea.py seo           (writes the districts file and prints what it holds)
"""
import csv
import io
import json
import os
import re
import sys

import numpy as np
import shapely
from shapely import make_valid, union_all
from shapely.geometry import mapping, shape

import build_city as b
import england

MOIS_URL = "https://jumin.mois.go.kr/downloadCsv.do?searchYearMonth=month&xlsStats=1"
# Province (sido) codes and names, as OSM and the registration table write them.
SIDO = {"11": "서울특별시", "28": "인천광역시", "41": "경기도"}
REGIONS = {"seo": {"sido": ["11", "28", "41"],
                   # Ongjin-gun's islands reach 200 km out to Baengnyeong by North Korea's coast (T-457)
                   "leave_out": ["2872000000"],
                   "month": ("2020", "11"),
                   # Incheon's July 2026 reorganisation: Jung-gu and Dong-gu became Jemulpo-gu and
                   # Yeongjong-gu, Seo-gu was split into Seohae-gu and Geomdan-gu; OSM has the new gu
                   "merge": {"2814000000": "2811000000"},
                   "osm_names": {"2811000000": ["제물포구", "영종구"], "2826000000": ["서해구", "검단구"]}}}


def raw(name):
    return os.path.join(b.RAW, "tier0", "korea", name)


def fetch(city):
    import requests
    year, month = REGIONS[city]["month"]
    os.makedirs(raw(""), exist_ok=True)
    for sido in REGIONS[city]["sido"]:
        out = raw(f"mois_{sido}_{year}{month}.csv")
        if os.path.exists(out):
            continue
        form = {"sltOrgType": "2", "sltOrgLvl1": sido + "00000000", "sltOrgLvl2": "A", "gender": "gender",
                "genderPer": "genderPer", "generation": "generation", "sltUndefType": "",
                "searchYearStart": year, "searchMonthStart": month, "searchYearEnd": year, "searchMonthEnd": month,
                "sltOrderType": "1", "sltOrderValue": "ASC", "category": "month", "state": "1"}
        r = requests.post(MOIS_URL, data=form, headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
        r.raise_for_status()
        text = r.content.decode("euc-kr")
        if "총인구수" not in text:
            raise SystemExit(f"{sido}: no population table in the response")
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  {out}: {len(text.splitlines()) - 1} rows", flush=True)


def units(city):
    """[(code, sido, name, people)] at the finest level: a si's gu replace the si."""
    year, month = REGIONS[city]["month"]
    rows = []
    for sido in REGIONS[city]["sido"]:
        with open(raw(f"mois_{sido}_{year}{month}.csv"), encoding="utf-8") as f:
            for r in list(csv.reader(f))[1:]:
                m = re.match(r"(.*)\((\d{10})\)", r[0])
                parts, code = m.group(1).split(), m.group(2)
                if len(parts) < 2 or code in REGIONS[city]["leave_out"]:
                    continue
                rows.append((code, sido, " ".join(parts[1:]), int(r[1].replace(",", ""))))
    with_gu = {(s, n.split()[0]) for _, s, n, _ in rows if len(n.split()) == 2}
    rows = [r for r in rows if (r[1], r[2]) not in with_gu]
    merge = REGIONS[city].get("merge", {})
    extra = {}
    for c, _, _, p in rows:
        if c in merge:
            extra[merge[c]] = extra.get(merge[c], 0) + p
    return [(c, s, n, p + extra.get(c, 0)) for c, s, n, p in rows if c not in merge]


def districts_file(city):
    """The districts (OSM, by name inside their province) clipped to land, as a GeoJSON with
    code, name, romanised name and people; its path."""
    out = raw(f"{city}_districts.geojson")
    if os.path.exists(out):
        return out
    import pyogrio
    import routes
    import osm_water
    admin = raw(f"{city}_admin.fgb")  # the extract's admin areas, kept so a rerun skips the slow read
    if not os.path.exists(admin):
        pbf = routes.download(b.CITIES[city]["extract"])
        df = pyogrio.read_dataframe(pbf, layer="multipolygons", columns=["osm_id", "name", "admin_level", "boundary", "other_tags"],
                                    where="boundary = 'administrative' AND admin_level IN ('4', '5', '6', '7')")
        os.makedirs(os.path.dirname(admin), exist_ok=True)
        pyogrio.write_dataframe(df, admin)
    df = pyogrio.read_dataframe(admin)
    prov = {}
    for sido in REGIONS[city]["sido"]:
        rows = df[(df.admin_level == "4") & (df["name"] == SIDO[sido])]
        if len(rows) != 1:
            raise SystemExit(f"{city}: {len(rows)} admin_level 4 boundaries named {SIDO[sido]}")
        prov[sido] = make_valid(rows.geometry.iloc[0]).buffer(0)
    lower = df[df.admin_level != "4"].copy()
    pts = np.array([make_valid(g).representative_point() for g in lower.geometry], dtype=object)
    in_prov = {sido: shapely.contains(p, pts) for sido, p in prov.items()}
    feats, missing = [], []
    w, s, e, n = union_all(list(prov.values())).bounds
    sea = union_all(osm_water.sea((w - 0.01, s - 0.01, e + 0.01, n + 0.01)))
    for code, sido, name, pop in units(city):
        parts, srcs, latins = [], [], []
        for osm_name in REGIONS[city].get("osm_names", {}).get(code, [name.split()[-1]]):
            sel = (lower["name"].to_numpy() == osm_name) & in_prov[sido]
            if len(name.split()) == 2:  # a gu of a si: inside its si
                si = lower[(lower["name"].to_numpy() == name.split()[0]) & in_prov[sido]]
                if len(si):
                    sel &= shapely.contains(make_valid(si.geometry.iloc[0]).buffer(0), pts)
            cand = lower[sel]
            if len(cand) == 0:
                missing.append(f"{SIDO[sido]} {name} ({osm_name})")
                continue
            cand = cand.iloc[[int(cand.geometry.area.values.argmax())]]
            parts.append(make_valid(cand.geometry.iloc[0]).buffer(0))
            srcs.append(f"r{cand.osm_id.iloc[0]} level {cand.admin_level.iloc[0]}")
            tags = cand.other_tags.iloc[0] or ""
            m = re.search(r'"name:ko-Latn"=>"([^"]*)"', tags) or re.search(r'"name:en"=>"([^"]*)"', tags)
            latins.append(m.group(1) if m else osm_name)
        if not parts:
            continue
        g = union_all(parts).buffer(0).difference(sea).buffer(0)
        if g.geom_type == "MultiPolygon":
            g = shapely.MultiPolygon([p for p in g.geoms if p.area > 1e-6])
        feats.append({"type": "Feature", "properties": {"code": code, "name": f"{SIDO[sido]} {name}",
                                                        "latin": " and ".join(latins), "pop": pop,
                                                        "source": "OSM " + ", ".join(srcs)},
                      "geometry": mapping(g)})
    if missing:
        raise SystemExit(f"{city}: no OSM boundary for {missing}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".part", "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False)
    os.replace(out + ".part", out)
    return out


def districts(city):
    with open(districts_file(city), encoding="utf-8") as f:
        return [(ft["properties"]["code"], shape(ft["geometry"]), ft["properties"]["pop"], ft["properties"]["latin"])
                for ft in json.load(f)["features"]]


def boundary_file(city):
    """The districts dissolved, for tier0's `boundary_file`; its path."""
    out = raw(f"{city}_outline.geojson")
    if not os.path.exists(out):
        g = union_all([d[1] for d in districts(city)]).buffer(0)
        with open(out + ".part", "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"city": city},
                                                                   "geometry": mapping(g)}]}, f)
        os.replace(out + ".part", out)
    return out


def registered_population(city, boundary, sig):
    """Registered residents by si-gun-gu (November 2020), spread inside each by GHSL; tier0's
    `population` hook."""
    ds = districts(city)
    units_ = [(d[0], d[1]) for d in ds]
    out = england.spread_totals(sig["lat"], sig["lon"], units_, {d[0]: d[2] for d in ds}, sig["pop_ghsl"])
    idx, _ = england.cell_units(sig["lat"], sig["lon"], units_)
    for i, (code, _, pop, latin) in enumerate(ds):
        have = sig["pop_ghsl"][idx == i].sum()
        print(f"  registered 2020-11 {latin}: GHSL {have:,.0f} -> {pop:,} ({pop / max(have, 1) - 1:+.1%})", flush=True)
    print(f"  registered by si-gun-gu: {len(ds)} districts, {out.sum():,.0f} people", flush=True)
    return out, ("resident registration population by si-gun-gu, November 2020 (Ministry of the Interior and Safety, "
                 "jumin.mois.go.kr; registered foreigners not included), spread inside each by GHSL GHS-POP E2020")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1] == "fetch":
        fetch(sys.argv[2])
        sys.exit()
    city = sys.argv[1]
    ds = districts(city)
    g = union_all([d[1] for d in ds])
    km2 = 111.32 * 110.57 * 0.80  # degrees squared to km2 near 37 N
    print(f"{city}: {len(ds)} districts, {sum(d[2] for d in ds):,} people, outline {g.geom_type}, "
          f"{len(getattr(g, 'geoms', [g]))} parts, about {g.area * km2:,.0f} km2, bounds {tuple(round(v, 3) for v in g.bounds)}")
    for d in ds:
        parts = list(getattr(d[1], "geoms", [d[1]]))
        print(f"  {d[0]} {d[3]}: {len(parts)} parts, {sum(len(p.interiors) for p in parts)} holes, about {d[1].area * km2:,.1f} km2")
