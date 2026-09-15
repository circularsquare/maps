"""The national estimate layer, spec §15: Pew 2020 totals plus hand rows, for the viewer.

    python estimates.py          writes data/processed/estimates.json and estimate_shapes.geojson
    python estimates.py --dry    prints what would ship and writes nothing

RUN IT AFTER tiles.py. §15.3's refusal reads each built country's `covers` out of counts.json.
Running it stale fails in the safe direction: a country built since the last run is still treated
as unbuilt and keeps its outlines until this runs again.

WHAT SHIPS, Anita's calls of 2026-09-14:
  * Pew's seven families for 2020, for every country Pew covers. Pew counts self-identification.
  * hand rows from `estimates_hand.py`, for anything Pew does not name, self-identification first.
  * NOTHING from the World Religion Project, which tools/scan_estimates.py found unfit (§15.4b).
  * the viewer's layer is OFF by default, and it fetches these two files only when turned on.

FOUR RULES ARE APPLIED HERE SO THAT THE VIEWER NEVER HAS TO KNOW THEM:
  1. §15.3. An estimate for a node a built country's own source measured, the node itself or
     anything below it, is not shown. It still counts toward the world total, because that total
     is Pew's number and not the map's.
  2. `within`. A hand row inside a Pew family in Pew's own count has its range taken out of that
     family, in its country and in the world. Pew has no Alevi category, so Türkiye's Alevis are
     inside its Muslims; the taxonomy keeps `alevism` apart, and so does this layer.
  3. Precedence (§15.5). A hand row for a node Pew also names replaces Pew's figure for that
     country, since it is the more specific source.
  4. A floor of 10,000. Pew's rounded counts print "<10,000" below that, and a figure the source
     will only state as "fewer than" is not a presence worth outlining. Such rows still count
     toward the world total.

A HAND ROW MAY BE A SHARE OF ONE OF PEW'S FAMILIES rather than of the country (`of`), which is
§15.4's chain: Yemen's Shia are a survey's share of Muslims times Pew's 2020 Muslims. The node must
sit directly under that family, and no other hand row may have changed Pew's figure for it there.

THE SHAPES ARE POLYGONS, because the country's interior is what answers a hover (spec §15.11).
The viewer makes the drawn outline's lines from them and leaves Natural Earth's 180° cut out of
those lines (`estimateLines` in index.html), since a line layer drawn off a polygon draws that cut
as a border running down the Pacific.
"""
import argparse
import csv
import io
import json
import math
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import shapely
import shapely.geometry

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw" / "estimates"
PEW_ZIP = RAW / "pew.zip"
NE = HERE / "data" / "geo" / "ne_10m_admin_0_countries.geojson"
PROCESSED = HERE / "data" / "processed"
COUNTS = PROCESSED / "counts.json"
BUILT_SHAPES = PROCESSED / "country_shapes.geojson"
TREE = HERE / "taxonomy" / "religions.json"
OUT = PROCESSED / "estimates.json"
OUT_SHAPES = PROCESSED / "estimate_shapes.geojson"

PEW_YEAR = "2020"
PEW_SOURCE = "Pew Research Center, Religious Composition by Country 2010-2020"
PEW_FAMILIES = {
    "Christians": "christianity", "Muslims": "islam", "Buddhists": "buddhism",
    "Hindus": "hinduism", "Jews": "judaism", "Religiously_unaffiliated": "unaffiliated",
    "Other_religions": None,
}
PEW_NUM_ALIAS = {"412": "XK"}         # Kosovo; Natural Earth carries no numeric code for it
FLOOR = 10_000
BASES = {"self_id", "estimate", "roll"}

# (cc, node) -> why an estimate restates a column the country's source measured under another
# node. §15.3's node test cannot see these (§15.4b); each is argued here.
SAME_COLUMN = {
    ("sg", "daoism"): "Singapore's census Taoism includes Chinese traditional beliefs and is "
                      "drawn as chinesefolk",
}

# Built countries keep close to the wash's own outline; unbuilt ones never sit beside a dot, so
# a coarser line costs nothing a reader can check (§15.7). Degrees: ~500 m and ~1 km.
SIMPLIFY_BUILT, SIMPLIFY_UNBUILT = 0.005, 0.01
ROUND = 3


def num(v):
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0.0


def cc_of(iso2):
    return "uk" if iso2 == "GB" else iso2.lower()


def approx(n):
    """Two significant figures, and nothing finer than Pew's own 10,000 (§15.4)."""
    if n is None:
        return "-"
    if n < FLOOR:
        return "<10k"
    d = 10 ** (math.floor(math.log10(n)) - 1)
    n = round(n / d) * d
    if n >= 1e9:
        return f"{n / 1e9:g}bn"
    if n >= 1e6:
        return f"{n / 1e6:g}m"
    return f"{n / 1e3:g}k"


# ------------------------------------------------------------------------------------------
# inputs
# ------------------------------------------------------------------------------------------

def load_ne():
    """Natural Earth admin 0: ISO numeric -> alpha-2, and alpha-2 -> its feature."""
    feats = json.loads(NE.read_text(encoding="utf-8"))["features"]
    by_num, by_iso2 = {}, {}
    for eh in (False, True):             # the plain codes first; the _EH ones only fill gaps
        for f in feats:
            p = f["properties"]
            iso2 = str(p.get("ISO_A2_EH" if eh else "ISO_A2") or "")
            if len(iso2) != 2:
                continue
            by_iso2.setdefault(iso2, f)
            n3 = str(p.get("ISO_N3_EH" if eh else "ISO_N3") or "")
            if n3.isdigit():
                by_num.setdefault(n3.zfill(3), iso2)
    return by_num, by_iso2


def load_pew(by_num):
    """alpha-2 -> {name, pop, <family>: count} for 2020, and Pew's own world row."""
    with zipfile.ZipFile(PEW_ZIP) as z:
        name = [n for n in z.namelist() if n.endswith("(unrounded counts).csv")][0]
        rows = list(csv.DictReader(io.StringIO(z.read(name).decode("utf-8-sig"))))
    out, world, unjoined = {}, None, []
    for r in rows:
        if r["Year"] != PEW_YEAR:
            continue
        rec = dict(name=r["Country"], pop=num(r["Population"]),
                   **{f: num(r[f]) for f in PEW_FAMILIES})
        if r["Level"] == "3":
            world = rec
        if r["Level"] != "1":
            continue
        code = r["Countrycode"].strip().zfill(3)
        iso2 = PEW_NUM_ALIAS.get(code) or by_num.get(code)
        if iso2:
            out[iso2] = rec
        else:
            unjoined.append((r["Country"], rec["pop"]))
    return out, world, unjoined


def refused(covers, cc, node):
    """§15.3: why this estimate may not be shown, or None. The one copy of the rule."""
    cov = covers.get(cc)
    if cov is None:
        return None
    hit = sorted(c for c in cov if c == node or c.startswith(node + "."))
    if hit:
        return f"the country's own source measures {hit[0]}"
    return SAME_COLUMN.get((cc, node))


# ------------------------------------------------------------------------------------------
# the table
# ------------------------------------------------------------------------------------------

def build():
    from estimates_hand import HAND

    by_num, by_iso2 = load_ne()
    pew, pew_world, unjoined = load_pew(by_num)
    counts = json.loads(COUNTS.read_text(encoding="utf-8"))["countries"]
    covers = {cc: set(c.get("covers") or []) for cc, c in counts.items()}
    tree = {n["id"] for n in json.loads(TREE.read_text(encoding="utf-8"))["nodes"]}
    pew_cc = {cc_of(iso2): rec for iso2, rec in pew.items()}

    rows = {}                            # (cc, node) -> row; lo == hi for a single figure
    for cc, p in pew_cc.items():
        for fam, node in PEW_FAMILIES.items():
            if node:
                rows[(cc, node)] = dict(cc=cc, node=node, lo=p[fam], hi=p[fam], src="pew",
                                        year=PEW_YEAR, source=PEW_SOURCE, basis="self_id")

    pew_column = {node: fam for fam, node in PEW_FAMILIES.items() if node}
    # a family figure a hand row replaces or takes something out of, per country
    changed = ({(h.get("cc"), h.get("node")) for h in HAND}
               | {(h.get("cc"), h.get("within")) for h in HAND if h.get("within")})
    for h in HAND:
        where = f"estimates_hand.py row {h.get('cc')}/{h.get('node')}"
        for k in ("cc", "node", "low", "high", "basis", "year", "source", "note"):
            if k not in h:
                sys.exit(f"!! {where}: missing `{k}`")
        if h["node"] not in tree:
            sys.exit(f"!! {where}: {h['node']} is not a node in religions.json")
        if h["cc"] not in pew_cc:
            sys.exit(f"!! {where}: Pew has no 2020 population for {h['cc']} to scale the share by")
        if h["basis"] not in BASES:
            sys.exit(f"!! {where}: basis must be one of {sorted(BASES)}")
        if not 0 <= h["low"] <= h["high"] <= 1:
            sys.exit(f"!! {where}: low and high are shares, low <= high")
        base = pew_cc[h["cc"]]["pop"]
        of = h.get("of")
        if of:
            # §15.4's chain: a share of one of Pew's families, times Pew's own 2020 figure for it
            if of not in pew_column:
                sys.exit(f"!! {where}: of={of} is not one of Pew's families {sorted(pew_column)}")
            if h["node"].rpartition(".")[0] != of:
                sys.exit(f"!! {where}: of={of}, and §15.4 takes a share only of the node directly "
                         "above")
            if h.get("within") or (h["cc"], of) in changed:
                sys.exit(f"!! {where}: of={of} needs Pew's own figure for {of} in {h['cc']}, and a "
                         "hand row changes that figure")
            base = pew_cc[h["cc"]][pew_column[of]]
        rows[(h["cc"], h["node"])] = dict(
            cc=h["cc"], node=h["node"], lo=h["low"] * base, hi=h["high"] * base, src="hand",
            year=h["year"], source=h["source"], basis=h["basis"])

    for h in HAND:
        fam = h.get("within")
        if not fam:
            continue
        base = rows.get((h["cc"], fam))
        if base is None or base["src"] != "pew":
            sys.exit(f"!! estimates_hand.py {h['cc']}/{h['node']}: within={fam}, and there is "
                     "no Pew figure for that family in that country to take it out of")
        inner = rows[(h["cc"], h["node"])]
        base["lo"], base["hi"] = base["lo"] - inner["hi"], base["hi"] - inner["lo"]
        base.setdefault("less", []).append(h["node"])

    for r in rows.values():
        r["why"] = refused(covers, r["cc"], r["node"])
        r["shown"] = r["why"] is None and r["hi"] >= FLOOR

    world = {}
    for r in rows.values():
        w = world.setdefault(r["node"], dict(lo=0.0, hi=0.0, countries=0, shown=[], ccs=[],
                                             src=set()))
        w["ccs"].append(r["cc"])
        w["lo"] += r["lo"]
        w["hi"] += r["hi"]
        w["countries"] += 1
        w["src"].add(r["src"])
        if r["shown"]:
            w["shown"].append(r["cc"])

    return dict(rows=rows, world=world, pew=pew_cc, pew_world=pew_world, unjoined=unjoined,
                counts=counts, by_iso2=by_iso2)


def _figure(lo, hi):
    lo, hi = round(lo), round(hi)
    return {"n": lo} if lo == hi else {"lo": lo, "hi": hi}


def to_json(t):
    countries = {}
    for (cc, node), r in sorted(t["rows"].items()):
        if not r["shown"]:
            continue
        c = countries.setdefault(cc, dict(
            name=t["counts"][cc]["name"] if cc in t["counts"] else t["pew"][cc]["name"],
            pop=round(t["pew"][cc]["pop"]), est={}))
        e = dict(_figure(r["lo"], r["hi"]), src=r["src"], year=r["year"], source=r["source"],
                 basis=r["basis"])
        if r.get("less"):
            e["less"] = r["less"]
        c["est"][node] = e
    world = {}
    for node, w in sorted(t["world"].items()):
        world[node] = dict(_figure(w["lo"], w["hi"]), countries=w["countries"],
                           shown=sorted(w["shown"]), pew="pew" in w["src"])
        if "pew" not in w["src"]:
            # a node only hand rows cover is named by its countries in the viewer, not "worldwide"
            world[node]["ccs"] = sorted(w["ccs"])
    return dict(pew_source=PEW_SOURCE, pew_year=int(PEW_YEAR), floor=FLOOR,
                countries=countries, world=world)


# ------------------------------------------------------------------------------------------
# the outlines
# ------------------------------------------------------------------------------------------

def _round(obj):
    if isinstance(obj, list):
        return [_round(o) for o in obj]
    if isinstance(obj, float):
        return round(obj, ROUND)
    return obj


def _polygons(geom):
    """Only the areas: make_valid can hand back stray lines and points along a simplified coast."""
    parts = []
    for p in shapely.get_parts(geom):
        if p.geom_type == "Polygon":
            parts.append(p)
        elif p.geom_type == "MultiPolygon":
            parts.extend(shapely.get_parts(p))
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else shapely.MultiPolygon(parts)


def shapes(t, ccs):
    built = defaultdict(list)
    for f in json.loads(BUILT_SHAPES.read_text(encoding="utf-8"))["features"]:
        built[f["properties"]["cc"]].append(shapely.geometry.shape(f["geometry"]))
    feats, missing = [], []
    for cc in sorted(ccs):
        if cc in t["counts"]:
            geoms, tol = built.get(cc, []), SIMPLIFY_BUILT
        else:
            f = t["by_iso2"].get("GB" if cc == "uk" else cc.upper())
            geoms, tol = ([shapely.geometry.shape(f["geometry"])] if f else []), SIMPLIFY_UNBUILT
        if not geoms:
            missing.append(cc)
            continue
        # one outline per country: the UK's three census regions dissolve into one border
        g = shapely.union_all(geoms) if len(geoms) > 1 else geoms[0]
        g = shapely.simplify(g, tol)
        if not shapely.is_valid(g):
            g = shapely.make_valid(g)
        g = _polygons(g)
        if g is not None:
            feats.append({"type": "Feature", "properties": {"cc": cc},
                          "geometry": _round(json.loads(shapely.to_geojson(g)))})
    return {"type": "FeatureCollection", "features": feats}, missing


# ------------------------------------------------------------------------------------------

def report(t, js):
    print(f"Pew countries {len(t['pew'])}; not joined {len(t['unjoined'])} "
          f"({', '.join(n for n, _ in t['unjoined'])}); built countries {len(t['counts'])}")
    print(f"\n{'node':16s} {'est':>4s} {'shown':>5s} {'unbuilt':>7s}  {'world':>13s}  source")
    for node, w in js["world"].items():
        shown = w["shown"]
        unbuilt = sum(1 for cc in shown if cc not in t["counts"])
        fig = approx(w["n"]) if "n" in w else f"{approx(w['lo'])}-{approx(w['hi'])}"
        print(f"{node:16s} {w['countries']:4d} {len(shown):5d} {unbuilt:7d}  {fig:>13s}  "
              + ("Pew 2020" if w["pew"] else "hand rows"))
    if t["pew_world"]:
        print("\nPew's own world row, for the check:")
        for fam, node in PEW_FAMILIES.items():
            if node:
                print(f"  {node:14s} {approx(t['pew_world'][fam])}")
    print("\nhand rows:")
    for (cc, node), r in sorted(t["rows"].items()):
        if r["src"] == "hand" or r.get("less"):
            state = "shown" if r["shown"] else f"not shown: {r['why'] or 'under the floor'}"
            print(f"  {cc} {node:12s} {approx(r['lo'])}-{approx(r['hi'])}  {state}"
                  + (f"  (less {', '.join(r['less'])})" if r.get("less") else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    t = build()
    js = to_json(t)
    report(t, js)
    geo, missing = shapes(t, js["countries"])
    if missing:
        sys.exit(f"!! no outline for {missing}: not in country_shapes.geojson or Natural Earth")
    if a.dry:
        print(f"\n--dry: would write {len(js['countries'])} countries and {len(geo['features'])} "
              "outlines")
        return 0
    OUT.write_text(json.dumps(js, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    OUT_SHAPES.write_text(json.dumps(geo, separators=(",", ":")), encoding="utf-8")
    print(f"\nwrote {OUT.name} ({OUT.stat().st_size / 1024:.0f} KB, {len(js['countries'])} "
          f"countries) and {OUT_SHAPES.name} ({OUT_SHAPES.stat().st_size / 1024:.0f} KB, "
          f"{len(geo['features'])} outlines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
