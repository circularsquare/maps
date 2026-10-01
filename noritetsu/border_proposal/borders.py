"""Where a railway crosses a national border: the points every country's build cuts at.

    python borders.py --fetch        # ERA RINF: every border point (op-type 90), two SPARQLs
    python borders.py                # how many points, by country pair

WHY A SHARED TABLE.  A route over a border is built in each country from its own extract, and
neither extract has the station on the far side, so the section between the last station on
one side and the first on the other belonged to neither build (TER K80: Mouscron and Tourcoing,
no line between). Each country now keeps its own side of that section, from its last station
to the border point (build_model.border_tails), and the two halves join in the app because
both builds name the border point with ONE id. That only works if both builds cut at the same
point, so the point comes from a table both read, not from geometry each computes.

RINF's border points are that table for Europe: one per line crossing, the same uopid in both
countries' registers (Mouscron-Frontière is EU00084 in Belgium's RINF and France's), and on
the track (median 2 m from OSM's rails, 90% within 9 m, over 415 measured crossings). A RINF
register's junction there is already "e" + uopid, so the OSM half ends at the register's own
junction station. Country outlines were tried and are far too coarse for this: Natural Earth
10m and religiondots' shapes put the border a median 600 m from RINF's points, up to 5.5 km.

EXTRA holds crossings RINF lacks (trams, Basel's German lines, borders outside the EU). Ids
there are "x" + a short name, never a uopid. A route end that meets no point is logged by
build_model ("no border point") and not drawn, which is how every crossing was before.

The table is border_points.json beside this file, tracked: [{id, name: {cc: name}, lon, lat,
countries: [cc, ...]}]. `countries` comes from which countries' sections of line end at the
point; a point only one country files (83 of 227) gets just that one.
"""
import json
import math
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TABLE = ROOT / "border_points.json"
ENDPOINT = "https://graph.data.era.europa.eu/repositories/rinf-plus"
USER_AGENT = "noritetsu-rail-map/0.1 (personal rail map research; python-urllib)"

ISO3 = {"AUT": "at", "BEL": "be", "BGR": "bg", "CHE": "ch", "CZE": "cz", "DEU": "de",
        "DNK": "dk", "EST": "ee", "ESP": "es", "FIN": "fi", "FRA": "fr", "GRC": "gr",
        "HRV": "hr", "HUN": "hu", "IRL": "ie", "ITA": "it", "LIE": "li", "LTU": "lt", "LUX": "lu",
        "LVA": "lv", "NLD": "nl", "NOR": "no", "POL": "pl", "PRT": "pt", "ROU": "ro",
        "SWE": "se", "SVN": "si", "SVK": "sk"}

# (id, name, lon, lat, [countries]): crossings RINF has no point for. Empty to start; filled
# from the build logs' "no border point" lines where a crossing is worth drawing.
EXTRA = []

# A border point this close to a route's track is where that track crosses.
NEAR_M = 60

Q_POINTS = """
PREFIX era: <http://data.europa.eu/949/>
PREFIX wgs: <http://www.w3.org/2003/01/geo/wgs84_pos#>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
SELECT DISTINCT ?op ?uopid ?name ?lat ?lon ?wkt ?country WHERE {
  ?op a era:OperationalPoint ; era:opType <http://data.europa.eu/949/concepts/op-types/90> .
  OPTIONAL { ?op era:uopid ?uopid }
  OPTIONAL { ?op era:opName ?name }
  OPTIONAL { ?op era:inCountry ?country }
  OPTIONAL { ?op era:netReference ?nr . ?nr wgs:lat ?lat ; wgs:long ?lon }
  OPTIONAL { ?op geo:hasGeometry ?g . ?g geo:asWKT ?wkt }
} ORDER BY ?op
"""
Q_PAIRS = """
PREFIX era: <http://data.europa.eu/949/>
SELECT DISTINCT ?uopid ?country WHERE {
  ?op a era:OperationalPoint ; era:opType <http://data.europa.eu/949/concepts/op-types/90> ;
      era:uopid ?uopid .
  ?sol a era:SectionOfLine ; era:inCountry ?country .
  { ?sol era:opStart ?op } UNION { ?sol era:opEnd ?op }
} ORDER BY ?uopid ?country
"""


def sparql(query):
    body = urllib.parse.urlencode({"query": query}).encode()
    req = urllib.request.Request(ENDPOINT, data=body, headers={
        "Accept": "application/sparql-results+json", "User-Agent": USER_AGENT,
        "Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    return [{v: b[v]["value"] for v in b} for b in d["results"]["bindings"]]


def cc_of(iri):
    t = (iri or "").rsplit("/", 1)[-1]
    return ISO3.get(t, t.lower())


def fetch():
    import re
    pairs = defaultdict(set)
    for r in sparql(Q_PAIRS):
        pairs[r["uopid"]].add(cc_of(r["country"]))
    out = {}
    for r in sparql(Q_POINTS):
        u = r.get("uopid")
        if not u:
            continue
        try:
            lon, lat = float(r["lon"]), float(r["lat"])
        except (KeyError, ValueError):
            m = re.search(r"POINT\s*\(\s*([-+\d.]+)\s+([-+\d.]+)", r.get("wkt", ""))
            if not m:
                continue
            lon, lat = float(m.group(1)), float(m.group(2))
        p = out.setdefault(u, {"id": f"e{u}", "name": {}, "lon": round(lon, 6),
                               "lat": round(lat, 6), "countries": set(pairs.get(u, ()))})
        if r.get("country"):
            p["countries"].add(cc_of(r["country"]))
            if r.get("name"):
                p["name"].setdefault(cc_of(r["country"]), r["name"])
    rows = [{**p, "countries": sorted(p["countries"])} for _u, p in sorted(out.items())]
    TABLE.write_text(json.dumps({"source": ENDPOINT, "fetched": date.today().isoformat(),
                                 "points": rows}, ensure_ascii=False, indent=0),
                     encoding="utf-8")
    print(f"wrote {len(rows)} border points to {TABLE}")


def load():
    """[{id, name, lon, lat, countries}], id being the station id every build gives it."""
    rows = json.loads(TABLE.read_text(encoding="utf-8"))["points"]
    pts = [{**r, "countries": set(r["countries"])} for r in rows]
    for pid, name, lon, lat, ccs in EXTRA:
        pts.append({"id": pid, "name": {"": name}, "lon": lon, "lat": lat, "countries": set(ccs)})
    return pts


def name_for(p, cc):
    return p["name"].get(cc) or next(iter(p["name"].values()), "") or p["id"]


class Index:
    """Border points by 0.02-degree cell, for "which points lie on this polyline"."""

    def __init__(self, points):
        self.pts = points
        self.cell = defaultdict(list)
        for i, p in enumerate(points):
            self.cell[(int(p["lon"] * 50), int(p["lat"] * 50))].append(i)

    def along(self, xy, near_m=NEAR_M):
        """Points within near_m of the polyline xy (rows of lon, lat), nearest-along first:
        (metres along xy to the foot of the perpendicular, point, segment index, foot lon,
        foot lat, metres off the track)."""
        if len(xy) < 2:
            return []
        cand = set()
        for x, y in xy:
            cx, cy = int(x * 50), int(y * 50)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cand.update(self.cell.get((cx + dx, cy + dy), ()))
        if not cand:
            return []
        kx = math.cos(math.radians(float(xy[0][1]))) * 111320
        ky = 110570
        cum = [0.0]
        for (x1, y1), (x2, y2) in zip(xy[:-1], xy[1:]):
            cum.append(cum[-1] + math.hypot((x2 - x1) * kx, (y2 - y1) * ky))
        out = []
        for i in cand:
            p = self.pts[i]
            best = None
            for k, ((x1, y1), (x2, y2)) in enumerate(zip(xy[:-1], xy[1:])):
                ax, ay = (x2 - x1) * kx, (y2 - y1) * ky
                px, py = (p["lon"] - x1) * kx, (p["lat"] - y1) * ky
                L2 = ax * ax + ay * ay
                t = 0.0 if L2 == 0 else max(0.0, min(1.0, (px * ax + py * ay) / L2))
                d = math.hypot(px - t * ax, py - t * ay)
                if best is None or d < best[0]:
                    best = (d, k, t)
            d, k, t = best
            if d <= near_m:
                (x1, y1), (x2, y2) = xy[k], xy[k + 1]
                out.append((cum[k] + t * (cum[k + 1] - cum[k]), p, k,
                            x1 + t * (x2 - x1), y1 + t * (y2 - y1), d))
        return sorted(out, key=lambda o: o[0])


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if "--fetch" in sys.argv:
        fetch()
    pts = load()
    by = defaultdict(int)
    for p in pts:
        by["-".join(sorted(p["countries"]))] += 1
    print(f"{len(pts)} border points: " + ", ".join(f"{k} {v}" for k, v in sorted(by.items())))
