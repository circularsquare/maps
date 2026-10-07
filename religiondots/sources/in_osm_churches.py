"""Kerala: an independent witness for the Kerala Migration Survey's church cells, from OpenStreetMap.

WHY THIS EXISTS
---------------
Anita asked (2026-09-15, runlog "rulings") for Kerala's St Thomas Christians to be shown if a route
exists, "wanted, not forced". The only district source is K.C. Zachariah's CDS Working Paper 468, Table 6
(KMS 2008-2014), which in_split_christian.py already reads. Its Catholic total passes the diocesan rolls
and its three rites do not (sources/in.md §9); nothing had checked its Jacobite, Orthodox and Mar Thoma
cells, so they stay on `christianity`. This script is that check, run 2026-10-03 (sources/in.md §14).

THE WITNESS
-----------
Every Christian place of worship OpenStreetMap holds in Kerala (`amenity=place_of_worship` +
`religion=christian` inside the IN-KL boundary; 6,532 on 2026-10-03), each put in its 2011 district by
point-in-polygon on data/geo/in/in_subdistricts.gpkg (kod[:5] is the district code in_split_christian.KERALA
uses), and given a church from its name and `denomination` tag (classify()). Kerala's Syrian churches
nearly always carry the church in the name ("St. Mary's Orthodox Syrian Church", "... Jacobite Syrian
Church", "... Mar Thoma Church", "... Malankara Catholic Church"); Catholic parishes mostly do not, so
the rites are not tested here.

The test: for each church, the share of a district's Christian places of worship that are that church,
against Table 6's share of the district's Christians, as a Spearman rank correlation over the 14
districts. Parish sizes differ between churches, so levels are not compared, only order. Mapping effort
differs between districts, which moves every church in a district together and so drops out of a share.

WHAT IT FOUND (2026-10-03): Pentecost/Brethren rho +0.89; CSI +0.48; Mar Thoma +0.45; Jacobite +0.16;
Orthodox +0.12; Jacobite + Orthodox together +0.06; Syro-Malankara -0.10. All four churches with
"Malankara" in their formal names together (Jacobite, Orthodox, Mar Thoma, Syro-Malankara) +0.60. So the
Syrian churches replicate as a group and not one by one, which is the label confusion §9 found among the
Catholic rites, reaching across the Catholic line. Not drawn; the record is sources/in.md §14.

Usage:
    python sources/in_osm_churches.py --fetch    query Overpass, save data/raw/in/osm/kerala_churches.json
    python sources/in_osm_churches.py            classify, place, and print the witness; writes nothing
"""

import json
import os
import re
import sys
import urllib.parse
import urllib.request

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import in_split_christian as isc   # TABLE6, DENOMS, KERALA

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAW = os.path.join(ROOT, "data", "raw", "in", "osm", "kerala_churches.json")
GEO = os.path.join(ROOT, "data", "geo", "in", "in_subdistricts.gpkg")
QUERY = """
[out:json][timeout:300];
area["ISO3166-2"="IN-KL"]["admin_level"="4"]->.k;
(nwr["amenity"="place_of_worship"]["religion"="christian"](area.k););
out center tags;
"""
# No personal identity in the agent string ([[feedback_no_identity_in_requests]]).
UA = "religiondots-research/1.0 (dot map of religion; one-off query)"
ENDPOINTS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]


def fetch():
    for url in ENDPOINTS:
        try:
            req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": QUERY}).encode(),
                                         headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=360) as r:
                body = r.read()
            n = len(json.loads(body)["elements"])
        except Exception as e:                     # 504 from the main instance on 2026-10-03
            print(f"  {url}: {e}")
            continue
        os.makedirs(os.path.dirname(RAW), exist_ok=True)
        tmp = RAW + ".tmp"
        with open(tmp, "wb") as f:
            f.write(body)
        os.replace(tmp, RAW)
        print(f"  {url}: {n:,} places saved to {RAW}")
        return
    raise SystemExit("every Overpass endpoint failed")


def classify(name, den):
    """A church from the name fields and the `denomination` tag. Order matters: the Catholic and
    Jacobite tests come before the bare `orthodox` and `mar thoma` ones, because "Mar Thoma Sleeha" is a
    Syro-Malabar dedication and the Jacobites call themselves the Jacobite Syrian Orthodox Church."""
    n = re.sub(r"\s+", " ", re.sub(r"[\.\-_,]", " ", name.lower()))
    dd = den.lower().replace("-", "_").replace(" ", "_")
    catholic_word = "catholic" in n or "sleeha" in n or "forane" in n or "latin" in n
    if ("malankara catholic" in n or "syro malankara" in n or "malankara syrian catholic" in n
            or dd in ("malankara_catholic", "syro_malankara_catholic")):
        return "Syro-Malankara"
    if "syro malabar" in n or "syromalabar" in n or dd in ("syro_malabar_catholic", "syro_malabar"):
        return "Syro-Malabar"
    if "knanaya" in n and ("jacobite" in n or "syrian orthodox" in n or "jacobite" in dd):
        return "Jacobite"
    if "knanaya" in n and "catholic" in n:
        return "Syro-Malabar"
    if "latin" in n or dd in ("latin_catholic", "latin"):
        return "Latin Catholics"
    if ("jacobite" in n or "yakob" in n or "syriac orthodox" in n or "simhasana" in n
            or "jacobite" in dd or dd in ("syriac_orthodox", "syrian_jacobite")):
        return "Jacobite"
    if (("mar thoma" in n or "marthoma" in n) and not catholic_word
            and "orthodox" not in n and "jacobite" not in n):
        return "Mar Thoma"
    if dd in ("marthoma", "marthomite", "mar_thoma", "mar_thoma_syrian_church_of_malabar"):
        return "Mar Thoma"
    if ("orthodox" in n and not catholic_word) or dd in ("indian_orthodox", "malankara_orthodox"):
        return "Orthodox"
    if re.search(r"\bcsi\b", n) or "church of south india" in n or dd in ("csi", "church_of_south_india",
                                                                         "anglican"):
        return "CSI"
    if ("pentecost" in n or re.search(r"\bipc\b", n) or "assemblies of god" in n or "assembly of god" in n
            or "church of god" in n or "brethren" in n or "sharon" in n or dd in ("pentecostal", "brethren")):
        return "Pentecost"
    if catholic_word or dd in ("catholic", "roman_catholic"):
        return "Catholic, rite not named"
    return "unclassified"


def load():
    if not os.path.exists(RAW):
        raise SystemExit(f"missing {RAW}; run --fetch")
    rows = []
    for e in json.load(open(RAW, encoding="utf-8"))["elements"]:
        t = e["tags"]
        c = e if e["type"] == "node" else e.get("center")
        if not c:
            continue
        name = " ".join(t.get(k, "") for k in ("name", "name:en", "official_name", "alt_name", "operator"))
        rows.append((c["lon"], c["lat"], classify(name, t.get("denomination", ""))))
    df = pd.DataFrame(rows, columns=["lon", "lat", "church"])

    import pyogrio
    import shapely
    geo = pyogrio.read_dataframe(GEO)
    geo["kod"] = geo["kod"].astype(str)
    geo = geo[geo["kod"].str.startswith(isc.KERALA_STATE)].reset_index(drop=True)
    pts = shapely.points(df["lon"].values, df["lat"].values)
    pi, gi = shapely.STRtree(geo.geometry.values).query(pts, predicate="within")
    code = np.full(len(df), None, dtype=object)
    code[pi] = [k[:5] for k in geo["kod"].values[gi]]
    df["district"] = pd.Series(code).map(isc.KERALA)
    print(f"  {len(df):,} Christian places of worship, {df['district'].notna().sum():,} placed in a district")
    if set(df["district"].dropna()) != set(isc.KERALA.values()):
        raise SystemExit("a Kerala district holds no place at all; the district codes no longer match")
    return df


GROUPS = {
    "Jacobite": ["Jacobite"], "Orthodox": ["Orthodox"], "Mar Thoma": ["Mar Thoma"],
    "Syro-Malankara": ["Syro-Malankara"], "CSI": ["CSI"], "Pentecost": ["Pentecost"],
    "Jacobite + Orthodox": ["Jacobite", "Orthodox"],
    "Jacobite + Orthodox + Mar Thoma": ["Jacobite", "Orthodox", "Mar Thoma"],
    "the four Malankara churches": ["Jacobite", "Orthodox", "Mar Thoma", "Syro-Malankara"],
}


def witness(df):
    from scipy.stats import spearmanr
    order = [isc.KERALA[k] for k in sorted(isc.KERALA, reverse=True)]
    ct = pd.crosstab(df["district"], df["church"]).reindex(order).fillna(0)
    print("\n  places by district and church:")
    print(ct.astype(int).to_string())
    tot = ct.sum(axis=1)
    t6 = pd.DataFrame({k: v for k, v in isc.TABLE6.items() if k != "KERALA"}, index=isc.DENOMS).T.reindex(order)
    print("\n  share of the district's places (OSM) against share of its Christians (WP468 Table 6), %:")
    cells = {}
    for g, cols in GROUPS.items():
        osm = sum(ct.get(c, 0) for c in cols) / tot * 100
        kms = t6[cols].sum(axis=1)
        rho, p = spearmanr(osm, kms)
        cells[g] = pd.DataFrame({"osm": osm.round(1), "kms": kms.round(1)})
        print(f"    {g:<34} Spearman rho {rho:+.2f}  p {p:.3f}")
    for g in ("Jacobite + Orthodox", "Mar Thoma", "Syro-Malankara"):
        print(f"\n  {g}:")
        print(cells[g].to_string())


def main():
    if "--fetch" in sys.argv:
        fetch()
        return
    witness(load())


if __name__ == "__main__":
    main()
