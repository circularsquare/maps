"""Slovakia: Štatistický úrad SR, Sčítanie obyvateľov, domov a bytov 2021 (SODB 2021), mother tongue
(materinský jazyk) by obec, all 28 published categories.

    python sources/sk_sodb.py --fetch    download the census portal's JSON and the GIS layer (~25 MB)
    python sources/sk_sodb.py            normalise from data/raw/sk/

-> data/normalized/sk.csv (levels `country`, `kraj`, `obec`; alternatives, never summed across levels)

THE QUESTION. One mother tongue per person, "the language spoken at home in childhood"; no second
answer. The census also asked the language used most at home and in public, which is not used here.

TWO PUBLICATIONS OF THE SAME TABLE, ONE THE SOURCE AND ONE THE CHECK.
  * scitanie.sk's results browser (indicator Z01/14, "Structure of population by mother tongue")
    reads static JSON files, one per area and level:
        https://www.scitanie.sk/themes/web-sodb/assets/public/disem/data/Z01_14_<t>_<spec>_<unit>.json
    `KR_<kraj>_OB` is every obec of a kraj with all 28 categories (27 languages, other, not
    ascertained), down to a count of 1. Eight files cover the 2,927 obce. `SR_SK0_SR` is the
    national row and `SR_SK0_KR` the 8 kraje. THIS IS THE SOURCE.
  * gis.scitanie.sk, the census's public ArcGIS server, layer `obyv_ekchar_matjaz_vekskup/4`, the
    same table folded to 10 languages plus `ostatné`, on the same LAU codes (`uzemie`) the
    religiondots build keys Slovakia's 1 km grid by. THE CHECK: every obec must agree on all 10
    languages and the total, and `ostatné` must equal the portal's other 18 categories summed.
  The GIS server also carries a nationality x mother tongue cross-table (`obyv_ekchar_nar_matjaz`),
  15 categories; it is 12,569 people short of the census (small cells dropped), so it is not used.

Both hosts need certifi's CA bundle; the Windows store fails (religiondots/sources/sk.py says why).
The GIS layer's maxRecordCount is 2,000 against 2,927 obce: paged, and the count asserted.

CHECKS (all asserted): 2,927 distinct obec codes, the same set in both publications; each obec's
28 categories sum to its total; the obce summed per category equal the national file and the
kraj file; the national total is 5,449,270; per obec the GIS layer agrees on all 10 languages and
the total, and its `ostatné` equals the other 18 categories exactly.
"""
import json
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "sk"
OUT = ROOT / "data" / "normalized" / "sk.csv"

PORTAL = "https://www.scitanie.sk/themes/web-sodb/assets/public/disem/data/Z01_14_{}.json?v=10"
KRAJE = ["SK010", "SK021", "SK022", "SK023", "SK031", "SK032", "SK041", "SK042"]
GIS = ("https://gis.scitanie.sk/server/rest/services/Hosted/obyv_ekchar_matjaz_vekskup/"
       "FeatureServer/4")
GIS_JSON = RAW / "sk_gis_obec_mother_tongue.json"
# GIS field -> portal code, for the 10 languages both print
GIS_CODES = {"cv_1": "01", "cv_2": "02", "cv_3": "03", "cv_4": "04", "cv_5": "05", "cv_6": "06",
             "cv_7": "07", "cv_8": "08", "cv_11": "11", "cv_14": "14"}
GIS_OTHER = "cv_50"

NATIONAL = 5_449_270
N_OBCE = 2_927
N_CATS = 28                      # 27 languages + "iný" (other) + "nezistený" (not ascertained)
NOT_STATED = "00"
SOURCE_ID = "sk_sodb2021_z01_14"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0"


def _get(url, timeout=300):
    import certifi
    ctx = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        body = r.read().strip()      # the portal's files start with whitespace
    if body[:1] != b"{":     # a 200 is not a download
        raise SystemExit(f"not JSON from {url}: {body[:120]!r}")
    d = json.loads(body)
    if "error" in d:
        raise SystemExit(f"error from {url}: {d['error']}")
    return d


def _portal_name(spec):
    return f"sk_z01_14_{spec}.json"


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    specs = ["SR_SK0_SR", "SR_SK0_KR"] + [f"KR_{k}_OB" for k in KRAJE]
    for spec in specs:
        p = RAW / _portal_name(spec)
        if p.exists() and p.stat().st_size > 1000:
            continue
        d = _get(PORTAL.format(spec))
        # keep the table only: the OB files also carry a map block nobody reads
        p.write_text(json.dumps({"meta": d["meta"], "table": d["table"]}, ensure_ascii=False),
                     encoding="utf-8")
        print(f"  {p.name}: {len(d['table']['data'])} units")
        time.sleep(0.5)
    if not GIS_JSON.exists():
        feats, off = [], 0
        while True:
            q = urllib.parse.urlencode({
                "where": "1=1", "outFields": ",".join(["uzemie", "nazov", "spolu", GIS_OTHER]
                                                       + list(GIS_CODES)),
                "returnGeometry": "false", "orderByFields": "objectid",
                "resultOffset": off, "resultRecordCount": 2000, "f": "json"})
            got = _get(f"{GIS}/query?{q}")["features"]
            feats += got
            if len(got) < 2000:
                break
            off += len(got)
        n = _get(f"{GIS}/query?where=1%3D1&returnCountOnly=true&f=json")["count"]
        if len(feats) != n or n != N_OBCE:
            raise SystemExit(f"GIS layer: paged {len(feats)}, server says {n}, expected {N_OBCE}")
        # the aliases name the languages; if the office renumbers cv_*, stop
        meta = _get(f"{GIS}?f=json")
        alias = {f["name"]: f.get("alias") for f in meta["fields"]}
        want = {"cv_1": "slovenský", "cv_2": "maďarský", "cv_3": "rómsky", "cv_4": "rusínsky",
                "cv_5": "ukrajinský", "cv_6": "český", "cv_7": "nemecký", "cv_8": "poľský",
                "cv_11": "ruský", "cv_14": "anglický", "cv_50": "ostatné"}
        bad = {k: alias.get(k) for k, v in want.items() if alias.get(k) != v}
        if bad:
            raise SystemExit(f"GIS field aliases moved: {bad}")
        GIS_JSON.write_text(json.dumps([f["attributes"] for f in feats], ensure_ascii=False),
                            encoding="utf-8")
        print(f"  {GIS_JSON.name}: {len(feats)} obce")


def _load(spec):
    p = RAW / _portal_name(spec)
    if not p.exists():
        raise SystemExit(f"missing {p}: run with --fetch")
    return json.loads(p.read_text(encoding="utf-8"))["table"]


def _values(unit):
    return {k: int(v["value"]) for k, v in unit["types"].items()}


def main():
    if "--fetch" in sys.argv:
        fetch()
    nat_t = _load("SR_SK0_SR")
    # "other" (ostatné) is the pie chart's fold of the small ones, typed "chart": not a category
    names = {k: v["name"] for k, v in nat_t["names"].items()
             if k != "total" and v["type"] in ("table", "both")}
    if len(names) != N_CATS:
        raise SystemExit(f"{len(names)} categories in the national file, expected {N_CATS}")
    nat = _values(nat_t["data"]["SK0"])
    kraj_t = _load("SR_SK0_KR")

    rows, obce = [], {}
    for k in KRAJE:
        t = _load(f"KR_{k}_OB")
        if {c for c, v in t["names"].items() if v["type"] in ("table", "both")} - {"total"} != set(names):
            raise SystemExit(f"{k}: category codes differ from the national file")
        for code, u in t["data"].items():
            if code in obce:
                raise SystemExit(f"obec {code} in two kraj files")
            v = _values(u)
            parts = sum(v.get(c, 0) for c in names)
            if parts != v["total"]:
                raise SystemExit(f"{u['name']} ({code}): categories {parts:,} != total {v['total']:,}")
            obce[code] = (u["name"], v)
    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    say(len(obce) == N_OBCE, f"{len(obce):,} obce in the 8 kraj files (expected {N_OBCE:,}), "
        "each one's 28 categories summing to its total")
    summed = defaultdict(int)
    for _, v in obce.values():
        for c, n in v.items():
            summed[c] += n
    say(nat["total"] == NATIONAL, f"national total {nat['total']:,}")
    say(all(summed[c] == nat.get(c, 0) for c in list(names) + ["total"]),
        "obce summed per category = the national file, all 28 categories")
    ksum = defaultdict(int)
    for u in kraj_t["data"].values():
        for c, n in _values(u).items():
            ksum[c] += n
    say(len(kraj_t["data"]) == 8 and all(ksum[c] == nat.get(c, 0) for c in list(names) + ["total"]),
        "the 8 kraje summed per category = the national file")

    # the second witness: the GIS layer, per obec
    gis = json.loads(GIS_JSON.read_text(encoding="utf-8")) if GIS_JSON.exists() else None
    if gis is None:
        raise SystemExit(f"missing {GIS_JSON}: run with --fetch")
    g = {a["uzemie"]: a for a in gis}
    say(set(g) == set(obce), f"the GIS layer has the same {len(g):,} obec codes")
    bad_lang, bad_other = [], []
    rest = [c for c in names if c not in GIS_CODES.values()]
    for code, (name, v) in obce.items():
        a = g[code]
        if any(int(a[f] or 0) != v.get(c, 0) for f, c in GIS_CODES.items()) \
                or int(a["spolu"] or 0) != v["total"]:
            bad_lang.append(name)
        if int(a[GIS_OTHER] or 0) != sum(v.get(c, 0) for c in rest):
            bad_other.append(name)
    say(not bad_lang, f"per obec, the GIS layer agrees on all 10 languages and the total"
        + (f" (differs in {len(bad_lang)}: {bad_lang[:5]})" if bad_lang else ""))
    say(not bad_other, "per obec, the GIS layer's `ostatné` = the portal's other 18 categories"
        + (f" (differs in {len(bad_other)}: {bad_other[:5]})" if bad_other else ""))

    for code, (name, v) in obce.items():
        for c, label in names.items():
            rows.append((code, "obec", name, label, v.get(c, 0)))
        rows.append((code, "obec", name, "total", v["total"]))
    for code, u in kraj_t["data"].items():
        v = _values(u)
        for c, label in names.items():
            rows.append((code, "kraj", u["name"], label, v.get(c, 0)))
        rows.append((code, "kraj", u["name"], "total", v["total"]))
    for c, label in names.items():
        rows.append(("SK0", "country", "Slovenská republika", label, nat.get(c, 0)))
    rows.append(("SK0", "country", "Slovenská republika", "total", nat["total"]))

    print("\n  national, by category:")
    for c in sorted(names, key=lambda c: -nat.get(c, 0)):
        print(f"    {nat.get(c, 0):>10,}  {100 * nat.get(c, 0) / NATIONAL:6.2f}%  {names[c]}")
    if not ok:
        raise SystemExit("reconciliation FAILED")

    import pandas as pd
    df = pd.DataFrame(rows, columns=["geo_id", "geo_level", "geo_name", "source_category", "count"])
    df["tier"] = "measured"
    df["year"] = 2021
    df["source_id"] = SOURCE_ID
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, OUT)
    print(f"\nwrote {OUT} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
