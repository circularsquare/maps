"""What is actually inside the MLIT railway dataset, before building anything on it.

    python probe_n02.py --zip data/raw/N02-24_GML.zip

国土数値情報 N02 is the Japanese government's own railway inventory. Unlike OpenStreetMap it
names the LINE on every section of track (N02_003 路線名) and its operator (N02_004 運営会社),
which is the thing OSM is missing in rural Japan: the San'in Main Line has no route relation
at all there, so the only thing calling at Matsue is a limited express.
"""
import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent

FIELD = {
    "N02_001": "railway class",
    "N02_002": "operator class",
    "N02_003": "line name",
    "N02_004": "operating company",
    "N02_005": "station name",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--line", default=None, help="report on one line name")
    args = ap.parse_args()
    z = Path(args.zip)
    if not z.is_absolute():
        z = ROOT / z

    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
        print(f"{len(names)} members")
        for n in names:
            print(f"  {n}  {zf.getinfo(n).file_size/1e6:.1f} MB")
        geo = [n for n in names if n.lower().endswith(".geojson")]
        if not geo:
            print("\nno GeoJSON inside; this edition ships GML/shapefile only")
            return
        for n in geo:
            with zf.open(n) as f:
                data = json.load(f)
            feats = data.get("features", [])
            print(f"\n=== {n}: {len(feats)} features")
            if not feats:
                continue
            props = feats[0].get("properties", {})
            print("  fields: " + ", ".join(
                f"{k} ({FIELD.get(k, '?')})" for k in props))
            lines = Counter(f["properties"].get("N02_003") for f in feats)
            ops = Counter(f["properties"].get("N02_004") for f in feats)
            print(f"  {len(lines)} distinct line names, {len(ops)} operating companies")
            print("  biggest lines by section count")
            for k, c in lines.most_common(8):
                print(f"    {k:<28} {c}")
            for probe in ("山陰本線", "山手線", "東海道本線"):
                n_sec = lines.get(probe, 0)
                print(f"  {probe}: {n_sec} features")
            print("  geometry types: " + str(Counter(
                f["geometry"]["type"] for f in feats)))
            for code in ("N02_001", "N02_002"):
                c = Counter(f["properties"].get(code) for f in feats)
                print(f"  {code} ({FIELD.get(code)}):")
                for k, n in c.most_common():
                    example = next(f["properties"]["N02_003"] for f in feats
                                   if f["properties"].get(code) == k)
                    print(f"    {k}: {n:>6}   e.g. {example}")
            if "N02_005g" in props:
                groups = Counter(f["properties"].get("N02_005g") for f in feats)
                multi = [g for g, n in groups.items() if n > 1]
                print(f"  {len(groups)} station groups; {len(multi)} of them hold more than "
                      f"one record, i.e. are interchanges")
                big = groups.most_common(5)
                for g, n in big:
                    nm = next(f["properties"]["N02_005"] for f in feats
                              if f["properties"].get("N02_005g") == g)
                    print(f"    group {g}: {n} records  {nm}")

            if args.line:
                hits = [f for f in feats
                        if f["properties"].get("N02_003") == args.line]
                print(f"\n  {args.line}: {len(hits)} features")
                for f in hits[:5]:
                    print(f"    {f['properties']}")


if __name__ == "__main__":
    main()
