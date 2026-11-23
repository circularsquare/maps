"""Export the drawn map's dots as one zip for people who want the data (2026-11-12).

    python tools/export_dots.py            -> data/export/languagedots_dots.zip

The zip holds dots.csv (every dot and sub-dot mark, exactly what tiles.py draws), languages.csv
(the tree from taxonomy/languages.json), countries.csv (each country's source notes from
counts.json) and README.txt (tools/export_README.txt). Countries are the last build tail's list
(data/processed/build_tail.json), so the export matches the published archive.

Upload: rclone copyto data/export/languagedots_dots.zip r2:anitamaps/languagedots/download/languagedots_dots.zip --s3-no-check-bucket
"""
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
PROC = HERE / "data" / "processed"
OUT = HERE / "data" / "export" / "languagedots_dots.zip"


def marks(path: Path, people=None):
    for f in json.loads(path.read_text(encoding="utf-8"))["features"]:
        p = people or f["properties"]["count"]
        if p > 0:   # tiles.py drops 0-people rings too
            lon, lat = f["geometry"]["coordinates"]
            yield lon, lat, f["properties"]["n"], int(round(p))


def main():
    ccs = json.loads((PROC / "build_tail.json").read_text(encoding="utf-8"))["countries"]
    counts = json.loads((PROC / "counts.json").read_text(encoding="utf-8"))
    nodes = json.loads((HERE / "taxonomy" / "languages.json").read_text(encoding="utf-8"))["nodes"]
    known = {n["id"] for n in nodes}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".zip.part")
    used, total_people, total_rows = set(), 0, 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        with z.open("dots.csv", "w", force_zip64=True) as raw:
            w = io.TextIOWrapper(raw, encoding="utf-8", newline="")
            c = csv.writer(w)
            c.writerow(["lon", "lat", "country", "language", "people"])
            for cc in ccs:
                n0 = total_rows
                for name, people in ((f"dots_{cc}.geojson", 1000), (f"rings_{cc}.geojson", None)):
                    p = PROC / name
                    if not p.exists():
                        continue
                    for lon, lat, n, k in marks(p, people):
                        c.writerow([f"{lon:.4f}", f"{lat:.4f}", cc, n, k])
                        used.add(n)
                        total_people += k
                        total_rows += 1
                print(f"{cc}: {total_rows - n0:,} rows", flush=True)
            w.flush()
            w.detach()

        buf = io.StringIO()
        c = csv.writer(buf)
        c.writerow(["id", "label", "parent", "color"])
        for n in nodes:
            c.writerow([n["id"], n["label"], n.get("parent") or "", n.get("color", "")])
        z.writestr("languages.csv", buf.getvalue().encode("utf-8"))

        buf = io.StringIO()
        c = csv.writer(buf)
        c.writerow(["country", "name", "source", "how", "grain", "gap", "note"])
        for cc in ccs:
            e = counts["countries"].get(cc, {})
            c.writerow([cc, e.get("name", ""), e.get("source", ""), e.get("how", ""),
                        e.get("grain", ""), e.get("gap", ""), e.get("note_public", "")])
        z.writestr("countries.csv", buf.getvalue().encode("utf-8"))

        z.write(HERE / "tools" / "export_README.txt", "README.txt")
    tmp.replace(OUT)
    missing = sorted(used - known)
    print(f"{total_rows:,} rows, {total_people:,} people, {len(ccs)} countries")
    print(f"language ids not in languages.csv: {missing or 'none'}")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
