"""Before/after check of register lines across a rebuild: name, English name, operator, km.

    python tools/compare_lines.py save be nl at     # before the change
    python tools/rebuild.py be nl at
    python tools/compare_lines.py diff be nl at     # what moved

A shared-file change meant for one country should leave every other country's register lines
identical; this is how each one was checked on 2026-09-30 and 2026-10-01. The snapshot is
data/logs/lines_snapshot.json (gitignored with the rest of data/).
"""
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "data" / "logs" / "lines_snapshot.json"


def current(cc):
    lines = json.loads((ROOT / "dist" / "data" / cc / "lines.json").read_text(
        encoding="utf-8"))["lines"]
    return {l["id"]: [l["name"], l.get("name_en", ""), l.get("operator", ""), round(l["km"], 2)]
            for l in lines if l.get("src") != "osm"}


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("save", "diff"):
        sys.exit(__doc__)
    mode, regions = sys.argv[1], sys.argv[2:]
    if mode == "save":
        old = json.loads(SNAP.read_text(encoding="utf-8")) if SNAP.exists() else {}
        old.update({cc: current(cc) for cc in regions})
        SNAP.parent.mkdir(parents=True, exist_ok=True)
        SNAP.write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
        print("saved", ", ".join(regions))
        return
    old = json.loads(SNAP.read_text(encoding="utf-8"))
    for cc in regions:
        a, b = old.get(cc, {}), current(cc)
        diff = [(k, a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
        print(f"{cc}: {len(a)} -> {len(b)} register lines, {len(diff)} differ")
        for d in diff[:20]:
            print("   ", d)


if __name__ == "__main__":
    main()
