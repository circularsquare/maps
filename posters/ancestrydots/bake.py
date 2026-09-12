"""
Bake the ancestrydots GeoJSON into a compact binary for fast poster re-renders.

The web pipeline's output (data/processed/dots_all_1per100.geojson, 662 MB US +
72 MB Canada) is far too slow to re-parse on every render iteration. This reads
it once and writes lon/lat/label-index arrays instead.

The GeoJSON is one enormous single line, so this scans it with a regex over
fixed-size chunks rather than parsing JSON.

    python bake.py

Output: build/dots_na.npz  (lon f32, lat f32, idx u16) + build/palette.json
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent.parent / "ancestrydots"
BUILD = HERE / "build"

LEGEND = ROOT / "combined" / "data" / "processed" / "legend.json"
SOURCES = [
    # NB: the US "--all" run already includes Puerto Rico (FIPS 72), so
    # dots_72_1per100.geojson must NOT be added here — it would double it.
    ("US", ROOT / "data" / "processed" / "dots_all_1per100.geojson"),
    ("CA", ROOT / "canada" / "data" / "processed" / "dots_all_1per100.geojson"),
]

# "coordinates": [-66.71, 18.21]}, "properties": {"label": "Puerto Rican"
FEATURE = re.compile(
    rb'\[(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\]\}[^{]*\{"label":\s*"([^"]*)"'
)
CHUNK = 64 << 20


def load_palette():
    entries = json.loads(LEGEND.read_text(encoding="utf-8"))
    index: dict[bytes, int] = {}
    for i, e in enumerate(entries):
        lab = e["label"]
        # the GeoJSON was written with ensure_ascii=True, so accented labels
        # appear as literal é escapes; key on both forms
        index[lab.encode("utf-8")] = i
        index[json.dumps(lab)[1:-1].encode("ascii", "backslashreplace")] = i
    print(f"legend: {len(entries)} labels from {LEGEND.relative_to(ROOT.parent)}")
    return entries, index


def scan(path: Path, index: dict[bytes, int], unknown: dict[bytes, int]):
    """Stream one GeoJSON, yielding (lon, lat, idx) arrays per chunk."""
    size = path.stat().st_size
    done = 0
    carry = b""
    t0 = time.time()

    with path.open("rb") as fh:
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            done += len(block)
            buf = carry + block

            lons, lats, idxs = [], [], []
            end = 0
            for m in FEATURE.finditer(buf):
                lons.append(m.group(1))
                lats.append(m.group(2))
                label = m.group(3)
                i = index.get(label)
                if i is None:
                    unknown[label] = unknown.get(label, 0) + 1
                    i = 0xFFFF
                idxs.append(i)
                end = m.end()

            # keep the unconsumed tail so a feature split across the chunk
            # boundary is still matched next time round
            carry = buf[end:] if end else buf[-4096:]

            if lons:
                # bytes -> float has to go via a fixed-width 'S' array;
                # numpy won't cast bytes straight to float32
                yield (np.array(lons, dtype="S24").astype(np.float32),
                       np.array(lats, dtype="S24").astype(np.float32),
                       np.array(idxs, dtype=np.uint16))

            pct = 100 * done / size
            rate = done / (1 << 20) / max(time.time() - t0, 1e-9)
            print(f"\r  {path.name[:34]:34s} {pct:5.1f}%  {rate:5.0f} MB/s",
                  end="", flush=True)
    print()


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    entries, index = load_palette()
    unknown: dict[bytes, int] = {}

    all_lon, all_lat, all_idx, all_src = [], [], [], []
    for si, (tag, path) in enumerate(SOURCES):
        if not path.exists():
            print(f"  !! missing {path} — skipping {tag}")
            continue
        print(f"{tag}: {path.stat().st_size / (1 << 20):.0f} MB")
        n_before = sum(len(a) for a in all_lon)
        for lon, lat, idx in scan(path, index, unknown):
            all_lon.append(lon)
            all_lat.append(lat)
            all_idx.append(idx)
        n = sum(len(a) for a in all_lon) - n_before
        all_src.append(np.full(n, si, dtype=np.uint8))
        print(f"  -> {n:,} dots")

    lon = np.concatenate(all_lon)
    lat = np.concatenate(all_lat)
    idx = np.concatenate(all_idx)
    src = np.concatenate(all_src)

    if unknown:
        print("\n!! labels not in the combined legend (rendered as index 65535):")
        for lab, n in sorted(unknown.items(), key=lambda kv: -kv[1])[:20]:
            print(f"     {lab.decode('utf-8', 'replace'):40s} {n:,}")

    # the US "--all" run is 50 states, so PR should only come from dots_72.
    # verify that, because a silent overlap would double Puerto Rico's density.
    pr_box = (lon > -67.5) & (lon < -65.0) & (lat > 17.8) & (lat < 18.6)
    per_src = [(SOURCES[s][0], int(((src == s) & pr_box).sum()))
               for s in np.unique(src[pr_box])]
    print("\nPuerto Rico bbox, dots by source: " +
          ", ".join(f"{t}={n:,}" for t, n in per_src))
    if len(per_src) > 1:
        print("  !! PR appears in more than one source — overlapping, de-dupe needed")

    out = BUILD / "dots_na.npz"
    np.savez(out, lon=lon, lat=lat, idx=idx, src=src)
    (BUILD / "palette.json").write_text(
        json.dumps(entries, ensure_ascii=False), encoding="utf-8")

    print(f"\ntotal {len(lon):,} dots = {len(lon) * 100 / 1e6:.1f}M people")
    print(f"wrote {out}  ({out.stat().st_size / (1 << 20):.0f} MB)")
    print(f"      {BUILD / 'palette.json'}")


if __name__ == "__main__":
    sys.exit(main())
