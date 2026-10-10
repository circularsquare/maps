"""Download Kontur population extracts (r8 hexes, 2023-11-01) that religiondots does not hold.

    python sources/kontur_fetch.py US GB CA AU IE

-> data/raw/kontur/kontur_population_<CC>_20231101.gpkg (the .gz is removed once unpacked).
religiondots' copies (religiondots/data/geo/kontur/) are read in place where they exist;
kontur_path() says which one to use.
"""
import gzip
import os
import shutil
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "data" / "raw" / "kontur"
RD = HERE.parent / "religiondots" / "data" / "geo" / "kontur"
BASE = "https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"


def name(cc):
    return f"kontur_population_{cc.upper()}_20231101.gpkg"


def kontur_path(cc):
    """The unpacked extract for a country: religiondots' if it has one, else ours."""
    for d in (RD, OUT):
        if (d / name(cc)).exists():
            return d / name(cc)
    raise SystemExit(f"no Kontur extract for {cc}; run python sources/kontur_fetch.py {cc.upper()}")


def fetch(cc):
    gpkg = OUT / name(cc)
    if gpkg.exists() or (RD / name(cc)).exists():
        print(f"{cc}: already here")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    gz = OUT / (name(cc) + ".gz")
    tmp = gz.with_suffix(".gz.part")
    print(f"{cc}: downloading…", flush=True)
    with urllib.request.urlopen(BASE + gz.name, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    os.replace(tmp, gz)
    part = gpkg.with_suffix(".gpkg.part")
    with gzip.open(gz, "rb") as src, open(part, "wb") as dst:
        shutil.copyfileobj(src, dst, 1 << 20)
    os.replace(part, gpkg)
    gz.unlink()
    print(f"{cc}: {gpkg.stat().st_size / 1e6:,.0f} MB unpacked", flush=True)


if __name__ == "__main__":
    for cc in sys.argv[1:]:
        fetch(cc)
