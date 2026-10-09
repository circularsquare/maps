"""T-056: MapLibre glyph tiles for the basemap's place names, in Zen Maru Gothic.

    ..\\venv\\Scripts\\python.exe pipeline\\glyphs.py

writes app/public/fonts/Zen Maru Gothic Regular,Noto Sans Regular/{start}-{end}.pbf, the font stack
basemap.ts gives every basemap label. Each file holds Zen Maru Gothic's glyphs for its 256
codepoints and, for codepoints Zen Maru lacks (Arabic, Devanagari, Thai...), OpenFreeMap's
Noto Sans Regular glyphs, the font the Positron style used before. MapLibre fetches one file per
stack and range, so the fallback has to live inside the file.

Left out, because MapLibre never fetches them:
- ranges whose every glyph is one MapLibre draws itself (kana, kanji, hangul and the other
  codepoints of its `localIdeographFontFamily` test, which are drawn by the browser);
- ranges with no glyph in either font. Missing, the file is a 404 and MapLibre draws the
  codepoint locally; an empty file would make it vanish.

Steps: download the font and its licence and OpenFreeMap's Noto ranges (cached in
data/raw/glyphs/, 39 MB), run `build_pbf_glyphs` (Stadia Maps' sdf_font_tools, Rust) to render
Zen Maru and combine it with Noto (Zen Maru first) in data/work/glyphs/, then copy the ranges
worth shipping. The tool: `cargo install build_pbf_glyphs --locked`; on Windows its bundled FreeType
build needs zlib's headers, so set CFLAGS=-I<cargo registry>/libz-sys-*/src/zlib first. Pass its
path with --tool if it is not on PATH. notes/T-056.md.
"""
import argparse
import json
import os
import pathlib
import shutil
import subprocess
import time
import urllib.parse
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "glyphs"
WORK = HERE / "data" / "work" / "glyphs"
OUT = HERE / "app" / "public" / "fonts"

FONT_URL = "https://github.com/google/fonts/raw/main/ofl/zenmarugothic/"
FONT_FILE = "ZenMaruGothic-Regular.ttf"
NOTO_OFL_URL = "https://github.com/google/fonts/raw/main/ofl/notosans/OFL.txt"
NOTO = "Noto Sans Regular"
NOTO_URL = "https://tiles.openfreemap.org/fonts/{stack}/{range}.pbf"
STACK = "Zen Maru Gothic Regular,Noto Sans Regular"
UA = {"User-Agent": "Mozilla/5.0 (glyph build)"}

# MapLibre GL 6's codePointUsesLocalIdeographFontFamily, the Basic Multilingual Plane part: these
# are drawn in the browser and never fetched while the map's localIdeographFontFamily is set
# (default "sans-serif").
LOCAL = [(0x02EA, 0x02EB), (0x1100, 0x11FF), (0x2E80, 0x2FDF), (0x3000, 0x30FF), (0x3105, 0x312F),
         (0x3131, 0x318E), (0x31A0, 0x4DBF), (0x4E00, 0xA48C), (0xA490, 0xA4C6), (0xA960, 0xA97C),
         (0xAC00, 0xD7C6), (0xD7CB, 0xD7FB), (0xF900, 0xFA6D), (0xFA70, 0xFAD9), (0xFE10, 0xFE1F),
         (0xFE30, 0xFE4F), (0xFF00, 0xFFEF)]


def local(cp):
    return any(a <= cp <= b for a, b in LOCAL)


def fetch(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        data = r.read()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    time.sleep(0.05)


def varint(b, i):
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return r, i


def fields(b):
    """(field number, value) of one protobuf message; values of length-delimited fields as bytes."""
    i = 0
    while i < len(b):
        key, i = varint(b, i)
        wire = key & 7
        if wire == 0:
            v, i = varint(b, i)
        elif wire == 2:
            n, i = varint(b, i)
            v, i = b[i:i + n], i + n
        elif wire == 5:
            v, i = b[i:i + 4], i + 4
        elif wire == 1:
            v, i = b[i:i + 8], i + 8
        else:
            raise ValueError(f"wire type {wire}")
        yield key >> 3, v


def glyph_ids(path):
    """Codepoints in a glyph PBF (glyphs.stacks[].glyphs[].id)."""
    ids = []
    for f, stack in fields(path.read_bytes()):
        if f == 1:
            ids += [next(v for k, v in fields(g) if k == 1) for k, g in fields(stack) if k == 3]
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tool", default=shutil.which("build_pbf_glyphs") or "build_pbf_glyphs")
    args = ap.parse_args()

    # 1. Inputs, cached.
    fetch(FONT_URL + FONT_FILE, RAW / "ttf" / FONT_FILE)
    fetch(FONT_URL + "OFL.txt", RAW / "OFL-ZenMaruGothic.txt")
    fetch(NOTO_OFL_URL, RAW / "OFL-NotoSans.txt")
    for r in range(256):
        rng = f"{r * 256}-{r * 256 + 255}"
        fetch(NOTO_URL.format(stack=urllib.parse.quote(NOTO), range=rng), RAW / "noto" / NOTO / f"{rng}.pbf")

    # 2. Render Zen Maru and combine it with Noto, Zen Maru first.
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    shutil.copytree(RAW / "noto" / NOTO, WORK / NOTO)
    combos = WORK.parent / "glyph_combinations.json"
    combos.write_text(json.dumps({STACK: ["ZenMaruGothic-Regular", NOTO]}), encoding="utf-8")
    env = {**os.environ, "RAYON_NUM_THREADS": "2"}
    subprocess.run([args.tool, "-c", str(combos), str(RAW / "ttf"), str(WORK)], check=True, env=env)

    # 3. Ship the ranges MapLibre can ask for.
    out = OUT / STACK
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shipped = size = zen = 0
    zen_ids = {}
    for p in (WORK / "ZenMaruGothic-Regular").glob("*.pbf"):
        zen_ids[p.name] = set(glyph_ids(p))
    for r in range(256):
        name = f"{r * 256}-{r * 256 + 255}.pbf"
        src = WORK / STACK / name
        ids = glyph_ids(src)
        if not ids or all(local(cp) for cp in ids):
            continue
        shutil.copyfile(src, out / name)
        shipped += 1
        size += src.stat().st_size
        zen += bool(zen_ids.get(name, set()) - {cp for cp in zen_ids.get(name, set()) if local(cp)})
    shutil.copyfile(RAW / "OFL-ZenMaruGothic.txt", OUT / "OFL-ZenMaruGothic.txt")
    shutil.copyfile(RAW / "OFL-NotoSans.txt", OUT / "OFL-NotoSans.txt")
    shutil.rmtree(WORK)     # 75 MB of intermediate ranges
    print(f"{out}: {shipped} ranges ({zen} with Zen Maru glyphs), {size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
