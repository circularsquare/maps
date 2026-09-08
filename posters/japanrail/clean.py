"""
Delete build artefacts that are no longer the current sheet.

Every render is tagged, and iterating leaves a tag's worth of layers behind
each time — plus the one-off `--crop` tests, which are named after wherever
they were cropped. This keeps the baked basemap, the layout, the current tag
and the composed poster, and removes the rest.

    python clean.py            # show what would go
    python clean.py --yes      # actually delete

--keep-tag defaults to whatever compose.py is currently pointed at.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

BUILD = Path(__file__).parent / "build"
INSETS = BUILD / "insets"

# Never removed: the bake (expensive to redo), the plan, the composed sheet.
KEEP_NAMES = {
    "colors.json", "jp_water.gpkg", "jp_coast.gpkg", "jp_lakes.gpkg",
    "jp_pref.gpkg", "layout.json", "layout_plan.png",
    "view_poster_dark.png", "view_poster_light.png",
    "view_poster_dark_ja.png", "view_poster_light_ja.png",
}
KEEP_PREFIXES = ("poster_",)

# Files named <kind>_<tag>.ext, where the tag is what a render was called.
TAGGED = re.compile(
    r"^(base_dark|base_light|bubbles_dark|bubbles_light|bubbles|glow|lines|"
    r"preview_dark|preview_light|frame)_(.+?)\.(png|json)$")

# Inset files that belong to a city currently in insets.CITIES.
INSET_KEEP = re.compile(
    r"^(?P<city>[a-z]+)"
    r"(|\.json|_base_dark|_base_light|_bubbles_dark|_bubbles_light|_glow|"
    r"_lines|_dark|_light)\.(png|json)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-tag", default=None,
                    help="render tag to keep; defaults to compose.py's own")
    ap.add_argument("--yes", action="store_true", help="actually delete")
    args = ap.parse_args()

    tag = args.keep_tag
    if tag is None:
        src = (Path(__file__).parent / "compose.py").read_text(encoding="utf-8")
        tag = re.search(r'"--tag", default="([^"]+)"', src).group(1)
    from insets import CITIES

    doomed = []
    for p in sorted(BUILD.glob("*")):
        if p.is_dir() or p.name in KEEP_NAMES:
            continue
        if any(p.name.startswith(k) for k in KEEP_PREFIXES):
            continue
        m = TAGGED.match(p.name)
        if m and m.group(2) == tag:
            continue
        doomed.append(p)

    for p in sorted(INSETS.glob("*")):
        m = INSET_KEEP.match(p.name)
        if m and m.group("city") in CITIES:
            continue
        doomed.append(p)

    total = sum(p.stat().st_size for p in doomed)
    for p in doomed:
        print(f"  {p.relative_to(BUILD).as_posix():<44} "
              f"{p.stat().st_size / (1 << 20):7.2f} MB")
    print(f"{len(doomed)} files, {total / (1 << 20):.0f} MB  "
          f"(keeping tag {tag!r} and {', '.join(sorted(CITIES))})")
    if not args.yes:
        print("dry run — pass --yes to delete")
        return
    for p in doomed:
        p.unlink()
    print("deleted")


if __name__ == "__main__":
    main()
