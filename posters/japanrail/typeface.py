"""
Register Nunito with matplotlib, matching the interactive map.

The static weights live in posters/ancestrydots/ (cut there by its
make_fonts.py from the variable Nunito.ttf, whose default instance is
ExtraLight — pointed at the variable file, matplotlib renders hairline-thin).
This loads them from there rather than keeping a second copy; if that ever
moves, add the new directory to CANDIDATES.

Latin only. Japanese station and line names need a CJK face, which is a
separate problem — see posters/NOTES.md.
"""

from pathlib import Path

from matplotlib import font_manager

HERE = Path(__file__).parent
CANDIDATES = [HERE, HERE.parent / "ancestrydots"]
FILES = ["Nunito-Regular.ttf", "Nunito-Bold.ttf"]

REGULAR = BOLD = None
_found = []
for _dir in CANDIDATES:
    if all((_dir / f).exists() for f in FILES):
        for _f in FILES:
            font_manager.fontManager.addfont(str(_dir / _f))
            _found.append(_dir / _f)
        REGULAR, BOLD = _dir / FILES[0], _dir / FILES[1]
        break

FAMILY = "Nunito" if REGULAR else "DejaVu Sans"

if not REGULAR:
    print("!! Nunito statics not found in " +
          ", ".join(str(d) for d in CANDIDATES) +
          " — falling back to DejaVu Sans")

# ── Japanese ───────────────────────────────────────────────────────────────
# Nunito has no CJK, so the Japanese edition needs a second face. Preference
# order: a Noto Sans JP dropped into this directory (SIL OFL, so it is the
# licence-clean choice if these are ever sold or redistributed), then Yu
# Gothic, which ships with Windows and is a real Japanese face with correct
# glyph forms. MS Gothic is the last resort — it is a 1990s screen font and
# looks it at print size.
#
# .ttc files are collections; the index picks the face inside. Yu Gothic's
# index 0 is "Yu Gothic", index 1 is the narrower "Yu Gothic UI".
_WIN = Path("C:/Windows/Fonts")
JP_CANDIDATES = [
    (HERE / "NotoSansJP-Regular.ttf", HERE / "NotoSansJP-Bold.ttf", 0),
    (_WIN / "YuGothR.ttc", _WIN / "YuGothB.ttc", 0),
    (_WIN / "msgothic.ttc", _WIN / "msgothic.ttc", 0),
]

JP_REGULAR = JP_BOLD = None
JP_INDEX = 0
for _reg, _bold, _idx in JP_CANDIDATES:
    if _reg.exists() and _bold.exists():
        JP_REGULAR, JP_BOLD, JP_INDEX = _reg, _bold, _idx
        break

JP_FAMILY = JP_REGULAR.stem if JP_REGULAR else None

if not JP_REGULAR:
    print("!! no Japanese face found — the --lang ja sheet will set as tofu. "
          "Drop NotoSansJP-Regular.ttf and NotoSansJP-Bold.ttf beside this "
          "file, or install Yu Gothic.")
