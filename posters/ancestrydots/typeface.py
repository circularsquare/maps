"""
Register Nunito with matplotlib so the poster matches the interactive map.

Nunito.ttf from Google Fonts is a variable font whose default instance is
ExtraLight, and matplotlib does not set variable axes — so pointing it at the
variable file directly would silently render everything hairline-thin. Run
make_fonts.py to cut the static Regular and Bold, which is what this loads.
"""

from pathlib import Path

from matplotlib import font_manager

HERE = Path(__file__).parent
FILES = ["Nunito-Regular.ttf", "Nunito-Bold.ttf"]

_found = []
for _f in FILES:
    _p = HERE / _f
    if _p.exists():
        font_manager.fontManager.addfont(str(_p))
        _found.append(_p)

FAMILY = "Nunito" if len(_found) == len(FILES) else "DejaVu Sans"
REGULAR = HERE / "Nunito-Regular.ttf"
BOLD = HERE / "Nunito-Bold.ttf"

if FAMILY != "Nunito":
    print("!! Nunito statics missing — falling back to DejaVu Sans. "
          "Run make_fonts.py (and download Nunito.ttf first).")
