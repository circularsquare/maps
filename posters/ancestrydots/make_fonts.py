"""
Cut static Nunito weights out of the variable font.

Nunito.ttf from Google Fonts is a variable font whose default instance is
ExtraLight — far too thin for print. matplotlib does not set variable axes, so
it would silently render everything in ExtraLight. This instances the two
weights the poster uses.

    python make_fonts.py

Writes Nunito-Regular.ttf and Nunito-Bold.ttf beside the variable font.
"""

from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

HERE = Path(__file__).parent
SRC = HERE / "Nunito.ttf"
WEIGHTS = {"Regular": 400, "Bold": 700}


def main():
    if not SRC.exists():
        raise SystemExit(
            f"missing {SRC}. Download it with:\n"
            "  curl -sSL -o Nunito.ttf "
            "'https://github.com/google/fonts/raw/main/ofl/nunito/"
            "Nunito%5Bwght%5D.ttf'")
    for name, wght in WEIGHTS.items():
        font = TTFont(SRC)
        inst = instantiateVariableFont(font, {"wght": wght}, updateFontNames=True)
        out = HERE / f"Nunito-{name}.ttf"
        inst.save(out)
        print(f"wrote {out.name}  wght={wght}  "
              f"{out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
