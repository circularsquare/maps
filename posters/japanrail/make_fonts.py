"""
Cut static Noto Sans JP weights out of the variable font.

The Japanese editions were rendering with Yu Gothic, which ships with Windows.
Noto Sans JP is SIL OFL, so it is the licence-clean face if these sheets are
ever sold; typeface.py already prefers it and falls back to Yu Gothic.

`NotoSansJP[wght].ttf` from Google Fonts carries the same trap as Nunito, and
worse: its variable **default instance is Thin (wght=100)**. Neither matplotlib
nor PIL sets variable axes, so pointing either at the variable file renders
every Japanese glyph hairline-thin — and at 8-11 pt on paper that is invisible
rather than merely wrong. This instances the two weights compose.py asks for.

    curl -sSL -o NotoSansJP.ttf \\
      'https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP%5Bwght%5D.ttf'
    python make_fonts.py

Writes NotoSansJP-Regular.ttf and NotoSansJP-Bold.ttf beside the variable font,
which is where typeface.py's JP_CANDIDATES looks first.
"""

from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

HERE = Path(__file__).parent
SRC = HERE / "NotoSansJP.ttf"
WEIGHTS = {"Regular": 400, "Bold": 700}


def main():
    if not SRC.exists():
        raise SystemExit(
            f"missing {SRC}. Download it with:\n"
            "  curl -sSL -o NotoSansJP.ttf "
            "'https://github.com/google/fonts/raw/main/ofl/notosansjp/"
            "NotoSansJP%5Bwght%5D.ttf'")
    for name, wght in WEIGHTS.items():
        font = TTFont(SRC)
        inst = instantiateVariableFont(font, {"wght": wght}, updateFontNames=True)
        out = HERE / f"NotoSansJP-{name}.ttf"
        inst.save(out)
        print(f"wrote {out.name}  wght={wght}  "
              f"{out.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
