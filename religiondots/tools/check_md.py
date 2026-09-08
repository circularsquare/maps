"""spec §7d: every markdown marker in a country note must turn into markup, not into text.

The notes in `countries.py` are authored in a markdown-ish voice because that is what a
docstring-shaped constant looks like. Nothing converted the markers until 2026-09-07, so 531
bold runs, 80 italics and 43 code spans were on screen in the about panel as raw asterisks and
backticks — France alone showed 106. This is the check that it stays fixed.

It holds a COPY of the viewer's `md()` regexes rather than importing them, because they live in
a <script> block in index.html and there is nothing to import. That copy is the thing to keep
in step: if the three lines below and the three in `md()` ever disagree, this check is
measuring something the reader never sees.

    python tools/check_md.py        -> "clean", or one line per note with what survived

The tempered `(?!\\*\\*)` in the bold pattern is load-bearing and three notes prove it: a bold
run may contain an italic one, as in Myanmar's `**They are drawn as *not enumerated* rather
than as Muslim**`. A `[^*]` class refuses that match, leaves the outer markers on screen, and
then lets the italic pass mangle the halves.
"""
import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from countries import COUNTRIES     # noqa: E402

# Keep in step with `md()` in index.html.
BOLD = re.compile(r"\*\*((?:(?!\*\*)[\s\S])+?)\*\*")
ITAL = re.compile(r"\*([^*]+?)\*")
CODE = re.compile(r"`([^`]+?)`")


def md(s):
    s = s.replace("&", "&amp;").replace("<", "&lt;")
    s = BOLD.sub(r"<b>\1</b>", s)
    s = ITAL.sub(r"<i>\1</i>", s)
    return CODE.sub(r"<code>\1</code>", s)


def main():
    bad = 0
    for cc, meta in sorted(COUNTRIES.items()):
        out = md(meta.get("note_public", "") or "")
        left = [ch for ch in out if ch in "*`"]
        if not left:
            continue
        bad += 1
        i = min((out.find(ch) for ch in "*`" if ch in out))
        print(f"{cc}: {len(left)} marker(s) survived -> ...{out[max(0, i - 70):i + 70]}...")
    print("clean" if not bad else f"{bad} note(s) with leftovers")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
