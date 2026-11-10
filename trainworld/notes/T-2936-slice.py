import sys, io, os, json, shutil
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf8")
from fontTools.ttLib import TTFont
from fontTools import subset

S = r"C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\fe4d1777-5b30-4243-ad0f-7ed61fc1aa3a\scratchpad"
SRC = S + r"\simpfont\chillM\ChillRoundM_v1.805\ChillRoundM.ttf"
APP = r"C:\Users\anita\projects\maps\.tw-trees\simpfont\trainworld\app"
FONTS = APP + r"\public\fonts"
FAMILY = "Hanzi Round"
PREFIX = "HanziRound-400-"

lvl = [l.strip() for l in open(S + r"\simpfont\tgscc\characters.txt", encoding="utf8") if l.strip()]
lvl1 = lvl[:3500]
gbh, gbo = [], []
for hi in range(0xA1, 0xF8):
    for lo in range(0xA1, 0xFF):
        try:
            c = bytes([hi, lo]).decode("gb2312")
        except Exception:
            continue
        (gbh if hi >= 0xB0 else gbo).append(c)
print("gb hanzi", len(gbh), "gb other", len(gbo))

base = TTFont(SRC)
cm = base.getBestCmap()

def has(c):
    return ord(c) in cm

# the game's own text first (every hanzi in zh-Hans messages and Chinese place names)
text = ""
msgs = json.load(open(APP + r"\src\i18n\messages\zh-Hans.json", encoding="utf8"))
text += "".join(v for v in msgs.values() if isinstance(v, str))
pn = open(APP + r"\src\i18n\place-names.json", encoding="utf8").read()
text += pn
isz = lambda c: 0x4e00 <= ord(c) <= 0x9fff
game = []
for c in text:
    if isz(c) and has(c) and c not in game:
        game.append(c)
print("game hanzi in font", len(game))

used = set()
chunks = []
# 0: punctuation, symbols, fullwidth forms (Nunito owns ASCII, Latin-1 and U+2000-206F)
punct = [ord(c) for c in gbo if has(c) and ord(c) > 0xFF and not 0x2000 <= ord(c) <= 0x206F]
punct += [u for u in list(range(0x3000, 0x3040)) + list(range(0xFF00, 0xFFF0)) if u in cm]
punct = sorted(set(punct))
chunks.append(punct)
used |= set(punct)
# 1: the game's hanzi
game_u = sorted({ord(c) for c in game})
chunks.append(game_u); used |= set(game_u)
def take(lst, n):
    cur = []
    for c in lst:
        u = ord(c)
        if u in used or not has(c):
            continue
        cur.append(u); used.add(u)
        if len(cur) == n:
            chunks.append(sorted(cur)); cur = []
    if cur:
        chunks.append(sorted(cur))
take(lvl1, 500)
take(gbh, 650)
print("chunks", [len(c) for c in chunks])

def ranges(us):
    us = sorted(us); out = []; a = b = us[0]
    for u in us[1:]:
        if u == b + 1:
            b = u
        else:
            out.append((a, b)); a = b = u
    out.append((a, b))
    return ", ".join("U+%x" % a if a == b else "U+%x-%x" % (a, b) for a, b in out)

for f in os.listdir(FONTS):
    if f.startswith(PREFIX):
        os.remove(os.path.join(FONTS, f))

css = ["/* Hanzi Round: Chill Round M (Warren2060/ChillRound, SIL OFL 1.1, public/fonts/OFL-HanziRound.txt), Simplified Chinese's face (T-2936).",
       "   The font's Reserved Font Name forbids keeping its name in a modified version, so the cut carries this family name inside and here.",
       "   Zen Maru Gothic completed to the GB2312 set; cut into woff2 slices by code point (punctuation, the hanzi this game's text uses,",
       "   then GB2312 level 1 and the rest in 500-650 character pieces), so a page loads only the slices its text uses; only zh-Hans's --font names it (style.css). */"]
total = 0
for i, us in enumerate(chunks):
    f = TTFont(SRC)
    opt = subset.Options()
    opt.layout_features = ["kern", "vert", "vrt2"]
    opt.name_IDs = [0, 1, 2, 3, 4, 5, 6, 13, 14]
    opt.notdef_outline = True
    opt.flavor = "woff2"
    s = subset.Subsetter(opt)
    s.populate(unicodes=us)
    s.subset(f)
    nt = f["name"]
    nt.names = [n for n in nt.names if n.nameID not in (1, 2, 3, 4, 6, 16, 17, 21, 22, 25)]
    for pid, eid, lid in ((3, 1, 0x409), (1, 0, 0)):
        nt.setName(FAMILY, 1, pid, eid, lid)
        nt.setName("Regular", 2, pid, eid, lid)
        nt.setName("HanziRound-Regular;slice %d" % i, 3, pid, eid, lid)
        nt.setName(FAMILY + " Regular", 4, pid, eid, lid)
        nt.setName("HanziRound-Regular", 6, pid, eid, lid)
    f.flavor = "woff2"
    p = os.path.join(FONTS, "%s%d.woff2" % (PREFIX, i))
    f.save(p)
    sz = os.path.getsize(p); total += sz
    css.append('@font-face { font-family: "%s"; font-style: normal; font-weight: 400; font-display: swap;\n  src: url(/fonts/%s%d.woff2) format("woff2");\n  unicode-range: %s; }' % (FAMILY, PREFIX, i, ranges(us)))
    print(i, len(us), sz)
print("total", total)
open(APP + r"\src\i18n\hanziround.css", "w", encoding="utf8", newline="\n").write("\n".join(css) + "\n")
shutil.copy(S + r"\simpfont\chillM\ChillRoundM_v1.805\LICENSE.txt", FONTS + r"\OFL-HanziRound.txt")
