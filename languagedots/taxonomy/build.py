"""Build taxonomy/languages.json (the tree and every node's colour) and check the mappings.

    python taxonomy/build.py
    python taxonomy/build.py --only in,np,et     only those countries' fragments and mappings
                                                 (the build tail's way of leaving out countries
                                                 still being built)

COLOUR, spec §6. One colour per node, the same in every country, hand-tunable here:

  * Each family owns a region of the wheel: Indo-Aryan the warm half (red, orange, yellow,
    pink), Dravidian greens and teals, Sino-Tibetan blues and violets, Austroasiatic magentas,
    Iranian sand and tan, Dardic cyan.
  * Inside a family, hue is NOT banded tightly. On a language map the interesting edges are
    between members of one family (Hindi | Bhojpuri | Maithili | Bengali), so the big languages
    are hand-picked in HAND for contrast with their neighbours on the ground, and only the
    small ones are generated near their group's colour.
  * A group node (Bihari, Bhil, Kiranti) has its own colour too, because it holds the people a
    census filed under the group without naming the language. Those are drawn desaturated, so
    "unnamed Indo-Aryan" reads as a remainder and not as one more language.

Colours are OKLCH (L 0-1, C, hue degrees), converted to sRGB hex here. Dark basemap, so most
sit at L 0.62-0.85.

GENERATED COLOURS ARE STABLE (Anita, 2026-10-05: "yes let's make stable"). A node with no colour
of its own (not in HAND or GROUP, no `L C h` in a fragment) gets one from a fixed grid of 45 slots
around its parent's colour: five lightnesses (parent's L -0.12, -0.06, 0, +0.06, +0.12, the whole
set shifted to fit 0.55-0.92) by nine hues (parent's hue -4 to +4 steps; a step is 10 degrees, wider
for a pale parent, up to 20 at low chroma, so its members still differ), at the parent's chroma.
  * The node's id, hashed (crc32), names its preferred slot. Its colour depends on that, on its
    parent's colour, and on nothing else unless the slot is taken.
  * A slot is taken if it is within 0.04 (OKLab) of an ancestor's colour (so never the parent's own
    colour: two opposite steps once gave Chalchiteko and Mopan exactly `mayan`'s), or of a sibling
    already coloured: every sibling with its own colour, and the generated siblings that come before
    it in id order. Then the node probes on through the grid (7 slots at a time, which changes both
    lightness and hue) to the first free one. If none is 0.04 clear of the siblings, it tries again
    at 0.025, then takes any slot no sibling sits on.
  * So adding or removing a sibling never moves another node, unless the new one takes or frees
    the slot that node wanted (or, rarely, one a node it moved then wanted). Before 2026-10-05 a
    colour was the node's place among its uncoloured siblings, and every added sibling shifted the
    rest (Slovak's hand-pick moved Poland's dialects a step; Russia froze four nodes against it).
  * A big sibling set spreads over the whole grid, +-40 degrees and all five lightnesses; the
    largest today (Bantu, Kiranti, Naga) have under 30 generated members.
To fix one by hand, give it a colour in its fragment: that moves no other node except a generated
sibling it lands on.

REGROUPING (2026-10-06): taxonomy/regroup.txt adds middle levels (Bantu zones, Austronesian
branches, an Arabic group, languages over their dialects) without rewriting ids where they are
written. The tree is coloured as written, then moved (taxonomy/regroup.py), so no node's colour
depends on the regrouping; new groups get theirs from the slot grid around their new parent.
languages.json holds the moved ids. See taxonomy/GROUPING.md.
"""
import json
import math
import os
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# ---- hand-picked colours: (L, C, h) ----
HAND = {
    # Indo-Aryan, west to east across the plain and then the rest
    "indoeuropean.indoaryan.northwestern.punjabi": (0.70, 0.17, 5),
    "indoeuropean.indoaryan.northwestern.saraiki": (0.80, 0.12, 30),
    # Hindko between Pashto (sand) and Punjabi (pink): a light purple, not a second pink
    "indoeuropean.indoaryan.northwestern.hindko": (0.76, 0.13, 315),
    "indoeuropean.indoaryan.northwestern.pahari_pothwari": (0.80, 0.10, 350),
    # Sindhi was red-orange (25) and the Punjab/Sindh line vanished against Punjabi's pink
    "indoeuropean.indoaryan.northwestern.sindhi": (0.80, 0.16, 75),
    "indoeuropean.indoaryan.northwestern.kachchhi": (0.74, 0.13, 5),   # 2026-10-06: further off Gujarati
    "indoeuropean.indoaryan.northwestern.dogri": (0.78, 0.13, 70),
    "indoeuropean.indoaryan.central.hindi": (0.75, 0.16, 58),
    "indoeuropean.indoaryan.central.urdu": (0.60, 0.20, 20),
    "indoeuropean.indoaryan.central.haryanvi": (0.86, 0.15, 95),
    "indoeuropean.indoaryan.central.braj": (0.66, 0.14, 45),
    "indoeuropean.indoaryan.central.bundeli": (0.84, 0.10, 40),
    "indoeuropean.indoaryan.eastcentral.awadhi": (0.82, 0.12, 15),
    "indoeuropean.indoaryan.eastcentral.bagheli": (0.66, 0.13, 70),
    "indoeuropean.indoaryan.eastcentral.chhattisgarhi": (0.84, 0.14, 85),
    "indoeuropean.indoaryan.bihari.bhojpuri": (0.62, 0.18, 35),
    "indoeuropean.indoaryan.bihari.magahi": (0.84, 0.13, 70),
    "indoeuropean.indoaryan.bihari.maithili": (0.70, 0.17, 0),
    "indoeuropean.indoaryan.bihari.bajjika": (0.80, 0.11, 25),
    "indoeuropean.indoaryan.bihari.angika": (0.72, 0.12, 330),
    "indoeuropean.indoaryan.bihari.khortha": (0.68, 0.14, 55),
    "indoeuropean.indoaryan.bihari.sadri": (0.86, 0.12, 60),
    "indoeuropean.indoaryan.tharu.tharu": (0.80, 0.14, 120),
    "indoeuropean.indoaryan.eastern.bengali": (0.88, 0.15, 100),
    "indoeuropean.indoaryan.eastern.assamese": (0.72, 0.15, 75),
    "indoeuropean.indoaryan.eastern.odia": (0.66, 0.16, 10),
    "indoeuropean.indoaryan.eastern.sambalpuri": (0.82, 0.10, 0),
    "indoeuropean.indoaryan.eastern.halbi": (0.76, 0.12, 40),
    "indoeuropean.indoaryan.rajasthani.rajasthani": (0.83, 0.12, 42),  # 2026-10-06: off Gujarati, and no longer equal to Saraiki
    "indoeuropean.indoaryan.pahari.western.pahari": (0.82, 0.12, 95),
    "indoeuropean.indoaryan.rajasthani.marwari": (0.66, 0.15, 40),
    "indoeuropean.indoaryan.rajasthani.mewari": (0.84, 0.11, 75),
    "indoeuropean.indoaryan.rajasthani.malvi": (0.70, 0.13, 10),
    "indoeuropean.indoaryan.rajasthani.lambadi": (0.70, 0.18, 330),
    "indoeuropean.indoaryan.rajasthani.gujari": (0.84, 0.12, 50),
    "indoeuropean.indoaryan.gujarati.gujarati": (0.78, 0.15, 30),
    "indoeuropean.indoaryan.bhil.bhili": (0.66, 0.15, 320),
    "indoeuropean.indoaryan.bhil.wagdi": (0.80, 0.12, 310),
    "indoeuropean.indoaryan.southern.marathi": (0.68, 0.17, 350),
    "indoeuropean.indoaryan.southern.konkani": (0.82, 0.12, 330),
    "indoeuropean.indoaryan.pahari.central.garhwali": (0.80, 0.13, 40),
    "indoeuropean.indoaryan.pahari.central.kumaoni": (0.68, 0.15, 15),
    # 2026-10-06 (Anita: "a bit more distinguishable from Hindi"): was (0.80, 0.16, 45), 0.062
    # from Hindi; now 0.086, and Garhwali 0.052 (was 0.033)
    "indoeuropean.indoaryan.pahari.eastern.nepali": (0.78, 0.17, 30),
    "indoeuropean.indoaryan.pahari.eastern.doteli": (0.68, 0.14, 25),
    "indoeuropean.indoaryan.dardic.kashmiri": (0.76, 0.12, 205),
    "indoeuropean.indoaryan.dardic.shina": (0.66, 0.10, 190),
    "indoeuropean.indoaryan.dardic.kohistani": (0.84, 0.08, 195),
    "indoeuropean.iranian.pashto": (0.82, 0.09, 105),
    "indoeuropean.iranian.balochi": (0.66, 0.09, 75),
    # a light but real blue, not near-white (Anita 2026-10-05: the US read drab)
    "indoeuropean.germanic.english": (0.90, 0.05, 245),
    # Dravidian
    "dravidian.southcentral.telugu": (0.72, 0.17, 145),
    "dravidian.southern.tamil": (0.70, 0.13, 185),
    "dravidian.southern.kannada": (0.84, 0.17, 125),
    "dravidian.southern.malayalam": (0.62, 0.13, 165),
    "dravidian.southern.tulu": (0.80, 0.11, 165),
    "dravidian.southcentral.gondi": (0.84, 0.13, 150),
    "dravidian.southcentral.kui": (0.64, 0.13, 135),
    "dravidian.northern.kurukh": (0.80, 0.12, 175),
    "dravidian.northern.brahui": (0.72, 0.15, 150),
    # Austroasiatic
    "austroasiatic.munda.santali": (0.70, 0.18, 320),
    "austroasiatic.munda.mundari": (0.82, 0.12, 330),
    "austroasiatic.munda.ho": (0.62, 0.16, 300),
    "austroasiatic.munda.sora": (0.80, 0.12, 305),
    "austroasiatic.munda.korku": (0.72, 0.14, 340),
    "austroasiatic.khasian.khasi": (0.70, 0.17, 330),
    "austroasiatic.khasian.pnar": (0.84, 0.10, 320),
    # Sino-Tibetan
    "sinotibetan.boro_garo.bodo": (0.70, 0.13, 250),
    "sinotibetan.boro_garo.garo": (0.64, 0.14, 270),
    "sinotibetan.boro_garo.kokborok": (0.78, 0.11, 240),
    "sinotibetan.meitei": (0.72, 0.15, 290),
    "sinotibetan.kukichin.mizo": (0.82, 0.10, 265),
    "sinotibetan.karbi": (0.80, 0.09, 230),
    "sinotibetan.tibetic.tibetan": (0.70, 0.11, 235),
    "sinotibetan.tibetic.balti": (0.62, 0.12, 250),
    "sinotibetan.tibetic.ladakhi": (0.80, 0.08, 245),
    "sinotibetan.tibetic.sherpa": (0.82, 0.09, 230),
    "sinotibetan.tamangic.tamang": (0.66, 0.15, 270),
    "sinotibetan.tamangic.gurung": (0.82, 0.10, 285),
    "sinotibetan.magaric.magar_dhut": (0.72, 0.13, 245),
    "sinotibetan.magaric.magar_kham": (0.84, 0.08, 255),
    "sinotibetan.newaric.newar": (0.62, 0.17, 295),
    "sinotibetan.kiranti.limbu": (0.78, 0.12, 300),
    "sinotibetan.kiranti.bantawa": (0.66, 0.12, 280),
    "sinotibetan.kiranti.chamling": (0.84, 0.08, 290),
    # Berber, more saturated (Anita, 2026-10-06: a minority beside the Maghreb Arabics' mint-teal,
    # it should stand out more). Yellows, golds, oranges and a lime, hue 42-125, C 0.14-0.19; was
    # C 0.12-0.15 in the fragments, with Siwi, Tamazight and Tunisian Berber generated toward green.
    # These override the fragments' own `L C h` (kept equal to them). taxonomy/COLOURS.md.
    "afroasiatic.berber.tachelhit": (0.87, 0.18, 97),
    "afroasiatic.berber.tamazight": (0.80, 0.19, 125),
    "afroasiatic.berber.tarifit": (0.64, 0.16, 42),
    "afroasiatic.berber.kabyle": (0.74, 0.17, 62),
    "afroasiatic.berber.chaouia": (0.86, 0.18, 108),
    "afroasiatic.berber.tumzabt": (0.70, 0.14, 88),
    "afroasiatic.berber.tamahaq": (0.90, 0.14, 96),
    "afroasiatic.berber.tamasheq": (0.77, 0.165, 72),
    "afroasiatic.berber.tamajaq": (0.72, 0.165, 62),
    "afroasiatic.berber.nafusi": (0.72, 0.16, 115),
    "afroasiatic.berber.tunisian_berber": (0.78, 0.17, 110),
    "afroasiatic.berber.siwi": (0.82, 0.17, 100),
    "afroasiatic.berber.zenaga": (0.76, 0.15, 70),
    # roots that hold people of their own
    "signlanguage": (0.90, 0.00, 0),
    "other": (0.62, 0.00, 0),
}

# Group colours: the base the generator works from, and what a group's own unnamed people get.
GROUP = {
    "indoeuropean": (0.74, 0.12, 50),
    "indoeuropean.indoaryan": (0.74, 0.12, 50),
    "indoeuropean.indoaryan.central": (0.74, 0.15, 50),
    "indoeuropean.indoaryan.eastcentral": (0.78, 0.13, 40),
    "indoeuropean.indoaryan.bihari": (0.74, 0.14, 30),
    "indoeuropean.indoaryan.tharu": (0.80, 0.14, 120),
    "indoeuropean.indoaryan.eastern": (0.80, 0.14, 85),
    "indoeuropean.indoaryan.rajasthani": (0.76, 0.14, 45),
    "indoeuropean.indoaryan.gujarati": (0.78, 0.15, 30),
    "indoeuropean.indoaryan.bhil": (0.72, 0.14, 320),
    "indoeuropean.indoaryan.southern": (0.70, 0.16, 350),
    "indoeuropean.indoaryan.northwestern": (0.70, 0.15, 10),
    "indoeuropean.indoaryan.pahari": (0.76, 0.13, 35),
    "indoeuropean.indoaryan.pahari.western": (0.76, 0.12, 90),
    "indoeuropean.indoaryan.pahari.central": (0.74, 0.14, 30),
    "indoeuropean.indoaryan.pahari.eastern": (0.76, 0.14, 40),
    "indoeuropean.indoaryan.dardic": (0.76, 0.10, 200),
    "indoeuropean.indoaryan.inner_terai": (0.78, 0.12, 110),
    "indoeuropean.indoaryan.sanskrit": (0.90, 0.06, 80),
    "indoeuropean.iranian": (0.76, 0.09, 90),
    "indoeuropean.germanic": (0.90, 0.05, 245),
    "dravidian": (0.74, 0.13, 155),
    "dravidian.southern": (0.72, 0.13, 175),
    "dravidian.southcentral": (0.74, 0.14, 145),
    "dravidian.central": (0.78, 0.12, 130),
    "dravidian.northern": (0.76, 0.12, 170),
    "austroasiatic": (0.72, 0.15, 320),
    "austroasiatic.munda": (0.72, 0.15, 320),
    "austroasiatic.khasian": (0.74, 0.15, 330),
    "austroasiatic.nicobarese": (0.76, 0.13, 345),
    "sinotibetan": (0.72, 0.12, 260),
    "sinotibetan.tibetic": (0.74, 0.10, 240),
    "sinotibetan.eastbodish": (0.74, 0.10, 225),
    "sinotibetan.tamangic": (0.72, 0.13, 275),
    "sinotibetan.magaric": (0.76, 0.12, 250),
    "sinotibetan.chepangic": (0.76, 0.12, 215),
    "sinotibetan.newaric": (0.66, 0.15, 295),
    "sinotibetan.kiranti": (0.74, 0.12, 290),
    "sinotibetan.westhimalayish": (0.76, 0.10, 220),
    "sinotibetan.dhimalish": (0.74, 0.10, 235),
    "sinotibetan.dura": (0.74, 0.10, 235),
    "sinotibetan.lepcha": (0.76, 0.12, 280),
    "sinotibetan.boro_garo": (0.72, 0.13, 255),
    "sinotibetan.meitei": (0.72, 0.15, 290),
    "sinotibetan.karbi": (0.80, 0.09, 230),
    "sinotibetan.kukichin": (0.76, 0.12, 270),
    "sinotibetan.naga": (0.74, 0.13, 245),
    "sinotibetan.tani": (0.74, 0.13, 220),
    "sinotibetan.mishmi": (0.74, 0.12, 205),
    "sinotibetan.burmish": (0.74, 0.12, 300),
    "afroasiatic": (0.82, 0.08, 160),
    "afroasiatic.arabic": (0.82, 0.08, 160),
    "afroasiatic.berber": (0.80, 0.16, 95),     # 2026-10-06, was 0.78 0.12 100 (ca, ly, pt)
}

# Groups whose unnamed people are drawn in the group's full colour, not washed out. "Chinese" is
# the one label most censuses print for Sinitic (US ACS 2.1M, Canada's "Chinese, n.o.s.", the
# UK's and Australia's "Chinese"): to a reader it is one commonly named language, not a
# remainder, and washed out it came out a pale blue-grey beside English's near-white in every
# American and Canadian city (Anita, 2026-10-04).
UNWASHED = {"sinotibetan.sinitic"}

# Generated colours: the slot grid around the parent (see the docstring).
SLOT_DL = (-0.12, -0.06, 0.0, 0.06, 0.12)
SLOT_DH = (-4, -3, -2, -1, 0, 1, 2, 3, 4)     # in hue steps
PROBE = 7                                     # stride through the 45 slots; coprime with 45
NEAR = (0.04, 0.025, 0.005)                   # OKLab: "taken" if this close; relaxed in turn
ROOT_BASE = (0.75, 0.05, 0)                   # for a root with no colour of its own


def oklch_to_hex(L, C, h):
    a = C * math.cos(math.radians(h))
    b = C * math.sin(math.radians(h))
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bl = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    def gamma(x):
        x = min(1.0, max(0.0, x))
        return 12.92 * x if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055
    return "#" + "".join(f"{round(gamma(v) * 255):02x}" for v in (r, g, bl))


def read_tree(only=None):
    """tree.txt, then every taxonomy/tree.d/<cc>.txt fragment, in file-name order.

    A FRAGMENT PER COUNTRY so that agents adding languages at once never edit the same file.
    Same format as tree.txt, with an optional third field, an OKLCH colour `L C h`, for a new
    group or family (a new language under an existing group needs none: it is generated near
    its group). A node may be defined in several fragments only with the same label; a second
    definition with another label, or a colour that differs, stops the build."""
    nodes, seen = [], {}
    files = [HERE / "tree.txt"] + sorted(f for f in (HERE / "tree.d").glob("*.txt")
                                         if only is None or f.stem in only)
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = [s.strip() for s in line.split("|")]
            nid, label = parts[0], parts[1]
            col = tuple(float(x) for x in parts[2].split()) if len(parts) > 2 and parts[2] else None
            if nid in seen:
                prev = seen[nid]
                if prev["label"] != label or (col and prev.get("lch") and col != prev["lch"]):
                    raise SystemExit(f"{nid} defined twice differently: {prev['src']} and {f.name}")
                # a colour may come from any fragment: the first to define a node is often a
                # country that only repeated it bare (uk.txt's Nguni before za.txt's, 2026-10-04)
                if col and not prev.get("lch"):
                    if len(col) != 3:
                        raise SystemExit(f"{f.name}: {nid}: colour must be 'L C h'")
                    prev["lch"] = col
                continue
            n = {"id": nid, "label": label, "src": f.name}
            if col:
                if len(col) != 3:
                    raise SystemExit(f"{f.name}: {nid}: colour must be 'L C h'")
                n["lch"] = col
            seen[nid] = n
            nodes.append(n)
    ids = [n["id"] for n in nodes]
    have = set(ids)
    for n in nodes:
        parent = n["id"].rsplit(".", 1)[0] if "." in n["id"] else None
        if parent and parent not in have:
            raise SystemExit(f"{n['id']}: parent {parent} is not a node")
        n["parent"] = parent
    return nodes


def _lab(c):
    L, C, h = c
    return (L, C * math.cos(math.radians(h)), C * math.sin(math.radians(h)))


def _dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _slots(base):
    """The 45 generated-colour slots around a parent's colour, lightness-major."""
    L0, C, h0 = base
    step = 20.0 if C <= 0 else min(20.0, max(10.0, 10 * 0.13 / C))
    lo, hi = L0 + SLOT_DL[0], L0 + SLOT_DL[-1]
    shift = (0.92 - hi if hi > 0.92 else 0) + (0.55 - lo if lo < 0.55 else 0)
    return [(round(L0 + dl + shift, 4), C, round((h0 + k * step) % 360, 2))
            for dl in SLOT_DL for k in SLOT_DH]


def colour(nodes):
    kids = {}
    for n in nodes:
        kids.setdefault(n["parent"], []).append(n["id"])
    lch = {}
    for n in nodes:
        nid = n["id"]
        if nid in HAND:
            lch[nid] = HAND[nid]
        elif nid in GROUP:
            lch[nid] = GROUP[nid]
        elif n.get("lch"):
            lch[nid] = n["lch"]
    # Generated colours, in id order: a parent ("a.b") sorts before its children ("a.b.c"), so it
    # is always coloured first, and siblings meet each other in id order (the docstring's rule).
    for nid in sorted(n["id"] for n in nodes if n["id"] not in lch):
        parent = nid.rsplit(".", 1)[0] if "." in nid else None
        base = lch[parent] if parent else ROOT_BASE
        anc, q = [], parent
        while q:
            anc.append(_lab(lch[q]))
            q = q.rsplit(".", 1)[0] if "." in q else None
        sibs = [_lab(lch[k]) for k in kids[parent] if k != nid and k in lch]
        slots = _slots(base)
        pref = zlib.crc32(nid.encode("utf-8")) % len(slots)
        lch[nid] = slots[pref]
        for near in NEAR:
            for j in range(len(slots)):
                s = slots[(pref + j * PROBE) % len(slots)]
                ls = _lab(s)
                if any(_dist(ls, a) < NEAR[0] for a in anc) or any(_dist(ls, b) < near for b in sibs):
                    continue
                lch[nid] = s
                break
            else:
                continue
            break
    for n in nodes:
        n["lch"] = lch[n["id"]]
    return nodes


def regroup_and_paint(nodes):
    """Move the coloured as-written tree to its regrouped ids (taxonomy/regroup.py), colour the
    new groups that have no colour (the slot grid around their parent, clear of ancestors and
    siblings), then write each node's hex colours. Colouring BEFORE regrouping is what keeps every
    existing node's colour exactly as it was where it is written."""
    import regroup
    nodes = regroup.apply(nodes)
    lch = {n["id"]: n["lch"] for n in nodes if n.get("lch")}
    kids = {}
    for n in nodes:
        kids.setdefault(n["parent"], []).append(n["id"])
    for n in nodes:                     # parents come before children
        if n["id"] in lch:
            continue
        nid, parent = n["id"], n["parent"]
        base = lch[parent] if parent else ROOT_BASE
        anc, q = [], parent
        while q:
            anc.append(_lab(lch[q]))
            q = q.rsplit(".", 1)[0] if "." in q else None
        sibs = [_lab(lch[k]) for k in kids[parent] if k != nid and k in lch]
        slots = _slots(base)
        pref = zlib.crc32(nid.encode("utf-8")) % len(slots)
        lch[nid] = slots[pref]
        for j in range(len(slots)):
            s = _lab(slots[(pref + j * PROBE) % len(slots)])
            if not any(_dist(s, a) < NEAR[0] for a in anc) and not any(_dist(s, b) < NEAR[0] for b in sibs):
                lch[nid] = slots[(pref + j * PROBE) % len(slots)]
                break
    for n in nodes:
        L, C, h = lch[n["id"]]
        n["color"] = oklch_to_hex(L, C, h)
        # a group's own (unnamed) people are drawn desaturated
        if kids.get(n["id"]) and n["id"] not in UNWASHED:
            n["color_own"] = oklch_to_hex(min(0.9, L + 0.04), C * 0.35, h)
    return nodes


def check_mappings(ids, only=None):
    """Every mapping module (taxonomy/<cc><year>.py): every string value of its NAMES or CODES
    dict, and every id in its optional EXTRA_NODES list (nodes resolve() can return that are not
    table values, such as India's Pahari split), must be a node."""
    import importlib
    import re
    from regroup import move
    bad = []
    for f in sorted(HERE.glob("*.py")):
        if not re.fullmatch(r"[a-z]{2}\d{4}[a-z_]*", f.stem):
            continue
        if only is not None and f.stem[:2] not in only:
            continue
        mod = importlib.import_module(f.stem)
        vals = []
        for attr in ("NAMES", "CODES"):
            vals += [(k, v) for k, v in getattr(mod, attr, {}).items() if isinstance(v, str)]
        vals += [("EXTRA_NODES", v) for v in getattr(mod, "EXTRA_NODES", [])]
        # mappings write ids where they were; regroup.txt says where they are drawn
        bad += [f"{f.stem}: {k} -> {v}" for k, v in vals if move(v) not in ids]
    if bad:
        raise SystemExit("mapped to nodes that do not exist:\n  " + "\n  ".join(bad))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated countries: read only their fragments and mappings")
    only = ap.parse_args().only
    # --only writes a SUBSET of the tree to languages.json, which the viewer then uses; an agent
    # ran it on its own country by mistake (cz, 2026-10-04). Only the build tail may.
    if only and os.environ.get("LD_BUILD_TAIL") != "1":
        raise SystemExit("--only is for tools/build_tail.py (it writes a partial languages.json); "
                         "run plain `python taxonomy/build.py`")
    only = set(only.split(",")) if only else None
    nodes = regroup_and_paint(colour(read_tree(only)))
    ids = {n["id"] for n in nodes}
    check_mappings(ids, only)
    out = HERE / "languages.json"
    for n in nodes:
        n.pop("src", None)
        n.pop("lch", None)
    # through a temp file: several agents run this, and the viewer reads it
    tmp = out.with_suffix(f".json.{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"nodes": nodes}, ensure_ascii=False, indent=0), encoding="utf-8")
    os.replace(tmp, out)
    print(f"wrote {out}: {len(nodes)} nodes, {sum(1 for n in nodes if n['id'] in HAND)} hand-coloured")


if __name__ == "__main__":
    main()
