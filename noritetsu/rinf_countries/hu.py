"""Hungary: RINF's ids are MÁV's line numbers (vonalszám), the numbers riders and hu.wikipedia
use ("1-es vasútvonal" redirects to "Budapest–Hegyeshalom–Rajka-vasútvonal"). Pulled
2026-09-30; every section is filed under one infrastructure-manager code, HU55_IM, GYSEV's
included.

HOW AN ID READS AS A NUMBER
- "30", "100/1", "20/2": line 30, 100, 20. A "/<digit>" is MÁV splitting one line into parts
  (20/1 Székesfehérvár - Celldömölk, 20/2 Celldömölk - Szombathely).
- "5/a", "1/d", "1D", "15R", "262/c": the letter belongs to the number, which is how Wikidata
  writes them ("5a" Bodajk - Balinka, "1d" Hegyeshalom - Rajka). Upper-case letters are
  GYSEV's (1D, 8G, 8R, 15D, 15R). Most letter ids are yard curves and freight branches that
  build_model then drops as unridden; the passenger stubs among them are folded into their
  line in FIXED.
- A "/<digit>" part can be a different public line: RINF's 120/1 is Rákos - Újszász -
  Szolnok, which MÁV numbers 120a; 120/3 is Szajol - Lőkösháza, line 120. Those few are in
  FIXED, checked against OSM's route=railway relations and Wikidata.
- The rule is CERTAIN (`rule_certain`): OSM's relations carry the same MÁV numbers, so they
  add nothing, and letting them overrule the id did harm. Where only a stub of a closed line
  traced, it lay on a neighbour's relation and joined it: 22 (Körmend - Zalalövő) became 25,
  64 (Pécs - Bátaszék) became 65, and Budapest's connecting curves 207, 223 and 347 became
  line 1, which made line 1 read long.
- Numbers 200-299 are Budapest's and other junctions' connecting curves, 300-499 freight and
  industrial lines; they keep their own number. The unridden ones end at junctions and
  build_model drops them.

NAMES are "<n>-es vasútvonal" with the suffix Hungarian vowel harmony gives the number as
spoken (1-es, 3-as, 5-ös, 6-os, 20-as, 40-es, 100-as). The English name, which is what the
app shows, is "Line <n> (<route>)", the route taken from the HUNGARIAN Wikidata label for the
number ("Pécs–Mohács-vasútvonal" -> "Line 65 (Pécs–Mohács)"). The Hungarian labels are
hu.wikipedia's article titles; the English ones are patchy and some are wrong (65's reads
"Villány–Magyarbóly", which is 66; 127's "Oradea–Kótpuszta"). So COUNTRY fetches Hungarian
labels only, which keeps rinf.py from appending an English one, and `HuNameEn` picks the item
the way rinf.py does (rinf.load_wikidata's order, then rinf.wikidata_item's length check
against RINF's own length for the number).

GYSEV (Győr-Sopron-Ebenfurti Vasút) has no IM code of its own in RINF, so `im_of` names it
from the line number. Its network before 2025 (8, 9, 15, 16, 17 to Zalaszentiván, 18, 21,
22, Hegyeshalom - Rajka, Sopron - Ebenfurth) and the lines MÁV handed it on 1 July 2025 (10
from Győrszabadhegy, 11 from Győrszabadhegy, 14, 17 on to Nagykanizsa, 20, 23, 24, 25, 26;
Világgazdaság, 2025-07, in hu_sources.md) are whole or nearly whole lines. It also took the
Murakeresztúr ends of 30 and 41 and stubs of 13 and 27, which leave MÁV the larger share of
those lines, so they stay MÁV. RINF itself has not caught up: its "pvh." points (pályavasúti
határ, the boundary between the two managers) are the pre-2025 ones, and a section's
validity date (2023-03-28 on some GYSEV sections, 1900-01-01 on others of the same line) does
not follow the boundary.
"""
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

# Ids whose public number is not what the rule reads. The first group is where MÁV's public
# number differs from the id (OSM's route=railway relations and Wikidata agree on these);
# the second is GYSEV's letter ids for stubs and curves that riders know as part of the line.
FIXED = {
    "120/1": "120a",  # Rákos - Újszász - Szolnok; 120 is Szajol - Lőkösháza (120/3)
    "120/2": "120a",  # its Szolnok end (Újszászi elágazás, Szolnok C and D elágazás)
    "125/a": "125",   # Mezőhegyes - Battonya end of Mezőtúr - Battonya (Wikidata 125)
    "265/a": "265",   # OSM and MÁV write these two without the letter
    "269/a": "269",
    "15D": "15",      # Harka - border towards Deutschkreutz (GYSEV's Sopron - Deutschkreutz trains)
    "15R": "15",      # Sopron - Sopron Rendező
    "8R": "8",        # Sopron Rendező curve
    "8G": "8",        # Győr-GYSEV - its western junction, where line 8 starts
    "1/d": "1D",      # the 0 km Hegyeshalom stub that joins 1D
}


def hu_ref(lid):
    m = re.fullmatch(r"(\d+)(?:/(\d+))?", lid or "")
    if m:
        return m.group(1)
    m = re.fullmatch(r"(\d+)/?([A-Za-z])", lid or "")
    if m:
        return m.group(1) + m.group(2).lower()
    return None


def hu_osm_ref(ref):
    """An OSM route=railway ref as a MÁV number, or None. MÁV's are "30", "120a", "113 (1)"
    (a line mapped in parts); an upper-case letter is an industrial siding ("11K", "100EK",
    "1AX") or a Slovak ŽSR relation ("120A", "101A"), and "M201", "RH2", "108 01" are
    Croatian and Austrian lines crossing the border, none of which may number a MÁV line.
    Romanian, Serbian and Slovak relations with a bare number ("200", "135") cannot be told
    apart by ref; they lie outside the country except at the border crossings."""
    m = re.fullmatch(r"(\d+)(?:\s*\(\d+\))?([a-z]?)", (ref or "").strip())
    return (m.group(1) + m.group(2)).upper() if m else None


_ONES = {1: "es", 2: "es", 3: "as", 4: "es", 5: "ös", 6: "os", 7: "es", 8: "as", 9: "es"}
_TENS = {1: "es", 2: "as", 3: "as", 4: "es", 5: "es", 6: "as", 7: "es", 8: "as", 9: "es"}
# A letter's name ends in a vowel (a, bé, cé, dé, e, ká...) except these (ef, el, em, en, er...).
_CONSONANT_NAMED = set("flmnrsx")


def hu_suffix(ref):
    """The case suffix a number takes in Hungarian, "1" -> "es", "120a" -> "s"."""
    m = re.fullmatch(r"(\d+)([A-Za-z]?)", ref or "")
    if not m:
        return "es"
    if m.group(2):
        return "es" if m.group(2).lower() in _CONSONANT_NAMED else "s"
    n = int(m.group(1))
    if n % 10:
        return _ONES[n % 10]
    if n % 100:
        return _TENS[(n // 10) % 10]
    if n % 1000:
        return "as"                       # száz
    return "es"                           # ezer


class HuName(str):
    """rinf.py writes a numbered line's name as COUNTRY["name"].format(ref=ref); a plain
    template cannot pick the suffix, so this one does."""
    def format(self, *args, **kw):
        ref = kw.get("ref", args[0] if args else "")
        return f"{ref}-{hu_suffix(ref)} vasútvonal"


# Lines GYSEV manages (docstring). 266, 292, 346 and 350-354 are the curves and freight
# branches on them that OSM's route=railway relations give operator=GYSEV.
GYSEV_LINES = {"8", "9", "10", "11", "14", "15", "16", "17", "18", "20", "21", "22", "23", "24",
               "25", "26", "524", "266", "292", "346", "350", "351", "352", "353", "354"}
GYSEV_IDS = {"1D", "1/d"}                  # Hegyeshalom - Rajka


_ROOT = Path(__file__).resolve().parent.parent
_ROUTES = None


def hu_routes():
    """Public number key ("65", "120A") -> the route in its Hungarian Wikidata label."""
    global _ROUTES
    if _ROUTES is not None:
        return _ROUTES
    import rinf                                          # only called from inside rinf.build
    raw = _ROOT / "data" / "raw" / "rinf" / "hu"
    ip = _ROOT / "data" / "proc" / "hu" / "infra.pkl"
    infra = {}
    if ip.exists():
        with open(ip, "rb") as f:
            infra = pickle.load(f)
    wd = rinf.load_wikidata(raw, frozenset(infra), None, "hu")
    km, seen = defaultdict(float), set()
    for r in json.loads((raw / "sections.json").read_text(encoding="utf-8"))["rows"]:
        if r["sol"] in seen:
            continue
        seen.add(r["sol"])
        ref = FIXED.get(r.get("line") or "") or hu_ref(r.get("line"))
        if ref:
            km[ref.upper()] += float(r.get("len") or 0)
    _ROUTES = {}
    for k, cands in wd.items():
        # A lone item names its number even when its length is off: 81's says 661 km, and
        # 64's and 66's are the whole historic line where RINF has the open stub.
        w = rinf.wikidata_item(cands, km.get(k)) or (cands[0] if len(cands) == 1 else {})
        lab = w.get("hu") or ""
        lab = re.sub(r"[\s-]*vasútvonal$", "", lab).strip()
        if lab:
            _ROUTES[k] = lab
    return _ROUTES


class HuNameEn(str):
    def format(self, *args, **kw):
        ref = kw.get("ref", args[0] if args else "")
        route = hu_routes().get(ref.upper())
        return f"Line {ref} ({route})" if route else f"Line {ref}"


def hu_im(sec):
    lid = sec.get("base") or sec.get("line") or ""
    m = re.match(r"\d+", lid)
    return "GYSEV" if lid in GYSEV_IDS or (m and m.group(0) in GYSEV_LINES) else "MÁV"


COUNTRY = {
    # Hungarian labels only: see NAMES in the docstring.
    "iso3": "HUN", "wikidata": "Q28", "langs": ["hu"],
    "fixed": FIXED, "ref": hu_ref, "rule_certain": True, "osm_ref": hu_osm_ref,
    # MÁV's RINF section lengths leave out the track inside stations (line 1: RINF 154.4 km,
    # 189.2 traced, 191 published; GYSEV's lines in the same file read 1.00), so a short
    # section between two big stations reads short by up to a kilometre. At rinf.py's 0.3
    # Tatabánya - Bánhida on line 12 (3.34 traced, 2.61 in RINF), which the S12 runs over,
    # was rejected.
    "tol_abs": 1.0,
    # MÁV and Wikidata write the letter lower case, "120a", "1d".
    "ref_display": lambda k: k.lower(),
    "name": HuName("{ref}-es vasútvonal"), "name_en": HuNameEn("Line {ref} (<route>)"),
    # Wikidata's trolleybus items share route numbers with MÁV lines ("9-es trolibusz" is
    # older than the Fertőszentmiklós - Pamhagen item); rinf.NOT_MAINLINE leaves them out.
    "im": {"HU55_IM": "MÁV"},
    "im_of": hu_im,
}
