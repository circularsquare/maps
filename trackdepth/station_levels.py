"""Absolute platform heights per subway station from the MTA elevator and escalator inventory.

Reads raw/mta_elevators.csv (data.ny.gov 94fv-bak7), raw/mta_stations.csv (39hk-dx4f) and the
3DEP DEM tiles in raw/dem/. Writes:

  evidence/station_levels.csv           one row per (station, platform or level) that can be pinned
  evidence/station_levels_unparsed.md   machines that could not be placed in a chain, and why

Method
------
Every machine with a rise (Elevator Rise or Escalator Rise) is one edge between two levels, read
from its Notes ("A to B ..."). Levels are nodes of a small graph per station complex: street,
mezzanines (upper / intermediate / lower / named by exit or by line), bridges, overpasses,
underpasses, platforms (by line and direction). Which end of an edge is higher follows from the
level kinds and whether the station is underground or elevated (MTA stations table, Structure).

Every street-end machine gets its own street node, pinned to the DEM ground at that machine's own
coordinate (median of a 5x5-pixel, 15 m window; the spread of the window widens the sd). The
heights of all levels in a complex then come from one weighted least-squares solve: where several
paths reach a level they are averaged by weight, and the misfit (chi-square per degree of freedom)
widens every sd in that complex. Assumptions are soft edges with their own sd and are listed in
`notes`:

  * a footbridge ("bridge") over the tracks is 4.0 +- 0.8 m above the platform it serves, an
    "overpass" 3.0 +- 1.5 m (it may be a concourse a few steps up, as at 168 St);
  * a complex with no street-end machine with a rise gets its topmost plain mezzanine at the
    typical street-to-mezzanine drop measured from the machines that do have one (see
    DEFAULT_MEZZ), with a wide sd;
  * an unnamed "mezzanine" is merged with a mezzanine named by an exit only when the complex has
    exactly one such named mezzanine of that tier;
  * station-specific readings (OVERRIDES) are written next to each case below.

Platform heights are floor heights; build.py takes rail as 1.1 m below.
"""
import collections
import math
import pathlib
import re

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

ROOT = pathlib.Path(__file__).parent
RAW = ROOT / "raw"
EVID = ROOT / "evidence"
FT = 0.3048
IN = 0.0254

to_utm = Transformer.from_crs(4326, 32618, always_xy=True)

# ---------------------------------------------------------------- rises

# Strings whose reading is a judgement call. raw -> (feet, inches, why)
WEIRD_RISES = {
    "56' - 2/25in": (56, 2.25, "read 2/25 as 2.25 in, as its siblings 2.025 / 21.4 / 21-4 at the same bank"),
    "56' - 2.025in": (56, 2.25, "read 2.025 as 2.25 in (same bank reads 2/25, 21.4, 21-4)"),
    "56' - 21.4in": (56, 2.25, "21.4 in is impossible; read as 2 1/4 in (siblings 2/25, 2.025, 21-4); 2.14 in would differ by 0.1 in"),
    "56' 21-4in": (56, 2.25, "read 21-4 as 2 1/4 in"),
    "39' 11/5in": (39, 11.5, "11/5 read as 11.5 in: the same bank's ES350 reads 39' 11 1/2in"),
    "40'39' 11 1/2in": (39, 11.5, "two values run together; took the precise one, 39' 11 1/2in"),
    "30'- 5/5in": (30, 0.5, "5/5 in is garbled; took 30' 0.5in (at most 1 in off)"),
    "10in - 2 3/8in": (10, 2.375, "'10in' is a typo for 10 ft (mezzanine-to-platform elevator)"),
    "10in- 2in": (10, 2.0, "'10in' is a typo for 10 ft (sibling EL220 reads 10' - 2in)"),
    "38-11 5/8in": (38, 11.625, "missing foot mark; read 38' 11 5/8in (sibling ES214 is 39')"),
    "19'=10in": (19, 10.0, "'=' read as '-'"),
    "14.8": (14.8, 0.0, "no unit; read as 14.8 ft (Pelham Bay Park's similar elevator is 14'8in)"),
    "83.3": (83.3, 0.0, "no unit; read as 83.3 ft (the parallel bank at this station is 81'8in)"),
    "25'11 3/4": (25, 11.75, "no inch mark; read 25' 11 3/4in"),
    "18'10 5/8": (18, 10.625, "no inch mark; read 18' 10 5/8in"),
    "16'-4in 3/4in": (16, 4.75, "read 16' 4 3/4in"),
}

FRAC = {"¼": " 1/4", "½": " 1/2", "¾": " 3/4"}


def norm_rise_text(s):
    s = s.strip()
    for a, b in (("’", "'"), ("‘", "'"), ("´", "'"), ("”", '"'), ("“", '"'), ("\xa0", " ")):
        s = s.replace(a, b)
    for a, b in FRAC.items():
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s)


def inches_of(rem):
    """Sum of whole and fractional inch tokens in a remainder like '- 8 9/32in'. None if odd."""
    rem = rem.strip().lstrip("-").strip()
    rem = re.sub(r'(inches|inch|in|")\s*$', "", rem).strip()
    if not rem:
        return 0.0
    toks = re.findall(r"\d+(?:\.\d+)?/\d+|\d+(?:\.\d+)?", rem)
    leftover = re.sub(r"\d+(?:\.\d+)?/\d+|\d+(?:\.\d+)?|[\s\-]|in|\"", "", rem)
    if leftover:
        return None
    whole = [t for t in toks if "/" not in t]
    if len(whole) > 1:
        return None
    v = 0.0
    for t in toks:
        if "/" in t:
            a, b = t.split("/")
            if float(a) >= float(b):
                return None
            v += float(a) / float(b)
        else:
            v += float(t)
    return v if v < 12 else None


def parse_ftin(s):
    """'12' - 1in' style. Returns feet (float) or None."""
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*'(.*)$", s)
    if not m:
        return None
    inch = inches_of(m.group(2))
    if inch is None:
        return None
    return float(m.group(1)) + inch / 12


def parse_rise(raw):
    """Returns (metres, note) where note is '' for a clean read, or (None, reason)."""
    if not isinstance(raw, str) or not raw.strip():
        return None, "no rise"
    s = norm_rise_text(raw)
    if s in WEIRD_RISES:
        f, i, why = WEIRD_RISES[s]
        return (f + i / 12) * FT, f"reinterpreted '{raw}': {why}"
    m = re.match(r"^([\d,]+(?:\.\d+)?)\s*mm\s*(?:\((.*)\))?\s*$", s)
    if m:
        mm = float(m.group(1).replace(",", ""))
        note = ""
        if m.group(2):
            ft = parse_ftin(m.group(2).replace(" '", "'"))
            if ft is None:
                note = f"mm value used; the feet-inches in brackets '{m.group(2)}' did not parse"
            elif abs(ft * FT - mm / 1000) > 0.03:
                note = (f"mm and feet-inches disagree ({mm:.0f} mm vs {m.group(2)} = {ft * FT * 1000:.0f} mm); "
                        f"used mm")
        return mm / 1000, note
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(?:ft|feet)\s*(?:(\d+)(?:\s+(\d+)/(\d+))?\s*inch(?:es)?)?\s*$", s)
    if m:
        ft = float(m.group(1))
        if m.group(2):
            ft += (float(m.group(2)) + (float(m.group(3)) / float(m.group(4)) if m.group(3) else 0)) / 12
        return ft * FT, ""
    ft = parse_ftin(s.replace(" '", "'"))
    if ft is not None:
        return ft * FT, ""
    return None, f"could not parse rise '{raw}'"


# ---------------------------------------------------------------- notes -> levels

LEVEL_WORDS = ["platform", "mezzanine", "mezzanie", "concourse", "bridge", "overpass", "underpass",
               "passageway", "landing", "lobby", "level"]
STREET_WORDS = re.compile(
    r"\b(st|street|ave|av|avenue|blvd|boulevard|rd|road|pl|place|plaza|pkwy|parkway|expwy|expressway|"
    r"hwy|highway|circle|sq|square|entrance|exit|corner|broadway|park|terrace|way|dr)\b", re.I)
TIERS = ["upper", "intermediate", "intemediate", "lower", "main", "central", "western", "eastern",
         "transfer", "path", "concourse"]
DIRS = re.compile(r"\b(uptown|downtown|northbound|southbound|[\w ]+?-bound|terminal)\b", re.I)


def norm_loc(s):
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"\((?:[ns][ew]|[ns][ew] corner|.*corner.*)\)", "", s)
    s = re.sub(r"\b([ns][ew]) corner\b|\bcorner\b|\bentrance\b|\bexit\b|\bnear\b", "", s)
    s = s.replace(" and ", " & ").replace("/", " & ")
    rep = [(r"\bavenue\b|\bave\b", "av"), (r"\bstreet\b", "st"), (r"\bfort\b", "ft"),
           (r"\bboulevard\b", "blvd"), (r"\b(\d+)(st|nd|rd|th)\b", r"\1"), (r"[^\w& ]", " ")]
    for a, b in rep:
        s = re.sub(a, b, s)
    parts = sorted(" ".join(p.split()) for p in s.split("&") if p.strip())
    return "&".join(parts)


def split_parens(seg):
    locs = re.findall(r"\(([^)]*)\)", seg)
    bare = re.sub(r"\([^)]*\)?", " ", seg)
    return " ".join(bare.split()), [l for l in locs if not re.fullmatch(r"\s*[NS][EW]( corner)?\s*", l, re.I)]


def classify(seg):
    """One end of a machine: dict(kind, tier, routes, dir, loc, text)."""
    text = seg.strip().strip(".")
    bare, locs = split_parens(text)
    low = bare.lower().replace("grand concourse", "grand cnc av")  # a street, not a concourse
    low = re.sub(r"\bplatforms\b", "platform", low)
    kind = None
    # the level word that comes first is the head noun: 'central mezzanine above 7 platform' is a
    # mezzanine, 'Platform level for Shuttle' a platform, 'concourse level of Penn' a mezzanine
    first = None
    for w in LEVEL_WORDS:
        mm = re.search(rf"\b{w}\b", low)
        if mm and (first is None or mm.start() < first[0]):
            first = (mm.start(), w)
    if first:
        w = first[1]
        kind = {"mezzanie": "mezzanine", "lobby": "mezzanine", "level": "platform",
                "concourse": "mezzanine"}.get(w, w)
    if kind is None and (low.startswith("street") or STREET_WORDS.search(low)):
        kind = "street"
    if kind is None:
        return None
    tier = ""
    if kind == "mezzanine":
        for t in TIERS:
            if re.search(rf"\b{t}\b", low):
                tier = {"intemediate": "intermediate"}.get(t, t)
                break
    routes = []
    if kind in ("platform", "mezzanine"):
        head = re.split(r"\b(platform|mezzanine|mezzanie|concourse)\b", bare, flags=re.I)[0]
        routes = re.findall(r"(?<![\w'])([1-7A-Z](?:/[1-7A-Z])*)(?![\w'])", head)
        routes = sorted({r for grp in routes for r in grp.split("/")} - {"I"})
    d = ""
    if kind == "platform":
        m = DIRS.search(bare)
        if m:
            d = re.sub(r"^(and|the)\s+", "", m.group(1).lower().strip())
    loc = ""
    if kind == "street":
        loc = norm_loc(text)
    elif locs:
        loc = norm_loc(locs[0])
    return {"kind": kind, "tier": tier, "routes": routes, "dir": d, "loc": loc, "text": text}


def parse_notes(notes):
    """'A to B[ for X exit| and C| ...]' -> (A, B, extras, exit_loc). None if no 'to'."""
    if not isinstance(notes, str):
        return None
    t = " ".join(notes.split()).rstrip(".")
    # mask ' to ' inside parentheses
    masked = re.sub(r"\([^)]*\)", lambda m: m.group(0).replace(" to ", " t0 "), t)
    m = re.search(r"\s+to\s+", masked, re.I)
    if not m:
        # 'Dekalb Ave & Flatbush Ave (SW corner) and mezzanine' style
        m = re.search(r"\s+and\s+(?=\w*\s*mezzanine|\w*\s*platform)", masked, re.I)
    if not m:
        return None
    a = t[:m.start()]
    rest = t[m.end():]
    exit_loc = ""
    me = re.search(r"\bfor (?:the )?(.+?) exit\b(?:\s*\(([^)]*)\))?", rest)
    if me:
        exit_loc = me.group(1) + ((" " + me.group(2)) if me.group(2) else "")
    cut = re.split(r"\s+(?:for|via|to)\s+|\s+and access\b|,\s*and access\b|\s+and transfer\b", rest, maxsplit=1)
    b = cut[0]
    extras = []
    mm = re.match(r"(.*?)\s+and\s+(.*)$", b)
    if mm and re.search(r"platform|mezzanine", mm.group(2), re.I) and re.search(r"platform|mezzanine", mm.group(1), re.I):
        b, extras = mm.group(1), [mm.group(2)]
    return a, b, extras, exit_loc


# ---------------------------------------------------------------- station-specific readings

# equipment code -> dict with any of:
#   a, b         replacement texts for the two ends
#   skip         reason the machine is left out of the chains (goes to the unparsed report)
#   up           'a' or 'b': which end is higher (when the kinds can't say)
#   sd           extra sd (m) on this edge
#   note         carried into the level notes
OVERRIDES = {
    # 191 St (1). Four elevators from the St Nicholas Av hilltop. EL105/106 (2 landings, 162'4")
    # reach the mezzanine above the platforms; EL103/104 (3 landings, 172'9") travel 10'5" further,
    # which is about a footbridge-to-platform drop, so their bottom landing is read as platform
    # level (Wikipedia: station ~173 ft below the street, some sources 180 ft).
    "EL103": {"b": "lower landing at platform level", "note": "191 St: the 3-landing elevators (172'9\") are read as reaching platform level, 10'5\" below the mezzanine the 2-landing ones (162'4\") reach"},
    "EL104": {"b": "lower landing at platform level"},
    # 181 St (1). 'mezzanine to bridge': the mezzanine is fare control under St Nicholas Av,
    # reached from the street by stairs (no rise in the data); the bridge is a footbridge over
    # the tracks. Street-to-mezzanine is an assumption (STREET_TO, below).
    # 168 St (1): 'transfer mezzanine' is the upper mezzanine shared with the A/C, reached from the
    # street by EL139, which has no rise: default street-to-mezzanine drop applies.
    # 181 St (A): upper mezzanine at 184 St is the elevator lobby at Fort Washington Av street level.
    # 34 St-Hudson Yards: ES633-635 (28', 28', 24') are listed as upper (35 St) to lower mezzanine,
    # beside ES636 (83.3 ft) between the same two levels, and the main bank's upper-to-lower is
    # 81'8"; the short ones must serve an intermediate landing. Left out.
    "ES633": {"skip": "listed as 35 St upper mezzanine to lower mezzanine at 28 ft, but ES636 joins the same levels at 83.3 ft and the main bank does 81'8\"; must serve an intermediate landing"},
    "ES634": {"skip": "as ES633"},
    "ES635": {"skip": "as ES633 (24 ft)"},
    # ES626: 'Hudson Blvd E to upper mezzanine' 12' beside ES627-629 at 30' for the same ends.
    # The park over the yards is terraced; keep it but loosely.
    "ES626": {"sd": 3.0, "note": "34 St-Hudson Yards: ES626 says 12 ft from Hudson Blvd E to the upper mezzanine where ES627-629 say 30 ft; given a 3 m sd"},
    # 72 St (Q): ES290 '69 St & 2nd Ave (NE corner) to upper mezzanine' carries 68'4", the rise of
    # ES288/289 (lower to upper mezzanine) beside it; the 70 St entrance does street-upper in 12'5".
    # ES288/289 serve the 69 St exit, ES291 the 70 St exit: one upper mezzanine (69-70 St block).
    "ES288": {"a": "lower mezzanine", "b": "upper mezzanine (70 St & 2nd Ave)"},
    "ES289": {"a": "lower mezzanine", "b": "upper mezzanine (70 St & 2nd Ave)"},
    "ES290": {"skip": "72 St (Q): street-to-upper-mezzanine escalator carries 68'4\", the rise of the lower-to-upper escalators beside it (copy error; the 70 St entrance does the same step in 12'5\")"},
    # Flatbush Av (2/5): 51'4" street to platform is identical to Grand Central's EL244 and twice
    # Church Av's 17'7" street-to-platform on the same shallow cut-and-cover line.
    "EL319": {"skip": "Flatbush Av-Brooklyn College: 51'4\" street-to-platform is the exact rise of Grand Central's EL244 and implausible for this shallow terminal (Church Av next door: 17'7\"); probably a copy error"},
    # W 4 St: upper level A/C/E, lower level B/D/F/M; the 3-landing elevators span both platforms.
    "EL334": {"a": "downtown A/C/E platform", "b": "downtown B/D/F/M platform", "up": "a"},
    "EL335": {"a": "uptown A/C/E platform", "b": "uptown B/D/F/M platform", "up": "a"},
    "ES325": {"up": "b"},
    "ES326": {"up": "b"},
    # Sutphin Blvd: E upper level, J/Z lower (ES436 'J/Z platform to E platform ... and access to
    # mezzanine' goes up; at Jamaica Center the J/Z are 28 ft under the mezzanine, the E 11 ft).
    "ES436": {"up": "b"},
    "ES437": {"up": "a"},
    # 59 St (4/5/6): 6 platforms above the 4/5 platforms.
    "ES211": {"up": "b"},
    "ES212": {"up": "b"},
    # Lexington Av/59 St: the BMT 60 St tunnel platform is below the IRT 6 platforms, and the
    # N/R/W transfer mezzanine is reached by going down from the 6.
    "ES215": {"up": "b"},
    "ES213": {"up": "b"},
    "ES214": {"up": "b"},
    # Crown Heights-Utica Av is a two-level station (one direction per level)
    "EL323": {"skip": "Crown Hts-Utica Av: 'mezzanine to platforms for service in both directions' serves both levels of a two-level station; can't tell which the rise is to"},
    # Broadway-Lafayette: the 6 (Bleecker St) is above the B/D/F/M
    "ES300": {"up": "b"},
    # 59 St-Columbus Circle: one elevator to the downtown IND platform and the 1 platform, which
    # are on different levels
    "EL622": {"skip": "59 St-Columbus Circle: 'mezzanine to downtown A/B/C/D & 1 platforms' serves two platforms on different levels; can't tell which the rise is to"},
    # Pelham Bay Park: one Amendola Plaza entrance
    "EL121": {"a": "Amendola Plaza entrance"},
    # 161 St: B/D mezzanine (underground) up to the elevated 4.
    "EL132": {"b": "Woodlawn-bound 4 platform", "up": "b"},
    "EL133": {"a": "B/D mezzanine", "up": "b"},
    "ES113": {"a": "B/D mezzanine", "b": "4 mezzanine", "up": "b"},
    "ES114": {"a": "4 mezzanine", "b": "B/D mezzanine", "up": "a"},
    "EL131": {"b": "B/D mezzanine"},
    "EL134": {"a": "B/D mezzanine"},
    "EL135": {"a": "B/D mezzanine"},
    # Jackson Hts: 7 is elevated above the E/F/M/R. 'upper mezzanine' for 7 sits at street level.
    "ES452": {"a": "E/F/M/R lower mezzanine", "b": "7 upper mezzanine", "up": "b"},
    "ES453": {"a": "7 upper mezzanine", "b": "E/F/M/R lower mezzanine", "up": "a"},
    "ES449": {"a": "7 underpass", "b": "E/F/M/R lower mezzanine", "up": "a"},
    "ES450": {"a": "E/F/M/R lower mezzanine", "b": "7 underpass", "up": "b"},
    "ES451": {"a": "E/F/M/R lower mezzanine", "b": "7 underpass", "up": "b"},
    "EL421": {"skip": "Jackson Hts: 3-landing elevator (transfer mezzanine, Manhattan-bound platform, mezzanine for Queens-bound); can't tell which two landings 19'3\" spans, and reading it as mezzanine-to-platform puts the two E/F/M/R platforms 7 m apart"},
    "EL422": {"skip": "Jackson Hts: 'mezzanine to Forest Hills-bound platform', but which mezzanine is unclear (its coordinates are at Sutphin Blvd) and it doesn't join a chain from the street"},
    "EL420": {"skip": "74 St-Broadway: no Notes, 3 landings (street, mezzanine, platform?); can't tell its ends"},
    # Broadway Junction: A/C mezzanine (below the els) up to J/Z/L upper mezzanine.
    "ES330": {"a": "A/C mezzanine", "b": "J/Z/L upper mezzanine", "up": "b"},
    "ES331": {"a": "J/Z/L upper mezzanine", "b": "A/C mezzanine", "up": "a"},
    "ES347": {"a": "A/C mezzanine", "b": "J/Z/L upper mezzanine", "up": "b"},
    # Myrtle-Wyckoff: L below, M elevated.
    "ES332": {"up": "b"},
    "ES333": {"up": "a"},
    # Stillwell Av: overpass above the platforms (at grade terminal).
    "EL701": {"up": "b"},
    "EL702": {"up": "b"},
    # Smith-9 Sts: 'upper mezzanine' is the platform-level mezzanine of the viaduct station.
    # (coordinates of ES314/ES316 are elsewhere in the city; replaced by the station's.)
    # Bowling Green: uptown 4/5 mezzanine above the lower mezzanine. The lower mezzanine (1977) is
    # under the tracks: EL710 runs 'plaza ... to uptown platform and lower mezzanine' (street,
    # platform, lower mezzanine, in that order) and EL711 'lower mezzanine to Brooklyn-bound
    # platform', so the four 23 ft escalators rise from the lower mezzanine to the platforms. Read
    # downward (as before 2026-11-02) they put the platforms 19 m down, 14 m below the line's
    # neighbours; read upward the platforms are 16 + 24 - 23 = 17 ft below the street.
    "ES336": {"up": "b"},
    "ES337": {"up": "a"},
    "ES340": {"up": "b", "note": "Bowling Green: the lower mezzanine is under the tracks (EL710/EL711 Notes), so its escalators rise to the platforms"},
    "ES341": {"up": "b"},
    "ES342": {"up": "b"},
    "ES343": {"up": "b"},
    # Penn Station 1/2/3: 'concourse level of Penn Station' is the Penn concourse. EL618 drops 46'10"
    # from the street to that concourse; EL216 joins it to the downtown 1 platform. Read downward the
    # platform came out 21.7 m deep, below Penn Station's own tracks, which the IRT crosses over; so
    # EL216 rises from the concourse to the platform (27 ft below the street). sd widened by 2 m:
    # 46'10" to the concourse is itself more than the usual Penn concourse depth.
    "EL216": {"a": "Penn Station concourse", "up": "b", "sd": 2.0,
              "note": "34 St-Penn (1/2/3): EL216 read as rising from the Penn concourse (46'10\" below the street) to the platform; read downward the platform was 21.7 m deep, under Penn Station's tracks. sd +2 m"},
    "EL618": {"b": "Penn Station concourse"},
    # 34 St-Penn (A/C/E): street to uptown C/E platform; then platform to lower mezzanine below.
    "EL226": {"up": "a"},
    "EL227": {"a": "Penn Station concourse IND"},
    # Times Sq: 7 platform belongs to the Flushing line station whatever MRN lists the machine.
    # Atlantic Av: Brooklyn-bound 2/3 platform up to the mezzanine; EL301 3-landing
    "EL304": {"b": "2/3/4/5 mezzanine"},
    "EL305": {"a": "2/3/4/5 mezzanine"},
    # Rockaway Blvd (elevated): street up to Manhattan-bound platform; transfer mezzanine between.
    "EL487": {"b": "Manhattan-bound A platform"},
    # Parkchester (elevated, 3 landings): street to platform via mezzanine.
    "EL521": {"b": "Pelham Bay Park-bound platform"},
    "EL522": {"b": "Manhattan-bound platform"},
    # Mosholu Pkwy: elevated; street to Woodlawn-bound platform via transfer mezzanine
    "EL523": {"b": "Woodlawn-bound platform"},
    # 149 St-Hostos: 'transfer mezzanine' below both platform levels? 4 is upper level (Jerome),
    # 2/5 lower; elevators from transfer mezzanine reach both. Two platforms at different levels on
    # one elevator: can't tell which landing the rise refers to.
    "EL101": {"skip": "149 St-Grand Concourse: elevator serves the 4 platform and the 2/5 platform, which are on different levels; can't tell which landing the rise is to"},
    "EL102": {"skip": "as EL101"},
    # Court St / Borough Hall: R lower mezzanine below the 2/3 mezzanine.
    "ES305": {"a": "R lower mezzanine", "b": "2/3 mezzanine"},
    "ES306": {"a": "2/3 mezzanine", "b": "R lower mezzanine"},
    # Fulton St: 2/3 mezzanine vs transfer mezzanine: transfer mezzanine (A/C/J/Z) is deeper.
    "ES367": {"up": "a"},
    "ES368": {"up": "b"},
    # South Ferry: '1 mezzanine' and '1/R/W mezzanine' are the same new-station mezzanine? The
    # Peter Minuit Plaza entrance serves both; keep as named.
    # World Trade Center EL326: 'Park Pl & Church St (SE corner to E Line access to downtown R/W' garbled.
    "EL326": {"skip": "World Trade Center EL326: Notes garbled ('Park Pl & Church St (SE corner to E Line access to downtown R/W'); can't tell the lower end"},
    # Jay St: 'A/C/F mezzanine', 'A/C/F/R mezzanine' and 'mezzanine for A/C/F/R service' are one.
    "ES357": {"b": "A/C/F/R mezzanine"},
    "ES356": {"a": "A/C/F/R mezzanine"},
    "ES318": {"b": "A/C/F/R mezzanine"},
    # 14 St (A/C/E) / 8 Av (L): one mezzanine at 8 Av & 14 St.
    "EL222": {"a": "A/C/E/L mezzanine"},
    "EL223": {"a": "A/C/E/L mezzanine"},
    "EL224": {"a": "A/C/E/L mezzanine"},
    "EL221": {"b": "A/C/E/L mezzanine"},
    # Herald Sq: 'N/Q/R/W mezzanine' (BMT mezzanine over Broadway) vs 'mezzanine' of EL209/210/211.
    "EL209": {"a": "N/Q/R/W mezzanine", "note": "34 St-Herald Sq: EL209/210's 'mezzanine' read as the N/Q/R/W mezzanine the escalators to the B/D/F/M start from (27'3\" vs 28-30')"},
    "EL210": {"a": "N/Q/R/W mezzanine"},
    "EL211": {"a": "N/Q/R/W mezzanine"},
    # Union Sq: one mezzanine over the BMT and L
    "EL217": {"skip": "14 St-Union Sq EL217: no Notes, and its coordinates are near 59 St-Columbus Circle"},
    # 57 St (F/M): EL621 'Mezzanine to M platform' flagged street access; it isn't.
    # Grand Central: EL244's 'mezzanine' is the main mezzanine (same shaft position as EL205/206).
    "EL244": {"a": "main mezzanine"},
    "ES207": {"a": "western mezzanine"},
    "ES209": {"b": "western mezzanine"},
    "ES203": {"a": "eastern mezzanine above 7 platform"},
    "ES204": {"b": "eastern mezzanine above 7 platform"},
    # Lexington Av/53 St: ES243/244 and ES269 start at the 6 mezzanine at 53 St & Lex; EL245,
    # ES245/246 at the 3rd Av end mezzanine.
    "ES243": {"b": "6 mezzanine (53 St & Lexington Av)"},
    "ES244": {"b": "6 mezzanine (53 St & Lexington Av)"},
    "ES269": {"a": "6 mezzanine (53 St & Lexington Av)"},
    "EL245": {"a": "mezzanine (53 St & 3 Av)"},
    # Lexington Av/63 St: the 3rd Av entrance's 'mezzanine' is the east end of the upper mezzanine?
    # Unknown, keep separate.
    # Times Sq: ES216/217 upper mezzanine (41 St & 7 Av) to 7 platform; ES218 the same.
    "ES218": {"b": "upper mezzanine (41 St & 7 Ave)"},
    # Cortlandt St: 'mezzanine at Oculus main level'
    # Penn 34 St EL225: street to uptown C/E platform (rise to the platform)
    "EL225": {"b": "uptown C/E platform"},
    # Jamaica-179 / Bedford Pk / etc: plain.
    # Woodhaven Blvd, Sheepshead Bay etc: 'Street to platform' 2-landing: fine.
    # Atlantic EL301 'to mezzanine for D/N/R ... via Brooklyn-bound 2/3 platform'
    "EL301": {"b": "mezzanine (4th Ave & Pacific St)"},
    "EL302": {"a": "mezzanine (4th Ave & Pacific St)"},
    "EL303": {"a": "mezzanine (4th Ave & Pacific St)"},
    # Pelham Bay Park ES121: 'mezzanine to platform' but flagged street access; fine.
    # Jackson Hts EL422: coordinates at Sutphin Blvd; replaced by station's.
    # Atlantic ES310: B/Q lower mezzanine to upper mezzanine (LIRR level)
    # Whitehall: ES311 '1/R/W upper mezzanine' and ES351 'upper mezzanine' are one
    "ES311": {"a": "upper mezzanine (Water St & Whitehall St)"},
    "ES351": {"b": "upper mezzanine (Water St & Whitehall St)"},
    # South Ferry: ES369/370 'mezzanine for 1 and R/W' and ES371/372 '1/R/W mezzanine' are one
    "ES369": {"b": "1/R/W mezzanine"},
    "ES370": {"a": "1/R/W mezzanine"},
    "ES373": {"a": "mezzanine (State St & Pearl St)", "note": "South Ferry: ES373 '1 mezzanine to Peter Minuit Plaza' 33'7\" read as the State St end mezzanine (its coordinates are 40 m west of the 1/R/W mezzanine bank, and 33'7\" doesn't fit the 20' of ES369/370)"},
    "ES375": {"a": "mezzanine (State St & Pearl St)"},
}

# MTA stations-table Structure values that are wrong for what the rises need to know (which way
# is up). Sheepshead Bay is on the Brighton Line embankment (OSM: bridge/embankment; its
# street-to-platform elevators rise from Sheepshead Bay Rd); the table says Open Cut, as it does
# for Avenue H to Neck Rd, which have no rises here.
STRUCTURE_FIX = {"54": "Embankment"}

# Assumed vertical offsets for levels reached without a rise. kind -> (metres above platform, sd)
ABOVE_PLATFORM = {"bridge": (4.0, 0.8), "overpass": (3.0, 1.5)}
# Assumed street -> topmost mezzanine when no street-end machine has a rise. Filled from the data
# in main() (median and spread of plain street-to-mezzanine rises), with a floor on the sd.
DEFAULT_MEZZ = {"mu": 4.7, "sd": 2.0}
# Per-complex assumptions: complex MRN -> list of (from_label, to_label, metres, sd, note)
# from is above to by `metres`. Labels are node texts as classified.
EXTRA_EDGES = {
    # 181 St (1): street -> fare-control mezzanine by stairs. Wikipedia gives 120 ft to the
    # platforms from the street; 103'5" + ~13 ft bridge-to-platform leaves ~1 m for the stairs,
    # so assume 1.5 +- 1.5 m.
    "301": [("street@station", "mezzanine", 1.5, 1.5,
             "181 St (1): street to the fare-control mezzanine is stairs with no rise in the data; assumed 1.5 +- 1.5 m")],
    # 181 St (A): the 184 St upper mezzanine is the elevator lobby at Fort Washington Av.
    "146": [("street@EL118", "upper mezzanine (184 St & Ft Washington Av)", 0.0, 1.0,
             "181 St (A): the 184 St upper mezzanine is the elevator lobby at Fort Washington Av street level (0 +- 1 m)"),
            ("lower mezzanine", "platform", 4.0, 1.0,
             "181 St (A): platforms reached from the lower mezzanine by a concourse over the tracks and stairs (Wikipedia), no rise in the data; assumed 4.0 +- 1.0 m, as a footbridge")],
    # W 8 St: ES335 rises 37 ft from this mezzanine to the upper-deck Q platform, about the full
    # height of a double-deck el, so the mezzanine is near street level, not one storey up.
    "57": [("mezzanine (W 5 St & Brighton Ave)", "street@station", 0.5, 1.5,
            "W 8 St: 37 ft from the mezzanine to the upper-deck platform is a whole double-deck el, so the mezzanine is taken at street level (+0.5 +- 1.5 m)")],
    # 34 St-Hudson Yards: no rise from the lower mezzanine to the platform (stairs, EL600).
    # Wikipedia: lower mezzanine 109 ft deep, platform 125 ft: 16 ft.
    "471": [("lower mezzanine", "platform", 16 * FT, 1.0,
             "34 St-Hudson Yards: lower mezzanine to platform has no rise in the data; 16 ft +- 1 m from Wikipedia (lower mezzanine 109 ft deep, platform 125 ft)")],
    # 181 St (A): a concourse runs over the platforms between the two mezzanines, with stairs
    # down to each platform (Wikipedia); EL517/518 (mezzanine to platform) have no rise.
    # W 4 St: the top mezzanine is over the upper (A/C/E) level, reached by stairs.
    "167": [("mezzanine (W 3rd St & 6th Ave)", "uptown A/C/E platform", 3.5, 1.0,
             "W 4 St: stairs from the mezzanine down to the upper (A/C/E) level, no rise in the data; assumed 3.5 +- 1.0 m (typical mezzanine-to-platform rise, 10-12 ft, in this data)"),
            ("mezzanine (W 3rd St & 6th Ave)", "downtown A/C/E platform", 3.5, 1.0,
             "W 4 St: as for the uptown side")],
    # Rockaway Blvd and Mosholu Pkwy: one platform is reached from the street, the other only from
    # the transfer mezzanine; side platforms of one el station are at one level.
    "192": [("Manhattan-bound A platform", "Lefferts Blvd, Far Rockaway, and Rockaway Park-bound A platform", 0.0, 0.3,
             "Rockaway Blvd: the two platforms taken at one level (+- 0.3 m)")],
    "379": [("Woodlawn-bound platform", "Manhattan-bound Platform", 0.0, 0.3,
             "Mosholu Pkwy: the two platforms taken at one level (+- 0.3 m)")],
    # Jackson Hts / 74 St: the 7's upper mezzanine is the street-level station house.
    "616": [("street@station", "7 upper mezzanine", 0.0, 1.5,
             "74 St-Broadway: the 7's upper mezzanine is taken as the street-level station house (0 +- 1.5 m)")],
    # Broadway Junction: the A/C mezzanine on Van Sinderen Av is the street entrance level.
    "621": [("street@ES330", "A/C mezzanine", 0.0, 1.0,
             "Broadway Junction: the A/C mezzanine (Van Sinderen Av) is at the street entrance, S=YES (0 +- 1 m)")],
    # 191 St: lower landing of EL103/104 at platform level
    "300": [("lower landing at platform level", "platform", 0.0, 1.0,
             "191 St: bottom landing of EL103/104 taken as platform level (0 +- 1 m)")],
}


# ---------------------------------------------------------------- ground

class Ground:
    def __init__(self):
        self.tiles = []
        for f in sorted((RAW / "dem").glob("dem_*.tif")):
            ds = rasterio.open(f)
            self.tiles.append(ds)
        self.cache = {}

    def window(self, lon, lat, half=2):
        x, y = to_utm.transform(lon, lat)
        for ds in self.tiles:
            b = ds.bounds
            if b.left <= x < b.right and b.bottom < y <= b.top:
                r, c = ds.index(x, y)
                key = (ds.name, r, c, half)
                if key not in self.cache:
                    win = rasterio.windows.Window(c - half, r - half, 2 * half + 1, 2 * half + 1)
                    a = ds.read(1, window=win, boundless=True, fill_value=-9999).astype(float)
                    a[a < -100] = np.nan
                    self.cache[key] = a
                return self.cache[key]
        return None

    def at(self, lon, lat):
        """(median, sd) of ground in a 15 m window; sd grows with the window's spread."""
        a = self.window(lon, lat)
        if a is None or np.isnan(a).all():
            return None, None, None
        v = a[~np.isnan(a)]
        p10, p50, p90 = np.percentile(v, [10, 50, 90])
        spread = p90 - p10
        return float(p50), float(math.hypot(0.3, 0.4 * spread)) + 0.2, float(spread)

    def street_by_embankment(self, lon, lat):
        """Ground for a street-end machine of a station on an embankment. The bare-earth DEM keeps
        an embankment (it is earth), so the ground at the machine's coordinate (often the station's
        own point, on the embankment) is the track bed, not the street the machine starts from. The
        street beside or under the embankment is taken as the 10th percentile within EMB_STREET_R."""
        a = self.window(lon, lat, half=int(EMB_STREET_R / 3))
        if a is None or np.isnan(a).all():
            return None, None, None
        v = a[~np.isnan(a)]
        p5, p25 = np.percentile(v, [10, 30])
        return float(p5), 0.5 + 0.3 * float(p25 - p5), float(p25 - p5)


EMB_STREET_R = 150.0   # m: search radius for the street beside an embankment station (Sheepshead Bay Rd is 130 m from the station point)


# ---------------------------------------------------------------- build

def fmt_ft(raw):
    return norm_rise_text(raw) if isinstance(raw, str) else ""


def main():
    el = pd.read_csv(RAW / "mta_elevators.csv", dtype=str)
    st = pd.read_csv(RAW / "mta_stations.csv", dtype=str)
    st["Station ID"] = st["Station ID"].str.lstrip("0")
    st["Complex ID"] = st["Complex ID"].str.lstrip("0")
    for sid, s in STRUCTURE_FIX.items():
        st.loc[st["Station ID"] == sid, "Structure"] = s
    el["Station MRN"] = el["Station MRN"].str.lstrip("0")
    el["Station Complex MRN"] = el["Station Complex MRN"].str.lstrip("0")
    ground = Ground()

    # station rows: (mrn, line) -> info; complex -> rows
    st_rows = {}
    by_complex = collections.defaultdict(list)
    for _, r in st.iterrows():
        d = {"mrn": r["Station ID"], "complex": r["Complex ID"], "line": r["Line"], "name": r["Stop Name"],
             "structure": r["Structure"], "routes": set(r["Daytime Routes"].split()),
             "lat": float(r["GTFS Latitude"]), "lon": float(r["GTFS Longitude"])}
        st_rows[(d["mrn"], d["line"])] = d
        by_complex[d["complex"]].append(d)
    rows_by_mrn = collections.defaultdict(list)
    for d in st_rows.values():
        rows_by_mrn[d["mrn"]].append(d)

    elevated = {"Elevated", "Viaduct", "Embankment", "At Grade"}

    unparsed = []   # (complex, station, equip, rise, notes, reason)
    rise_log = []   # (equip, station, raw, note)
    coord_log = []
    edges = collections.defaultdict(list)  # complex -> edges

    def station_for(mrn, routes, cx):
        """(mrn, line) of the station a level with these route letters belongs to."""
        cands = by_complex.get(cx, [])
        if routes:
            best = [(len(set(routes) & d["routes"]), d) for d in cands]
            top = max((b for b, _ in best), default=0)
            hit = [d for b, d in best if b == top and b > 0]
            if len(hit) == 1:
                return hit[0], ""
            if len(hit) > 1:
                own = [d for d in hit if d["mrn"] == mrn]
                if len(own) == 1:
                    return own[0], ""
                return None, f"routes {'/'.join(routes)} match several stations"
        own = rows_by_mrn.get(mrn, [])
        if len(own) == 1:
            return own[0], ""
        if len(own) > 1:
            return None, "station has several lines and the level names no route"
        return None, "station not in stations table"

    # ---- empirical default for street -> plain mezzanine
    mezz_rises = []

    machines = {}
    for _, r in el.iterrows():
        eq = r["Equipment Code"]
        is_el = r["Elevator or Escalator"] == "Elevator"
        raw = r["Elevator Rise"] if is_el else r["Escalator Rise"]
        cx = r["Station Complex MRN"]
        mrn = r["Station MRN"]
        sname = str(r["Station Description"]).replace(" - Station", "")
        notes = r["Notes"]
        ov = OVERRIDES.get(eq, {})
        rise, rnote = parse_rise(raw)
        if rise is None:
            if isinstance(raw, str):
                unparsed.append((cx, sname, eq, raw, notes, rnote))
                rise_log.append((eq, sname, raw, rnote))
            else:
                unparsed.append((cx, sname, eq, "", notes, "no rise in the inventory"))
            continue
        if rnote:
            rise_log.append((eq, sname, raw, rnote))
        if "skip" in ov:
            unparsed.append((cx, sname, eq, raw, notes, "left out: " + ov["skip"]))
            continue
        pn = parse_notes(notes)
        if pn is None and not ("a" in ov and "b" in ov):
            unparsed.append((cx, sname, eq, raw, notes, "Notes don't say 'A to B'"))
            continue
        a_txt, b_txt, extras, exit_loc = pn if pn else (None, None, [], "")
        a_txt = ov.get("a", a_txt)
        b_txt = ov.get("b", b_txt)
        A, B = classify(a_txt), classify(b_txt)
        if A is None or B is None:
            unparsed.append((cx, sname, eq, raw, notes, f"can't tell what level '{a_txt if A is None else b_txt}' is"))
            continue
        if exit_loc and "b" not in ov and "a" not in ov:
            for X in (B, A):
                if X["kind"] == "mezzanine" and not X["loc"]:
                    X["loc"] = norm_loc(exit_loc)
                    break
        # an unnamed mezzanine on a street machine is the mezzanine at that street
        for X, Y in ((A, B), (B, A)):
            if X["kind"] == "mezzanine" and Y["kind"] == "street" and not X["loc"] and not X["routes"]:
                X["loc"] = Y["loc"]
        # coordinates
        try:
            lat, lon = float(r["X Coordinate"]), float(r["Y Coordinate"])
        except (TypeError, ValueError):
            lat = lon = float("nan")
        own = rows_by_mrn.get(mrn, [])
        if own:
            sx, sy = to_utm.transform(own[0]["lon"], own[0]["lat"])
            if np.isnan(lat):
                d = float("inf")
            else:
                mx, my = to_utm.transform(lon, lat)
                d = math.hypot(mx - sx, my - sy)
            if d > 400:
                coord_log.append((eq, sname, r["X Coordinate"], r["Y Coordinate"], d))
                lat, lon = own[0]["lat"], own[0]["lon"]
        machines[eq] = {"eq": eq, "cx": cx, "mrn": mrn, "station": sname, "raw": raw, "rise": rise,
                        "rnote": rnote, "A": A, "B": B, "lat": lat, "lon": lon, "ov": ov,
                        "notes": notes, "extras": extras, "is_el": is_el}
        if A["kind"] == "street" and B["kind"] == "mezzanine" and B["tier"] in ("", "upper") or \
           B["kind"] == "street" and A["kind"] == "mezzanine" and A["tier"] in ("", "upper"):
            srow = own[0] if own else None
            if srow is not None and srow["structure"] not in elevated:
                mezz_rises.append((cx, is_el, rise))

    # One value per complex (an escalator bank of five shouldn't count five times). Entrances
    # deeper than 12 m are the deep-station kind, not the typical mezzanine under the street.
    per_cx = collections.defaultdict(list)
    for cx_, is_el_, x in mezz_rises:
        if x < 12:
            per_cx[cx_].append(x)
    plain = np.array([np.median(v) for v in per_cx.values()])
    elev = np.array([np.median([x for c, e, x in mezz_rises if c == cx_ and e and x < 12])
                     for cx_ in per_cx if any(c == cx_ and e for c, e, x in mezz_rises)])
    if len(plain) >= 10:
        DEFAULT_MEZZ["mu"] = float(np.median(plain))
        DEFAULT_MEZZ["sd"] = max(2.0, 1.5 * float(np.std(plain)))
    print(f"street-to-mezzanine (underground, < 12 m, one per complex): n={len(plain)}, median {np.median(plain):.2f} m, "
          f"sd {np.std(plain):.2f} m; elevators only n={len(elev)} median {np.median(elev):.2f} m "
          f"-> default {DEFAULT_MEZZ['mu']:.2f} +- {DEFAULT_MEZZ['sd']:.2f} m")

    # ---- nodes
    def node_key(X, m):
        """Hashable node id for one end of machine m."""
        cx = m["cx"]
        if X["kind"] == "street":
            return ("street", m["eq"])
        if X["kind"] == "platform":
            srow, why = station_for(m["mrn"], X["routes"], cx)
            if srow is None:
                return ("?", why)
            d = X["dir"]
            sub = tuple(X["routes"]) if X["routes"] and set(X["routes"]) < srow["routes"] else ()
            return ("platform", cx, srow["mrn"], srow["line"], d, sub)
        if X["kind"] in ("bridge", "overpass", "underpass", "passageway", "landing"):
            q = "/".join(X["routes"]) or m["mrn"]
            if "landing" in X["text"].lower() and "platform level" in X["text"].lower():
                return ("landing", cx, "platform-level", m["mrn"])
            return (X["kind"], cx, q, X["loc"])
        # mezzanine
        if X["routes"]:
            q = "r:" + "/".join(X["routes"])
        elif X["loc"]:
            q = "l:" + X["loc"]
        else:
            q = "m:" + m["mrn"]
        return ("mezzanine", cx, X["tier"], q)

    def structure_of(key, m):
        if key[0] == "platform":
            return st_rows[(key[2], key[3])]["structure"]
        own = rows_by_mrn.get(m["mrn"], [])
        if key[0] == "mezzanine" and key[3].startswith("r:"):
            srow, _ = station_for(m["mrn"], key[3][2:].split("/"), m["cx"])
            if srow:
                return srow["structure"]
        return own[0]["structure"] if own else "Subway"

    TIER_H = {"upper": 3, "main": 2.5, "intermediate": 2, "": 1, "transfer": 1, "central": 1, "western": 1,
              "eastern": 1, "path": 0.5, "concourse": 1, "lower": 0}

    def pseudo_h(key, X, m):
        """Rough height for ordering: negative below street for subway, positive for elevated."""
        if key[0] == "street":
            return 0.0
        el_ = structure_of(key, m) in elevated
        k = key[0]
        if k == "mezzanine":
            t = TIER_H.get(key[2], 1)
            return (1 + 0.1 * t) if el_ else (-4 + 0.5 * t)
        if k in ("bridge", "overpass"):
            return 3.5 if el_ else -4.5
        if k == "underpass":
            return 0.5 if el_ else -6
        if k in ("passageway", "landing"):
            return 2 if el_ else -4.8
        if k == "platform":
            return 3 if el_ else -5
        return 0.0

    def lbl(k):
        if k[0] == "street":
            return "street"
        if k[0] == "platform":
            srow = st_rows[(k[2], k[3])]
            routes = k[5] if k[5] else sorted(srow["routes"])
            return f"{k[4] + ' ' if k[4] else ''}{'/'.join(routes)} platform".strip()
        if k[0] == "mezzanine":
            q = k[3][2:] if k[3][:2] in ("l:", "r:") else ""
            return f"{k[2] + ' ' if k[2] else ''}mezzanine{' (' + q + ')' if q else ''}"
        if k[0] == "landing":
            return "lower landing"
        return f"{k[0]}{' (' + k[3] + ')' if len(k) > 3 and k[3] else ''}"

    # Merge mezzanines that are one level under different names, per complex and tier:
    #  * 'mezzanine' (unnamed, of station X) and 'X-routes mezzanine' (named by X's own lines);
    #  * an unnamed mezzanine and one named by an exit, when it is the only named one of that
    #    tier, or the nearest by machine position within 150 m with the next one 75 m farther.
    all_keys = collections.defaultdict(set)
    kpos = collections.defaultdict(list)
    for m in machines.values():
        for X in (m["A"], m["B"]):
            k = node_key(X, m)
            if k[0] == "mezzanine":
                all_keys[(k[1], k[2])].add(k)
                if not np.isnan(m["lat"]):
                    kpos[k].append(to_utm.transform(m["lon"], m["lat"]))
    parent = {}

    def find(k):
        while parent.get(k, k) != k:
            k = parent[k]
        return k

    alias_note = {}

    def union(a, b, why):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            alias_note.setdefault(rb, []).append(why)

    def mezz_name(k):
        return f"{k[2] + ' ' if k[2] else ''}mezzanine{' (' + k[3][2:] + ')' if k[3][:2] in ('l:', 'r:') else ''}"

    for (cx, tier), ks in all_keys.items():
        unnamed = [k for k in ks if k[3].startswith("m:")]
        routed = [k for k in ks if k[3].startswith("r:")]
        named = [k for k in ks if k[3].startswith("l:")]
        for u in unnamed:
            mrn = u[3][2:]
            own_routes = set().union(*[d["routes"] for d in rows_by_mrn.get(mrn, [])]) if rows_by_mrn.get(mrn) else set()
            hits = [r for r in routed if set(r[3][2:].split("/")) <= own_routes]
            if len(hits) == 1:
                union(u, hits[0], f"'{mezz_name(u)}' and '{mezz_name(hits[0])}' read as one level")
        for u in unnamed:
            if not named:
                continue
            if len(named) == 1:
                union(u, named[0], f"the unnamed '{mezz_name(u)}' is taken to be the one at {named[0][3][2:]} (the only named one of that tier)")
                continue
            if not kpos[u]:
                continue
            ux, uy = np.mean(kpos[u], axis=0)
            ds = sorted((math.hypot(*(np.mean(kpos[n], axis=0) - (ux, uy))), n) for n in named if kpos[n])
            if ds and ds[0][0] < 150 and (len(ds) == 1 or ds[1][0] - ds[0][0] > 75):
                union(u, ds[0][1], f"the unnamed '{mezz_name(u)}' is taken to be the one at {ds[0][1][3][2:]} "
                                   f"(its machines are {ds[0][0]:.0f} m away; the next named one {ds[1][0]:.0f} m)"
                      if len(ds) > 1 else f"the unnamed '{mezz_name(u)}' is taken to be the one at {ds[0][1][3][2:]}")
    alias = {k: find(k) for ks in all_keys.values() for k in ks if find(k) != k}
    alias_note = {k: " ; ".join(v) for k, v in alias_note.items()}

    # ---- edges
    for m in machines.values():
        ka, kb = node_key(m["A"], m), node_key(m["B"], m)
        ka, kb = alias.get(ka, ka), alias.get(kb, kb)
        if ka[0] == "?" or kb[0] == "?":
            unparsed.append((m["cx"], m["station"], m["eq"], m["raw"], m["notes"],
                             f"platform's line unclear: {ka[1] if ka[0] == '?' else kb[1]}"))
            continue
        if ka == kb:
            unparsed.append((m["cx"], m["station"], m["eq"], m["raw"], m["notes"], "both ends read as the same level"))
            continue
        up = m["ov"].get("up")
        if up is None:
            ha, hb = pseudo_h(ka, m["A"], m), pseudo_h(kb, m["B"], m)
            if ha == hb:
                unparsed.append((m["cx"], m["station"], m["eq"], m["raw"], m["notes"],
                                 "can't tell which end is higher (same kind of level at both ends)"))
                continue
            up = "a" if ha > hb else "b"
        hi, lo = (ka, kb) if up == "a" else (kb, ka)
        # a rise is read off drawings to the inch, but which floor it is measured to (landing sill,
        # finished floor, a step or two) varies: 0.10 m + 0.5 % of the rise
        sd = 0.10 + 0.005 * m["rise"] + (0.25 if m["rnote"] else 0) + m["ov"].get("sd", 0)
        edges[m["cx"]].append({"hi": hi, "lo": lo, "d": m["rise"], "sd": sd, "eq": m["eq"], "m": m,
                               "kind": "rise"})

    # ---- solve per complex
    out_rows = []
    report = []
    for cx, es in sorted(edges.items(), key=lambda kv: kv[0].zfill(4)):
        ms = {e["eq"]: e["m"] for e in es}
        notes_cx = []
        nodes = set()
        for e in es:
            nodes.add(e["hi"]); nodes.add(e["lo"])
        # node positions (for ground of synthetic streets and output lat/lon)
        pos = collections.defaultdict(list)
        for e in es:
            for k in (e["hi"], e["lo"]):
                pos[k].append((e["m"]["lon"], e["m"]["lat"]))
        crow = by_complex.get(cx, [None])[0]
        # assumption edges: bridge/overpass -> platform(s) of the same station
        plats = [k for k in nodes if k[0] == "platform"]
        for k in list(nodes):
            if k[0] in ABOVE_PLATFORM:
                linked = any((e["hi"] == k and e["lo"][0] == "platform") or (e["lo"] == k and e["hi"][0] == "platform") for e in es)
                if linked:
                    continue
                mrn = k[2]
                rows = rows_by_mrn.get(mrn, [])
                if len(rows) != 1:
                    continue
                pk = ("platform", cx, mrn, rows[0]["line"], "", ())
                mu, sd = ABOVE_PLATFORM[k[0]]
                es.append({"hi": k, "lo": pk, "d": mu, "sd": sd, "eq": f"assumed", "m": None, "kind": "assume",
                           "note": f"platform assumed {mu} +- {sd} m below the {k[0]} (no rise between them)"})
                pos[pk].extend(pos[k])
                nodes.add(pk)
        # complex-specific assumptions
        for fr, to, mu, sd, note in EXTRA_EDGES.get(cx, []):
            def find_lbl(text):
                if text.startswith("street@"):
                    tag = text.split("@")[1]
                    if tag == "station":
                        return ("street", f"station {cx}")
                    return ("street", tag)
                if text == "platform":
                    rows = by_complex[cx]
                    return ("platform", cx, rows[0]["mrn"], rows[0]["line"], "", ())
                X = classify(text)
                fake = {"cx": cx, "mrn": by_complex[cx][0]["mrn"], "eq": "x"}
                k = node_key(X, fake)
                return alias.get(k, k)
            kf, kt = find_lbl(fr), find_lbl(to)
            if kf[0] == "street" and kf[1].startswith("station"):
                pos[kf].append((crow["lon"], crow["lat"]))
            for kk, tx in ((kf, fr), (kt, to)):
                if kk not in nodes and kk[0] not in ("platform", "street"):
                    print(f"warning: complex {cx} assumption names a level not in the data: {tx} -> {kk}")
            es.append({"hi": kf, "lo": kt, "d": mu, "sd": sd, "eq": "assumed", "m": None, "kind": "assume", "note": note})
            nodes.add(kf); nodes.add(kt)
            if kt not in pos and kt[0] == "platform":
                pos[kt].extend(pos.get(kf, []))

        # components
        adj = collections.defaultdict(set)
        for e in es:
            adj[e["hi"]].add(e["lo"]); adj[e["lo"]].add(e["hi"])
        seen = set()
        comps = []
        for n in nodes:
            if n in seen:
                continue
            stack, comp = [n], set()
            while stack:
                x = stack.pop()
                if x in comp:
                    continue
                comp.add(x)
                stack.extend(adj[x] - comp)
            seen |= comp
            comps.append(comp)

        for comp in comps:
            ces = [e for e in es if e["hi"] in comp]
            cnotes = []
            streets = [k for k in comp if k[0] == "street"]
            m0 = next((e["m"] for e in ces if e["m"] is not None), None)
            if not streets:
                # default street -> topmost plain mezzanine
                cand = [k for k in comp if k[0] == "mezzanine" and k[2] in ("", "upper", "main", "transfer")]
                if not cand:
                    for e in ces:
                        unparsed.append((cx, e["m"]["station"] if e["m"] else "", e["eq"], e["m"]["raw"] if e["m"] else "",
                                         e["m"]["notes"] if e["m"] else "",
                                         "chain never reaches the street and has no plain mezzanine to hang a default on"))
                    continue
                # topmost: one not below another candidate
                below = {e["lo"] for e in ces}
                above = {e["hi"] for e in ces}
                tops = [k for k in cand if k not in below] or cand
                for k in tops:
                    sk = ("street", f"default {k}")
                    el_ = structure_of(k, m0) in elevated
                    note = (f"no street-end machine has a rise: {lbl(k)} assumed {DEFAULT_MEZZ['mu']:.1f} +- "
                            f"{DEFAULT_MEZZ['sd']:.1f} m {'above' if el_ else 'below'} the street (typical street-to-mezzanine "
                            f"rise at stations that have one)")
                    if el_:
                        ces.append({"hi": k, "lo": sk, "d": DEFAULT_MEZZ["mu"], "sd": DEFAULT_MEZZ["sd"], "eq": "default", "m": None, "kind": "assume", "note": note})
                    else:
                        ces.append({"hi": sk, "lo": k, "d": DEFAULT_MEZZ["mu"], "sd": DEFAULT_MEZZ["sd"], "eq": "default", "m": None, "kind": "assume", "note": note})
                    pos[sk] = pos[k]
                    comp.add(sk)
                    streets.append(sk)
            # pins
            pins = {}
            for sk in streets:
                if sk[1] in ms:
                    mm = ms[sk[1]]
                    lon, lat = mm["lon"], mm["lat"]
                else:
                    pts = pos.get(sk) or [(crow["lon"], crow["lat"])]
                    lon = float(np.median([p[0] for p in pts])); lat = float(np.median([p[1] for p in pts]))
                g = ground.at(lon, lat)
                srow_ = (rows_by_mrn.get(ms[sk[1]]["mrn"]) or [None])[0] if sk[1] in ms else None
                if srow_ is not None and srow_["structure"] == "Embankment" and g[0] is not None:
                    g2 = ground.street_by_embankment(lon, lat)
                    if g2[0] is not None and g2[0] < g[0] - 1.0:
                        cnotes.append(f"embankment station: street at {sk[1]} taken as the low ground within "
                                      f"{EMB_STREET_R:.0f} m ({g2[0]:.1f} m; the DEM at the machine, {g[0]:.1f} m, is the embankment)")
                        print(f"  embankment street: {srow_['name']} {sk[1]}: {g[0]:.1f} -> {g2[0]:.1f} m")
                        g = g2
                if g[0] is None:
                    continue
                pins[sk] = (g[0], g[1], g[2], lon, lat)
            if not pins:
                for e in ces:
                    if e["m"] is not None:
                        unparsed.append((cx, e["m"]["station"], e["eq"], e["m"]["raw"], e["m"]["notes"], "no ground at the street end"))
                continue
            idx = {k: i for i, k in enumerate(sorted(comp, key=str))}
            n = len(idx)
            Arows, b, w = [], [], []
            for sk, (g, gsd, spread, _, _) in pins.items():
                row = np.zeros(n); row[idx[sk]] = 1
                Arows.append(row); b.append(g); w.append(1 / gsd ** 2)
            for e in ces:
                row = np.zeros(n); row[idx[e["hi"]]] = 1; row[idx[e["lo"]]] = -1
                Arows.append(row); b.append(e["d"]); w.append(1 / e["sd"] ** 2)
            A = np.array(Arows); b = np.array(b); w0 = np.array(w)
            labels = [f"street pin {sk[1]}" for sk in pins] + [f"{e['eq']} ({lbl(e['hi'])} to {lbl(e['lo'])})" for e in ces]
            # Robust weighted least squares: where paths disagree, the edges that misfit by more than
            # 2 sd get their sd scaled up until they fit within 2 sd. This widens the sd of exactly
            # the levels that depend on the disagreeing machines, instead of every level in a complex.
            scale = np.ones(len(b))
            ok = True
            for _ in range(30):
                w = w0 / scale ** 2
                N = A.T @ (A * w[:, None])
                try:
                    cov = np.linalg.inv(N)
                except np.linalg.LinAlgError:
                    ok = False
                    break
                z = cov @ (A.T @ (w * b))
                res = (A @ z - b) * np.sqrt(w)
                bad = np.abs(res) > 2.0
                if not bad.any():
                    break
                scale[bad] *= np.abs(res[bad]) / 2.0
            if not ok:
                for e in ces:
                    if e["m"] is not None:
                        unparsed.append((cx, e["m"]["station"], e["eq"], e["m"]["raw"], e["m"]["notes"], "chain not tied down"))
                continue
            zsd = np.sqrt(np.diag(cov))
            raw_res = A @ z - b
            off = []
            for j in np.where(scale > 1.01)[0]:
                off.append(f"{labels[j]} off by {raw_res[j]:+.2f} m (sd widened from {1 / math.sqrt(w0[j]):.2f} to {1 / math.sqrt(w0[j]) * scale[j]:.2f})")
            if off:
                report.append((cx, "; ".join(off)))
                cnotes.append("paths disagree: " + "; ".join(off))

            # shortest (least-variance) path for `how`
            gadj = collections.defaultdict(list)
            for e in ces:
                gadj[e["hi"]].append((e["lo"], e, -1))
                gadj[e["lo"]].append((e["hi"], e, +1))

            def how_for(target):
                import heapq
                best = {target: (0.0, None)}
                pq = [(0.0, str(target), target)]
                done = set()
                while pq:
                    c, _, x = heapq.heappop(pq)
                    if x in done:
                        continue
                    done.add(x)
                    if x in pins:
                        break
                    for y, e, sgn in gadj[x]:
                        nc = c + e["sd"] ** 2
                        if y not in best or nc < best[y][0]:
                            best[y] = (nc, (x, e, sgn))
                            heapq.heappush(pq, (nc, str(y), y))
                starts = [k for k in pins if k in best]
                if not starts:
                    return "", float("nan")
                s = min(starts, key=lambda k: best[k][0] + pins[k][1] ** 2)
                g, gsd, spread, lon, lat = pins[s]
                if s[1] in ms:
                    mm_ = ms[s[1]]
                    st_end = [X["text"] for X in (mm_["A"], mm_["B"]) if X["kind"] == "street"]
                    where = f"{s[1]} ({st_end[0]})" if st_end else f"{s[1]}'s position"
                else:
                    where = f"{lat:.5f},{lon:.5f}"
                parts = [f"ground {g:.1f} at {where}"]
                x = s
                while best[x][1] is not None:
                    prev, e, sgn = best[x][1]
                    # moving from x toward target: prev is nearer the target
                    step = e["d"]
                    sign = "-" if e["hi"] == x else "+"
                    if e["m"] is not None:
                        parts.append(f"{sign} {fmt_ft(e['m']['raw'])} {e['eq']} {lbl(e['hi'] if sign == '-' else e['lo'])} to {lbl(prev)}")
                    else:
                        parts.append(f"{sign} {step:.1f} ({e['eq']}: {lbl(x)} to {lbl(prev)})")
                    x = prev
                return " ".join(parts), g

            used_eq = sorted({e["eq"] for e in ces if e["m"] is not None})
            for k in comp:
                if k[0] == "street":
                    continue
                i = idx[k]
                if k[0] == "platform":
                    srow = st_rows[(k[2], k[3])]
                else:
                    mrn = k[3][2:] if k[0] == "mezzanine" and k[3].startswith("m:") else (m0["mrn"] if m0 else crow["mrn"])
                    rr = rows_by_mrn.get(mrn) or [crow]
                    srow = rr[0]
                pts = pos.get(k) or [(srow["lon"], srow["lat"])]
                lon = float(np.median([p[0] for p in pts])); lat = float(np.median([p[1] for p in pts]))
                g_st = ground.at(srow["lon"], srow["lat"])[0]
                how, local = how_for(k)
                kn = [e["note"] for e in ces if e.get("note")] + list(cnotes)
                if k in alias_note:
                    kn.append(alias_note[k])
                for e in ces:
                    if e["m"] is not None and e["m"]["ov"].get("note"):
                        kn.append(e["m"]["ov"]["note"])
                    if e["m"] is not None and e["m"]["rnote"]:
                        kn.append(f"{e['eq']} rise {e['m']['rnote']}")
                out_rows.append({
                    "station_mrn": srow["mrn"], "station": srow["name"], "line": srow["line"],
                    "level_name": lbl(k), "z_navd88_m": round(float(z[i]), 2), "sd_m": round(float(zsd[i]), 2),
                    "depth_below_local_street_m": round(local - float(z[i]), 2),
                    "how": how, "machines": " ".join(used_eq),
                    "lat": round(lat, 6), "lon": round(lon, 6), "notes": " | ".join(dict.fromkeys(kn)),
                    "_complex": cx, "_kind": k[0], "_station_ground": None if g_st is None else round(g_st, 2),
                    "_structure": srow["structure"],
                })

    df = pd.DataFrame(out_rows)
    df = df.sort_values(["_complex", "station_mrn", "line", "level_name"])
    EVID.mkdir(exist_ok=True)
    cols = ["station_mrn", "station", "line", "level_name", "z_navd88_m", "sd_m", "depth_below_local_street_m",
            "how", "machines", "lat", "lon", "notes"]
    df[cols].to_csv(EVID / "station_levels.csv", index=False)
    plat = df[df["_kind"] == "platform"]
    print(f"{len(df)} levels in {df['_complex'].nunique()} complexes; {len(plat)} platform rows at "
          f"{plat[['station_mrn', 'line']].drop_duplicates().shape[0]} (station, line) pairs")

    # ---- unparsed report
    L = ["# Machines not placed in a chain",
         "",
         "Written by `station_levels.py`. Each machine below has no usable rise, or could not be",
         "joined to a chain that reaches the street. A later pass (more rises, a layout drawing, a",
         "barometer ride) could pick these up.",
         "",
         "## Rises that needed reading or could not be read",
         "",
         "| machine | station | raw | what was done |", "|---|---|---|---|"]
    for eq, sname, raw, note in sorted(rise_log):
        L.append(f"| {eq} | {sname} | `{raw}` | {note} |")
    L += ["", "## Coordinates replaced by the station's (more than 400 m from it)", "",
          "| machine | station | listed lat, lon | distance |", "|---|---|---|---|"]
    for eq, sname, x, y, d in coord_log:
        L.append(f"| {eq} | {sname} | {x}, {y} | {'missing' if d == float('inf') else f'{d / 1000:.1f} km'} |")
    L += ["", "## Paths that disagree (misfit over 2.5 sd)", ""]
    for cx, txt in report:
        L.append(f"- complex {cx}: {txt}")
    have_rise = [u for u in unparsed if u[3]]
    no_rise = [u for u in unparsed if not u[3]]
    L += ["", "## Machines with a rise that were not placed", "",
          "| complex | station | machine | rise | notes | why |", "|---|---|---|---|---|---|"]
    for cx, sname, eq, raw, notes, why in sorted(have_rise, key=lambda u: (str(u[0]).zfill(4), u[2])):
        nt = (notes if isinstance(notes, str) else "").replace("|", "/")
        L.append(f"| {cx} | {sname} | {eq} | `{raw}` | {nt} | {why} |")
    L += ["", f"## Machines with no rise in the inventory ({len(no_rise)})", "",
          "They still say what connects to what; listed only where they would complete a chain that",
          "has rises elsewhere in the same complex.", ""]
    cx_with = set(edges)
    for cx, sname, eq, raw, notes, why in sorted(no_rise, key=lambda u: (str(u[0]).zfill(4), u[2])):
        if cx in cx_with:
            L.append(f"- {sname} {eq}: {notes}")
    (EVID / "station_levels_unparsed.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    df.to_csv(pathlib.Path(__file__).parent / "data" / "_station_levels_debug.csv", index=False) if False else None
    return df


if __name__ == "__main__":
    main()
