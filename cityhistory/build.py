"""build.py — turn Stadester's city JSON into a compact cities.json for the viewer.

Steps:
  1. load stadester_cities.json (per-city annual population series + coords + provenance)
  2. clip corrupt year keys, drop empty/no-coord entries
  3. drop parenthetical metro variants ("(agglomeration)", "(Greater ...)", "(inner ...)")
     -> v1 keeps city-proper figures only (see FINDINGS.md / todo.txt)
  4. Douglas-Peucker simplify each series in log-population space -> a handful of
     control points per city (the browser re-interpolates in log space, giving smooth
     year-to-year transitions from a tiny payload)
  5. emit data/cities.json: {yearMin, yearMax, cities:[{n,la,lo,t,p:[[year,pop],...]}]}

Source-defect repairs live in the tables below (DROP_KEYS / CLIP_BEFORE / RENAME / CF_*) and
in data/coord_fixes.json. Run `python validate.py` after building: it re-runs the five checks
that found each class of defect, so a regression -- or a fresh one from a source update --
shows up as a count going the wrong way. Regenerate coord_fixes.json with the two scripts in
tools/ (see tools/make_coordfix.py).
"""
import json, math, os, re, sys, unicodedata
import provenance                      # per-year source attribution; see source_codes()
from collections import defaultdict

SRC = "data/stadester/stadester_cities.json"
GHSL = "data/stadester/wup2025.json"  # UN WUP 2025 agglomerations (annual 1975-2025, GHSL-lineage,
                                      # cleaner+broader than the old ghsl.json; same schema). Run prep_wup.py.
FUA = "data/stadester/fua2025.json"   # metropolitan-level series for the WUP centres that sit
                                      # inside a functional urban area. Run prep_fua.py. See
                                      # load_ghsl() for why, and --no-fua to build without it.
AFRICAPOLIS = "data/stadester/africapolis.json"   # OECD/SWAC Africapolis agglomerations, run
                                      # prep_africapolis.py. OFF unless --africapolis is passed;
                                      # see load_ghsl() for the evidence behind that default.
OUT = "data/cities.json"
COORDFIX = "data/coord_fixes.json"   # repaired coords for geocoder-fallback entries

YEAR_LO, YEAR_HI = -4000, 2035      # clip range (kills the one corrupt key 19690310)
DP_EPS_REL = 0.01                   # linear-space relative tolerance for line simplification
# Two jobs, deliberately two tuples. DROP_MARKERS says "never draw this as its own dot" AND
# doubles as index_variants' splice source, because for these suffixes the two roles coincide:
# a "(agglomeration)" entry is both a duplicate of its base and a usable metro measurement.
DROP_MARKERS = ("(agglomeration)", "(greater", "(inner", "(metropolitan", "(metro")
# DUP_MARKERS is only the first job. populstat writes the same idea a dozen other ways and 254
# entries slipped through on spelling alone -- "(agglomeration?)" with a question mark (all 20
# are Nigerian: Kano 2.17M, Ibadan 1.84M, Kaduna, Benin City, Port Harcourt, Maiduguri),
# "(agglom./municip.)" (La Habana 2.19M, San Salvador 1.80M), "(aglomeration)" (a typo), and
# the "(municip.)" family. 154 of them have a plain base entry as well, so those cities were
# being DRAWN TWICE, at full agglomeration size on top of themselves.
#
# These are NOT added to DROP_MARKERS, for two separate reasons. First, that tuple is now the
# variant FINDER, and broadening it changes which entry wins each (name, country) slot -- it
# swaps Birmingham and Liverpool for tiny homonyms. Second, dropping on the suffix alone would
# delete cities outright: 100 of the 254 have NO plain base entry, and "Zagreb (municip.)" at
# 850k is the only Zagreb in the source. So these drop ONLY when a plain entry for the same
# city exists; otherwise the entry is kept and just loses its parenthetical.
DUP_MARKERS = ("(agglomeration?)", "(agglom.", "(aglomeration)", "(municip.", "(municipality)")

# --- New World antiquity ramp -------------------------------------------------
# populstat/Chandler-Modelski file several American cities with Classic-era peak
# populations stamped back into deep antiquity: Teotihuacan 150k @ 800 BC, and
# Tikal/Caracol/Tula/Tiwanaku as flat ~100k *constants* from 900 BC onward (the peak
# value entered for every year). These are archaeologically impossible -- no city in
# the Americas approached that size before ~1 CE, and most only urbanized in the Late
# Preclassic. A gap/shape filter can't catch them without also deleting the real Old
# World giants (Pi-Ramesses, Nineveh, Anyang share the same sparse "one ancient
# benchmark, then modern data" shape), so we discriminate on region instead.
#
# Rather than a hard cutoff (which makes American cities pop into existence full-size
# at the boundary), we apply a time-varying CAP that ramps up: nothing American is
# visible before NW_RAMP_START, and real magnitudes are only allowed from NW_RAMP_FULL
# on. Between them the cap rises log-linearly, so a city fades in gradually. The cap
# only ever *reduces* an implausibly-large early value; a genuinely small early city
# is left untouched. Points before NW_RAMP_START are dropped outright.
NW_RAMP_START = -200                # American cities invisible before here (Late Preclassic)
NW_RAMP_FULL  = 400                 # real magnitudes fully allowed from here (Classic era)
NW_CAP_TOP    = 2e7                 # cap ceiling at NW_RAMP_FULL (effectively "no cap")

# Anglo/French North America had no cities before ~1700, so ALL pre-1700 population there is
# populstat placeholder, not data: "Cincinnati" held 10k from 1425, "Saint Louis" 40k from
# 1100, even "Boston" has a fabricated 1000->5k ramp from year 1000 (founded 1630!). Every
# real colonial city (Boston, New York, Philadelphia, Quebec) has its genuine series start
# ~1700, so clipping pre-1700 keeps all real data. Latin America is excluded -- it had genuine
# cities from the 1500s (Potosi, Mexico City, Cusco).
NA_CLIP_BEFORE = 1700
# ...with exactly one exception, and it is the one the sentence above gets wrong. Cahokia was a
# real city of tens of thousands opposite the Missouri confluence from about 1050, and the
# "Saint Louis 40k from 1100" this rule was written to delete IS Cahokia's benchmark -- the
# fusion filed Chandler's row on the modern city 12km away, where held verbatim to 1830 it is
# indeed a placeholder and the clip is right about it. See SYNTHETIC, which gives the figure an
# entry of its own; this set keeps the clip off that entry.
NA_CLIP_KEEP = {"Cahokia-United States"}

# --- cities the fusion lost entirely -----------------------------------------------------
# Entries built here and injected into `raw` before anything reads it, so every rule downstream
# treats them exactly like source entries. This is the only place a city is CREATED, so the bar
# is the strictest in the file:
#   1. every figure is a verbatim row from a source already in the repo (chandlerV2.csv), not
#      a number of ours -- this table restores what the fusion dropped, it does not add;
#   2. no entry on the map carries the series, and the note names the entry that was hiding it
#      and says why that entry cannot simply be relabelled;
#   3. the coordinates are the archaeological site, not the modern city Chandler geocoded to.
SYNTHETIC = {
    # Cahokia. chandlerV2.csv has `Cahokia / Missouri / United States of America AD_1100 =
    # 40,000, AD_1400 = 4,000` -- the decline is in the row too. Stadester fused it into
    # "Saint Louis-Missouri", where it appears as 39,999 held from 1100 to 1830 and hands off to
    # the real 1840 census; NA_CLIP_BEFORE then deletes it as a placeholder, correctly, because
    # on that entry it is one. It cannot be a RENAME: Saint Louis has its own genuine history
    # from 1764 and the mounds are 12km from downtown, so this is the Danapur/Pataliputra shape
    # -- two places, one entry -- except that here the ancient half has nowhere to go.
    # Result: North America went from zero cities at every pre-colonial year to one.
    # Chandler's 40,000 is at the high end; current archaeology puts Cahokia proper nearer
    # 10,000-15,000 with the wider American Bottom at 20,000-30,000. Carried as Chandler states
    # it, like every other figure on the map, rather than silently re-estimated.
    "Cahokia-United States": {
        "name": "Cahokia", "country": "United States", "type": "chandler_modelski",
        "coords": [38.6553, -90.0618],          # Monks Mound, not Chandler's downtown St Louis
        "population": {"1100": 40000, "1400": 4000},
    },
    # Chan Chan, the Chimu capital and the largest adobe city ever built. chandlerV2.csv files it
    # as `Chanchan / Chancán / Peru (-8.106, -79.0745) AD_1400 = 25,000`, and Stadester fused
    # that row into "Trujillo-Peru" 5km away, where it became 25,000 held from 1400 to 1790
    # against a real 1791 count of 9,000 -- i.e. the map asserted a 25,000-person Trujillo more
    # than a century before the Spanish founded it in 1534. CLIP_BEFORE now takes that half off
    # Trujillo, so without this entry the figure would simply be lost.
    # Chandler's coordinate is already the site, so it is used as given.
    "Chan Chan-Peru": {
        "name": "Chan Chan", "country": "Peru", "type": "chandler_modelski",
        "coords": [-8.106, -79.0745],
        "population": {"1400": 25000},
    },
    # Mayapan, the Postclassic Maya capital -- founded c. 1220 (Chandler says 1194) and destroyed
    # in the Cocom revolt of c. 1441. Chandler has the row, but geocoded to (20.9785, -89.5934),
    # which is Merida, 40km north; the real site is Telchaquillo, Yucatan. Stadester duly fused
    # it into the Merida entry, where its 1194 value sat beside Chandler's genuine Merida series.
    # DROP_YEARS removes that stray row so the figure is not in two places at once.
    # NOTE Merida itself is NOT a conflation and needs no split: Chandler files it as
    # "Merida / Tiho", one continuous city, because the Spanish built Merida in 1542 directly on
    # top of the Maya T'ho using its stones -- so its AD 900 = 40,000 is the same site.
    # 25,000 is high; excavation-based estimates for Mayapan run 15,000-17,000.
    "Mayapan-Mexico": {
        "name": "Mayapán", "country": "Mexico", "type": "chandler_modelski",
        "coords": [20.6294, -89.4590],          # Telchaquillo, not Chandler's Merida geocode
        "population": {"1194": 25000, "1441": 25000},
    },
}

# --- the archaeological tier -------------------------------------------------------------
# SYNTHETIC's rule 1 is "every figure is a verbatim row from a source already in the repo", and
# for the pre-Columbian Americas that rule makes the map unfixable. The audit is in GAPS.md
# 2.7/2.8; the short version is that chandlerV2.csv contains THREE rows with any value before
# AD 600 (Izapa, Teotihuacan, Tres Zapotes) and stadester has seven entries, four of which are a
# flat Modelski 100,000 stamped back to 900 BC. Chandler compiled in 1987 from the historical
# record, and for Mesoamerica and the Andes there essentially is no historical record before the
# Spanish -- the evidence is excavation and survey, published after him and never folded in. So
# the corpus is not thin here, it is absent, and no rule change reaches it.
#
# This table is therefore a DELIBERATE loosening of that bar, opened for this case only:
#   1. every figure is a published estimate from the archaeological literature, and the comment
#      NAMES the source and gives the RANGE where the field disagrees -- these are survey and
#      excavation estimates, not counts, and several are actively contested;
#   2. no entry on the map carries the series (checked by name across stadester and chandlerV2,
#      and by a 25km radius search of the built map);
#   3. coordinates are the archaeological site;
#   4. figures are round, and the number of points is the number the phase sequence supports --
#      no interpolated filler. Where a value is a mid-range choice between competing estimates
#      the comment says so rather than presenting it as settled.
#
# These carry `type: "archaeological"` and get their own provenance character, `a`, so a reader
# can tell this tier from Chandler at a glance and the growth-mode source rule marks the joins
# (see SRC_ARCH). They are also exempt from the New World ramp -- nw_cap exists to suppress
# UNATTESTED compiler stamps, and an entry whose comment cites its source is the case the ramp
# was never aimed at. Without that exemption the ramp would delete most of this table outright,
# since everything before NW_RAMP_START is dropped.
#
# NOT INCLUDED, and the reason is the display floor rather than doubt: Caral/Norte Chico
# (c. 2600-1800 BC, ~3,000 at the site on Shady Solis' estimates), Chavin de Huantar
# (2,000-3,000), San Lorenzo (~5,500) and La Venta (3,000-8,000) all sit at or under
# MINPOP = 5000 and would be entries that never draw. Palenque (~6,000-8,000 at AD 700, when the
# era floor is already 10,000) is out for the same reason. The Preclassic Americas is genuinely
# a handful of dots; padding it with invisible entries would imply coverage that is not there.
ARCHAEOLOGICAL = {
    # CUICUILCO -- the structural gap, because it is why Teotihuacan exists. It was the Basin of
    # Mexico's dominant centre through the Late Formative and Teotihuacan's rival; the eruption
    # of Xitle ended it and the Basin's population concentrated at Teotihuacan afterwards. With
    # Cuicuilco missing the map showed the effect and not the cause.
    # Figures: Sanders, Parsons & Santley, "The Basin of Mexico" (1979) -- ~20,000 at the
    # Patlachique-phase peak, c. 150-1 BC, declining as Teotihuacan drew the Basin in. The
    # eruption is conventionally dated c. AD 100; a later radiocarbon/paleomagnetic case puts
    # Xitle nearer AD 245-315, which would extend the tail but not change the peak.
    "Cuicuilco-Mexico": {
        "name": "Cuicuilco", "country": "Mexico", "type": "archaeological",
        "coords": [19.3011, -99.1808],
        "population": {"-400": 5000, "-150": 20000, "100": 10000},
    },
    # EL MIRADOR -- the largest Late Preclassic Maya centre, and the most contested figure in
    # this table. Site-core estimates run 10,000-15,000; Richard Hansen's claims for the wider
    # Mirador basin run to tens of thousands and have been read as 100,000+. The basin figure is
    # a REGION, not a city, and this map draws cities, so the conservative core estimate is used
    # and the disagreement is recorded here rather than split the difference silently.
    # Abandoned c. AD 150, with a partial Late Classic reoccupation not represented here.
    "El Mirador-Guatemala": {
        "name": "El Mirador", "country": "Guatemala", "type": "archaeological",
        "coords": [17.7550, -89.9200],
        "population": {"-200": 10000, "1": 15000, "150": 5000},
    },
    # KAMINALJUYU -- the highland Maya centre under modern Guatemala City, occupied from the
    # Middle Preclassic and at its height in the Miraflores phase (c. 100 BC - AD 200) at around
    # 10,000. It is the same site relationship as Cahokia/St Louis: Ciudad de Guatemala is drawn
    # from 1500 and the two never overlap in time, so both can stand.
    # Most of the mounds were destroyed by 20th-century urban growth, which is part of why the
    # estimates are loose; 10,000 is the common figure for the Miraflores peak.
    "Kaminaljuyu-Guatemala": {
        "name": "Kaminaljuyú", "country": "Guatemala", "type": "archaeological",
        "coords": [14.6250, -90.5417],
        "population": {"-100": 8000, "200": 10000, "900": 4000},
    },
    # MOCHE (Huacas de Moche / Cerro Blanco) -- the urban core of the Southern Moche state, and
    # the only Andean city of the period the map can support. ~10,000-15,000 at its AD 300-700
    # height on Claude Chapdelaine's urban-zone excavations. Note this is 5km from Chan Chan,
    # which SYNTHETIC already carries for the later Chimu capital -- different city, different
    # millennium, and the two do not overlap.
    "Moche-Peru": {
        "name": "Moche", "country": "Peru", "type": "archaeological",
        "coords": [-8.1330, -78.9950],
        "population": {"200": 5000, "500": 15000, "800": 3000},
    },
    # WARI (Huari) -- capital of the Middle Horizon Wari state, flagged absent in GAPS.md 2.5.
    # Estimates range widely, 10,000-40,000, with William Isbell's mapping of the urban core at
    # the upper end; 30,000 is taken as the mid-to-upper figure for the AD 750-900 peak. The
    # nearest drawn entry is Ayacucho, 25km away and starting in 1574, so nothing carries this.
    "Wari-Peru": {
        "name": "Wari (Huari)", "country": "Peru", "type": "archaeological",
        "coords": [-13.0417, -74.1417],
        "population": {"600": 10000, "800": 30000, "1000": 5000},
    },
    # CALAKMUL -- the Kaan capital and Tikal's rival for the whole Classic period. The map has
    # carried Tikal since the beginning and never had its opponent, which makes the Classic
    # Peten read as a single-centre landscape. ~50,000 in the urban core and immediate
    # hinterland at the AD 600-700 peak, from the PACUNAM LiDAR survey work reported in Canuto
    # et al. 2018 (Science 361:eaau0137) and Sprajc's mapping. Abandoned c. AD 900.
    "Calakmul-Mexico": {
        "name": "Calakmul", "country": "Mexico", "type": "archaeological",
        "coords": [18.1053, -89.8117],
        "population": {"250": 20000, "650": 50000, "900": 8000},
    },
    # COPAN -- the southeastern Classic Maya capital. The figure is the Copan VALLEY at its
    # AD 750-800 height, ~20,000-25,000, from the Harvard/Penn State settlement survey
    # (Webster, Freter & Gonlin); the urban core alone is nearer 10,000. Taking the valley
    # figure keeps it on the same footing as the other entries here, which are all settlement
    # rather than ceremonial-core counts. Dynasty ends 822 and the valley depopulates over the
    # following century and a half.
    "Copan-Honduras": {
        "name": "Copán", "country": "Honduras", "type": "archaeological",
        "coords": [14.8400, -89.1417],
        "population": {"400": 8000, "780": 20000, "950": 5000},
    },
    # CHICHEN ITZA -- called out in GAPS.md 2.2 as "what should be filling the 900-1050 frames".
    # It is the Terminal Classic/Early Postclassic capital of the northern lowlands, commonly put
    # at 30,000-50,000 around AD 900-1050. This coordinate was OCCUPIED until recently: the
    # Caracol entry was geocoded onto El Caracol, the observatory building here, and drew
    # Caracol's population on Chichen Itza's site 400 years early. SITE_COORDS moved Caracol to
    # Belize, which is what makes room for the real city.
    "Chichen Itza-Mexico": {
        "name": "Chichén Itzá", "country": "Mexico", "type": "archaeological",
        "coords": [20.6829, -88.5686],
        "population": {"800": 10000, "1000": 40000, "1200": 5000},
    },
}

# --- year 0 does not exist -------------------------------------------------------------
# 1 BC is followed by AD 1 in the Julian and Gregorian calendars alike, so no source can have
# observed a year 0 and every year-0 row here is a processing artifact. Measured across the
# 102 entries that have one: 68 (67%) are an EXACT duplicate of the entry's previous negative
# year, and the other 34 all sit on the straight-line fill between their neighbours (Chengdu's
# 0 = 248,218 is precisely 100 years along the line from -100:70,000 to 1:250,000). Not one is
# a datum, so dropping them costs nothing.
#
# What it buys is the AD 1 cliff. populstat holds the 100 BC benchmark verbatim onto year 0 and
# then starts the AD table at year 1, so three top-12 cities fall off a one-year edge:
# Alexandria 1,000,000 -> 400,000, Patna 200,000 -> 100,000, Taxila 150,000 -> 100,000. Total
# drawn urban population dropped 6.16M -> 5.61M (-9.0%) in a single year, ~87% of it Alexandria,
# and Alexandria handed the world's-largest slot to Rome overnight -- at a year everybody
# scrubs to. With year 0 gone the same decline is spread over the 101 years from -100 to AD 1,
# which is a claim the source is actually making rather than an artifact of its table layout.
#
# No check could see it: whipsaw needs 8x (this is 2.5x), oscillation needs a return inside 40y
# (it never returns), and despike is up-only.
YEAR_ZERO = 0

# --- which region an entry is in --------------------------------------------------
# These were lat/lon boxes (lon < -30 for "the Americas"; lat > 32 and lon in -130..-55 for
# Anglo/French North America) and both misfired. ALASKA AND HAWAII fall outside the NA box
# entirely, so neither the pre-1700 clip nor the leading-flat-run strip ever ran there -- which
# is why "Barrow" held 56,000 from 1900 to 1989. That figure is Barrow-in-Furness, England,
# which the source has at 57,010 in 1900, filed against Utqiagvik (real population ~4,000).
# The lon box also swept in 143 Pacific-island entries (French Polynesia, Wallis and Futuna,
# American Samoa, Tonga) as "the Americas".
#
# The fix is the source's own `country` field, which is populated on all 24,219 entries with no
# blanks, 280 distinct values. US entries carry the STATE ('Alaska', 'Ohio', 'Tennessee') --
# 1,936 of them across all 50 states, and no Canadian provinces at all.
#
# WHY NOT A SHAPEFILE. Point-in-polygon reads the COORDINATE, and the coordinate is the field
# that is wrong in exactly the cases where the two disagree: the source has 'Hamilton-Canada'
# geocoded into India, 'El Tajin-Mexico' at lon 97.38, 'Los Angeles-Equatorial Guinea' at LA's
# real coordinates. What these rules actually encode is a PROVENANCE fact -- "populstat's
# US/Canada pages contain fabricated pre-1700 numbers" -- and `country` records that directly
# where a coordinate only proxies it. It is also free: measured against the repo's own vendored
# land geojson, ray-casting 24,198 points costs 2.8s at 110m and 23.9s at 50m, on a 10s build.
# And no box can express the rule anyway: Hawaii (lat 19.7-21.5) sits inside Cuba's, Puerto
# Rico's and Mexico's latitude ranges, so it is separable only on longitude.
#
# Unknown values default to Old World, which fails safe -- no clip, no deletion.
_US_STATES = frozenset("""Alabama Alaska Arizona Arkansas California Colorado Connecticut
    Delaware Florida Hawaii Idaho Illinois Indiana Iowa Kansas Kentucky Louisiana Maine
    Maryland Massachusetts Michigan Minnesota Mississippi Missouri Montana Nebraska Nevada
    Ohio Oklahoma Oregon Pennsylvania Tennessee Texas Utah Vermont Virginia Washington
    Wisconsin Wyoming""".split()) | {
    "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota",
    "Rhode Island", "South Carolina", "South Dakota", "West Virginia"}
# Anglo/French North America: the pre-1700 clip's actual scope. NOT Puerto Rico or the US
# Virgin Islands -- US territories, but Spanish-colonial cities from the 1520s, so they belong
# with Latin America for this test.
ANGLO_NA = _US_STATES | {"Canada", "United States", "United States of America", "USA"}
# The Americas, for the New World antiquity ramp. Everything above plus Latin America and the
# Caribbean. Pacific islands are deliberately absent.
AMERICAS = ANGLO_NA | {
    "Anguilla", "Antigua and Barbuda", "Argentina", "Aruba", "Barbados", "Belize", "Bermuda",
    "Bolivia", "Brazil", "British Virgin Islands", "Cayman Islands", "Chile", "Colombia",
    "Costa Rica", "Cuba", "Dominica", "Dominican Republic", "Ecuador", "El Salvador",
    "Falkland Islands", "French Guiana", "Greenland", "Grenada", "Guadeloupe", "Guatemala",
    "Guyana", "Haiti", "Honduras", "Jamaica", "Martinique", "Mexico", "Montserrat",
    "Netherlands Antilles", "Nicaragua", "Panama", "Paraguay", "Peru", "Puerto Rico",
    "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines",
    "South Georgia and the South Sandwich Islands", "Suriname", "The Bahamas",
    "Trinidad and Tobago", "Turks and Caicos Islands", "US Virgin Islands", "Uruguay",
    "Venezuela"}
# 'Georgia' is the one genuinely ambiguous string in the source -- 80 entries spanning the US
# state and the Caucasus country. Split on longitude; none of the Caucasus ones has pre-modern
# data, so an imperfect split costs nothing.
def in_americas(country, lon):
    if country == "Georgia":
        return lon < -30
    return country in AMERICAS

def in_anglo_na(country, lon):
    if country == "Georgia":
        return lon < -30
    return country in ANGLO_NA

def nw_cap(y):
    """Max allowed population for an American city at year y (None = uncapped)."""
    if y >= NW_RAMP_FULL:
        return None
    f = (y - NW_RAMP_START) / (NW_RAMP_FULL - NW_RAMP_START)
    return 10 ** (math.log10(PEAK_FLOOR) + f * (math.log10(NW_CAP_TOP) - math.log10(PEAK_FLOOR)))

# --- ...but the cap must not eat a real benchmark ----------------------------------------
# The ramp is aimed at Modelski's placeholders -- a round 100,000 stamped on Tikal, Caracol and
# Tiahuanaco at every century from 900 BC -- and at its own start year it is at its most
# brutal, because nw_cap(NW_RAMP_START) == PEAK_FLOOR == 2,000 by construction. That is exactly
# where Chandler's only two BC benchmarks for the Americas sit, so the ramp was deleting the
# very evidence it exists to protect:
#     Izapa        200 BC = 35,000   (chandlerV2 `Izapa-Mexico`)         -> drawn 2,000
#     Tres Zapotes 200 BC = 30,000   (chandlerV2 `Tres Zapotes-Mexico`)  -> drawn 2,000
# Both are ordinary Chandler rows, no different in kind from the Old World benchmarks the map
# draws without argument, and between them they are 100% of the corpus's pre-AD-1 New World
# evidence. Capping them is why the Americas were literally empty before AD 1 -- not sparse,
# empty -- while Mesoamerica's Late Preclassic was in the source the whole time.
#
# So: exempt a point that IS a Chandler benchmark, verbatim. The join is provenance.py's own --
# the entry's `chandler_modelski_key` (Tapachula's says `Izapa-Mexico`, which is the only reason
# that one is findable at all), falling back to the stadester key, then the year lookup that
# knows chandlerV2's BC_100 column is really AD 100.
#
# The test is a VALUE match, not provenance.classify()'s label, and the difference is the whole
# point: classify() calls Tikal's 900 BC row 'chandler' as well, on the entry's TYPE rather than
# on any value, and Chandler has no Tikal benchmark before AD 751. Asking chandlerV2 directly is
# what separates a benchmark from a stamp. Measured against the current source it exempts the
# two points above and nothing else -- Teotihuacan's 100 BC = 45,000 is the third candidate on
# paper, but fix_chandler_ad100 refiles it to AD 100 before this runs and the cap is already
# 200,000 by then, so the build reports 2. What stays capped is 5 Modelski year-0 100,000s,
# which YEAR_ZERO deletes regardless. The phantoms the ramp was built for all sit before
# NW_RAMP_START and are still dropped outright, so nothing it was suppressing comes back.
def chandler_benchmark(key, entry, y, v):
    """True if (y, v) is a verbatim chandlerV2 row for this entry, so the cap must leave it."""
    rows = provenance.chandler()
    row = rows.get(entry.get("chandler_modelski_key") or "") or rows.get(key)
    if row is None:
        return False
    cv = provenance._chandler_at(row, y)
    return cv is not None and abs(cv - v) <= provenance.CHANDLER_TOL * max(cv, v)

# --- entries to drop outright (mislabeled duplicates of a city already present) ---
DROP_KEYS = {
    "Kensington and Chelsea-United Kingdom",  # sub-borough carrying Greater London's figure;
                                              # a real "London" entry already exists -> duplicate

    # Same shape as Kensington: "Honolulu CDP" is not a badly-named city but a second copy of
    # one. "Honolulu-Hawaii" sits at the IDENTICAL coordinate (21.310,-157.858) with the longer
    # record and the modern graft on top of it (peak 1.01M against the CDP entry's 372,000), so
    # the CDP is a smaller dot drawn inside a bigger one, and its census-place figure is exactly
    # the city-proper number the graft is there to replace. Renaming it to "Honolulu" -- the
    # obvious first instinct -- would produce two identically-labelled dots at one point.
    "Honolulu CDP-United States",

    # Richmond, Virginia, filed a THIRD time and at longitude +77.4667 -- the same lost minus
    # sign as El Tajin and Hamilton, drawing a 526,000-person city in the Taklamakan. Unlike
    # those two this one is dropped rather than moved, because Richmond is not missing:
    # "Richmond-Virginia" and "Richmond-United States" both already sit at the real coordinate
    # with the longer record (1790-2000, and 1.37M once the metro graft lands), and this entry's
    # 1900-1975 span is a strict subset of theirs. Moving it would just add a fourth co-located
    # duplicate for dedup to arbitrate. Its 526,000 is a metro figure the graft supplies anyway.
    "Richmond-United States of America",

    # A garbage-bin entry: populstat's Ruhr page nests several geographic units and they were
    # all merged under one key. Its series swings town -> Regierungsbezirk -> Landkreis ->
    # Ruhrgebiet and back (8k in 1905, 1.8M in 1914, 9.4k in 1933, 5.46M in 1975, 29.9k in
    # 2000) -- only the 5-figure values are Wetter. Essen/Dortmund/Duisburg/Bochum all exist
    # separately, each on its own city figure (see GRAFT_DENY -- Essen no longer carries the
    # conurbation either), so an entry that is four different units at once adds nothing.
    "Wetter (Ruhr)-Germany",

    # "Atomgrad" is not a place. The entry has no coordinate (so the geocoder dumped it on
    # Russia's centroid, where it draws as a 97,500-person city in central Siberia), no
    # other_names to identify it by, and a single row -- 97,500 in 1991. Every other entry on
    # that fallback point carries its standard transliteration in other_names and so can at
    # least in principle be re-homed; this one cannot be placed, named or checked against
    # anything. "Atom city" was a nickname several Soviet nuclear towns shared, and picking one
    # would be a guess. See §6.11 for the 22 entries still stacked there.
    "Atomgrad-Russia",

    # Hong Kong: populstat's district table was scattered across two country keys and badly
    # geocoded. "Beijiao" (= North Point, a district) holds the whole TERRITORY's series and
    # is renamed to Hong Kong below; everything here is either a district inside that figure
    # or a mis-geocoded duplicate of it, so all of them are double-counts.
    "Xianggang-Hong Kong",    # city-proper Hong Kong, superseded by the territory series
    "Jiulong-Hong Kong",      # Kowloon        - inside the territory total
    "Xinjiulong-Hong Kong",   # New Kowloon    - ditto
    "Quanwan-Hong Kong",      # Tsuen Wan      - ditto
    "Shatian-Hong Kong",      # Sha Tin        - ditto
    "Tunmen-Hong Kong",       # Tuen Mun       - ditto
    # ...and the same districts again, mis-keyed to China and geocoded to its centroid or to
    # random mainland towns of similar name (Jiulong -> a Sichuan county, Quanwan -> Quannan).
    "Xianggang-China",        # 28.8km from the 27.5M Guangzhou centre -> was stealing it
    "Jiulong-China", "Quanwan-China", "Tuen Mun-China",
    "Tia Po-China", "Sha Tin-China", "Yuen Long-China",

    # --- "<City>-United States" twins carrying the right series at the WRONG PLACE ------------
    # A different failure from the co-located metro twins in MERGE_INTO. Here the two entries do
    # NOT share a coordinate, so dedup never compared them and both are drawn: the "-United
    # States" copy carries the series of the LARGEST city of that name and the COORDINATE of a
    # different, smaller one. The result is a full-size phantom in the wrong state --
    # "Hollywood" at 139,400 drawn inside Los Angeles, 3,900km from the Florida entry holding
    # the identical series.
    # Dropped rather than merged: the correctly-placed twin already has the whole series (the
    # peaks match to the digit), so there is nothing to fold in, and no coordinate to repair
    # that would not just duplicate a city already on the map.
    "Hollywood-United States",      # = Hollywood-Florida's series, at 34.09,-118.33 (LA)
    "Ontario-United States",        # = Ontario-California's,       at 42.85,-77.29  (NY)
    "Mesquite-United States",       # = Mesquite-Texas's,           at 36.81,-114.07 (NV)
    "Aurora-United States",         # = Aurora-Illinois's,          at 39.73,-104.83 (CO)
    "Westminster-United States",    # = Westminster-Colorado's,     at 33.76,-117.99 (CA)
    "Springfield-United States",    # = Springfield-Missouri's,     at 39.92,-83.81  (OH)
    # Hartford is the one that does NOT fit and is deliberately left: its "-United States" copy
    # is at Hartford, Wisconsin but its series (164,440) is Hartford CT's CITY PROPER, which is
    # the more accurate of the two -- Hartford-Connecticut carries the 905,091 metro. Dropping
    # it keeps the worse number; keeping it keeps the wrong place. Wants a coord fix, not this.

    # Mesquita: the entry is Nova Iguaçu's series entire. Mesquita was split OUT of Nova Iguaçu
    # in 1999 and has ~168,000 people, but this runs 93,564 (1970) to 1,293,600 (1991) and then
    # STOPS -- so §3.11 held 1.29M forward to 2025 and the map drew a second Nova Iguaçu 3km
    # from the first. Nothing here is Mesquita's, and Nova Iguaçu is already on the map, so
    # there is no year to clip to. Same family as Belford Roxo and Queimados below, which do
    # have a real figure of their own and are clipped instead.
    "Mesquita-Brazil",

    # Town of Mount Royal, a ~20k enclave 0.6km from Montreal, whose entry carries MONTREAL's
    # series (381,076 in 1900, 526,000 in 1914, 728,000 held 1925-1940, ramping to 2.75M by 1974)
    # with two genuine TMR rows dropped in (4,900 in 1941, 14,800 in 1950). "Montreal-Canada"
    # exists with a fuller, cleaner series, so this is a duplicate -- Kensington and Chelsea again.
    "Mont-Royal-Canada",
}

# --- clip everything before a year (entry mixes two different places' histories) ----
# Danapur is a cantonment 10km from Patna carrying PATALIPUTRA's ancient figures, so both
# entries sat in the top-20 from 400 BC to AD 200 -- one city, counted twice. Patna keeps
# the ancient series; Danapur keeps only its own (real, modern) census record.
CLIP_BEFORE = {
    "Dânâpur-India": 1891,   # its own first census; everything before is Pataliputra's
    # Stadester fuzzy-matched "Shanghai" to Chandler's "Shangqi" (a Shang-dynasty city) and
    # stapled its 100-130k antiquity series onto modern Shanghai's coordinates -- see the
    # entry's own chandler_modelski_key. Shanghai was a fishing and salt village until the
    # Ming; chandlerV2's own Shanghai row correctly starts at 1554, so start there.
    "Shanghai-China": 1554,
    # Qingdao was a fishing village until the 1897 German lease -- the first real count, in 1911,
    # is 34,000. The 800,000 held from 1879 to 1910 painted a phantom top-30 city on the Shandong
    # coast for thirty years.
    "Qingdao-China": 1911,
    # Pyongyang. The MERGE_INTO donor carries Chandler's -1000: 25,000, -800: 25,000,
    # -650: 30,000 and 0: 42,858, and only the last of those is supportable. The Korean
    # peninsula in 1000 BC is the Mumun pottery period -- agricultural villages, no urbanism of
    # any kind -- and while the Guanzi mentions Gojoseon as a kingdom by the 7th century BC,
    # where its capital SAT is disputed, with much of the literature placing early Gojoseon in
    # Liaodong rather than on the Taedong. What is not disputed is the later city: Wanggeom-seong
    # was the capital of Wiman Joseon from 194 BC, Han Wudi took it in 108 BC, and the Lelang
    # commandery seat that followed is one of the best-attested sites in Korean archaeology
    # (the Tosong-ni walled town and thousands of Lelang tombs). So -194 is where the claim
    # becomes a claim about a place rather than about a polity, and it is the one year to edit
    # if that reading changes -- deleting this line restores Chandler's Bronze Age figures.
    # In PRACTICE the entry has no anchor between -650 and 0 and YEAR_ZERO drops the year-0 row,
    # so -194 and the strictly-archaeological -108 both resolve to the same thing: the series
    # now opens at AD 100. Anything between -649 and -1 gives that result; the year is written
    # as -194 to record which claim is being made, not because the arithmetic needs it.
    "P'yõngyang (agglomeration)-North Korea": -194,
    # Yiyang, Hunan: 200,000 held VERBATIM from -300 to 1982, the Tula pattern again, with only
    # -400: 100,000 and -300: 200,000 distinct. Both are impossible where they are drawn --
    # Yiyang county was founded in 221 BC, after Qin conquered Chu, so the place did not exist
    # at either date, let alone at that size, and the Zi valley was Chu backcountry.
    # Unlike Tula and Cahokia there is nowhere to send the figures: the entry is populstat's
    # alone, chandlerV2.csv has NOTHING within 60km of Yiyang in Henan (the Han-state city this
    # is most likely meant to be) nor within 120km of Yiyang in Hunan, and Modelski's own 400/300
    # BC lists name no city rendered Yiyang or Iyang. So there is no source-backed target and a
    # relocation would be our guess rather than a recovery. Clipping to the first real datum
    # deletes a false claim without asserting a new one -- the Qingdao case exactly.
    "Yiyang-China": 1983,
    # Trujillo, Peru: 25,000 held from 1400 to 1790, then a real count of 9,000 in 1791. The
    # Spanish city was founded in 1534, so the pre-1534 half cannot be it -- and the 25,000 is
    # Chandler's `Chanchan / Chancán AD_1400 = 25,000`, geocoded 5km onto the colonial city.
    # See SYNTHETIC, which gives Chan Chan its own entry; this stops the figure being counted
    # twice and stops the map claiming a 25,000-person Trujillo 134 years before it was founded.
    "Trujillo-Peru": 1791,
    # Gentofte is Frederiksberg all over again: a Copenhagen suburb whose entry carries the
    # CAPITAL's series (102,514 in 1800 rising to 354,917 by 1890) with genuine Gentofte only from
    # 1921 (34,500 -> 87,000). Copenhagen is already supplied by the renamed Frederiksberg entry,
    # so everything here before 1921 is the capital counted a second time.
    "Gentofte-Denmark": 1921,
    # Vorst (Forest) is a Brussels commune of ~55,000 whose entry carries BRUSSELS before 1901:
    # 162,972 (1880), 173,486 (1890) and the same held to 1900, then 10,600 in 1901 -- which is
    # Forest's own first real figure and a sixteen-fold cliff. Frederiksberg and Gazi again, and
    # unlike those two Brussels is already on the map in its own right, so the early half is a
    # straight double-count rather than the only copy of the capital. See DROP_YEARS for the
    # matching pair at the other end of the same entry.
    "Vorst-Belgium": 1901,
    # The Baixada Fluminense municipalities west of Rio, both split out of Nova Iguaçu in 1990,
    # both filed `is_agglomeration_of: rio de janeiro`, and both carrying the parent's figures
    # for every year before their own first census. Belford Roxo ramps to 1,208,448 by 1990 and
    # holds it verbatim to 1999 before reading 433,100 in 2000; Queimados does the identical
    # thing at 1,193,373. That put three ~1.2M dots within 15km of each other through the 1990s
    # (Mesquita was the fourth -- see DROP_KEYS, which has no year to clip to).
    # Caveat worth recording: Queimados' surviving 410,998 still looks high against a 2000
    # census of ~122,000, so the entry may be mis-assigned outright rather than merely early.
    # Clipping removes the phantom, which is the visible defect; the level is a separate call.
    "Belford Roxo-Brazil": 2000,
    "Queimados-Brazil": 2000,
    # Kaohsiung, and the Qingdao shape down to the round numbers: 220,000 held 1880-1890, then
    # 100,000 held 1898-1920, then 35,400 in 1921. Two flat blocks of a round figure, each
    # ending on a cliff, and the last of them landing on the first real municipal count -- which
    # is the signature of an administrative area handed to a city series, the same defect
    # trim_admin_tail() removes from the Chinese entries except that here it is at the FRONT.
    # 220,000 is roughly Qing Fengshan county, whose territory modern Kaohsiung sits on; 100,000
    # is plausibly the Japanese Takow district (打狗支廳); 35,400 is Takao town (高雄街), created
    # in 1920, which is what every later figure in the entry continues.
    # Takow was a treaty port from 1864 and the consular trade reports of the 1880s describe a
    # small harbour settlement at Kihou and Takow -- thousands, not hundreds of thousands. There
    # is no corroboration anywhere in the shipped sources: chandlerV2.csv's Kaohsiung row starts
    # at 1950: 261,000, and Chandler carries nothing at all for Taiwan before Taipei's 1900.
    # What it cost on the map: Taipei's entry does not open until 1898 and Tainan -- the island's
    # capital and its largest city throughout the Qing -- is 70,000, so the map made Kaohsiung
    # the biggest city in Taiwan by three-fold for the whole late Qing, at a size it did not
    # reach until the 1950s. Clipping to 1921 deletes a false claim without asserting a new one.
    "Kaohsiung-Taiwan": 1921,
    # Abomey-Calavi carries ABOMEY's history, 85km away, and the Chan Chan shape is exact:
    # chandlerV2.csv files the row as `Abomey-Calavi / Benin` but geocodes it (7.18286,
    # 1.99119) -- which is Abomey, 300m from Stadester's own `Abomey-Benin` entry and 85km
    # north-west of the Abomey-Calavi this entry is filed at (6.4503, 2.3468). The entry says
    # so itself: `particulars: agglomeration of Cotonou`, `is_agglomeration_of: cotonou`, and
    # `chandler_modelski_coords: [7.18286, 1.99119]`.
    # So the four figures (1750/1780/1800: 24,000, 1861: 20,000) are the Dahomey royal capital
    # -- the state that ran the Ouidah slave trade -- drawn on a Cotonou suburb, and Abomey
    # itself starts at 1921: 9,200. Nothing else in any shipped source covers that coast before
    # the twentieth century (Elmina 1880, Ouidah 1921, and neither has a Chandler row), so this
    # is the whole of the Bight of Benin's pre-colonial record and it is in the wrong place.
    # See CENSUS, which puts the same four figures on Abomey; without both halves the numbers
    # would either be lost or drawn twice.
    # 1862 rather than a full clip: 1992: 21,300 is a real populstat figure for Abomey-Calavi
    # town and the 1870-1991 ramp between it and Chandler's 1861 is fill either way. The entry
    # is left as the ~21,000 dot it already was, which is a separate (small) question -- it sits
    # 6.5km from Cotonou's 2.5M centre, inside it, so it double-counts whatever we do.
    "Abomey-Calavi-Benin": 1862,
}

# --- Chandler's AD 100 benchmark, filed 200 years early --------------------------------
# `data/chandlerV2.csv` has both a `BC_100` and an `AD_100` column, and the BC_100 one is
# actually Chandler's AD 100 table. Three independent lines of evidence, none of which needs
# a judgement call:
#
#   1. The two columns are COMPLEMENTARY, not overlapping. 27 cities have BC_100, one has
#      AD_100, and only Cadiz has both. Real benchmark columns do not partition like that.
#   2. ROME is the sole occupant of AD_100 (450,000) and is ABSENT from BC_100. No 100 BC
#      world-cities table omits Rome; no AD 100 table consists of Rome alone. Read together
#      the two columns are one coherent AD 100 list -- Rome 450k, Seleucia 250k, Antioch 150k,
#      Anuradhapura 130k, Carthage 100k, Smyrna 90k, Athens 75k, Chengdu 70k ... London 30k.
#   3. The contents are impossible as 100 BC. LONDON 30,000 -- Londinium was founded c. AD 47.
#      LYON 50,000 -- Lugdunum was founded in 43 BC and was a colony of a few thousand.
#      NIMES 44,000 -- colonia from c. 28 BC. CORINTH 50,000 and CARTHAGE 100,000 -- both were
#      razed by Rome in 146 BC and lay derelict until Caesar refounded them in the 40s BC.
#      OSTIA 30,000 -- its boom is the Claudian and Trajanic harbours, AD 42 onwards.
#      Every one of those figures is right for AD 100 and impossible for 100 BC.
#
# Stadester absorbed the mistake, so it arrives here as a -100 anchor and then gets straight-
# line-filled forward for centuries. The visible cost is a city drawn at full Roman size
# through the two centuries before it existed, and a fade-out or a gap that starts 200 years
# too early (Ostia currently fades out at AD 26, which is sixteen years before its harbour was
# dug). Note the correction generally makes a gap LONGER, not shorter, so it feeds the
# real-year arm of fade_long_gaps rather than competing with it.
#
# Listed by hand rather than derived, because build.py does not otherwise read chandlerV2.csv
# and one CSV load is not worth it for a closed set of twenty. `--ad100` prints what moved,
# and main() reports the count, so a source update that renames a key goes loud instead of
# quiet. Each entry's chandler_modelski_key is given, since several are filed under a modern
# name that hides which city this is.
CHANDLER_AD100 = {
    "Ankara-Turkey",                   # Ankara       34,000
    "Duhuang-China",                   # Dunhuang     32,000  (single-point entry)
    "Fiumicino-Italy",                 # -> Ostia     30,000
    "Gazi-Greece",                     # -> Athens    75,000  (see todo: Gazi carries Athens)
    "Kórinthos-Greece",                # -> Corinth   50,000
    "London-United Kingdom",           # London       30,000
    "Lyon-France",                     # Lyon         50,000
    "Milano-Italy",                    # Milan        30,000
    "Mires-Greece",                    # -> Gortyn    30,000
    "Nîmes-France",                    # Nimes        44,000
    "Oxyrhyncus-Egypt",                # Oxyrhynchus  34,000  (single-point entry)
    "Teotihuacán-Mexico",              # Teotihuacan  45,000
    "Thessaloniki-Greece",             # Thessalonica 35,000
    "Wâdî Moosa-Jordan",               # -> Petra     30,000
    "Yarîm-Yemen",                     # -> Zafar     60,000
}
# NOT shifted, deliberately. Five more entries carry a BC_100 value but ALSO hold a real anchor
# inside the window the shift would have to clear -- Carthage (Al Marsâ, 0 = 175,000),
# Chengdu (1 = 250,000), Cadiz (0 = 63,500), Smyrna (Izmir, 0 = 107,500) and Capua (Santa
# Maria Capua Vetere, 0 = 103,000). There the mis-filed figure is a duplicate of a value the
# series already has at the right date rather than the thing anchoring the era, so moving it
# would collide; deleting it instead is a separate call and Carthage in particular wants one
# (razed 146 BC, refounded 29 BC, so its real shape is a trough, not either figure).
# Cadiz is the one city Chandler lists in BOTH columns, at 62,000 and 65,000 -- which is either
# the single genuine BC_100 entry or the duplication that gives the whole error away.
# Istanbul is also left alone: its chandler key is misspelled "Instanbul-Turkey", and 36,000
# for Byzantium in 100 BC is defensible on its own, so there is nothing to buy by touching the
# highest-profile series on the map.
AD100_FROM, AD100_TO = -100, 100
# Tolerance for "this point is straight-line fill". Not 1e-9: Stadester's spline leaves float
# dust (200000 vs 200000.00000000006) and an exact test promotes dust into a fake anchor,
# which would stop the shift dead. Same reasoning as CF_EPS.
AD100_EPS = 1e-6
AD100_LOG = []

def fix_chandler_ad100(raw):
    """Move each mis-filed benchmark from 100 BC to AD 100, taking its stale fill with it.

    The fill matters. Stadester interpolates from the -100 anchor forward to whatever the next
    REAL anchor is, so those intervening annual values were all computed from a point that is
    about to move -- leaving them behind would join AD 100 to a line drawn from 100 BC and
    manufacture a cliff (Istanbul would read 36,000 at AD 100 against 254,400 at AD 200).
    So the whole span from -100 up to the next real anchor goes, and the value is re-planted
    at AD 100. Deleting fill costs nothing: dp_simplify would have collapsed it anyway, and
    the viewer re-interpolates between control points.

    Runs on the raw dict before anything else reads it, so the New World cap, the carry-forward
    strip and the fade rules all see the corrected dates."""
    for key in sorted(CHANDLER_AD100):
        c = raw.get(key)
        pop = c and c.get("population")
        if not pop:
            continue
        yrs = {}
        for ystr, v in pop.items():
            try:
                y = int(ystr)
            except ValueError:
                continue
            v = v[0] if isinstance(v, list) else v
            if v:
                yrs[ystr] = (y, float(v))
        start = next((ys for ys, (y, _) in yrs.items() if y == AD100_FROM), None)
        if start is None:
            continue
        v0 = yrs[start][1]
        pts = sorted((y, v) for y, v in yrs.values())
        anchor_years = [y for y, _ in dp_simplify(pts, AD100_EPS)]
        nxt = next((y for y in anchor_years if y > AD100_FROM), None)
        if nxt is not None and nxt <= AD100_TO:
            continue                       # a real anchor already occupies the window
        end = nxt if nxt is not None else max(y for y, _ in pts) + 1
        for ys, (y, _) in list(yrs.items()):
            if AD100_FROM <= y < end:
                del pop[ys]
        pop[str(AD100_TO)] = v0
        AD100_LOG.append((key, v0, end))
    return AD100_LOG


# --- individual bad data points to delete ------------------------------------------
# For one-off definition switches too small to trip the whipsaw check (<2x) but plainly
# visible on the graph. London's 1875/1880 pair is Greater-London-sized inside an otherwise
# County-of-London run, so the line jumps to 4.24M and falls back to 3.81M in 1881.
DROP_YEARS = {
    # 1998-1999: the BASE bleeding back in after the Greater London variant ends. PREFER_VARIANT
    # keeps base years outside the variant's range, the variant stops at 1997, and the base is
    # holding its 1974 figure of 10,427,084 verbatim all the way to 1999 -- so London was drawn
    # at 10.4M for two years between 7.19M and 7.07M. Invisible to every check because it is an
    # exactly-EQUAL PAIR: despike sees 10.4/10.4 = 1.0 on one leg and check F sees no flip.
    "London-United Kingdom": {1875, 1880, 1998, 1999},
    # Paris has the identical defect in the identical years -- populstat's 1914 and 1925 rows
    # are agglomeration figures (4.0M, 4.8M) dropped into a Ville-de-Paris series that is
    # ~2.9M on either side. Unlike London there is no separate variant entry to switch to,
    # so the spikes just go. (Its 1946->1962 climb is still a definition change rather than
    # growth, but it is at least monotone; check F keeps reporting it.)
    "Paris-France": {1914, 1920, 1925},
    # Rome's late antiquity, and the most editorial entry in this table -- it deletes a real
    # Chandler benchmark, so the reasoning had better be visible.
    #
    # As drawn, Rome fell 999,497 (300) -> 150,000 (361), a 6.7x collapse in 61 years, and then
    # held 150,000 FLAT for the next 239. Neither half is a measurement of anything. The cliff
    # is a SOURCE SWITCH: 300 is populstat and 361 is Chandler (the new `s` string says so, and
    # the growth mode now draws that step white), and the two disagree 2.2x about how big Rome
    # ever was -- Chandler's own AD 100 figure is 450,000 against populstat's ~1,000,000. The
    # plateau is then Chandler's 361 benchmark carried forward, while Chandler's own row says
    # 100,000 at 500 and 50,000 at 600.
    #
    # We already prefer populstat over Chandler on both sides of this: 999,497 over 450,000 at
    # 100, and 150,000 over 50,000 at 600. The 361 row is the ONE place the entry switches, and
    # it is exactly what manufactures the cliff. Removing it makes the stretch internally
    # consistent and lets 300 -> 600 interpolate: 729k at 350, 531k at 400, 282k at 500, 206k
    # at 550. That is a late-antique decline rather than a cliff and a plateau, and it matches
    # the mainstream reading (Rome near a million through the 2nd century, still several hundred
    # thousand in 400, collapsing after the 410 and 455 sacks and the Gothic War).
    #
    # What it costs: the 600 -> 622 step (150,000 -> 50,000) becomes the visible seam instead,
    # and that one is honest -- it is populstat handing to Chandler again, it lands in the right
    # era for Rome's real collapse, and the growth mode now colours it white and says so.
    # Range, not a bare year, so the 400 and 500 fill goes with the anchor.
    "Roma-Italy": set(range(301, 600)),
    # Vorst's other end, and the same defect mirrored: CLIP_BEFORE takes Brussels' pre-1901
    # figures off the front, and these two put them back on the tail. The commune's own record
    # runs to 54,876 (held 1969-1998, which is right -- Forest is ~55,000), then reads 94,438 in
    # 1999 and 134,000 in 2000. Neither is Forest; both are agglomeration-scale.
    "Vorst-Belgium": {1999, 2000},
    # Carthage's -100: 100,000 is Chandler's AD 100 figure filed 200 years early -- the same
    # error CHANDLER_AD100 corrects for fifteen other entries. It is NOT in that set because the
    # shift needs an empty window and Carthage holds a real anchor at 0 (175,000), so the two
    # would collide; §3.1's note says so and adds that "deleting it instead is a separate call
    # and Carthage in particular wants one". This is that call. Without it the razed century is
    # anchored at 100,000 by a figure that belongs to the Roman colony 250 years later, and the
    # DISAPPEARED span below would have to be planted across a real-looking datum.
    "Al Marsâ-Tunisia": {-100},
    # Merida's 1194 row is Chandler's MAYAPAN benchmark, which Chandler geocoded onto Merida and
    # Stadester then fused in. See SYNTHETIC, which gives Mayapan its own entry at the real site.
    # Harmless as drawn -- 25,000 is also what Chandler's own Merida row says at 1500 and 1528,
    # so removing it changes no pixel -- but leaving it would put one figure in two places.
    "Mérida-Mexico": {1194},
    # Frederiksberg is an enclave municipality inside Copenhagen, and its entry is really
    # COPENHAGEN's series (12,000 in 1101 rising to 1,268,867 in 1974 -- Frederiksberg itself
    # has never exceeded ~105k) with a handful of genuine Frederiksberg census rows dropped
    # into it. See RENAME below for why this entry is the one carrying the capital. These are
    # those rows: 1921 and the whole 1927-1969 block (103,400 rising only to 113,200, held
    # verbatim from 1941), plus a final 91,400 in 2003. 1970-1973 go with them because they
    # are the source's straight-line fill FROM 113,200 up to Copenhagen's 1,268,867, so they
    # interpolate between two different cities and mean nothing on their own. What survives is
    # Copenhagen: 620,000 in 1914 and 1920, 790,000 in 1925, 1,268,867 from 1974.
    "Frederiksberg-Denmark": {1921, 2003} | set(range(1927, 1974)),
    # Manchester, now that PREFER_VARIANT no longer masks it (see below). The base is a clean
    # city-proper census run -- 338,300 (1861), 355,700 (1871), 341,500 (1881), 505,300 (1891),
    # 543,900 (1901), 714,300 (1911), 730,000 (1921), 766,400 (1931), 732,900 (1938) -- with
    # three SE Lancashire conurbation rows dropped into it, each held to the next round year in
    # populstat's usual way. Ranges, so the straight-line fill on either side goes too and does
    # not get promoted into a fresh anchor.
    "Manchester-United Kingdom": set(range(1872, 1881)) | set(range(1912, 1921)) | set(range(1922, 1931)),
    # Memphis. This entry is TWO SOURCES fused: its chandler_modelski_key is "Memphis-Egypt",
    # so Chandler-Modelski supply ancient Memphis (-2500..-300) and populstat supplies the
    # modern Egyptian town of Al-Badrashayn (1974..2002). Stadester glued them together and
    # filled the 2,274-year hole between with a straight line at a dead-constant +327.91/yr.
    # The right-hand anchor of that line, 845,672 in 1974, is not a plausible figure for the
    # town and the dataset says so on its own: of 98 Egyptian entries carrying both a 1974 and
    # a 2002 value, 95 GREW, and Badrashayn is the only one that falls by more than 3x -- it
    # falls by 14.5x, to 58,500. Its neighbours in Giza province were 39,184 (Awsim), 64,192
    # (Al-Hawamidiyah) and 110,826 (Warraq al-'Arab) in 1974, which is exactly the tier its own
    # 2002 figure sits in. Nothing else in the dataset carries a value near it. Most likely a
    # markaz or governorate total filed against the town; either way it is ~20x too big, and it
    # is then held verbatim for 28 years.
    # The range covers the FILL as well as the anchor, per the note above about DROP_YEARS
    # needing to be a range: dropping only 1974-2001 would promote 1973 (845,344) into a fresh
    # anchor and change nothing. Note this also removes the AD 0 value of 198,374, which reads
    # as data on the chart but is computed -- 100,000 + 327.91 x 300, a point on the fill line.
    "Badrashayn, Al--Egypt": set(range(-299, 2002)),

    # --- populstat's conurbation benchmark rows -----------------------------------------
    # The alternating half of check F, and the reason despike() is structurally blind to it:
    # populstat enters a metro benchmark and HOLDS IT VERBATIM to the next round year, so the
    # wrong-unit value arrives as an exactly-EQUAL PAIR -- 1875=1880, 1914=1920, 1925=1930 --
    # wedged between decennial censuses that are city-proper. Neither member of a pair is out
    # of line with BOTH its neighbours, so no single-point rule can see it. Same defect as the
    # London and Paris entries at the top of this table.
    # Every census flank below was checked against the real returns: Birmingham CB 1921 =
    # 919,438 and 1931 = 1,002,603; Liverpool 1921 = 802,940, 1931 = 855,688; Newcastle 1921 =
    # 274,955, 1931 = 283,155. The paired values are the West Midlands, Merseyside, Tyneside,
    # Clydeside and Upper Silesian conurbations.
    # Ranges, not bare years, so the source's straight-line fill goes too -- dropping only the
    # anchor promotes the adjacent fill into a fresh anchor and changes nothing.
    "Birmingham-United Kingdom":   set(range(1872, 1881)) | set(range(1912, 1921)) | set(range(1922, 1931)),
    "Liverpool-United Kingdom":    set(range(1872, 1881)) | set(range(1912, 1921)) | set(range(1922, 1931)),
    "Newcastle upon Tyne-United Kingdom": set(range(1912, 1921)) | set(range(1922, 1931)),
    "Dublin-Ireland":              set(range(1872, 1881)),   # 333,000 against 245,700 / 249,600
    "Edinburgh-United Kingdom":    set(range(1872, 1881)),   # 273,000 against 197,600 / 228,200
    "Glasgow-United Kingdom":      set(range(1922, 1931)),   # 1,396,000 = Clydeside (1921: 1,034,200)
    "Katowice-Poland":             set(range(1922, 1931)),   # 565,000 = the Upper Silesian
                                                             # district; the town was 104,900 in 1921
    # Chorzów is Katowice's neighbour and has the same conurbation rows, one round earlier and
    # far bigger: 66,100 (1905) -> 307,156 (1910) -> 500,000 (1914, held verbatim to 1920) ->
    # 74,800 (1921). So a 500,000 dot sat on a town of ~75,000 for seven years while Katowice
    # next door read 105,000. Invisible to every check for the reason §6.3 gives -- the equal
    # pair means neither 1914 nor 1920 is out of line with BOTH its neighbours. The range starts
    # at 1906 to take 1910 and the fill with it; stopping at the anchors would promote the fill
    # rising out of 1905 into a fresh 300k anchor and change nothing on screen.
    "Chorzów-Poland":              set(range(1906, 1921)),
    # Calcutta's 1912-30 rows are the usual pair. The 1810-20 pair is the same defect a century
    # earlier and much bigger: 584,208 held verbatim across BOTH rows, against 152,625 in 1790
    # and the 1821 census of the town, 179,917 -- a 3.2x round trip that reads on the graph as a
    # spike and a cliff. Chandler is the authority for this city before the censuses and settles
    # it: his own row has Calcutta at 162,000 in 1800 and 179,917 in 1821, so there is no room
    # for 584,208 in 1810 on any definition of the town. It is Calcutta-with-suburbs or a
    # district total. Range starts at 1800 to take the fill (1800 = 368,417 is the exact linear
    # midpoint of 1790 and 1810, not a datum); what survives is 1790 -> 1821, which passes
    # within a few percent of Chandler's 1800.
    "Calcutta-India":              set(range(1800, 1821)) | set(range(1912, 1921))
                                                          | set(range(1922, 1931)),
    "Johannesburg-South Africa":   set(range(1922, 1932)),   # 538,000 exceeds the 1936 census
                                                             # (519,384): it is the Witwatersrand
    "Antwerpen-Belgium":           set(range(1922, 1927)),   # 466,000 vs the city's 300,200 in 1927
    "Frankfurt am Main-Germany":   set(range(1912, 1916)),   # 550,000 = agglomeration; the city
                                                             # was 414,576 in 1910
    "Bordeaux-France":             set(range(1921, 1926)),   # 369,000 = agglomeration; the commune
                                                             # was 267,409 (1921) and 258,914 (1926)
    "Marseille-France":            set(range(1872, 1876)),   # a DIP: 224,000 against the 1876
                                                             # commune census of 318,868 a year later
    "Riga-Latvia":                 set(range(1921, 1925)),   # 558,000 is the 1913 pre-war figure
                                                             # mis-dated; the 1925 census was 337,699.
                                                             # The 1917 fall to 225,000 STAYS -- that
                                                             # is the real wartime evacuation.

    # --- China: city / county (xian) / prefecture / tri-city figures interleaved ---------
    # Different compilers on a shared set of anchor years (1875, 1879, 1911, 1914, 1918, 1925,
    # 1926, 1936, 1948). Which level is right differs per city, so each is judged on its own.
    "Beijing-China":   set(range(1876, 1911)) | set(range(1912, 1926)),
        # 1,648,800 held from 1879 to 1910 is Chandler's 1850 Peking figure (1,648,000) re-entered
        # and carried forward for thirty years -- the 1910 police census counted 785,442. Keeps
        # 1875: 900,000 and 1911: 693,000 and 1926: 811,100, which are the real counts, and keeps
        # the 1.56M from 1936 on, which is legitimately the enlarged Beiping municipality.
    "Wuhan-China":     {1914, 1925},
        # The one that INVERTS the rule: here the HIGH rows are correct. 1,444,000 (1918) and
        # 1,583,900 (1926) are the tri-city Hankou+Wuchang+Hanyang; 700,000 and 818,000 are Hankou
        # alone, and no Hankou entry exists anywhere in the source, so an entry named Wuhan must
        # carry the tri-city figure. 1939: 760,000 stays -- the city was evacuated under occupation.
    "Hangzhou-China":  set(range(1841, 1875)) | set(range(1876, 1911)) | {1926, 1930},
        # 200,000 in 1875 is right: the Taiping destroyed Hangzhou in 1861-64. The 424,400 held to
        # 1870 and the 400,000 at 1879-1890 are the pre-Taiping level re-asserted across the
        # catastrophe. 1926=1930: 1,000,000 is the county. NB the range STOPS at 1910 -- 594,000
        # (1911) and 684,100 (1918) are genuine and a wider range would leave a 50-year hole.
    "Changsha-China":  set(range(1912, 1925)) | set(range(1926, 1936)),
        # 535,800 (1918) and 607,000 (1926) are Changsha county; 250,000 (1911) and 260,000 (1925)
        # are the city.
    "Chongqing-China": set(range(1891, 1912)),   # 598,000 (1911) breaks a smooth 292,033 ->
                                                 # 437,600 -> 608,000
    "Chengdu-China":   set(range(1937, 1948)),   # 315,000 (1939) sits between 458,000 (1936) and
                                                 # 727,000 (1948), and Chengdu GAINED wartime refugees
    "Harbin-China":    set(range(1912, 1926)),   # 28,600 (1918) is below the 1911 value in a city
                                                 # that only grew; 381,000 (1925) is the greater
                                                 # urban area against the 1926 city census of 164,900
    "Wenzhou-China":   set(range(1841, 1882)),   # 500,000 (1879) is the prefecture, 57,000 (1882)
                                                 # the city; everything between 1840 and it is fill
    # Held back deliberately: Guangzhou and Shanghai are compilers disagreeing rather than two
    # definitions, and St Petersburg's 2,318,600 is held flat 1915-1924 (a carry-forward, not a
    # flip) while the 1920 census of 722,000 that would replace it is not in the source at all.

    # --- Baghdad's 1150, and the third editorial deletion of a Chandler benchmark ----------
    # chandlerV2.csv's Baghdad row reads 1100: 150,000 · 1150: 10,000 · 1200: 100,000. The middle
    # figure is contradicted by its own neighbours in its own row: a 93% loss and a tenfold
    # recovery, each inside fifty years, is not a demographic trajectory, and no destruction of
    # Baghdad happened between 1100 and 1200. The city in 1150 was the seat of al-Muqtafi, whose
    # restored caliphate was strong enough to withstand a Seljuk siege in 1157. Read as a digit
    # rather than a datum -- 100,000, the value Chandler gives at 1200, 1250 and 1350 -- it stops
    # being anomalous at all.
    # It matters more than a stray row usually would because of what sits on top of it: with
    # CF_KEEP withdrawn and CENSUS restoring the rest of the row, 1150 is the single remaining
    # point that keeps Baghdad dead through the 12th century, and the events file has a 1258
    # Mongol note whose whole difficulty is that the map has already killed the city a hundred
    # years early. Nothing in the pipeline can reach it: despike() is up-only by construction
    # and check F needs a return inside OSC_SPAN = 20 years.
    # This is the Roma 361 call again and it is deleting real source data, so: if the reading
    # ever changes, deleting this one line restores it.
    "Baghdád-Iraq": {1150},
    # Benin City's 1900, and a defect the MERGE_INTO fix for §3.4 created. The donor variant
    # carries Chandler's `Benin / Oedo` benchmarks to 1854: 60,000 and then a populstat
    # agglomeration count of 762,700 at 1991 -- the Nigerian census figure -- with 137 years of
    # stadester's straight line between them and not one datum on it. The merge folds in every
    # donor year outside the base's 1901..1995, so the whole ramp came with it, and the map drew
    # Benin City at 295,943 in 1900 (a fill point, `i` in the provenance string) handing to the
    # base's real 1901 count of 15,000. A 19.7x cliff in one year, and the sixth largest jump in
    # the dataset by analyze_jumps.py -- larger than any seam.
    # Dropping the ramp leaves the two things that are measurements: Chandler to 1854 and
    # populstat from 1901, interpolating 60,000 -> 15,000 across the 47 years that contain the
    # 1897 Benin Expedition, which burned the city and exiled the Oba. That is a claim both
    # sources support at their own ends and neither contradicts in between.
    # The range stops at 1990 rather than 1991 to record that the 762,700 is real data and is
    # being declined on definition grounds, not deleted as fill: it is an agglomeration count
    # and the base is city proper. It never reaches the map either way -- 1991 is inside the
    # base's range, so the merge does not fold it -- but it is the figure to reach for if the
    # 1995 -> 1996 seam (224,000 -> 1.05M, Africapolis) is ever softened.
    "Benin City (agglomeration?)-Nigeria": set(range(1855, 1991)),
}

# --- prefer a metro variant's series over the base entry's -------------------------
# v1's rule is "city-proper history, agglomeration only via the modern graft", and the
# parenthetical variants are dropped. That breaks down when the BASE entry is itself
# internally inconsistent -- populstat's London interleaves two definitions year by year:
#   1911: 4,521,700   (County of London)      1914: 7,419,000   (Greater London)
#   1921: 4,484,500   (County of London)      1925: 7,742,000   (Greater London)
# so the map showed London sawtoothing between 4.5M and 7.7M through the 1910s-20s, then
# sitting at ~4M until 1960 while the real city was over 8M -- and 1961-1968 is a straight
# interpolation across the definition change, not data. The "(Greater London)" variant is a
# clean, consistent series over exactly that span (6.34M in 1901 -> 8.20M in 1931), so for
# the years it covers we take it and let the base entry supply the deep history.
PREFER_VARIANT = {
    "London-United Kingdom": "London (Greater London)-United Kingdom",
    # Manchester was here for the same reason and has been REMOVED: its variant carries only 3
    # real anchors between 1900 and 1975 against the base's 13, so the override was buying a
    # consistent definition by deleting the 1911, 1921, 1931, 1938 and 1947 censuses and
    # drawing 1901 -> 1970 as one straight line. That is the trade VARIANT_MIN_ANCHOR_FRAC now
    # refuses automatically; the hand entry was bypassing it. London stays because its variant
    # has 8 real anchors (1901-1950) and loses nothing -- it is the exception, not the pattern.
    # Manchester's base flips are handled the cheap way instead, in DROP_YEARS above.
}

# --- hand-entered census figures ---------------------------------------------------------
# The ONLY place in the pipeline where a population is typed by hand instead of derived from a
# source file. Everything else here deletes, clips or reassigns what the sources say; the coord
# repairs deliberately resolve to real WUP centroids rather than typed coordinates. So the bar
# is high, and an entry needs all four of:
#   1. a NAMED census, not a recollection or a round number;
#   2. a geographic definition that MATCHES what the surrounding series measures -- a correct
#      figure on the wrong unit is worse than no figure, since it manufactures exactly the
#      definition oscillation the rest of the pipeline exists to remove;
#   3. a real gap to fill, i.e. the source has no measurement of its own there;
#   4. a span of source fill/hold to CLEAR, so the new anchors are not fighting a held value
#      on either side of them.
# Format: key -> (clear_from, clear_to, {year: population}).
US1950 = "tools/us1950.csv"     # 91 hand-verified 1950 US censuses; see the file header


US_MODERN = "tools/us_modern.csv"   # 2010/2020/2024 place figures; see the file header
US_METRO  = "tools/us_metro.csv"    # annual MSA 2000-2024; see tools/make_us_metro.py


def load_us_metro():
    """Read tools/us_metro.csv into {key: {year: pop}} -- US metropolitan statistical areas.

    Highest-precedence modern source, above the FUA layer and above WUP. eFUA models an MSA;
    for the United States the Census Bureau publishes the real one annually, so where both
    exist there is no reason to prefer the model. It matters most where the model is worst:
    eFUA puts Boston at 2.79M against an MSA of 4.90M.

    Replaces the modern series outright rather than being spliced into it, exactly as the FUA
    layer does -- one definition per city across the whole modern range. It does NOT reduce the
    handover step, because populstat's American tail is city proper (its Atlanta 2000 is
    416,000 against a census place count of 416,474); going from that to any metropolitan
    figure is a real change of unit. What it fixes is which metropolitan figure.

    Returns (series, cbsa) -- {key: {year: pop}} and {key: cbsa code}. The second is only for
    reporting: one MSA is emitted under both "Dallas-Texas" and "Dallas-United States" because
    which of the two exists as an entry varies, so "did this metro land anywhere" has to be
    asked per CBSA and not per key."""
    out, cbsa_of = {}, {}
    if not os.path.exists(US_METRO):
        print(f"note: {US_METRO} missing -- no US metro figures applied")
        return out, cbsa_of
    with open(US_METRO, encoding="utf-8") as f:
        years = None
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# key,"):
                years = [int(c[1:]) for c in line[2:].split(",")[3:]]
                continue
            if not line.strip() or line.startswith("#"):
                continue
            # key,cbsa,"title with commas",v...
            key, rest = line.split(",", 1)
            cbsa, rest = rest.split(",", 1)
            _title, vals = rest[1:].split('"', 1)      # title is quoted; drop it
            pts = {}
            for y, v in zip(years or (), vals.lstrip(",").split(",")):
                if v.strip():
                    pts[y] = float(v)
            if pts:
                out[key] = pts
                cbsa_of[key] = cbsa
    return out, cbsa_of


def load_us_modern():
    """Read tools/us_modern.csv into {key: {year: pop}}. See load_us1950 for why this is a file."""
    out = {}
    if not os.path.exists(US_MODERN):
        print(f"note: {US_MODERN} missing -- no US modern figures applied")
        return out
    with open(US_MODERN, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, y10, y20, y24, _err = (line.split(",") + [""] * 5)[:5]
            pts = {}
            for y, v in ((2010, y10), (2020, y20), (2024, y24)):
                if v:
                    pts[y] = float(v)
            if pts:
                out[key] = pts
    return out


def load_us1950():
    """Read tools/us1950.csv into (census points, merge pairs).

    Kept as a data file rather than 91 more dict literals: the provenance, the unit check and
    the per-row error belong next to the numbers, and a hundred-entry Python dict is unreadable.
    Each row also names the co-located twin it must outrank, because a 1950 city-proper figure
    is only safe on an entry that is itself city proper -- see MERGE_INTO's US note."""
    pts, merges = {}, {}
    if not os.path.exists(US1950):
        print(f"note: {US1950} missing -- no US 1950 censuses applied")
        return pts, merges
    with open(US1950, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, pop, _err, _basis, merge = (line.split(",") + [""] * 5)[:5]
            pts[key] = (1950, 1950, {1950: float(pop)})
            if merge:
                merges[merge] = key
    return pts, merges


CENSUS = {
    # London's Greater London variant holds 8,322,100 verbatim from 1950 to 1996 -- 46 years of
    # carry-forward, well under CF_MIN_SPAN -- and then reports 7,187,300 for 1997. So the whole
    # postwar story is missing: Greater London fell by about a fifth to its early-80s trough and
    # then began recovering, and the map drew it as a flat line at its 1950 peak.
    # Figures are the decennial census counts for the Greater London area (UK ONS / GRO), which
    # is the unit the variant is on throughout. 1981 and 1991 are the unrevised usually-resident
    # counts; both were later revised upward (to roughly 6.81M and 6.83M) for census undercount,
    # and the unrevised ones are used here to keep the whole series on one basis.
    "London-United Kingdom": (1951, 1996, {
        1951: 8196978,
        1961: 7977178,
        1971: 7452346,
        1981: 6608598,
        1991: 6679699,
    }),

    # --- US rust belt: city proper, decennial census ---------------------------------------
    # Source for all of these: US Census Bureau Working Paper 27 (Gibson 1998), "Population of
    # the 100 Largest Cities and Other Urban Places in the United States: 1790 to 1990", tables
    # 13-22, cross-checked against Census table HS-7 for 2000. Unit is the incorporated place,
    # which is what these series already end on -- every surviving 2000 anchor matches its
    # census count to within 0.2%.
    # Keyed to the STATE entries, which MERGE_INTO now guarantees are the ones drawn. Keying
    # them to "-United States" (the metro twin) was the earlier mistake and would have put
    # city-proper figures onto an agglomeration series.
    #
    # CLEVELAND is the worst-drawn city in the dataset and the confirmed case of the trim
    # deleting a real peak: the source HAS 1906: 460,300, 1910: 545,820, 1916: 674,100,
    # 1920: 784,678 and 1925: 922,900, and the terminal unit switch arbitrated for the 2000
    # census and its backward ramp removal took all five with it -- so the city never peaked
    # and read 52% low at mid-century.
    "Cleveland-Ohio": (1900, 1999, {
        1900: 381768, 1910: 560663, 1920: 796841, 1930: 900429, 1940: 878336,
        1950: 914808, 1960: 876050, 1970: 750903, 1980: 573822, 1990: 505616,
    }),
    # Saint Louis and Pittsburgh were the two worst after Cleveland -- the drawn line ran ~95%
    # and ~92% above the truth at their 1970s trough.
    "Saint Louis-Missouri": (1931, 1999, {
        1940: 816048, 1950: 856796, 1960: 750026, 1970: 622236, 1980: 453085, 1990: 396685,
    }),
    "Pittsburgh-Pennsylvania": (1931, 1999, {
        1940: 671659, 1950: 676806, 1960: 604332, 1970: 520117, 1980: 423938, 1990: 369879,
    }),
    "Detroit-Michigan": (1931, 1999, {
        1940: 1623452, 1950: 1849568, 1960: 1670144, 1970: 1511482, 1980: 1203339, 1990: 1027974,
    }),
    "Chicago-Illinois": (1931, 1999, {
        1940: 3396808, 1950: 3620962, 1960: 3550404, 1970: 3366957, 1980: 3005072, 1990: 2783726,
    }),
    "Philadelphia-Pennsylvania": (1931, 1999, {
        1940: 1931334, 1950: 2071605, 1960: 2002512, 1970: 1948609, 1980: 1688210, 1990: 1585577,
    }),
    "Baltimore-Maryland": (1931, 1999, {
        1940: 859100, 1950: 949708, 1960: 939024, 1970: 905759, 1980: 786775, 1990: 736014,
    }),
    "Milwaukee-Wisconsin": (1941, 1999, {
        1950: 637392, 1960: 741324, 1970: 717099, 1980: 636212, 1990: 628088,
    }),
    # --- superseded note (kept so the reasoning is not lost) --------------------------------
    # Cleveland, Detroit, Chicago, Philadelphia, St Louis, Pittsburgh, Baltimore and Milwaukee
    # all have verified city-proper census series ready (US Census Bureau Working Paper 27,
    # Gibson 1998, tables 13-22), and Cleveland is the worst-drawn city in the dataset -- the
    # source holds 1906: 460,300 through 1925: 922,900 and the terminal-unit trim deleted all
    # of it, so the city never peaks and reads 52% low at mid-century.
    # They are blocked on a prior defect: each is filed TWICE at identical coordinates, as
    # "<city>-<State>" (city proper) and "<city>-United States" (metro), e.g. Cleveland-Ohio
    # peak 922,900 beside Cleveland-United States peak 2,134,395. Which one the map draws is
    # decided by dedup on point count, and the trim then resolves whichever tail it gets. Until
    # that pair is resolved -- MERGE_INTO or a drop, see the orphans/duplicates section -- a
    # hand census cannot be aimed reliably at the entry that will actually be drawn, and would
    # be papering over the duplicate rather than fixing it.
    # Seoul was investigated and needs NOTHING: the gap is in "Sõul (agglomeration)", which
    # DROP_MARKERS drops, while the entry actually drawn ("Sõul-South Korea", renamed to Seoul)
    # already carries 1935: 444,000 -- the Keijo census 444,098 -- plus 1938, 1949, 1959 and
    # 1965. Recorded so the 41-year "gap" is not re-investigated from the dropped variant.
    # --- ISTAT comune censuses --------------------------------------------------------------
    # Unit confirmed against the series' own values: Genova 1936: 635,000 vs the census 634,646
    # and Roma 1931: 1,010,000 vs 1,008,083. That second check also settles WHICH ISTAT series
    # to use -- these are the census-day counts, not the boundary-homogenised ones (which give
    # Roma 1931 = 916,858).
    "Roma-Italy":   (1968, 2000, {1971: 2781385, 1981: 2839638, 1991: 2775250}),
    "Torino-Italy": (1968, 2000, {1971: 1167968, 1981: 1117154, 1991: 962507}),
    "Genova-Italy": (1968, 2000, {1971: 816872,  1981: 762895,  1991: 678771}),
    # --- All-Union censuses, city of Tashkent ----------------------------------------------
    # Unit is city proper: the entry's own 2000 value (2,142,000) matches the published 2000
    # estimate. Current line reads +21% in 1959.
    "Tashkent-Uzbekistan": (1941, 1999, {
        1959: 911930, 1970: 1384509, 1979: 1780002, 1989: 2072459,
    }),
    # --- Amsterdam: gemeente ----------------------------------------------------------------
    # The source holds the 1959 peak (868,232) verbatim to 1970, so the postwar suburbanisation
    # trough is missing and the line reads +19% in 1985. Sources: Gemeente Amsterdam Onderzoek
    # en Statistiek ("De Amsterdamse bevolking sinds 1900") for 1985; CBS for 1 Jan 1995.
    # 1970/1980/1990 could not be sourced to the same standard and are omitted -- two points
    # are enough to put the shape right.
    "Amsterdam-Netherlands": (1960, 1999, {1985: 675570, 1995: 740275}),

    # --- recovered source figures, NOT hand-typed censuses ----------------------------------
    # The four conditions above are about typing a population we found ourselves. These are a
    # different thing using the same mechanism: the figure is already in `data/chandlerV2.csv`,
    # which we ship, and the Stadester fusion simply lost it. So the bar is:
    #   1. the figure is a row in a source file already in the repo, quoted verbatim;
    #   2. no entry on the map carries it, and the note says which entry was hiding it;
    #   3. the span being cleared is fabrication, not data.
    # Tula. The source entry is 100,000 held VERBATIM from -300 to 1999 -- 2,300 years of one
    # number, the largest single block of fabrication in the corpus. Toltec Tula flourished
    # c. 900-1150, so the entry was drawing a 100,000-person city for the twelve centuries
    # before it existed, and the strip then blanked it for the whole period it did. Meanwhile
    # chandlerV2.csv has `Tollan / Tula / Mexico (20.0637, -99.3410) AD_900 = 50,000` -- the
    # right city, the right date, at these coordinates -- and nothing carries it. Clearing the
    # whole fabricated span leaves the modern town's real 2000 census (86,799) untouched.
    "Tula de Allende-Mexico": (-300, 1999, {900: 50000}),
    # Lhasa. Chandler carries 765 and 800 at 100,000 (the Tibetan Empire's capital), then goes
    # SILENT for nine centuries, and picks the city back up at 1700: 80,000, 1750/1800/1840:
    # 50,000 -- the Dalai Lamas' Lhasa. Stadester filled the silence with the imperial figure AND
    # overwrote those four later benchmarks with it, producing 100,000 held flat from 765 to 1850.
    # The strip then correctly deleted the fabrication and, with it, the only record of Lhasa
    # between the empire and the 1854 census: the map blanked the city from 840 to 1830, erasing
    # the whole Ganden Phodrang era, the Potala and the city every 19th-century account describes.
    # Restoring Chandler's own four figures is what puts it back. The gap that remains is real --
    # see DISAPPEARED, where the dates are Langdarma's death and the Fifth Dalai Lama.
    "Lhasa-China": (766, 1853, {1700: 80000, 1750: 50000, 1800: 50000, 1840: 50000}),
    # Thanjavur, and NOT the case it first looked like. Chandler has no pre-1750 benchmark for it
    # at all -- its row opens at 1750: 30,000 -- so the 100,000 stadester holds from 900 to 1830 is
    # populstat's alone and contradicts Chandler three-fold where they meet. That is the Guangzhou
    # shape, not the Dali one, and the 900 figure is left as the single unsupported claim it is
    # rather than being propped up. What IS being lost is the other end: the run overwrote
    # Chandler's 1750 and 1800, so the Nayak and Maratha capital vanished with the fabrication and
    # the city did not reappear until 1831.
    "Thanjâvûr-India": (901, 1830, {1750: 30000, 1800: 30000}),

    # TEOTIHUACAN, and it is the Lhasa/Thanjavur shape again in the New World: stadester
    # overwrote the compiler's own later benchmark with a carry-forward and the city lost its
    # peak. chandlerV2's `Teotihuacan-Mexico` row reads
    #     AD 100 = 45,000 · AD 361 = 90,000 · AD 500 = 125,000 · AD 622 = 60,000
    # -- a rise to a Classic-period maximum, then the collapse. What stadester holds instead is
    #     400 = 103,168 · 500 = 103,168 · 600 = 103,168 · 622 = 60,000
    # i.e. an interpolated 400 value repeated across the two benchmark years after it. The
    # 125,000 is simply gone, and with it the shape: the map drew Teotihuacan PEAKING AT 400 and
    # sliding from there, when 400 is the middle of its growth and the peak is a century later.
    # provenance.py sees this directly -- it labels the 361 point ('chandler','exact') and the
    # 400 one ('populstat','default'), with 500 falling through to ('fill','fill').
    #
    # The consequence is a ranking error, not a cosmetic one. At AD 500 Teotihuacan was the
    # largest city in the Americas and one of the largest anywhere; the map instead drew it at
    # 80,746 and BELOW Caracol (120,000) at every year of the Classic -- and Caracol's figure is
    # Modelski's, with no Chandler row behind it at all (see GAPS.md 2.3/2.6).
    #
    # Window clears stadester's 400/500/600 and restores the benchmark, leaving Chandler's own
    # four points and nothing else. No number here is ours. The 622 = 60,000 collapse figure is
    # already correct in the source and is left alone; DISAPPEARED still dates the abandonment.
    "Teotihuacán-Mexico": (400, 621, {500: 125000}),

    # MONTE ALBAN, the one case in the Americas pass where the city already had an entry and
    # simply started 1,300 years too late. Chandler has exactly one row for it, AD 800 = 30,000,
    # so the Zapotec capital appeared on the map only as it was being abandoned -- founded
    # c. 500 BC, the dominant centre of the Valley of Oaxaca for a millennium, and drawn for the
    # last century of that. This is a CENSUS rather than an ARCHAEOLOGICAL entry precisely
    # because the entry exists: a second dot on the same hill would be the Danapur/Pataliputra
    # error, not a fix.
    # Figures: Richard Blanton, "Monte Alban: Settlement Patterns at the Ancient Zapotec
    # Capital" (1978) and the Valley of Oaxaca settlement survey -- roughly 5,000 in Period I
    # (500-200 BC), ~15,000 through Period II, and a Period IIIb peak usually given as
    # 16,000-25,000 around AD 500-700. 25,000 is the top of that range, chosen to meet Chandler's
    # own 30,000 at 800 without manufacturing a step; a reader who prefers Blanton's lower
    # figure should read the AD 600 point as 16,000-25,000.
    # Window ends at 799 so Chandler's benchmark is untouched -- and because the two tiers meet
    # there, index.html marks that segment as a source change, which is exactly right.
    "Santa Cruz Xoxocotlán-Mexico": (-500, 799, {-300: 5000, -100: 15000, 600: 25000}),

    # --- the three unrevised Modelski entries -----------------------------------------------
    # Different from everything above this line, and worth being explicit about it: these
    # OVERRIDE a figure the map already draws, rather than filling a hole. That is the case the
    # Guangzhou lesson says to be slowest about, so the test applied to each was "does the
    # entry's own benchmark source contradict it, or is it unsupported by ANY source" -- not
    # "does newer literature prefer a different number".
    #
    # All three are the same defect: Modelski's round 100,000 stamped on every century of a
    # New World site, which fixes the magnitude at a plausible peak but attaches it to the wrong
    # 500 years. nw_cap suppresses the deep-antiquity end of that stamp and always has; what it
    # cannot do is correct the DATE, so each of these was drawn at its peak size four to six
    # centuries early and then declining through its actual floruit. The intermediate points
    # below are phase-level shape from the settlement literature, not counts -- the comments say
    # which figure is load-bearing in each case.

    # CARACOL. The starkest of the three: it has NO chandlerV2 row at all, so its
    # `500 = 120,000 / 800 = 100,000` was the last New World figure on the map with no benchmark
    # source of any kind behind it -- and it made Caracol the largest city in the Americas for
    # six straight centuries and top-12 in the world at AD 700 and 800.
    # The magnitude is not the problem. Arlen and Diane Chase's LiDAR mapping of the Caracol
    # settlement system supports ~100,000 across its ~200 km2 at the peak, which is the same
    # kind of measure as the Copan valley figure in ARCHAEOLOGICAL. The DATE was: Caracol's
    # Late Classic peak is c. AD 650-700, after its defeat of Tikal in 562, and the site was
    # abandoned c. AD 900. The map had it peaking at 500 and dead at 800 -- backwards on both.
    # Carried at Chase & Chase's figure rather than silently re-estimated, as Cahokia is carried
    # at Chandler's: more conservative readings put the densely settled core nearer 50,000-65,000.
    # Window keeps the -200 emergence seed, which is right -- Preclassic occupation from c. 600 BC.
    "Caracol-Mexico": (0, 900, {1: 5000, 400: 30000, 700: 100000, 900: 15000}),

    # TIKAL, and here the entry's own source is the one being contradicted. chandlerV2 has
    # exactly two Tikal rows -- `751 = 63,000` and `800 = 40,000` -- and both are already on the
    # map and already right. What sat in front of them was Modelski's flat 100,000 from AD 100,
    # so Tikal was drawn 60% ABOVE its own benchmark four centuries before that benchmark's
    # date, and then declining across the Late Classic, which is when the great temples were
    # built. Chandler's 63,000 at 751 also matches Culbert et al. (1990) on the central 120 km2,
    # so the benchmark and the settlement survey agree and only the Modelski stamp dissents.
    # Window STOPS AT 750 so both Chandler rows survive untouched and keep their `c` -- the
    # inserted points are only the rise into them. 900: 10,000 dates the terminal decline
    # (abandoned c. AD 900) rather than letting the city vanish at Chandler's last row.
    "Tikal-Guatemala": (0, 750, {1: 10000, 400: 30000, 900: 10000}),

    # TIAHUANACO is barely an override at all -- it is the Lhasa shape, and the compiler's own
    # number has been sitting unused this whole time. chandlerV2 says `800 = 20,000`; stadester
    # holds 44,721 at that year and a flat 100,000 before it, so the map was drawing Tiwanaku at
    # 2.2x Chandler's figure at the one year Chandler speaks, and at 5x it for the seven
    # centuries before. Restoring 20,000 at 800 is therefore a CENSUS recovery, not a new claim,
    # and it happens to sit at the conservative end of the archaeological range too (Janusek and
    # others at 10,000-20,000; Kolata argued 30,000-60,000 for the urban core).
    # The dating is the rest of it: Tiwanaku's urban phase is c. AD 500-1000 with the peak at
    # 800-1000 and collapse c. AD 1000-1100, so the flat 100,000 from AD 100 predates the city
    # by four centuries. Window clears the -200 seed deliberately -- unlike Caracol and Tikal
    # there was no Preclassic city here to emerge from; the site is occupied from c. AD 110.
    "Tiahuanaco-Bolivia": (-500, 1200, {200: 3000, 500: 8000, 800: 20000, 1000: 22000,
                                        1150: 4000}),

    # Baghdad, 933-1400, and the largest single recovery in this table -- see the withdrawn
    # CF_KEEP entry for how the span came to be drawn the way it was.
    # Stadester overwrote SIX consecutive Chandler benchmarks here with two holds and one fill:
    #
    #   year   chandlerV2.csv   stadester        what stadester did
    #   1000        125,000     1,100,000        held the 932 value forward
    #   1100        150,000     1,100,000        held the 932 value forward
    #   1200        100,000        55,000        the arithmetic mean of 1150 and 1250
    #   1300         40,000       100,000        held the 1250 value forward
    #   1350        100,000       100,000        held (agrees by coincidence)
    #   1400         90,000       100,000        held the 1250 value forward
    #
    # so the only Chandler years that survived the fusion were 1150 and 1250, and the strip then
    # deleted 1300/1350/1400 as the tail of a carry-forward run (validate check G reports exactly
    # that, `2 lost, run 1250..1400`). Every figure below is quoted verbatim from Chandler's row,
    # 1250 included, so the whole span is one source rather than a patchwork -- CF_KEEP's absence
    # would otherwise leave 1250 as an island of Chandler between two of ours.
    #
    # What it restores is the shape of five centuries. As drawn before this the city fell 110x
    # between 1100 and 1150 and then recovered tenfold, and held 100,000 flat from 1250 to 1401
    # straight through the Mongol sack. As drawn after it, it declines from the Abbasid peak
    # across the Buyid and Seljuk periods, is 100,000 at 1250, is 40,000 at 1300 -- Hulagu took
    # the city in 1258 -- recovers to 100,000 under the Jalayirids by 1350, and is 90,000 at 1401,
    # the year Timur sacked it for the second time and the year Chandler's own next benchmark
    # falls on. None of that is our reading imposed on the source; all six numbers are his.
    # Honest caveat, the same one Cahokia's SYNTHETIC note makes: Chandler's 932: 1,100,000 is at
    # the very top of the literature for Abbasid Baghdad (400,000-700,000 is the commoner range),
    # and his 1000: 125,000 makes the fall from it very fast. Carried as he states it.
    "Baghdád-Iraq": (933, 1400, {
        1000: 125000, 1100: 150000, 1200: 100000, 1250: 100000, 1300: 40000, 1350: 100000,
        1400: 90000,
    }),
    # Abomey, capital of Dahomey. The receiving half of the CLIP_BEFORE above, and the same
    # mechanism as Chan Chan except that Abomey has a modern entry of its own, so the figures
    # move to it rather than needing a SYNTHETIC city. All four are chandlerV2.csv's
    # `Abomey-Calavi / Benin` row verbatim, at the coordinate Chandler gave them.
    # The window clears nothing -- Abomey's own record opens at 1921 -- and is written as
    # 1749..1920 to say which span the entry is being given rather than to delete anything.
    # What it buys: the Bight of Benin has a city on it between 1750 and 1861. Ouidah, the port
    # Dahomey conquered in 1727 and ran as the second-busiest slaving harbour in Africa, has no
    # record before 1921 in any shipped source and cannot be drawn; Abomey is the state behind
    # it and can be. The 1861: 20,000 lands two years before the French take Porto-Novo and
    # eleven before the first Franco-Dahomean war.
    # Junction check: 1861: 20,000 -> Abomey's own 1921: 9,200, interpolated across 60 years.
    # That is a decline the period supports (the 1892-94 conquest burned the palaces and the
    # court dispersed) and it is two real anchors from two sources with fill between, not a
    # claim of ours.
    "Abomey-Benin": (1749, 1920, {1750: 24000, 1780: 24000, 1800: 24000, 1861: 20000}),
    # Cordoba, 1100-1500, and read the second half of this note before touching the LEVEL.
    # The hole: stadester holds 79,125 verbatim at 1000, 1100, 1200, 1300, 1400 AND 1500 -- a
    # 500-year carry-forward -- so the strip keeps 1000 and the map draws one straight line from
    # there to 1550: 33,000. Five and a half centuries in a single segment, through the fitna of
    # 1009-1031 and through the Reconquista of 1236. The 1248 Sevilla note in events.json says so
    # from the other side: "Cordoba ... has no control point between 1000:79k and 1550:33k, so its
    # fall reads as a smooth 550-year decay rather than a conquest".
    # chandlerV2.csv has all five years and they are coherent and monotone -- 1100/1200: 60,000,
    # 1300: 40,000, 1400: 36,000, 1500: 30,000 -- and they meet the entry's own next figure
    # cleanly, 30,000 against stadester's 1550: 33,000, a 1.1x junction. The front junction is
    # 79,125 -> 60,000, 1.32x. Neither is a compiler collision; each year takes the one source
    # that has it.
    # Check G is BLIND to this and the reason is worth recording: it joins entries to Chandler
    # rows by coordinate at GBM_JOIN_KM = 5, and Chandler geocodes Cordoba to (38.046, -4.894),
    # 20km north-west of the city. See spec.md 6.13 -- 16 further runs sit in the 5-10km band.
    #
    # NOT DONE HERE, deliberately: the 1000 figure. Chandler says 450,000 and is the source of
    # the familiar "largest city in the world in AD 1000" claim; this entry is a BURINGH one and
    # says 79,125. Three reasons to leave it:
    #   1. it would OVERRIDE a drawn figure rather than fill a hole, which is the act the
    #      Guangzhou lesson says to be slowest about, and which GAPS.md Part 2 already flags as
    #      needing its own decision each time;
    #   2. Chandler's Cordoba row is not internally coherent where it matters -- 800: 160,000,
    #      900: 20,000, 1000: 450,000, 1100: 60,000 -- so taking his 1000 means also explaining
    #      why his 900 is ignored. Post-1100 the same row IS coherent, which is why only that
    #      half is used here;
    #   3. there is no consensus to defer to. Published estimates for caliphal Cordoba run from
    #      about 90,000 to a million; 100,000 is the classic built-area estimate, 250,000-450,000
    #      the range popular accounts quote, and Bosker-Buringh-van Zanden -- a modern peer-
    #      reviewed dataset, and the one this entry IS -- sit at the bottom of it.
    # So the map now draws Cordoba's decline with the shape its sources give it, and its
    # caliphal peak at the low end of a live dispute rather than at the high end.
    "Córdoba-Spain": (1001, 1549, {
        1100: 60000, 1200: 60000, 1300: 40000, 1400: 36000, 1500: 30000,
    }),
    # --- the rest of what check G's 5km join cannot see (spec 6.13) -------------------------
    # These three came out of measuring that blind spot rather than from the report, and they
    # are here rather than in a widened check because the measurement already did the finding:
    # a scan that runs once and produces four table rows is cheaper than a rule that runs every
    # build and needs a name gate to keep Lille out of Antwerp.
    #
    # CHANG'AN. Stadester holds 600,000 at 805, 900 AND 1000 and then reads 45,000 at 1077, so
    # the strip keeps 805 and the map draws the Tang capital sliding evenly across 272 years --
    # 243,000 in the year 900, against Chandler's own 500,000. Chandler's row is the right shape
    # and the map had lost it: 805: 600,000 -> 900: 500,000 (a ninth century of gentle decline)
    # -> 1077: 45,000 (Huang Chao took the city in 881 and Zhu Wen dismantled it in 904, moving
    # the capital to Luoyang; it never was one again). One figure restores the difference between
    # a slide and a catastrophe. Chandler has nothing at 1000, so nothing is planted there.
    # His row joins 8.5km from the entry -- outside the 5km join, which is the whole reason this
    # sat unreported.
    "Xi'an-China": (806, 1076, {900: 500000}),
    # LUOYANG, and the receiving end of that same collapse. Stadester holds one SPLINE value --
    # 264,157, not a measurement of anything -- at 700, 800, 900 and 1000, so the strip keeps 700
    # and the map runs 264,000 (700) to 40,000 (1077) as a single 377-year line. Chandler has two
    # real benchmarks inside it, 800: 300,000 and 1000: 50,000, which say the Tang eastern capital
    # was still at its height in 800 and finished by 1000 -- the An Lushan sacks of 756 and 762
    # are behind the first and the Song move to Kaifeng behind the second.
    # NOT restored: Chandler's 100: 420,000, which check G also lists. The map draws populstat's
    # 260,000 there and taking Chandler's would OVERRIDE a drawn figure on a 1.6x disagreement.
    # That is the Guangzhou case and it is declined, here as elsewhere.
    "Luoyang-China": (701, 1076, {800: 300000, 1000: 50000}),
    # BASRA, two runs and the same treatment. Stadester holds 100,000 from 717 to 1100 and then
    # 60,000 from 1123 to 1500, so 783 years of the city arrive as two flat blocks and the map
    # draws one line from 717 to 1123 and another to 1525. Chandler has five benchmarks across
    # them and they are a real curve: 100,000 through the eighth century, HALVED to 50,000 by
    # 1000 -- the Zanj revolt of 869-883 and the silting of the canal country -- a recovery to
    # 60,000 across the twelfth, and 50,000 at 1200 before the long fall to his 1525: 10,000,
    # which spans Hulagu's 1258 sack. 1123 and 1150 are quoted too although the first is already
    # drawn and the second is collinear, so the whole span is one source rather than a patchwork.
    "Basrah-Iraq": (718, 1524, {
        800: 100000, 1000: 50000, 1100: 60000, 1123: 60000, 1150: 60000, 1200: 50000,
    }),
}

# --- typed coordinates, for sites the repair machinery structurally cannot reach ----------
# The coord_fixes pipeline resolves a bad coordinate by matching the entry's NAME to a WUP
# urban centre and taking its centroid, which is why CENSUS can say the repairs "deliberately
# resolve to real WUP centroids rather than typed coordinates". That works because almost every
# mis-geocoded entry is a living city filed at the wrong living city.
#
# It cannot work for an archaeological site, because there is no modern centre to resolve TO --
# and those are exactly the entries a geocoder gets wrong, since the fallback it reaches for is
# a modern place-name that happens to match. So this is the coordinate equivalent of CENSUS: the
# one place a lat/lon is typed rather than derived, with the same kind of bar.
#   1. the entry's SERIES is not in doubt -- we are moving a known city, not inventing one;
#   2. the current coordinate is demonstrably a different place, not merely imprecise;
#   3. the destination is a specific identified site, and the note says which and how far.
# Format: key -> (lat, lon, note). Applied after coord_fixes.json so it wins any overlap.
SITE_COORDS = {
    # Chandler's Anyi is the capital of the state of Wei, occupied 562-339 BC and sitting at
    # Xia County in Yuncheng, Shanxi -- the Yuwangcheng site. The geocoder found Anyi COUNTY in
    # Jiangxi instead, ~800km south-east, and put a 100,000-person Warring States capital in the
    # Gan valley, which at 500 BC was Chu/Yue frontier with no urbanism of that scale anywhere
    # near it. Chandler's own figure (BC_500/400/300 = 100,000) is entirely defensible for Wei's
    # capital and impossible where it is drawn.
    "Anyi-China": (35.1394, 111.2206, "Yuwangcheng, Xia County, Shanxi -- moved ~800km NW"),
    # "Caracol" is the Maya city in the Chiquibul, Belize. The geocoder matched the name to
    # El Caracol, the observatory BUILDING at Chichen Itza, and filed the entry as Mexico at
    # (20.679, -88.571) -- 0.1km from Chichen Itza and 440km from Caracol. The series is
    # Caracol's and is right for it (500: 120,000, 800: 100,000 matches its Late Classic peak),
    # so this is purely a location error -- but as drawn it puts Caracol's population on Chichen
    # Itza's site, 400 years before Chichen Itza's own floruit, and then blanks it at 800 just
    # as the real city there was starting.
    "Caracol-Mexico": (16.7636, -89.1178, "Caracol, Cayo District, Belize -- moved ~440km S"),
    # Djenne is filed by Chandler itself as "Dienne / Guinea, Jenne / Senegal" at (15.033,
    # -16.350), which is in Senegal near the Atlantic. The city is on the Bani in the Niger
    # inland delta, Mali, ~700km east-south-east, beside Djenne-Djenno. Chandler's certainty on
    # this row is 3, its lowest, and the coordinate is part of why.
    "Dienne-Senegal": (13.9061, -4.5550, "Djenne, Mopti, Mali -- moved ~700km ESE"),
    # El Tajin is filed at longitude +97.3782. The site is at -97.3778: the minus sign is simply
    # missing, and the entry is drawn in the hills east of Lashio, Myanmar. It is not a geocoder
    # miss -- the digits are exactly right -- which is why no name-matching repair ever touched
    # it, and why it survived the region audit: `country` says Mexico, so in_americas() is
    # correct about it and the New World ramp treats it properly. Only the DOT is on the wrong
    # continent. The cost is not small: this is a 40,000-50,000 city at AD 622-1000, i.e. the
    # single largest Mesoamerican entry the map has between Teotihuacan's collapse and Tula, and
    # the whole Epiclassic Gulf coast is missing without it. (build.py's own region-conflict
    # counter cannot see this one -- it tests `lon < -30 and not new_world`, which catches an
    # Old World entry sitting in the Americas but not an American entry sitting in Asia.)
    "El Tajin-Mexico": (20.4472, -97.3778, "El Tajin, Papantla, Veracruz -- longitude sign flip"),
    # Hamilton, Ontario -- the same lost minus sign as El Tajin, landing a 515,000-person city
    # in the Tian Shan foothills of Xinjiang. Unlike Richmond (see DROP_KEYS) there is no other
    # Hamilton-Ontario entry, so this one has to be moved rather than dropped: the alternative
    # is that Canada's ninth-largest city is absent from the map. The series is a single 1975
    # point, which the modern graft then joins to the real Hamilton CMA.
    "Hamilton-Canada": (43.2557, -79.8711, "Hamilton, Ontario -- longitude sign flip"),
}

# --- display renames ----------------------------------------------------------
# Stadester files ancient capitals under the modern town sitting on the ruins, so
# the antiquity charts read "Faqus / Al-Uqsur / Hillah" instead of the names people
# know (and search for). Keyed by the exact source key to avoid renaming a same-named
# entry elsewhere (there are two "Hillah", two "Anyang"). Modern name kept in parens
# where the modern city is itself notable. NOTE: not for sub-district duplicates like
# "Kensington and Chelsea" (a real London already exists) — those need dedup, not rename.
RENAME = {
    # Copenhagen was missing from the map entirely. Both of its own entries -- "København
    # [stad]" (890k) and "København [agglom.]" (1.44M) -- have coords: null in the source, so
    # they are dropped as no-coord, and the propose_coords tooling never saw them either
    # because it can only repair an entry that HAS a coordinate to move. What was standing in
    # for the capital was this Frederiksberg entry, an enclave 1.8km away carrying Copenhagen's
    # whole series. Relabelling it is what puts Copenhagen back, and it keeps the deep history
    # the [stad] entry lacks (that one starts at 1750; this goes back to 1101). The figures are
    # agglomeration-sized from 1880 on, which the modern graft would have supplied anyway.
    # Costs Frederiksberg itself as a separate dot -- a 105k enclave inside the city drawn over
    # it, i.e. exactly the Kensington-and-Chelsea case the coord-fix tooling already drops.
    "Frederiksberg-Denmark":          "København (Copenhagen)",
    # The source files The Hague under its formal Dutch name, inverted for alphabetisation:
    # "Gravenhage, 's-". That is the correct city -- its own other_names are Den Haag, The Hague,
    # La Haye -- but nobody reads the inverted form as the seat of Dutch government, and there is
    # no other Hague entry for it to be confused with. Label only; the series is untouched.
    "Gravenhage, 's--Netherlands":    "Den Haag (The Hague)",
    "Fâqûs-Egypt":                    "Pi-Ramesses (Faqus)",   # Ramesside capital at Qantir/Avaris
    "Uqsur, Al--Egypt":               "Thebes (Luxor)",        # Al-Uqsur = Luxor = ancient Thebes
    "Badrashayn, Al--Egypt":          "Memphis",               # Al-Badrashayn sits on Memphis
    "Hillah-Iraq":                    "Babylon (Hillah)",      # Al-Hillah beside the Babylon ruins
    "Mosul-Iraq":                     "Nineveh (Mosul)",       # Nineveh across the Tigris from Mosul
    "Anyang-China":                   "Anyang (Yin)",          # Shang capital Yin / Yinxu
    "Drigh Road Cantonment-Pakistan": "Karachi",               # Karachi's number filed under a cantonment
    "Sõul-South Korea":               "Seoul",                 # McCune-Reischauer "Sŏul" mojibake
    "Nunjiang-China":                 "Nanjing",               # garbled name; coords are Nanjing (Ming capital)
    "Hospet-India":                   "Vijayanagara (Hospet)", # modern Hospet on the Vijayanagara/Hampi site

    # Hong Kong's territory series is filed under one of its districts (Beijiao = North Point).
    "Beijiao-Hong Kong":              "Hong Kong",
    # More ancient capitals filed under the modern town on the ruins (see note above).
    "Al Marsâ-Tunisia":               "Carthage (La Marsa)",   # La Marsa on the Carthage headland
    "Al-bu Kamal-Syria":              "Mari (Abu Kamal)",      # Mari, destroyed by Hammurabi c.1761 BC
    "Selçuk-Turkey":                  "Ephesus (Selçuk)",      # Selçuk = the Ephesus site
    "Bergama-Turkey":                 "Pergamon (Bergama)",
    "Sûr-Lebanon":                    "Tyre (Sur)",
    "Al-Quds-Palestine":              "Jerusalem",             # Al-Quds = Jerusalem
    "Al Khums-Libya":                 "Leptis Magna (Al Khums)",
    "Salihli-Turkey":                 "Sardis (Salihli)",
    "Kaspican-Bulgaria":              "Pliska (Kaspichan)",    # First Bulgarian Empire capital
    "Santa Maria Capua Vetere-Italy": "Capua (S.M. Capua Vetere)",
    # Crete: the Bronze Age figures belong to Knossos and Gortyn, not to the modern towns
    # beside them. A real "Iraklio" entry already exists with its own 700 AD -> present series.
    "Nea Alikarnassos-Greece":        "Knossos",               # Minoan palace, 5km from the suburb
    "Mires-Greece":                   "Gortyn",                # Roman capital of Crete, beside Mires

    # --- more of the same, each confirmed against its chandlerV2.csv row -------------------
    # The identification is the SOURCE's, not a guess: in every case the drawn series and the
    # Chandler row agree on both the benchmark year and the value, and the coordinates are
    # within a few km. Two of these are not merely unhelpful labels but false claims -- as
    # drawn the map asserts a 24,000-person Bronze Age Safed and a 27,000-person Iron Age
    # Tulkarm, neither of which was ever a town of any size.
    "Zefat-Israel":                   "Hazor",                 # Chandler Hazor (33.017,35.569)
                                                               # BC_1600 = BC_1360 = 24,000
    "Tûlkarm-Palestine":              "Samaria",               # Chandler Samaria (32.276,35.190)
                                                               # BC_800 = 27,000 -- Omri's capital
    "Basyûn-Egypt":                   "Sais",                  # Chandler Sais (30.965,30.769)
                                                               # BC_650 = 48,000; 26th-dyn capital
    "Bahtîm-Egypt":                   "Heliopolis (On)",       # Chandler Heliopolis (30.130,31.289)
                                                               # BC_1360 = 30,000; now a Cairo suburb
    "Mallawî-Egypt":                  "Amarna (Akhetaten)",    # Chandler Amarna (27.645,30.896)
                                                               # BC_1360 = 30,000, 11km from Mallawi
    "Amaliada-Greece":                "Elis",                  # Chandler Elis (37.892,21.373)
                                                               # BC_430 = 30,000
    "Marv Dasht-Iran":                "Persepolis (Marv Dasht)",   # Chandler "Perspolis" BC_430
                                                                   # = 50,000; burned 330 BC
    "Zhengzhou-China":                "Zhengzhou (Ao)",        # Chandler Ao (34.767,113.65)
                                                               # BC_1360 = 32,000; Shang capital
    "Zimbabwe-Zimbabwe":              "Great Zimbabwe",        # the site, and the source's own
                                                               # other_names value. Drawn 1300:
                                                               # 25,000 -> 1450: 40,000, so it is
                                                               # the largest city in southern
                                                               # Africa for those frames and the
                                                               # label reads as a country.
                                                               # (GAPS 3.6 says it "reaches
                                                               # nothing" -- that is stale; it is
                                                               # on the map, just unrecognisable.)
    "Xiaguan-China":                  "Dali (Xiaguan)",        # Chandler Dali/Tali (25.606,
                                                               # 100.268), 4km away -- Xiaguan is
                                                               # the modern urban district of Dali
                                                               # City. Nanzhao and Dali Kingdom
                                                               # capital; see its CF_KEEP entry.
    # New World. Same test, same result -- the drawn series matches the Chandler row exactly.
    "Tlalnepantla de Baz-Mexico":     "Tenayuca",              # Chandler Tenayuca (19.532,-99.168)
                                                               # 1200: 50,000, 1250: 54,000,
                                                               # 1565: 4,100 -- the Chichimec
                                                               # capital, inside modern Tlalnepantla
    "Santa Cruz Xoxocotlán-Mexico":   "Monte Albán",           # the municipality containing the
                                                               # site; 800: 30,000, dead by 875
    "Tapachula-Mexico":               "Izapa (Tapachula)",     # Chandler Izapa (14.923,-92.180)
                                                               # BC_200 = 35,000
    "Tula de Allende-Mexico":         "Tula (Tollan)",         # see CENSUS -- the entry now
                                                               # carries Chandler's Tollan row

    # --- swept systematically out of chandler_modelski_key ---------------------------------
    # The field is on 1,497 entries and was previously read only as prose evidence in these
    # comments. Used as a scanner -- CM name materially different from the DRAWN name (difflib
    # <=0.72, neither a substring, not in the entry's own other_names), and pre-1500 data
    # >=5,000 -- it reproduces the table above rather than contradicting it, which is what
    # makes the remainder worth acting on.
    #
    # Most of the 65 candidates are NOT defects and were left alone: the great majority are
    # EXONYMS, where the map's local-name style is already the right answer (Halab/Aleppo,
    # Dimashq/Damascus, Napoli/Naples, Wien/Vienna, Moskva/Moscow, Köln/Cologne, Kyjiv/Kiev,
    # Makkah/Mecca, Venezia/Venice, Firenze/Florence, and ~30 more). Renaming those would be a
    # policy change, not a repair. What is left is this table's actual subject: a modern town
    # standing in for the ancient city underneath it.
    "Shûsh-Iran":                     "Susa (Shush)",          # Chandler Susa: -800 = 27,000,
                                                               # -650 = 30,000, -430 = 70,000,
                                                               # all three matching to the digit,
                                                               # 1.0km. Held a top-12 slot for
                                                               # 1,200 sampled years labelled
                                                               # "Shûsh", a town of 63,000.
    "Mary-Turkmenistan":              "Merv (Mary)",           # Chandler Merv 1141 = 180,000 and
                                                               # 1150 = 200,000, both exact. The
                                                               # ruins are 29km east of modern
                                                               # Mary, which is the whole reason
                                                               # the coordinate looks off.
    # Kathmandu -- a three-part repair, and the CM key is what exposed it. See MERGE_INTO for
    # the other half and MANUAL in make_coordfix.py for the coordinate. This rename is the
    # small part: "Kâthmândau" is a romanization nobody searches for, and more practically it
    # fails names_agree() against WUP's own "Kathmandu" centre (norm() gives "kathmandau" vs
    # "kathmandu" -- neither a substring of the other), so the label is load-bearing.
    "Kâthmândau-Nepal":               "Kathmandu",
    "Shahat-Libya":                   "Cyrene (Shahat)",       # Chandler Cyrene -430 = 35,000,
                                                               # -200 = 30,000, both exact, 1.8km
    "Wâdî Moosa-Jordan":              "Petra (Wadi Musa)",     # Chandler Petra, 3.6km
    "Fiumicino-Italy":                "Ostia (Fiumicino)",     # Chandler Ostia, 4.6km
    "Yarîm-Yemen":                    "Zafar (Yarim)",         # Chandler Zafar, 9.9km
    # Lower confidence than the rest, and flagged rather than hidden: chandlerV2 has no
    # "Anshan-Iran" row, so this is the SOURCE's identification (its own chandler_modelski_key)
    # with the coordinate agreeing to 8.6km, but without the value corroboration every other
    # entry here has. Included because the alternative is worse: as drawn, a top-5 city of the
    # 3000 BC board is labelled "Aliabad", which is a generic Iranian village name.
    "Aliabad-Iran":                   "Anshan (Aliabad)",      # Tall-e Malyan, Elamite capital
    "Durg-India":                     "Durg-Bhilai",           # see MERGE_INTO: one conurbation,
                                                               # WUP's centre is named for the
                                                               # smaller half
}
PEAK_FLOOR = 2000                   # drop cities that never reach this (runtime cutoff is higher)

# --- one city split across two entries ------------------------------------------
# {donor key: target key}. The donor is not drawn; the years it covers OUTSIDE the target's own
# range are folded into the target, and nothing else. That "outside only" rule is the whole
# safety of this table -- the two entries are usually on different definitions where they
# overlap, so mixing them inside the covered span would manufacture exactly the unit flips the
# rest of the pipeline exists to remove. Extending the record at either end is safe because
# there is nothing there to contradict.
#
# dedup cannot do this job: it PICKS the richer entry and discards the other, so for Athens it
# would have kept Gazi's 2,400 Buringh points and thrown away Athinai's better modern detail,
# its spliced agglomeration variant and its WUP graft. Merging keeps both halves.
#
# Athens is the case that motivated it. "Gazi-Greece" is a neighbourhood 1.6km from the centre
# whose chandler_modelski_key is literally "Athens-Greece": it carries 155,000 at -430, the
# Periclean city, and is the ONLY ancient Athens in the source. The real "Athinai-Greece" entry
# starts at 1750. Dropping Gazi -- the obvious move, since it is drawn as a separate 3.1M dot on
# top of Athens -- would have deleted Athens from antiquity entirely. Same trap as Copenhagen.
MERGE_INTO = {
    "Gazi-Greece":       "Athínai-Greece",     # -430..1749 (Buringh) onto populstat's 1750..2001

    # --- the whole pre-modern record is in the parenthetical entry ---------------------------
    # Athens' variant of the same defect, found by sweeping every variant that starts >50y
    # before its base: of 144, only six are pre-1700 and these three are the unhandled ones.
    # In each case DROP_MARKERS/DUP_MARKERS was deleting the city's entire antiquity for a
    # NAMING reason, and the plain entry left behind starts in the 19th century. The
    # outside-only rule is exactly right here -- the two never overlap.
    #
    # prefer_agglomeration cannot do this job and the reason is structural, not tuning: it
    # returns early on `if not V or not S or not G`, so with no WUP centre there is no splice
    # at all, and its cost model prices only the modern switch step against the break the
    # splice opens. A variant whose entire worth is 2,300 years before the seam is invisible
    # to it. Sparta's modern self is 19,100, below WUP's 50k floor, so `G` is empty outright.
    "Sparti (agglomeration)-Greece":           "Spartí-Greece",
                             # -430: 40,000 .. AD 200: 30,000 (Chandler BC_430 = 40,000,
                             # BC_200 = 30,000) onto a plain entry that opens at 1861: 2,000.
                             # Classical Sparta was missing from the map entirely.
    "P'yõngyang (agglomeration)-North Korea":  "P'yõngyang-North Korea",
                             # see CLIP_BEFORE -- the donor is clipped to -194 first, because
                             # only the late half of Chandler's series is defensible.
    # Kathmandu, in two halves like Athens, and found by sweeping chandler_modelski_key.
    # "Sihara" is not a place -- it sits 0.15km from WUP's own Kathmandu centre, its CM key IS
    # "Kathmandu-Nepal", and its series is Chandler's Kathmandu row (630 = 22,000, matching to
    # the digit) running 630..1910. The modern half lives on "Kâthmândau-Nepal", 1911..2001,
    # which is the target. The two never overlap, so the outside-only rule is exactly right.
    #
    # Neither half worked alone, which is why this needed all three tables. Kâthmândau was
    # geocoded 90km NW into the Gandaki hills, so it never reached its own WUP centre and was
    # drawn frozen at 697,000; the centre was taken instead by "Pâtan" -- Lalitpur, the twin
    # city 4.6km south, whose own record peaks at 161,600 -- so the map drew Kathmandu as Patan
    # at 3.23M. And Sihara could not simply be promoted: at a peak of 40,000 against a 3.2M
    # centre it is below GRAFT_MIN_FRAC, and rightly so. Fixing the coordinate is what lets the
    # real record win the centre on its own merits (697,000 is 21.6% of it), and this merge is
    # what gives it back the 1,281 years of history that were filed under the wrong name.
    "Sihara-Nepal":                            "Kâthmândau-Nepal",

    # Durg and Bhilai are one urban area and WUP has one centre for it, named "Durg"
    # (1,393,235). Durg won it, so the map drew Durg at 1.39M -- the whole conurbation --
    # while Bhilai was held forward at its own 685,000 from 1991 right on top of it, i.e.
    # 2.08M drawn for 1.4M of people. The source itself says they are one thing: Durg's entry
    # is filed `is_agglomeration_of: bhilai`.
    # Durg is the target because it holds the WUP centre and the deeper record (1941 vs 1961),
    # and the merge folds in nothing at all -- Bhilai's 1961..1991 sits entirely inside Durg's
    # 1941..1991 -- which is exactly the point: the whole value here is removing the second dot.
    # RENAME then labels it "Durg-Bhilai", the conurbation's actual name, so the bigger and
    # better-known half is not silently erased from the map.
    "Bhilai-India":                            "Durg-India",

    "Qâhirah, Al- (agglomeration)-Egypt":      "Qâhirah, Al--Egypt",
                             # Cairo, and the largest single omission the variant sweep found:
                             # the plain entry starts at 1859, so the map had NO Cairo for the
                             # whole period it was among the largest cities on earth. The
                             # variant carries 700: 100,000 -> 1348: 500,000 -> 1350: 350,000
                             # (the Black Death, in the data) -> 1650: 175,000 -> 1798: 186,000.
                             # The junction is unusually safe for an agglomeration pair: at 1859
                             # the variant reads 295,000 against the base's 254,000, a 1.16x
                             # step, because Cairo had essentially no suburbs before the
                             # Ismailia expansion. Nothing in the folded span trips the
                             # carry-forward strip either -- its longest flat run is 40 years.
    "Benin City (agglomeration?)-Nigeria":     "Benin City-Nigeria",
                             # Chandler `Benin / Oedo`: 1600: 50,000, 1650: 60,000, 1668:
                             # 65,000, 1750: 50,000, 1850: 60,000. The walled Edo capital the
                             # Portuguese described, against a plain entry starting 1901.

    # --- US metro twins ---------------------------------------------------------------------
    # Every large US city is filed TWICE at identical coordinates: "<city>-<State>" and
    # "<city>-United States". They carry the same name, type and particulars, and agree exactly
    # at 2000 -- but they diverge through the 20th century, and the one holding a
    # `chandler_modelski_key` is the larger of the two in 49 of 51 pairs where only one has it
    # (Boston 2.73x, San Francisco 2.68x, Albany 2.17x). That is Chandler-Modelski's urban-
    # agglomeration figure fused in; the plain entry is populstat's city proper.
    #
    # Which one got drawn was decided by dedup ranking on (point count, peak), so the metro twin
    # usually won and the map showed US cities at metro size for the whole century. Scored
    # against the real city-proper counts at 1900/1950/1990 (US Census Bureau Working Paper 27),
    # the state entry wins every one of these by a wide margin -- Milwaukee 7.7% mean error vs
    # 38.5%, Pittsburgh 33.3% vs 144.1% -- so the state entry is the one to keep.
    #
    # MERGE_INTO rather than DROP_KEYS because the metro twin sometimes starts earlier, and the
    # outside-only rule folds exactly that in and nothing else: Philadelphia-United States runs
    # from 1700 against Pennsylvania's 1750, and Baltimore's from 1775 against 1790. Inside the
    # overlap the two are different definitions and must not be mixed.
    "Cleveland-United States":     "Cleveland-Ohio",
    "Detroit-United States":       "Detroit-Michigan",
    "Chicago-United States":       "Chicago-Illinois",
    "Philadelphia-United States":  "Philadelphia-Pennsylvania",
    "Saint Louis-United States":   "Saint Louis-Missouri",
    "Pittsburgh-United States":    "Pittsburgh-Pennsylvania",
    "Baltimore-United States":     "Baltimore-Maryland",
    "Milwaukee-United States":     "Milwaukee-Wisconsin",
    # San Bernardino is the same pair and was missed only because it is too small for
    # tools/us1950.csv, which covers the 100 largest places of 1950 and is where the rest of
    # these come from. Identical coordinates, and the metro twin had won dedup on peak: the map
    # held 524,663 -- San Bernardino COUNTY -- flat from 1974 to 1999 before dropping to 185,400,
    # on a city people know. The state entry peaks at 185,400, which is the city.
    "San Bernardino-United States": "San Bernardino-California",
}

# --- joining a historical city to its modern WUP centre -----------------------
# This join is the one place in the pipeline where WE invent a fact rather than repeat a
# source's, so it is deliberately dumb: a city is matched to its OWN urban centre, by name
# or by near-coincident location, and to nothing else.
#
# The previous rule scored every centre within 50km by peak_population/distance and took the
# best. That is a heuristic with no referent -- it silently paired cities that merely sit near
# each other, and 6.6% of its 6,318 matches were name-mismatched AND >5km apart: Jieyang took
# Chaozhou's 6.4M centre from 21km away, Obock took Djibouti's from 47km, Moyo took Nimule's
# from 38km across an international border. It also failed in the other direction, because
# scoring by size let a big neighbour outbid a city's own centre: Brazzaville's centre went to
# Kinshasa across the river, leaving a 3.7M agglomeration with no history attached to it.
#
# So: two rules, neither of which requires a judgement call.
#   NAME  -- the names agree and the centre is within GRAFT_NAME_KM. Distance is generous here
#            because a population-weighted centroid can sit well off the historic core
#            (Guangzhou's is 31.6km from the old town).
#   TIGHT -- no name agreement, but the centroids are within GRAFT_TIGHT_KM. Catches the cases
#            where the two sources romanize differently -- Stadester carries the older form and
#            WUP the current one: Bangalore/Bengaluru (6.0km), Taegu/Daegu (6.6km), Jiddah/
#            Jeddah (5.5km), T'aipei/Taipei (10.3km), Saint Louis/St. Louis (8.5km).
# Ties break on distance, and a name match always beats a tight match. Anything else gets no
# graft at all: the city simply ends when its own data ends, which is the honest outcome.
#
# TIGHT can afford to be this loose -- 15km puts plenty of suburbs inside their parent's centre
# -- because it never decides who gets the tail. The principal rule below hands that to the
# name match, so Iztapalapa and Nezahualcóyotl reach Mexico City's centre and still lose it to
# Mexico City. What broke the old rule was not its radius but its scoring by centre SIZE, which
# let any big neighbour outbid a city's own centre.
GRAFT_NAME_KM  = 40
GRAFT_TIGHT_KM = 15
TIGHT_MIN_FRAC = 0.2                # a TIGHT match must also be plausibly the same place by
                                    # size. WUP splits some agglomerations into a main centre
                                    # plus small satellite clusters named after a neighbourhood,
                                    # and at 15km a big city whose own centre is farther off (or
                                    # already taken) lands on the satellite instead: Kobe joined
                                    # "Hinomine 3-chome" (110k), Fort Worth "River Oaks" (62k),
                                    # Nashville "Cherokee Park" (55k), Perm a village called
                                    # Kondratovo. Requiring the centre to be >=20% of the city's
                                    # peak kills all 27 without a single false positive.
                                    # NOT applied to name matches: Detroit/Detroit (0.24),
                                    # Hartford/Hartford (0.19) and Albany/Albany (0.14) are the
                                    # RIGHT centres, and their ratios are low only because
                                    # populstat's US figures are metro-wide while a GHSL urban
                                    # centre is the dense core. That gap is the seam we already
                                    # know about, not a bad join.
GRAFT_MIN_FRAC = 0.05               # only graft if the principal is >=5% of the centre's size
                                    # (stops a tiny orphan town inheriting a whole megacity)
NAME_MIN_LEN = 5                    # below this, substring matching is meaningless -- "Ur"
                                    # and "Aba" are inside hundreds of unrelated names

# --- who wins a contested centre ---------------------------------------------
# When two entries match one centre the rank is (names_agree, peak) and the NAME is checked
# first, which is right for Dongguan/Shenzhen but wrong whenever WUP labels a centre after a
# district instead of after the city. Then a suburb wins on its name and the city it is a
# suburb OF gets no modern tail at all: Cleveland (2.13M, from 1820) lost to Parma OH (98k,
# from 1940) and ends in 2000, Venezia lost to Mestre (which is administratively part of
# Venice), and Antioch -- 2,000 years of history at the same coordinates -- lost to the
# "Hatay-Turkey" entry, so the map draws a city that dies in 1856 and a separate dot 0.9km
# away that begins in 1884. 24 centres are decided this way.
#
# There is no general rule here, and the evidence is what says so. A size gate ("the name only
# wins if it is within Nx of the biggest claimant") does not separate: at 3x it hands Jerusalem
# -- our own RENAME of Al-Quds, 2,800 years deep -- to the modern "Yerushalayim" entry, which
# is this very defect inverted. Adding "and starts earlier" fixes that but still fires on ten
# pairs of genuinely separate cities: Rawalpindi takes Islamabad's centre, Haifa takes Hadera's
# 40km away, Menton takes Monaco's, and Savannakhet takes Mukdahan's across the Mekong and an
# international border. Precision lands around 65%, which is not a rule, it is a coin toss with
# a good press release. The source's own is_agglomeration_of would be the principled signal and
# it is populated for only 4 of the 45 candidates -- it covers suburbs of famous cities, not
# the Parma/Bethany/Cannock tier -- so it cannot carry this either.
#
# So: a hand list, same as RENAME and CF_END. Every entry below is a centre whose WUP name is a
# district, suburb or later name of the winner. Named by source key; the winner takes the
# centre whatever it is called.
GRAFT_PRINCIPAL_WINS = {
    "Cleveland-Ohio", "Cleveland-United States",   # WUP splits Cleveland into Parma + Euclid
    "Antioch-Syria/Turkey",                        # vs "Hatay", the province name, same place
    "Venezia-Italy",                               # Mestre is part of the comune of Venice
    "Oklahoma City-Oklahoma", "Oklahoma City-United States",   # Bethany is an OKC enclave
    "Tegal-Indonesia",                             # Slawi -- source's own is_agglomeration_of
    "Garut-Indonesia",                             # Tarogong -- likewise
    "Alacant-Spain",                               # loses to "San Juan de Alicante" on substring
    "Wolverhampton-United Kingdom",                # vs Cannock, 15km and a fifth the size
    "Bytom-Poland",                                # Radzionkow was inside Bytom 1975-1997
    "Des Moines-Iowa", "Des Moines-United States", # Urbandale is a Des Moines suburb
    "Versailles-France",                           # Maurepas is a 1960s ville nouvelle nearby
    "Rostov-na-Donu-Russia",                       # the loser is a mis-geocoded Rostov Veliky
    "Mons-Belgium",                                # Saint-Ghislain is a small town 10km west
    "Bayamón-Puerto Rico",                         # Catano is a small San Juan suburb
    "Manukau-New Zealand",                         # vs Papakura, a seventh its size
    "Hefa-Israel",                                 # vs Qiryat Motzkin, a ~40k suburb that had
                                                   # taken the 484,770 centre WUP itself calls
                                                   # "Haifa" -- see MANUAL in make_coordfix.py
}

# --- entries where the modern layer is worse than the historical one ---------------------
# A graft is normally an improvement: WUP is a real measurement of the same place, later and on
# a stated definition. These are the cases where it is not, and where the honest outcome is the
# one we already have for an unmatched entry -- the city ends when its own data ends and is held
# forward, drawn dimmed.
#
# Three failure modes, one table. In the first two the centre is the right one and its
# NUMBER is not the city's; in the third the centre is the wrong place outright:
#
#   CONURBATION.  GHS merges the western Ruhr -- Essen, Duisburg, Oberhausen, Bochum,
#     Gelsenkirchen, Mulheim, Bottrop -- into ONE 2.72M urban centre (wup 675, centroid between
#     Essen and Oberhausen), while keeping Dortmund (748k) and Dusseldorf (907k) separate. Essen
#     wins that centre on its name, so the map drew a 2.72M "Essen" sitting on top of
#     Gelsenkirchen (279k), Duisburg (515k), Bochum (391k) and Oberhausen (222k), which are all
#     still drawn individually from their own populstat records. The conurbation was being
#     counted twice: once whole, once in parts.
#     It is self-reinforcing, which is why it needs a table rather than a threshold. populstat's
#     Essen entry carries BOTH definitions -- 3,609,289 (the Ruhrgebiet) held through the 1990s,
#     then 595,100 for 2000, which is Essen the city. §3.6 arbitrates that split against the WUP
#     handover, and because the handover is the 2.72M conurbation it kept the Ruhrgebiet plateau
#     and deleted the one correct figure in the entry. Denying the graft breaks the loop: with
#     no WUP to arbitrate, §3.6's local arbiter weighs the plateau against Essen's own earlier
#     record instead, drops it, and the entry ends on 595,100 -- Essen, held forward.
#
#   ABSENT.  WUP's Noril'sk centre (wup 20872, 1.0km away, unambiguously the right one) runs
#     144,964 in 1975 down to 56,272 in 2010 and then STOPS -- it falls under the 50k threshold
#     and drops out of the dataset. The real city was 174,673 at the 1989 census and 182,701 in
#     2021; it did not lose two thirds of its people. GHS's density segmentation does not cope
#     with an Arctic industrial city (the neighbouring centre 17km away shows the same decline),
#     so grafting it replaced populstat's 143,100 with a figure a third the size and then froze
#     that. populstat is not right either -- it misses the Talnakh/Kayerkan mergers -- but it is
#     wrong by 20%, not by 70%.
#
#   WRONG PLACE.  Soweto is the third entry and not the same kind of case as the two above: its
#     centre is not right-with-a-bad-number, it is a DIFFERENT TOWNSHIP. TIGHT matched it to
#     wup 3855 "Lenasia" 6.9km away, so the map handed Soweto's 596,600 (1991) to Lenasia's
#     57,272 (2000) -- a 10.4x drop at the handover -- and then drew Lenasia's growth to 214,770
#     under Soweto's name for the next quarter century. Soweto is ~1.3M; Lenasia is a separate
#     township with its own history.
#     Nothing was going to match, and that is the honest answer here rather than a failure: WUP
#     has no Soweto centre because Soweto is contiguous with Johannesburg and falls inside wup
#     3664. TIGHT_MIN_FRAC should have caught it and did not -- Lenasia reaches 36% of Soweto's
#     peak by 2025, over the 20% gate -- because the gate compares all-time peaks and the two
#     series barely overlap in time. The diagnostic quantity is the step AT the handover, and
#     for a TIGHT (name-mismatched) match a 10x drop there is not the seam, it is a different
#     place; see GAPS.md for that as a check-C candidate. Not turned into a rule here.
#     Denied, the entry ends on its own last real figure -- the 1991 census, 596,600 -- held
#     forward by §3.11. That is ~53% below Soweto's 2011 count of 1,271,628 and still by far the
#     best available: it is the right place measured on the right unit, which the graft was not.
#     Note the dot double-counts against Johannesburg's 10.7M FUA either way, as Tembisa, Evaton,
#     Katlehong and Diepmeadow already do. That is a separate question and a bigger one.
# Deliberately NOT a rule. "Deny when the centre is much bigger than the entry" is the seam
# itself (§3.7) and would fire on every US city; "deny when the centre shrinks" would fire on
# the whole post-Soviet and Rust Belt map, where the decline is real.
GRAFT_DENY = {
    "Essen-Germany",        # wup 675 is the whole western Ruhr, 2.72M
    "Noril'sk-Russia",      # wup 20872 ends 2010 at 56k; the city is ~175k
    "Soweto-South Africa",  # wup 3855 is Lenasia, a different township 6.9km away
}

# --- populstat carry-forward -------------------------------------------------
# populstat repeats a single pre-modern estimate VERBATIM for every year until the first
# modern census, then cliffs to it. That leaves dead cities frozen at their peak for
# centuries: Vijayanagara was sacked and abandoned in 1565 but sits at 480,000 until 1890;
# Kamakura holds 200,000 for the 690 years after the shogunate fell; Mari holds its Bronze
# Age figure for 3,780 years. On the map they occupy top-10 slots for centuries.
#
# Genuine data essentially never repeats a value to the byte across 150+ years, so an exact
# flat run of that length is the signature. We keep the run's FIRST point (the real estimate)
# and delete the rest, then drop a floor marker so the city fades out after its last real
# datum rather than plateauing. All 136 detected runs were reviewed individually; only the
# four below were judged genuine, so the rule is strip-by-default with an explicit keep-list.
CF_MIN_SPAN = 150                   # exact-flat run this long = carry-forward, not data
CF_MIN_VAL  = 20000                 # ignore villages; their flat runs don't distort anything
CF_EPS      = 1e-9                  # relative tolerance for "the same value". Stadester's
                                    # spline leaves float dust (200000 vs 200000.00000000006),
                                    # and an exact == splits one run into two, leaving a
                                    # surviving anchor at the wrong end of it.
# Keyed by (source key, the run's FIRST year) -- a city can have one genuine plateau and one
# bogus one, so a key-wide exemption is too blunt. Istanbul is exactly that case: its
# 944-1200 plateau is real, but exempting the whole entry also protected a 750,000 held flat
# from 1690 to 1790, which left Istanbul the largest city in the world for that century.
CF_KEEP = {
    ("Istanbul-Turkey", 944),  # Constantinople really did hold ~330k across the Macedonian
                               # and Komnenian periods; the drop is the 1204 Crusader sack
    ("Badrashayn, Al--Egypt", -1100),
                               # Memphis. Chandler-Modelski hold 100,000 from -1100 to -300,
                               # and it is a benchmark rather than annual data -- but it is
                               # their assertion that the city was still that size through the
                               # Late Period, which it was. Without this the strip collapses it
                               # to the -1100 anchor and, now that DROP_YEARS has removed the
                               # fill that used to follow, Memphis fades out from about -950 --
                               # eight centuries early, and through the era of its Saite
                               # revival and its Persian and Ptolemaic administration.
    # --- WITHDRAWN: ("Baghdád-Iraq", 932) --------------------------------------------------
    # It read: "the same case as Istanbul above, and found by validate check G. Chandler states
    # 1,100,000 at 932, 1000 AND 1100 -- three benchmarks", and kept the plateau on that basis.
    # Both halves of that are wrong, and the entry is withdrawn rather than deleted because the
    # mistake is an easy one to make again from check A's output alone.
    #   1. chandlerV2.csv's Baghdad row states 1,100,000 at 932 and then **125,000 at 1000 and
    #      150,000 at 1100**. There is one benchmark at that level, not three. provenance.py
    #      agrees and always did: it classifies 932 `chandler exact`, 1000 `fill` and 1100
    #      `populstat`, which is the signature of a hold, not of a repeated assertion.
    #   2. check G never reported this run. Its value test (GBM_VALUE_TOL = 2.0) exists for
    #      exactly this -- the Guangzhou guard -- and 1.1M against Chandler's 125,000 is 8.8x
    #      out, so 1000 and 1100 are not "lost benchmarks", they are stadester contradicting
    #      Chandler nine-fold. Check G's real Baghdad hit is a different run, 1250..1400, which
    #      eats Chandler's 1350 and 1400 behind a 1.1x cliff -- cosmetic, and now restored below.
    # What the exemption cost: it propped up stadester's carry-forward at the Abbasid peak and
    # then handed straight to Chandler's 1150: 10,000, so the map drew 1.1M held to 1100 and a
    # 110x collapse in the next 50 years, with the recovery to 100,000 by 1250 on top of it.
    # That is the largest unexplained cliff on the medieval map and it is entirely ours.
    # Chandler's own figures for the span are restored in CENSUS instead; see DROP_YEARS for
    # the one row of his that is not.
    ("Xiaguan-China", 860),    # Dali -- Nanzhao's late capital and then the Dali Kingdom's, until
                               # the Mongols took it in 1253. Chandler carries NINE benchmarks
                               # across the plateau (860: 100,000, 900 and 1000: 90,000, then
                               # 1100/1150/1200/1250: 100,000) before dropping to 25,000 at 1377,
                               # so the flat run is the compiler tracking a city that really did
                               # hold its size for four centuries. The strip kept 860 alone and
                               # drew a 517-year slide to 1377: Dali read 46,000 in 1150 and
                               # 35,000 in 1250, a third to two thirds low through the whole
                               # kingdom. Stadester flattens Chandler's 90,000 dip at 900-1000 to
                               # 100,000; keeping its version loses that detail and is still far
                               # closer than the slide.
                               # NOTE the entry is filed under Xiaguan, the modern urban district
                               # 4km from Chandler's Dali coordinate -- same city, so RENAME
                               # relabels it rather than SITE_COORDS moving it.
    ("Shîrâz-Iran", 1350),     # continuously one of Iran's largest cities; it submitted to
                               # Timur rather than being sacked, and declined only after 1722
    ("Umma-Sumer", -2500),     # 200yr plateau covers Umma's real peak under Lugalzagesi
    ("Bolgary-Russia", 1200),  # two defensible endpoints: pre-Mongol capital, then the
                               # rebuilt Golden Horde city; the drop is its 1430s destruction
}
# `end` is the last year of GENUINE data in a run, and the strip drops only `y > end`. It has
# two uses, in opposite directions, and both are legitimate because the field means the same
# thing either way:
#   BACKWARD (the original) -- the run's own starting value is already too late or too high, so
#     the city was finished before the run even began. Kaifeng, Shanghai, Corinth below.
#   FORWARD  -- the run FUSED real benchmarks with the fill that follows them, because the
#     compiler asserted one value at several successive benchmark years and stadester's hold
#     echoes it to within CF_EPS (~1e-16 of float dust). Then the measured half of the run is
#     data and only the tail is repetition. validate.py check G finds these; it reports 207,
#     and the ones worth acting on are those followed by a cliff, where a plateau the source
#     actually asserts gets redrawn as a centuries-long slide.
# Note either direction forces a fade (see below), so a city that did not die also wants a
# DISAPPEARED entry -- which is why the three forward entries here all have one.
CF_END = {
    "Gelibolu-Turkey":      999,    # Gallipoli was never a 300k city; the whole plateau is fake
    "Kaifeng-China":       -225,    # Qin drowned Daliang in 225 BC; Kaifeng's real peak is Song
    "Shanghai-China":      -301,    # a fishing village until the Ming; see the Shangqi note
    "Bûr Sa'îd-Egypt":     1149,    # Port Said was founded in 1859 for the Suez Canal
    "Aksum-Ethiopia":       700,    # a small ceremonial town once Red Sea trade was lost
    "T'bilisi-Georgia":    1226,    # sacked by Jalal al-Din the year before the run starts
    "Siracusa-Italy":      -212,    # Marcellus's sack ended Syracuse; the -200 value is post-collapse
    "Kórinthos-Greece":     400,    # Corinth was literally uninhabited at the 100 BC anchor
    "Yamaguchi-Japan":     1551,    # ended by Sue Harukata's coup, before the 1575 anchor
    "Sagaing-Myanmar":     1364,    # Ava was founded across the river in 1364
    "Agadez-Niger":        1550,    # already in decline after Askia Muhammad's 1500 sack
    "Huancavelica-Peru":   1786,    # the Santa Barbara mine collapsed in 1786
    "Krivodol-Bulgaria":   1396,    # 982 inhabitants in 1880; 40k is implausible at any date
    # --- forward: keep the measured half of a fused run (check G) --------------------------
    # Keyed by (source key, the run's FIRST year), so they scope to one run -- see the lookup
    # in strip_carry_forward for why a bare key is wrong in this direction.
    ("Gao-Mali", 1550):               1591,   # Chandler states 75,000 at 1550, 1575, 1585 and
                                              # 1591 -- the Songhai capital at its height, four
                                              # separate benchmarks -- and the 1600..1930 hold
                                              # sits 5e-16 away, so all four fused into one
                                              # 380-year run and three were deleted. Askia
                                              # Muhammad's Gao was drawn as a slide starting in
                                              # 1550 instead of a plateau collapsing after the
                                              # Moroccan conquest of 1591, which is the date.
    ("Tombouctou-Mali", 1500):        1600,   # same shape, one benchmark: Chandler has 25,000
                                              # at BOTH 1500 and 1600 and the hold to 1820
                                              # echoes them, so 1600 went with the repetition.
    ("Sparti (agglomeration)-Greece", -200): 200,
                                              # Chandler has 30,000 at 0, 100 and 200; the run
                                              # ran on to 1800 as fill, taking Roman Sparta's
                                              # two benchmarks with it. Keyed on the DONOR --
                                              # the strip runs before MERGE_INTO folds it in.
    ("Allada-Benin", 1682):           1700,   # Ardra, the kingdom the Slave Coast is named for.
                                              # Chandler has 40,000 at 1682 AND 1700 -- both
                                              # certainty 1 -- and the hold to 2001 echoes the
                                              # second, so 1700 fused into a 319-year run and
                                              # went with it. Check G reports it (`1 lost,
                                              # 1682..2001`) but ranks it 1.7x, far below the
                                              # printed head, which is why it sat unnoticed.
                                              # The 18 years matter more than they look: they
                                              # carry Allada to the eve of Agaja's conquest,
                                              # so the decline is drawn 1700 -> 1724 rather
                                              # than starting in 1682, when the kingdom was at
                                              # its height. See DISAPPEARED for the date.
}
# Whether a stripped run also gets a FADE (see plant_fades) -- i.e. whether deleting it means
# "nobody was recording this place" or merely "the source rounds to one number per era".
#
# The rule above is a good detector of repetition and a bad detector of ABANDONMENT, because
# populstat encodes its pre-modern estimates as a STEP FUNCTION: one benchmark repeated
# verbatim across an interval, then a new benchmark. For a stepped series EVERY interval trips
# the flat-run test, so the strip fired on all of them and plant_fades then blanked the city
# across each one. Roma is three steps (50,000 held 622-900, 35,000 held 970-1300, 17,000 at
# 1377) and was drawn as absent from 697 to 1302; Istanbul (400,000 held 622-900, then 330,000
# from 944) vanished through the entire Macedonian revival; Dimashq kept the 300,000 Umayyad
# benchmark at 705 and then died on the spot. 881 of the 1,150 planted holes were this, and the
# casualty list was Guangzhou, Chengdu, Xi'an under the Han, Cordoba at its peak, Milano, Kyjiv,
# Halab, Wuhan, Makkah and Mosul.
#
# What separates the two families is where the run ENDS, and the split is clean because the
# defect this exists for is literally "repeats one estimate until the FIRST MODERN CENSUS":
#   dead   -- the run ends AT that census and the next datum is a year or two later.
#             Vijayanagara 1550..1890 -> 1891, Kamakura 1250..1940 -> 1946, Nara 709..1870 ->
#             1877, Samarra 889..1840 -> 1843, Bergama 100..1900 -> 1901, Mari -1800..1980.
#   alive  -- the run ends centuries earlier and is followed by another PRE-MODERN estimate,
#             which is the compiler still tracking the city. Roma 900 -> 970, Istanbul 900 ->
#             944, Dimashq 1100 -> 1150, Guangzhou 1000 -> 1067, Kunming 1600 -> 1694.
# 1800 is where the two populations sit either side of a real valley: only 3 of 386 runs end in
# 1700-1749 at all, the alive family tops out at Changsha 1637 and Kaifeng 1750 (which todo.txt
# already lists as one that must NOT be blanked), and the dead family starts at Luoyang 1810.
# The gate is on the fade only -- the repeated points are still deleted either way, so a step
# is drawn as one interpolated segment between its benchmarks (quarter weight, half opacity,
# which is what those affordances are for) instead of as a hole in the city's existence.
CF_MODERN = 1800
# ...except where CF_END has already been used to declare the city finished by hand. That is a
# researched death, so it fades whatever the arithmetic says: it is what carries Sagaing (run
# ends 1750, Ava founded across the river in 1364) and T'bilisi (run ends 1770, sacked 1226).

# --- definition spikes (validate.py check F) ---------------------------------------
# One row measured on a different geographic unit, dropped into an otherwise consistent
# series. See despike() for why these three thresholds and why upward excursions only.
OSC_SPAN  = 20          # the excursion must leave and return within this many years. 40 was
                        # too loose in BOTH directions. A definition flip is a census-ROW
                        # artifact -- two adjacent rows of one table using different units --
                        # and rows are a few years apart: every unambiguous catch here is
                        # within 18y (Catania 1840/1853/1858, Qiqihar 1900/1911/1918, the
                        # Chinese prefecture rows 5-10y). Past ~20y the shape stops being
                        # diagnostic and starts being history: Moskva 75k(1553) -> 120k(1570)
                        # -> 60k(1575) is the real peak before Devlet Giray burned the city in
                        # 1571. Widening it for antiquity -- where anchors ARE centuries apart
                        # -- would be worse still, because over centuries a rise and fall is
                        # simply what cities DO. At a 200y window this rule would delete the
                        # Ramesside peak of Thebes, Carthage before the Punic Wars, Timur's
                        # Samarkand, Mansa Musa's Mali in 1324, Delhi's peak before Timur
                        # sacked it in 1398, and Naples either side of the Black Death. All ten
                        # wide-span candidates checked were genuine history, none were flips.
OSC_AGREE = 1.35        # ...with the values either side this close to each other
OSC_AMP   = 1.5         # ...and the odd row out beating both of them by at least this
OSC_AMP_HUGE = 4.0      # ...or beating both by THIS much, in which case the ends need not
                        # agree: no trend explains a 13x round trip, and the fast-growing
                        # Chinese county seats are 1.5-2x bigger after the spike than before
                        # (Guixian 62k -> 1,485,000 -> 115k) so a strict agreement test, which
                        # is really a "is this just a trend?" test, was letting them through.
OSC_SEAM_AMP = 6.0      # when the right-hand witness is a GRAFTED WUP point rather than
                        # populstat, demand this much before believing it. Below it the graft
                        # may simply be undercounting a real city -- see despike()'s note on
                        # Henderson NV -- and every one of those false positives was 1.5-2.8x.
                        # Above it that excuse dies: WUP would have to be wrong by six-fold,
                        # whereas a county-or-prefecture row being 6-15x its town is exactly
                        # what China's 1990/1994 censuses did (Xuanwei 70k, 1,174,700, 78k --
                        # and the spike is the LAST historical year, so its only witness on
                        # the right is WUP, which here is right and rises smoothly to 330k).
OSC_MIN   = 50000       # below this nobody can see the flicker
# Keyed by (source key, the spike's year), for excursions that really did happen this fast.
DESPIKE_KEEP = set()


def strip_carry_forward(S, key):
    """Collapse populstat carry-forward runs in a {year: pop} dict.

    Returns (S, fade_years) where fade_years are the years a floor point should be planted
    at, so the viewer shows the city fading out after its last real datum instead of holding
    a dead value. Every detected run is stripped; only the ones that ran up into the modern
    record earn a fade year (see CF_MODERN -- a step in a stepped estimate series is not an
    abandonment). Leaves S untouched for CF_KEEP entries."""
    if len(S) < 3:
        return S, []
    pts = sorted(S.items())
    fades, drop = [], set()
    i = 0
    while i < len(pts):
        j = i
        while j + 1 < len(pts) and abs(pts[j + 1][1] - pts[i][1]) <= CF_EPS * max(pts[i][1], 1.0):
            j += 1
        if (key, pts[i][0]) in CF_KEEP:
            i = j + 1
            continue
        if pts[j][0] - pts[i][0] >= CF_MIN_SPAN and pts[i][1] >= CF_MIN_VAL:
            # (key, run start) first, then bare key. A bare key applies to EVERY run in the
            # entry, which is what the backward entries want -- "finished by 1226" should also
            # flatten any later run. It is wrong for a forward one: CF_END["Gao-Mali"] = 1591
            # also un-stripped Gao's 800..1300 plateau, a 500-year hold that stadester invented
            # and that contradicts Chandler's own 1300 figure of 25,000. So a forward entry is
            # scoped to the single run it is about.
            here = CF_END.get((key, pts[i][0]))
            end = pts[i][0] if here is None else here
            if here is None:
                end = CF_END.get(key, end)            # last year of genuine data
            for y, _ in pts[i:j + 1]:
                if y > end:
                    drop.add(y)
            if pts[j][0] >= CF_MODERN or key in CF_END or (key, pts[i][0]) in CF_END:
                fades.append(end)
        i = j + 1
    if not drop:
        return S, []
    return {y: v for y, v in S.items() if y not in drop}, fades


def despike(control, key, seam=None):
    """Delete control points that jump UP off both neighbours and straight back down.

    This is validate.py's check F. populstat drops a row measured on a different geographic
    unit into an otherwise consistent series -- the 1990 Chinese census reported many county
    seats at prefecture level (Xuanwei 70k -> 1,174,700 -> 78k in six years), and European
    entries carry an occasional "Greater ..." row (Berlin 2.0M in 1905, 3.5M in 1914, 1.7M
    in 1916). A city cannot grow seventeen-fold and shrink back inside a decade, so the odd
    row out is a unit, not a population.

    Three conditions, and all three are load-bearing:
      span   the excursion leaves and returns within OSC_SPAN years -- this is the "how big
             is the time gap" test. Across centuries a rise and fall is ordinary history and
             none of this reasoning applies, so those are never touched.
      agree  the values either side agree with each other. If the series does NOT come back
             it is a trend with noise on it, not a definition flip: Aleppo goes 2.4M (2005)
             -> 3.8M (2010) -> 1.1M (2015) and that collapse is the war, so it must survive.
      amp    the row beats BOTH neighbours by a real margin, measured on the WEAKER leg.

    UP ONLY, which is the important asymmetry. A switch to a larger unit (prefecture,
    agglomeration, "Greater X") always makes the number bigger; catastrophes make it
    smaller. Downward excursions that recover are usually real -- Hiroshima 344k (1940) ->
    137k (1945) -> 286k (1950) passes every test above, and deleting it would erase the
    atomic bomb from the map. Worse, when a conurbation series is interrupted by a single
    city-proper census (Birmingham 1.5M, 922k in 1921, 1.7M) the dip is the one ACCURATE
    figure in the run. Both stay. Only the inflated rows go.

    ENTIRELY INSIDE THE HISTORICAL ERA -- at or before `seam`, the last year the historical
    source has, after which merge_series hands over to WUP. The whole rule rests on believing
    the two neighbours, and past the seam they are not populstat at all but a different
    source. When that graft is wrong the neighbours are wrong together, and the rule then
    deletes the one CORRECT row for agreeing with neither.
    That is what it did to the fast-growing suburbs: Henderson NV 61k (1989) -> 175,400 (2000)
    -> 61,863 (2015) got its 2000 row deleted, but Henderson really did hit 175,381 that year
    and 317k by 2020 -- the 2015 value is a bad graft (it sits inside Las Vegas's centre), and
    the deleted row was the true one. Cary, Roseville, Corona, Palmas, Namyangju and
    Brownsville were all the same story. A WUP value is not a witness against a census.

    Runs on the DP-simplified control points, never on the raw series: Stadester fills gaps
    with straight lines, so raw neighbours are interpolated FROM the spike (Berlin's raw 1910
    is 2,851,156, a fill between 1905 and the 1914 spike) and no amplitude test can see it.

    Returns (control, removed) where removed is a list of (year, value) for reporting."""
    if len(control) < 3 or "--no-despike" in sys.argv:      # --no-despike to A/B the effect
        return control, []
    pts = [list(p) for p in control]
    removed = []
    changed = True
    while changed and len(pts) >= 3:                 # to a fixpoint: killing a spike can
        changed = False                              # expose the next row of the same run
        for i in range(1, len(pts) - 1):
            (y0, v0), (y1, v1), (y2, v2) = pts[i - 1], pts[i], pts[i + 1]
            if (key, y1) in DESPIKE_KEEP:
                continue
            if y2 - y0 > OSC_SPAN or min(v0, v1, v2) <= 0:
                continue
            if max(v0, v1, v2) < OSC_MIN:
                continue
            amp = min(v1 / v0, v1 / v2)                    # the weaker of the two legs
            if amp < OSC_AMP:                              # too shallow, or a dip
                continue
            if max(v0, v2) / min(v0, v2) > OSC_AGREE and amp < OSC_AMP_HUGE:
                continue                                   # never came back, and not huge
                                                           # enough to rule out a trend
            if seam is not None and y2 > seam and amp < OSC_SEAM_AMP:
                continue                                   # WUP witness, not big enough
            removed.append((y1, v1, y0, v0, y2, v2))
            del pts[i]
            changed = True
            break
    return pts, removed


def norm(s):
    return "".join(ch for ch in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(ch)).lower().strip()


# --- display-name repairs -------------------------------------------------------------------
# Two source-wide spelling defects, fixed mechanically because they are defects of ENCODING and
# WORD ORDER, not of naming policy. The map's policy is the local name (see the note above the
# chandler_modelski_key block in RENAME) and neither of these changes it: "Ni`'znij Novgorod"
# becomes "Nizhnij Novgorod", not "Nizhny Novgorod", and "Qahirah, Al-" becomes "Al-Qahirah",
# not "Cairo". Turning an endonym into its English exonym stays a hand decision in RENAME.
#
# (1) The source writes a caron with a backtick, in two spellings for the same mark -- "`z" and
#     "`'z" both mean z-caron, as its own other_names confirm ("Vorone`'z" -> "Voronezh",
#     "`Cernivci" -> "Chernivtsi"). 481 entries carry it, and untouched they are unreadable.
#     Restoring the real letter is always right; whether to go further and romanise it depends
#     on the script the country actually uses, which is why that part is NOT done here. Cyrillic
#     countries are then romanised by hand where it matters; Latin-script ones must keep the
#     caron, because Kosice and Siauliai ARE spelled Kosice and Siauliai.
CARON = {"c": "č", "s": "š", "z": "ž", "C": "Č", "S": "Š", "Z": "Ž",
         "e": "ě", "r": "ř", "d": "ď", "t": "ť", "n": "ň"}
_CARON_RE = re.compile(r"`'?(.)")

# (2) Names inverted for alphabetisation, as a printed gazetteer does it: "Qahirah, Al-",
#     "Havre, Le -", "Sables-d'Olonne, Les -". 94 entries. Nobody writes a city's name that way,
#     and on a map there is nothing to alphabetise. The article rejoins the front -- hyphenated
#     for the Arabic ones, spaced for the Romance ones, and tight for "L'" and "'s", which carry
#     their own punctuation. Entries with a second comma ("Mahallah, Al-, al-Kubra") are left
#     alone: the tail is part of the name and re-assembling it is a guess.
_INV_AR = re.compile(r"^([^,]+), (A[dlnrstz]|Ash|El)-? ?-?$")
_INV_LA = re.compile(r"^([^,]+), (La|Le|Les|De|Den|Het)\s?-$")
_INV_TIGHT = re.compile(r"^([^,]+), (L'|'s)\s?-$")

def clean_name(s):
    s = _CARON_RE.sub(lambda m: CARON.get(m.group(1), m.group(1)), s)
    for rx, joiner in ((_INV_AR, "-"), (_INV_LA, " "), (_INV_TIGHT, "")):
        m = rx.match(s)
        if m:
            return m.group(2) + joiner + m.group(1)
    return s


def load_ghsl():
    """Grid-index GHSL urban centres by 1-deg cell for fast nearest/largest lookup.
    Returns (centres, grid) where centres[i] = (lat, lon, {year:pop}, name, is_fua).

    METROPOLITAN OVERRIDE. Where prep_fua.py found a functional urban area around a centre,
    that centre's ENTIRE population dict is replaced by the FUA series. Nothing downstream
    changes: match_centre still joins on WUP's name and coordinate, merge_series still hands
    over at the same seam, and a city therefore runs on one definition across the whole modern
    range. The two are never spliced together -- see prep_fua.py's header for the construction
    and §3.7 for why splicing a definition change into the middle of a series is the one thing
    this pipeline will not do.

    WHY IT IS NEEDED. WUP 2025 measures the Degree-of-Urbanisation urban centre, contiguous
    cells above 1,500/km2. American suburbia sits nearer 1,000/km2 and fails that test, so US
    metros shatter: WUP has 357 US centres against 3,143 counties, and it puts Chicago at
    3.68M, Dallas at 1.52M, Atlanta at 458k. Grafted onto populstat's metro-wide American
    figures that produced a systematically DOWNWARD seam -- median switch step 0.80x for US
    cities peaking >=100k, with a quarter of them losing more than half their population in a
    single year, against 1.08x and 8.5% worldwide. The fix is a metropolitan delineation, not
    a different population raster.

    ONLY THE PRINCIPAL of each FUA is overridden; the other members keep their own urban-centre
    series. Anything else double-counts -- New York's FUA holds 10 WUP centres, Dallas's 19 --
    and the map would draw the same 19.8M people ten times. So Dallas takes the whole
    Dallas-Fort Worth figure and Fort Worth keeps its own centre, which is the same choice any
    metro table makes. The demoted members are still drawn from their own populstat records, so
    the conurbation is counted once whole and once in parts; that is the Essen problem of
    GRAFT_DENY, tolerated here because it is already how every populstat suburb behaves."""
    with open(GHSL, encoding="utf-8") as f:
        g = json.load(f)
    fua = {}
    if os.path.exists(FUA) and "--no-fua" not in sys.argv:
        with open(FUA, encoding="utf-8") as f:
            fua = json.load(f)
    # AFRICAPOLIS IS OFF BY DEFAULT, and this is a judgement rather than a technical limit --
    # `--africapolis` turns it on and the file is built and matched either way.
    #
    # It is the purpose-built African urban record (national censuses plus imagery, 8,763 usable
    # agglomerations, 5.9 observed epochs each) and by the argument that put us_metro above eFUA
    # in the States -- a real national measurement beats a global model -- it should win here.
    # Two things say otherwise, both measured:
    #   1. It changes almost nothing where the map is actually read. Against the FUA layer:
    #      Lagos x0.97, Kinshasa x1.06, Abidjan x1.03, Kano x0.98, Kampala x1.02, Casablanca
    #      x0.95, Johannesburg x1.08. The big cities already agree.
    #   2. It shrinks everything else. Median across 1,638 matched centres is 1.50x, p90 3.73x,
    #      so mid-size African cities would be drawn ~a third smaller than their non-African
    #      peers -- reintroducing exactly the cross-region definition break §3.5c removed.
    # And its observed record stops at 2020 (2025+ are projections, which this pipeline does not
    # draw as data), so every African city would sit in a dimmed `hx` hold for the last five
    # years while the rest of the world runs to 2025 -- a continent dimmed at the present day.
    afp = {}
    if os.path.exists(AFRICAPOLIS) and "--africapolis" in sys.argv:
        with open(AFRICAPOLIS, encoding="utf-8") as f:
            afp = json.load(f)
    centres, grid = [], defaultdict(list)
    n_fua = n_afp = 0
    for code, c in g.items():
        co = c.get("coords")
        if not co or len(co) != 2:
            continue
        src = afp.get(code) or fua.get(code)
        raw = (src or c).get("population", {})
        pop = {int(y): v for y, v in raw.items() if v and v > 0}
        if not pop:
            continue
        lat, lon = co
        i = len(centres)
        centres.append((lat, lon, pop, c.get("name", ""), src is not None))
        if code in afp:
            n_afp += 1
        elif src is not None:
            n_fua += 1
        grid[(round(lat), round(lon))].append(i)
    print(f"ghsl centres indexed: {len(centres):,}")
    if fua:
        print(f"  FUA override: {n_fua:,} centres carry a metropolitan series "
              f"({len(fua) - n_fua:,} FUA rows had no matching centre)")
    elif "--no-fua" in sys.argv:
        print("  FUA override: disabled (--no-fua)")
    if afp:
        print(f"  Africapolis override: {n_afp:,} African centres (above FUA)")
    elif os.path.exists(AFRICAPOLIS):
        print("  Africapolis override: available but OFF -- pass --africapolis to enable")
    return centres, grid


def name_variants(name):
    """Normalized forms of a display name to test against a WUP centre name.

    Our own RENAME entries read "Babylon (Hillah)" / "Thebes (Luxor)" -- the historical name
    outside the parens, the modern town inside. WUP knows only the modern one, so both halves
    are offered separately or those cities would fail the name test against their own centre."""
    out = {norm(name)}
    if "(" in name and ")" in name:
        out.add(norm(name[:name.index("(")]))
        out.add(norm(name[name.index("(") + 1:name.rindex(")")]))
    return {v for v in out if v}


def names_agree(city_name, centre_name):
    """True if these two names refer to the same place, as far as strings can tell.

    Substring either way, because the two sources disagree about qualifiers ("New Delhi" /
    "Delhi", "Roma (Rome)" / "Roma"). Guarded by NAME_MIN_LEN: without it every three-letter
    name is inside something, and "Ur" would match "Urumqi"."""
    cn = norm(centre_name)
    if not cn:
        return False
    for v in name_variants(city_name):
        if v == cn:
            return True
        if min(len(v), len(cn)) >= NAME_MIN_LEN and (v in cn or cn in v):
            return True
    return False


def match_centre(name, lat, lon, peak, centres, grid):
    """Index of the WUP urban centre that IS this city, else None. See the GRAFT_* notes.

    Two accept rules -- names agree within GRAFT_NAME_KM, or centroids coincide within
    GRAFT_TIGHT_KM and the centre is at least TIGHT_MIN_FRAC of the city -- and no scoring
    beyond that. A name match outranks a tight match, and within a rule the nearer centre
    wins. Notably absent is any ranking by population: size is what let the old rule hand a
    city to whichever big neighbour happened to be in range. It appears only as the tight
    rule's floor, where it rejects rather than ranks."""
    coslat = math.cos(math.radians(lat))
    best = None                                   # (rule, distance, index); lower is better
    for dla in (-1, 0, 1):
        for dlo in (-1, 0, 1):
            for i in grid.get((round(lat) + dla, round(lon) + dlo), []):
                gla, glo = centres[i][0], centres[i][1]
                dx = (glo - lon) * coslat * 111.32
                dy = (gla - lat) * 110.57
                d = math.hypot(dx, dy)
                if d > GRAFT_NAME_KM:
                    continue
                if names_agree(name, centres[i][3]):
                    rule = 0
                elif d <= GRAFT_TIGHT_KM and \
                        max(centres[i][2].values()) >= TIGHT_MIN_FRAC * peak:
                    rule = 1
                else:
                    continue
                if best is None or (rule, d) < best[:2]:
                    best = (rule, d, i)
    return best[2] if best else None


BLEND_LO, BLEND_HI = 1975, 2000     # morph city-proper history -> GHSL agglomeration here

def interp_log(series_sorted, y):
    """log-linear interpolate a sorted [(year,pop)] list at year y; None if outside."""
    if y < series_sorted[0][0] or y > series_sorted[-1][0]:
        return None
    lo, hi = 0, len(series_sorted) - 1
    while hi - lo > 1:
        m = (lo + hi) // 2
        if series_sorted[m][0] <= y: lo = m
        else: hi = m
    (y0, v0), (y1, v1) = series_sorted[lo], series_sorted[hi]
    if y1 == y0:
        return v0
    f = (y - y0) / (y1 - y0)
    return 10 ** (math.log10(v0) + (math.log10(v1) - math.log10(v0)) * f)


def merge_series(S, G):
    """Run the historical series S to its own last year, then switch to the WUP series G.

    This used to blend the two geometrically across 1975-2000, on the theory that morphing
    one definition into the other hides the seam. It hides it by SPREADING it: the step
    becomes a 25-year trend, and because that trend runs against real growth for any city
    whose populstat figure was metro-wide and whose WUP urban centre is only the dense core,
    the map showed 358 cities DECLINING through the late 20th century when the source says
    they grew -- Houston x0.63, Brisbane x0.63, Adelaide x0.89, Montreal x0.77, plus most of
    the post-Soviet industrial belt. That is a false claim about history, and the trajectory
    graph drew it solid, because blended points are dense enough to count as real data.

    A hard switch cannot remove the step -- populstat's terminal figure and a GHSL urban
    centre are different measurements and one of them has to give -- but it puts the error
    in one frame at a stated year, where it reads as "the source changes here", instead of
    spreading it over the quarter-century viewers know best. Median step is 1.06x; the tail
    is the cities whose populstat series ends on an administrative figure (Manila's 1.65M is
    the city proper inside a 17.8M metro), which is what prefer_agglomeration below is for."""
    if not G:
        return S
    sw = max(S)                                  # last year the historical source really has
    out = {y: v for y, v in S.items() if y <= sw}
    nxt = min((y for y in G if y > sw), default=None)
    if nxt is not None and S[sw] > 0:
        SWITCH_STEPS.append((G[nxt] / S[sw], sw))
    for y, v in G.items():
        if y > sw:
            out[y] = v
    return out


EXTEND_FROM = 1990        # only hold forward a record that runs at least this late
YEAR_NOW    = 2025        # ...and hold it to here, the map's present

TRIM_FLAT_EPS    = 0.02   # values within 2% of each other count as the same held figure
TRIM_PLATEAU_MIN = 2      # a couple of points before the break is enough. It was 10, on the
                          # assumption that the earlier level always sits in a long flat hold
                          # the way New York's 16.15M does. Half the cases are not shaped like
                          # that: Frankfurt/Mannheim/Dusseldorf hold for only 6 years, and
                          # Nantes and Lyon hold for 2 at the end of an interpolated ramp. They
                          # never entered arbitration at all, which is why the French and German
                          # dips survived the first pass.
TRIM_TAIL_MAX    = 8      # ...and the run after it at most this long, or it is not a tail
TRIM_DROP        = 1.3    # the two levels must disagree by at least this much
TRIM_BREAK_MAX   = 3      # and the disagreement must appear within this many years. This is
                          # what makes the loose plateau rule safe: a 1.3x fall inside 3 years
                          # is not demography, it is a change of unit. Every case here is a
                          # ONE-year fall (Nantes 540k->270k over 1999->2000, Lens 499k->36k),
                          # and a real catastrophe that size does not land on the last row of
                          # every French and German city simultaneously. Genuine collapses are
                          # also mid-series, not terminal -- Hiroshima 1945 has 55 years of data
                          # after it, so it is never the terminal run this function looks at.
TRIM_WALK_MAX_YEARS = 20  # how far back the ramp-removal walk may reach, in YEARS. The fills
                          # it exists to remove are short -- Philadelphia's 1960-1974 is the
                          # longest at 14 -- while the 40-POINT limit it replaces was 170 years
                          # of decadal censuses for Cleveland. 20 clears Philadelphia and stops
                          # Cleveland at its 1906 anchor.
# --- local arbitration, for the entries WUP never matched ---------------------------------
# See the second half of trim_terminal_unit_switch's docstring. These only ever take effect
# when G is empty, and they can only ever DROP THE PLATEAU -- never the final row.
TRIM_LOCAL_LOOKBACK = 40   # years of the entry's own record, before the plateau, used as the
                           # reference level. Taken as the MINIMUM over that window rather than
                           # the nearest anchor, because populstat's straight-line fill ramps
                           # monotonically UP into the plateau, so every point in the fill sits
                           # between the real anchor and the plateau and any "nearest point"
                           # rule reads the fill instead of the data. The minimum cannot: the
                           # fill never goes below the anchor it started from. Klaten is the
                           # case -- 29 years of fill from 32,700 (1960) to 1,020,000 (1989),
                           # where a nearest-anchor walk bounded at TRIM_WALK_MAX_YEARS lands
                           # on 1969, a fill point at ~339,000, and the test then reads the
                           # plateau as continuous with the record.
TRIM_LOCAL_MARGIN   = 0.5  # log10. The plateau must be at least this much FURTHER from that
                           # reference level than the final row is -- ~3.2x -- before the
                           # plateau is called an excursion. Below it we do nothing.
TRIM_LOCAL_TAIL_FLOOR = 3  # ...and the final row may not be more than this far BELOW the
                           # reference level. Closer-than-the-plateau is not enough on its own:
                           # Sefton's record runs back to a 4,636 village figure, so its junk
                           # final row of 1,000 -- for a borough that had just reported 305,813
                           # -- still scored as the closer of the two, and dropping the plateau
                           # in its favour froze 1,000 people on screen until 2025. A final row
                           # below even the LOWEST figure of the preceding window is not a unit
                           # switch, it is a bad row, and neither level should be trusted. Kept
                           # loose (a third of the minimum) so it only catches that.
TRIM_LOCAL_WHOLE_MAX = 25  # years. Where the plateau is the entry's ENTIRE record there is no
                           # reference level at all, and dropping it deletes the city from the
                           # map for the whole span it covered -- so it may only be done to a
                           # short stub. The cases worth having are ~11 years (Cibinong and
                           # Cikampek hold 1990-2001, Japeri 1991-1999), where the cost is a
                           # decade of a figure that was never a measurement. Islington,
                           # Tower Hamlets and Brent are the reason for the bound: each is a
                           # single borough figure carried from 1901 to 2001, and deleting it
                           # erases their entire 20th century rather than fixing anything.
TRIMS = []                # (city, kept level, dropped level, years dropped) for the report
LOCAL_TRIMS = []          # the subset arbitrated locally, reported separately

def trim_terminal_unit_switch(S, G, name=""):
    """Populstat's late series often carries TWO definitions: a metro figure held flat for
    decades, then a final census at the administrative city. New York holds 16,150,000 from
    1974 to 1999 and then reports 8,008,300 for 2000; Boston holds 3.46M and then 589,000;
    Detroit 4.43M then 951,000. The low values are exactly the 2000 US Census city figures,
    the high ones are the metro area, and both are 'real' -- they just measure different
    things, so whichever one we hand to the WUP switch, the other becomes a cliff. Left alone
    it draws a V: the city collapses for one frame and then leaps back at the graft.

    We cannot know which level is meant, but we do not have to guess: WUP is a third
    measurement of the same city, so let it arbitrate. Whichever of the two levels its
    handover value is closer to (in log space) is the one on the same footing as the modern
    layer; the other is deleted. That is 'use populstat as late as it is CONSISTENT' -- the
    inconsistency is between the two ends, and the tiebreak is evidence rather than a rule
    of thumb. It cuts both ways in practice: New York keeps its 16.15M metro plateau and
    drops the census point, Detroit keeps the 951,000 census point and drops the plateau.

    WHEN WUP NEVER MATCHED THE ENTRY there is no third measurement to arbitrate with, and this
    used to return immediately -- which left the largest jumps on the whole map untouched, and
    made them worse than the grafted ones, because the contradicted final row is also the value
    §3.11 then freezes flat to 2025. Sidoarjo runs 38,700 (1963) -> 53,500 (1979) -> 1,070,000
    held 1989-2001 -> 76,900, and that 76,900 is what the map shows for the next 23 years.

    The fallback arbiter is the entry's OWN earlier record: 53,500 growing to 76,900 over 23
    years is a city, 53,500 to 1,070,000 is a kabupaten. Where the plateau is that far out of
    line with what the entry says about itself, it is an excursion and goes.

    It is deliberately ONE-DIRECTIONAL -- it can drop the plateau, never the final row -- and
    that asymmetry is the whole safety argument. Funza has the same terminal shape but is
    Bogota's series top to bottom (19,500 in 1776 rising smoothly to 4,530,000) with one final
    row at 49,900 that is the only genuinely-Funza figure in it. Its plateau is CONTINUOUS with
    its record, so the margin test refuses it and nothing happens. Were the rule symmetric it
    would instead delete that 49,900 and hold a 4.5M phantom Bogota-under-another-name to 2025,
    which is far worse than the cliff. An entry whose every row is the parent agglomeration is
    a bad join (§6.5), not a unit switch, and it needs the hand table, not a guess."""
    if len(S) < 4:
        return S
    pts = sorted(S.items())
    def run_back(end_i):
        """extent of the flat run ending at index end_i -> (start_i, value)"""
        v = pts[end_i][1]
        i = end_i
        while i > 0 and abs(pts[i - 1][1] - v) <= TRIM_FLAT_EPS * max(v, 1.0):
            i -= 1
        return i, v
    tail_i, tail_v = run_back(len(pts) - 1)
    if tail_i == 0:
        return S
    plat_i, plat_v = run_back(tail_i - 1)
    tail_span = pts[-1][0] - pts[tail_i][0]
    plat_span = pts[tail_i - 1][0] - pts[plat_i][0]
    if tail_span > TRIM_TAIL_MAX or plat_span < TRIM_PLATEAU_MIN:
        return S
    if pts[tail_i][0] - pts[tail_i - 1][0] > TRIM_BREAK_MAX:
        return S                               # gradual -- that is a decline, not a unit change
    if plat_v < TRIM_DROP * tail_v:            # the two levels agree well enough -- no break
        return S
    if tail_v <= 0 or plat_v <= 0:
        return S
    local = False
    if G:
        sw = pts[-1][0]
        nxt = min((y for y in G if y > sw), default=None) or max(G)
        g = G[nxt]
        if g <= 0:
            return S
        keep_plateau = abs(math.log10(g / plat_v)) < abs(math.log10(g / tail_v))
    else:
        # No WUP row for this entry, so arbitrate against the entry's own earlier record.
        y_plat = pts[plat_i][0]
        window = [v for y, v in pts[:plat_i]
                  if y_plat - TRIM_LOCAL_LOOKBACK <= y < y_plat and v > 0]
        if not window:
            # No reference level. Two very different reasons for that, and only one is safe.
            if plat_i > 0:
                return S      # there IS earlier record, just none recent enough to compare
            # plat_i == 0: the plateau is the entry's ENTIRE record -- one figure repeated,
            # then a final row contradicting it. Nothing supports the plateau except its own
            # repetition, and a carry-forward is not a measurement (§1), so there is no level
            # to weigh the final row against and nothing worth keeping. Cibinong, Cikampek,
            # Ciparay, Majalaya and Japeri are all this shape: ~1.8M held 1990-2001, then
            # 142,000. Majalaya and Ciparay give the game away by carrying the SAME 1,909,500
            # four km apart -- one Kabupaten Bandung figure stamped on both entries.
            if plat_span > TRIM_LOCAL_WHOLE_MAX:
                return S
            keep_plateau, local = False, True
            base = None
        else:
            base = min(window)
        if base is not None:
            if abs(math.log10(plat_v / base)) - abs(math.log10(tail_v / base)) < TRIM_LOCAL_MARGIN:
                return S                       # not clearly an excursion -- leave it alone
            if tail_v * TRIM_LOCAL_TAIL_FLOOR < base:
                return S                       # the final row is junk too; trust neither level
            keep_plateau, local = False, True
    if keep_plateau:
        dropped = [y for y, _ in pts[tail_i:]]
        TRIMS.append((name, plat_v, tail_v, len(dropped)))
    else:
        # Deleting the plateau is not enough: Stadester fills the gap before it with a
        # straight line, so the RAMP up to the plateau survives and keeps the trough.
        # Philadelphia's real anchors are 2.27M (held to 1959) and 4.85M (1974); the 15
        # points between are interpolation toward a figure we just decided was the wrong
        # unit, so they go too. Walk back through that fill and stop at the previous real
        # anchor, which must survive: Detroit's 1.87M held 1940-1959 is Detroit city proper
        # and correct.
        #
        # "Stop at the first flat pair" was the whole test, and it assumes the fill is always
        # bracketed by a flat hold. Where the pre-plateau data is a genuine census SERIES
        # there is no flat pair to find and the walk ran to its 40-POINT limit, which in a
        # decadal series is a century and a half: Cleveland holds 922,900 from 1925 to 1999
        # (74 years -- a carry-forward, but too short for CF_MIN_SPAN) and reports 478,400 in
        # 2000, so the plateau is rightly dropped, and the walk then took every census from
        # 1830 to 1920 with it, leaving a city that went 600 in 1820 straight to 478,000.
        # TRIM_WALK_MAX_YEARS bounds it in years instead, which is the unit the fill is
        # actually measured in. A slope test was tried first and is wrong: populstat's fill
        # is piecewise-linear between several anchors of the SAME rejected definition, so it
        # stopped Philadelphia at 1969 (85,387/yr up to there, 346,351/yr after) and left the
        # trough this function exists to remove.
        start = plat_i
        while start > 1 and (plat_i - start) < 40 \
                and pts[plat_i][0] - pts[start - 1][0] <= TRIM_WALK_MAX_YEARS \
                and abs(pts[start - 1][1] - pts[start - 2][1]) > TRIM_FLAT_EPS * max(pts[start - 1][1], 1.0):
            start -= 1
        dropped = [y for y, _ in pts[start:tail_i]]
        (LOCAL_TRIMS if local else TRIMS).append((name, tail_v, plat_v, len(dropped)))
    if len(dropped) >= len(S):
        return S
    return {y: v for y, v in S.items() if y not in set(dropped)}


# --- China: populstat's terminal figures are administrative areas --------------------
# China's late populstat rows report the COUNTY, prefecture-level city or municipality rather
# than the settlement. Chongqing is the clearest: 4,644,814 held 1984-1994 (the real urban core)
# then 21,834,938 in 1996 and 30,979,100 in 2001 -- the municipality, a province-sized unit
# created in 1997. On the map it is the largest city in the world for those years.
#
# It is not a handful of entries, and it is specific to China. Comparing populstat's last figure
# with WUP's first, on the finished build (so despike and the terminal trim have already run):
#
#                      n     >=2x    >=3x    >=5x
#     China          375    41.6%   25.9%    9.3%
#     everywhere    5,528    4.2%    1.3%    0.3%
#
# Ten times the rate at 2x, twenty at 3x, thirty at 5x; 156 Chinese cities disagree by >=2x and
# 63 of those carry a terminal figure over a million. The census years cluster on 1990/1994/
# 1995/2000, which is the county-and-prefecture reporting signature.
#
# WHY THIS IS SCOPED TO CHINA AND NOT A GENERAL RULE. The worst non-Chinese cases on the same
# measure are Jacksonville, Nashville, Charlotte, Fort Worth and Indianapolis -- and there the
# disagreement has the OPPOSITE cause: WUP's density-defined core undercounts a sprawling city,
# while populstat is right. Identical numbers, inverted meaning. A global "trust WUP when they
# disagree by 3x" would fix China and wreck the American South. China is the one place we have
# evidence that populstat is the unreliable side, so it is the only place the rule fires.
CHINA_ADMIN_RATIO = 2.5     # terminal run this many times the WUP handover = an administrative
                            # area, not a city. Chongqing is 5.9x, Yushu 21.6x, Zaozhuang 11.0x;
                            # the Chinese cities where the two sources actually agree sit under
                            # 1.5x, so the gap either side of 2.5 is wide.
CHINA_ADMIN_ACCEPT = 1.5    # ...and once triggered, keep walking back until the series is THIS
                            # close to WUP again. Without the second threshold the walk stops at
                            # the first point under the trigger and leaves the tail sitting just
                            # below it, still on the wrong unit: Harbin dropped 9,411,000 and
                            # landed on 8,241,248, which is still the prefecture (WUP says
                            # 3,570,000). populstat ramps INTO the administrative figure, so the
                            # intermediate fill points are administrative too.
CHINA_ADMIN_MIN_KEEP = 2    # never leave a city with fewer control points than this
ADMIN_TRIMS = []            # (city, dropped level, kept level, n points) -- `--admin` prints all

def trim_admin_tail(S, G, name="", country=""):
    """Drop a terminal run that WUP says is an administrative area rather than a city.

    trim_terminal_unit_switch handles the case where populstat holds a METRO figure and then
    steps DOWN to the administrative city. This is the mirror image -- populstat steps UP into a
    county/prefecture/municipality figure and stays there -- so that function never sees it (its
    `plat_v < TRIM_DROP * tail_v` guard returns early when the plateau is the lower level)."""
    if country != "China" or not G or len(S) < CHINA_ADMIN_MIN_KEEP + 1:
        return S
    pts = sorted(S.items())
    nxt = min((y for y in G if y > pts[-1][0]), default=None)
    if nxt is None or G[nxt] <= 0:
        return S
    if pts[-1][1] < CHINA_ADMIN_RATIO * G[nxt]:        # tail agrees well enough -- leave it
        return S
    lim = CHINA_ADMIN_ACCEPT * G[nxt]
    i = len(pts)
    while i > 0 and pts[i - 1][1] >= lim:
        i -= 1
    if i == len(pts) or i < CHINA_ADMIN_MIN_KEEP:      # nothing to do, or it would gut the city
        return S
    ADMIN_TRIMS.append((name, pts[-1][1], pts[i - 1][1], len(pts) - i))
    return {y: v for y, v in S.items() if y <= pts[i - 1][0]}


SWITCH_STEPS = []       # (WUP/historical ratio, switch year) -- reported at the end of a build

def report_switch_steps():
    """Print the size of the seam. This is the pipeline's headline honesty number: the one
    frame where a city stops being a populstat administrative figure and starts being a WUP
    urban centre. It cannot go to 1.0 -- they are different measurements -- but it should not
    get worse, and the tail is the worklist."""
    if not SWITCH_STEPS:
        return
    rr = sorted(r for r, _ in SWITCH_STEPS)
    n = len(rr)
    q = lambda p: rr[int(p / 100 * (n - 1))]
    within = sum(1 for r in rr if 0.8 <= r <= 1.25)
    beyond = sum(1 for r in rr if r > 2 or r < 0.5)
    print(f"source-switch step (WUP / historical) over {n:,} grafted cities:")
    print(f"  p10 {q(10):.2f}x  p25 {q(25):.2f}x  median {q(50):.2f}x  p75 {q(75):.2f}x  p90 {q(90):.2f}x")
    print(f"  within 0.8-1.25x: {within:,} ({100*within/n:.0f}%)   beyond 2x either way: "
          f"{beyond:,} ({100*beyond/n:.0f}%)")


def index_variants(raw):
    """(bare name, country) -> key of the agglomeration variant entry, where one exists.

    populstat files a metro figure as a separate parenthetical entry -- "Manila
    (agglomeration)" beside "Manila" -- and build.py has always dropped those, keeping
    city-proper history only. That was the right call while the modern tail was blended in,
    and the wrong one now: the series has to END on something comparable to a WUP urban
    centre, and the variant is exactly that measurement. 552 of them have a matching base
    entry and overlapping years."""
    out = {}
    for k, c in raw.items():
        nm = c.get("name", "")
        if not any(m in nm.lower() for m in DROP_MARKERS):
            continue
        if k in MERGE_INTO:
            continue        # handled by the merge, which folds only the years the target
                            # lacks; letting the splice also see it would overwrite the base
        bare = norm(nm[:nm.index("(")]) if "(" in nm else norm(nm)
        idk = (bare, c.get("country", ""))
        pop = c.get("population") or {}
        if idk not in out or len(pop) > out[idk][1]:
            out[idk] = (k, len(pop))
    return {k: v[0] for k, v in out.items()}


VARIANT_TAIL_SLACK = 15     # the variant must run to within this many years of the base's end
VARIANT_MAX_GAP = 30        # the variant may not leave a hole longer than this...
VARIANT_GAP_ANCHORS = 4     # ...anywhere the base has this many real anchors of its own.
                            # 30y is where the data splits cleanly: every legitimate splice
                            # sits under it (Manila's worst hole is 28y) and the losses run
                            # 50-140y -- Lisboa 139, Berlin 89, Yerushalayim 78, Glasgow 69,
                            # Jekaterinburg 72, Charkiv 69, Odesa 54 -- each with 9-15 real
                            # base censuses inside the hole.
VARIANT_MIN_ANCHORS = 5     # ...and carry at least this many real anchors of its own over the
                            # years it would overwrite. Manila has 7 and splices; Manchester,
                            # Liverpool and Newcastle have 3 and do not.
                            #
                            # This started life as a RATIO against the base's anchor count and
                            # that was subtly wrong, because DP counts shape complexity rather
                            # than data density: a clean census run is nearly collinear and so
                            # scores LOW. Cleaning Newcastle's conurbation rows out of the base
                            # left 266,600 / 275,000 / 283,200 for 1911/1921/1931, DP folded
                            # those to two points, the base looked sparse, and the splice fired
                            # and re-hollowed the city the fix had just repaired. An absolute
                            # floor on the variant cannot be moved by improving the base.


def _real_anchors(S, lo, hi):
    """How many points in [lo,hi] are actual data rather than Stadester's straight-line fill.

    DP at DP_EPS_REL is exactly the question "which of these points is not on the line between
    its neighbours", which is what the rest of the pipeline already uses to recover anchors."""
    pts = sorted((y, v) for y, v in S.items() if lo <= y <= hi and v > 0)
    return len(dp_simplify(pts, DP_EPS_REL)) if len(pts) > 2 else len(pts)

def _step(G, S):
    """|log10| of the WUP-switch ratio for a historical series S, or 0 if there is no switch."""
    if not S or not G:
        return 0.0
    sw = max(S)
    nxt = min((y for y in G if y > sw), default=None)
    if nxt is None or S[sw] <= 0 or G[nxt] <= 0:
        return 0.0
    return abs(math.log10(G[nxt] / S[sw]))


def prefer_agglomeration(S, V, G):
    """Splice an agglomeration variant V over the base city-proper series S -- but only when
    that lowers the TOTAL definitional discontinuity of the result.

    The appeal is obvious: populstat files a metro figure separately, and the series has to
    end on something a WUP urban centre can land on. Manila's base ends at 1,654,000, the
    city proper inside a 17.8M metro, so the switch is a 10.8x cliff; its variant ends at
    9,650,000 and the cliff becomes 1.8x.

    The catch is that splicing is not free -- it creates a NEW break where the variant starts,
    and across all 552 pairs the median ratio there is 1.37x. Applying it to every pair traded
    552 fresh mid-series breaks for a median switch-step of 1.12x -> 1.09x and 22 fewer cities
    stepping past 2x. That is a bad bargain, so we price both sides and take the variant only
    when it wins: the switch step it saves must exceed the break it introduces.

    Everything is measured in |log10| so the two are commensurable and direction-free."""
    if not V or not S or not G:
        return S, False
    if max(V) < max(S) - VARIANT_TAIL_SLACK:
        return S, False                     # variant stops early -> cannot improve the ending
    v0 = min(V)
    spliced = {y: v for y, v in S.items() if y < v0}
    spliced.update(V)
    if not spliced:
        return S, False
    # The splice replaces the base OUTRIGHT from v0 on, so it can trade a dense census record
    # for a nearly-empty variant -- and because Stadester fills gaps with straight lines, an
    # empty variant does not look empty, it looks smooth. Manchester ran 1901:765,000 ->
    # 1970:2,840,000 as ONE LINE, its real 1911/1921/1931/1951/1961 censuses gone. Counting
    # points that survive DP (i.e. that are not just fill) over the overwritten years:
    # Manchester 13 base vs 3 variant, Liverpool 15 vs 3, Newcastle 12 vs 3 -- against Manila
    # 11 vs 7, which is the case this function was written for and which is a fair trade.
    # The cost model above prices the step it saves and the break it opens, but detail
    # destroyed in between costs it nothing, so those three looked like clean wins.
    if _real_anchors(V, v0, max(S)) < VARIANT_MIN_ANCHORS:
        return S, False
    # ...and those anchors must be DISTRIBUTED, not clustered at the ends. Counting them was
    # not enough: Lisbon's variant has 7 and sails past the floor above, but 139 of the years
    # it overwrites (1820-1959) contain none of them, while the base entry holds 15 real
    # censuses across exactly that span (1911: 435,400 = INE's 435,359). The splice was
    # trading a century and a half of Portuguese census data for a straight line.
    # The test is deliberately conditional on the BASE: a hole is only a loss if there was
    # something in it. Where neither series has data the splice is harmless and still runs.
    vys = [y for y, _ in dp_simplify(sorted((y, v) for y, v in V.items()
                                            if v0 <= y <= max(S)), DP_EPS_REL)]
    for a, z in zip(vys, vys[1:]):
        if z - a > VARIANT_MAX_GAP and _real_anchors(S, a, z) >= VARIANT_GAP_ANCHORS:
            return S, False
    # the break the splice introduces, at the variant's first year
    before = [y for y in S if y < v0]
    if before and S[max(before)] > 0 and V[v0] > 0:
        introduced = abs(math.log10(V[v0] / S[max(before)]))
    else:
        introduced = 0.0                    # nothing before it -> the variant IS the series
    if _step(G, spliced) + introduced < _step(G, S):
        return spliced, True
    return S, False


def dp_simplify(pts, releps):
    """Douglas-Peucker on a (year, population) series in LINEAR space, with deviation
    measured RELATIVE to the local magnitude.

    Why linear+relative (not log): Stadester fills gaps with straight-line (linear)
    interpolation between its real anchors -- confirmed by e.g. Yuzhou, whose absolute
    increment is a dead-constant +438/yr across 2,300 years. In linear space that fill
    is perfectly collinear, so this collapses it back to the two real anchors; the
    viewer's log-linear interpolation then reconnects them GEOMETRICALLY (a plausible
    growth curve instead of the smear). Genuine data (incl. annual GHSL modern series)
    deviates from the straight chord by more than releps, so it survives -- which also
    stops the modern era from being over-simplified into fake 20-year gaps."""
    if len(pts) <= 2:
        return pts[:]
    x0, y0 = pts[0]
    x1, y1 = pts[-1]
    # find point of max deviation from the chord y0..y1 evaluated at its x, relative to size
    dmax, idx = 0.0, 0
    dx = x1 - x0
    for i in range(1, len(pts) - 1):
        x, y = pts[i]
        ychord = y0 + (y1 - y0) * ((x - x0) / dx) if dx else y0
        d = abs(y - ychord) / max(abs(ychord), abs(y), 1.0)
        if d > dmax:
            dmax, idx = d, i
    if dmax > releps:
        left = dp_simplify(pts[:idx + 1], releps)
        right = dp_simplify(pts[idx:], releps)
        return left[:-1] + right
    return [pts[0], pts[-1]]


# --- adjusted time ------------------------------------------------------------------
# A gap of 700 years means completely different things in 2500 BC and in 1750: the first is
# ordinary sparsity for the period, the second is a city that genuinely went unrecorded through
# an era that otherwise has records every generation. So gaps are measured along the viewer's
# WARPED timeline instead of in raw years -- the same warp the time slider uses, where recent
# centuries get much more of the track than ancient millennia.
#
# One adjusted year == one real year in the oldest stretch (before AD 1), which is what makes
# the thresholds below readable: FADE_GAP 700 is literally 700 years in antiquity, and shrinks
# to ~350 across AD 1-1400, ~110 across 1400-1900 and ~28 after 1900 -- eras whose real gaps
# would be the same number are not comparable, adjusted ones are.
#
# MUST MATCH segY()/segS() in index.html. Outside the end edges this extrapolates the nearest
# segment's rate, so a city starting at -3700 still measures sensibly.
ADJ_EDGES = [-3500, 1, 1400, 1900, 2025]   # real-year segment boundaries

def _adj_frac():
    """Slider fraction at each edge. 1400 stays at 50% and 1900 at 75%; the first half is
    split so the pre-AD-1 stretch runs twice as many years per unit of track as AD 1-1400."""
    a = (ADJ_EDGES[1] - ADJ_EDGES[0]) / 2.0    # pre-AD 1, counted at half rate = 2x compressed
    b = ADJ_EDGES[2] - ADJ_EDGES[1]
    return [0.0, 0.5 * a / (a + b), 0.5, 0.75, 1.0]

ADJ_FRAC = _adj_frac()
# scale the whole track so the first segment runs at exactly 1 adjusted year per real year
ADJ_SCALE = (ADJ_EDGES[1] - ADJ_EDGES[0]) / ADJ_FRAC[1]

def adj(y):
    """real year -> adjusted year"""
    for i in range(4):
        if y <= ADJ_EDGES[i+1] or i == 3:
            f = (y - ADJ_EDGES[i]) / (ADJ_EDGES[i+1] - ADJ_EDGES[i])
            return (ADJ_FRAC[i] + f * (ADJ_FRAC[i+1] - ADJ_FRAC[i])) * ADJ_SCALE

def unadj(a):
    """adjusted year -> real year"""
    s = a / ADJ_SCALE
    for i in range(4):
        if s <= ADJ_FRAC[i+1] or i == 3:
            f = (s - ADJ_FRAC[i]) / (ADJ_FRAC[i+1] - ADJ_FRAC[i])
            return ADJ_EDGES[i] + f * (ADJ_EDGES[i+1] - ADJ_EDGES[i])

# ADJUSTED-year gap that reads as "no data for a very long time" = 3,000 years before AD 1,
# 1,500 across AD 1-1400, ~480 across 1400-1900, ~120 after. Calibrated against the cities
# either side of the line rather than picked round: it has to clear Kaifeng's 1232-1751 hole
# (2,547 adj) and Rome's 622-1377 (1,510), because both were continuously inhabited and merely
# under-recorded, while still catching Kannauj 1150-1848 (3,322), Kamakura 1250-1946 (4,609),
# Handan (6,586), Memphis 0-1974 (7,813) and Yuzhou -300-1975 (8,138), which are the smears.
# Anything lower starts deleting real cities: at 700 this blanked Rome from 675 to 1350.
# Note what the warp does at the ends -- a purely ancient gap almost never trips it now (the
# whole pre-AD-1 stretch is only 3,501 adjusted years), which is right, because sparse anchors
# in the third millennium BC are the norm and the viewer already shows those years at half
# opacity. The cross-era smears still trip it easily: a gap that ENDS in the modern era is
# thousands of adjusted years long however early it starts.
FADE_GAP = 3000
FADE_YEARS = 150        # ADJUSTED-year decay-out / rise-in ramp at each end of such a gap
FADE_FLOOR = 1000       # floor pop held across the gap middle (below MINPOP -> invisible)
FADES_ON = True
# Safety net rather than an active rule at FADE_GAP 3000: it stops any future loosening from
# reaching into the modern era, where a short control-point gap means the series was a straight
# line there (DP dropped the collinear middle), NOT that the city went unrecorded.
FADE_GAP_MIN_REAL = 200

# --- a real-year second arm was tried here and REJECTED --------------------------------
# The motivation is sound and worth keeping written down. FADE_GAP alone cannot catch the
# cross-era smears, and the reason is arithmetic rather than taste: London's 100..1199 hole
# measures 2,198 adjusted years and Kaifeng's 1232..1751 hole measures 2,547. They want
# opposite verdicts, they are ~14% apart, and the band between them is packed with 1900-2000
# pairs that must not fade at all. The adjusted threshold is SATURATED -- there is no value
# that fades London and spares Song Kaifeng.
#
# The obvious second arm is raw years, since London's gap is 1,099 of them and Kaifeng's 519.
# It was implemented and measured at FADE_GAP_REAL = 900. It does not survive: of the 21 gaps
# it newly fades, roughly half are cities that were CONTINUOUSLY INHABITED and merely under-
# recorded, and it asserts their absence --
#
#     Halab       -1700..622   Aleppo, one of the longest continuously occupied sites anywhere
#     Varanasi     -430..622   continuously inhabited and continuously significant
#     Samarqand    -400..622   Achaemenid Marakanda -> Sogdian capital, thriving at both ends
#     Trabzon       260..1204  a Byzantine port that never stopped being one
#     Changsha     -200..1077  a Chinese prefectural seat throughout
#     Messina      -200..1223  Roman, Byzantine and Norman in turn
#     Memphis     -2500..-1360 the Old and Middle Kingdom capital -- the largest city on earth
#                              for much of the span the rule wanted to blank
#
# The failure is not the threshold, it is the QUESTION. Gap length measures how sparse the
# record is; it cannot distinguish "derelict" from "under-recorded", and those are the two
# hypotheses. No function of (y0, y1, v0, v1) separates Aleppo from London, because the data
# genuinely looks the same in both -- the difference lives entirely outside the dataset.
# This is the same lesson as the note above about not widening the span for older history.
#
# A fade asserts ABSENCE, which is a strong claim and worse than a wrong-but-present line when
# it is false. So the cross-era smears are handled case by case in DISAPPEARED instead, and
# a city not on that list keeps its (fabricated, but non-committal) interpolation.
# Left as a named constant rather than deleted, so nobody re-derives the idea from scratch.
FADE_GAP_REAL = None
# plant_fades is triggered by the carry-forward strip, not by gap length, so FADE_GAP does not
# apply -- but it still needs a floor, or it fades cities across gaps that are just ordinary
# early-modern sparsity. 300 real years is what the old real-year ramp arithmetic enforced
# implicitly; it keeps the 130 cases this exists for (Vijayanagara 341, Kamakura 696) and
# suppresses 53 that would be false abandonments -- Sakai 1582-1877, Constantine 1515-1808,
# Shizuoka 1600-1877, Samarkand 1575-1834, none of which stopped being cities.
# (This constant was originally set by Baghdad's 932-1150 hole, which no longer exists: the
# CENSUS recovery of Chandler's 1000/1100/1200 benchmarks fills it.)
FADE_STRIP_MIN_REAL = 300

def fade_long_gaps(control):
    """Across gaps longer than FADE_GAP adjusted years, insert floor points so a city fades OUT
    after the earlier anchor and back IN before the later anchor, instead of the viewer smearing
    a smooth line across centuries of missing data. This is what kills the interpolation smears:
    Yuzhou / Memphis etc. have a single ancient anchor and a modern one with a ~2,300-yr void
    between -- geometric interp still ramps them to 400-700k across the whole medieval period.
    Absence of data across a millennium means 'not a recorded major city', not 'draw a line'.
    Cities with genuine continuity (Baghdad, Song Kaifeng, Constantinople) have dense anchors
    and short gaps, so they are untouched.

    Measuring the ramps in adjusted years too is what stops them being the eyesore they were:
    a fade-in ending at a 1940 census used to rise out of nothing across 1790-1940, a seventh of
    the visible timeline; now it is the same 150 adjusted years, which is about six real ones."""
    if not FADES_ON or len(control) < 2:
        return control
    out = [control[0]]
    for (y0, v0), (y1, v1) in zip(control, control[1:]):
        if (adj(y1) - adj(y0) > FADE_GAP
                or (FADE_GAP_REAL is not None and y1 - y0 >= FADE_GAP_REAL)) \
                and y1 - y0 >= FADE_GAP_MIN_REAL and max(v0, v1) > FADE_FLOOR:
            out += fade_pts(y0, y1, max(v0, v1))
        out.append([y1, v1])
    return out


FADE_LOG = []   # (name, y0, y1, peak) per planted fade; `build.py --fades` prints it

def fade_pts(y0, y1, v):
    """The pair of floor points bracketing a gap, each FADE_YEARS ADJUSTED years inside its end
    (so the ramp is ~150 real years in antiquity and ~6 after 1900). Empty if they don't fit."""
    a0, a1 = adj(y0), adj(y1)
    if a0 + FADE_YEARS >= a1 - FADE_YEARS:
        return []
    p0, p1 = round(unadj(a0 + FADE_YEARS)), round(unadj(a1 - FADE_YEARS))
    if not y0 < p0 < p1 < y1:
        return []
    FADE_LOG.append([None, y0, y1, v])
    return [[p0, FADE_FLOOR], [p1, FADE_FLOOR]]


def plant_fades(control, fades):
    """Fade a city out after each year in `fades` -- the last genuinely-recorded year before
    a carry-forward run we deleted. Without this the viewer just draws a long geometric line
    from the real estimate down to the first census, which reads as a slow decline when what
    actually happened is 'no record for four centuries'. Mirrors fade_long_gaps' shape, but
    is triggered by the strip rather than by gap length (Vijayanagara's gap is only 341 years),
    so it deliberately skips FADE_GAP / FADE_GAP_MIN_REAL -- we already know the record ends."""
    if not FADES_ON or not fades:
        return control
    out = [control[0]]
    for (y0, v0), (y1, v1) in zip(control, control[1:]):
        if any(y0 <= f < y1 for f in fades) and v0 > FADE_FLOOR \
                and y1 - y0 > FADE_STRIP_MIN_REAL:
            out += fade_pts(y0, y1, max(v0, v1))
        out.append([y1, v1])
    return out


# --- cities that genuinely went away, one at a time -------------------------------------
# The case-by-case half of the cross-era smear problem (see FADE_GAP_REAL for why the general
# rule was rejected). Each entry names a span the city spent below visibility, as
#     "source key": [(y_out, y_in), ...]
# where y_out is roughly when it fell under ~5,000 and y_in roughly when it climbed back. Both
# are planted as FADE_FLOOR points, so the viewer draws a decline INTO the gap and a rise out
# of it rather than a straight line across.
#
# Why explicit years rather than reusing fade_pts' mechanical ramp: fade_pts puts its floor
# points a fixed FADE_YEARS inside each end of the gap, which is fine when the only claim is
# "unrecorded" but wrong when we know the actual dates. Londinium was still substantial into
# the third century and gone by the fifth; the mechanical ramp fades it out by AD 175, which
# is both wrong and checkable. Where we are asserting a collapse we should assert the right
# one, and a reader who knows the period will check it.
#
# Bar for entry: the city has to have gone genuinely quiet, not merely unrecorded. "Aleppo has
# no data between 1700 BC and AD 622" is not a qualification -- Aleppo was there the whole
# time. The test is whether a historian would say the site was abandoned or reduced to a
# village, and it is a judgement call every time, which is exactly why this is a table.
# This table OVERRIDES the generic rules for the cities in it: any floor points fade_long_gaps
# or plant_fades left behind are removed first, then the named spans are planted. An entry
# mapping to [] therefore reads "this city was continuous, take the fade off", which is the
# single commonest correction here -- the generic rules blank six cities that never went away.
#
# Every entry below was researched individually and the dates are the claim, not decoration.
# Where the record is genuinely disputed the entry is [] rather than a guess: a fade asserts
# absence, and a wrong-but-present interpolation is the cheaper error.
DISAPPEARED = {
    # --- collapses no rule catches, because the smear is one straight line ---------------
    # Londinium was "almost completely abandoned" by 457 and there is no evidence of anyone
    # inside the walls for the next two centuries. Lundenwic grows as a separate port at
    # Covent Garden from the 670s and is Bede's emporium by 731.
    "London-United Kingdom": [(450, 670)],
    # Athens. The Herulian sack of 267 shrank the city into a ~10 ha circuit, the Slav sack of
    # 582 and the Byzantine dark age finished the job -- the Agora lay abandoned from the 7th
    # century to the 10th and coins stop appearing by the mid-7th. Taking the CONSERVATIVE end
    # of the range: some read the 7th-8th century trough at 5,000-8,000 rather than below the
    # display floor, since Athens stayed an archbishopric and Constans II wintered his fleet
    # there in 662/3. 650 rather than 620 for the same reason.
    # Keyed on the MERGE_INTO *target*: ancient Athens arrives on the Gazi-Greece record and is
    # merged into Athínai-Greece, and this table runs after that, on the surviving key.
    # (CHANDLER_AD100 keys the same city as "Gazi-Greece" because it runs before the merge.)
    "Athínai-Greece": [(650, 900)],
    # Corinth's real hole hides INSIDE the -430 -> AD 100 line and no gap rule can see it:
    # Mummius destroyed the city in 146 BC and the site held squatters and huts until Caesar
    # refounded it as Colonia Laus Iulia Corinthiensis in 44 BC. The medieval floor the generic
    # rule planted (175..1526) goes -- Byzantine Corinth was capital of the theme of Hellas,
    # a silk town whose weavers the Normans carried off in 1147, and 20,000-25,000 in the 11th
    # century. That leaves AD 100 -> 1550 as an undivided decline, which is wrong in detail but
    # does not claim the city vanished. See §6 note: it wants a positive anchor, not a fade.
    "Kórinthos-Greece": [(-146, -44)],
    # Zhang Xianzhong burned Chengdu on his withdrawal in 1646; an inspector in 1664 found
    # ruins, tiger tracks and a few dozen residents, and the provincial capital sat at Baoning
    # until 1665. The source has none of it -- 120,000 at 1649 held verbatim to 1720, which is
    # populstat's equal-pair carry-forward straddling the one event that mattered.
    "Chengdu-China": [(1650, 1670)],
    # Carthage, the last of the Corinth family and the one §3.1 already flags as wanting a
    # deletion of its own. Scipio razed it in 146 BC; the Gracchan colony of 122 failed, Caesar
    # refounded the site in 44 and Augustus settled it in 29. Between those dates it was farmland.
    # The series runs -200: 173,000 -> -100: 100,000 -> 0: 175,000 straight through, and the
    # -100 row is the SAME mis-filed AD 100 benchmark CHANDLER_AD100 documents and deliberately
    # leaves alone because the entry holds a real anchor at 0 -- so the wrong figure was
    # anchoring the era. DROP_YEARS removes it; this plants the trough it was hiding.
    "Al Marsâ-Tunisia": [(-146, -35)],

    # --- exposed by the new MERGE_INTO entries --------------------------------------------
    # Both of these arrive with a generic fade covering most of their history, because folding
    # the donor in gives the strip a long series to work on for the first time. Keyed on the
    # merge TARGET, like Athens.
    #
    # Sparta. The donor holds Chandler at -430: 40,000, -200: 30,000, 0/100/200: 30,000 -- and
    # that last trio is a 400-year exact-flat run, so the strip ate the AD 100 and AD 200
    # benchmarks and faded from -50. (Same defect as Gao below; check G reports it.) The real
    # shape: Roman Sparta was a working town and something of a tourist attraction into the 4th
    # century, Alaric sacked it in 396, and Byzantine Lakedaimonia on the site was a bishopric of
    # a few thousand -- under the display floor but not nothing. Villehardouin built Mystras in
    # 1249 and the population moved there wholesale; the site was empty until modern Sparti was
    # founded by decree in 1834. 400 rather than 1249 because the test this table uses is "fell
    # under ~5,000", which happened at the earlier date; 1249 is when it reached zero.
    "Spartí-Greece": [(400, 1834)],
    # Pyongyang. Tang and Silla destroyed Goguryeo in 668 and Silla took the site in 676, after
    # which it sat abandoned on the Silla-Balhae frontier until Wang Geon recovered it and made
    # it Goryeo's Sogyong. From 918 it is continuously a major Korean city, so the generic
    # 697..1676 floor is right at its start and a thousand years too long at its end -- it was
    # blanking the whole Goryeo and Joseon city.
    "P'yõngyang-North Korea": [(668, 918)],

    # Tula, whose CENSUS entry leaves a single anchor at 900 and the modern town's 2000 census.
    # The generic rule fades that gap, but its mechanical ramp puts the fade-out at ~975 -- a
    # century and a half before the city fell. Toltec Tula's ceremonial centre was burned around
    # 1150-1200; 1180 is the middle of that range. The recovery date is the softer end, as it is
    # for Ostia above: Tula de Allende is a small Hidalgo town and 1950 is roughly when it passed
    # the display floor, not a census.
    "Tula de Allende-Mexico": [(1180, 1950)],

    # Lhasa's nine-century hole is real, and now that CENSUS has restored the 1700-1840 figures
    # it can be dated instead of ramped. Langdarma was assassinated in 842 and the Tibetan Empire
    # dissolved into the Era of Fragmentation -- no central authority in Tibet between 842 and
    # 1247, and the political centre later sat at Sakya, Nedong and Shigatse rather than Lhasa.
    # Lhasa is restored as the capital in 1642, when Gushi Khan installs the Fifth Dalai Lama and
    # the Ganden Phodrang is founded; the Potala follows in 1645. The Jokhang kept Lhasa a
    # pilgrimage town throughout, so this asserts "under the display floor", not "empty".
    "Lhasa-China": [(842, 1642)],
    # Thanjavur was continuously a capital -- Chola to 1279, then Pandya, Vijayanagara, the
    # Thanjavur Nayaks from 1532 and the Marathas from 1674 -- so the 975..1807 floor the generic
    # rule planted is a false abandonment of an 800-year run of court cities. Taking it off leaves
    # 900: 100,000 declining to Chandler's 1750: 30,000 as one long line. Wrong in detail, since
    # the fall belongs after 1279, but it claims a decline rather than an absence, and the source
    # gives nothing to date a sharper shape with.
    "Thanjâvûr-India": [],

    # --- Africa: the strip's fade misreads "unrecorded" as "dead" -------------------------
    # These are all one defect, and it is the carry-forward strip working exactly as documented
    # rather than a bug. A Sahelian or Nubian city's record is "one pre-modern estimate, then
    # nothing until the colonial era", which produces the exact flat run the strip exists to
    # detect; the run then ENDS at the first modern figure, which the CF_MODERN gate reads as
    # "dead". For these cities that inference is backwards -- the run ends there because that is
    # when European record-keeping STARTS. The rule is detecting the arrival of the observer.
    # Measured at 1575: sub-Saharan Africa drew 9 cities and blanked 8.
    #
    # Gao is the sharpest case and worth keeping written down, because it shows the strip
    # deleting DATA rather than repetition. Its raw series reads 1550: 75000, 1575: 75000,
    # 1585: 75000, 1591: 75000, 1600: 74999.99999999996 ... 1930: 74999.99999999996. Those first
    # four are four separate Chandler benchmarks that happen to carry the same value, and the
    # spline's echo of the last one differs by 5e-16 -- far inside CF_EPS. So the strip sees ONE
    # flat run of 380 years, keeps its first point and deletes the rest, taking three real
    # benchmarks with it, and the fade lands at 1574. Gao at 75,000 in 1575 is the largest city
    # in sub-Saharan Africa and it was invisible. validate.py check G now reports this class.
    "Gao-Mali": [],
    "Tombouctou-Mali": [],       # Timbuktu declines after the 1591 Moroccan conquest but is
                                 # never abandoned -- Caillié found ~12,000 there in 1828, and
                                 # Chandler has 25,000 at 1500 AND at 1600, the second of which
                                 # the strip consumed. Was blanked 1524..1804.
    "Dunqulah-Sudan": [],        # Old Dongola, capital of Makuria. Chandler runs 975: 25,000 ->
                                 # 1200: 30,000 -> 1400: 25,000 -> 1500: 20,000 -> 1800: 10,000,
                                 # a continuous series, and the rule blanked 1165..1924 -- i.e.
                                 # all of it. Declines after 1365; never emptied.
    "Sinnar-Sudan": [],          # founded 1504, capital of the Funj sultanate until 1821. The
                                 # blank started at 1624, inside its working life.
    "Shaki-Nigeria": [],         # Chandler has 1510: 60,000 and still 40,000 at 1850/1880; the
                                 # rule blanked 1534..1946, straight through both later anchors.
    "Dienne-Senegal": [],        # Djenne. Chandler 800: 10,000 -> 1300: 20,000 -> 1500: 20,000,
                                 # blanked 1399..1876. One of the great Niger-delta trade and
                                 # scholarship towns, continuously occupied. See SITE_COORDS --
                                 # it is also filed in the wrong country.
    "M'Banza Congo-Angola": [],  # Sao Salvador, capital of Kongo -- Chandler 1500: 40,000,
                                 # 1543: 50,000; blanked 1567..1964. The Jaga did sack and empty
                                 # it in 1568, but the Portuguese restored Alvaro I in 1571, so
                                 # the real hole is three years and below anything the viewer
                                 # can draw. [] rather than (1568, 1571) for that reason.
    # Allada (Ardra), the kingdom the Slave Coast is named for, and NOT the same case as the
    # seven above -- which is the point of writing it out. The generic rule blanked it
    # 1706..1996 off a single 1682 benchmark, and the first instinct was `[]` like the others.
    # That is wrong here: Gao, Timbuktu and Dongola all have LATER Chandler benchmarks proving
    # the compiler kept tracking them, and Allada has none. Removing the fade outright draws a
    # 320-year line from 40,000 (1682) to 23,400 (2002), i.e. an Allada of ~30,000 through the
    # eighteenth and nineteenth centuries, and that is certainly false -- Agaja of Dahomey took
    # and sacked the town in March 1724, the court moved to Abomey, and Allada spent the rest
    # of its Dahomean and colonial history as a provincial seat on the Ouidah-Abomey road.
    # So the collapse is real and the fade should stay; what was wrong was its DATES.
    #   y_out 1724 is the conquest, and is the claim. With the CF_END entry above restoring
    #     Chandler's 1700 benchmark, the decline is now drawn across those 24 years instead of
    #     out of 1682, so the town is on the map at full size for the whole period the Allada
    #     and Ouidah slave trade ran through it.
    #   y_in 1960 is the weakest number in this entry and is flagged as such: NOTHING measures
    #     Allada between 1700 and the 2002 count of 23,400. It is that count back-projected at
    #     the growth Benin's small towns actually had after 1950, which puts the 5,000 crossing
    #     somewhere mid-century. If it is wrong it is wrong by decades, at a size the viewer can
    #     barely draw, and the alternative -- a visible 30,000 Allada for two centuries -- is
    #     wrong by a factor of ten for two hundred years.
    "Allada-Benin": [(1724, 1960)],

    # --- collapses the generic rule finds but dates badly --------------------------------
    # All of these had their fade-out planted at 175, which is fade_pts' mechanical ramp off
    # an AD 100 anchor rather than a claim about the city. Each is now the real date.
    "Fiumicino-Italy": [(550, 1930)],    # Ostia: last Ostians in a fortified theatre by 537;
                                         # Gregoriopolis (830) held 50-100 people and was still
                                         # 260 souls in the 1600s; Lido di Ostia passes 5,000
                                         # only around 1930. Was 26..1985.
    "Wâdî Moosa-Jordan": [(650, 1985)],  # Petra: the 363 quake wrecked the hydraulics, 551
                                         # finished it, the Petra papyri stop c. 593. Wadi Musa
                                         # had 654 people in 1961 and 10,998 in 1994.
    "Yarîm-Yemen": [(550, 1900)],        # Zafar: Sanaa replaced it as Himyarite capital c.
                                         # 537-548 and the record breaks off. The site holds
                                         # ~450 people today; only Yarim, 10 km away, is a town.
    "Izmir-Turkey": [(1402, 1580)],      # Smyrna's collapse is MEDIEVAL, not ancient: Timur
                                         # razed it in 1402, the 1530 Ottoman survey lists 304
                                         # taxpayers (~1,400 people) and 1576 about 3,500-5,000,
                                         # before the Levant trade boom. The old floor started
                                         # at 75 and blanked the whole Byzantine city.
    "Cádiz-Spain": [(600, 1500)],        # Islamic Qadis was a minor port, "almost deserted"
                                         # when Alfonso X repopulated it in 1262; still 1,303
                                         # houses in 1596 and only ~7,000 by 1600. Old floor
                                         # started at 75, ~525y early. Ends 1500 rather than
                                         # 1550 because 1550 is itself an anchor (3,000) and a
                                         # span has to fit strictly inside a gap -- which is
                                         # right anyway, since the recovery through 3,000 (1550)
                                         # to 10,000 (1594) is data and should be drawn.
    # Nineveh was destroyed in 612 BC and Xenophon found the site deserted in 401; the city
    # revives on the far bank as Mosul, a Sasanian fortress town and then an Arab garrison and
    # capital of the Jazira after 641, reaching Chandler's 40,000 by 752. The generic rule had
    # this backwards -- it blanked 1325..1813, when Mosul was battered but never emptied and
    # Chandler has it at 34,000 in 1800.
    "Mosul-Iraq": [(-600, 640)],

    # --- continuous: the generic fade is a false abandonment, take it off -----------------
    # Varanasi is among the longest continuously inhabited cities anywhere and Xuanzang
    # describes it as dense and prosperous c. 635. It was on the 1,000 floor from 697 to 1616.
    "Vârânasi-India": [],
    # Bactra was first-rank throughout -- Greco-Bactrian capital, Kushan, Sasanian, the great
    # Nawbahar, then the Samanids' "mother of cities" -- and Chandler has it at 30,000 in 1150,
    # inside the 1043..1769 span the rule blanked. Its real destruction is Genghis Khan in 1220
    # (Ibn Battuta still found ruins in 1333), recovering from about 1450; that is a narrower
    # and later claim than anything currently in the data, and wants the Chandler anchor first.
    "Balkh-Afghanistan": [],
    # Ancyra contracted hard in the 650s-660s, to a 350x150m citadel holding perhaps 2,000-3,000
    # (Foss, DOP 31) -- but it stayed a theme capital and metropolitan see, Michael III rebuilt
    # it after 838, and it was ~15,000 by the 1520s tahrir. The planted floor ran 175..1501,
    # which is wrong at both ends and blanks a 30,000-class late-Roman city. Stripping it is
    # unambiguous; replacing it with (650, 900) is defensible but the reading is contested, so
    # this asserts nothing rather than asserting the wrong thing.
    "Ankara-Turkey": [],
}
DISAPPEARED_LOG = []

LONE_ANCHOR_BEFORE = 1800   # a lone PRE-MODERN row is a benchmark; a lone modern one is a district

def bracket_lone_anchor(control, name=""):
    """Give a city whose entire record is ONE point a visible extent.

    `popAt()` returns 0 outside [p[0], p[-1]], so a single-point series is drawn in exactly one
    frame of the timeline and in none of the ~5,500 others -- it is data we hold and never show.
    29 pre-1800 cities were in that state, and every one is a real city: Miletus, Troy, Ani,
    Khajuraho, Prambanan, Istakhr, Dvin, Vallabhi, Tamralipti, Anbar, Loango, Dongo, Surame,
    Dzibalchaltun, Chan Chan.

    The shoulders are the same ones the fade machinery plants everywhere else, FADE_YEARS
    adjusted years either side. That is not an invented population -- it is exactly the claim
    the record supports, "this size at this date, nothing either side" -- and it draws as a
    bubble rising to the benchmark and falling away from it. source_codes() already marks any
    FADE_FLOOR point `f`, so the viewer does not read the shoulders as measurements.

    Pre-modern only, and the gate matters: the other 639 single-point entries are 20th-century
    census rows for city districts and English boroughs -- Vale Royal, Thamesdown, Warringah,
    Landhi Korangi -- which are the duplicate/sub-district problem (§6.5), not this one.
    Bracketing those would put 639 spurious dots on the modern map."""
    if not FADES_ON or len(control) != 1:
        return control
    y, v = control[0]
    if y >= LONE_ANCHOR_BEFORE or v <= FADE_FLOOR:
        return control
    a = adj(y)
    p0, p1 = round(unadj(a - FADE_YEARS)), round(unadj(a + FADE_YEARS))
    if not p0 < y < p1:
        return control
    LONE_ANCHOR_LOG.append((name, y, v))
    return [[p0, FADE_FLOOR], [y, v], [p1, FADE_FLOOR]]


LONE_ANCHOR_LOG = []


def plant_disappearances(control, key):
    """Replace the generic fades for one city with the spans named in DISAPPEARED.

    Two steps, in order. First every FADE_FLOOR point goes: for a city in this table the
    table is the authority, and the commonest correction is deleting a fade rather than
    adding one. Then each named span is planted, in whichever gap contains it.

    A span must land strictly inside one gap between consecutive control points. If it does
    not it is NOT planted and main() reports it as NOT APPLIED -- that means the record has
    a real anchor inside the span, so either the dates or the anchor is wrong and the answer
    is to look, not to overwrite the anchor.

    One sharp edge: Buringh writes a literal 1,000 as its nominal value for a town too small
    to model, and that is numerically identical to FADE_FLOOR (Nimes carries a real 1,000 at
    1399, Drapetsona opens on one at 1300). So the strip cannot tell a planted floor from a
    Buringh datum, and every entry in the table is checked against its built series by hand
    for exactly that reason. Do not make this table long enough that that stops happening."""
    if key not in DISAPPEARED:
        return control
    spans = DISAPPEARED[key]
    kept = [p for p in control if p[1] != FADE_FLOOR]
    n_stripped = len(control) - len(kept)
    if len(kept) < 2:
        return control
    out, planted = [kept[0]], set()
    for (y0, v0), (y1, v1) in zip(kept, kept[1:]):
        for i, (y_out, y_in) in enumerate(spans):
            if y0 < y_out < y_in < y1:
                out += [[y_out, FADE_FLOOR], [y_in, FADE_FLOOR]]
                planted.add(i)
        out.append([y1, v1])
    DISAPPEARED_LOG.append((key, n_stripped, len(planted), len(spans), out))
    return out


# --- per-point provenance ---------------------------------------------------------------
# Stadester is four datasets in a trench coat and labels only the ENTRY, not the year, so a
# single line on the chart can be a Chandler benchmark, a Buringh model value and a populstat
# census in consecutive segments with nothing to say so. provenance.py recovers it; this turns
# its answer into one character per control point, emitted as `s` alongside `p`.
#
# The most valuable distinction is not which of the four -- it is `i`, INTERPOLATION. Only 19%
# of the corpus is a real data point; the other 81% is Stadester's straight-line gap fill, and
# a viewer currently cannot tell the difference between a measured century and an invented one.
SRC_CODES = {
    provenance.CHANDLER:  "c",   # Chandler-Modelski benchmark, matched to chandlerV2 by value
    provenance.BURINGH:   "b",   # Bosker-Buringh-van Zanden, matched on their benchmark grid
    provenance.POPULSTAT: "p",   # populstat's census scrape -- the backbone, ~89% of anchors
    provenance.UNKNOWN:   "u",
}
SRC_FILL, SRC_WUP, SRC_FADE = "i", "w", "f"
# The archaeological tier. provenance.py cannot attribute these -- it recovers which COMPILER a
# year came from, and these entries predate every compiler's coverage -- so the code is assigned
# from the table rather than derived. Worth its own character rather than reusing `u`: these are
# survey and excavation estimates, a different kind of claim from a census or a benchmark, and
# index.html's sourceSpans() then marks the join wherever one meets a Chandler row (Monte Alban
# is the case that matters -- archaeological to AD 700, Chandler's benchmark at 800).
SRC_ARCH = "a"
# CENSUS entries whose figures are archaeological-tier rather than enumerations. Needed because
# a CENSUS lands on an EXISTING entry, so its points go through the normal attribution path and
# provenance.py -- which knows only about compilers -- calls them `i`, straight-line fill. That
# is worse than unattributed: it says "not a measurement" about a cited survey estimate, and it
# suppresses the source-change mark at the point where the tiers actually meet.
#
# Scoped to the years the table INSERTS, not to its clear-window, and the difference is
# load-bearing: Tikal's window stops at 750 precisely so Chandler's own 751 and 800 benchmarks
# survive it, and those must keep reading `c`. Using the window would have been right for Monte
# Alban and wrong for Tikal; using the inserted years is right for both, and it degrades safely
# -- a point DP drops simply never asks.
ARCH_CENSUS = {"Santa Cruz Xoxocotlán-Mexico", "Caracol-Mexico", "Tikal-Guatemala",
               "Tiahuanaco-Bolivia"}

def source_codes(control, r, seam):
    """One character per control point, in the same order as `p`.

    Precedence matters and runs floor -> graft -> attributed -> fill:
      f  a planted FADE_FLOOR point. Ours, not anyone's data, and it must not read as a
         measurement of 1,000 people -- see plant_disappearances on the Buringh collision.
      w  past `seam`, so this is UN WUP 2025 and provenance.py has never seen it; it only
         knows the historical source file.
      c/b/p/u  provenance.py attributed the year.
      i  a control point provenance.py considers straight-line fill. Not a contradiction:
         dp_simplify keeps a point at 1% relative tolerance and provenance splits at 1e-6,
         so DP legitimately keeps points on a fill line (and merge/trim/hold add their own).
         Calling those `i` is the honest answer -- they are not measurements.

    Three records can contribute years to one series and all three are classified, in the order
    the pipeline applies them, so the last writer of a value is the one that names it:
      donors    a MERGE_INTO donor, which only ever supplies years OUTSIDE the target's own
                range -- so there is no overlap to resolve, and the target winning is exact
                rather than a tie-break. Athens proved this one: its entire ancient record is
                Gazi's, and classifying only the survivor reported all of it as fill.
      entry     the entry itself.
      variant   the agglomeration variant PREFER_VARIANT or prefer_agglomeration spliced in,
                which OVERWRITES the base across the years it covers, so it is applied last.
                Alexandria proved this one -- its whole series arrives via its (agglomeration)
                entry, so every ancient point read as interpolation until the variant was
                classified too. Cheap to get wrong and very visible: that is a top-3 city for
                the entire classical era."""
    codes = []
    # Assigned, not derived -- see SRC_ARCH. Fade points still win, so a planted floor does not
    # read as an excavation estimate of 1,000 people.
    if r.get("key") in ARCHAEOLOGICAL:
        return "".join(SRC_FADE if v == FADE_FLOOR else SRC_ARCH for _, v in control)
    # The years an archaeological-tier CENSUS wrote; see ARCH_CENSUS.
    arch_years = set(CENSUS[r["key"]][2]) if r.get("key") in ARCH_CENSUS else ()
    attr = {}
    for donor in r.get("donor_entries") or ():
        attr.update(provenance.classify(donor))
    if r.get("entry") is not None:
        attr.update(provenance.classify(r["entry"]))
    if r.get("variant_entry") is not None:
        attr.update(provenance.classify(r["variant_entry"]))
    for y, v in control:
        if v == FADE_FLOOR:
            codes.append(SRC_FADE)
        elif y in arch_years:
            codes.append(SRC_ARCH)
        elif seam is not None and y > seam:
            codes.append(SRC_WUP)
        else:
            src, _ = attr.get(y, (provenance.FILL, None))
            codes.append(SRC_CODES.get(src, SRC_FILL))
    return "".join(codes)


def sig3(x):
    """round to 3 significant figures, integer-valued."""
    if x <= 0:
        return 0
    d = math.floor(math.log10(x))
    p = 10 ** (d - 2)
    return int(round(x / p) * p)


FALLBACK_MIN_STACK = 3    # distinct city names on one exact point = a geocoder fallback
FALLBACK_DP        = 4    # ~11m. See the docstring for why this is not validate's 2dp.

def drop_fallback_stacks(recs, centres, grid):
    """Remove entries still sitting on a country centroid after coord_fixes.json has run.

    `propose_coords.py` re-homes a stranded entry by matching it to a WUP centre in the same
    country, and that fixes everything WUP knows about -- 32 of the 55 on Russia's centroid.
    It cannot reach the rest, for a structural reason: WUP's floor is 50,000 and the leftovers
    are all under it (Kandalaksha 48.5k, Kurchatov 48.2k, Gryazi 48.3k). There is no centre to
    match them to, so they stayed where the geocoder put them and drew as a 22-deep pile of
    bubbles in central Siberia, with another 20 in Morocco and 18 in the Netherlands.

    A city drawn 3,000km from itself is worse than one absent, so they go. They are not lost
    work: every one carries its standard transliteration in `other_names` (`Gat`'cina` ->
    Gatchina), so a coordinate source that goes below 50k -- GeoNames, per §7 -- could restore
    the lot. §6.11 has the reasoning.

    Detection is validate check D's rule with ONE change: coordinates must match EXACTLY
    (FALLBACK_DP), where check D rounds to 2dp. Reporting can afford to be loose; deleting
    cannot. Every genuine fallback stack in the data resolves to a single coordinate to the
    metre -- a geocoder returns one centroid, not a scatter -- while 2dp is a ~1km cell, and
    the only thing it catches that exact matching does not is Senglea / Vittoriosa / Isla, the
    three towns of Malta's Three Cities, which really are ~500m apart with real distinct
    coordinates. So check D is expected to keep reporting that one point after this runs;
    that is the rule being loose, not a stack being missed."""
    names_at, at = defaultdict(set), defaultdict(list)
    for r in recs:
        pt = (round(r["la"], FALLBACK_DP), round(r["lo"], FALLBACK_DP))
        names_at[pt].add(r["n"].split(" (")[0].lower())
        at[pt].append(r)
    bad = {p for p, v in names_at.items() if len(v) >= FALLBACK_MIN_STACK}
    # A merged twin city and its two halves land on one point legitimately -- Villingen +
    # Schwenningen + Villingen-Schwenningen, Sekondi + Takoradi + Sekondi-Takoradi. The
    # hyphenated name being exactly the other two entries is the proof, and it is a much
    # narrower test than plain containment: Greece's centroid holds "Eleftherios" and
    # "Eleftherios Venizelos", which containment would rescue and this does not.
    for p in list(bad):
        for nm in names_at[p]:
            parts = [q.strip() for q in nm.split("-") if len(q.strip()) >= 4]
            if len(parts) >= 2 and all(q in names_at[p] for q in parts):
                bad.discard(p)
                break
    if not bad:
        return recs, [], 0, []

    # Not every crowded point is a centroid. Where a geocoder collapses a city AND its own
    # districts onto the city's real location the result looks identical to a fallback, and
    # deleting the lot would take a real city with it: Panama City (704,100) sits under Bella
    # Vista and Rufino Alfaro, two of its own corregimientos; Sekondi-Takoradi under Sekondi
    # and Takoradi; Villingen-Schwenningen under Villingen and Schwenningen.
    # WUP settles it. An entry that belongs where it sits has a same-NAMED urban centre on top
    # of it; the towns stranded on Russia's or Germany's centroid have nothing there. Name
    # agreement is required, not mere proximity -- Spain's fallback point is Madrid itself, so
    # a distance test alone would keep Erandio and Galdakao on the strength of Madrid's centre.
    def belongs(r):
        ci = match_centre(r["n"], r["la"], r["lo"], r["peak"], centres, grid)
        return ci is not None and names_agree(r["n"], centres[ci][3])

    keep, dropped, rescued = [], 0, []
    for r in recs:
        pt = (round(r["la"], FALLBACK_DP), round(r["lo"], FALLBACK_DP))
        if pt not in bad or belongs(r):
            if pt in bad:
                rescued.append(r["n"])
            keep.append(r)
        else:
            dropped += 1
    pts = [(p, sorted(x["n"] for x in at[p]), max(x["peak"] for x in at[p])) for p in bad]
    return keep, pts, dropped, rescued


def load_coord_fixes():
    """Repairs for entries the source's geocoder dumped on a country centroid.

    Two kinds, both keyed by stadester key. An entry with lat/lon is relocated to a real WUP
    urban centroid (never a hand-typed coordinate). An entry with "drop": true is a district
    or a since-merged town whose parent city is already in the dataset -- Lyubertsy and
    Mytishchi inside Moscow, Tottenham inside London, Fuse inside Higashiosaka. There is no
    correct standalone location for those, and leaving them stranded paints a phantom city in
    the middle of Siberia, so they go the way of Kensington and Chelsea."""
    if not os.path.exists(COORDFIX):
        print(f"note: {COORDFIX} missing -- no coordinate repairs applied")
        return {}, set()
    with open(COORDFIX, encoding="utf-8") as f:
        fixes = json.load(f)
    moves = {k: (v["lat"], v["lon"]) for k, v in fixes.items() if not v.get("drop")}
    drops = {k for k, v in fixes.items() if v.get("drop")}
    return moves, drops


def main():
    with open(SRC, encoding="utf-8") as f:
        raw = json.load(f)
    print(f"loaded {len(raw):,} entries")
    # Cities the fusion lost, injected before anything reads the dict so they are indistinguish-
    # able from source entries from here on. Refuses to overwrite a real key, since a collision
    # would mean the entry exists after all and the table is out of date.
    for skey, sentry in SYNTHETIC.items():
        if skey in raw:
            print(f"  note: SYNTHETIC {skey!r} already in source -- NOT injected")
            continue
        raw[skey] = dict(sentry, key=skey, other_names=None, province="")
        print(f"  SYNTHETIC: injected {skey} "
              f"({min(sentry['population'], key=int)}..{max(sentry['population'], key=int)})")
    # Same injection, different evidentiary tier -- see ARCHAEOLOGICAL for why the bar differs.
    for akey, aentry in ARCHAEOLOGICAL.items():
        if akey in raw:
            print(f"  note: ARCHAEOLOGICAL {akey!r} already in source -- NOT injected")
            continue
        raw[akey] = dict(aentry, key=akey, other_names=None, province="")
        print(f"  ARCHAEOLOGICAL: injected {akey} "
              f"({min(aentry['population'], key=int)}..{max(aentry['population'], key=int)})")
    # Before anything else reads the years: Chandler's AD 100 table is filed at 100 BC.
    fix_chandler_ad100(raw)
    print(f"chandler AD-100 benchmarks re-dated: {len(AD100_LOG)} of {len(CHANDLER_AD100)}")
    if "--ad100" in sys.argv:
        for k, v, end in AD100_LOG:
            print(f"     {k[:34]:34s} {v:>10,.0f} moved -100 -> 100, fill cleared to {end}")
    # Derive the source year-grid AFTER the AD-100 re-dating, so provenance.py caches anchors
    # for the corrected series rather than the ones it is about to stop seeing.
    provenance.init(raw)
    print(f"provenance grid derived: {len(provenance.year_grid())} Buringh benchmark years")
    us_census, us_merges = load_us1950()
    us_modern = load_us_modern()
    us_metro, us_metro_cbsa = load_us_metro() if "--no-us-metro" not in sys.argv else ({}, {})
    us1950_keys = set(us_census)
    centres, grid = load_ghsl()
    coord_fix, coord_drop = load_coord_fixes()
    variant_index = index_variants(raw)
    print(f"agglomeration variants indexed: {len(variant_index):,}")
    print(f"coordinate repairs available: {len(coord_fix):,} moves, {len(coord_drop):,} drops")

    dropped_variant = dropped_nocoord = dropped_empty = dropped_small = dropped_dup = 0
    clipped_newworld = n_year0 = kept_newworld_benchmark = 0
    n_coordfix = n_clipped = n_carryfwd = n_variant = n_agglom = n_sitecoord = 0

    # --- pass 1: parse clean city records + their historical population dict ---
    # dedup exact duplicate entries: same name within ~11km (keeps richer series)
    by_id = {}
    # (bare name, country) of every entry the source files WITHOUT a parenthetical -- i.e. the
    # cities that already have a plain dot, which is what makes a DUP_MARKERS entry a duplicate.
    plain_entries = {(norm(c.get("name", "")), c.get("country", ""))
                     for c in raw.values() if "(" not in (c.get("name") or "")}
    dropped_dupmark = n_dupmark_kept = n_merged = n_census = n_census_cities = 0
    donors = {}
    region_conflicts = {}

    for key, c in raw.items():
        if key in DROP_KEYS or key in coord_drop:
            dropped_variant += 1
            continue
        name = c.get("name", "")
        low = name.lower()
        # A MERGE_INTO donor is exempt from both marker drops. Naming a key in that table IS
        # the decision to keep it, and the drops run first, so without this the donor never
        # reaches the fold below and the merge reports "donor not present". That is not
        # hypothetical: Sparta and Pyongyang have their ENTIRE pre-modern record in the
        # "(agglomeration)" entry and their plain entries start in 1861 and 1890, so the
        # parenthesis was deleting classical Sparta and Lelang-era Pyongyang outright. Benin
        # City is the DUP_MARKERS half of the same defect -- its variant carries 1600..1900
        # and the plain entry starts at 1901.
        donor = key in MERGE_INTO
        if not donor and any(m in low for m in DROP_MARKERS):
            dropped_variant += 1
            continue
        if not donor and any(m in low for m in DUP_MARKERS):
            if (norm(name[:name.index("(")]), c.get("country", "")) in plain_entries:
                dropped_dupmark += 1          # drawn twice: the plain entry already has it
                continue
            name = name[:name.index("(")].strip() or name   # sole entry for the city: keep it,
            n_dupmark_kept += 1                             # minus the parenthetical
            low = name.lower()
        co = c.get("coords")
        if key in coord_fix:                       # geocoder fallback -> repaired location
            co = list(coord_fix[key])
            n_coordfix += 1
        if key in SITE_COORDS:                     # ...and the hand table wins over both
            co = list(SITE_COORDS[key][:2])
            n_sitecoord += 1
        if not co or len(co) != 2 or (co[0] == 0 and co[1] == 0):
            dropped_nocoord += 1
            continue
        ctry = c.get("country", "")
        new_world = in_americas(ctry, co[1])
        if co[1] < -30 and not new_world:
            region_conflicts[ctry] = region_conflicts.get(ctry, 0) + 1
        S = {}
        for ystr, v in c.get("population", {}).items():
            try:
                y = int(ystr)
            except ValueError:
                continue
            if y == 0:                             # see YEAR_ZERO
                n_year0 += 1
                continue
            if y < YEAR_LO or y > YEAR_HI or not v or v <= 0:
                continue
            v = float(v)
            # ARCHAEOLOGICAL entries skip the ramp entirely. The ramp suppresses UNATTESTED
            # compiler stamps; an entry whose comment cites its source is the case it was never
            # aimed at, and since it drops everything before NW_RAMP_START outright it would
            # otherwise delete most of that table (Cuicuilco's whole rise, El Mirador's).
            if new_world and key not in ARCHAEOLOGICAL:
                if y < NW_RAMP_START:                  # deep-antiquity phantom -> gone
                    clipped_newworld += 1
                    continue
                cap = nw_cap(y)
                if cap is not None and v > cap:        # ramp: suppress implausibly-large early value
                    if chandler_benchmark(key, c, y, v):   # ...unless it is a real benchmark
                        kept_newworld_benchmark += 1
                    else:
                        v = cap
                        clipped_newworld += 1
            S[y] = v
        # North-American placeholder data (see ANGLO_NA above). No Anglo/French city existed
        # before ~1700, so (a) drop everything pre-1700, then (b) strip a long flat leading run --
        # the placeholders are either fabricated ramps (Boston-MA-type, gone via (a)) or a
        # constant held for a century+ that spills past 1700 (Cincinnati holds 10k to 1810).
        if in_anglo_na(ctry, co[1]) and key not in NA_CLIP_KEEP:
            for y in [y for y in S if y < NA_CLIP_BEFORE]:
                del S[y]; clipped_newworld += 1
            pts = sorted(S.items())
            k = 0
            while k < len(pts) and pts[0][1] and abs(pts[k][1] / pts[0][1] - 1) < 0.05:
                k += 1
            if k >= 2 and pts[k - 1][0] - pts[0][0] > 80:      # flat for a century+ = placeholder
                for y, _ in pts[:k]:
                    del S[y]; clipped_newworld += 1
        for y in DROP_YEARS.get(key, ()):        # individual bad points
            S.pop(y, None)
        # Two different reasons to reach for the parenthetical variant, so two rules.
        # PREFER_VARIANT (hand table): the BASE entry interleaves two definitions year by
        # year -- London sawtoothing 4.5M/7.7M through the 1920s -- so the variant is taken
        # only across the years it covers and the base supplies the deeper history.
        # index_variants (automatic): the base is internally consistent but ends on a
        # city-proper figure, which the WUP switch then has to jump off. Here the variant
        # takes over for good from its first year. See prefer_agglomeration.
        vkey = PREFER_VARIANT.get(key) or variant_index.get((norm(name), c.get("country", "")))
        hand = key in PREFER_VARIANT
        variant_entry = None
        if vkey and vkey in raw:
            variant_entry = raw[vkey]        # travels to source_codes(); see its docstring
            V = {}
            for ystr, v in (raw[vkey].get("population") or {}).items():
                try:
                    vy = int(ystr)
                except ValueError:
                    continue
                # `vy != 0` matters here as much as in the base loop, and is easy to miss:
                # the variant is read straight off `raw` rather than through it, so Alexandria
                # -- whose whole series arrives via its (agglomeration) entry -- kept its
                # year-0 row and its AD 1 cliff after the base-loop fix. See YEAR_ZERO.
                if vy != YEAR_ZERO and YEAR_LO <= vy <= YEAR_HI and v and float(v) > 0:
                    V[vy] = float(v)
            if V and hand:
                lo, hi = min(V), max(V)
                S = {y: v for y, v in S.items() if y < lo or y > hi}
                S.update(V)
                n_variant += 1
            elif V:
                # needs the city's WUP centre to price the trade, so match it here; pass 2
                # re-matches on the final (possibly spliced) peak.
                gi = match_centre(RENAME.get(key, name), co[0], co[1],
                                  max(S.values()) if S else 0, centres, grid)
                S, took = prefer_agglomeration(S, V, centres[gi][2] if gi is not None else None)
                n_variant += took
                n_agglom += took
        # entry mixes two places' histories (Danapur holding Pataliputra's figures)
        clip = CLIP_BEFORE.get(key)
        if clip is not None:
            for y in [y for y in S if y < clip]:
                del S[y]
                n_clipped += 1
        # populstat carry-forward: keep the real estimate, drop its centuries of repetition
        n_before_cf = len(S)
        S, cf_fades = strip_carry_forward(S, key)
        if len(S) < n_before_cf:            # counts strips, not fades -- since CF_MODERN most
            n_carryfwd += 1                 # stripped runs are steps and earn no fade year
        if not S:
            dropped_empty += 1
            continue
        # Antiquity-emerging New World cities (Teotihuacan etc.) get a sub-visibility seed so
        # they fade in over ~300 yrs instead of popping in. Modern emergents (colonial cities)
        # don't -- seeding them would paint a spurious pre-founding presence.
        if new_world:
            y0 = min(S)
            if y0 <= NW_RAMP_FULL:
                seed = max(NW_RAMP_START, y0 - 300)
                if seed < y0:
                    S[seed] = PEAK_FLOOR * 0.4   # below peak+display floors: invisible, never resurrects
        # RENAME wins outright -- its values are already written the way they should read, and
        # putting them through clean_name() would only risk it re-editing a hand decision.
        #
        # This is not purely cosmetic, and the second effect is the useful one. The pass-1
        # match_centre() above still passes the RAW name, unchanged; but pass 2 matches on
        # rec["n"], which is this, so a repaired name is also a repaired join against WUP's
        # centre names. Both repairs move it the right way -- norm() leaves "ni`'znij novgorod"
        # and "qahirah, al-" as-is, against WUP's "nizhniy novgorod" and "al-qahirah", while the
        # cleaned forms drop the backticks and put the article back on the front. Expect the
        # graft counts to move slightly; watch validate.py C and H.
        disp = RENAME.get(key) or clean_name(name)
        rec = {"n": disp, "la": co[0], "lo": co[1], "t": c.get("type", "?"),
               "S": S, "peak": max(S.values()), "fades": cf_fades, "key": key,
               "country": c.get("country", ""),
               # the RAW entry, kept for source_codes(). Deliberately the untouched source
               # object rather than S: provenance.py answers about what stadester actually
               # held, and it caches anchors by the identity of this population dict, so
               # handing it a cleaned copy would silently miss the cache and recompute.
               "entry": c, "variant_entry": variant_entry}
        if key in MERGE_INTO:                   # donor: held aside, folded in below
            donors[key] = rec
            continue
        idk = (norm(disp), round(co[0], 1), round(co[1], 1))
        old = by_id.get(idk)
        if old is None:
            by_id[idk] = rec
        else:                                   # keep the richer duplicate
            dropped_dup += 1
            # ...unless one of the pair is a us1950 target. Richness is the wrong tiebreak for
            # the US twins: the metro entry is bigger BECAUSE it is the metro, so it won every
            # time and the map drew US cities at agglomeration size for the whole 20th century.
            # A us1950 key has been checked against the census directly (see the file header),
            # so that check outranks the heuristic. Without this the merge below cannot help --
            # it runs after dedup, and by then the verified twin is already gone.
            rank_new = (rec["key"] in us1950_keys, len(S), rec["peak"])
            rank_old = (old["key"] in us1950_keys, len(old["S"]), old["peak"])
            if rank_new > rank_old:
                by_id[idk] = rec

    # --- fold each donor's out-of-range years into its target (see MERGE_INTO) ---
    by_key = {r["key"]: r for r in by_id.values()}
    merges_all = dict(us_merges)
    merges_all.update(MERGE_INTO)        # the hand table wins any conflict
    n_us_merge = sum(1 for d in us_merges if d not in MERGE_INTO)
    print(f"  us1950: {len(us_census)} census points, {n_us_merge} twin merges from {US1950}")
    for dkey, tkey in merges_all.items():
        d, t = donors.get(dkey), by_key.get(tkey)
        if d is None or t is None:
            print(f"  MERGE_INTO: skipped {dkey!r} -> {tkey!r} "
                  f"({'donor' if d is None else 'target'} not present)")
            continue
        lo, hi = min(t["S"]), max(t["S"])
        add = {y: v for y, v in d["S"].items() if y < lo or y > hi}
        if not add:
            continue
        t["S"].update(add)
        # the donor's raw entry has to travel with its years, or source_codes() classifies the
        # merged series against the target alone and reports the donor's half as interpolation.
        # Athens is the whole point of MERGE_INTO and was the case that showed this: its entire
        # ancient record is Gazi's, so all of it read as fill.
        t.setdefault("donor_entries", []).append(d["entry"])
        t["fades"] = sorted(set(t.get("fades") or []) |
                            {f for f in (d.get("fades") or []) if f < lo or f > hi})
        t["peak"] = max(t["S"].values())
        n_merged += 1
        print(f"  MERGE_INTO: {d['n']} -> {t['n']}, {len(add):,} years outside {lo}..{hi} "
              f"(now {min(t['S'])}..{max(t['S'])})")
    recs = list(by_id.values())

    # Hand-entered censuses go in AFTER dedup, and the ordering is load-bearing. Clearing a span
    # removes points, dedup ranks co-located duplicates by (point count, peak), and every US city
    # here is filed TWICE at identical coordinates -- "Cleveland-Ohio" (city proper, peak 922,900)
    # beside "Cleveland-United States" (metro, peak 2,134,395). Running this in pass 1 shrank the
    # city-proper entry until the METRO one won dedup, so filling Cleveland with its real censuses
    # silently swapped the map onto a 2.1M metro series. Applied here it cannot influence that
    # choice at all.
    by_key = {r["key"]: r for r in recs}
    census_all = dict(us_census)
    census_all.update(CENSUS)            # hand entries win over the bulk file
    for ckey, (clear_lo, clear_hi, pts) in census_all.items():
        r = by_key.get(ckey)
        if r is None:                        # lost to dedup or dropped upstream
            print(f"  note: CENSUS entry {ckey!r} is not on the map -- skipped")
            continue
        S = r["S"]
        for y in [y for y in S if clear_lo <= y <= clear_hi]:
            del S[y]
        S.update(pts)
        r["peak"] = max(S.values())
        n_census += len(pts)
        n_census_cities += 1

    # --- drop the entries the geocoder dumped on a country centroid ---------------------
    recs, fb_pts, fb_dropped, fb_kept = drop_fallback_stacks(recs, centres, grid)
    if fb_pts:
        print(f"geocoder fallback points: dropped {fb_dropped:,} entries stranded on "
              f"{len(fb_pts):,} centroids"
              + (f"; kept {len(fb_kept)} that WUP puts there by name ("
                 + ", ".join(sorted(fb_kept)[:6]) + ")" if fb_kept else ""))
        for pt, names, pk in sorted(fb_pts, key=lambda t: -t[2])[
                :999 if "--stacks" in sys.argv else 10]:
            print(f"     {pt[0]:>8.3f},{pt[1]:<9.3f} {len(names):>3} entries  "
                  f"peak {pk:>9,.0f}   " + ", ".join(names[:3])[:52])

    # --- join each city to its own WUP centre; one principal per centre ---
    # Two cities can still land on one centre (adjacent towns both inside GRAFT_TIGHT_KM of it),
    # and only one of them can carry its modern series, so rank by (name matches, peak). The
    # name test settles the case where a city sits inside a bigger neighbour's polygon and would
    # otherwise win it on size alone: WUP has no Dongguan centre (it falls inside Shenzhen's),
    # and Dongguan's 1.74M history outweighs Shenzhen's own, so without this Dongguan takes
    # Shenzhen's modern tail and Shenzhen is left ending in 2001.
    # GRAFT_PRINCIPAL_WINS overrides the name test where WUP labels the centre after a district
    # of the city rather than the city -- see the table for why it is a hand list.
    principal = {}   # centre idx -> rec index that gets the graft
    n_named = n_tight = 0
    for ri, r in enumerate(recs):
        ci = match_centre(r["n"], r["la"], r["lo"], r["peak"], centres, grid)
        r["centre"] = ci
        if ci is None:
            continue
        named = names_agree(r["n"], centres[ci][3])
        n_named += named
        n_tight += not named
        rank = (r["key"] in GRAFT_PRINCIPAL_WINS, named, r["peak"])
        if ci not in principal or rank > principal[ci][0]:
            principal[ci] = (rank, ri)
    principal = {ci: ri for ci, (rank, ri) in principal.items()}

    # --- US modern era, for the cities the graft cannot reach --------------------------------
    # Applied HERE, after the graft principal is known and before pass 2 uses it, because it is
    # conditional on that decision. A grafted city already ends on a WUP agglomeration tail, and
    # a city-proper 2020 dropped on top of it would manufacture the exact definition oscillation
    # the rest of the pipeline exists to remove -- New York's WUP 2025 is ~18M against a city
    # proper of 8.2M. So these go only to cities with NO graft, which is precisely the set that
    # was dead: populstat stops about 2000 and WUP maps agglomerations >=50k rather than places,
    # so ~1,300 US cities had nothing after 2000 and drew as held, dimmed, no-data.
    n_mod = n_mod_skip = 0
    for ri, r in enumerate(recs):
        pts = us_modern.get(r["key"])
        if not pts:
            continue
        ci = r.get("centre")
        if ci is not None and principal.get(ci) == ri:
            n_mod_skip += 1                  # grafted: WUP owns its modern era
            continue
        r["S"].update(pts)
        r["peak"] = max(r["S"].values())
        n_mod += 1
    print(f"  us_modern: {n_mod:,} ungrafted US cities given 2010/2020/2024 place figures "
          f"({n_mod_skip:,} skipped as grafted)")
    print(f"WUP join: {n_named:,} by name, {n_tight:,} by proximity only "
          f"-> {len(principal):,} centres claimed of {len(centres):,}")

    # --- pass 2: merge modern tail onto principals, simplify, emit ---
    cities = []
    grafted = n_extended = n_msa = 0
    msa_used = set()
    n_despiked = despiked_pts = 0
    despike_log = []
    ymin, ymax = 9999, -9999
    total_pts_before = total_pts_after = 0
    for ri, r in enumerate(recs):
        S = r["S"]
        total_pts_before += len(S)
        ci = r["centre"]
        gpop = centres[ci][2] if ci is not None else None
        # US metro override, the top of the modern-source stack: us_metro > FUA > WUP. Swapped
        # in whole, before trim_admin_tail and trim_terminal_unit_switch see it, so those
        # arbitrate populstat's two-definition tail against the figure that will actually be
        # drawn rather than against one this line then replaces.
        if gpop is not None and r["key"] in us_metro:
            gpop = us_metro[r["key"]]
            n_msa += 1
            msa_used.add(r["key"])
        if (ci is not None and principal.get(ci) == ri
                and r["key"] not in GRAFT_DENY
                and r["peak"] >= GRAFT_MIN_FRAC * max(gpop.values())):
            # resolve populstat's two-definition tail before deciding where the seam falls,
            # or the switch lands on whichever level happened to be last
            # administrative-area tail first, so the terminal-unit trim below sees the city's
            # own figures rather than a county's
            S = trim_admin_tail(S, gpop, r["n"], r.get("country", ""))
            S = trim_terminal_unit_switch(S, gpop, r["n"])
            seam = max(S)                        # last historical year; past it the points
            merged = merge_series(S, gpop)       # are WUP's, a different source entirely
            if max(merged) > max(S):
                grafted += 1
        else:
            # An ungrafted entry reaches the map as pure populstat, so its two-definition tail
            # is still there and nothing downstream will ever contradict it -- and the hold
            # below then freezes whichever level it ended on all the way to YEAR_NOW. Same
            # trim, arbitrated against the entry's own record instead of against WUP.
            S = trim_terminal_unit_switch(S, None, r["n"])
            seam = None                          # nothing grafted: it is all historical
            merged = S
        pts = sorted(merged.items())

        if max(v for _, v in pts) < PEAK_FLOOR:
            dropped_small += 1
            continue

        simp = dp_simplify(pts, DP_EPS_REL)
        # definition spikes are only visible once DP has collapsed the source's linear fill
        # back to its real anchors, so this runs here and not on the raw series in pass 1.
        simp, spikes = despike(simp, r["key"], seam)
        if spikes:
            n_despiked += 1
            despiked_pts += len(spikes)
            for y, v, y0, v0, y2, v2 in spikes:
                despike_log.append((v, y, r["n"], y0, v0, y2, v2))
        nf = len(FADE_LOG)
        control = plant_fades([[y, sig3(v)] for y, v in simp], r.get("fades"))
        control = fade_long_gaps(control)
        control = plant_disappearances(control, r["key"])
        control = bracket_lone_anchor(control, r["n"])
        for e in FADE_LOG[nf:]:
            e[0] = r["n"]
        # --- hold a recent-but-unmatched city forward to the present -----------------
        # Without this the map empties out after 2000: 17,517 cities visible in 1990, 6,372 in
        # 2025. That is not "we stopped knowing", it is drawn as the towns ceasing to exist,
        # and it makes every aggregate read of the modern era wrong in the same direction.
        # 79% of the cities that stop are under 50,000, i.e. below WUP's threshold, so no
        # modern figure for them exists in any source we have -- holding the last one is the
        # only way to keep them on the map at all.
        # Held FLAT rather than grown: a growth factor would be an assumption applied to
        # 14,000 towns at once, and plenty of the world (rural North America, inland Japan,
        # the ex-DDR) is flat or shrinking. Flat says "this is the last measurement", which is
        # exactly what it is.
        # The honesty is carried by EXTEND_FROM: the held run is excluded from the viewer's
        # data-windows via "hx", so it draws at quarter weight on the graph and dimmed on the
        # map, the same treatment as any other stretch with no data behind it. And the run is
        # short by construction -- a city whose record stops in 1955 is not dragged 70 years
        # forward, it still ends where its data ends.
        held_from = None
        if control[-1][0] < YEAR_NOW and control[-1][0] >= EXTEND_FROM \
                and control[-1][1] >= PEAK_FLOOR:
            held_from = control[-1][0]
            control = control + [[YEAR_NOW, control[-1][1]]]
            n_extended += 1
        total_pts_after += len(control)

        ymin = min(ymin, control[0][0])
        ymax = max(ymax, control[-1][0])
        rec_out = {
            "n": r["n"],
            "la": round(r["la"], 4),
            "lo": round(r["lo"], 4),
            "t": r["t"],
            "p": control,
            "s": source_codes(control, r, seam),
        }
        # last historical year before the WUP handover. The viewer ignores this; validate.py
        # uses it so check F stops re-reporting the seam step as source oscillation -- the
        # step is real but it is merge_series', already measured by SWITCH_STEPS, and at one
        # point it was 89% of everything check F reported.
        if seam is not None and max(y for y, _ in control) > seam:
            rec_out["sw"] = seam
        if held_from is not None:            # last year with data; past it the line is a hold
            rec_out["hx"] = held_from
        cities.append(rec_out)

    cities.sort(key=lambda c: -max(p[1] for p in c["p"]))  # biggest first (draw order)
    out = {"yearMin": ymin, "yearMax": ymax, "cities": cities}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))

    sz = os.path.getsize(OUT) / 1e6
    print(f"kept {len(cities):,} cities  |  year range {ymin}..{ymax}  |  grafted modern tail onto {grafted:,}")
    if us_metro:
        # An unused key is an MSA whose city never grafted -- no WUP centre matched it, or a
        # co-located duplicate entry won the centre. Reported biggest-first because those are
        # the ones worth a coord_fixes or GRAFT_PRINCIPAL_WINS entry.
        landed = {us_metro_cbsa[k] for k in msa_used}
        best = {}
        for k, v in us_metro.items():
            c = us_metro_cbsa[k]
            if c in landed:
                continue
            p = max(v.values())
            if c not in best or p > best[c][0]:
                best[c] = (p, k)
        unused = sorted(best.values(), reverse=True)
        print(f"  US metro: {n_msa:,} grafted cities took a Census MSA series "
              f"-- {len(landed):,} of {len(set(us_metro_cbsa.values())):,} MSAs landed, "
              f"{len(unused):,} never reached a graft")
        for p, k in unused[:12]:
            print(f"     {k:34}{p:>12,.0f}")
    if n_census:
        print(f"hand-entered {n_census} census figures (CENSUS table) into {n_census_cities} cities")
    print(f"DUP_MARKERS: dropped {dropped_dupmark:,} entries already drawn by a plain entry; "
          f"kept {n_dupmark_kept:,} that were the only entry for their city (parenthetical stripped)")
    print(f"dropped: {dropped_variant:,} metro-variants, {dropped_dup:,} duplicates, "
          f"{dropped_nocoord:,} no-coord, {dropped_empty:,} empty, "
          f"{dropped_small:,} below peak floor {PEAK_FLOOR}")
    print(f"ramped/clipped {clipped_newworld:,} New World antiquity points (cap ramp "
          f"{NW_RAMP_START}..{NW_RAMP_FULL}); kept {kept_newworld_benchmark} verbatim Chandler "
          f"benchmarks the ramp would have flattened")
    print(f"dropped {n_year0} year-0 rows (no such year; see YEAR_ZERO)")
    print(f"repaired {n_coordfix:,} geocoder-fallback coordinates "
          f"({n_sitecoord} typed by hand from SITE_COORDS); "
          f"stripped carry-forward from {n_carryfwd:,} cities; clipped {n_clipped:,} pre-merge points; "
          f"took {n_variant:,} metro-variant series ({n_agglom:,} automatic agglomeration splices)")
    print(f"removed {despiked_pts:,} definition spikes from {n_despiked:,} cities "
          f"(up-and-back >={OSC_AMP}x inside {OSC_SPAN}y, ends agreeing within {OSC_AGREE}x)")
    # every deletion is a judgement call, so `--spikes` prints the lot with the neighbours
    # that justified it; the default shows only the biggest.
    for v, y, n, y0, v0, y2, v2 in sorted(despike_log, reverse=True)[:999 if "--spikes" in sys.argv else 12]:
        print(f"     {n:<22} {y0}:{v0:>9,.0f} -> [{y} {v:>9,.0f}] -> {y2}:{v2:>9,.0f}"
              f"   ({y2-y0}y, {min(v/v0, v/v2):.1f}x)")
    # fades: which cities got zeroed out across a gap, bucketed by where the gap starts. The
    # era split matters because the threshold is in ADJUSTED years -- `--fades` prints the lot.
    if FADES_ON:
        eras = [("pre-AD1", -9999, 1), ("1-1400", 1, 1400), ("1400-1900", 1400, 1900),
                ("1900+", 1900, 9999)]
        buck = {e[0]: 0 for e in eras}
        for _, y0, _, _ in FADE_LOG:
            for nm, a, b in eras:
                if a <= y0 < b:
                    buck[nm] += 1
        arm = "" if FADE_GAP_REAL is None else f" OR >= {FADE_GAP_REAL} real yr"
        print(f"faded {len(FADE_LOG):,} gaps (gap > {FADE_GAP} adjusted yr{arm} and >= "
              f"{FADE_GAP_MIN_REAL} real yr, ramp {FADE_YEARS} adjusted yr): "
              + ", ".join(f"{buck[nm]} starting {nm}" for nm, _, _ in eras))
        for n, y0, y1, v in sorted(FADE_LOG, key=lambda e: -e[3])[:999 if "--fades" in sys.argv else 12]:
            print(f"     {n:<26} {y0:>6} -> {y1:<6} ({y1-y0:>4}y real, "
                  f"{adj(y1)-adj(y0):>6,.0f}y adj)  peak {v:,.0f}")
    # named disappearances: the case-by-case half. `NOT APPLIED` means the table names a city
    # or a span the build never reached -- a renamed source key, or a gap that turned out to
    # have an anchor inside it -- and is a prompt to fix the table, not a warning to live with.
    if DISAPPEARED:
        seen = {e[0] for e in DISAPPEARED_LOG}
        miss = [k for k in DISAPPEARED if k not in seen]
        short = [(k, p, n) for k, _, p, n, _ in DISAPPEARED_LOG if p < n]
        print(f"named disappearances: {len(seen)}/{len(DISAPPEARED)} cities reached, "
              f"{sum(e[2] for e in DISAPPEARED_LOG)} spans planted, "
              f"{sum(e[1] for e in DISAPPEARED_LOG)} generic floor points removed")
        if miss:
            print(f"     NOT REACHED (renamed or dropped key?): {', '.join(miss)}")
        for k, p, n in short:
            print(f"     NOT APPLIED: {k} planted {p}/{n} spans -- a real anchor sits inside one")
    if LONE_ANCHOR_LOG:
        print(f"lone anchors bracketed: {len(LONE_ANCHOR_LOG)} pre-{LONE_ANCHOR_BEFORE} cities "
              f"whose whole record is one benchmark, drawn in a single frame before this: "
              + ", ".join(f"{n} ({y})" for n, y, _ in
                          sorted(LONE_ANCHOR_LOG, key=lambda t: -t[2])[:6]))
        for k, strip, p, n, out in sorted(DISAPPEARED_LOG):
            fl = [y for y, v in out if v == FADE_FLOOR]
            print(f"     {k[:28]:28s} -{strip} generic, +{p} named"
                  + (f"   dark {fl[0]}..{fl[-1]}" if fl else "   no floor (continuous)"))
    print(f"control points: {total_pts_before:,} -> {total_pts_after:,} "
          f"({total_pts_after/max(total_pts_before,1)*100:.1f}%), avg {total_pts_after/len(cities):.1f}/city")
    if TRIMS:
        keep_hi = sum(1 for _, k, d, _ in TRIMS if k > d)
        print(f"terminal unit-switch: resolved {len(TRIMS):,} two-definition tails "
              f"({keep_hi:,} kept the held plateau, {len(TRIMS)-keep_hi:,} kept the final census)")
        for nm, kept, drop, n in sorted(TRIMS, key=lambda t: -max(t[1], t[2]))[:12]:
            print(f"     {nm[:24]:24s} kept {kept:12,.0f}  dropped {drop:12,.0f}  ({n} pts)")
    if LOCAL_TRIMS:
        print(f"  ...of which {len(LOCAL_TRIMS):,} had no WUP row and were arbitrated against "
              f"the entry's own record (plateau dropped; never the final row)")
        for nm, kept, drop, n in sorted(LOCAL_TRIMS, key=lambda t: -t[2])[
                :999 if "--trims" in sys.argv else 12]:
            print(f"     {nm[:24]:24s} kept {kept:12,.0f}  dropped {drop:12,.0f}  ({n} pts)")
    print(f"held forward to {YEAR_NOW}: {n_extended:,} cities whose record stops in "
          f"{EXTEND_FROM}..{YEAR_NOW-1} (drawn as no-data)")
    if ADMIN_TRIMS:
        print(f"China administrative tails trimmed: {len(ADMIN_TRIMS):,} "
              f"(terminal >= {CHINA_ADMIN_RATIO}x the WUP handover)")
        for nm, drop, kept, n in sorted(ADMIN_TRIMS, key=lambda t: -t[1])[
                :999 if "--admin" in sys.argv else 12]:
            print(f"     {nm[:24]:24s} dropped {drop:12,.0f}  back to {kept:12,.0f}  ({n} pts)")
    # Surface disagreements between the coordinate and the region tables, so a source update
    # that adds a country we do not classify shows up instead of silently changing behaviour.
    # The standing entries are Pacific islands (correctly NOT the Americas) plus a handful of
    # mis-geocoded ones; a NEW name appearing here is the signal.
    if region_conflicts:
        top = sorted(region_conflicts.items(), key=lambda kv: -kv[1])
        print(f"region check: {sum(region_conflicts.values()):,} entries lie west of lon -30 but "
              f"are not in AMERICAS -- " + ", ".join(f"{c}({n})" for c, n in top[:10]))
    report_switch_steps()
    print(f"wrote {OUT}  ({sz:.2f} MB)")


if __name__ == "__main__":
    main()
