"""
Scatter ancestry dots within census tract polygons.
Reads raw ACS CSVs + cartographic boundary shapefiles, outputs GeoJSON.

Tract shapefiles are the cb_ (cartographic boundary) versions, pre-clipped to the
coastline. Pass --areawater to also subtract inland water bodies (rivers, lakes, bays).
Download shapefiles first with download_shapefiles.py.

Usage:
    python scatter_dots.py --state 36 --scale 100               # NYC, coast-clipped
    python scatter_dots.py --state 36 --scale 100 --areawater   # also clip inland water
    python scatter_dots.py --all --scale 1000 --areawater       # nationwide

Outputs to data/processed/dots_<state|all>_1per<scale>.geojson
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

# Maps B04006 variable codes → (label, color_group)
# Source: https://api.census.gov/data/2024/acs/acs5/groups/B04006.json
# NOTE: Skip parent/subtotal rows (_006E Arab total, _073E Subsaharan African total,
#       _094E West Indian total) to avoid double-counting with their sub-categories.
# NOTE: Asian, Latino, Native American, and Pacific Islander are NOT in B04006 —
#       they're captured by race/Hispanic origin questions. Pull from B02015 + B03001.
ANCESTRY_GROUPS = {
    # --- Western European → blue (Germany/Italy/Finland and west) ---
    "B04006_004E": ("Alsatian", "western"),
    "B04006_018E": ("Australian", "western"),
    "B04006_019E": ("Austrian", "western"),
    "B04006_020E": ("Basque", "western"),
    "B04006_021E": ("Belgian", "western"),
    "B04006_023E": ("British", "western"),
    "B04006_028E": ("Celtic", "western"),
    "B04006_033E": ("Danish", "western"),
    "B04006_034E": ("Dutch", "western"),
    "B04006_036E": ("English", "western"),
    "B04006_038E": ("European", "western"),
    "B04006_039E": ("Finnish", "western"),
    "B04006_040E": ("French", "western"),
    "B04006_042E": ("German", "western"),
    "B04006_047E": ("Icelander", "western"),
    "B04006_049E": ("Irish", "western"),
    "B04006_051E": ("Italian", "western"),
    "B04006_054E": ("Luxembourger", "western"),
    "B04006_056E": ("Maltese", "western"),
    "B04006_058E": ("Northern European", "western"),
    "B04006_059E": ("Norwegian", "western"),
    "B04006_062E": ("Portuguese", "western"),
    "B04006_065E": ("Scandinavian", "western"),
    "B04006_066E": ("Scotch-Irish", "western"),
    "B04006_067E": ("Scottish", "western"),
    "B04006_089E": ("Swedish", "western"),
    "B04006_090E": ("Swiss", "western"),
    "B04006_093E": ("Welsh", "western"),

    # --- Eastern European → sky blue (east of Germany/Italy/Finland line) ---
    "B04006_003E": ("Albanian", "eastern"),
    "B04006_024E": ("Bulgarian", "eastern"),
    "B04006_027E": ("Carpatho Rusyn", "eastern"),
    "B04006_029E": ("Croatian", "eastern"),
    "B04006_031E": ("Czech", "eastern"),
    "B04006_032E": ("Czechoslovakian", "eastern"),
    "B04006_035E": ("Eastern European", "eastern"),
    "B04006_037E": ("Estonian", "eastern"),
    "B04006_043E": ("German Russian", "eastern"),
    "B04006_044E": ("Greek", "eastern"),
    "B04006_046E": ("Hungarian", "eastern"),
    "B04006_052E": ("Latvian", "eastern"),
    "B04006_053E": ("Lithuanian", "eastern"),
    "B04006_055E": ("Macedonian", "eastern"),
    "B04006_061E": ("Polish", "eastern"),
    "B04006_063E": ("Romanian", "eastern"),
    "B04006_064E": ("Russian", "eastern"),
    "B04006_068E": ("Serbian", "eastern"),
    "B04006_069E": ("Slavic", "eastern"),
    "B04006_070E": ("Slovak", "eastern"),
    "B04006_071E": ("Slovene", "eastern"),
    "B04006_072E": ("Soviet Union", "eastern"),
    "B04006_092E": ("Ukrainian", "eastern"),
    "B04006_107E": ("Yugoslavian", "eastern"),

    # --- American → indigo (New World identity) ---
    "B04006_005E": ("American", "american"),
    "B04006_025E": ("Cajun", "american"),
    "B04006_026E": ("Canadian", "american"),
    "B04006_041E": ("French Canadian", "american"),
    "B04006_060E": ("Pennsylvania German", "american"),

    # --- MENA → purple ---
    # B04006_006E is the Arab subtotal — skipped to avoid double-counting sub-categories below
    "B04006_007E": ("Egyptian", "mena"),
    "B04006_008E": ("Iraqi", "mena"),
    "B04006_009E": ("Jordanian", "mena"),
    "B04006_010E": ("Lebanese", "mena"),
    "B04006_011E": ("Moroccan", "mena"),
    "B04006_012E": ("Palestinian", "mena"),
    "B04006_013E": ("Syrian", "mena"),
    "B04006_014E": ("Arab", "mena"),           # generic "Arab" responses
    "B04006_015E": ("Other Arab", "mena"),
    "B04006_016E": ("Armenian", "mena"),
    "B04006_017E": ("Assyrian/Chaldean/Syriac", "mena"),
    "B04006_048E": ("Iranian", "mena"),
    "B04006_050E": ("Israeli", "mena"),
    "B04006_091E": ("Turkish", "mena"),

    # --- Subsaharan African → blue-green ---
    # B04006_073E is the Subsaharan subtotal — skipped
    "B04006_074E": ("Cape Verdean", "african"),
    "B04006_075E": ("Ethiopian", "african"),
    "B04006_076E": ("Ghanaian", "african"),
    "B04006_077E": ("Kenyan", "african"),
    "B04006_078E": ("Liberian", "african"),
    "B04006_079E": ("Nigerian", "african"),
    "B04006_080E": ("Senegalese", "african"),
    "B04006_081E": ("Sierra Leonean", "african"),
    "B04006_082E": ("Somali", "african"),
    "B04006_083E": ("South African", "african"),
    "B04006_084E": ("Sudanese", "african"),
    "B04006_085E": ("Ugandan", "african"),
    "B04006_086E": ("Zimbabwean", "african"),
    "B04006_087E": ("African", "african"),     # generic "African" responses
    "B04006_088E": ("Other Subsaharan African", "african"),
    # --- Afro-Caribbean → yellow-green ---
    # B04006_094E is the West Indian subtotal — skipped
    "B04006_095E": ("Bahamian", "afro_carib"),
    "B04006_096E": ("Barbadian", "afro_carib"),
    "B04006_097E": ("Belizean", "afro_carib"),
    "B04006_098E": ("Bermudan", "afro_carib"),
    "B04006_099E": ("British West Indian", "afro_carib"),
    "B04006_100E": ("Dutch West Indian", "afro_carib"),
    "B04006_101E": ("Haitian", "afro_carib"),
    "B04006_102E": ("Jamaican", "afro_carib"),
    "B04006_103E": ("Trinidadian and Tobagonian", "afro_carib"),
    "B04006_104E": ("U.S. Virgin Islander", "afro_carib"),
    "B04006_105E": ("West Indian", "afro_carib"),
    "B04006_106E": ("Other West Indian", "afro_carib"),

    # --- South & Central Asian (partial — B02015 fills the rest) → purple ---
    "B04006_002E": ("Afghan", "s_c_asian"),

    # --- Guyanese (mixed Indo/Afro — mapped afro_carib as Caribbean context) ---
    "B04006_045E": ("Guyanese", "afro_carib"),

    # --- Brazilian → latino (self-identification pattern on census) ---
    "B04006_022E": ("Brazilian", "latino"),

    # Asian, Latino, Native American, Pacific Islander:
    # not present in B04006 — covered by B02015 + B02016 + B02020 + B03001 below
}

# Maps B02015 variable codes → (label, color_group)
# Source: https://api.census.gov/data/2024/acs/acs5/groups/B02015.json
# Skipped: _001E (total), _035E (Two or more Asian)
# Note: _029E Afghan overlaps with B04006_002E — omitted here to avoid double-counting
B02015_GROUPS = {
    # --- East Asian → orangey red ---
    "B02015_002E": ("Chinese", "east_asian"),
    "B02015_003E": ("Hmong", "east_asian"),
    "B02015_004E": ("Japanese", "east_asian"),
    "B02015_005E": ("Korean", "east_asian"),
    "B02015_006E": ("Mongolian", "east_asian"),
    "B02015_007E": ("Okinawan", "east_asian"),
    "B02015_008E": ("Taiwanese", "east_asian"),
    "B02015_009E": ("Other East Asian", "east_asian"),
    # --- Southeast Asian → magenta/red-purple ---
    "B02015_010E": ("Burmese", "se_asian"),
    "B02015_011E": ("Cambodian", "se_asian"),
    "B02015_012E": ("Filipino", "se_asian"),
    "B02015_013E": ("Indonesian", "se_asian"),
    "B02015_014E": ("Laotian", "se_asian"),
    "B02015_015E": ("Malaysian", "se_asian"),
    "B02015_016E": ("Mien", "se_asian"),
    "B02015_017E": ("Singaporean", "se_asian"),
    "B02015_018E": ("Thai", "se_asian"),
    "B02015_019E": ("Vietnamese", "se_asian"),
    "B02015_020E": ("Other Southeast Asian", "se_asian"),
    # --- South & Central Asian → purple ---
    "B02015_021E": ("Indian", "s_c_asian"),
    "B02015_022E": ("Bangladeshi", "s_c_asian"),
    "B02015_023E": ("Bhutanese", "s_c_asian"),
    "B02015_024E": ("Nepalese", "s_c_asian"),
    "B02015_025E": ("Pakistani", "s_c_asian"),
    "B02015_026E": ("Sikh", "s_c_asian"),
    "B02015_027E": ("Sri Lankan", "s_c_asian"),
    "B02015_028E": ("Other South Asian", "s_c_asian"),
    # B02015_029E Afghan omitted — already in B04006_002E
    "B02015_030E": ("Kazakh", "s_c_asian"),
    "B02015_031E": ("Uzbek", "s_c_asian"),
    "B02015_032E": ("Other Central Asian", "s_c_asian"),
    # --- Other ---
    "B02015_033E": ("Other Asian, specified", "east_asian"),
    "B02015_034E": ("Other Asian, not specified", "east_asian"),
}

# Maps B03001 variable codes → (label, color_group)
# Source: https://api.census.gov/data/2024/acs/acs5/groups/B03001.json
# Skipped: _001E (total), _002E (Not Hispanic or Latino), _003E (Hispanic total),
#          _008E (Central American total), _016E (South American total), _027E (Other Hispanic total)
# Note: Spaniard/Spanish kept as latino — they self-identify Hispanic on census form
B03001_GROUPS = {
    "B03001_004E": ("Mexican", "latino"),
    "B03001_005E": ("Puerto Rican", "latino"),
    "B03001_006E": ("Cuban", "latino"),
    "B03001_007E": ("Dominican", "latino"),
    # Central American sub-categories
    "B03001_009E": ("Costa Rican", "latino"),
    "B03001_010E": ("Guatemalan", "latino"),
    "B03001_011E": ("Honduran", "latino"),
    "B03001_012E": ("Nicaraguan", "latino"),
    "B03001_013E": ("Panamanian", "latino"),
    "B03001_014E": ("Salvadoran", "latino"),
    "B03001_015E": ("Other Central American", "latino"),
    # South American sub-categories
    "B03001_017E": ("Argentinean", "latino"),
    "B03001_018E": ("Bolivian", "latino"),
    "B03001_019E": ("Chilean", "latino"),
    "B03001_020E": ("Colombian", "latino"),
    "B03001_021E": ("Ecuadorian", "latino"),
    "B03001_022E": ("Paraguayan", "latino"),
    "B03001_023E": ("Peruvian", "latino"),
    "B03001_024E": ("Uruguayan", "latino"),
    "B03001_025E": ("Venezuelan", "latino"),
    "B03001_026E": ("Other South American", "latino"),
    # Other Hispanic
    "B03001_028E": ("Spaniard", "western"),
    "B03001_029E": ("Spanish", "latino"),
    "B03001_030E": ("Spanish American", "latino"),
    "B03001_031E": ("Other Hispanic or Latino", "latino"),
}

# Maps B02016 variable codes → (label, color_group)
# Source: https://api.census.gov/data/2024/acs/acs5/groups/B02016.json
# Skipped: _001E (total), _014E (Two or more Pacific Islander)
B02016_GROUPS = {
    "B02016_002E": ("Native Hawaiian", "pacific"),
    "B02016_003E": ("Samoan", "pacific"),
    "B02016_004E": ("Tongan", "pacific"),
    "B02016_005E": ("Other Polynesian", "pacific"),
    "B02016_006E": ("Chamorro", "pacific"),
    "B02016_007E": ("Chuukese", "pacific"),
    "B02016_008E": ("Guamanian", "pacific"),
    "B02016_009E": ("Marshallese", "pacific"),
    "B02016_010E": ("Other Micronesian", "pacific"),
    "B02016_011E": ("Fijian", "pacific"),
    "B02016_012E": ("Other Melanesian", "pacific"),
    "B02016_013E": ("Other Pacific Islander", "pacific"),
}

# Maps B02020 variable codes → (label, color_group)
# Source: https://api.census.gov/data/2024/acs/acs5/groups/B02020.json
# Skipped: _001E (total), _002E/_023E/_031E (subtotals), _041E (Two or more AIAN)
B02020_GROUPS = {
    "B02020_003E": ("Blackfeet", "native"),
    "B02020_004E": ("Cherokee", "native"),
    "B02020_005E": ("Cheyenne River Sioux", "native"),
    "B02020_006E": ("Comanche", "native"),
    "B02020_007E": ("Crow", "native"),
    "B02020_008E": ("Gila River", "native"),
    "B02020_009E": ("Hopi", "native"),
    "B02020_010E": ("Lumbee", "native"),
    "B02020_011E": ("Navajo", "native"),
    "B02020_012E": ("Oglala Sioux", "native"),
    "B02020_013E": ("Pascua Yaqui", "native"),
    "B02020_014E": ("Rosebud Sioux", "native"),
    "B02020_015E": ("Chickasaw", "native"),
    "B02020_016E": ("Choctaw", "native"),
    "B02020_017E": ("Muscogee (Creek)", "native"),
    "B02020_018E": ("Tohono O'odham", "native"),
    "B02020_019E": ("Turtle Mountain Chippewa", "native"),
    "B02020_020E": ("White Mountain Apache", "native"),
    "B02020_021E": ("All other American Indian", "native"),
    "B02020_022E": ("American Indian, not specified", "native"),
    "B02020_024E": ("Aztec", "native"),
    "B02020_025E": ("Inca", "native"),
    "B02020_026E": ("Maya", "native"),
    "B02020_027E": ("Mixtec", "native"),
    "B02020_028E": ("Taino", "native"),
    "B02020_029E": ("Tarasco (Purepecha)", "native"),
    "B02020_030E": ("All other Latin American Indian", "native"),
    "B02020_032E": ("Tlingit and Haida", "native"),
    "B02020_033E": ("Metlakatla", "native"),
    "B02020_034E": ("Inupiat (Barrow)", "native"),
    "B02020_035E": ("Yup'ik (Hooper Bay)", "native"),
    "B02020_036E": ("Inupiat (Kotzebue)", "native"),
    "B02020_037E": ("Nome Eskimo Community", "native"),
    "B02020_038E": ("All other Alaska Native", "native"),
    "B02020_039E": ("Alaska Native, not specified", "native"),
    "B02020_040E": ("AIAN, not specified", "native"),
}

# Combined lookup: table name → its variable mapping
ALL_TABLES = {
    "B04006": ANCESTRY_GROUPS,
    "B02015": B02015_GROUPS,
    "B02016": B02016_GROUPS,
    "B02020": B02020_GROUPS,
    "B03001": B03001_GROUPS,
}

# Residual labels: people captured by the race question but not by any ancestry write-in.
# B02009 = Black or African American alone or in any combination
# B02008 = White alone or in any combination
# We subtract the already-represented ancestry groups from each race total to avoid double-counting.
RESIDUAL_LABELS = {
    "Black, no ancestry reported": "no_ancestry",
    "White, no ancestry reported": "no_ancestry",
}
BLACK_SUBTRACT_GROUPS = {"african", "afro_carib"}
WHITE_SUBTRACT_GROUPS = {"western", "eastern", "american"}

# Subgroup color definitions: subgroup_key → (hue, sat, lit).
SUBGROUP_COLORS: dict = {
    # western
    "uk_ireland":        (204, 0.7, 0.55),
    "nordic":            (210, 0.7, 0.70),
    "france_benelux":    (218, 0.7, 0.58),
    "germany_rest":      (230, 0.7, 0.60),
    # eastern
    "north_balkans":     (240, 0.7, 0.58),
    "balkan":            (250, 0.7, 0.53),
    "russia_ukraine":    (250, 0.7, 0.64),
    # latino
    "central_american":  (62,  0.7, 0.55),
    "south_american":    (50,  0.7, 0.55),
    # s_c_asian
    "south_asian":       (295, 0.698, 0.649),
    "central_asian":     (295, 0.698, 0.709),
    # residuals (special saturation)
    "white_unspecified": (220, 0.2, 0.70),
    "black_unspecified": (180, 0.2, 0.40),
}

# Maps each label to its subgroup key. Labels absent here get the group center color.
LABEL_SUBGROUP: dict = {
    # western: UK + Ireland (and Australian, Celtic)
    "Australian":  "uk_ireland",
    "British":     "uk_ireland",
    "Celtic":      "uk_ireland",
    "English":     "uk_ireland",
    "Irish":       "uk_ireland",
    "Scotch-Irish":"uk_ireland",
    "Scottish":    "uk_ireland",
    "Welsh":       "uk_ireland",
    # western: Nordic
    "Danish":           "nordic",
    "Finnish":          "nordic",
    "Icelander":        "nordic",
    "Northern European":"nordic",
    "Norwegian":        "nordic",
    "Scandinavian":     "nordic",
    "Swedish":          "nordic",
    # western: France / Spain / Benelux
    "Alsatian":     "france_benelux",
    "Basque":       "france_benelux",
    "Belgian":      "france_benelux",
    "Dutch":        "france_benelux",
    "French":       "france_benelux",
    "Luxembourger": "france_benelux",
    "Portuguese":   "france_benelux",
    "Spaniard":     "france_benelux",
    # western: Germany + rest
    "Austrian": "germany_rest",
    "European": "germany_rest",
    "German":   "germany_rest",
    "Italian":  "germany_rest",
    "Maltese":  "germany_rest",
    "Swiss":    "germany_rest",
    # eastern: North of Balkans (Poland, Hungary, Czech, Baltics)
    "Czech":            "north_balkans",
    "Eastern European": "north_balkans",
    "Estonian":         "north_balkans",
    "Hungarian":        "north_balkans",
    "Latvian":          "north_balkans",
    "Lithuanian":       "north_balkans",
    "Polish":           "north_balkans",
    "Slovak":           "north_balkans",
    # eastern: Balkans
    "Albanian":    "balkan",
    "Bulgarian":   "balkan",
    "Croatian":    "balkan",
    "Greek":       "balkan",
    "Macedonian":  "balkan",
    "Romanian":    "balkan",
    "Serbian":     "balkan",
    "Slovene":     "balkan",
    "Yugoslavian": "balkan",
    # eastern: Russia / Ukraine
    "Russian":   "russia_ukraine",
    "Ukrainian": "russia_ukraine",
    # latino: Central American
    "Costa Rican":          "central_american",
    "Guatemalan":           "central_american",
    "Honduran":             "central_american",
    "Mexican":              "central_american",
    "Nicaraguan":           "central_american",
    "Other Central American":"central_american",
    "Panamanian":           "central_american",
    "Salvadoran":           "central_american",
    # latino: South American
    "Argentinean":        "south_american",
    "Bolivian":           "south_american",
    "Brazilian":          "south_american",
    "Chilean":            "south_american",
    "Colombian":          "south_american",
    "Ecuadorian":         "south_american",
    "Other South American":"south_american",
    "Paraguayan":         "south_american",
    "Peruvian":           "south_american",
    "Uruguayan":          "south_american",
    "Venezuelan":         "south_american",
    # s_c_asian: South Asian
    "Afghan":          "south_asian",
    "Bangladeshi":     "south_asian",
    "Bhutanese":       "south_asian",
    "Indian":          "south_asian",
    "Nepalese":        "south_asian",
    "Other South Asian":"south_asian",
    "Pakistani":       "south_asian",
    "Sikh":            "south_asian",
    "Sri Lankan":      "south_asian",
    # s_c_asian: Central Asian
    "Kazakh":           "central_asian",
    "Other Central Asian":"central_asian",
    "Uzbek":            "central_asian",
    # residuals
    "White, no ancestry reported": "white_unspecified",
    "Black, no ancestry reported": "black_unspecified",
}

GROUPS_FILE  = Path(__file__).parent / "groups.json"
COLORS_FILE  = Path(__file__).parent / "ancestry_colors.csv"
LEGEND_FILE  = Path(__file__).parent / "data" / "processed" / "legend.json"


def load_groups() -> dict:
    with open(GROUPS_FILE) as f:
        return json.load(f)["groups"]


def build_ancestry_colors(groups: dict) -> dict:
    """Return {label: hex_color} using SUBGROUPS colors where defined, group center otherwise."""
    all_labels: dict = {}
    for _code, (label, group) in {**ANCESTRY_GROUPS, **B02015_GROUPS, **B02016_GROUPS, **B02020_GROUPS, **B03001_GROUPS}.items():
        all_labels[label] = group
    for label, group in RESIDUAL_LABELS.items():
        all_labels[label] = group

    color_map: dict = {}
    for label, group_name in all_labels.items():
        subgroup = LABEL_SUBGROUP.get(label)
        if subgroup:
            hue, sat, lit = SUBGROUP_COLORS[subgroup]
        else:
            g = groups.get(group_name, {"hue": 0, "lit": 0.55})
            hue = g["hue"]
            sat = 0.7
            lit = g["lit"]
        if group_name != "no_ancestry":
            lit = _equalized_lit(hue, sat)
        color_map[label] = hue_to_hex(hue, sat, lit)
    return color_map


def _hsl_luminance(hue: float, sat: float, lit: float) -> float:
    """Relative luminance of an HSL color (hue in degrees)."""
    import colorsys
    r, g, b = colorsys.hls_to_rgb(hue / 360, lit, sat)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

# Reference luminance: blue (240°) at S=0.7, L=0.6 — the darkest of our hues.
# Other hues get their lit pulled down to match, then blended halfway toward 0.5.
_TARGET_LUM = _hsl_luminance(240, 0.7, 0.6)

def _equalized_lit(hue: float, sat: float) -> float:
    """Find L so HSL(hue, sat, L) has the same luminance as blue at L=0.6,
    then blend halfway toward 0.5 (half-strength correction)."""
    lo, hi = 0.1, 0.95
    for _ in range(50):
        mid = (lo + hi) / 2
        if _hsl_luminance(hue, sat, mid) < _TARGET_LUM:
            lo = mid
        else:
            hi = mid
    equalized = (lo + hi) / 2
    return (equalized + 2 * 0.5) / 3  # 1/3-strength: blend toward 0.5


def hue_to_hex(hue: float, sat: float, lit: float) -> str:
    import colorsys
    r, g, b = colorsys.hls_to_rgb(hue / 360, lit, sat)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


_GROUP_ORDER    = ['western', 'eastern', 'american', 'east_asian', 'se_asian', 's_c_asian', 'african', 'afro_carib', 'latino', 'mena', 'native', 'pacific', 'no_ancestry']
_SUBGROUP_ORDER = ['uk_ireland', 'nordic', 'france_benelux', 'germany_rest', 'north_balkans', 'balkan', 'russia_ukraine', 'central_american', 'south_american', 'south_asian', 'central_asian', 'white_unspecified', 'black_unspecified']


def write_colors_csv(color_map: dict, group_map: dict = None, pop_map: dict = None):
    group_map = group_map or {}
    pop_map   = pop_map   or {}
    group_rank = {g: i for i, g in enumerate(_GROUP_ORDER)}

    def csv_label(s):
        if ',' in s or '"' in s:
            return '"' + s.replace('"', '""') + '"'
        return s

    rows = []
    for label, hex_color in color_map.items():
        import colorsys
        r = int(hex_color[1:3], 16) / 255
        g = int(hex_color[3:5], 16) / 255
        b = int(hex_color[5:7], 16) / 255
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        rows.append((group_map.get(label, ""), label, round(h * 360, 1), round(s, 3), round(l, 3), pop_map.get(label, 0)))

    rows.sort(key=lambda row: (group_rank.get(row[0], 99), row[1].lower()))

    lw = max(len(csv_label(row[1])) for row in rows)
    gw = max(len(row[0]) for row in rows)
    pw = max(len(f"{row[5]:,}") for row in rows)

    def fmt(group, label, hue, sat, lit, pop):
        lbl = csv_label(label)
        return f"{lbl:<{lw}}, {group:<{gw}}, {hue:>5}, {sat:>5}, {lit:>5}, {pop:>{pw},}"

    with open(COLORS_FILE, "w", newline="\n") as f:
        f.write(f"{'label':<{lw}}, {'group':<{gw}}, {'hue':>5}, {'sat':>5}, {'lit':>5}, {'population':>{pw}}\n")
        for row in rows:
            f.write(fmt(*row) + "\n")


def write_legend_json(color_map: dict, pop_map: dict = None):
    """Write legend.json from the color map (label → hex). Reads group from ALL_TABLES + RESIDUAL_LABELS, overridden by CSV."""
    import csv as _csv
    pop_map = pop_map or {}
    all_var_map = {**ANCESTRY_GROUPS, **B02015_GROUPS, **B02016_GROUPS, **B02020_GROUPS, **B03001_GROUPS}
    group_map = {info[0]: info[1] for info in all_var_map.values()}
    group_map.update(RESIDUAL_LABELS)
    # CSV group column takes precedence over hardcoded tables
    with open(COLORS_FILE, newline="") as f:
        reader = _csv.DictReader(f)
        reader.fieldnames = [n.strip() for n in reader.fieldnames]
        for row in reader:
            label = row["label"].strip()
            group = row.get("group", "").strip()
            if group:
                group_map[label] = group
    group_rank    = {g: i for i, g in enumerate(_GROUP_ORDER)}
    subgroup_rank = {s: i for i, s in enumerate(_SUBGROUP_ORDER)}

    items = []
    for label, hex_color in color_map.items():
        subgroup = LABEL_SUBGROUP.get(label, "")
        items.append({
            "label":      label,
            "group":      group_map.get(label, ""),
            "subgroup":   subgroup,
            "color":      hex_color,
            "population": pop_map.get(label, 0),
        })
    items.sort(key=lambda x: (
        group_rank.get(x["group"], 99),
        subgroup_rank.get(x["subgroup"], 99),
        x["label"].lower(),
    ))

    LEGEND_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LEGEND_FILE, "w") as f:
        json.dump(items, f)
    print(f"Wrote {len(items)} entries to {LEGEND_FILE}")


def _pop_map_from_geojsons(scale: int) -> dict:
    """Aggregate label counts across all processed dot GeoJSONs, return {label: population}.
    Seeds from CSV population column as a baseline so labels absent from the geojson
    (e.g. no_ancestry residuals) still carry their correct population."""
    import csv as csv_mod
    from collections import Counter
    # Baseline: read populations from ancestry_colors.csv
    pop_map = {}
    if COLORS_FILE.exists():
        with open(COLORS_FILE, newline="") as f:
            reader = csv_mod.DictReader(f)
            reader.fieldnames = [n.strip() for n in reader.fieldnames]
            for row in reader:
                label = row["label"].strip().strip('"')
                try:
                    pop_map[label] = int(row["population"].strip().replace(",", ""))
                except (ValueError, KeyError):
                    pass
    # Override with actual dot counts from geojson where available
    processed = Path(__file__).parent / "data" / "processed"
    total: Counter = Counter()
    candidates = sorted(processed.glob("dots_*_1per*.geojson"))
    all_file = processed / f"dots_all_1per{scale}.geojson"
    if all_file.exists():
        candidates = [all_file]
    for path in candidates:
        with open(path) as f:
            data = json.load(f)
        total += Counter(feat["properties"]["label"] for feat in data["features"])
    for label, count in total.items():
        pop_map[label] = max(count * scale, pop_map.get(label, 0))
    return pop_map


def load_colors_csv() -> dict:
    import colorsys, csv
    color_map = {}
    with open(COLORS_FILE, newline="") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        for row in reader:
            label = row["label"].strip()
            h = float(row["hue"].strip()) / 360
            s = float(row["sat"].strip())
            l = float(row["lit"].strip()) if "lit" in row else float(row["val"].strip())
            r, g, b = colorsys.hls_to_rgb(h, l, s)
            color_map[label] = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
    return color_map


def random_points_in_polygon(polygon, n: int) -> list:
    """Generate n random points inside polygon using numpy + shapely vectorized contains."""
    if n == 0:
        return []
    minx, miny, maxx, maxy = polygon.bounds
    pts: list = []
    needed = n
    while needed > 0:
        batch = max(needed * 4, 32)
        xs = np.random.uniform(minx, maxx, batch)
        ys = np.random.uniform(miny, maxy, batch)
        mask = shapely.contains_xy(polygon, xs, ys)
        valid_xs, valid_ys = xs[mask], ys[mask]
        take = min(needed, len(valid_xs))
        if take > 0:
            pts.extend(zip(valid_xs[:take].tolist(), valid_ys[:take].tolist()))
            needed -= take
    return pts if pts else [(polygon.centroid.x, polygon.centroid.y)] * n


def process_state(state_fips: str, scale: int, raw_dir: Path, shp_dir: Path, groups: dict, ancestry_colors: dict, use_areawater: bool = False) -> list:
    """Return list of GeoJSON feature dicts for one state."""
    # Use pre-clipped shapefile if available, otherwise fall back to coastline-only
    clipped_path = shp_dir / f"cb_2024_{state_fips}_tract_clipped.shp"
    standard_path = shp_dir / f"cb_2024_{state_fips}_tract_500k" / f"cb_2024_{state_fips}_tract_500k.shp"

    if use_areawater and clipped_path.exists():
        shp_path = clipped_path
    elif use_areawater:
        print(f"  Clipped shapefile not found — run download_shapefiles.py --areawater first. Falling back to coastline-only.")
        shp_path = standard_path
    else:
        shp_path = standard_path

    if not shp_path.exists():
        print(f"  Shapefile not found: {shp_path}")
        return []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*winding order.*")
        tracts = gpd.read_file(shp_path).to_crs(epsg=4326)
    tracts["geometry"] = tracts["geometry"].make_valid()
    tracts["GEOID"] = tracts["GEOID"].astype(str).str.zfill(11)

    # Load and merge all three ACS tables
    merged = tracts.copy()
    for table in ALL_TABLES:
        csv_path = raw_dir / f"{table}_{state_fips}.csv"
        if not csv_path.exists():
            print(f"  Warning: {csv_path} not found, skipping {table}")
            continue
        acs = pd.read_csv(csv_path, dtype=str)
        acs["GEOID"] = acs["state"].str.zfill(2) + acs["county"].str.zfill(3) + acs["tract"].str.zfill(6)
        acs = acs.drop(columns=["state", "county", "tract", "NAME", "GEO_ID"], errors="ignore")
        merged = merged.merge(acs, on="GEOID", how="left")

    # Melt to long form and filter to nonzero counts — much faster than iterrows
    all_var_map = {**ANCESTRY_GROUPS, **B02015_GROUPS, **B02016_GROUPS, **B02020_GROUPS, **B03001_GROUPS}
    present_cols = [c for c in all_var_map if c in merged.columns]
    long = merged[["GEOID", "geometry"] + present_cols].melt(
        id_vars=["GEOID", "geometry"], var_name="var_code", value_name="raw"
    )
    long["count"] = pd.to_numeric(long["raw"], errors="coerce").fillna(0).astype(int)
    long = long[long["count"] > 0].copy()
    if long.empty:
        return []
    long["label"] = long["var_code"].map(lambda c: all_var_map[c][0])
    long["color"] = long["label"].map(ancestry_colors)
    long["group"] = long["var_code"].map(lambda c: all_var_map[c][1])

    # --- Residual Black / White ---
    # For each tract: load the race-question total, subtract already-represented ancestry counts,
    # and emit the remainder as a generic "no ancestry reported" dot color.
    geoid_geom = merged.set_index("GEOID")["geometry"]
    for table, total_col, subtract_groups, res_label, res_group in [
        ("B02009", "B02009_001E", BLACK_SUBTRACT_GROUPS, "Black, no ancestry reported", "american"),
        ("B02008", "B02008_001E", WHITE_SUBTRACT_GROUPS, "White, no ancestry reported", "american"),
    ]:
        csv_path = raw_dir / f"{table}_{state_fips}.csv"
        if not csv_path.exists():
            continue
        race_df = pd.read_csv(csv_path, dtype=str)
        race_df["GEOID"] = race_df["state"].str.zfill(2) + race_df["county"].str.zfill(3) + race_df["tract"].str.zfill(6)
        race_counts = pd.to_numeric(
            race_df.set_index("GEOID")[total_col], errors="coerce"
        ).fillna(0).astype(int)

        already = long[long["group"].isin(subtract_groups)].groupby("GEOID")["count"].sum()
        residual = race_counts.subtract(already, fill_value=0).clip(lower=0).astype(int)
        residual = residual[(residual > 0) & residual.index.isin(geoid_geom.index)]
        if residual.empty:
            continue

        res_color = ancestry_colors.get(res_label, "#888888")
        res_rows = pd.DataFrame({
            "GEOID": residual.index,
            "geometry": residual.index.map(geoid_geom),
            "var_code": f"RESIDUAL_{table}",
            "raw": residual.values.astype(str),
            "count": residual.values,
            "label": res_label,
            "color": res_color,
            "group": res_group,
        })
        long = pd.concat([long, res_rows], ignore_index=True)

    # Generate dots with fractional carry-forward across tracts (sorted by GEOID = county-ordered).
    # Remainders < 1 dot accumulate across tracts so small populations aren't silently dropped.
    features = []
    remainders = {}  # var_code -> fractional dot remainder from previous tract
    for geoid, grp in long.groupby("GEOID"):
        geom = grp.iloc[0]["geometry"]
        if geom is None or geom.is_empty:
            continue

        dot_rows = []
        for _, row in grp.iterrows():
            vc = row["var_code"]
            total = row["count"] / scale + remainders.get(vc, 0.0)
            n = int(total)
            remainders[vc] = total - n
            if n > 0:
                dot_rows.append((n, row["label"], row["group"], row["color"]))

        if not dot_rows:
            continue

        total_dots = sum(r[0] for r in dot_rows)
        all_pts = random_points_in_polygon(geom, total_dots)
        idx = 0
        for n, label, group, color in dot_rows:
            for pt in all_pts[idx : idx + n]:
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": list(pt)},
                    "properties": {"label": label, "group": group, "color": color},
                })
            idx += n

    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", help="Single state FIPS (e.g. 36)")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--no-areawater", action="store_true", help="Skip inland water subtraction (faster, less accurate)")
    parser.add_argument("--write-colors", action="store_true", help="Write current algorithmic colors to ancestry_colors.csv and exit")
    parser.add_argument("--write-legend", action="store_true", help="Write legend.json from ancestry_colors.csv and exit")

    args = parser.parse_args()
    scale = 100

    if args.write_legend:
        colors = load_colors_csv()
        pop_map = _pop_map_from_geojsons(scale)
        write_legend_json(colors, pop_map)
        return

    if args.write_colors:
        groups = load_groups()
        colors = build_ancestry_colors(groups)
        all_var_map = {**ANCESTRY_GROUPS, **B02015_GROUPS, **B02016_GROUPS, **B02020_GROUPS, **B03001_GROUPS}
        group_map = {info[0]: info[1] for info in all_var_map.values()}
        group_map.update(RESIDUAL_LABELS)
        pop_map = _pop_map_from_geojsons(scale)
        write_colors_csv(colors, group_map, pop_map)
        print(f"Wrote {len(colors)} ancestry colors to {COLORS_FILE}")
        write_legend_json(load_colors_csv(), pop_map)
        return

    raw_dir = Path("data/raw")
    shp_dir = Path("data/shapefiles")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    from fetch_data import STATE_FIPS
    states = STATE_FIPS if args.all else [args.state.zfill(2)]
    tag = "all" if args.all else args.state.zfill(2)

    groups = {}
    if COLORS_FILE.exists():
        ancestry_colors = load_colors_csv()
    else:
        groups = load_groups()
        ancestry_colors = build_ancestry_colors(groups)

    use_areawater = not args.no_areawater
    all_features = []
    if len(states) > 1:
        from concurrent.futures import ProcessPoolExecutor
        print(f"Processing {len(states)} states in parallel...")
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(process_state, fips, scale, raw_dir, shp_dir, groups, ancestry_colors, use_areawater): fips
                for fips in states
            }
            for future, fips in futures.items():
                print(f"  State {fips} done")
                all_features.extend(future.result())
    else:
        for fips in states:
            print(f"Processing state {fips}...")
            all_features.extend(process_state(fips, scale, raw_dir, shp_dir, groups, ancestry_colors, use_areawater))

    # Sort so rarest ancestries (fewest total dots) are drawn last (on top).
    from collections import Counter
    label_counts = Counter(f["properties"]["label"] for f in all_features)
    all_features.sort(key=lambda f: label_counts[f["properties"]["label"]], reverse=True)

    out_path = out_dir / f"dots_{tag}_1per{scale}.geojson"
    geojson = {"type": "FeatureCollection", "features": all_features}
    with open(out_path, "w") as f:
        json.dump(geojson, f)
    print(f"Wrote {len(all_features)} dots to {out_path}")

    # Write legend.json — uses write_legend_json so subgroup sorting and no_ancestry group are included.
    pop_map = {lbl: label_counts[lbl] * scale for lbl in label_counts}
    write_legend_json(ancestry_colors, pop_map)


if __name__ == "__main__":
    main()
