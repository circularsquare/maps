"""
Classify the 250 StatCan DA-level ethnic/cultural origins into color groups and
generate ancestry_colors_ca.csv (first pass; hand-tune afterwards, like the US file).

Taxonomy mirrors the US map's 12 groups (groups.json), with:
  - `native`   relabeled "Indigenous" (First Nations, Métis, Inuit, Cree, ...)
  - `american` relabeled "Canadian & regional" (Canadian, Québécois, Acadian, ...)
  - `other`    NEW neutral-grey bucket for pan religion/race labels with no ethnicity
               (Christian n.i.e., Muslim, Buddhist, Caucasian n.o.s.).
Within each group, origins are spread across the group's hue_range ordered by
national population (rarest = edge of range), luminance-equalized like the US map.

Usage:
    python make_colors.py            # writes ancestry_colors_ca.csv + groups_ca.json
"""

from __future__ import annotations

import colorsys
import csv
import json
from collections import defaultdict
from pathlib import Path

from labels import canonical

HERE = Path(__file__).parent
US_COLORS = HERE.parent / "ancestry_colors.csv"  # the US map's hand-tuned palette

# ---- group definitions (center hue/sat/lit + spread), extends US groups.json ----
GROUPS = {
    "western":   {"label": "Western European",         "hue": 220, "sat": 0.70, "lit": 0.55, "hue_range": 45},
    "eastern":   {"label": "Eastern European",         "hue": 250, "sat": 0.76, "lit": 0.65, "hue_range": 28},
    "american":  {"label": "North American",           "hue": 191, "sat": 0.49, "lit": 0.46, "hue_range": 14},  # Canadian (top pop) lands ~hue184 = #3ca7af
    "east_asian":{"label": "East Asian",               "hue":  12, "sat": 0.70, "lit": 0.55, "hue_range": 22},
    "se_asian":  {"label": "Southeast Asian",          "hue": 320, "sat": 0.70, "lit": 0.60, "hue_range": 25},
    "s_c_asian": {"label": "South & Central Asian",    "hue": 295, "sat": 0.70, "lit": 0.65, "hue_range": 25},
    "mena":      {"label": "MENA",                     "hue": 270, "sat": 0.70, "lit": 0.65, "hue_range": 30},
    "latino":    {"label": "Latin American",           "hue":  55, "sat": 0.70, "lit": 0.55, "hue_range": 35},
    "native":    {"label": "Indigenous",               "hue":  30, "sat": 0.72, "lit": 0.55, "hue_range": 22},
    "african":   {"label": "African",                  "hue": 145, "sat": 0.70, "lit": 0.55, "hue_range": 32},
    "afro_carib":{"label": "West Indian",              "hue":  88, "sat": 0.70, "lit": 0.55, "hue_range": 25},
    "pacific":   {"label": "Pacific Islander",         "hue": 347, "sat": 0.70, "lit": 0.60, "hue_range": 18},
    "other":     {"label": "Other (Canada)",           "hue": 275, "sat": 0.18, "lit": 0.74, "hue_range": 44},
}

# --- targeted tweaks on top of US-palette matching ---
# Groups coloured UNIFORMLY (ignore US per-label colours; spread with the group's own
# light/pale params): North American stays near the light "Canadian" teal; Other is a
# pale desaturated purple.
CUSTOM_GROUP = {"american", "other"}
# Force these canonical labels into a different group than US/ORIGIN_GROUP would give.
GROUP_OVERRIDE = {"Asian": "east_asian", "New Zealander": "american", "Jewish": "other"}
# Francophone regional identities take the French colour instead of the pale-blue spread.
FRANCOPHONE = {"French Canadian", "Québécois", "Acadian", "Franco Ontarian"}
# French nudged a smidge purpler + more saturated (also used for FRANCOPHONE).
FRENCH_HS = (216.0, 0.86)
# Indigenous origins hand-placed to reflect linguistic family + region, harmonized with
# the US "native" scheme (lightness = latitude: Arctic pale -> eastern/plains dark;
# hue = family: Athabaskan 26 -> Ojibwe/plains 27 -> Cree/Algonquin 28 -> eastern
# Algonquian 29-30 -> Iroquoian 31-32 -> Métis 33). Direct US equivalents: Ojibway=Chippewa,
# Blackfoot=Blackfeet, Inuit~Alaska Native. Cherokee & Maya keep their US-matched colours.
INDIGENOUS_COLOR = {
    "Inuit": (31, 0.60, 0.53),                     # Arctic, palest (Eskaleut, like Alaska Native)
    "Dene": (26, 0.63, 0.50),                      # subarctic Athabaskan (Navajo/Apache kin)
    "First Nations": (30, 0.65, 0.46),             # Canada-wide umbrella (~AIAN not specified, paler)
    "North American Indigenous": (30, 0.63, 0.47), # broadest umbrella
    "Métis": (33, 0.61, 0.49),                     # mixed nation, its own tone
    "Cree": (28, 0.66, 0.47),
    "Plains Cree": (27, 0.66, 0.46),
    "Woodland Cree": (28, 0.65, 0.49),
    "Oji-Cree": (27, 0.66, 0.48),
    "Atikamekw": (28, 0.66, 0.47),
    "Innu/Montagnais": (28, 0.66, 0.48),
    "Ojibway": (27, 0.66, 0.46),                   # = US Chippewa
    "Anishinaabe": (27, 0.66, 0.47),
    "Saulteaux": (27, 0.66, 0.45),
    "Algonquin": (28, 0.66, 0.45),
    "Mi'kmaq": (29, 0.66, 0.44),                   # eastern Algonquian (Maritimes), dark brown
    "Qalipu Mi'kmaq": (29, 0.66, 0.45),
    "Maliseet": (30, 0.66, 0.44),
    "Abenaki": (30, 0.66, 0.43),
    "Blackfoot": (27, 0.66, 0.44),                 # = US Blackfeet
    "Mohawk": (32, 0.65, 0.45),                    # Iroquoian pocket
    "Iroquois (Haudenosaunee)": (32, 0.65, 0.46),
    "Huron (Wendat)": (31, 0.65, 0.46),
}
# Explicit per-label colours (hue, sat, lit) that win over everything.
SPECIAL_COLOR = {"Asian": (342.0, 0.35, 0.62), **INDIGENOUS_COLOR}  # + desaturated-pink Asian

# ---- 250 origins -> group.  Order-independent; keyed by exact StatCan label. ----
ORIGIN_GROUP = {
    # Canadian & regional (New-World rooted identities)
    "Canadian": "american", "Québécois": "american", "French Canadian": "american",
    "Acadian": "american", "American": "american", "Newfoundlander": "american",
    "Ontarian": "american", "Nova Scotian": "american", "New Brunswicker": "american",
    "British Columbian": "american", "Franco Ontarian": "american", "Albertan": "american",
    "Saskatchewanian": "american", "Gaspesian": "american", "Manitoban": "american",
    "Cape Bretoner": "american", "Prince Edward Islander": "american",
    "United Empire Loyalist": "american", "North American, n.o.s.": "american",

    # Indigenous
    "First Nations (North American Indian), n.o.s.": "native", "Métis": "native",
    "Cree, n.o.s.": "native", "North American Indigenous, n.o.s.": "native",
    "Mi'kmaq, n.o.s.": "native", "Ojibway": "native", "Inuit, n.o.s.": "native",
    "Algonquin": "native", "Mohawk": "native", "Innu/Montagnais, n.o.s.": "native",
    "Dene, n.o.s.": "native", "Blackfoot, n.o.s.": "native", "Abenaki": "native",
    "Iroquois (Haudenosaunee), n.o.s.": "native", "Plains Cree": "native",
    "Huron (Wendat)": "native", "Saulteaux": "native", "Anishinaabe, n.o.s.": "native",
    "Oji-Cree": "native", "Cherokee": "native", "Qalipu Mi'kmaq": "native",
    "Atikamekw": "native", "Woodland Cree": "native", "Maliseet": "native",

    # Western / Northern / Southern European
    "English": "western", "Irish": "western", "Scottish": "western", "French, n.o.s.": "western",
    "German": "western", "Italian": "western", "Dutch": "western", "British Isles, n.o.s.": "western",
    "European, n.o.s.": "western", "Norwegian": "western", "Welsh": "western", "Portuguese": "western",
    "Spanish": "western", "Swedish": "western", "Austrian": "western", "Belgian": "western",
    "Swiss": "western", "Finnish": "western", "Danish": "western", "Icelandic": "western",
    "Greek": "western", "Northern European, n.o.s.": "western", "Western European, n.o.s.": "western",
    "Northern Irish": "western", "Celtic, n.o.s.": "western", "Maltese": "western",
    "Breton": "western", "Norman": "western", "Flemish": "western", "Sicilian": "western",
    "Azorean": "western", "Basque": "western", "Mennonite": "western", "Pennsylvania Dutch": "western",

    # Eastern European / Slavic / Baltic
    "Ukrainian": "eastern", "Polish": "eastern", "Russian": "eastern", "Hungarian": "eastern",
    "Croatian": "eastern", "Czech": "eastern", "Serbian": "eastern", "Slovak": "eastern",
    "Eastern European, n.o.s.": "eastern", "Lithuanian": "eastern", "Romanian": "eastern",
    "Bulgarian": "eastern", "Slavic, n.o.s.": "eastern", "Latvian": "eastern", "Estonian": "eastern",
    "Byelorussian": "eastern", "Moldovan": "eastern", "Albanian": "eastern", "Macedonian": "eastern",
    "Slovenian": "eastern", "Bosnian": "eastern", "Czechoslovakian, n.o.s.": "eastern",
    "Yugoslavian, n.o.s.": "eastern", "Roma": "eastern",

    # MENA (Middle East + North Africa + Caucasus + Jewish)
    "Arab, n.o.s.": "mena", "Lebanese": "mena", "Iranian": "mena", "Persian": "mena",
    "Egyptian": "mena", "Moroccan": "mena", "Syrian": "mena", "Turkish": "mena",
    "Armenian": "mena", "Iraqi": "mena", "Algerian": "mena", "Tunisian": "mena",
    "Palestinian": "mena", "Kurdish": "mena", "Assyrian": "mena", "Berber": "mena",
    "Kabyle": "mena", "Coptic": "mena", "Israeli": "mena", "Jewish": "mena",
    "North African, n.o.s.": "mena", "West or Central Asian or Middle Eastern, n.o.s.": "mena",
    "Azerbaijani": "mena", "Chaldean": "mena", "Jordanian": "mena", "Yemeni": "mena", "Libyan": "mena",

    # South & Central Asian
    "Indian (India)": "s_c_asian", "Pakistani": "s_c_asian", "Punjabi": "s_c_asian",
    "Sri Lankan": "s_c_asian", "Tamil": "s_c_asian", "South Asian, n.o.s.": "s_c_asian",
    "Bangladeshi": "s_c_asian", "Gujarati": "s_c_asian", "Nepali": "s_c_asian", "Bengali": "s_c_asian",
    "Sinhalese": "s_c_asian", "Malayali": "s_c_asian", "Kashmiri": "s_c_asian", "Pashtun": "s_c_asian",
    "Goan": "s_c_asian", "Sikh": "s_c_asian", "Hindu": "s_c_asian", "Jatt": "s_c_asian",
    "Afghan": "s_c_asian", "Telugu": "s_c_asian", "Tajik": "s_c_asian",

    # East Asian
    "Chinese": "east_asian", "Korean": "east_asian", "Japanese": "east_asian",
    "Taiwanese": "east_asian", "Hong Konger": "east_asian", "Mongolian": "east_asian",
    "Tibetan": "east_asian", "East or Southeast Asian, n.o.s.": "east_asian",

    # Southeast Asian
    "Filipino": "se_asian", "Vietnamese": "se_asian", "Cambodian (Khmer)": "se_asian",
    "Indonesian": "se_asian", "Laotian": "se_asian", "Ilocano": "se_asian", "Thai": "se_asian",
    "Malaysian": "se_asian", "Malay": "se_asian", "Burmese": "se_asian", "Igorot": "se_asian",
    "Karen": "se_asian", "Singaporean": "se_asian",

    # Pacific
    "Fijian": "pacific",

    # Latin American / Hispanic
    "Mexican": "latino", "Colombian": "latino", "Salvadorean": "latino", "Chilean": "latino",
    "Peruvian": "latino", "Brazilian": "latino", "Cuban": "latino", "Venezuelan": "latino",
    "Guatemalan": "latino", "Ecuadorian": "latino", "Argentinian": "latino", "Nicaraguan": "latino",
    "Honduran": "latino", "Dominican": "latino", "Uruguayan": "latino", "Costa Rican": "latino",
    "Paraguayan": "latino", "Latin, Central or South American, n.o.s.": "latino",
    "Hispanic, n.o.s.": "latino", "Mayan": "latino",

    # African (Sub-Saharan + Black North American diaspora)
    "African, n.o.s.": "african", "Nigerian": "african", "Somali": "african", "Ethiopian": "african",
    "Congolese": "african", "Eritrean": "african", "Ghanaian": "african", "Cameroonian": "african",
    "Sudanese": "african", "Ivorian": "african", "Igbo": "african", "Yoruba": "african",
    "Kenyan": "african", "Rwandan": "african", "Burundian": "african", "Tigrinya": "african",
    "Senegalese": "african", "Bantu, n.o.s.": "african", "Zimbabwean": "african", "Bamileke": "african",
    "Ugandan": "african", "Oromo": "african", "Tanzanian": "african", "Central African": "african",
    "Akan, n.o.s.": "african", "Central or West African, n.o.s.": "african", "Fulani": "african",
    "Guinean": "african", "Beninese": "african", "Malagasy": "african", "Amhara": "african",
    "Southern or East African, n.o.s.": "african", "Black, n.o.s.": "african", "Mauritian": "african",
    "African American": "african", "African Canadian": "african", "African Nova Scotian": "african",
    "Edo": "african",

    # Afro-Caribbean / West Indian
    "Jamaican": "afro_carib", "Haitian": "afro_carib", "Guyanese": "afro_carib",
    "Trinidadian/Tobagonian": "afro_carib", "Barbadian": "afro_carib", "African Caribbean": "afro_carib",
    "Caribbean, n.o.s.": "afro_carib", "West Indian, n.o.s.": "afro_carib", "Vincentian": "afro_carib",
    "Grenadian": "afro_carib", "St. Lucian": "afro_carib", "Indo-Caribbean": "afro_carib",
    "Indo-Guyanese": "afro_carib",

    # Other / unspecified (pan religion/race, ambiguous settler)
    "Caucasian (White), n.o.s.": "other", "Christian, n.i.e.": "other", "Muslim": "other",
    "Buddhist": "other", "Asian, n.o.s.": "other", "Australian": "other", "New Zealander": "other",
    "South African": "other",
}


def equalized_lit(hue, sat):
    """Match luminance to blue@L0.6, blended 1/3 toward 0.5 (from scatter_dots.py)."""
    def lum(l):
        r, g, b = colorsys.hls_to_rgb(hue / 360, l, sat)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    target = 0.2126, 0.7152, 0.0722
    tr, tg, tb = colorsys.hls_to_rgb(240 / 360, 0.6, 0.7)
    target_lum = 0.2126 * tr + 0.7152 * tg + 0.0722 * tb
    lo, hi = 0.1, 0.95
    for _ in range(50):
        mid = (lo + hi) / 2
        if lum(mid) < target_lum:
            lo = mid
        else:
            hi = mid
    eq = (lo + hi) / 2
    return (eq + 2 * 0.5) / 3


def load_us_colors() -> dict:
    """US label -> {group, hue, sat, lit} from the US map's hand-tuned palette."""
    us = {}
    with open(US_COLORS, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        r.fieldnames = [n.strip() for n in r.fieldnames]
        for row in r:
            us[row["label"].strip()] = {
                "group": row["group"].strip(),
                "hue": float(row["hue"]), "sat": float(row["sat"]), "lit": float(row["lit"]),
            }
    return us


def main():
    us = load_us_colors()

    # national populations, aggregated by CANONICAL label (raw StatCan -> canonical)
    pops: dict = defaultdict(int)
    canon_raw: dict = {}   # canonical -> the raw StatCan label (for group lookup)
    with open(HERE / "data" / "raw" / "national_pops.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = row["origin"]
            canon = canonical(raw)
            pops[canon] += int(row["population"])
            canon_raw[canon] = raw

    def final_group(canon, raw):
        if canon in GROUP_OVERRIDE:
            return GROUP_OVERRIDE[canon]
        if canon in us:
            return us[canon]["group"]
        return ORIGIN_GROUP.get(raw, "other")

    rows = []
    n_shared = 0
    to_spread: dict = defaultdict(list)   # group -> [canon] coloured by group spread
    for canon in pops:
        raw = canon_raw[canon]
        grp = final_group(canon, raw)
        # explicit French / francophone colour
        if canon == "French" or canon in FRANCOPHONE:
            hue, sat = FRENCH_HS
            rows.append((canon, grp, round(hue, 1), round(sat, 3), round(equalized_lit(hue, sat), 3), pops[canon]))
        # explicit per-label colour
        elif canon in SPECIAL_COLOR:
            hue, sat, lit = SPECIAL_COLOR[canon]
            rows.append((canon, grp, round(hue, 1), round(sat, 3), round(lit, 3), pops[canon]))
        # custom uniform groups (North American teal, Other pale purple) ignore US colours
        elif grp in CUSTOM_GROUP:
            to_spread[grp].append(canon)
        # otherwise adopt the US palette where shared
        elif canon in us:
            c = us[canon]
            rows.append((canon, grp, round(c["hue"], 1), round(c["sat"], 3), round(c["lit"], 3), pops[canon]))
            n_shared += 1
        else:
            to_spread[grp].append(canon)

    # spread the remaining labels across their group's hue_range by population
    for grp, labels in to_spread.items():
        labels.sort(key=lambda c: -pops[c])
        g = GROUPS[grp]
        n = len(labels)
        span = g["hue_range"]
        for i, canon in enumerate(labels):
            frac = 0 if n == 1 else (i / (n - 1)) - 0.5
            hue = (g["hue"] + frac * span) % 360
            sat = g["sat"]
            # custom groups keep their flat light/pale lit; others get luminance-equalized
            lit = g["lit"] if grp in CUSTOM_GROUP else equalized_lit(hue, sat)
            rows.append((canon, grp, round(hue, 1), round(sat, 3), round(lit, 3), pops[canon]))

    print(f"{len(rows)} origins: {n_shared} share the US palette")

    # sort output by group order then population desc for readability
    order = list(GROUPS.keys())
    rows.sort(key=lambda r: (order.index(r[1]) if r[1] in order else 99, -r[5]))

    out = HERE / "ancestry_colors_ca.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "group", "hue", "sat", "lit", "population"])
        w.writerows(rows)
    print(f"wrote {len(rows)} origins -> {out}")

    with open(HERE / "groups_ca.json", "w", encoding="utf-8") as f:
        json.dump({"groups": GROUPS}, f, ensure_ascii=False, indent=2)

    # summary
    from collections import Counter
    c = Counter(r[1] for r in rows)
    print("\ngroup counts:")
    for grp in order:
        if c[grp]:
            print(f"  {GROUPS[grp]['label']:22} ({grp:11}) {c[grp]:3}")


if __name__ == "__main__":
    main()
