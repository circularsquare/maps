"""
Where inside a US county the dots go — spec §8.4.

ASARB is county-level and there is nothing finer: the study does not collect congregation
addresses, and PL 94-521 keeps the Census Bureau out of the question entirely. So Cook County
is 5.3M people in one polygon and Los Angeles is 10.0M in another, and §8.2's placement rule
spreads each county's adherents evenly over its tracts — every neighbourhood of Chicago drawn
with an identical religious mix.

§8.3 tried to fix that with church locations and it failed: parish density is a fossil of 1920s
settlement, so the model rated Englewood more Catholic than Mount Greenwood. The lesson it left
was that a model built from a proxy needs an external check against something known.

This is the second attempt, and the proxy is demographic composition — ancestry, birthplace and
race, which the ACS publishes at tract level and ancestrydots already has on disk for every
state.

THE CHECK. ASARB's own county numbers are ground truth at roughly the granularity wanted, so:
fit the model on counties, hold out WHOLE METROS, and see whether it predicts the within-metro
variation it never saw. 448 counties in 48 metros. Scored on population-weighted correlation of
the within-metro deviation, because the allocation is raked to the ASARB county total and the
level is therefore fixed for free — only the relative pattern has to be right.

    Church of God in Christ          r = 0.61      Hindu Temples            r = 0.20
    National Baptist Convention USA      0.57      Jehovah's Witnesses          0.21
    Seventh-day Adventist                0.57      Lutheran (Missouri Synod)    0.12
    Reform Judaism                       0.46      United Church of Christ      0.09
    Catholic Church                      0.45      Assemblies of God            0.07
    Episcopal, AME, American Baptist  0.42-0.45    ORTHODOX JUDAISM             0.05

Calibration slopes run 0.7–0.96 for those, so the predicted spread is about the right size
rather than a muted smear. Against §8.3's parish model, which had no discrimination at all
(34% vs 37% across Chicago's most and least Catholic neighbourhoods), this is a real signal.

So it ships PER NODE, gated on that number: a node whose held-out-metro correlation clears
R_MIN gets demographic weights, every other node stays population-uniform. That is a per-node
confidence claim rather than an all-or-nothing switch, which is what §7 asks for anyway.

THREE THINGS IT IS NOT, all of which belong in the about panel:

1. It is an estimate, not a measurement. Nothing below county level here was counted.
2. It is partly the race map. "Englewood is Black Protestant" is the input, not a finding. The
   part that is not circular is the ancestry and birthplace layer — Guyanese Hindus in Richmond
   Hill, Armenians in Glendale, Assyrians in Skokie — which no race map contains.
3. The check is BETWEEN counties and the use is WITHIN one, and the coefficients get
   extrapolated well past their fitted range: metro counties run 5–25% Black, Cook County
   tracts run 0–100%. The direction is the favourable one — more demographic contrast, not
   less — but it is extrapolation and it is not validated there, because nothing to validate
   it against exists.

ORTHODOX JUDAISM WAS THE CONSPICUOUS FAILURE and is now the best-placed body on the US map.
The first cut had no Jewish marker at all — ancestry does not carry one, `israeli` is Israelis
rather than Haredim, and the Haredi neighbourhoods report European ancestries that this file
read as `euro_catholic`. So Borough Park drew 58% Catholic and 1.0x Orthodox Judaism, in the
most Jewish neighbourhood in America. A missing predictor is never neutral: Orthodox Judaism
correctly got no weights, and Catholic, which did, took the space instead.

The fix was LANGUAGE SPOKEN AT HOME (B16001), and the reason it was not in the first cut is
worth keeping: api.census.gov refuses unkeyed requests now, and that was taken as the end of
the road. It is not — the summary file on www2.census.gov is the same data as flat files and
needs no key. See LANG_VARS.

    Borough Park          1.0x -> 4.4x        Kew Gardens Hills (Modern Orth)  4.3x
    Williamsburg               -> 2.7x        Pico-Robertson, LA               5.8x
    Midwood                    -> 2.2x        Beverly-Fairfax, LA              8.0x
    Bed-Stuy (no Jews)         -> 1.1x        West Rogers Park, Chicago        2.5x

It still fails the fitted gate (r = 0.07) and always will: 79% of Orthodox Judaism is in one
metro, so whole-metro holdout has nothing to learn from. It is AUTHORED instead, on the same
grounds as Armenian Apostolic, and the ground-truth check is that the top Yiddish PUMAs in the
country are Monsey, Borough Park, Kiryas Joel, Williamsburg and Lakewood in that order.

UJA-Federation's 2023 Jewish Community Study — Jewish population by sub-county ZIP cluster for
the eight New York counties — is still the measured source and would beat this. It is now an
improvement rather than the only hope.

Usage:
    python us_weights.py --build      # tract segments + fit; writes both artefacts
    python us_weights.py --report     # print the validation table; fits, writes nothing
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).parent
ANCESTRYDOTS = HERE.parent / "ancestrydots"
SEGMENTS_CSV = HERE / "data" / "geo" / "us_tract_segments.csv"
MODEL_JSON = HERE / "data" / "processed" / "us_weight_model.json"
CBSA_XLSX = HERE / "data" / "geo" / "cbsa_list1_2023.xlsx"
CBSA_URL = ("https://www2.census.gov/programs-surveys/metro-micro/geographies/"
            "reference-files/2023/delineation-files/list1_2023.xlsx")
CES_FEATHER = HERE / "data" / "raw" / "ces_cumulative_2006-2025.feather"
CES_URL = "https://dataverse.harvard.edu/api/access/datafile/14076522"
EDU_DAT = HERE / "data" / "geo" / "acsdt5y2023-b15003.dat"
AGE_DAT = HERE / "data" / "geo" / "acsdt5y2023-b01002.dat"
SF = ("https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/"
      "data/5YRData/acsdt5y2023-{}.dat")
LANG_DAT = HERE / "data" / "geo" / "acsdt5y2023-b16001.dat"
LANG_URL = ("https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/"
            "data/5YRData/acsdt5y2023-b16001.dat")
TRACT_PUMA = HERE / "data" / "geo" / "tract_to_puma_2020.txt"
TRACT_PUMA_URL = ("https://www2.census.gov/geo/docs/maps-data/data/rel2020/"
                  "2020_Census_Tract_to_2020_PUMA.txt")

# Language spoken at home, B16001 — the segments the ancestry tables cannot supply.
#
# THE API IS NOT THE ONLY WAY IN. api.census.gov now refuses unkeyed requests, which is why
# the first cut of this file had no language data and drew Borough Park as Catholic. The
# summary file on www2.census.gov is the same data as flat .dat files and needs no key at all.
# Worth remembering before concluding a census table is unreachable.
#
# The catch is geography: B16001's finest published level is the PUMA, ~100,000 people, not the
# tract. That is still twenty-eight units inside Brooklyn against the one that ASARB gives, and
# PUMAs are drawn on neighbourhood lines — Borough Park is essentially its own. So a tract
# inherits its PUMA's language shares, and everything sourced here is PUMA-resolution however
# finely the dots are drawn. That is a real limit and it is the difference between "somewhere
# in Brooklyn" and "in Borough Park", which is the whole complaint.
#
# `yiddish` is the load-bearing one and it is really "Yiddish, Pennsylvania Dutch or other West
# Germanic" — the Census does not split them. In Brooklyn and Rockland that is Haredi Jews; in
# Lancaster County it is the Amish. Both are true and both are wanted, so the lump is left as
# it is rather than being forced to mean one thing.
LANG_VARS = {
    "lang_yiddish": "B16001_E021",      # Yiddish, Pennsylvania Dutch or other West Germanic
    "lang_hebrew": "B16001_E108",
    "lang_armenian": "B16001_E039",
    "lang_persian": "B16001_E042",
    "lang_arabic": "B16001_E105",
    "lang_afroasiatic": "B16001_E111",  # Amharic, Somali or other Afro-Asiatic
    "lang_gujarati": "B16001_E045",
    "lang_hindi": "B16001_E048",
    "lang_urdu": "B16001_E051",
    "lang_punjabi": "B16001_E054",
    "lang_bengali": "B16001_E057",
    "lang_dravidian": "B16001_E072",    # Malayalam, Kannada or other Dravidian
}
LANG_SEGS = list(LANG_VARS)

# Which tract-level ancestry carries each language, used to spread a PUMA's speakers across
# its tracts instead of smearing them evenly.
#
# THE CASE THAT FORCED THIS: PUMA 3604303 is "Bedford-Stuyvesant & Crown Heights North" and is
# 6.2% Yiddish, because Crown Heights is the world centre of Chabad. Spread evenly it made
# Bed-Stuy — which has essentially no Jews — 1.3x the borough rate for Orthodox Judaism. The
# two halves of that PUMA are not alike in any other respect either: one is Black, one is not.
#
# Yiddish-speaking Haredim are recorded as white by the race question, so allocating a PUMA's
# Yiddish speakers in proportion to its white population is a sound disaggregation rather than
# a new assumption. Same for the rest: Armenian speakers are of Armenian ancestry, Malayalam
# speakers are South Asian. Where a PUMA has none of the carrier at all, it falls back to
# population, which is the old behaviour.
LANG_CARRIER = {
    "lang_yiddish": ["white_resid", "euro_catholic", "euro_protestant", "euro_orthodox"],
    "lang_hebrew": ["white_resid", "euro_catholic", "euro_protestant", "euro_orthodox",
                    "israeli"],
    "lang_armenian": ["armenian"],
    "lang_persian": ["iranian_turkic"],
    "lang_arabic": ["arab"],
    "lang_afroasiatic": ["african"],
    "lang_gujarati": ["south_asian"],
    "lang_hindi": ["south_asian"],
    "lang_urdu": ["south_asian", "pakistani"],
    "lang_punjabi": ["south_asian", "sikh_resp"],
    "lang_bengali": ["south_asian"],
    "lang_dravidian": ["south_asian"],
}

# A node needs this held-out-metro correlation before its dots are moved. 0.25 is a judgement:
# it keeps Catholic, the Black Protestant bodies, Adventist, Episcopal, PCUSA, LDS, ELCA, UMC,
# nondenominational and Southern Baptist — 91% of the adherents tested — and drops Assemblies
# of God, Missouri Synod, UCC, Churches of Christ, Hindu Temples and Orthodox Judaism, which
# would otherwise be moved on noise.
R_MIN = 0.25
RIDGE = 0.15        # as a fraction of trace(X'X)/k; where the sweep in _ridge puts slope ~1
N_FOLDS = 5
FOLD_SEED = 1
MIN_ADHERENTS = 20_000      # below this a node has too few metro counties to score honestly
CES_MIN_N_FIT = 50          # respondents before a county informs the residual fit
CES_MIN_N_SCORE = 150       # ...and before it is scored against, since a survey share is noisy

# Nodes whose ASARB figure is a compiler ESTIMATE rather than a body's own return — group codes
# 267, 890, 891, 892 and 895, per sources.md §9 and spec §3.1. Their held-out r is not
# independent evidence, because a number that was itself modelled from population can be
# reproduced by a model of population; `islam` scores 0.64, the highest here, for that reason.
# They still take weights when they clear the bar — uniform is not the safer answer, it is just
# a different wrong one — but the flag travels with them into the model file so the r is never
# read as validation.
ESTIMATE_NODES = {"islam", "hinduism", "buddhism.mahayana", "buddhism.theravada",
                  "buddhism.vajrayana"}

# The segments. Coarse enough to fit on 448 counties, fine enough to separate the things that
# actually differ in religion — Irish and Polish from German and Norwegian, Mexican from
# Dominican, Armenian and Assyrian from everything around them.
_SEG = {}


def _add(seg, *labels):
    for lab in labels:
        _SEG[lab] = seg


_add("euro_catholic", "Irish", "Italian", "Polish", "Portuguese", "French Canadian", "Slovak",
     "Slovene", "Croatian", "Lithuanian", "Hungarian", "Czech", "Czechoslovakian", "Austrian",
     "Belgian", "Luxembourger", "Maltese", "Basque", "French", "Alsatian", "Cajun",
     "Acadian/Cajun")
_add("euro_protestant", "English", "German", "Scandinavian", "Norwegian", "Swedish", "Danish",
     "Dutch", "Scottish", "Scotch-Irish", "Welsh", "Finnish", "Swiss", "British", "Icelander",
     "Northern European", "Celtic", "Pennsylvania German", "Australian", "European",
     "Western European", "Estonian", "Latvian", "German Russian", "Canadian", "Slavic")
_add("american_anc", "American", "Southern", "Appalachian", "United States", "U.S.")
_add("euro_orthodox", "Russian", "Ukrainian", "Serbian", "Greek", "Romanian", "Bulgarian",
     "Carpatho Rusyn", "Macedonian", "Belarusian", "Albanian", "Eastern European",
     "Yugoslavian", "Bosnian", "Montenegrin", "Soviet Union", "Moldovan", "Georgian", "Cypriot")
_add("armenian", "Armenian")
_add("assyrian", "Assyrian/Chaldean/Syriac", "Assyrian", "Chaldean")
_add("arab", "Lebanese", "Syrian", "Palestinian", "Jordanian", "Egyptian", "Iraqi", "Yemeni",
     "Moroccan", "Arab", "Arab/Arabic", "Algerian", "Libyan", "Tunisian", "Kuwaiti",
     "Saudi Arabian", "Emirati", "Bahraini", "Qatari", "Omani", "Middle Eastern", "Other Arab",
     "North African", "Sudanese", "Berber")
_add("iranian_turkic", "Iranian", "Turkish", "Afghan", "Kurdish", "Azerbaijani", "Kazakh",
     "Uzbek", "Other Central Asian", "Tatar")
_add("israeli", "Israeli")
_add("african", "Nigerian", "Ghanaian", "Ethiopian", "Somali", "Kenyan", "Eritrean", "Liberian",
     "Sierra Leonean", "Cameroonian", "Senegalese", "Cape Verdean", "Sub-Saharan African",
     "African", "Congolese", "Ugandan", "Zimbabwean", "South African", "Togolese", "Gambian",
     "Guinean", "Ivoirian", "Malian", "Burkinabe", "Nigerien", "Chadian", "Angolan",
     "Mozambican", "Rwandan", "Tanzanian", "Zambian", "Botswanan", "Namibian",
     "Other Subsaharan African")
_add("afro_carib", "Jamaican", "Haitian", "Trinidadian and Tobagonian", "Barbadian",
     "West Indian", "Bahamian", "Belizean", "Bermudan", "British West Indian",
     "Dutch West Indian", "U.S. Virgin Islander", "Other West Indian",
     "Trinidadian & Tobagonian")
# Guyanese is its own segment on purpose: Richmond Hill is the largest Hindu neighbourhood in
# the United States and nothing else in the ACS marks it.
_add("guyanese", "Guyanese")
_add("brazilian", "Brazilian")
_add("black_resid", "Black, no ancestry reported")
_add("white_resid", "White, no ancestry reported")
_add("mexican", "Mexican")
_add("puerto_rican", "Puerto Rican")
_add("cuban", "Cuban")
_add("dominican", "Dominican")
_add("central_am", "Salvadoran", "Guatemalan", "Honduran", "Nicaraguan", "Costa Rican",
     "Panamanian", "Other Central American")
_add("south_am", "Colombian", "Ecuadorian", "Peruvian", "Venezuelan", "Argentinean", "Bolivian",
     "Chilean", "Paraguayan", "Uruguayan", "Other South American")
_add("other_hisp", "Spaniard", "Spanish", "Spanish American", "Other Hispanic or Latino")
_add("south_asian", "Indian", "Bangladeshi", "Nepalese", "Sri Lankan", "Bhutanese",
     "Other South Asian")
_add("pakistani", "Pakistani")
_add("sikh_resp", "Sikh")
_add("east_asian", "Chinese", "Japanese", "Korean", "Taiwanese", "Mongolian", "Okinawan",
     "Other East Asian", "Other Asian, specified", "Other Asian, not specified", "Hmong")
_add("filipino", "Filipino")
_add("se_asian", "Vietnamese", "Cambodian", "Laotian", "Thai", "Burmese", "Indonesian",
     "Malaysian", "Singaporean", "Mien", "Other Southeast Asian")

# --------------------------------------------------------------------------------------
# AUTHORED ties — where the ethnicity is constitutive, not correlated
# --------------------------------------------------------------------------------------
# The fit learns from variation BETWEEN counties, so it can only see a body whose share moves
# across counties by enough to stand out of the noise. That excludes exactly the bodies a
# reader most wants to find in Glendale, Skokie or Richmond Hill: at county scale `armenian`
# is 0.2% of the population nationally, and no penalty setting makes it legible (see _ridge).
#
# For these bodies the tie is not a statistical regularity to be estimated, it is what the body
# IS. The Armenian Apostolic Church is Armenian by constitution; the Ethiopian Orthodox Tewahedo
# Church is Ethiopian; Mar Thoma is Kerala. So they are authored, in the same spirit as the 372
# hand-mapped taxonomy placements of §2.4, and the affinity is a claim this file is making
# rather than a number it measured. That distinction travels into the model file as
# `basis: authored`, and it should reach the reader too.
#
# The bar for entry is narrow: the body's own name or canon names the ethnicity. Bodies that
# merely SKEW ethnic — Southern Baptist, the Church of God in Christ, Greek Catholic parishes —
# stay with the fit, which is what evidence is for.
#
# JUDAISM IS DELIBERATELY ABSENT. There is no honest ACS segment for it: `israeli` is Israelis,
# who are not most American Jews and are barely any Haredim, and the Haredi neighbourhoods that
# most want drawing report European ancestries that this file reads as `euro_catholic`. The
# result is the Borough Park failure — the map filling the most Jewish neighbourhood in America
# with Catholics. Until ACS B16001 (Yiddish and Hebrew at home) is fetched, or UJA-Federation's
# 2023 study is read for the eight New York counties, Judaism stays uniform.
# `floor` is added to every tract's affinity before weighting, in the same units (share of
# population). It is the difference between "concentrated here" and "absent everywhere else",
# and without it a body lands entirely in whichever tracts happen to carry its marker. Orthodox
# Judaism is the case that needs it: Yiddish nails Borough Park, Williamsburg, Monsey and
# Lakewood, and says nothing at all about the 80,000 Orthodox Jews in Queens or the 44,400 in
# Los Angeles, who are Modern Orthodox and speak English. Hebrew carries some of those and the
# floor carries the rest.
AUTHORED = {
    "christianity.oriental.armenian-etchmiadzin": {"armenian": 1.0, "lang_armenian": 1.0},
    "christianity.oriental.armenian-cilicia": {"armenian": 1.0, "lang_armenian": 1.0},
    "christianity.oriental.coptic": {"arab": 0.5, "lang_arabic": 1.0},          # Egyptian
    # Amharic is the Ethiopian Orthodox liturgical and household language, and a far tighter
    # marker than `african`, which is every Sub-Saharan ancestry at once.
    "christianity.oriental.ethiopian": {"lang_afroasiatic": 1.0, "african": 0.3},
    "christianity.oriental.eritrean": {"lang_afroasiatic": 1.0, "african": 0.3},
    "christianity.oriental.syriac": {"assyrian": 1.0},
    # Kerala. Malayalam is a Dravidian language and the Dravidian lump is mostly Malayalam and
    # Tamil; for these four bodies it is much sharper than "South Asian".
    "christianity.oriental.malankara-orthodox": {"lang_dravidian": 1.0, "south_asian": 0.3},
    "christianity.oriental.malankara-syriac": {"lang_dravidian": 1.0, "south_asian": 0.3},
    "christianity.oriental.marthoma": {"lang_dravidian": 1.0, "south_asian": 0.3},
    "christianity.oriental.knanaya": {"lang_dravidian": 1.0, "south_asian": 0.3},
    "christianity.orthodox.canonical.greek": {"euro_orthodox": 1.0},
    "christianity.orthodox.canonical.serbian": {"euro_orthodox": 1.0},
    "christianity.orthodox.canonical.macedonian": {"euro_orthodox": 1.0},
    "christianity.orthodox.canonical.ukrainian": {"euro_orthodox": 1.0},
    "christianity.orthodox.canonical.romanian": {"euro_orthodox": 1.0},
    "christianity.orthodox.canonical.bulgarian": {"euro_orthodox": 1.0},
    "christianity.orthodox.canonical.carpatho-russian": {"euro_orthodox": 1.0},
    "christianity.orthodox.canonical.georgian": {"euro_orthodox": 1.0},
    "christianity.orthodox.canonical.albanian": {"euro_orthodox": 1.0},
    "christianity.orthodox.canonical.rocor": {"euro_orthodox": 1.0},
    "christianity.orthodox.canonical.patriarchal-parishes": {"euro_orthodox": 1.0},
    # Antiochian is the Levantine one, and in the US it is largely Lebanese and Syrian.
    "christianity.orthodox.canonical.antiochian": {"arab": 1.0},
    # The OCA is the one canonical body here with no ethnicity in its name — it is the old
    # Russian mission turned deliberately pan-ethnic and English-speaking. Half weight on the
    # Slavic segment says "still visibly Slavic, no longer only Slavic", which is the honest
    # reading of a body whose whole point was to stop being an ethnic church.
    "christianity.orthodox.canonical.oca": {"euro_orthodox": 0.5, "euro_protestant": 0.1},
    # Guyanese carries as much weight as the whole South Asian segment because Richmond Hill
    # is the largest Hindu neighbourhood in the country and reads as Caribbean, not Indian.
    # `south_asian` is held to half because it contains Bangladeshis, who are Muslim; Gujarati
    # and Hindi at home pull the weight back toward the Hindu half of the diaspora.
    "hinduism": {"south_asian": 0.5, "guyanese": 1.0,
                 "lang_gujarati": 1.0, "lang_hindi": 0.5},
    "hinduism.vedanta": {"south_asian": 0.5, "guyanese": 1.0,
                         "lang_gujarati": 1.0, "lang_hindi": 0.5},
    # Punjabi at home is about as clean a Sikh marker as the census contains.
    "sikhism": {"lang_punjabi": 1.0, "sikh_resp": 1.0, "south_asian": 0.2},
    "jainism": {"lang_gujarati": 1.0, "south_asian": 0.3},
    "zoroastrianism": {"lang_persian": 1.0, "iranian_turkic": 0.5, "south_asian": 0.3},
    "buddhism.mahayana": {"east_asian": 1.0, "se_asian": 0.5},
    "buddhism.theravada": {"se_asian": 1.0, "south_asian": 0.3},
    "buddhism.vajrayana": {"east_asian": 0.5, "south_asian": 0.5},

    # ORTHODOX JUDAISM — the node this whole language block exists for. Fitted r is 0.07 and
    # will not improve: 79% of it is in one metro, so whole-metro holdout has nothing to learn
    # from, exactly as with Armenian Apostolic. But the raw signal is strong — county Orthodox
    # share against Yiddish share is r = 0.52 over 991 counties — and the top Yiddish PUMAs in
    # the country are Monsey, Borough Park, Kiryas Joel, Williamsburg and Lakewood, in that
    # order. That is the ground-truth check §8.3 demands, and it passes on the nose.
    #
    # Yiddish and Hebrew go in at equal weight because they balance themselves: Brooklyn is
    # 4.4% Yiddish against 0.8% Hebrew and lands on Borough Park, while Queens and Los Angeles
    # are Hebrew-dominant and land on Kew Gardens Hills and Pico-Robertson. No tuning needed.
    "judaism.orthodox": {"lang_yiddish": 1.0, "lang_hebrew": 1.0, "floor": 0.002},
    "judaism.chabad": {"lang_yiddish": 1.0, "lang_hebrew": 1.0, "floor": 0.002},
}

# The design matrix. `sikh_resp`, `other_hisp`, `brazilian` and `white_resid` are built but held
# out of the fit: the first three are tiny or residual categories, and `white_resid` is the
# complement of the rest, so including it makes the matrix singular.
SEGS = ["black_resid", "african", "afro_carib", "guyanese", "euro_catholic", "euro_protestant",
        "american_anc", "euro_orthodox", "armenian", "assyrian", "arab", "iranian_turkic",
        "israeli", "mexican", "puerto_rican", "cuban", "dominican", "central_am", "south_am",
        "south_asian", "pakistani", "east_asian", "filipino", "se_asian", "native", "pacific"]

# Education and age, at tract level from the same keyless summary file. Ancestry and language
# say who people descend from; these two say the things that actually predict IRRELIGION, and
# the residual model (§8.4a) is nearly useless without them — degree share alone correlates
# +0.56 with the atheist-and-agnostic share of a county.
EXTRA_SEGS = ["ba_share", "age"]

# What the ridge actually sees: ancestry, then language, then the two rates. Order is
# load-bearing — `beta` is written out zipped against this list.
FIT_SEGS = SEGS + LANG_SEGS + EXTRA_SEGS


def design(frame, pop):
    """The design matrix, from raw count columns. ONE definition, used by both fits and by
    the Weighter — three blocks with three denominators, and if any caller disagrees about
    them every coefficient is silently misapplied to the wrong scale.

    `frame` carries counts (ancestry segments, language speakers, ba_plus, adults25,
    age_x_pop) at whatever geography; `pop` is that geography's population.
    """
    anc = [c for c in frame.columns
           if c not in set(LANG_SEGS) | {"pop", "ba_plus", "adults25", "age_x_pop"}]
    denom = frame[anc].sum(axis=1)
    pop = pop.where(pop > 0)
    out = pd.concat([
        frame.reindex(columns=SEGS).div(denom.where(denom > 0), axis=0),
        frame.reindex(columns=LANG_SEGS).div(pop, axis=0),
        pd.DataFrame({
            "ba_share": frame["ba_plus"] / frame["adults25"].where(frame["adults25"] > 0),
            # /40 only to put it on the same order as a share, so one raw-scale ridge
            # penalty means roughly the same thing for it as for everything else.
            "age": frame["age_x_pop"] / pop / 40.0,
        }, index=frame.index),
    ], axis=1).fillna(0.0)
    return out[FIT_SEGS]


# --------------------------------------------------------------------------------------
# stage 1 — tract segments
# --------------------------------------------------------------------------------------

def _seg_of(label, group):
    if label in _SEG:
        return _SEG[label]
    if group == "native":
        return "native"
    if group in ("pacific", "pac_isl", "pacific_islander"):
        return "pacific"
    return None


def _add_language_segments(seg):
    """Give every tract its PUMA's language mix, as estimated COUNTS.

    Counts rather than shares because fit() aggregates tracts to counties with a plain sum;
    summing shares would be meaningless. share = sum(count) / sum(pop) at whatever level.
    """
    if not (LANG_DAT.exists() and TRACT_PUMA.exists()):
        print(f"  !! no B16001 ({LANG_DAT.name}) or crosswalk — no language segments.\n"
              f"     curl -sSL -o {LANG_DAT} {LANG_URL}\n"
              f"     curl -sSL -o {TRACT_PUMA} {TRACT_PUMA_URL}")
        for c in LANG_SEGS:
            seg[c] = 0.0
        return seg

    cols = ["GEO_ID", "B16001_E001"] + list(LANG_VARS.values())
    lang = pd.read_csv(LANG_DAT, sep="|", usecols=lambda c: c in cols, dtype={"GEO_ID": str})
    lang = lang[lang["GEO_ID"].str.startswith("795P200US")]          # PUMAs only
    lang["PUMA"] = lang["GEO_ID"].str[len("795P200US"):]             # state(2) + puma(5)
    base = pd.to_numeric(lang["B16001_E001"], errors="coerce").fillna(0)
    shares = pd.DataFrame({name: pd.to_numeric(lang[var], errors="coerce").fillna(0)
                           / base.where(base > 0)
                           for name, var in LANG_VARS.items()}).fillna(0.0)
    shares.index = lang["PUMA"].to_numpy()

    xw = pd.read_csv(TRACT_PUMA, dtype=str)
    xw.columns = [c.strip().lstrip("﻿") for c in xw.columns]
    xw["GEOID"] = xw["STATEFP"] + xw["COUNTYFP"] + xw["TRACTCE"]
    xw["PUMA"] = xw["STATEFP"] + xw["PUMA5CE"]
    puma_of = dict(zip(xw["GEOID"], xw["PUMA"]))

    p = pd.Series(seg.index.map(puma_of), index=seg.index)
    got = shares.reindex(p.to_numpy())
    got.index = seg.index

    # A PUMA's total speakers, then split across its tracts by whoever carries the language.
    puma_pop = seg["pop"].groupby(p).sum()
    for c in LANG_SEGS:
        total = got[c].fillna(0.0) * p.map(puma_pop).fillna(0.0)      # same for every tract
        carrier = seg.reindex(columns=LANG_CARRIER.get(c, []), fill_value=0.0).sum(axis=1)
        denom = carrier.groupby(p).transform("sum")
        frac = np.where(denom > 0, carrier / denom.where(denom > 0),
                        seg["pop"] / p.map(puma_pop).replace(0, np.nan))
        seg[c] = (total * pd.Series(frac, index=seg.index).fillna(0.0)).round(1)

    hit = p.notna() & p.isin(shares.index)
    print(f"  language: {int(hit.sum()):,} of {len(seg):,} tracts matched a PUMA; "
          f"{seg['lang_yiddish'].sum():,.0f} Yiddish/West-Germanic speakers placed, "
          f"spread within each PUMA by carrier ancestry")
    return seg


def _add_extra_segments(seg):
    """Tract-level education and median age, from the keyless summary file.

    ACS JAM VALUES ARE A TRAP AND THEY COST A WHOLE ROUND OF THIS WORK. `-666666666` and
    friends mean "no estimate", not a number. Left in, median age had a minimum of -52,769 and
    a standard deviation of 2,560 — one poisoned column dominated the ridge trace, shrank
    EVERY coefficient to about zero, and the fit reported no signal at all while the raw
    correlations behind it were 0.4 to 0.6. A model that reports nothing is a claim about the
    fitter first; check the raw relationship before believing it (the same lesson as §8.4's
    first three specifications, arrived at from the other direction).
    """
    missing = [p for p in (EDU_DAT, AGE_DAT) if not p.exists()]
    if missing:
        print(f"  !! no {', '.join(p.name for p in missing)} — no education or age segments.")
        for c in ("ba_plus", "adults25", "age_x_pop"):
            seg[c] = 0.0
        return seg

    def tract_col(path, cols):
        d = pd.read_csv(path, sep="|", usecols=lambda c: c in (["GEO_ID"] + cols), dtype=str)
        d = d[d["GEO_ID"].str.startswith("1400000US")]
        d.index = d["GEO_ID"].str[len("1400000US"):]
        return {c: pd.to_numeric(d[c], errors="coerce").where(
            lambda v: v > -1e8) for c in cols}

    edu = tract_col(EDU_DAT, ["B15003_E001", "B15003_E022", "B15003_E023",
                              "B15003_E024", "B15003_E025"])
    age = tract_col(AGE_DAT, ["B01002_E001"])
    ba = sum(edu[f"B15003_E{n:03d}"] for n in (22, 23, 24, 25))
    seg["ba_plus"] = ba.reindex(seg.index).fillna(0.0)
    seg["adults25"] = edu["B15003_E001"].reindex(seg.index).fillna(0.0)
    med = age["B01002_E001"].reindex(seg.index).clip(10, 80).fillna(38.0)
    seg["age_x_pop"] = med * seg["pop"]           # a county sum / county pop is then the mean
    ok = seg["adults25"] > 0
    print(f"  education & age: {int(ok.sum()):,} of {len(seg):,} tracts; "
          f"{seg['ba_plus'].sum() / seg['adults25'].sum():.1%} of adults hold a degree")
    return seg


def _ces_targets(year_from=2016):
    """County religion shares from the Cooperative Election Study — the residual's ground truth.

    ASARB cannot supply one: the residual is by definition the people no roll holds. PRRI's
    county file looks like the obvious answer and is disqualified, because those estimates are
    themselves a Bayesian small-area model built from ACS demographics — scoring a demographic
    model against them would prove nothing, the same circularity as ASARB's "Muslim Estimate".
    CES is raw survey microdata with a county FIPS on each row and is genuinely independent.

    Pooled from `year_from` rather than 2006 because the unaffiliated share moved a great deal
    over the full run and a county's sample is not evenly spread across the years.
    """
    if not CES_FEATHER.exists():
        raise SystemExit(f"missing {CES_FEATHER}\n  curl -sSL -o {CES_FEATHER} {CES_URL}")
    import pyarrow.feather as pf

    d = pf.read_table(CES_FEATHER, columns=["year", "county_fips", "religion",
                                            "weight_cumulative"]).to_pandas()
    d = d[(d["year"] >= year_from) & d["county_fips"].notna() & d["religion"].notna()]
    w = d["weight_cumulative"].fillna(1.0).clip(lower=0).to_numpy()
    r = d["religion"]
    # CES religion codes: 1 Protestant, 2 Catholic, 3 Mormon, 4 Orthodox, 5 Jewish, 6 Muslim,
    # 7 Buddhist, 8 Hindu, 9 Atheist, 10 Agnostic, 11 Nothing in particular, 12 Something else.
    cats = {"unaffiliated": [11], "secular": [9, 10], "christianity": [1, 2, 3, 4],
            "judaism": [5], "islam": [6], "buddhism": [7], "hinduism": [8],
            "other.us": [12]}
    out = pd.DataFrame({"FIPS": d["county_fips"].astype(int).astype(str).str.zfill(5).to_numpy(),
                        "w": w})
    for name, codes in cats.items():
        out[name] = r.isin(codes).to_numpy() * w
    g = out.groupby("FIPS").sum()
    g["n"] = out.groupby("FIPS").size()
    for name in cats:
        g[name] = g[name] / g["w"]
    print(f"  CES {year_from}+: {len(out):,} respondents in {len(g):,} counties")
    return g[["n"] + list(cats)]


def _reindex_connecticut(seg):
    """Connecticut is §8.1 for the third time, and the direction is the new one.

    §8.1's rule is that boundaries must be the vintage the data was published on. Here two
    vintages are both required at once: the placement layer has to be 2020 tracts, because
    their county prefix is what joins to ASARB, and the ACS 2020-2024 release publishes
    Connecticut on the 2022 PLANNING REGIONS — 09110-09190 against the old 09001-09015 — so
    879 of 884 Connecticut tracts fail the GEOID join and the whole state silently falls back
    to uniform placement.

    The tracts themselves did not move, only their numbering, so a representative-point join
    from the 2020 polygons onto the 2024 ones recovers the mapping exactly. ancestrydots
    already downloaded the 2024 Connecticut tracts, so nothing has to be fetched.

    Every state that adopts a new county-equivalent scheme will need this; nothing else has yet.
    """
    old_shp = HERE / "data" / "geo" / "tracts2020" / "cb_2020_us_tract_500k.shp"
    new_shp = (ANCESTRYDOTS / "data" / "shapefiles" / "cb_2024_09_tract_500k"
               / "cb_2024_09_tract_500k.shp")
    if not (old_shp.exists() and new_shp.exists()):
        print("  !! Connecticut crosswalk skipped — a tract shapefile is missing; "
              "CT will place uniformly")
        return seg

    import geopandas as gpd

    old = gpd.read_file(old_shp, columns=["GEOID", "STATEFP"])
    old = old[old["STATEFP"] == "09"]
    new = gpd.read_file(new_shp, columns=["GEOID"])
    if new.crs != old.crs:
        new = new.to_crs(old.crs)
    pts = old.copy()
    pts["geometry"] = old.geometry.representative_point()
    j = gpd.sjoin(pts[["GEOID", "geometry"]], new[["GEOID", "geometry"]],
                  how="left", predicate="within", lsuffix="old", rsuffix="new")
    j = j[~j.index.duplicated(keep="first")]
    xw = dict(zip(j["GEOID_new"], j["GEOID_old"]))     # ACS id -> 2020 id
    hit = seg.index.isin(xw)
    moved = seg[hit].rename(index=xw)
    seg = pd.concat([seg[~hit], moved])
    print(f"  Connecticut: {len(moved):,} of {len(j):,} tracts crosswalked "
          f"2022 planning regions -> 2020 counties")
    return seg


def build_tract_segments():
    """Collapse ancestrydots' tract ACS tables into one segment matrix per census tract.

    Reuses ancestrydots' variable→label mapping rather than re-deriving it, so the two maps
    cannot drift apart on what "Assyrian/Chaldean/Syriac" means. `pop` is the ACS tract total
    (B03001_001E), which is a real population and replaces §8.2's equal-share approximation —
    §8.2 already said to prefer a shipped population where the placement layer carries one.
    """
    sys.path.insert(0, str(ANCESTRYDOTS))
    import scatter_dots as sd

    raw = ANCESTRYDOTS / "data" / "raw"
    states = sorted({p.name.split("_")[1].split(".")[0]
                     for p in raw.glob("B04006_*.csv")})
    if not states:
        raise SystemExit(f"no ancestrydots ACS tables under {raw} — run its fetch_data.py --all")

    frames, unmapped = [], {}
    for st in states:
        acc, cg_acc, pop = {}, {}, None
        for table, mapping in sd.ALL_TABLES.items():
            p = raw / f"{table}_{st}.csv"
            if not p.exists():
                continue
            want = set(mapping) | {"state", "county", "tract", f"{table}_001E"}
            df = pd.read_csv(p, usecols=lambda c: c in want,
                             dtype={"state": str, "county": str, "tract": str})
            geoid = df["state"] + df["county"] + df["tract"]
            if table == "B03001":
                pop = pd.to_numeric(df["B03001_001E"], errors="coerce").fillna(0)
                pop.index = geoid
            for var, (label, group) in mapping.items():
                if var not in df.columns:
                    continue
                seg = _seg_of(label, group)
                v = pd.to_numeric(df[var], errors="coerce").fillna(0).clip(lower=0)
                v.index = geoid
                cg_acc[group] = cg_acc.get(group, 0) + v
                if seg is None:
                    unmapped[label] = unmapped.get(label, 0) + 1
                    continue
                acc[seg] = acc.get(seg, 0) + v

        # Race residuals: the people a race question catches whom no ancestry write-in names.
        # Same subtraction ancestrydots makes, so the two agree on who is left over.
        for tbl, seg, groups in (("B02009", "black_resid", sd.BLACK_SUBTRACT_GROUPS),
                                 ("B02008", "white_resid", sd.WHITE_SUBTRACT_GROUPS)):
            p = raw / f"{tbl}_{st}.csv"
            if not p.exists():
                continue
            df = pd.read_csv(p, usecols=lambda c: c in (f"{tbl}_001E", "state", "county", "tract"),
                             dtype={"state": str, "county": str, "tract": str})
            tot = pd.to_numeric(df[f"{tbl}_001E"], errors="coerce").fillna(0)
            tot.index = df["state"] + df["county"] + df["tract"]
            sub = sum((cg_acc.get(g, 0) for g in groups), start=pd.Series(0.0, index=tot.index))
            acc[seg] = (tot - sub.reindex(tot.index).fillna(0)).clip(lower=0)

        wide = pd.DataFrame(acc).fillna(0.0)
        wide.insert(0, "pop", pop.reindex(wide.index).fillna(0.0))
        wide.index.name = "GEOID"
        frames.append(wide)
        print(f"  state {st}: {len(wide):>6,} tracts, {wide['pop'].sum():>11,.0f} people",
              flush=True)

    out = pd.concat(frames).fillna(0.0)
    out = _reindex_connecticut(out)
    out = _add_language_segments(out)
    out = _add_extra_segments(out)
    SEGMENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SEGMENTS_CSV, float_format="%.1f")
    print(f"\n{len(out):,} tracts, {out['pop'].sum():,.0f} people -> {SEGMENTS_CSV}")
    if unmapped:
        print(f"labels with no segment ({len(unmapped)}): {', '.join(sorted(unmapped))}")
    return out


# --------------------------------------------------------------------------------------
# stage 2 — fit, and score by held-out metro
# --------------------------------------------------------------------------------------

def _load_cbsa():
    if not CBSA_XLSX.exists():
        raise SystemExit(f"missing {CBSA_XLSX}\n  download: {CBSA_URL}")
    d = pd.read_excel(CBSA_XLSX, skiprows=2, dtype=str)
    d.columns = [str(c).strip() for c in d.columns]
    fips = d["FIPS State Code"].str.zfill(2) + d["FIPS County Code"].str.zfill(3)
    is_metro = d["Metropolitan/Micropolitan Statistical Area"].str.contains("Metro", na=False)
    return (pd.Series(d["CBSA Code"].values, index=fips.values)[is_metro.values],
            dict(zip(d["CBSA Code"], d["CBSA Title"])))


def _asarb_by_node():
    """ASARB county x taxonomy node, plus the county population it was measured against."""
    from countries import _us_counts

    df = _us_counts()
    y = df.pivot_table(index="unit", columns="node", values="count", aggfunc="sum").fillna(0.0)
    summ = pd.read_excel(HERE / "data" / "raw" / "2020_USRC_Summaries.xlsx",
                         sheet_name="2020 County Summary", dtype={"FIPS": str})
    summ["FIPS"] = summ["FIPS"].astype(str).str.zfill(5)
    pcol = [c for c in summ.columns if "Population" in str(c)][0]
    pop = summ.set_index("FIPS")[pcol].astype(float)
    return y, pop[pop > 0]


def _wcorr(a, b, w):
    a = a - np.average(a, weights=w)
    b = b - np.average(b, weights=w)
    den = np.sqrt(np.sum(w * a * a) * np.sum(w * b * b))
    return float(np.sum(w * a * b) / den) if den > 0 else np.nan


def _ridge(A, b, lam_frac):
    """Ridge on the RAW shares — deliberately, and this is the one tuning decision here that
    was made by measurement rather than by taste.

    A single penalty against columns whose scale differs by three orders of magnitude —
    `euro_catholic` is a fifth of some counties, `armenian` a twentieth of one percent of
    almost all of them — shrinks the small ones far harder. That is usually described as a
    bug, and standardising the columns is the textbook fix. Swept both ways:

        raw  λ=0.15   26 nodes   66.6M adherents   mean r 0.42   median slope 0.92
        std  λ=0.05   12 nodes   40.4M adherents   mean r 0.36   median slope 0.37

    Standardising recovers `armenian` beautifully — it goes from rank 15 to rank 1 with a
    +0.55 coefficient — and wrecks everything else, because a slope of 0.37 means the
    predictions are running about three times as far as the truth. Shrinking a small segment
    harder is not a defect, it is the correct response to there being less information in it:
    at county scale `armenian` genuinely has almost no variance to learn from.

    So neither penalty can serve both cases, and the honest reading is that a body defined by
    a 0.2%-of-population ethnicity is NOT LEARNABLE from between-county variation at all. That
    is what AUTHORED is for. λ = 0.15 is where the raw sweep puts the slope nearest 1.
    """
    k = A.shape[1]
    G = A.T @ A
    lam = lam_frac * np.trace(G) / k
    return np.linalg.solve(G + lam * np.eye(k), A.T @ b)


def fit(seg_tracts=None, verbose=True):
    """Fit one ridge per node on within-metro deviations; score by held-out whole metro."""
    if seg_tracts is None:
        seg_tracts = pd.read_csv(SEGMENTS_CSV, dtype={"GEOID": str}).set_index("GEOID")
    y, pop = _asarb_by_node()
    cbsa, cbsa_name = _load_cbsa()

    county = seg_tracts.groupby(seg_tracts.index.str[:5]).sum()
    idx = county.index.intersection(y.index).intersection(pop.index).intersection(cbsa.index)
    grp = cbsa.reindex(idx)
    size = pop.reindex(idx).groupby(grp).sum().sort_values(ascending=False)
    ncty = grp.value_counts()
    keep = [c for c in size.index[:60] if ncty[c] >= 3]
    idx = idx[grp.isin(keep).values]
    grp = cbsa.reindex(idx)

    cpop = pop.reindex(idx)
    C = county.reindex(idx)
    Xs = design(C, cpop)

    def demean(s):
        m = s.groupby(grp).transform(lambda h: np.average(h, weights=cpop.reindex(h.index)))
        return s - m

    Xd = Xs.apply(demean).to_numpy(float)
    P = cpop.to_numpy(float)
    metros = pd.factorize(grp)[0]
    fold = np.random.default_rng(FOLD_SEED).integers(0, N_FOLDS, metros.max() + 1)[metros]

    if verbose:
        print(f"fitting on {len(idx):,} counties in {metros.max() + 1} metros "
              f"({P.sum() / 1e6:.0f}M people)\n")
        print(f"{'node':<40} {'adherents':>12} {'held-out r':>11} {'slope':>7}  use")
        print("-" * 84)

    model, rows = {}, []
    for node in y.columns:
        adh = float(y[node].reindex(idx).fillna(0).sum())

        # Authored first, and with no r reported at all: these are not fitted, so there is no
        # held-out number to quote and quoting one would imply a validation that did not
        # happen. They also skip MIN_ADHERENTS — a body of 2,817 Knanaya Catholics is small
        # precisely because it is one community, which is the case authoring exists for.
        if node in AUTHORED:
            model[node] = {"basis": "authored", "affinity": AUTHORED[node]}
            rows.append((node, adh, np.nan, np.nan, True))
            if verbose:
                print(f"{node[:39]:<40} {adh:>12,.0f} {'—':>11} {'—':>7}  "
                      f"AUTHORED -> {'+'.join(AUTHORED[node])}")
            continue

        if adh < MIN_ADHERENTS:
            continue
        yd = demean(y[node].reindex(idx).fillna(0) / cpop).to_numpy(float)
        if not np.isfinite(yd).all() or yd.std() == 0:
            continue
        pred = np.zeros_like(yd)
        for k in range(N_FOLDS):
            tr, te = fold != k, fold == k
            if not tr.any() or not te.any():
                continue
            beta = _ridge(Xd[tr] * np.sqrt(P[tr])[:, None], yd[tr] * np.sqrt(P[tr]), RIDGE)
            pred[te] = Xd[te] @ beta
        r = _wcorr(yd, pred, P)
        pp = np.sum(P * pred * pred)
        slope = float(np.sum(P * yd * pred) / pp) if pp > 0 else np.nan
        beta = _ridge(Xd * np.sqrt(P)[:, None], yd * np.sqrt(P), RIDGE)
        use = np.isfinite(r) and r >= R_MIN
        circular = node in ESTIMATE_NODES
        if use:
            model[node] = {"basis": "fitted", "r": round(r, 3), "slope": round(slope, 3),
                           "r_is_circular": circular,
                           "beta": {s: float(b) for s, b in zip(FIT_SEGS, beta)}}
        rows.append((node, adh, r, slope, use))
        if verbose:
            note = "weights" if use else "—"
            if circular:
                note += "  (r not independent — ASARB figure is itself an estimate)"
            print(f"{node[:39]:<40} {adh:>12,.0f} {r:>11.2f} {slope:>7.2f}  {note}")

    if verbose:
        tot = sum(a for _, a, _, _, _ in rows)
        won = sum(a for _, a, _, _, u in rows if u)
        n_auth = sum(1 for n in model if model[n].get("basis") == "authored")
        print(f"\n{len(model) - n_auth} fitted nodes clear r >= {R_MIN}, {n_auth} authored — "
              f"{won / 1e6:.1f}M of {tot / 1e6:.1f}M adherents ({won / tot:.0%}); "
              f"the rest stay population-uniform")
    return model, rows


def fit_residual(seg_tracts=None, verbose=True):
    """spec §8.4a — a SEPARATE model for the §3.5a residual, fitted against CES.

    The roll model must not be reused here and countries.py says why: applying a model of
    where ASARB's adherents live to the people on nobody's roll would place the residual
    exactly where the measured people already are, which is the one place they are not. So
    the residual gets its own coefficients from its own target.

    Same protocol otherwise — demeaned within metro, whole metros held out. Two differences,
    both because the target is a survey rather than a census:
      - counties are weighted by CES sample size, not population, since that is the precision
        of the target;
      - only counties with enough respondents are scored, because a share off 30 people is
        mostly noise and would understate any model.
    """
    if seg_tracts is None:
        seg_tracts = pd.read_csv(SEGMENTS_CSV, dtype={"GEOID": str}).set_index("GEOID")
    _, pop = _asarb_by_node()
    cbsa, _ = _load_cbsa()
    ces = _ces_targets()

    county = seg_tracts.groupby(seg_tracts.index.str[:5]).sum()
    idx = county.index.intersection(ces.index).intersection(pop.index).intersection(cbsa.index)
    grp = cbsa.reindex(idx)
    size = pop.reindex(idx).groupby(grp).sum().sort_values(ascending=False)
    ncty = grp.value_counts()
    keep = [c for c in size.index[:60] if ncty[c] >= 3]
    idx = idx[grp.isin(keep).values]
    grp = cbsa.reindex(idx)

    C, cpop = county.reindex(idx), pop.reindex(idx)
    Xs = design(C, cpop)

    def demean(s):
        m = s.groupby(grp).transform(lambda h: np.average(h, weights=cpop.reindex(h.index)))
        return s - m

    Xd = Xs.apply(demean).to_numpy(float)
    P = cpop.to_numpy(float)
    N = ces["n"].reindex(idx).fillna(0).to_numpy(float)
    metros = pd.factorize(grp)[0]
    fold = np.random.default_rng(FOLD_SEED).integers(0, N_FOLDS, metros.max() + 1)[metros]
    fit_m, score_m = N >= CES_MIN_N_FIT, N >= CES_MIN_N_SCORE

    if verbose:
        print(f"\nresidual model (§8.4a), CES-validated: {len(idx):,} metro counties, "
              f"{int(score_m.sum()):,} with n >= {CES_MIN_N_SCORE} to score against\n")
        print(f"{'node':<24}{'CES share':>11}{'held-out r':>12}{'slope':>8}  use")
        print("-" * 62)

    model = {}
    for node in [c for c in ces.columns if c != "n"]:
        yd = demean(ces[node].reindex(idx).fillna(0)).to_numpy(float)
        pred = np.zeros_like(yd)
        for k in range(N_FOLDS):
            tr, te = (fold != k) & fit_m, fold == k
            if not tr.any():
                continue
            wt = np.sqrt(N[tr])
            pred[te] = Xd[te] @ _ridge(Xd[tr] * wt[:, None], yd[tr] * wt, RIDGE)
        a, b, w = yd[score_m], pred[score_m], P[score_m]
        r = _wcorr(a, b, w)
        bb = float(np.sum(w * b * b))
        slope = float(np.sum(w * a * b) / bb) if bb > 0 else np.nan
        wt = np.sqrt(N[fit_m])
        beta = _ridge(Xd[fit_m] * wt[:, None], yd[fit_m] * wt, RIDGE)
        use = np.isfinite(r) and r >= R_MIN
        if use:
            model[node] = {"basis": "fitted_ces", "r": round(r, 3), "slope": round(slope, 3),
                           "beta": {s: float(v) for s, v in zip(FIT_SEGS, beta)}}
        if verbose:
            share = float(np.average(ces[node].reindex(idx).fillna(0), weights=P))
            print(f"{node:<24}{share:>10.1%}{r:>12.2f}{slope:>8.2f}  "
                  f"{'weights' if use else '—'}")
    if verbose:
        print(f"\n{len(model)} residual nodes clear r >= {R_MIN}; the rest stay "
              f"population-uniform")
    return model


# --------------------------------------------------------------------------------------
# stage 3 — the weighter scatter.py calls
# --------------------------------------------------------------------------------------

def _cap_to_population(w, pop, count, iters=4):
    """No tract may be handed more adherents of one body than it has residents.

    §8.3's parish model put 13% of Cook County's Catholic dots into tracts it implied were over
    100% Catholic, and that was the first sign the model was wrong. The fitted track cannot do
    it — it clips a RATE, so its weight is bounded by population by construction. The authored
    track can, because it weights on a segment share with no reference to the county rate at
    all: a body concentrated on a small marker in a big county will pile onto a few tracts.

    So: convert the weight to implied people, clip anything over the tract's population, and
    hand the excess to the tracts with room, a few times. Converges immediately in practice and
    is a no-op for every body that was not going to overflow.
    """
    w = np.asarray(w, dtype=float)
    if count <= 0 or w.sum() <= 0:
        return w
    for _ in range(iters):
        implied = w / w.sum() * count
        over = implied > pop
        if not over.any():
            break
        # freeze the full tracts at their population, redistribute the rest by weight
        room = ~over
        if not room.any() or w[room].sum() <= 0:
            break
        spill = float((implied[over] - pop[over]).sum())
        w = w.copy()
        w[over] = pop[over] / max(count, 1e-9) * w.sum()
        w[room] = w[room] * (1 + spill / max(implied[room].sum(), 1e-9))
    return w


class Weighter:
    """Per-(county, node) tract weights.

    weight = tract population x predicted rate, where

        predicted rate = county rate + (tract segment shares - county segment shares) . beta

    The county rate is ASARB's own, so the level never moves and raking is automatic; beta only
    redistributes. Nodes with no beta fall back to population alone, which is still better than
    §8.2's equal-share-per-tract — the ACS tract total is a real population and tracts run
    1,200–8,000 people.
    """

    def __init__(self, geoids, model=None, segments=None):
        blob = (json.loads(MODEL_JSON.read_text(encoding="utf-8"))
                if MODEL_JSON.exists() else {})
        if model is None:
            model = blob.get("nodes", {})
        # §8.4a: the residual gets its own coefficients, fitted against CES rather than ASARB.
        self.residual_model = blob.get("residual_nodes", {})
        if segments is None:
            segments = pd.read_csv(SEGMENTS_CSV, dtype={"GEOID": str}).set_index("GEOID")
        self.model = model
        geoids = pd.Index(geoids).astype(str)
        seg = segments.reindex(geoids)
        # Tracts the ACS does not cover — chiefly Connecticut, whose 2020 tract GEOIDs carry the
        # old county codes while ACS 2020-2024 publishes them on the 2022 planning regions, so
        # the join fails for the whole state. Those fall back to population, and with no
        # population either they fall back to equal shares.
        self.missing = seg["pop"].isna().to_numpy()
        self.pop = seg["pop"].fillna(0.0).to_numpy(float)
        # Carry EVERY segment, not just the fitted subset: AUTHORED reaches for `sikh_resp`,
        # which is deliberately not in the design matrix but is exactly the right column for
        # a Sikh gurdwara.
        # Same two denominators as fit(): ancestry against its own total, language against
        # population. If these disagree with fit(), every coefficient is silently misapplied.
        # TWO matrices, and they are not interchangeable.
        #
        # `X` is design() — the exact columns, order and denominators the ridge was fitted on,
        # so a beta means here what it meant there. Building it by hand instead was a silent
        # bug: `ba_share` and `age` were not in the hand-built list, `beta.get(name, 0.0)`
        # returned 0 for both, and the residual model's two strongest predictors were dropped
        # on the floor. The dots came out flat across education deciles (0.98x top to bottom)
        # while the model itself was predicting a 2.5x spread, and only a check against the
        # DRAWN OUTPUT caught it — the weights, inspected directly, looked fine.
        #
        # `A` is every raw segment as a share of the ancestry total, for AUTHORED affinities,
        # which reach for columns design() deliberately leaves out (`sikh_resp`, `white_resid`).
        p = seg["pop"]
        self.X = design(seg.fillna(0.0), p).to_numpy(float)
        self.fit_pos = {s: i for i, s in enumerate(FIT_SEGS)}
        anc = [c for c in seg.columns
               if c not in set(LANG_SEGS) | {"pop", "ba_plus", "adults25", "age_x_pop"}]
        a = seg[anc].fillna(0.0)
        tot = a.sum(axis=1)
        self.all_segs = anc + LANG_SEGS
        self.seg_pos = {s: i for i, s in enumerate(self.all_segs)}
        lg = seg.reindex(columns=LANG_SEGS).fillna(0.0).div(p.where(p > 0), axis=0)
        self.segshare = pd.concat([a.div(tot.where(tot > 0), axis=0), lg],
                                  axis=1).fillna(0.0).to_numpy(float)
        self.n_weighted = 0
        self.n_authored = 0
        self.n_residual = 0
        self.n_uniform = 0

    def weights(self, node, idx, count, plain=False):
        """Weights for the tracts at positions `idx`, given the unit's adherent count.

        Returns None to mean "no better idea than equal shares" — which happens only where the
        placement layer has no population at all, e.g. a state the ACS join missed entirely.

        `plain` means the row is not a measurement (spec §7), and the answer is then tract
        population and nothing else. Every beta below was fitted to predict ASARB's OWN
        within-metro variation, so it says where the people on a roll live; §3.5a's residual
        is by construction the people on no roll, and running it through this model would
        place them on top of the congregations they are defined by not belonging to. The real
        tract populations still apply — that half of §8.4 is a measurement of everybody.
        """
        pop = self.pop[idx]
        cpop = pop.sum()
        if cpop <= 0:
            self.n_uniform += 1
            return None
        # §8.4a: a derived row is the people on nobody's roll, so it gets the
        # CES-fitted residual model rather than the roll model — never the roll
        # model, and no longer nothing at all.
        m = (self.residual_model if plain else self.model).get(node)
        if m is None:
            self.n_uniform += 1
            return pop
        S = self.segshare[idx]

        if m.get("basis") == "authored":
            # Weight straight on the segment, no base rate and no deviation: the claim is that
            # these people ARE that community, so the dots go where the community is. Falls
            # back to population where the county holds none of the segment at all — an
            # Armenian parish in a county the ACS records no Armenians in is a real thing,
            # and refusing to draw it would be worse than drawing it flat.
            aff = np.zeros(len(self.all_segs))
            floor = 0.0
            for seg, v in m["affinity"].items():
                if seg == "floor":
                    floor = float(v)
                elif seg in self.seg_pos:
                    aff[self.seg_pos[seg]] = float(v)
            w = pop * (S @ aff + floor)
            if w.sum() <= 0:
                self.n_uniform += 1
                return pop
            self.n_authored += 1
            return _cap_to_population(w, pop, count)

        residual = m.get("basis") == "fitted_ces"
        # design() columns, not the affinity matrix — see the note in __init__.
        Xi = self.X[idx]
        missing = [s for s in m["beta"] if s not in self.fit_pos]
        if missing:
            raise KeyError(f"{node}: beta names not in the design matrix: {missing[:5]}")
        beta = np.array([m["beta"].get(s, 0.0) for s in FIT_SEGS], dtype=float)
        cshare = np.average(Xi, axis=0, weights=pop)
        rate = count / cpop + (Xi - cshare) @ beta
        # A rate cannot be negative and cannot exceed everyone in the tract. Clipping is what
        # keeps the linear model honest out at tract-level segment shares, which run far past
        # anything it was fitted on.
        rate = np.clip(rate, 0.0, 1.0)
        w = pop * rate
        if w.sum() <= 0:                      # county rate 0, or clipped flat — nothing to say
            self.n_uniform += 1
            return pop
        if residual:
            self.n_residual += 1
        else:
            self.n_weighted += 1
        return w


def load_weighter(place):
    """countries.py hook. `place` is the tract GeoDataFrame scatter.py has already read."""
    if not SEGMENTS_CSV.exists():
        print(f"  !! {SEGMENTS_CSV.name} missing — run `python us_weights.py --build`; "
              f"placing uniformly (§8.2)")
        return None
    if not MODEL_JSON.exists():
        print(f"  !! {MODEL_JSON.name} missing — run `python us_weights.py --build`; "
              f"placing on tract population only")
    return Weighter(place["GEOID"].to_numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true",
                    help="rebuild tract segments and refit; writes both artefacts")
    ap.add_argument("--refit", action="store_true",
                    help="refit from the cached tract segments")
    ap.add_argument("--report", action="store_true",
                    help="fit and print the validation table, write nothing")
    args = ap.parse_args()

    if args.build:
        print("building tract segments from ancestrydots ACS tables…")
        build_tract_segments()

    if args.build or args.refit or args.report:
        model, rows = fit()
        residual = fit_residual()
        if not args.report:
            MODEL_JSON.parent.mkdir(parents=True, exist_ok=True)
            MODEL_JSON.write_text(json.dumps({
                "_": "spec §8.4. Redistributes a county's dots across its tracts; the county "
                     "total is ASARB's and never moves. basis=fitted carries `beta` and the "
                     "held-out-metro correlation `r` that earned it; basis=authored carries "
                     "`affinity` and no r, because it was asserted rather than measured.",
                "r_min": R_MIN, "ridge": RIDGE, "segments": FIT_SEGS, "nodes": model,
                "__residual": "spec §8.4a. Applied to `derived` rows only — the §3.5a "
                              "residual, the people on nobody's roll. Fitted against CES "
                              "county shares, never against ASARB.",
                "residual_nodes": residual,
            }, indent=1), encoding="utf-8")
            print(f"\nwrote {len(model)} roll models and {len(residual)} residual models "
                  f"-> {MODEL_JSON}")
    else:
        ap.error("nothing to do: pass --build, --refit or --report")


if __name__ == "__main__":
    main()
