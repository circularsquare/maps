"""Shared tables for the Kazakhstan fetcher and boundary prep.

Units are keyed by the current KATO code (classifier of administrative-
territorial objects, edition of 18.09.2026): 9 digits, the first two the region.
Level 1 is the region's code (e.g. 100000000 = Abay), level 2 the rayon's or
city administration's code (e.g. 103400000 = Aksuat district).
"""
import re
from pathlib import Path

HELPER = Path(__file__).resolve().parents[2]
REPO = HELPER.parent
DATA = HELPER / "data" / "kazakhstan"
RAW = DATA / "raw"

KATO_XLSX = RAW / "KATO_18.09.2026.xlsx"
KATO_URL = ("https://stat.gov.kz/upload/iblock/a7d/eqogm92xc0udumlzfzffhcrutk0fjhb4/"
            "%D0%9A%D0%90%D0%A2%D0%9E_18.09.2026.xlsx")

# Region code (first two KATO digits) -> English name used in the viewer.
REGIONS = {
    "10": "Abay Region", "11": "Akmola Region", "15": "Aktobe Region",
    "19": "Almaty Region", "23": "Atyrau Region", "27": "West Kazakhstan Region",
    "31": "Jambyl Region", "33": "Jetisu Region", "35": "Karaganda Region",
    "39": "Kostanay Region", "43": "Kyzylorda Region", "47": "Mangystau Region",
    "55": "Pavlodar Region", "59": "North Kazakhstan Region", "61": "Turkistan Region",
    "62": "Ulytau Region", "63": "East Kazakhstan Region", "71": "Astana",
    "75": "Almaty", "79": "Shymkent",
}

# OCHA COD-AB adm1 English name -> region code (used only to place OSM rayons
# in their region, so twins like the three "Abay District"s resolve).
COD_ADM1 = {v: k for k, v in REGIONS.items()}

# Bulletin region headers -> region code. Matched after norm().
BULLETIN_REGIONS = {
    "абаи": "10", "акмолинская": "11", "актюбинская": "15", "алматинская": "19",
    "атырауская": "23", "западноказахстанская": "27", "жамбылская": "31",
    "жетису": "33", "карагандинская": "35", "костанаиская": "39",
    "кызылординская": "43", "мангистауская": "47", "павлодарская": "55",
    "североказахстанская": "59", "туркестанская": "61", "улытау": "62",
    "восточноказахстанская": "63", "астана": "71", "алматы": "75", "шымкент": "79",
}

# Units merged in the shipped geography, because no boundary exists for the
# current split. Key = shipped code, value = the KATO units summed into it.
#  * Shymkent's Turan district (791910000) was created in 2022 and the other
#    four were redrawn around it; OSM has only the four pre-2022 districts, and
#    their 2025 populations no longer fit those shapes (Al-Farabi 191,578 in
#    2021 against 266,806 in 2025). So Shymkent ships as one unit.
MERGES = {
    "790000000": ["791110000", "791310000", "791510000", "791710000", "791910000"],
}
MERGED_NAMES = {
    "790000000": "Shymkent (all five districts)",
}

# Units OSM lacks that are exactly the part of a city no OSM district covers.
# Astana's Saraishyk district (split from Almaty district, 29 Jan 2025): OSM's
# Almaty district is already the reduced one (93 km² by our measure against
# the 8,518 ha published for it after the split), and Astana minus its five
# OSM districts leaves 72 km² in the south-east, against Saraishyk's published
# 6,953 ha. Value = OSM relation id of the city.
CITY_GAPS = {"711610000": 3087155}

_TR = str.maketrans("әіңғүұқөһйё", "аингуукохие")


def norm(x):
    """Lower-case Cyrillic key: Kazakh letters folded to Russian, no spaces or
    punctuation, and the type words (district, city administration) dropped."""
    x = str(x).lower().replace("ё", "е")
    x = re.sub(r"г\.\s*а\.|городская администрация|городской акимат|"
               r"\bрайон\b|\bим\.|\bимени\b|\bг\.|\bгород\b|\bобласть\b", "", x)
    return re.sub(r"[^а-я0-9]", "", x.translate(_TR))


def norm_settlement(x):
    """Settlement-name key: drops the type prefix (с., п., г., ст., рзд., ...)."""
    x = str(x).lower().translate(_TR).strip()
    x = re.sub(r"^(с\.а\.|п\.а\.|с\.|п\.|г\.|город|уч\.|ст\.|рзд\.|разъезд)\s*", "", x)
    return re.sub(r"[^а-я0-9]", "", x)


def load_kato():
    """KATO 2026 as a DataFrame of 9-digit rows with an `l2` column (the
    level-2 unit each row belongs to) and `level` (1, 2, or 3 = anything below)."""
    import pandas as pd
    k = pd.read_excel(KATO_XLSX, dtype=str)
    k = k[k.te.str.len() == 9].copy()
    city = k.ab.isin(["71", "75", "79"])
    k["l2"] = k.te.str[:4] + "00000"
    k.loc[city, "l2"] = k.te.str[:6] + "000"
    k["level"] = 3
    k.loc[(k.cd == "00") & (k.ef == "00") & (k.hij == "000"), "level"] = 1
    k.loc[~city & (k.cd != "00") & (k.ef == "00") & (k.hij == "000"), "level"] = 2
    k.loc[city & (k.cd != "00") & (k.hij == "000"), "level"] = 2
    k.loc[k.level == 1, "l2"] = None
    return k


def shipped_code(kato_l2):
    """The level-2 code a KATO unit is shipped under (itself unless merged)."""
    for key, members in MERGES.items():
        if kato_l2 in members:
            return key
    return kato_l2
