"""Austria: RINF's ids are ÖBB's route numbers ("10101"), which ÖBB and Wikidata's P1671 write
"101 01"; names are Wikidata's for that number. The seven other infrastructure managers have
no name in RINF; theirs below are read off the lines they hold."""
import re

COUNTRY = {
    "iso3": "AUT", "wikidata": "Q40", "langs": ["de", "en"],
    "ref": lambda lid: lid if re.fullmatch(r"\d{5}", lid or "") else None,
    # The id IS the number. OSM's route=railway relations here carry Kursbuch
    # (timetable) numbers, 100, 300, 901, which must never override it.
    "rule_certain": True,
    "ref_display": lambda k: f"{k[:3]} {k[3:]}" if re.fullmatch(r"\d{5}", k) else k,
    "generic_label": r"^Bahnstrecke\b",
    "im": {"0081_IM": "ÖBB-Infrastruktur", "3023_IM": "Steiermärkische Landesbahnen",
           "3035_IM": "Montafonerbahn", "3787_IM": "Raaberbahn",
           "3782_IM": "Neusiedler Seebahn", "3865_IM": "Hafen Krems",
           "3764_IM": "WienCont", "3882_IM": "Linz AG"},
}
