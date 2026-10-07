"""Denmark's rules for build_model.py (country_rules): which route=train relations are named
trains rather than lines.

What OSM Denmark has (tags of all 211 Danish rail route relations read 2026-10-03): DSB's
InterCity and InterCityLyn by line number ("InterCity 1: København => Aalborg", "InterCityLyn
5: Københavns Lufthavn => Sønderborg", "IC 81: Flensburg => Fredericia"), all tagged
service=long_distance; the regional trains ("Regionaltog 50", "RE69", "DSB Vores Tog Struer -
Thisted", "Aarhus - Fredericia - Odense"); Lokaltog's numbered lines ("Lokaltog 920R: Hillerød =>
Hundested"); Midtjyske Jernbaner's "Lokalbane 92/93"; Øresundståg ("Train 90: Helsingør =>
København => Malmø => Karlskrona"); SJ's "Tog 80: København => Stockholm"; Arriva/NAH.SH's "RB
66: Niebüll => Tønder"; and the S-tog A-H, Metro M1-M4 and the light rails as their own route
kinds. All of these are interval products a rider uses as lines, SJ's Stockholm table included
(Sweden's rule keeps SJ's tables as lines, and it is the same relation on both sides).

Named trains: the international EC/ICE/EuroCity Express and night trains (Hamburg, Berlin, the
Stockholm - Berlin night trains: Snälltåget, SJ EuroNight), none of which OSM maps in Denmark
today, so the rule is written for them by name; anything tagged service=night or car; and the
Tønder - Højer museum train (route 21097554, "Tønder-Højer-banen"; summer heritage trips, not
scheduled weekly), which has no percentage of its own that way.

service=long_distance is NOT a marker here: every DSB InterCity carries it.
"""
import re

DK_TRAIN = re.compile(
    r"^(?:Train\s+|Tog\s+|Tåg\s+|Zug\s+)?(?:EC|ECE|EN|NJ|ICE(?!\s?\d{1,2}(?:\.\d)?(?!\d)))"
    r"(?:[\s\d:]|$)"
    r"|\bEuro(?:City|Night)\b|\bNightjet\b|\bEuropean Sleeper\b|\bSnälltåget\b"
    r"|\bNatt(?:åg|og)\b|\bNachtzug\b|\bBerlin Night Express\b"
    r"|^Tønder-Højer-banen\b")
DK_TRAIN_OPERATORS = {"Snälltåget", "Snälltåget AB"}


def looks_like_service(tags, name, name_en):
    if set((tags.get("service") or "").split(";")) & {"night", "car"}:
        return True
    if tags.get("operator") in DK_TRAIN_OPERATORS:
        return True
    return bool(DK_TRAIN.search(name or "") or DK_TRAIN.search(tags.get("ref") or ""))
