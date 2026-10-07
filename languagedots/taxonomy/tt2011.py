"""Trinidad and Tobago, Census 2011: ethnic group of the native-born and country of birth of the
foreign-born -> node. No language question; every row `derived` (sources/tt.md). Built as
Barbados (taxonomy/bb2010.py).

  native-born, Caucasian       English (as bb's white Barbadians)
  native-born, everyone else   Trinidadian Creole English (trin1276); in Tobago (CSO area 98)
                               Tobagonian Creole English (toba1282). Indo-Trinidadians included:
                               Trinidad Bhojpuri (Glottolog trin1268, a dialect of Caribbean
                               Hindustani) survives among a few elderly speakers, and no source
                               gives a figure, so none is drawn.
  foreign-born                 by country of birth: the English-speaking Caribbean on its own
                               creole (bb.txt's nodes), Saint Lucia on Antillean Creole (dm.md's
                               Kweyol node), UK, US and Canada English; the rest via
                               sources/origin_mix.py; CSO's pooled "Other" on `other`.
"""
CR = "creole.english_based"
EN = "indoeuropean.germanic.english"
TRINIDADIAN = f"{CR}.trinidadian"
TOBAGONIAN = f"{CR}.tobagonian"

COUNTRY = {
    "Barbados": f"{CR}.bajan", "Grenada": f"{CR}.grenadian", "Guyana": f"{CR}.guyanese",
    "Jamaica": f"{CR}.jamaican", "Saint Vincent and the Grenadines": f"{CR}.vincentian",
    "Saint Lucia": "creole.french_based.antillean",
    "United Kingdom": EN, "United States of America": EN, "Canada": EN,
    "Other": "other",
}
ORIGIN = {"China": "CN", "India": "IN", "Venezuela": "VE"}
CODES = {**COUNTRY, "native": TRINIDADIAN, "native Tobago": TOBAGONIAN, "native Caucasian": EN}


def native(ethnic, area):
    if ethnic == "Caucasian":
        return {EN: 1.0}
    return {TOBAGONIAN if area == "98" else TRINIDADIAN: 1.0}


def born_in(country):
    if country in COUNTRY:
        return {COUNTRY[country]: 1.0}
    if country in ORIGIN:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sources"))
        import origin_mix
        return origin_mix.mix(ORIGIN[country], "tt")
    raise SystemExit(f"tt2011: country of birth {country!r} not planned")
