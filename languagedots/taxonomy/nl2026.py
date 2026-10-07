"""Netherlands, 2026: language labels written by sources/nl_build.py -> node.

The Dutch census asks no language, so every label here is the build's own (sources/nl.md):
  * "Dutch": everyone the proxies do not place elsewhere, plus the "dialect" answers Glottolog
    files under Dutch (Brabants, Hollandic, Town Frisian, Bildts...).
  * regional languages from CBS's 2019 survey of the language most spoken at home, named by
    where the respondent lives, following Glottolog: Frisian (Western Frisian), Gronings,
    Westphalian (Glottolog's Westphalic: Drents, Twents, Sallands, Achterhoeks, Stellingwerfs,
    Veluws), Limburgish, Zeeuws.
  * immigrant languages: CBS's population by country of origin, each country on its main
    language (France's table, sources/fr_build.py COUNTRY_LANG, with nl_build.NL_OVERRIDES and
    SPLITS). "Chinese" (origin China) is the Sinitic group: the source names a country.
"""
import fr2023

GE = "indoeuropean.germanic"

NAMES = dict(fr2023.NAMES)
NAMES.update({
    # Western Frisian (west2354). Canada and Finland already draw "Frisian" on this node.
    "Frisian": f"{GE}.frisian",
    "Gronings": f"{GE}.continental.lowgerman.gronings",          # gron1242
    "Westphalian": f"{GE}.continental.lowgerman.westphalian",    # west2356
    "Limburgish": f"{GE}.continental.limburgish",                # limb1263
    "Zeeuws": f"{GE}.continental.zeeuws",                        # zeeu1238
    "Afrikaans": f"{GE}.continental.afrikaans",
    "Sranan Tongo": "creole.english_based.sranan",
    "Sarnami": "indoeuropean.indoaryan.bihari.sarnami",
    "Javanese": "austronesian.javanese",                         # Caribbean Javanese
    "Kurdish": "indoeuropean.iranian.kurdish",
    "Pashto": "indoeuropean.iranian.pashto",
    "Cantonese": "sinotibetan.sinitic.cantonese",
})
EXTRA_NODES = []


def resolve(label):
    if label[:1].islower():   # a node id, from sources/origin_mix.py
        return label
    if label not in NAMES:
        raise KeyError(f"nl2026: unmapped label {label!r}")
    return NAMES[label]
