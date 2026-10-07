"""Italy, 2025: language labels written by sources/it_istat.py -> node.

Italy's census asks no language, so every label here is the build's own, not a census category
(sources/it.md says how each is derived):
  * "Italian": everyone the proxies do not place elsewhere, plus South Tyrol's Italian group.
  * the Italo-Romance languages ISTAT's 2024 survey calls "dialetto", named by where the
    respondent lives (sources/it_regional.py); Tuscan, Romanesco and the other central dialects
    are Italian in Glottolog and are drawn as Italian.
  * minority languages: German and Ladin (South Tyrol 2024), Ladin, Mocheno and Cimbrian
    (Trentino 2021), Slovenian, Griko and Calabrian Greek (survey knowledge x family use),
    Arbereshe, Slavomolisano, Occitan, Franco-Provencal, Gallurese, Sassarese, Catalan
    (Alghero) and Ligurian (Tabarchino) by the comuni that speak them.
  * immigrant languages: ISTAT's foreign residents by citizenship, each country on its main
    language (France's table, sources/fr_build.py COUNTRY_LANG, with it_istat.ITALY_OVERRIDES).
"""
import fr2023

RO = "indoeuropean.romance"

NAMES = dict(fr2023.NAMES)
NAMES.update({
    "Neapolitan": f"{RO}.neapolitan",
    "Sicilian": f"{RO}.sicilian",
    "Venetian": f"{RO}.venetian",
    "Lombard": f"{RO}.lombard",
    "Piedmontese": f"{RO}.piedmontese",
    "Ligurian": f"{RO}.ligurian",
    "Emilian": f"{RO}.emilian",
    "Romagnol": f"{RO}.romagnol",
    "Friulian": f"{RO}.friulian",
    "Ladin": f"{RO}.ladin",
    "Sardinian": f"{RO}.sardinian",
    "Gallurese": f"{RO}.gallurese",
    "Sassarese": f"{RO}.sassarese",
    "Franco-Provencal": f"{RO}.francoprovencal",
    "Arbereshe": "indoeuropean.albanian.arberesh",
    # Griko (Salento) and Greko (Bova): one language in Glottolog, apul1236
    "Griko": "indoeuropean.hellenic.italiot_greek",
    "Calabrian Greek": "indoeuropean.hellenic.italiot_greek",
    "Slavomolisano": "indoeuropean.slavic.south.slavomolisano",
    "Mocheno": "indoeuropean.germanic.continental.mocheno",
    "Cimbrian": "indoeuropean.germanic.continental.cimbrian",
    "Sinhala": "indoeuropean.indoaryan.sinhala",
})
EXTRA_NODES = []


def resolve(label):
    if label[:1].islower():   # a node id, from sources/origin_mix.py
        return label
    if label not in NAMES:
        raise KeyError(f"it2025: unmapped label {label!r}")
    return NAMES[label]
