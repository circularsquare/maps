"""Cyprus, Census of Population and Housing 2021, native language -> node. Keyed by the label
as CYSTAT-DB table 1891616E prints it (sources/cy_census.py keeps it verbatim, misspellings
included: `Ukranian`, `Moldovian`).

34 rows: 32 named, `Other languages`, `Not stated`. No new nodes; every language already has
its node elsewhere on the map.

CALLS (sources/cy.md says more):
  * `Indian` (8,370) and `Sri Lankan` (6,060) name a country, not a language. Indian could be
    Hindi, Punjabi, Malayalam or any other language of India, Indo-Aryan or Dravidian; Sri
    Lankan could be Sinhala or Tamil. The narrowest node holding either is the root, so both go
    on `other`, as us2024 ("India N.E.C.") and zm2022 ("Indian") already file it.
  * `Filipino` on Filipino, `Nepalese` on Nepali, `Persian` on Persian, `Chinese` on Sinitic
    (drawn unwashed, build.py UNWASHED), `Moldovian` on Moldovan, `Yugoslavian` on
    Serbo-Croatian (au2021's "Serbo-Croatian/Yugoslavian" did the same), `Slovakian` on Slovak.
  * Greek is Greek: the census does not ask about Cypriot Greek and nothing here splits it.
  * Cypriot Maronite Arabic and Kurbetcha (the Roma language the ethnic-group question names as
    Roma/Gurbeti) are not printed; whoever named them is inside `Other languages` or a
    neighbouring row, and no table says which.
  * `Other languages` (3,491) on `other`. No indigenous language of Cyprus could be in it apart
    from the two above, so there is nothing to keep apart from it.
  * `Not stated` (11,111, 1.2%) is not drawn.
"""
IE = "indoeuropean"
NAMES = {
    "Greek": f"{IE}.hellenic.greek",
    "English": f"{IE}.germanic.english",
    "Russian": f"{IE}.slavic.east.russian",
    "Arabic": "afroasiatic.arabic",
    "Romanian": f"{IE}.romance.romanian",
    "Bulgarian": f"{IE}.slavic.south.bulgarian",
    "Indian": "other",                       # a country, not a language: see above
    "Filipino": "austronesian.philippine.filipino",
    "Sri Lankan": "other",                   # Sinhala or Tamil: see above
    "Ukranian": f"{IE}.slavic.east.ukrainian",
    "Georgian": "kartvelian.georgian",
    "Nepalese": f"{IE}.indoaryan.pahari.eastern.nepali",
    "French": f"{IE}.romance.french",
    "Vietnamese": "austroasiatic.vietnamese",
    "German": f"{IE}.germanic.continental.german",
    "Chinese": "sinotibetan.sinitic",
    "Polish": f"{IE}.slavic.west.polish",
    "Turkish": "turkic.turkish",
    "Persian": f"{IE}.iranian.persian",
    "Armenian": f"{IE}.armenian.armenian",
    "Hebrew": "afroasiatic.hebrew",
    "Italian": f"{IE}.romance.italian",
    "Latvian": f"{IE}.baltic.latvian",
    "Slovakian": f"{IE}.slavic.west.slovak",
    "Moldovian": f"{IE}.romance.moldovan",
    "Bengali": f"{IE}.indoaryan.eastern.bengali",
    "Spanish": f"{IE}.romance.spanish",
    "Swedish": f"{IE}.germanic.north.swedish",
    "Hungarian": "uralic.hungarian",
    "Urdu": f"{IE}.indoaryan.central.urdu",
    "Yugoslavian": f"{IE}.slavic.south.serbocroatian",
    "Dutch": f"{IE}.germanic.continental.dutch",
    "Other languages": "other",
    "Not stated": None,
}

EXCLUDED = ("Not stated",)


def resolve(label):
    if label not in NAMES:
        raise KeyError(f"cy2021: unmapped label {label!r}")
    return NAMES[label]
