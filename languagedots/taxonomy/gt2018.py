"""Guatemala, Censo 2018: PCP15 "Idioma en el que aprendio a hablar" (the language the person
learned to speak in), everyone aged 4 and over. Labels exactly as sources/gt_censo.py writes them
into data/normalized/gt.csv, which is INE's REDATAM label set (INE spells Kaqchikel "Kaqchiquel").

Every one of INE's 27 language labels is one node (spec §3.1). The two that are not a language:
  "Señas"        sign language, unnamed (almost all of it will be Lengua de Señas de Guatemala,
                 but INE does not say so): the `signlanguage` root, as au2021's "nfd".
  "Otro idioma"  any other language: foreign languages and indigenous languages of other
                 countries in one line, which the census does not let us tell apart, so `other`
                 rather than `americas_other` (AGENT_BRIEF §3).
  "No habla"     does not speak (13,996): not drawn, in `gap`.
"""

NAMES = {
    # Mayan
    "Achí": "mayan.kichean.achi",
    "Akateko": "mayan.qanjobalan.akateko",
    "Awakateko": "mayan.mamean.awakateko",
    "Ch'orti'": "mayan.cholan_tzeltalan.chorti",
    "Chalchiteko": "mayan.mamean.chalchiteko",
    "Chuj": "mayan.qanjobalan.chuj",
    "Itza'": "mayan.yucatecan.itza",
    "Ixil": "mayan.mamean.ixil",
    "Jakalteko/Popti'": "mayan.qanjobalan.jakalteko",
    "K'iche'": "mayan.kichean.kiche",
    "Kaqchiquel": "mayan.kichean.kaqchikel",
    "Mam": "mayan.mamean.mam",
    "Mopan": "mayan.yucatecan.mopan",
    "Poqomam": "mayan.kichean.poqomam",
    "Poqomchi'": "mayan.kichean.poqomchi",
    "Q'anjob'al": "mayan.qanjobalan.qanjobal",
    "Q'eqchi'": "mayan.kichean.qeqchi",
    "Sakapulteko": "mayan.kichean.sakapulteko",
    "Sipakapense": "mayan.kichean.sipakapense",
    "Tektiteko": "mayan.mamean.teko",            # Mexico's "Teko": the same language
    "Tz'utujil": "mayan.kichean.tzutujil",
    "Uspanteko": "mayan.kichean.uspanteko",
    # others
    "Xinka": "isolate.xinka",
    "Garífuna": "arawakan.garifuna",
    "Español": "indoeuropean.romance.spanish",
    "Inglés": "indoeuropean.germanic.english",
    "Señas": "signlanguage",
    "Otro idioma": "other",
    "No habla": None,
}


def resolve(label):
    if label not in NAMES:
        raise KeyError(f"gt2018: unmapped label {label!r}")
    return NAMES[label]
