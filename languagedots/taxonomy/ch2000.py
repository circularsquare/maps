"""Switzerland, Volkszählung 2000, main language (Hauptsprache), px-x-4003000000_123
(sources/ch_vz2000.py) -> node. Keyed by BFS's German labels exactly as data/normalized/ch.csv
carries them.

The question asked for ONE language, "the language in which you think and which you know best".
The census imputed non-response, so there is no "not stated" row: the 36 categories partition
all 7,287,357 residents.

CALLS (sources/ch.md says more):
  "Deutsch": the main-language answer covers Standard and Swiss German alike. Since 2026-10-06
    (Anita: draw Swiss German as its own language in Switzerland; Bavarian, Alemannic etc. stay
    German in Germany and Austria) sources/ch_vz2000.py splits it per commune into
    "Deutsch: Schweizerdeutsch (modelliert)" -> `swiss_german` and "Deutsch: Hochdeutsch
    (modelliert)" -> `german`, at the same census's family-language rates (Schweizerdeutsch and
    Hochdeutsch were separate boxes there) by language region and nationality. Plain "Deutsch"
    remains only on the canton and country rows, which are not drawn.
  "Rätoromanisch": the existing `rhaetoromance` node, "Rhaeto-Romance", which is the literal
    English of BFS's term (uk2021 and pl2021 put their censuses' "Rhaeto-Romance" there too). In
    Switzerland every one of these 35,097 people speaks Romansh (Glottolog roma1326); Ladin and
    Friulian are Italian.
  BFS'S OWN MERGES, drawn on the named language as ONS's "Bengali (with Sylheti)" is: the census
    analysis (Lüdi and Werlen 2005, BFS, p. 11 note 2) says "Spanisch" includes Catalan and
    Galician, "Englisch" Scots, "Türkisch" the other Turkic languages, and "Russisch" Belarusian and
    Ukrainian. The national figures here match that publication's (Russian 9,005 against its
    9,003, the gap being the other residence concept), and the residuals they would otherwise
    sit in are tiny ("Übrige westeuropäische" 175, "Übrige slawische" 114), so the px table makes
    the same merges. The map cannot unpick them.
  "Serbisch und Kroatisch": Serbo-Croatian, one category in the census (Bosnian included).
  REGIONAL REMAINDERS whose members cross families go on `other`: nothing narrower is sure to
    contain them (uk2021's rule). "Übrige westeuropäische" (can hold Basque and Celtic),
    "Übrige nordeuropäische" (Finnish is listed among the northern ones, so it can hold Estonian
    or Sami), "Übrige osteuropäische", "Andere europäische", "Westasiatische" (Kurdish, Persian,
    Armenian, Hebrew...), "Indoarische und drawidische" (28,101, mostly Tamil, 21,816 nationally
    in Lüdi and Werlen, but Indo-Aryan and Dravidian are two families), "Ostasiatische" (Chinese,
    Thai, Vietnamese, Japanese, Tagalog...), "Übrige Sprachangaben".
  "Uebrige slawische Sprachen": Slavic.
  "Afrikanische Sprachen" (9,202): `africa_other`, which zm2022 and cf2003 use for African
    languages the census does not name. Arabic is printed apart, so this is sub-Saharan and
    Berber speech, possibly a few Afrikaans speakers.
"""
GE = "indoeuropean.germanic"
RO = "indoeuropean.romance"
SL = "indoeuropean.slavic"

NAMES = {
    "Deutsch": f"{GE}.continental.german",
    "Deutsch: Schweizerdeutsch (modelliert)": f"{GE}.continental.swiss_german",
    "Deutsch: Hochdeutsch (modelliert)": f"{GE}.continental.german",
    "Französisch": f"{RO}.french",
    "Italienisch": f"{RO}.italian",
    "Rätoromanisch": f"{RO}.rhaetoromance",
    "Englisch": f"{GE}.english",
    "Niederländisch": f"{GE}.continental.dutch",
    "Spanisch": f"{RO}.spanish",            # with Catalan and Galician (BFS's merge)
    "Portugiesisch": f"{RO}.portuguese",
    "Übrige westeuropäische Sprachen": "other",
    "Dänisch": f"{GE}.north.danish",
    "Norwegisch": f"{GE}.north.norwegian",
    "Schwedisch": f"{GE}.north.swedish",
    "Finnisch": "uralic.finnish",
    "Übrige nordeuropäische Sprachen": "other",
    "Serbisch und Kroatisch": f"{SL}.south.serbocroatian",
    "Russisch": f"{SL}.east.russian",       # with Belarusian and Ukrainian (BFS's merge)
    "Polnisch": f"{SL}.west.polish",
    "Tschechisch": f"{SL}.west.czech",
    "Slowakisch": f"{SL}.west.slovak",
    "Mazedonisch": f"{SL}.south.macedonian",
    "Slowenisch": f"{SL}.south.slovenian",
    "Bulgarisch": f"{SL}.south.bulgarian",
    "Uebrige slawische Sprachen": SL,
    "Albanisch": "indoeuropean.albanian.albanian",
    "Türkisch": "turkic.turkish",           # with the other Turkic languages (BFS's merge)
    "Übrige osteuropäische Sprachen": "other",
    "Ungarisch": "uralic.hungarian",
    "Rumänisch": f"{RO}.romanian",
    "Griechisch": "indoeuropean.hellenic.greek",
    "Andere europäische Sprachen": "other",
    "Afrikanische Sprachen": "africa_other",
    "Arabisch": "afroasiatic.arabic",
    "Westasiatische Sprachen": "other",
    "Indoarische und drawidische Sprachen": "other",
    "Ostasiatische Sprachen": "other",
    "Übrige Sprachangaben": "other",
}
NOT_STATED = {"Hauptsprachen - Total"}


def resolve(label):
    if label in NOT_STATED:
        return None
    if label not in NAMES:
        raise KeyError(f"ch2000: unmapped label {label!r}")
    return NAMES[label]
