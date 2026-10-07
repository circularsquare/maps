"""Türkiye, KONDA *Biz Kimiz?* 2006, Tablo 7 (mother tongue, 15 answers), plus DGMM's Syrians
under temporary protection -> node. sources/tr_konda.py; keyed by KONDA's own Turkish labels.

CALLS (sources/tr.md says more):
  Kürtçe: `indoeuropean.iranian.kurdish`. KONDA's answer is "Kurdish"; in Türkiye that is almost
    all Kurmanji, but the survey does not say so and the node other countries use is kept, so
    Kurdish reads as one colour across the border.
  Zazaca: `indoeuropean.iranian.zazaki`, its own answer in the survey, as in the 1965 census.
  Arapça and the Syrians: `afroasiatic.arabic`. The Syrians are a separate label so the record
    can tell them apart; some are Kurds or Turkmens, and DGMM does not say how many.
  Rumca: Greek. Ermenice: Armenian. Çerkesçe: `abkhazadyghe.circassian` (KONDA does not say
    Adyghe or Kabardian). Lazca: `kartvelian.laz`, new (Glottolog lazz1240, Kartvelian).
    Kıptice ("Gypsy"): Romani.
  Yahudice ("Jewish"): `other.jewish`, the node Azerbaijan made for the same unspecific answer;
    in Türkiye it is Ladino or Hebrew and the survey does not say which.
  Türki Diller ("Turkic languages", Azerbaijani, Turkmen, Uzbek, Kazakh and the rest, Turkish
    being its own answer): the group node `turkic`, drawn as a Turkic language not named.
  Balkan ("Balkan languages": Bosnian, Albanian, Pomak, Bulgarian, Macedonian...) and Batı
    Avrupa ("Western European"): the root `indoeuropean`, the narrowest node holding all of
    each; both are drawn as "language not named".
  Kafkas ("Caucasian languages" other than Circassian: Georgian, Abkhaz, Chechen and the rest)
    spans three families, so `other.caucasian`, new, a leaf under `other` like Russia's
    `other.dagestani`.
  Diğer: bare `other`.
"""
NAMES = {
    "Türkçe": "turkic.turkish",
    "Kürtçe": "indoeuropean.iranian.kurdish",
    "Zazaca": "indoeuropean.iranian.zazaki",
    "Arapça": "afroasiatic.arabic",
    "Arapça (Suriyeli, geçici koruma)": "afroasiatic.arabic",
    "Ermenice": "indoeuropean.armenian.armenian",
    "Rumca": "indoeuropean.hellenic.greek",
    "Yahudice": "other.jewish",
    "Balkan": "indoeuropean",
    "Kafkas": "other.caucasian",
    "Lazca": "kartvelian.laz",
    "Çerkesçe": "abkhazadyghe.circassian",
    "Türki Diller": "turkic",
    "Kıptice": "indoeuropean.indoaryan.romani.romani",
    "Batı Avrupa": "indoeuropean",
    "Diğer": "other",
}
EXTRA_NODES = []


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"tr2006: unmapped label {label!r}")
