"""Ukraine 2001 census, native language (sources/ua_c01.py) -> node.

Keyed by the labels exactly as data/normalized/ua.csv carries them: Ukrstat's own column headers
for the 17 languages of its by-unit table (accusative, as printed: "named as native ...
language"), the U.S. Census Bureau's original field names for the eight nationalities whose
own-language counts are carved out of the remainder, and two of sources/ua_c01.py's own.

CALLS.
  "єврейську" ("Jewish"): Yiddish. The Soviet and post-Soviet census term "єврейська/еврейский
    мова" is Yiddish; the census database has no separate Hebrew line in this table. 3,096
    people. If it held some Hebrew too, the narrowest node would be the root and the label is
    named, so Yiddish it is.
  "грецьку" (Greek): Greek. Some Mariupol Greeks speak Rumeic (a Greek variety) and some Urum
    (Turkic); the census printed one label, which says Greek. 5,589 people.
  "циганську" (Romani): the leaf `romani.romani`, variety not stated (tree.d/ua.txt).
  "молдовську" and "румунську": two answers, two nodes, though Glottolog has one language.
  "Native Language, Turks": Turkish; in Ukraine these are mostly Meskhetian Turks (Kherson oblast).
  "Native Language, Arabs": Arabic, the macrolanguage node; mostly students in the big cities.
  "Native Language, Koreans": Korean (Koryo-saram, Koryo-mal is a Korean variety).
  REMAINDER: every other native language, unnamed, on `other`.
  NOT_STATED ("Did Not Indicate", 200,951): not drawn; it is the entry's gap.
"""
SL = "indoeuropean.slavic"
TU = "turkic"

NAMES = {
    "українську": f"{SL}.east.ukrainian",
    "російську": f"{SL}.east.russian",
    "білоруську": f"{SL}.east.belarusian",
    "болгарську": f"{SL}.south.bulgarian",
    "польську": f"{SL}.west.polish",
    "словацьку": f"{SL}.west.slovak",
    "вірменську": "indoeuropean.armenian.armenian",
    "гагаузьку": f"{TU}.gagauz",
    "кримсько-татарську": f"{TU}.crimean_tatar",
    "караїмську": f"{TU}.karaim",
    "молдовську": "indoeuropean.romance.moldovan",
    "румунську": "indoeuropean.romance.romanian",
    "німецьку": "indoeuropean.germanic.continental.german",
    "єврейську": "indoeuropean.germanic.continental.yiddish",
    "циганську": "indoeuropean.indoaryan.romani.romani",
    "угорську": "uralic.hungarian",
    "грецьку": "indoeuropean.hellenic.greek",
    "Native Language, Tatars": f"{TU}.tatar",
    "Native Language, Azerbaijanians": f"{TU}.azerbaijani",
    "Native Language, Turks": f"{TU}.turkish",
    "Native Language, Uzbeks": f"{TU}.uzbek",
    "Native Language, Georgians": "kartvelian.georgian",
    "Native Language, Arabs": "afroasiatic.arabic",
    "Native Language, Vietnamiens": "austroasiatic.vietnamese",
    "Native Language, Koreans": "koreanic.korean",
    "other native languages (remainder)": "other",
}
NOT_STATED = {"Did Not Indicate"}


def resolve(label):
    if label in NOT_STATED:
        return None
    if label not in NAMES:
        raise KeyError(f"ua2001: unmapped label {label!r}")
    return NAMES[label]
