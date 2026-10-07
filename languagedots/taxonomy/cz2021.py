"""Czechia, SLDB 2021, mother tongue -> node. Keyed by ČSÚ's own labels (sldb2021_jazyk1.csv).

56 labels at country and kraj level, 13 of them also at obec level (sources/cz_sldb.py says how
the other 43 reach an obec). Calls (sources/cz.md says more):
  * "Moravský" (Moravian, 22,585 drawn) is its own node beside Czech under West Slavic, not a
    child of Czech: Glottolog files Czecho-Moravian (czec1259) and Lach (lach1246) as Czech
    dialects, but a child would make Czech a group, drawn washed out.
  * "Slezský" (Silesian, 1,272 drawn) goes on the Silesian node Poland uses. Its people are mostly
    in Těšín Silesia (Třinec, Jablunkov, Český Těšín, Karviná), where the local speech is the
    Cieszyn Silesian that Poland's census also files under Silesian or its own gwara; a smaller
    part is in Opava and Ostrava, where it would be Lach. The census label is one word for both.
  * "Čínský" (Chinese) names a group, not a language: it sits on Sinitic, as pl2021, us2024 and
    uk2021 put unspecified Chinese.
  * "Moldavský" (Moldovan) and "Rumunský" (Romanian) stay apart, as ČSÚ prints them; likewise
    "Srbochorvatský" (Serbo-Croatian) beside Serbian, Croatian and Bosnian, and "Černohorský"
    (Montenegrin), a new leaf under South Slavic.
  * "Znakový jazyk" (sign language) does not name which: the `signlanguage` root, as za2011 and
    np2021 put a bare "sign language". Almost all of it will be Czech Sign Language.
  * "Jiný jazyk" (another language) is `other`.
"""
IE = "indoeuropean"
SL = f"{IE}.slavic"
SW, SE, SS = f"{SL}.west", f"{SL}.east", f"{SL}.south"
GE = f"{IE}.germanic"
RO = f"{IE}.romance"
IA, IR = f"{IE}.indoaryan", f"{IE}.iranian"
TK = "turkic"

NAMES = {
    "Český jazyk": f"{SW}.czech",
    "Moravský jazyk": f"{SW}.moravian",
    "Slezský jazyk": f"{SW}.silesian",
    "Slovenský jazyk": f"{SW}.slovak",
    "Polský jazyk": f"{SW}.polish",
    "Ruský jazyk": f"{SE}.russian",
    "Ukrajinský jazyk": f"{SE}.ukrainian",
    "Běloruský jazyk": f"{SE}.belarusian",
    "Rusínský jazyk": f"{SE}.rusyn",
    "Bulharský jazyk": f"{SS}.bulgarian",
    "Srbský jazyk": f"{SS}.serbian",
    "Chorvatský jazyk": f"{SS}.croatian",
    "Bosenský jazyk": f"{SS}.bosnian",
    "Černohorský jazyk": f"{SS}.montenegrin",
    "Srbochorvatský jazyk": f"{SS}.serbocroatian",
    "Makedonský jazyk": f"{SS}.macedonian",
    "Slovinský jazyk": f"{SS}.slovenian",
    "Litevský jazyk": f"{IE}.baltic.lithuanian",
    "Lotyšský jazyk": f"{IE}.baltic.latvian",
    "Německý jazyk": f"{GE}.continental.german",
    "Nizozemský jazyk": f"{GE}.continental.dutch",
    "Anglický jazyk": f"{GE}.english",
    "Francouzský jazyk": f"{RO}.french",
    "Italský jazyk": f"{RO}.italian",
    "Španělský jazyk": f"{RO}.spanish",
    "Rumunský jazyk": f"{RO}.romanian",
    "Moldavský jazyk": f"{RO}.moldovan",
    "Řecký jazyk": f"{IE}.hellenic.greek",
    "Albánský jazyk": f"{IE}.albanian.albanian",
    "Arménský jazyk": f"{IE}.armenian.armenian",
    "Romský jazyk": f"{IA}.romani.romani",
    "Hindský jazyk": f"{IA}.central.hindi",
    "Urdský jazyk": f"{IA}.central.urdu",
    "Bengálský jazyk": f"{IA}.eastern.bengali",
    "Paňdžábský jazyk": f"{IA}.northwestern.punjabi",
    "Perský jazyk": f"{IR}.persian",
    "Paštunský jazyk": f"{IR}.pashto",
    "Tádžický jazyk": f"{IR}.tajik",
    "Maďarský jazyk": "uralic.hungarian",
    "Turecký jazyk": f"{TK}.turkish",
    "Azerbájdžánský jazyk": f"{TK}.azerbaijani",
    "Kazašský jazyk": f"{TK}.kazakh",
    "Kyrgyzský jazyk": f"{TK}.kyrgyz",
    "Uzbecký jazyk": f"{TK}.uzbek",
    "Turkmenský jazyk": f"{TK}.turkmen",
    "Mongolský jazyk": "mongolic.mongolian",
    "Gruzínský jazyk": "kartvelian.georgian",
    "Čečenský jazyk": "nakhdaghestanian.nakh.chechen",
    "Arabský jazyk": "afroasiatic.arabic",
    "Hebrejský jazyk": "afroasiatic.hebrew",
    "Čínský jazyk": "sinotibetan.sinitic",
    "Vietnamský jazyk": "austroasiatic.vietnamese",
    "Japonský jazyk": "japonic.japanese",
    "Korejský jazyk": "koreanic.korean",
    "Znakový jazyk": "signlanguage",
    "Jiný jazyk": "other",
}
NOT_STATED = {"Nezjištěno"}


def resolve(label):
    if label in NOT_STATED:
        return None
    if label not in NAMES:
        raise KeyError(f"cz2021: unmapped label {label!r}")
    return NAMES[label]
