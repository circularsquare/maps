"""Slovakia, SODB 2021, mother tongue -> node. Keyed by the census portal's own labels
(sources/sk_sodb.py; indicator Z01/14), the same 28 at country, kraj and obec level.

Calls (sources/sk.md says more):
  * "rómsky" (Romani, 100,526) goes on the generic Romani leaf, as cz2021's "Romský jazyk" does. Most
    of it is Carpathian Romani (Glottolog's West and East Slovakian Romani, under carp1235), with
    some Vlax, but the census does not say which, so it is not put on a dialect.
  * "rusínsky" (Rusyn, 38,679) and "ukrajinský" (Ukrainian, 7,608) stay apart, as printed.
  * "čínsky" (Chinese) names a group, not a language: it sits on Sinitic, as cz2021, pl2021, us2024
    and uk2021 put unspecified Chinese.
  * "jidiš alebo hebrejský" (Yiddish or Hebrew, 273) is ONE answer covering two languages of two
    families: a named leaf of its own under `other` (tree.d/sk.txt), not guessed into either.
  * "slovenský posunkový jazyk" (Slovak Sign Language) is a new leaf under `signlanguage`.
  * "iný" (another language, 3,952) is `other`.
  * "nezistený" (not ascertained, 312,364, 5.7%) is not drawn; it is the gap.
"""
IE = "indoeuropean"
SL = f"{IE}.slavic"
SW, SE, SS = f"{SL}.west", f"{SL}.east", f"{SL}.south"
GE = f"{IE}.germanic"
RO = f"{IE}.romance"

NAMES = {
    "slovenský": f"{SW}.slovak",
    "maďarský": "uralic.hungarian",
    "rómsky": f"{IE}.indoaryan.romani.romani",
    "rusínsky": f"{SE}.rusyn",
    "český": f"{SW}.czech",
    "ukrajinský": f"{SE}.ukrainian",
    "ruský": f"{SE}.russian",
    "anglický": f"{GE}.english",
    "nemecký": f"{GE}.continental.german",
    "poľský": f"{SW}.polish",
    "vietnamský": "austroasiatic.vietnamese",
    "slovenský posunkový jazyk": "signlanguage.spj",
    "taliansky": f"{RO}.italian",
    "čínsky": "sinotibetan.sinitic",
    "srbský": f"{SS}.serbian",
    "arabský": "afroasiatic.arabic",
    "španielsky": f"{RO}.spanish",
    "rumunský": f"{RO}.romanian",
    "chorvátsky": f"{SS}.croatian",
    "bulharský": f"{SS}.bulgarian",
    "francúzsky": f"{RO}.french",
    "albánsky": f"{IE}.albanian.albanian",
    "turecký": "turkic.turkish",
    "kórejský": "koreanic.korean",
    "jidiš alebo hebrejský": "other.yiddish_hebrew",
    "perzský": f"{IE}.iranian.persian",
    "iný": "other",
}
NOT_STATED = {"nezistený"}
TOTAL = "total"


def resolve(label):
    if label in NOT_STATED or label == TOTAL:
        return None
    if label not in NAMES:
        raise KeyError(f"sk2021: unmapped label {label!r}")
    return NAMES[label]
