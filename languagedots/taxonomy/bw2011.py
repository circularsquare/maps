"""Botswana, 2011 census, language spoken at home (persons aged 2+) -> node. Keyed by the column
names sources/bw_census.py writes. Sixteen answers and Not Stated.

NAMED AS LANGUAGES, each its own node (families from Glottolog, data/raw/glottolog/values.csv):
  * Setswana: za.txt's node under Sotho-Tswana.
  * Shekgalagadi (Sekgalagadi): Glottolog kgal1244, inside Sotho-Tswana (soth1248), so a new leaf
    beside Setswana. The 2022 report's Appendix 1 lists its varieties (Seboloongwe, Sengologa,
    Seshaga...), national only.
  * Sekalanga (Kalanga): Glottolog kala1384, inside Shona (S.10). us.txt's `nigercongo.bantu.shona`
    is a language leaf other countries draw on, so Kalanga is a new leaf beside it under Bantu
    rather than a child of it.
  * Zezuru/Shona -> Shona. Zezuru is a Shona dialect; one answer, us.txt's node.
  * Ndebele -> uk.txt's Ndebele (Zimbabwe). Botswana's Ndebele speakers are in the North East,
    Tutume and the towns, beside the Zimbabwe border, and the language is Zimbabwe's; South African
    Ndebele is spoken nowhere near.
  * Sesubiya -> zm.txt's Subiya (Botatwe), Chobe.
  * Sembukushu -> zm.txt's Mbukushu (under Luyana; Glottolog puts it in Greater Luyana).
  * Seyeyi (Yeyi): Glottolog yeyi1239, directly under Eastern Narrow Bantu; a new leaf under Bantu.
  * Seherero -> na.txt's Otjiherero. (pl.txt also has a bare `nigercongo.bantu.herero`; the
    southern African node is na's, which Namibia draws.)
  * English, Afrikaans: as named.

NAMED AS A GROUP OF DIFFERENT LANGUAGES, on the narrowest node holding all of them (spec 3.2):
  * Sesarwa -> the Khoisan root. "Sesarwa" is Setswana for "language of the Basarwa", the San. The
    2022 report's Appendix 1 says what the census files there: Naro, Gana, Gwi, Sekhwedam (Khwe),
    Shua, Tsowa, Kua (Khoe), Ju|'hoan and Kx'au||'ein (Kx'a), !Xoo and |Hua (Tuu), and Nama. That
    spans Khoe-Kwadi, Kx'a and Tuu, which share only na.txt's Khoisan root; na2011.py puts
    Namibia's "San languages" there for the same reason. Drawn washed out as "language not named",
    which is what it is: 31,778 people whose language the census did not name further.

REMAINDERS. Other African languages on `africa_other` (kept apart from `other`, Anita 2026-10-04).
Other European languages, Other Asian languages and Other (NEC) name parts of the world or nothing,
and no node narrower than `other` holds every language each could be (na2011.py, the same call).

NOT DRAWN: Not Stated (888 people), and the under-twos, who were not asked.
"""
BANTU = "nigercongo.bantu"
NAMES = {
    "Setswana": f"{BANTU}.sotho_tswana.setswana",
    "English": "indoeuropean.germanic.english",
    "Sekalanga": f"{BANTU}.kalanga",
    "Shekgalagadi": f"{BANTU}.sotho_tswana.shekgalagadi",
    "Sesubiya": f"{BANTU}.botatwe.subiya",
    "Sesarwa": "khoisan",
    "Seyeyi": f"{BANTU}.yeyi",
    "Sembukushu": f"{BANTU}.luyana.mbukushu",
    "Afrikaans": "indoeuropean.germanic.continental.afrikaans",
    "Ndebele": f"{BANTU}.nguni.ndebele_zw",
    "Zezuru/Shona": f"{BANTU}.shona",
    "Seherero": f"{BANTU}.otjiherero",
    "Other African languages": "africa_other",
    "Other European languages": "other",
    "Other Asian languages": "other",
    "Other (NEC)": "other",
    "Not Stated": None,
}
EXCLUDED = ("Not Stated",)


def resolve(name):
    return NAMES[name]
