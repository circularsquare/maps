"""Namibia, 2011 census, main language spoken in the household (H13) -> node. Keyed by the
DDI's label, as sources/na_pums.py writes it. Thirteen answers, Don't know, and the blank of the
households that were not asked.

NAMED AS LANGUAGES, each its own node:
  * Oshiwambo languages -> Oshiwambo. The dialects of Owambo (Kwanyama, Ndonga, Kwambi,
    Ngandjera, Mbalantu, Kwaluudhi, Kolonkadhi, Eunda), which Glottolog lists as the Ndonga
    (R.20) group. In Namibia Oshiwambo is named and taught as one language with two written
    standards (Oshikwanyama, Oshindonga), and the census counts it as one answer; drawn as one
    language, not as a group washed out as "language not named", which would leave half the
    country's dots looking like a remainder.
  * Herero languages -> Otjiherero. Herero with its Mbanderu and Himba varieties (Glottolog
    here1253 and its dialects). Glottolog's Herero (R.30) group also holds Zemba, spoken by a few
    thousand in Kunene; the census does not separate them and they are drawn as Otjiherero.
  * Nama/Damara languages -> Khoekhoegowab. Nama and Damara are two peoples' names for one
    language (Glottolog nama1264, Khoekhoe), Namibia's official name for it Khoekhoegowab.
  * Setswana, Afrikaans, German, English: as named.

NAMED AS GROUPS OF DIFFERENT LANGUAGES, on the narrowest node holding all of them (spec §3.2),
drawn washed out as "language not named":
  * Kavango languages: Rukwangali and Rumanyo (Gciriku) are Glottolog's Kwangali-Diriku, under
    Southern Njila; Thimbukushu is Greater Luyana (zm.txt has it under Luyana). Not mutually
    intelligible, and nothing in the census says which a household spoke. Narrowest node: Bantu.
  * Caprivi languages: Silozi (Sotho-Tswana), Subiya, Fwe and Totela (Botatwe), Yeyi, Mbukushu.
    Narrowest node: Bantu.
  * San languages: Ju|'hoan and !Xun (Kx'a), Khwe and Naro (Khoe), !Xoon (Tuu), and the
    Hai||om, whose speech is a Khoekhoe variety. Narrowest node: the Khoisan root.

REMAINDERS. Other African languages (1.7%; Kavango's profile puts 12.6% of its households here,
largely Angolan languages) on `africa_other`, kept apart from `other` (Anita, 2026-10-04).
Other European languages and Asian languages name parts of the world, not languages, and no
node narrower than `other` holds every language each could be; on `other`.

NOT DRAWN: Don't know (535 people) and the households not asked (49,952 people in hostels,
barracks, prisons, hospitals, hotels and the like).
"""
BANTU = "nigercongo.bantu"
NAMES = {
    "Oshiwambo languages": f"{BANTU}.oshiwambo",
    "Herero languages": f"{BANTU}.otjiherero",
    "Nama/Damara languages": "khoisan.khoe.khoekhoe",
    "Setswana": f"{BANTU}.sotho_tswana.setswana",
    "Afrikaans": "indoeuropean.germanic.continental.afrikaans",
    "German": "indoeuropean.germanic.continental.german",
    "English": "indoeuropean.germanic.english",
    "Kavango languages": BANTU,
    "Caprivi languages": BANTU,
    "San languages": "khoisan",
    "Other African languages": "africa_other",
    "Other European languages": "other",
    "Asian languages": "other",
    "Don't know": None,
    "Not asked (institutions and special populations)": None,
}
EXCLUDED = ("Don't know", "Not asked (institutions and special populations)")


def resolve(name):
    return NAMES[name]
