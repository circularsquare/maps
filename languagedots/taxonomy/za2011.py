"""South Africa, Census 2011, first of the two languages spoken most often in the household
(questionnaire P-06) -> node. Keyed by the census's own spelling, as sal-lang.csv prints it.

Twelve coded answers (the eleven official languages and Sign language), Other, and two that are
not drawn: Not applicable (808,905, mostly people in prisons, mine hostels and other collective
quarters) and Unspecified (0 in the small-area table).

  * Sepedi is Stats SA's name for Northern Sotho (Sesotho sa Leboa), the whole Northern Sotho
    cluster as answered, not only Pedi proper; the census offers no finer code.
  * IsiNdebele is Southern (South African) Ndebele, the Nguni language of Mpumalanga and
    Gauteng; Zimbabwe's Northern Ndebele is a separate node. Speakers of Limpopo's Sotho-
    influenced Northern Transvaal Ndebele had no code of their own and may be in either this
    or Sepedi; nothing in the census separates them.
  * Sign language is the census's code 09, not a named sign language (overwhelmingly South
    African Sign Language, but the code does not say), so it sits on the sign-language root.
  * Other (826,707) is every answer outside the twelve codes, unnamed; on `other`.

Tree placement checked against Glottolog (data/raw/glottolog): Nguni (S.40) holds Zulu, Xhosa,
Swati and Southern Ndebele; Sotho-Tswana (S.30) holds Northern Sotho, Southern Sotho and
Tswana; Tsonga is in Tswa-Ronga (S.50); Venda stands alone among them. Afrikaans is Germanic.
"""
BANTU = "nigercongo.bantu"

NAMES = {
    "Afrikaans": "indoeuropean.germanic.continental.afrikaans",
    "English": "indoeuropean.germanic.english",
    "IsiNdebele": f"{BANTU}.nguni.ndebele_za",
    "IsiXhosa": f"{BANTU}.nguni.xhosa",
    "IsiZulu": f"{BANTU}.nguni.zulu",
    "SiSwati": f"{BANTU}.nguni.siswati",
    "Sepedi": f"{BANTU}.sotho_tswana.sepedi",
    "Sesotho": f"{BANTU}.sotho_tswana.sesotho",
    "Setswana": f"{BANTU}.sotho_tswana.setswana",
    "Tshivenda": f"{BANTU}.venda",
    "Xitsonga": f"{BANTU}.tswa_ronga.tsonga",
    "Sign language": "signlanguage",
    "Other": "other",
    "Not applicable": None,
    "Unspecified": None,
}

EXCLUDED = ("Not applicable", "Unspecified")


def resolve(name):
    return NAMES[name]
