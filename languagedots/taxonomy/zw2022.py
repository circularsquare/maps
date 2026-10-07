"""Zimbabwe, 2022 census, mother tongue ("the language usually spoken in the individual's home
in his/her early childhood"), persons aged 3+ -> node. Keyed by Table 2.17's labels as
sources/zw_census.py writes them. Sixteen answers and Other; no not-stated row.

NAMED AS LANGUAGES, each its own node (Glottolog, data/raw/glottolog/languages.csv):
  * Shona -> us.txt's leaf (shon1251). The census prints one Shona: Zezuru, Karanga, Manyika,
    Korekore are not separated (the Afrobarometer names them; sources/zw.md), so they are not
    drawn apart.
  * Ndau -> mz.txt's Cindau (ndau1241, MZ;ZW), a leaf beside Shona as there.
  * Kalanga -> bw.txt's leaf (kala1384). Nambya (namb1291): a new leaf beside it, same reason
    bw.txt gives for Kalanga (Shona is a leaf other countries draw on).
  * Ndebele -> Ndebele (Zimbabwe). Xhosa -> Xhosa (Mbembesi, Matabeleland North).
  * Tonga -> zm.txt's "Tonga (Zambia)", tong1318: the Zambezi valley Tonga of Binga and Hwange
    speak the same language as Zambia's Southern Province (Glottolog lists it for ZM, NA, ZW).
  * Shangani -> Xitsonga (Changana; Chiredzi and Mwenezi).
  * Venda -> Tshivenda. Tswana -> Setswana. Chewa -> Chewa (Malawian-origin farm and mine
    communities). Chibarwe -> mz.txt's Barwe (barw1243).
  * Sotho -> a new leaf "Sotho (Zimbabwe)" under Sotho-Tswana. Zimbabwe's Sotho live in Gwanda
    and Beitbridge (32,758 of 39,155 in Matabeleland South), where the speech is the Birwa /
    Northern Sotho of the Limpopo valley rather than Lesotho's Sesotho; which one the census
    means is not said, so it gets its own leaf rather than either.
  * English: as named. Sign Language -> signlanguage.

NAMED AS A GROUP, on the narrowest node holding it (spec 3.2):
  * Koisan -> the Khoisan root, as bw2011.py's Sesarwa. Zimbabwe's are the Tshwao (Kalahari
    Khoe) of Tsholotsho and Bulilima; the census names no language. 305 people.

REMAINDER: Other -> `other` (the census does not say African or foreign).

NOT DRAWN: children under 3, not in the table (1,265,704 of 15,178,957).
"""
BANTU = "nigercongo.bantu"
NAMES = {
    "Shona": f"{BANTU}.shona",
    "Ndebele": f"{BANTU}.nguni.ndebele_zw",
    "English": "indoeuropean.germanic.english",
    "Kalanga": f"{BANTU}.kalanga",
    "Koisan": "khoisan",
    "Nambya": f"{BANTU}.nambya",
    "Ndau": f"{BANTU}.ndau",
    "Chibarwe": f"{BANTU}.nyanja_sena.barwe",
    "Shangani": f"{BANTU}.tswa_ronga.tsonga",
    "Chewa": f"{BANTU}.nyanja_sena.chewa",
    "Sign Language": "signlanguage",
    "Sotho": f"{BANTU}.sotho_tswana.sotho_zw",
    "Tonga": f"{BANTU}.botatwe.tonga",
    "Tswana": f"{BANTU}.sotho_tswana.setswana",
    "Venda": f"{BANTU}.venda",
    "Xhosa": f"{BANTU}.nguni.xhosa",
    "Other": "other",
}


def resolve(name):
    return NAMES[name]
