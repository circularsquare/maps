"""Timor-Leste, Census 2015, mother tongue (Volume 2 table 12) -> node. Keyed by the labels
sources/tl_census.py writes, which are the workbook's own (curly apostrophes included).
Thirty-two Timorese languages, five foreign ones and Other; one answer per person.

FAMILIES (Glottolog, data/raw/glottolog/values.csv; tree.d/tl.txt says how they are grouped):
  * Timor-Alor-Pantar (timo1261), the Papuan languages of Timor: Bunak, Fataluku, Makasae
    (Glottolog's Makasae-Makalero holds Makasae and Makalero as its two halves), Makalero, and
    Sa'ani, which the census prints separately and which is spoken in Lautem's Luro beside
    Makasae (5,465 of its 5,787 speakers are in Lautem).
  * Everything else Timorese is Austronesian and goes in `austronesian.timoric`.

EVERY LABEL ITS OWN LEAF (AGENT_BRIEF §3), and five deserve a word:
  * Idalaka and Kawaimina are cover terms Geoffrey Hull proposed (Idalaka for Idaté and Lakalei,
    Kawaimina for Kairui, Midiki, Waima'a and Naueti). The census lists them as answers beside
    their members, so 211 and 41 people gave the cover term; each is a leaf of its own, labelled
    with what it covers, rather than a group holding its members (a group would wash out every
    Idaté and Lakalei speaker as "language not named").
  * Kairui and Midiki are one language to Glottolog (Kairui-Midiki, kair1265) but two answers
    in the census, 3,946 and 14,616, with different homes (Kairui in Viqueque, Midiki in Baucau
    and Viqueque); two leaves.
  * Atauran is "the language of Atauro" without a dialect; Adabe, Dadu'a, Rahesuk, Raklungu and
    Resuk are Atauro's dialect names as the census lists them. Dadu'a is also a dialect name in
    Manatuto (Glottolog dadu1237 puts its point there), and 2015 has 1,863 in Manatuto against 35
    in Dili; one label, one leaf, drawn where the census counted it.
  * Lolein (Glottolog lists a Lolei among Mambae's dialects) and Nanaek (among Galoli's) are
    answers in their own right here, 1,155 and 321; leaves.
  * Makuva (Maku'a, Glottolog maku1277) is an Austronesian language of Lautem's Tutuala, near
    extinct; the census counts 121 people scattered across all thirteen municipalities, which
    may be miscodes, and they are drawn as counted.

FOREIGN: Portuguese, Indonesian, English, Malay, Chinese as named (Chinese on `sinotibetan.sinitic`,
labelled Chinese, as every other country draws an unspecified Chinese). English in Ermera and
Manufahi is withheld in countries/tl.py, not here (sources/tl.md §4).

REMAINDER: Other (617) on `other`; the table does not say whether it holds Timorese languages.
"""
AN = "austronesian.timoric"
TAP = "papuan.timor_alor_pantar"
NAMES = {
    "Tetun Prasa": f"{AN}.tetun_prasa",
    "Tetun Terik": f"{AN}.tetun_terik",
    "Adabe": f"{AN}.adabe",
    "Atauran": f"{AN}.atauran",
    "Baikenu": f"{AN}.baikenu",
    "Bekais": f"{AN}.bekais",
    "Bunak": f"{TAP}.bunak",
    "Dadu’a": f"{AN}.dadua",
    "Fataluku": f"{TAP}.fataluku",
    "Galoli": f"{AN}.galoli",
    "Habun": f"{AN}.habun",
    "Idalaka": f"{AN}.idalaka",
    "Idate": f"{AN}.idate",
    "Isni": f"{AN}.isni",
    "Kairui": f"{AN}.kairui",
    "Kawaimina": f"{AN}.kawaimina",
    "Kemak": f"{AN}.kemak",
    "Lakalei": f"{AN}.lakalei",
    "Lolein": f"{AN}.lolein",
    "Makalero": f"{TAP}.makalero",
    "Sa’ani": f"{TAP}.saani",
    "Makasai": f"{TAP}.makasae",
    "Makuva": f"{AN}.makuva",
    "Mambai": f"{AN}.mambai",
    "Midiki": f"{AN}.midiki",
    "Nanaek": f"{AN}.nanaek",
    "Naueti": f"{AN}.naueti",
    "Rahesuk": f"{AN}.rahesuk",
    "Raklungu": f"{AN}.raklungu",
    "Resuk": f"{AN}.resuk",
    "Tokodede": f"{AN}.tokodede",
    "Waima’a": f"{AN}.waimaa",
    "Portuguese": "indoeuropean.romance.portuguese",
    "Indonesian": "austronesian.malayic.indonesian",
    "English": "indoeuropean.germanic.english",
    "Malay": "austronesian.malayic.malay",
    "Chinese": "sinotibetan.sinitic",
    "Other": "other",
}

# the languages of Atauro island, which countries/tl.py places on the island inside Dili's column
ATAURO = [f"{AN}.{k}" for k in ("atauran", "adabe", "dadua", "rahesuk", "raklungu", "resuk")]


def resolve(name):
    return NAMES[name]
