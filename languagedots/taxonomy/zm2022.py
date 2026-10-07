"""Zambia, 2022 census, "widely spoken language of communication" -> node. Keyed by the label
as Table C1.0 prints it (sources/zm_census.py keeps it verbatim, `bemba` in lower case too).

93 rows: 80 named Zambian languages and English in the census's sub-groups A to L, nine foreign
rows, Sign Language, `Other African`, `Other Language`, and two that are not drawn: `Babies not
yet able to speak` (1,108,980) and `Unable to speak - Hearing impaired & dumb (Not Applicable)`
(107,783). Every label the census prints as a language has its own node, including the ones
linguists treat as dialects of a bigger language (Ngumbo, Unga, Mukulu as Bemba; Senga, Fungwe,
Yombe as Tumbuka; Ndembu as Lunda; the Luyana varieties); the census asked them apart.

TREE (taxonomy/tree.d/zm.txt). Middle levels are the conventional Zambian groupings, with
Guthrie zones in the labels, each checked against Glottolog's classification
(data/raw/glottolog):
  * Bemba (M.40): Bemba and Aushi, Chishinga, Kabende, Bwile (Glottolog: Bemba M.40 or Malungu-
    Central Sabi); Shila and Tabwa are Glottolog's Taabwa, the census's Group A, kept here.
    Mukulu, Ngumbo, Unga are Bemba dialects of the Bangweulu area (no Glottolog entry for
    Ngumbo; Glottolog's Unga and Mukulu hits are unrelated languages elsewhere, not used).
  * Bisa-Lamba (M.50): Bisa, Lala, Ambo, Luano, Swaka, Lamba, Lima (Glottolog bisa1262).
  * Nyanja-Sena (N.30-40): Nyanja, Chewa, Nsenga, Ngoni, Kunda, Chikunda. Glottolog files
    Nsenga (and Kunda as its dialect) under Sabi beside Bemba; the conventional N.41 grouping
    with the Nyanja-Sena languages of Eastern Province is used instead, because that is where a
    Zambian reader looks for it. Ngoni is Glottolog's "Ngoni (Nyanja)", the Nguni-descended
    community's Nyanja speech, not the Nguni language.
  * Tumbuka (N.20): Tumbuka, Senga, Fungwe, Yombe (all Glottolog Tumbukic).
  * Botatwe (M.60, and K.40): Tonga, Toka, Toka-Leya, Ila, Lundwe, Lumbu, Sala, Gowa, Lenje,
    Soli, Twa (Eastern Botatwe) and Totela, Subiya, Fwe (Western Botatwe).
  * Luyana (K.30): Luyana and the Luyi and Simaa varieties, Mashi and Mbukushu (Glottolog
    Greater Luyana).
  * Lozi is Sotho-Tswana (S.30, Glottolog Sesotho-Lozi), under za.txt's existing group.
  * Chokwe-Lunda (K.10-20, L.50): Lunda (North-Western), Ndembu, Luvale, Luchazi, Mbunda, Chokwe.
  * Luban (L.40-60): Kaonde; Nkoya with Lukolwe (Mbwela), Lushangi, Mashasha (Glottolog Nkoya
    holds Mbowela, Lushangi, Mashasha).
  * Mambwe-Nyiha (M.10-20, Glottolog Mbozi): Mambwe, Lungu, Namwanga, Iwa, Tambo, Lambya, Nyiha,
    Wandya.

REMAINDERS. `Other African` (38,293) goes on `africa_other`, kept apart from `other` (Anita,
2026-10-04: indigenous remainders apart from other remainders). The regional foreign rows
(American, European, Indian, Asian, Oceanian) name a part of the world, not a language, and no
node narrower than `other` holds every language each could be; on `other`, with `Other
Language`. `Swahili Tanzania` is Swahili; `Swahili Congo` is the Congolese variety, its own node.
"""
B = "nigercongo.bantu"
BEMBA = f"{B}.bemba"
BISA = f"{B}.bisa_lamba"
NYA = f"{B}.nyanja_sena"
TUM = f"{B}.tumbuka"
BOT = f"{B}.botatwe"
LUY = f"{B}.luyana"
CL = f"{B}.chokwe_lunda"
LUB = f"{B}.luban"
MBO = f"{B}.mambwe_nyiha"

NAMES = {
    # Group A (Northern)
    "Aushi": f"{BEMBA}.aushi",
    "Chishinga": f"{BEMBA}.chishinga",
    "Kabende": f"{BEMBA}.kabende",
    "Mukulu": f"{BEMBA}.mukulu",
    "Ngumbo": f"{BEMBA}.ngumbo",
    "Unga": f"{BEMBA}.unga",
    "bemba": f"{BEMBA}.bemba",
    "Bwile": f"{BEMBA}.bwile",
    "Lunda (Luapula)": f"{BEMBA}.lunda_luapula",   # Kazembe's Lunda, who speak a Bemba variety
    "Shila": f"{BEMBA}.shila",
    "Tabwa": f"{BEMBA}.tabwa",
    "Ushi": f"{BEMBA}.ushi",
    # Group A (borders Eastern & Northern)
    "Kunda": f"{NYA}.kunda",
    "Bisa": f"{BISA}.bisa",
    "Chikunda": f"{NYA}.chikunda",
    # Group A (borders Central & Eastern), (borders Central & Copperbelt)
    "Lala": f"{BISA}.lala",
    "Ambo": f"{BISA}.ambo",
    "Luano": f"{BISA}.luano",
    "Swaka": f"{BISA}.swaka",
    "Lamba": f"{BISA}.lamba",
    "Lima": f"{BISA}.lima",
    # Group B
    "Kaonde": f"{LUB}.kaonde",
    # Group C1, C2
    "Lozi": f"{B}.sotho_tswana.lozi",
    "Luyana": f"{LUY}.luyana",
    "Kwandi": f"{LUY}.kwandi",
    "Kwangwa": f"{LUY}.kwangwa",
    "Mbowe": f"{LUY}.mbowe",
    "Mbumi": f"{LUY}.mbumi",
    "Wina": f"{LUY}.wina",
    "Simaa": f"{LUY}.simaa",
    "Imilangu": f"{LUY}.imilangu",
    "Mwenyi": f"{LUY}.mwenyi",
    "Nyengo": f"{LUY}.nyengo",
    "Koma": f"{LUY}.koma",
    "Liyuwa": f"{LUY}.liyuwa",
    "Mulonga": f"{LUY}.mulonga",
    "Mashi": f"{LUY}.mashi",
    "Mbukushu": f"{LUY}.mbukushu",
    # Group D, E
    "Lunda (North-Western)": f"{CL}.lunda",
    "Ndembu": f"{CL}.ndembu",
    "Luvale": f"{CL}.luvale",
    "Luchazi": f"{CL}.luchazi",
    "Mbunda": f"{CL}.mbunda",
    "Chokwe": f"{CL}.chokwe",
    # Group F, G
    "Mambwe": f"{MBO}.mambwe",
    "Lungu": f"{MBO}.lungu",
    "Namwanga": f"{MBO}.namwanga",
    "Iwa": f"{MBO}.iwa",
    "Tambo": f"{MBO}.tambo",
    "Lambya": f"{MBO}.lambya",
    "Nyiha": f"{MBO}.nyiha",
    "Wandya": f"{MBO}.wandya",
    # Group H
    "Nkoya": f"{LUB}.nkoya",
    "Lukolwe (Mbwela)": f"{LUB}.lukolwe",
    "Lushangi": f"{LUB}.lushangi",
    "Mashasha": f"{LUB}.mashasha",
    # Group I, J
    "Nsenga": f"{NYA}.nsenga",
    "Ngoni": f"{NYA}.ngoni",
    "Chewa": f"{NYA}.chewa",
    "Nyanja": f"{NYA}.nyanja",
    # Group K
    "Tonga": f"{BOT}.tonga",
    "Toka": f"{BOT}.toka",
    "Totela": f"{BOT}.totela",
    "Toka-Leya": f"{BOT}.toka_leya",
    "Subiya": f"{BOT}.subiya",
    "Twa": f"{BOT}.twa",
    "Fwe": f"{BOT}.fwe",
    "Ila": f"{BOT}.ila",
    "Lundwe": f"{BOT}.lundwe",
    "Lumbu": f"{BOT}.lumbu",
    "Sala": f"{BOT}.sala",
    "Gowa": f"{BOT}.gowa",
    "Lenje": f"{BOT}.lenje",
    "Soli": f"{BOT}.soli",
    # Group L
    "Tumbuka": f"{TUM}.tumbuka",
    "Fungwe": f"{TUM}.fungwe",
    "Senga": f"{TUM}.senga",
    "Yombe": f"{TUM}.yombe",
    # Official language
    "English": "indoeuropean.germanic.english",
    # Foreign languages
    "French": "indoeuropean.romance.french",
    "Mandarin": "sinotibetan.sinitic.mandarin",
    "Swahili Congo": f"{B}.swahili_congo",
    "Swahili Tanzania": f"{B}.swahili",
    "American": "other",
    "European": "other",
    "Indian": "other",
    "Asian": "other",
    "Oceanian": "other",
    # the rest
    "Sign Language": "signlanguage",
    "Other African": "africa_other",
    "Other Language": "other",
    "Babies not yet able to speak": None,
    "Unable to speak - Hearing impaired & dumb (Not Applicable)": None,
}

EXCLUDED = ("Babies not yet able to speak",
            "Unable to speak - Hearing impaired & dumb (Not Applicable)")


def resolve(name):
    return NAMES[name]
