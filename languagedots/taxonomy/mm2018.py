"""Myanmar, GAD 2018 Township Profiles Table 14 (1 April 2017), ETHNIC NATIONALITY -> the language
node it is drawn as. A proxy under AGENT_BRIEF section 2's ethnicity rule (Anita, 2026-10-05);
every row `derived`. Keyed by USCB's English column label (sources/mm_gad.py); the Burmese field
name GAD printed is in the comment where USCB's English needed checking.

Broad national races that cover several languages sit on a GROUP node, drawn washed out as
"language not named": Karen, Chin (Kuki-Chin), Kachin, Kayah, Naga, Chinese. The table splits
off Pa'o, Kayan, Lisu, Palaung and others separately, so those have their own leaves.

Glottolog checks (data/raw/glottolog/languages.csv, by name and ISO code), each as commented.
sources/mm.md has the reasoning.
"""
ST = "sinotibetan"
AA = "austroasiatic"
BU = f"{ST}.burmish"
KA = f"{ST}.karen"
LL = f"{ST}.loloish"

NAMES = {
    "Burmese": f"{BU}.burmese",              # mya, Burmish
    "Rakhine": f"{BU}.rakhine",              # rki, Burmish
    "Danu": f"{BU}.danu",                    # dnv, a Burmese dialect in Glottolog (Intha-Danu)
    "Intha": f"{BU}.intha",                  # Glottolog inth1239, dialect of Intha-Danu
    "Taungyo": f"{BU}.taungyo",              # tco, Burmish
    # Karenic (Glottolog kare1337). "Karen" is the national race covering S'gaw, Pwo and
    # others, not split anywhere: on the group.
    "Karen": KA,
    "Pa'o": f"{KA}.pao",                     # blk, Pa'o Karen
    "Kayan": f"{KA}.kayan",                  # pdu, Kayan Lahwi (Padaung)
    "Kayah": f"{KA}.kayah",                  # Glottolog Kayah family kaya1317 (Eastern eky,
                                             # Western kyu); a group node, no source splits it
    # Kuki-Chin: the Chin national race is some fifty Kuki-Chin languages (Hakha, Tedim, Falam,
    # Mara, Khumi, Asho...), none split by GAD. On the group.
    "Chin": f"{ST}.kukichin",
    # Kachin: the national race's members speak Jingpho (Jingpho-Luish), Zaiwa, Lhaovo and Lachik
    # (Burmish) and Rawang (Nungish). No genealogical node holds only them, so a group node of
    # their own (tree.d/mm.txt). Lisu, also a Kachin subgroup, is its own GAD column.
    "Kachin": f"{ST}.kachin",
    "Naga": f"{ST}.naga",                    # Konyak, Tangkhul, Makury... not split: the group
    "Lisu": f"{LL}.lisu",                    # lis
    # "Liz" is USCB's rendering of GAD's li-shaw, an older spelling of Lisu (Lishaw); "Kho Lone
    # Li Shaw" is a second Lisu label in Kutkai and Lashio. Spelling variants of one answer.
    "Liz": f"{LL}.lisu",
    "Kho Lone Li Shaw": f"{LL}.lisu",
    "Lahu": f"{LL}.lahu",                    # lhu
    "Akha": f"{LL}.akha",                    # ahk
    "Kadu": f"{ST}.kadu",                    # zkd, Luish (Chakpa-Kadu-Ganan)
    "Kanan": f"{ST}.ganan",                  # Ganan, Luish, Banmauk township as Kadu
    # Chinese. Kokang and Mong Wong speak Yunnanese, a Southwestern Mandarin; GAD's "Chinese"
    # (5,638, an undercount) is unsplit and sits on the group, as every country files it.
    "Kokang": f"{ST}.sinitic.mandarin",
    "Mong Wong": f"{ST}.sinitic.mandarin",
    "Chinese": f"{ST}.sinitic",
    # Kra-Dai
    "Shan": "kradai.shan",                   # shn. Eastern Shan's Tai Khun and Tai Lue are filed
                                             # under Shan by GAD and cannot be split out
    # Austroasiatic
    "Mon": f"{AA}.mon",                      # mnw
    "Palaung": f"{AA}.palaung",              # Ruching, Rumai and Shwe Palaung, Palaungic
    "Wa": f"{AA}.wa",
    "Yinn": f"{AA}.yinchia",                 # "Yinn (Kya and/or Net)": Yinchia (yin), Palaungic,
                                             # Nansang and Monghsu; Yinnet beside it
    "Htanot": f"{AA}.danau",                 # GAD's htanot (Kalaw only) is Danau (dnu), Palaungic
    # Hmong-Mien: GAD's myaung-zi (USCB "Myaing") is the Hmong of northern Shan (Miao-zi)
    "Myaing": "hmongmien.hmong",
    # Austronesian
    "Moken": "austronesian.moken",           # mwt, the sea people of the Mergui islands
    # no language named
    "Indian": "other.indian",
    "Foreign": "other",
    "Other": "other",
}


def resolve(label):
    return NAMES.get(label)
