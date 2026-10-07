"""Togo, Afrobarometer R5-R7 (2012-2017) first-language answers by region (sources/tg_afro.py)
-> node. Labels are the survey's language card, spellings merged in sources/tg_afro.py.

  * Gbe (Kwa): Ewe; Mina (Gen, genn1243, bj.txt's leaf); Ouatchi (Waci Gbe, waci1239, a leaf
    of its own); Aja and Fon (bj.txt).
  * Ghana-Togo Mountain (Kwa, under gh.txt's `kwa.gtm`): Ikposo (ikpo1238), Akebu (akeb1238).
  * Gur: Kabiyè, Tem, Ntcham (Bassar), Konkomba, Gourmanchéma, Akaselem ("Tchamba", the
    Tchamba people's language), existing leaves; Moba (bj.txt); Nawdm (Losso) and Lama (Lamba,
    lamb1271) new; Ngangam (Gangam, Gurma group; no glottocode found) new.
  * Anufo (Tchokossi), Kwa: gh.txt's leaf. Ifè (Ana): bj.txt's `voltaniger.ife`.
  * "Other" (the card's own catch-all, 2.8%; 13% of Centrale): `africa_other`.
"""
G = "nigercongo.gur"
K = "nigercongo.kwa"

NAMES = {
    "Ewe": f"{K}.ewe",
    "Mina (Gen)": f"{K}.gen",
    "Ouatchi (Waci)": f"{K}.waci",
    "Aja": f"{K}.aja",
    "Fon": f"{K}.fon",
    "Ikposo (Akposso)": f"{K}.gtm.ikposo",
    "Akebu": f"{K}.gtm.akebu",
    "Anufo (Tchokossi)": f"{K}.anufo",
    "Kabiyè": f"{G}.kabiye",
    "Moba": f"{G}.moba",
    "Tem (Kotokoli)": f"{G}.tem",
    "Nawdm (Losso)": f"{G}.nawdm",
    "Lama (Lamba)": f"{G}.lama",
    "Gourmanchéma": f"{G}.gourmanchema",
    "Konkomba": f"{G}.konkomba",
    "Ntcham (Bassar)": f"{G}.ntcham",
    "Ngangam (Gangam)": f"{G}.ngangam",
    "Tchamba (Akaselem)": f"{G}.akaselem",
    "Ifè (Ana)": "nigercongo.voltaniger.ife",
    "Yoruba": "nigercongo.voltaniger.yoruba",
    "Hausa": "afroasiatic.chadic.hausa",
    "Fulfulde": "nigercongo.atlantic.fulah",
    "French": "indoeuropean.romance.french",
    "Other": "africa_other",
}


def resolve(name):
    return NAMES.get(name)
