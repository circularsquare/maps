"""Togo, MICS6 2017 head-of-household mother tongue by region, split inside each MICS group by
Afrobarometer R5-R7 (2012-2017) first-language answers (sources/tg_mics.py; until 2026-10-09
Afrobarometer alone, sources/tg_afro.py) -> node. Labels are Afrobarometer's language card,
spellings merged in sources/tg_afro.py, plus MICS's own "Foreign language" and "French".

  * Gbe (Kwa): Ewe; Mina (Gen, genn1243, bj.txt's leaf); Ouatchi (Waci Gbe, waci1239, a leaf
    of its own); Aja and Fon (bj.txt).
  * Ghana-Togo Mountain (Kwa, under gh.txt's `kwa.gtm`): Ikposo (ikpo1238), Akebu (akeb1238).
  * Gur: Kabiyè, Tem, Ntcham (Bassar), Konkomba, Gourmanchéma, Akaselem ("Tchamba", the
    Tchamba people's language), existing leaves; Moba (bj.txt); Nawdm (Losso) and Lama (Lamba,
    lamb1271) new; Ngangam (Gangam, Gurma group; no glottocode found) new.
  * Anufo (Tchokossi), Kwa: gh.txt's leaf. Ifè (Ana): bj.txt's `voltaniger.ife`.
  * "Other" (the card's own catch-all; since MICS, its share of MICS's "autres langues
    nationales", 1.4%): `africa_other`.
  * "Foreign language" (MICS's LANGUES ETRANGERES, 4.7%): `africa_other`, as Burkina Faso's
    "Autre langue africaine" (bf2006.py). MICS does not say which, but 236 of its 308 heads give
    "other nationalities" as ethnic group, 58% are Muslim and a third live in rural households
    in every region: residents from the neighbouring countries, so African languages, the
    narrowest node that holds nearly all of it.
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
    "Foreign language": "africa_other",
}


def resolve(name):
    return NAMES.get(name)
