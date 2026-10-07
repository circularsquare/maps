"""Bangladesh 2011 census ETHNIC GROUP -> the language node it is drawn as. A proxy: the census
asked no language question. Keyed by the USCB file's English spelling of BBS's groups.

Anita allowed the proxy on 2026-10-05 (relayed by the supervisor); every row is `derived`.
sources/bd.md has the reasoning and the known misfits, which are the point of reading it:

  * `Not an ethnic minority` (98.9%) is drawn as Bengali. It also holds Sylheti and Chittagonian
    speakers (counted as Bengalis, and languages of their own in Glottolog), the Urdu-speaking
    "Bihari" camps (not an ethnic category), and the Rohingya outside the camps.
  * A group is drawn on its heritage language, so language shift is invisible: many Garo,
    Santal, Oraon and Munda households speak Bengali or Sadri at home.
  * Groups with no established language of their own (Barmon, Dalu, Mong) and the census's
    `Other ethnicity` sit on `bangladesh_other`, not guessed into Bengali or into a language.
"""
IA = "indoeuropean.indoaryan"
ST = "sinotibetan"
AA = "austroasiatic"

NAMES = {
    "Not an ethnic minority": f"{IA}.eastern.bengali",
    "Chakma": f"{IA}.eastern.chakma",
    "Tanchaynga": f"{IA}.eastern.tanchangya",
    "Hajong": f"{IA}.eastern.hajong",
    # Mal Paharia (ISO mkb), Indo-Aryan; the census's separate "Pahari" is the Sauria Paharia
    "Malpahari": f"{IA}.eastern.mal_paharia",
    "Pahari": "dravidian.northern.malto",
    "Orao": "dravidian.northern.kurukh",
    "Sawntal": f"{AA}.munda.santali",
    "Monda": f"{AA}.munda.mundari",
    "Cool": f"{AA}.munda.kol",
    "Khasia": f"{AA}.khasian.khasi",
    "Marma": f"{ST}.burmish.marma",
    "Rakhain": f"{ST}.burmish.rakhine",
    "Tripura": f"{ST}.boro_garo.kokborok",
    "Uchai": f"{ST}.boro_garo.usoi",
    "Garo": f"{ST}.boro_garo.garo",
    "Coach": f"{ST}.boro_garo.koch",
    # "Monipuri" covers both Meitei and Bishnupriya Manipuri (Indo-Aryan) speakers, mostly in
    # Kamalganj; the census does not split them and the two share no node short of the root
    # `other`, so the group is drawn as Meitei, the language the name usually means.
    "Monipuri": f"{ST}.meitei",
    "Mro": f"{ST}.mruic.mru",
    "Chak": f"{ST}.luish.chak",
    "Bawm": f"{ST}.kukichin.bawm",
    "Khumi": f"{ST}.kukichin.khumi",
    "Khiyang": f"{ST}.kukichin.khyang",
    "Pangkhua": f"{ST}.kukichin.pangkhua",
    "Lusai": f"{ST}.kukichin.mizo",
    # no language of their own is established: Barman speak a Bengali dialect or Sadri by
    # source, Dalu a Bengali-Hajong speech, and "Mong" (263 people, scattered) is unidentified
    "Barmon": "bangladesh_other",
    "Dalu": "bangladesh_other",
    "Mong": "bangladesh_other",
    "Other ethnicity": "bangladesh_other",
}

EXCLUDED = set()


def resolve(label):
    return NAMES.get(label)
