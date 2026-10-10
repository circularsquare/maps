"""Bangladesh 2011 census ETHNIC GROUP -> the language node it is drawn as. A proxy: the census
asked no language question. Keyed by the USCB file's English spelling of BBS's groups.

Anita allowed the proxy on 2026-10-05 (relayed by the supervisor); every row is `derived`.
sources/bd.md has the reasoning and the known misfits, which are the point of reading it:

  * `Not an ethnic minority` (98.9%) is Bengali in NAMES, but countries/bd.py splits it by place
    through split_remainder() below (2026-10-07): Sylheti in the Sylheti-speaking upazilas,
    Chittagonian in Chittagong (but Sandwip) and Cox's Bazar, 56% of it in Chittagong city.
    It also holds the Urdu-speaking "Bihari" camps (not an ethnic category) and the Rohingya
    outside the camps, drawn with their neighbours.
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

# ---- Place split of `Not an ethnic minority` (spec §3.3; 2026-10-07, fix-bd, Anita asked for it).
# The census's Bengali remainder holds the Sylheti and Chittagonian speakers, which Glottolog
# files as languages of their own (sylh1242 under Eastern Bengali, chit1275 under Southeastern
# Bengali). No source counts them, so the remainder is drawn by where it lives. Boundaries and
# their sources are in sources/bd.md §4; countries/bd.py applies split_remainder().
SYLHETI = f"{IA}.eastern.sylheti"
CHITTAGONIAN = f"{IA}.eastern.chittagonian"
ROHINGYA = f"{IA}.eastern.rohingya"   # the camps, from UNHCR (sources/bd_unhcr.py), not the census
EXTRA_NODES = [SYLHETI, CHITTAGONIAN, ROHINGYA]

# Sylheti: all of Sylhet and Moulvibazar districts, eastern Sunamganj and north-eastern Habiganj.
# Western Sunamganj (Derai, Dharampasha, Jamalganj, Sulla, Tahirpur, the haors next to Netrokona)
# and the rest of Habiganj speak varieties closer to Mymensingh and Brahmanbaria, and stay Bengali.
SYLHETI_UNITS = (
    {f"BGD_08_04_{i:02d}" for i in range(1, 13)}            # Sylhet district, 12
    | {f"BGD_08_02_{i:02d}" for i in range(1, 8)}           # Moulvibazar district, 7
    | {"BGD_08_03_01", "BGD_08_03_02", "BGD_08_03_03",      # Bishwambarpur, Chhatak, Dakshin Sunamganj
       "BGD_08_03_06", "BGD_08_03_07", "BGD_08_03_10"}      # Dowarabazar, Jagannathpur, Sunamganj Sadar
    | {"BGD_08_01_02", "BGD_08_01_08"}                      # Bahubal, Nabiganj (Habiganj)
)
# Chittagonian: Chittagong district but Sandwip (whose island speech goes with Noakhali), and all
# of Cox's Bazar (Teknaf's and Ukhia's local speech included; it is Chittagonian, close to Rohingya).
CHITTAGONIAN_UNITS = (
    {f"BGD_02_04_{i:02d}" for i in range(1, 26)} - {"BGD_02_04_23"}
    | {f"BGD_02_06_{i:02d}" for i in range(1, 9)}
)
# Chittagong City Corporation's eleven 2011 thanas: a city of in-migrants. The World Bank's 2019
# Dhaka-Chittagong spatial survey found 48.0% of Chittagong respondents born in the community and
# 8.2% from elsewhere in Chittagong; that 56% is drawn as Chittagonian, the rest as Bengali.
CHITTAGONG_CITY = {f"BGD_02_04_{i:02d}" for i in (2, 4, 7, 8, 9, 11, 13, 14, 17, 18, 20)}
CITY_SHARE = 0.48 + 0.082

assert len(SYLHETI_UNITS) == 27 and len(CHITTAGONIAN_UNITS) == 32 and CHITTAGONG_CITY < CHITTAGONIAN_UNITS


def split_remainder(unit):
    """[(node, share)] for the Bengali remainder of one upazila; shares sum to 1."""
    bengali = NAMES["Not an ethnic minority"]
    if unit in SYLHETI_UNITS:
        return [(SYLHETI, 1.0)]
    if unit in CHITTAGONG_CITY:
        return [(CHITTAGONIAN, CITY_SHARE), (bengali, 1.0 - CITY_SHARE)]
    if unit in CHITTAGONIAN_UNITS:
        return [(CHITTAGONIAN, 1.0)]
    return [(bengali, 1.0)]


def resolve(label):
    return NAMES.get(label)
