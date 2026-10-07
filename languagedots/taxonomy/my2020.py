"""Malaysia Census 2020, ethnic and sub-ethnic group -> language node (AGENT_BRIEF §2, ethnicity).

Keyed by the labels sources/my_census.py writes: Table 5's sub-ethnic leaves (state level, as
DOSM prints them, Malay and English run together) and the district table's broad groups.

  * Every named sub-ethnic group is read as its language, all of it: no source gives a
    retention share for any Malaysian group (sources/my.md, "Retention"), so §2's default holds.
  * Malay: `malay`. Melayu Brunei (Sabah only): `brunei_malay`. Kadayan/Kedayan, in both Sabah
    and Sarawak: `kedayan`.
  * Chinese (Cina): `sinitic` itself, "Chinese, language not named". No 2020 or 2010 table splits
    Malaysian Chinese into Hokkien, Hakka, Cantonese, Teochew, Foochow...; the 1970-2000
    censuses did, but only as microdata behind an application (sources/my.md).
  * Indians (India): split by INDIAN below, 80% Tamil and 20% on `other` (Malayalam, Telugu,
    Punjabi and the rest cross Dravidian and Indo-Aryan). The 80% is the Tamil share of the
    Indian migration to Malaya (R. Rajoo, "Malaysian World-view", ed. Mohd. Taib Osman, 1985,
    pp. 149-150), the only figure found; no census since 1970 splits Indians.
  * Lain-lain (Others: Thai, Eurasian, Portuguese-Eurasian and others, unlisted): `other`.
  * Melayu Proto (Jakun, Temuan, Semelai, Orang Kanaq, Orang Seletar, Orang Kuala): on
    `malayic`, unnamed. All but Semelai speak Malayic languages; Semelai (Aslian) is a few
    thousand of the 95,000 and nothing separates them.
  * Senoi and Negrito: `aslian`, unnamed (peoples, each speaking several Aslian languages).
  * Bumiputera Sabah Lain / Sarawak Lain (other Sabah / Sarawak Bumiputera, unlisted: Tidung,
    Kagayan, Ubian, Cocos and many more): `austronesian`, unnamed.
  * Lun Bawang/Murut (Sarawak) and Sabah's Lundayuh/Lundayeh are one people and one language
    (Glottolog lunb1237 is a dialect of lund1271), so one node. Sarawak's "Murut" is this
    people, not Sabah's Murut.
  * Tagal (Sarawak) is a Murutic language (taga1273) but printed apart from Murut, so its own
    leaf. Bisayah (Sarawak) and Sabah's Bisaya/Bisayah share `bisaya`.
  * Javanese or Jawa (Sarawak): `javanese`.
  * Non-citizens are not drawn: the census gives them no ethnic group (sources/my.md).
"""
AN = "austronesian"
NB = f"{AN}.north_borneo"
ML = f"{AN}.malayic"
PH = f"{AN}.philippine"
TAMIL = "dravidian.southern.tamil"

NAMES = {
    # broad groups (district table, and Table 5's three non-Bumiputera rows)
    "Malay": f"{ML}.malay",
    "Melayu Malay": f"{ML}.malay",
    "Chinese": "sinotibetan.sinitic",
    "Cina Chinese": "sinotibetan.sinitic",
    "Indians": TAMIL,                 # shared by INDIAN in countries/my.py
    "India Indians": TAMIL,
    "Others": "other",
    "Lain-lain Others": "other",
    # Orang Asli
    "Negrito": "austroasiatic.aslian",
    "Senoi": "austroasiatic.aslian",
    "Melayu Proto": ML,
    # Sabah
    "Bajau": f"{AN}.sama_bajaw.bajau",
    "Balabak/ Molbog": f"{PH}.molbog",
    "Bisaya/ Bisayah": f"{NB}.bisaya",
    "Bulongan": f"{NB}.bulongan",
    "Idahan/ Ida'an": f"{NB}.idaan",
    "Iranun/ Ilanun": f"{PH}.iranun",
    "Kadayan/ Kedayan": f"{ML}.kedayan",
    "Kadazan/ Dusun": f"{NB}.kadazandusun",
    "Melayu Brunei": f"{ML}.brunei_malay",
    "Murut": f"{NB}.murut",
    "Orang Sungai/ Sungoi": f"{NB}.orang_sungai",
    "Rungus": f"{NB}.rungus",
    "Suluk": f"{PH}.tausug",
    "Lundayuh/ Lundayeh": f"{NB}.lundayeh",
    "Bumiputera Sabah Lain": AN,
    "Bumiputera Sabah Lain Other Sabah Bumiputera": AN,
    # Sarawak
    "Iban": f"{ML}.iban",
    "Bidayuh": f"{AN}.bidayuh",
    "Melanau": f"{NB}.melanau",
    "Bakong": f"{NB}.bakong",
    "Berawan": f"{NB}.berawan",
    "Dali'": f"{NB}.dali",
    "Javanese or Jawa (Sarawak)": f"{AN}.javanese",
    "Bisayah (Sarawak)": f"{NB}.bisaya",
    "Bukitan": f"{NB}.bukitan",
    "Kadayan (Sarawak)": f"{ML}.kedayan",
    "Kajang": f"{NB}.kajang",
    "Kanowit": f"{NB}.kanowit",
    "Kayan": f"{NB}.kayan",
    "Kiput or Lakiput": f"{NB}.kiput",
    "Kalabit": f"{NB}.kelabit",
    "Kenyah": f"{NB}.kenyah",
    "Miriek (Miri)": f"{NB}.miriek",
    "Narom": f"{NB}.narom",
    "Lugat": f"{NB}.lugat",
    "Sa'ban/Saben": f"{NB}.saban",
    "Lun Bawang/ Murut (Sarawak)": f"{NB}.lundayeh",
    "Segan (Baie)": f"{NB}.segan",
    "Penan": f"{NB}.penan",
    "Sihan": f"{NB}.sihan",
    "Sabup": f"{NB}.sabup",
    "Tabun": f"{NB}.tabun",
    "Tetau or Tatau": f"{NB}.tatau",
    "Tagal": f"{NB}.tagal",
    "Tanjong": f"{NB}.tanjong",
    "Ukit": f"{NB}.ukit",
    "Bumiputera Sarawak Lain": AN,
    "Bumiputera Sarawak Lain Other Sarawak Bumiputera": AN,
}

# Indians, shared (see the docstring)
INDIAN = [(TAMIL, 0.8), ("other", 0.2)]
EXTRA_NODES = ["other"]

# Which 2010 district column (sources/my_census.py, level seed2010) gives each Sabah or Sarawak
# group its spread over districts inside the state; anything not listed takes "Other Bumiputera".
SEED = {
    "12": {"Melayu Malay": "Malay", "Melayu Brunei": "Malay", "Kadazan/ Dusun": "Kadazan Dusun",
           "Bajau": "Bajau", "Murut": "Murut",
           "Lundayuh/ Lundayeh": "Murut"},
    "13": {"Melayu Malay": "Malay", "Iban": "Iban", "Bidayuh": "Bidayuh", "Melanau": "Melanau"},
}

# HOME DISTRICTS for the Sabah and Sarawak groups that the 2010 table does not name: without
# this, each would be spread wherever 2010's "Other Bumiputera" lives in its state (Rungus over
# Tawau). Each is the set of 2020 districts its language is spoken in, from the language's
# Glottolog point and the district each Wikipedia language and people article names; the
# census count for the state is unchanged, only where in the state it goes. Groups not listed
# (and every group outside Sabah and Sarawak) follow their seed alone.
HOME = {
    "12": {
        "Rungus": ["Kudat", "Kota Marudu", "Pitas"],
        "Orang Sungai/ Sungoi": ["Kinabatangan", "Beluran", "Tongod", "Telupid", "Sandakan",
                                 "Pitas"],
        "Suluk": ["Sandakan", "Tawau", "Lahad Datu", "Semporna", "Kunak", "Kudat",
                  "Kota Kinabalu", "Kalabakan"],
        "Iranun/ Ilanun": ["Kota Belud", "Lahad Datu", "Kudat"],
        "Bisaya/ Bisayah": ["Beaufort", "Kuala Penyu", "Sipitang"],
        "Kadayan/ Kedayan": ["Sipitang", "Beaufort", "Kuala Penyu", "Papar"],
        "Idahan/ Ida'an": ["Lahad Datu", "Kinabatangan", "Sandakan"],
        "Melayu Brunei": ["Papar", "Beaufort", "Kuala Penyu", "Sipitang", "Kota Kinabalu",
                          "Putatan", "Penampang"],
        "Balabak/ Molbog": ["Kudat"],
    },
    "13": {
        "Kayan": ["Belaga", "Telang Usan", "Marudi", "Miri", "Bintulu", "Kapit"],
        "Kenyah": ["Belaga", "Telang Usan", "Marudi", "Miri", "Bintulu", "Kapit"],
        "Penan": ["Belaga", "Telang Usan", "Marudi", "Limbang"],
        "Kalabit": ["Telang Usan", "Miri", "Marudi"],
        "Lun Bawang/ Murut (Sarawak)": ["Lawas", "Limbang"],
        "Berawan": ["Telang Usan", "Marudi", "Miri"],
        "Kadayan (Sarawak)": ["Miri", "Subis", "Limbang", "Lawas", "Beluru"],
        "Bisayah (Sarawak)": ["Limbang"],
        "Kajang": ["Belaga", "Kapit"],
        "Tagal": ["Lawas"],
        "Sa'ban/Saben": ["Telang Usan"],
        "Lugat": ["Telang Usan"],
        "Ukit": ["Belaga", "Kapit", "Bukit Mabong"],
        "Bukitan": ["Belaga", "Kapit", "Bukit Mabong"],
        "Sihan": ["Belaga"],
        "Kanowit": ["Kanowit"],
        "Tanjong": ["Kanowit"],
        "Bakong": ["Marudi", "Beluru"],
        "Kiput or Lakiput": ["Marudi", "Beluru"],
        "Narom": ["Miri", "Beluru"],
        "Miriek (Miri)": ["Miri"],
        "Dali'": ["Marudi", "Beluru"],
        "Tetau or Tatau": ["Tatau"],
    },
}


def resolve(label):
    return NAMES[label]
