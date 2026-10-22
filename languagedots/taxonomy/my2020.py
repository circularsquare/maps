"""Malaysia Census 2020, ethnic and sub-ethnic group -> language node (AGENT_BRIEF §2, ethnicity).

Keyed by the labels sources/my_census.py writes: Table 5's sub-ethnic leaves (state level, as
DOSM prints them, Malay and English run together) and the district table's broad groups.

  * Every named sub-ethnic group is read as its language, all of it: no source gives a
    retention share for any Malaysian group (sources/my.md, "Retention"), so §2's default holds.
  * Malay: `malay`. Melayu Brunei (Sabah only): `brunei_malay`. Kadayan/Kedayan, in both Sabah
    and Sarawak: `kedayan`.
  * Chinese (Cina): reworked 2026-10-19. The 2020 count per district is split by the census's
    own dialect groups (Hokkien, Hakka, Cantonese, Teochew, Foochow, Hainanese, Kwongsai; the
    2000 census, by state, DOSM's unpublished tabulation as printed in Voon 2024), and only
    RETAIN of each group still speaks it: the rest are drawn as Mandarin. The group that no
    column names (Henghua, Hockchia, other Chinese, 4.5% nationally) stays on `sinitic`.
    sources/my_extra.py and countries/my.py; the calls are sources/my.md §3.
  * Indians (India): reworked 2026-10-19: the sub-group shares of INDIANS in sources/my_extra.py
    (Tamil 88%, Telugu, Malayalam, Punjabi, Bengali, Sinhala, Pakistan's languages, 2.6% other
    Indian on `other`). It replaces the 80/20 split (Rajoo 1985, the Tamil share of the
    migration to Malaya), which is no longer used.
  * Non-citizens: drawn since 2026-10-19, from the census's district counts, on the languages of
    their nationalities (sources/my_extra.py, sources/my.md §6). Not a table value of this
    mapping: counts() carries their node ids.
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

# Nodes counts() returns that are not table values: the Chinese varieties and the Indian
# languages (sources/my_extra.py). The non-citizens' ~58 nodes are generated into
# tree.d/my.txt's borrowed block by `python sources/origin_mix.py --fragment my`.
SINITIC = "sinotibetan.sinitic"
EXTRA_NODES = [
    "other", f"{SINITIC}.mandarin", f"{SINITIC}.min_nan", f"{SINITIC}.hakka",
    f"{SINITIC}.cantonese", f"{SINITIC}.teochew", f"{SINITIC}.hainanese", f"{SINITIC}.min_dong",
    f"{SINITIC}.kwongsai", TAMIL, "dravidian.southern.malayalam", "dravidian.southcentral.telugu",
    "indoeuropean.indoaryan.northwestern.punjabi", "indoeuropean.indoaryan.eastern.bengali",
    "indoeuropean.indoaryan.sinhala",
]

# Share of each Chinese dialect group that still speaks it at home; the rest is drawn as
# Mandarin. Carstens (Portland State University), a study of Malaysian Chinese language practice
# presented at UKM, 27 May 2014 (ukm.my/news/?p=4668): "currently 53% of the respondents speak
# Chinese dialects with their parents compared with 42% in 1970"; 93% of families use Mandarin
# alongside dialects; none speak Malay at home, 25% use English with their families. A small
# qualitative sample of working people in towns, so the 53% is the best figure found, not a
# national measure. One rate for every group: nothing separates them (sources/my.md §3).
RETAIN = 0.53

# WHERE inside a state a dialect group lives, as soft weights on the district's Chinese count:
# a group's `home` districts weigh 1, the rest `out`. The state's total for the group is the
# 2000 census's; only where it goes changes. Each line has its source in sources/my.md §3.
# Names are the OpenDOSM district names of sources/my_census.py.
_RIVER = ["Sibu", "Sarikei", "Maradong", "Dalat", "Mukah", "Kanowit", "Selangau", "Kapit", "Song",
          "Julau", "Bintulu", "Tatau", "Daro"]
DIALECT_PLACE = {
    # (state code, group) -> (home districts, weight of the others)
    ("13", "foochow"): (_RIVER, 0.1),        # Rejang basin: "clustered in the Sibu area" (Voon 2024)
    ("08", "foochow"): (["Manjung"], 0.05),  # Sitiawan: Methodist Foochow settlement (Voon 2024)
    ("06", "kwongsai"): (["Bentong", "Raub"], 0.1),            # Wikipedia, Malaysian Chinese
    ("10", "hokkien"): (["Klang"], 0.5),                       # same passage
    ("01", "hokkien"): (["Batu Pahat", "Muar"], 0.5),
    ("07", "hokkien"): (["Timur Laut"], 0.5),
    ("14", "cantonese"): (["W.P. Kuala Lumpur"], 0.5),
    ("08", "cantonese"): (["Kinta"], 0.5),
    ("05", "cantonese"): (["Seremban"], 0.5),
    ("06", "cantonese"): (["Kuantan"], 0.5),
    ("06", "hakka"): (["Cameron Highlands"], 0.5),
    ("01", "hakka"): (["Kulai"], 0.5),
    ("08", "teochew"): (["Manjung", "Hilir Perak", "Larut Dan Matang", "Kerian"], 0.5),
    ("01", "teochew"): (["Johor Bahru"], 0.5),
}

# Non-citizens in Sabah: which districts each nationality leans to, as soft weights (home 1,
# the rest `SABAH_OUT`). Wan Shawaluddin Wan Hassan, Marja Azlima Omar and Ramli Dollah, "The
# Illegal Immigrants in Sabah: Why Do They Come Back?", Borneo Research Journal 4, 2010: the
# Indonesians "in Tawau, Kunak, Lahad Datu, Sandakan and also in Kota Kinabalu", the Filipinos
# "in the eastern part of Sabah especially Sandakan, Lahad Datu, Kudat, the towns closer to the
# Philippines" (Semporna and Tawau are among their fieldwork sites); plantation districts for the
# Indonesians (Bernama 2016 via MPOB: 90% of Sabah's plantation workers Indonesian).
SABAH_HOME = {
    "ID": ["Tawau", "Kunak", "Lahad Datu", "Sandakan", "Kota Kinabalu", "Kinabatangan", "Beluran",
           "Tongod", "Telupid", "Kalabakan"],
    "PH": ["Sandakan", "Lahad Datu", "Kudat", "Semporna", "Tawau", "Kunak"],
}
SABAH_OUT = 0.3

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
