"""Mongolia 2020 census ETHNIC GROUP -> the language node it is drawn as. A proxy: the census
asked no language question (questionnaire Q6 is ethnicity). Keyed by the English national
report's spelling, as sources/mn_census.py writes it. Every ethnic row is `derived`.

Mongol subgroups are read by the dialect group they speak (Glottolog, and Ethnologue's
Kalmyk-Oirat entry for Mongolia):
  Oirat (kalm1243): Durvud, Bayad, Zakhchin, Torguud, Uuld, Myangad, Khoshuud, the Altai
    Uriankhai, the Khoton (a Turkic-origin group that has spoken Dorbet Oirat for centuries) and
    the Darkhad (Glottolog files Darkhat, dark1243, under Kalmyk-Oirat).
  Buryat (bua): Buriad and Barga (Bargu Buryat).
  Khamnigan Mongol (kham1281).
  Mongolian (Halh, khk) for everyone else: Khalkh, Dariganga, Khotgoid, Sartuul, Eljigen,
    Uzemchin, Kharchin.
The census's "Uriankhai" is mostly the Altai Uriankhai of Khovd and Bayan-Olgii (Oirat speakers);
Khovsgol's 10.8% of them are the Uriankhai of the Darkhad depression, who speak Darkhad too, so
the one node serves both. No retention source exists (sources/mn.md): nobody is moved onto
Mongolian, though urban Oirat and Buryat speakers have largely shifted to Khalkha.
"""

NAMES = {
    "Khalkh": "mongolic.mongolian",
    "Dariganga": "mongolic.mongolian",
    "Khotgoid": "mongolic.mongolian",
    "Sartuul": "mongolic.mongolian",
    "Eljigen": "mongolic.mongolian",     # the Eljigen Khalkha of Uvs
    "Uzemchin": "mongolic.mongolian",
    "Kharchin": "mongolic.mongolian",
    "Durvud": "mongolic.oirat",
    "Bayad": "mongolic.oirat",
    "Zakhchin": "mongolic.oirat",
    "Torguud": "mongolic.oirat",
    "Uuld": "mongolic.oirat",
    "Myangad": "mongolic.oirat",
    "Khoshuud": "mongolic.oirat",
    "Uriankhai": "mongolic.oirat",
    "Khoton": "mongolic.oirat",
    "Darkhad": "mongolic.oirat",
    "Buriad": "mongolic.buryat",
    "Barga": "mongolic.buryat",
    "Khamnigan": "mongolic.khamnigan",
    "Kazakh": "turkic.kazakh",
    "Tuva": "turkic.tuvan",
    "Tsaatan (Dukha)": "turkic.dukha",
    "Uzbek (Chantuu)": "turkic.uzbek",
    # Tsakhar, Khorchin, Khalimag, Tumed, Sunud, Tuved, Balba and "other" (143 people), which
    # table 3.6 folds into one column: an unnamed Mongol remainder sits on the family node
    "Other Mongolian ethnic groups": "mongolic",
    # Mongolian citizens of a foreign ethnicity (naturalised), not split by the census
    "Other foreign /Mongolian citizen/": "other",
    # Foreign citizens by citizenship (table 3.6's national shares), read as the country's language
    "Foreign citizen: China": "sinotibetan.sinitic.mandarin",
    "Foreign citizen: Russia": "indoeuropean.slavic.east.russian",
    "Foreign citizen: Korea": "koreanic.korean",
    "Foreign citizen: USA": "indoeuropean.germanic.english",
    "Foreign citizen: other": "other",
}

EXTRA_NODES = []


def resolve(label):
    return NAMES.get(label)
