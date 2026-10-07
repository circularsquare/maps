"""Thailand 2010 Population and Housing Census, language usually spoken in the household -> node.

Keyed by the row labels sources/th_census.py writes (the reports' English labels, canonicalised;
the Thai label beside each is in data/normalized/th.csv). The three-way split rows ("Only Thai
language", "Thai and other languages", "Only other languages") are not languages: countries/th.py
turns them into Thai and into the scale of the language rows, so only the 40 language rows and
"Thai" are resolved here.

  * "Thai" is every household that answered Thai. The census has no box for Isan, Kham Mueang
    (Northern Thai) or Pak Tai (Southern Thai), and its manual counts them as Thai. Since
    2026-10-06 (Anita's go-ahead for a proxy that changes counts) countries/th.py splits each
    changwat's Thai into Central Thai (`kradai.thai`), Isan, Northern Thai and Southern Thai by
    World Values Survey shares (sources/th_wvs.py); those three are EXTRA_NODES below.
  * "Thaikueng" (ไทยขึน/ไทยเลย/ลาวเลย, Tai Khün / Thai Loei / Lao Loei) is one census answer
    covering two Tai varieties: Khün in Chiang Mai, Lamphun, Chiang Rai and Mae Hong Son, and the
    speech of Loei province (364,000 of its 788,000 are in Loei). It also holds 60,000 in Kalasin
    and 49,000 in Mukdahan, where neither is spoken and Phu Thai is; the census does not say what
    those households said. One leaf, `khun_loei`, labelled with the census's three names, rather
    than a split by province that would put a name on the Kalasin and Mukdahan households.
  * "Lao-krung" (ลาวครั่ง/ลาวขี้ครั่ง) is Lao Khrang, a Lao variety of the central plains
    (Phetchabun, Suphan Buri, Uthai Thani): its own leaf beside `lao`. "Lao" is Lao.
  * "Morn" is Mon. "Hmong/Mea" is Hmong (Meo is an old exonym). "Burmese", "Vietnamese",
    "Korean", "Japanese" as named. "Cambodia" (เขมร) is Khmer: in this table it is mostly
    Cambodian Khmer in Bangkok and the east (Trat, Rayong, Chon Buri); the Northern Khmer of
    Surin, Buri Ram and Si Sa Ket sits almost entirely in "Local languages" (below).
  * "Karen" (กะเหรี่ยง) names the Karen languages as a group (S'gaw, Pwo, Pa-O, Kayah), not one
    of them, so it sits on the Karen group node as au2021's and ca2021's "Karen" do.
  * "Chinese" (จีน) names no variety (Teochew, Hokkien, Hakka, Yunnanese Mandarin in the north):
    `sinitic`, which build.py draws unwashed as "Chinese".
  * "Malay/yawi" (มลายูถิ่น/นายู/ยาวี, local Malay, Nayu, Yawi) is the Malay of the far south:
    Pattani Malay, its own leaf (Glottolog patt1254). Satun's 16,600 speak a Kedah-type Malay,
    but the census gives them the same box. "Malaysia" (มาเลเซีย) is Malaysian Malay: `malay`.
    "Indonesia" is Indonesian.
  * "Local languages" (ภาษาถิ่น) and "Dialect and others in Thailand" (ภาษาพื้นเมืองและชาวเขาอื่นๆ,
    other indigenous and hill-tribe languages) are the census's two unnamed boxes for languages
    of Thailand: on `seasia_other`. The first is 552,000 in Surin, 180,000 in Buri Ram and
    131,000 in Si Sa Ket, the Northern Khmer and Kuy country; the second is mostly Chiang Rai and
    Chiang Mai (Akha, Lahu, Lisu, Mien, Lawa). Neither is guessed into a language.
  * "Tagalog/Filipino" is Tagalog; "Bengali/Banca Lee/Bangladesh" is Bengali; "India/Hindi"
    (อินเดีย/ฮินดี) is Hindi, "ภาษาอินเดีย" being the everyday Thai name for Hindi; "Arab" Arabic.
  * "Mexican" and "Cuban" are answers naming a country whose language is Spanish: Spanish.
    "Portugal" is Portuguese. The other European rows as named.
  * "Other languages in Asia" and "Other languages in Europe, America, Australia" are foreign
    remainders: `other`. "Africa" and "Other languages in Africa": `africa_other`.
  * Not drawn: households that answered Thai and another language or another language only,
    but whose other language the table gives no row (81,519 people in the kingdom).
"""
IA = "indoeuropean.indoaryan"
ROM = "indoeuropean.romance"

NAMES = {
    "Thai": "kradai.thai",
    "Karen": "sinotibetan.karen",
    "Thaikueng": "kradai.khun_loei",
    "Morn": "austroasiatic.mon",
    "Lao-krung": "kradai.lao_khrang",
    "Hmong/Mea": "hmongmien.hmong",
    "Local languages": "seasia_other",
    "Malay/yawi": "austronesian.malayic.pattani_malay",
    "Dialect and others in Thailand": "seasia_other",
    "Chinese": "sinotibetan.sinitic",
    "Burmese": "sinotibetan.burmish.burmese",
    "Vietnamese": "austroasiatic.vietnamese",
    "Lao": "kradai.lao",
    "Cambodia": "austroasiatic.khmer",
    "Korean": "koreanic.korean",
    "Japanese": "japonic.japanese",
    "Tagalog/Filipino": "austronesian.philippine.tagalog",
    "Bengali/Banca Lee/Bangladesh": f"{IA}.eastern.bengali",
    "Malaysia": "austronesian.malayic.malay",
    "Indonesia": "austronesian.malayic.indonesian",
    "India/Hindi": f"{IA}.central.hindi",
    "Arab": "afroasiatic.arabic",
    "Other languages in Asia": "other",
    "English": "indoeuropean.germanic.english",
    "German": "indoeuropean.germanic.continental.german",
    "Greek": "indoeuropean.hellenic.greek",
    "Spanish": f"{ROM}.spanish",
    "Polish": "indoeuropean.slavic.west.polish",
    "Portugal": f"{ROM}.portuguese",
    "Russian": "indoeuropean.slavic.east.russian",
    "Swedish": "indoeuropean.germanic.north.swedish",
    "Finnish": "uralic.finnish",
    "French": f"{ROM}.french",
    "Danish": "indoeuropean.germanic.north.danish",
    "Italian": f"{ROM}.italian",
    "Hungarian": "uralic.hungarian",
    "Mexican": f"{ROM}.spanish",
    "Cuban": f"{ROM}.spanish",
    "Other languages in Europe, America, Australia": "other",
    "Africa": "africa_other",
    "Other languages in Africa": "africa_other",
}

# countries/th.py's split of "Thai" by the World Values Survey (sources/th_wvs.py variety -> node;
# "central" stays on NAMES["Thai"])
VARIETY_NODES = {
    "central": "kradai.thai",
    "isan": "kradai.isan",
    "northern": "kradai.northern_thai",
    "southern": "kradai.southern_thai",
}
EXTRA_NODES = ["kradai.isan", "kradai.northern_thai", "kradai.southern_thai"]

# the three-way split: turned into Thai and a scale by countries/th.py, never drawn as such
SPLIT = ("Total", "Only Thai language", "Thai and other languages", "Only other languages")


def resolve(label):
    return NAMES[label]
