"""Indonesia: share each province's "other regional languages" remainder out among named
languages. Imported by sources/id_sp2010.py; every row it produces is `modelled`.

THE PROBLEM. BPS's 2010 census tables name eight home languages by province (L4.5). The other
regional languages, 42,560,185 people, are a single number per province (L4.2's regional box less
the seven regional languages of L4.5). What BPS does publish around it:
  * L4.1: the same remainder NATIONALLY, in 24 language groups ("Bali", "Batak", "bahasa-bahasa
    asal NTT", "Melayu Perdagangan", ...) plus sign language. Their sum is the remainder exactly
    (id_sp2010.py check 5).
  * L2.6: province x 31 ethnic groups ("Batak", "Suku asal NTT", "Suku Asal Sulawesi lainnya").
Outside BPS's own tables:
  * Ananta, Arifin, Hasbullah, Handayani and Pramono, *Demography of Indonesia's Ethnicity*
    (ISEAS 2015) pp.119-122, the national count of the 145 largest ethnic groups of the same
    census under their "new classification" (NC) of BPS's 1,331 ethnic codes, which takes BPS's
    "Suku asal X" groups apart (Atoni, Manggarai, Toraja, Kaili, Dani ...). Read through the copy
    in English Wikipedia's "List of ethnic groups in Indonesia by population" (2026-10-05); the
    book itself is not open. National only.
  * Ananta et al., "Changing Ethnic Composition: Indonesia, 2000-2010" (IUSSP 2013), Table 4: the
    share of each large ethnic group that used its own language at home in 2010 (Batak 43.1%,
    Betawi 25.4%, Balinese 92.7%, ... "Others" 31.6%).
  * Katadata Insight Center (databoks, 2021-10-07), North Sumatra's ethnic groups attributed to
    BPS's 2010 census: Tapanuli/Toba 25.62%, Mandailing 11.27%, Karo 5.09%, Simalungun 2.04%,
    Pakpak 0.73% (Batak 44.75%). The only split of "Batak" found.
No table of language or ethnic group by regency was found (sources/id.md §2).

THE MODEL, three stages, all at the 33 provinces:
  0. Fine ethnic groups per province. Each L2.6 group is shared among its NC members by a rake
     (IPF) to L2.6's province counts and the NC national counts, seeded by how near each
     province's people live to the member's Glottolog point(s) (a 50 km kernel, 95% of the seed;
     the rest even). Members BPS's group holds that NC moved elsewhere, or that NC does not
     name, are the group's residual.
  1. The remainder per province x L4.1 group. A rake to the provinces' measured remainders (rows)
     and L4.1's measured national totals (columns), seeded by stage 0's groups times the share
     of that group that used its own language at home (IUSSP Table 4), each group seeding the
     L4.1 group(s) its language is filed in. Ethnic groups whose language has no L4.1 group of
     its own (Ambonese, Minahasa, Papuans, NTT) also seed "Melayu Perdagangan", the trade and
     creole Malays, which no ethnic group stands for. L4.1's "Sulawesi Utara" and "lain asal
     Sulawesi" are raked as one, because nothing says which of the two holds Gorontalo.
  2. Each province x L4.1 group shared among the languages of its members, by stage 0 x
     retention. "Batak" is split by Katadata's North Sumatra shares in every province.
     "Melayu Perdagangan" is named by place: Manado Malay in North and Central Sulawesi and
     Gorontalo, Ambonese Malay in Maluku, North Moluccan Malay in North Maluku, Papuan Malay in
     the two Papuas, Kupang Malay in NTT, "trade Malay, variety not named" elsewhere. Sign
     language follows population. "Bahasa-bahasa asal Papua" (since 2026-10-06) is shared
     among Indonesian New Guinea's 234 Glottolog languages by sources/id_papua.py's regency
     model (Ananta et al. 2016), the province's own mix in Papua and Papua Barat and both
     together elsewhere; Papua's NC clusters below now only seed stages 0 and 1.
Rounded per province by largest remainder, so each province's rows sum to its measured
remainder exactly and each L4.1 group's national total is met to rounding.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
UNITS = ROOT / "data" / "geo" / "id" / "id_units.csv"
GLOTTOLOG = ROOT / "data" / "raw" / "glottolog" / "languages.csv"

KERNEL0_KM = 50.0   # stage 0: how far a Glottolog point pulls a group between provinces
LAMBDA0 = 0.95

# IUSSP 2013, Table 4, "own language" (persons aged 5+, 2010). "other" is the table's "Others".
RET = {"aceh": .8417, "batak": .4311, "betawi": .2541, "bali": .9269, "dayak": .6162,
       "sasak": .9394, "other": .3159}

# L4.1's 24 regional language groups other than the seven named in L4.5, plus sign language.
# SULUT is "Bahasa-bahasa asal Sulawesi Utara" + "Bahasa-bahasa lain asal Sulawesi", raked as one.
SULUT = "Bahasa-bahasa asal Sulawesi Utara + lain asal Sulawesi"
COLS = {
    "Aceh": ["Aceh"], "Aceh Lainnya": ["Bahasa-bahasa asal Aceh Lainnya"], "Batak": ["Batak"],
    "Nias": ["Nias"], "Sumatera": ["Bahasa-bahasa asal Sumatera"],
    "Musi": ["Musi/ Palembang/ Sekayu"], "Melayu Tengah": ["Melayu Tengah"],
    "Lampung": ["Bahasa-bahasa asal Lampung"], "Betawi": ["Betawi"],
    "Cirebon": ["Cirebon-Indramayu"], "Bali": ["Bali"], "Sasak": ["Lombok/Sasak"],
    "NTB": ["Bahasa-bahasa asal NTB Lainnya"], "NTT": ["bahasa-bahasa asal NTT"],
    "Dayak": ["Dayak"], "Kalimantan": ["Bahasa-bahasa lainnya asal Kalimantan"],
    "Makassar": ["Makassar"], "Sulselbar": ["Bahasa-bahasa asal Sulselbar"],
    "STT": ["Bahasa-bahasa asal Sulawesi Timur Tenggara"],
    "SULUT": ["Bahasa-bahasa asal Sulawesi Utara", "Bahasa-bahasa lain asal Sulawesi"],
    "Maluku": ["Bahasa-bahasa asal Maluku"], "Papua": ["Bahasa-bahasa asal Papua"],
    "MP": ["Melayu Perdagangan"], "Isyarat": ["Bahasa Isyarat"],
}

# Languages: key -> (label written to the normalized CSV, Glottolog codes that place it).
# The node each label goes on is taxonomy/id2010.py's business.
A = "Austronesian"
LANG = {
    "acehnese": ("Aceh", ["achi1257"]),
    "gayo": ("Gayo", ["gayo1244"]),
    "alas_kluet": ("Alas-Kluet", ["bata1292"]),
    "singkil": ("Singkil", ["sing1240"]),
    "simeulue": ("Simeulue", ["sime1241"]),
    "aneuk_jamee": ("Aneuk Jamee", []),       # Glottolog's point is Minangkabau's, in Padang
    "tamiang": ("Tamiang", []),
    "toba": ("Batak Toba", ["bata1289"]),
    "mandailing": ("Batak Mandailing dan Angkola", ["bata1291", "bata1290"]),
    "karo": ("Batak Karo", ["bata1293"]),
    "simalungun": ("Batak Simalungun", ["bata1288"]),
    "pakpak": ("Batak Pakpak Dairi", ["bata1294"]),
    "nias": ("Nias", ["nias1242"]),
    "kerinci": ("Kerinci", ["keri1250"]),
    "musi": ("Musi/Palembang", ["musi1241"]),
    "central_malay": ("Melayu Tengah (Besemah, Semendo, Ogan, Enim)", ["cent2053"]),
    "col": ("Lembak (Col)", ["coll1240"]),
    "komering": ("Komering", ["kome1238"]),
    "lampung": ("Lampung", ["lamp1242", "lamp1243"]),
    "rejang": ("Rejang", ["reja1240"]),
    "mentawai": ("Mentawai", ["ment1249"]),
    "pekal": ("Pekal", ["peka1242"]),
    "kaur": ("Kaur", ["kaur1269"]),
    "betawi": ("Betawi", ["beta1252"]),
    "cirebonese": ("Cirebon", ["cire1240"]),
    "balinese": ("Bali", ["bali1278"]),
    "sasak": ("Sasak", ["sasa1249"]),
    "bima": ("Bima", ["bima1247"]),
    "sumbawa": ("Sumbawa", ["sumb1241"]),
    "uab_meto": ("Uab Meto (Dawan)", ["uabm1237"]),
    "manggarai": ("Manggarai", ["mang1405"]),
    "sumba": ("Sumba", ["kamb1299", "weje1237", "kodi1247", "anak1240", "wanu1241",
                        "lamb1273", "mamb1305"]),
    "lamaholot": ("Lamaholot", ["lama1277"]),
    "ngada": ("Ngada", ["ngad1261"]),
    "tetun": ("Tetun", ["tetu1245"]),
    "rote": ("Rote", ["dela1251", "term1237", "deng1253"]),
    "alor": ("Alor-Pantar", ["abui1241", "blag1240", "kabo1247", "kuii1253", "kama1365",
                             "kelo1247", "adan1251", "alor1247"]),
    "lio": ("Lio", ["lioo1240"]),
    "hawu": ("Hawu (Sabu)", ["sabu1255"]),
    "dayak": ("Dayak", None),                  # None: every Dayak point in Glottolog, below
    "kutai": ("Kutai", ["teng1267", "kota1275"]),
    "paser": ("Paser", ["pasi1257"]),
    "makassarese": ("Makassar", ["maka1311"]),
    "minahasan": ("Minahasa", ["tont1239", "tond1251", "tomb1243", "tons1240", "tons1239"]),
    "gorontalo": ("Gorontalo", ["goro1259"]),
    "toraja": ("Toraja", ["tora1261"]),
    "mandar": ("Mandar", ["mand1442"]),
    "tae": ("Tae' (Luwu)", ["taee1237"]),
    "duri": ("Duri", ["duri1242"]),
    "mamasa": ("Mamasa", ["mama1276"]),
    "selayar": ("Selayar", ["sela1260"]),
    "mamuju": ("Mamuju", ["mamu1255"]),
    "buton": ("Buton (Wolio, Cia-Cia, Tukang Besi)", ["woli1241", "ciac1237", "tuka1248",
                                                       "tuka1249", "lasa1237", "kumb1274"]),
    "tolaki": ("Tolaki", ["tola1247"]),
    "muna": ("Muna", ["muna1247"]),
    "moronene": ("Moronene", ["moro1287"]),
    "banggai": ("Banggai", ["bang1368"]),
    "saluan": ("Saluan", ["salu1253"]),
    "sangir": ("Sangir", ["sang1336"]),
    "mongondow": ("Mongondow", ["mong1342"]),
    "talaud": ("Talaud", ["tala1285"]),
    "kaili": ("Kaili", ["ledo1238", "daak1235", "unde1235"]),
    "pamona": ("Pamona", ["pamo1252"]),
    "buol": ("Buol", ["buol1237"]),
    "tomini": ("Tomini", ["tomi1243"]),
    "lauje": ("Lauje", ["lauj1238"]),
    "bajau": ("Bajau", ["indo1317"]),
    "kei": ("Kei", ["keii1239"]),
    "seram": ("Seram", ["alun1238", "nort2864", "manu1258", "sepa1242", "nort2867", "sout2895",
                        "bobo1254", "masi1266"]),
    "tanimbar": ("Tanimbar (Yamdena, Fordata, Selaru)", ["yamd1240", "ford1242", "sela1259"]),
    "sula": ("Sula", ["sula1248"]),
    "buru": ("Buru", ["buru1303"]),
    "geser_gorom": ("Geser-Gorom", ["gese1240"]),
    "aru": ("Aru", ["ujir1237", "kola1285", "dobe1238"]),
    "kisar": ("Kisar", ["kisa1266"]),
    "babar": ("Babar", ["nort2860", "dawe1237", "empl1237", "tela1241", "sout2883"]),
    "ternate": ("Ternate", ["tern1247"]),
    "tidore": ("Tidore", ["tido1248"]),
    "tobelo": ("Tobelo", ["tobe1252"]),
    "galela": ("Galela", ["gale1259"]),
    "loloda": ("Loloda", ["lolo1264"]),
    "tabaru": ("Tabaru", ["taba1263"]),
    "biak": ("Biak", ["biak1248"]),
    "yapen": ("Yapen", ["ansu1237", "seru1244", "amba1265", "busa1254", "pomm1237",
                        "mara1397"]),
    "waropen": ("Waropen", ["waro1242"]),
    "dani": ("Dani", ["west2594", "midg1235", "uppe1430", "lowe1415", "wala1269"]),
    "ekari": ("Ekari (Mee)", ["ekar1243"]),
    "yali": ("Yali", ["angg1239", "nini1235"]),
    "nduga": ("Nduga", ["ndug1245"]),
    "moni": ("Moni", ["moni1261"]),
    "hupla": ("Hupla", ["hupl1238"]),
    "damal": ("Damal", ["dama1272"]),
    "ketengban": ("Ketengban", ["kete1254"]),
    "kamoro": ("Kamoro", ["kamo1255"]),
    "asmat": ("Asmat", ["cent2117", "casu1237", "yaos1235"]),
    "citak": ("Citak", ["cita1245", "tamn1235"]),
    "ngalum": ("Ngalum", ["ngal1298"]),
    "marind": ("Marind", ["nucl1622", "bian1251"]),
    "yaqay": ("Yaqay", ["yaqa1246"]),
    "arfak": ("Arfak (Hatam, Meyah, Sougb)", ["hata1243", "meya1236", "mani1235", "mosk1236",
                                               "mans1260"]),
    "maybrat": ("Maybrat", ["maib1239"]),
    "moi": ("Moi", ["moii1235"]),
    "sentani": ("Sentani", ["nucl1632"]),
    "baham": ("Baham", ["baha1258"]),
    "manado_malay": ("Melayu Perdagangan: Melayu Manado", ["mala1481"]),
    "ambonese_malay": ("Melayu Perdagangan: Melayu Ambon", ["ambo1250"]),
    "north_moluccan_malay": ("Melayu Perdagangan: Melayu Maluku Utara", ["nort2828"]),
    "papuan_malay": ("Melayu Perdagangan: Melayu Papua", ["papu1250"]),
    "kupang_malay": ("Melayu Perdagangan: Melayu Kupang", ["kupa1239"]),
    "trade_malay": ("Melayu Perdagangan: ragam tidak disebut", []),
    "sign": ("Bahasa Isyarat", []),
    "unnamed": ("tidak disebut", []),
}

# Indonesian New Guinea's languages (sources/id_papua.py, 2026-10-06): "Bahasa-bahasa asal
# Papua" is split among them in stage 2, in every province, instead of among the NC clusters
# below, which now only seed stages 0 and 1. Keyed "pap_<glottocode>".
PAP_LANG = ROOT / "data" / "normalized" / "id_papua_languages.csv"
PAPUA_STAGE0_ONLY = {"biak", "yapen", "waropen", "dani", "ekari", "yali", "nduga", "moni",
                     "hupla", "damal", "ketengban", "kamoro", "asmat", "citak", "ngalum",
                     "marind", "yaqay", "arfak", "maybrat", "moi", "sentani", "baham"}


def _add_papua_langs():
    if not PAP_LANG.exists():
        return
    d = pd.read_csv(PAP_LANG, dtype=str)
    for r in d.itertuples():
        LANG[f"pap_{r.glottocode}"] = (r.label, [r.glottocode])


_add_papua_langs()

# Melayu Perdagangan by province (stage 2): a label whose meaning depends on place.
MP_NODE = {"71": "manado_malay", "72": "manado_malay", "75": "manado_malay",
           "81": "ambonese_malay", "82": "north_moluccan_malay", "91": "papuan_malay",
           "94": "papuan_malay", "53": "kupang_malay"}

# Katadata (BPS 2010, North Sumatra): Batak's split, applied in every province.
BATAK_SPLIT = {"toba": 25.62, "mandailing": 11.27, "karo": 5.09, "simalungun": 2.04,
               "pakpak": 0.73}

# L2.6 group -> members: (language key, NC national count, {L4.1 col: share}, retention).
# A count of None marks the group's residual (L2.6's count less its NC members; floored at 0).
# cols {} means the members' language is already counted in L4.5 (Malay): no share.
_O, _MP = "other", "MP"
def _m(lang, n, cols, ret=_O):
    return (lang, n, cols, ret)
EAST = {"Maluku": .5, _MP: .5}
PAP = {"Papua": .5, _MP: .5}
NTT = {"NTT": .85, _MP: .15}
SUL = {"SULUT": .8, _MP: .2}
GROUPS = {
    "Suku asal Aceh": [
        _m("acehnese", 3_404_109, {"Aceh": 1}, "aceh"),
        _m("gayo", 336_856, {"Aceh Lainnya": 1}), _m("alas_kluet", 98_223, {"Aceh Lainnya": 1}),
        _m("simeulue", 67_722, {"Aceh Lainnya": 1}),
        _m("aneuk_jamee", 63_357, {"Aceh Lainnya": 1}),
        _m("singkil", 52_982, {"Aceh Lainnya": 1}), _m("tamiang", 52_901, {"Aceh Lainnya": 1}),
        _m("unnamed", None, {"Aceh Lainnya": 1})],
    "Batak": [_m("batak", 8_466_969, {"Batak": 1}, "batak")],
    "Nias": [_m("nias", 1_041_925, {"Nias": 1})],
    # NC moved Jambi Malay into Malay; Jambi's whole remainder is 33,000 against 1.34M ethnic
    # Jambi there, so their language was coded Melayu, which L4.5 counts
    "Suku asal Jambi": [_m("kerinci", 303_550, {"Sumatera": 1}), _m(None, None, {})],
    # the residual is NC's Melayu Lahat (Pasemah, Lintang, Lematang, Kikim, Gumai, Kisam) and
    # Semendo, which NC moved into Malay and Glottolog files as Central (South Barisan) Malay
    "Suku asal Sumatera Selatan": [
        _m("musi", 1_252_258, {"Musi": 1}), _m("musi", 654_105, {"Musi": 1}),
        _m("musi", 192_705, {"Musi": 1}),
        _m("central_malay", 721_613, {"Melayu Tengah": 1}),
        _m("central_malay", 163_628, {"Melayu Tengah": 1}),
        _m("col", 163_262, {"Melayu Tengah": .5, "Sumatera": .5}),
        _m("komering", 370_119, {"Lampung": .5, "Sumatera": .5}),
        _m("unnamed", 144_986 + 121_289, {"Musi": .5, "Sumatera": .5}),   # Rambang, Daya
        _m("central_malay", None, {"Melayu Tengah": 1})],
    "Suku asal Lampung": [_m("lampung", 1_376_390, {"Lampung": 1})],
    # Bangka, Belitung and Akit speak Malay (Bangka Belitung's remainder is 24,000)
    "Suku asal Sumatera Lainnya": [
        _m("rejang", 454_673, {"Sumatera": 1}), _m("mentawai", 69_145, {"Sumatera": 1}),
        _m("pekal", 29_173, {"Sumatera": 1}), _m("kaur", 40_863, {"Sumatera": .5, "Melayu Tengah": .5}),
        _m(None, 683_193 + 201_068 + 27_769, {}),
        _m("unnamed", None, {"Sumatera": .5, "Melayu Tengah": .5})],
    "Betawi": [_m("betawi", 6_807_968, {"Betawi": 1}, "betawi")],
    "Cirebon": [_m("cirebonese", 1_877_514, {"Cirebon": 1})],
    "Bali": [_m("balinese", 3_946_416, {"Bali": 1}, "bali")],
    "Sasak": [_m("sasak", 3_173_127, {"Sasak": 1}, "sasak")],
    # Bima, Mbojo, Dompu and Kore are NC's four Bima groups
    "Suku Nusa Tenggara Barat lainnya": [
        _m("bima", 665_383 + 127_972 + 61_817 + 16_313, {"NTB": 1}),
        _m("sumbawa", 396_906, {"NTB": 1}), _m("unnamed", None, {"NTB": 1})],
    # NC's members sum to 58,835 more than BPS's group; scaled down, no residual. "Flores" names
    # an island, not a language. "Timor Leste origin" is read as Tetun (sources/id.md §5).
    "Suku asal Nusa Tenggara Timur": [
        _m("uab_meto", 933_093, NTT), _m("manggarai", 737_615, NTT), _m("sumba", 658_721, NTT),
        _m("lamaholot", 294_615, NTT), _m("ngada", 289_950, NTT), _m("tetun", 269_368, NTT),
        _m("unnamed", 260_069, NTT), _m("rote", 239_346, NTT), _m("alor", 196_529, NTT),
        _m("lio", 187_155, NTT), _m("hawu", 177_297, NTT), _m("unnamed", None, NTT)],
    "Dayak": [_m("dayak", 3_009_494, {"Dayak": 1}, "dayak")],
    "Suku Asal Kalimantan lainnya": [
        _m("kutai", 279_055, {"Kalimantan": 1}), _m("paser", 73_350, {"Kalimantan": 1}),
        _m("unnamed", None, {"Kalimantan": 1})],
    "Makassar": [_m("makassarese", 2_672_590, {"Makassar": 1})],
    "Minahasa": [_m("minahasan", 1_237_177, {"SULUT": .5, _MP: .5})],
    "Gorontalo": [_m("gorontalo", 1_251_494, SUL)],
    "Suku Asal Sulawesi lainnya": [
        _m("toraja", 857_250, {"Sulselbar": 1}), _m("mandar", 684_688, {"Sulselbar": 1}),
        _m("tae", 420_117, {"Sulselbar": 1}), _m("duri", 238_084, {"Sulselbar": 1}),
        _m("mamasa", 133_659 + 34_962, {"Sulselbar": 1}),      # Pattae' is Mamasa (patt1250)
        _m("selayar", 131_213, {"Sulselbar": 1}), _m("mamuju", 108_229, {"Sulselbar": 1}),
        _m("buton", 937_761, {"STT": 1}), _m("tolaki", 425_938, {"STT": 1}),
        _m("muna", 332_437, {"STT": 1}), _m("moronene", 40_025, {"STT": 1}),
        _m("banggai", 165_381, {"STT": 1}), _m("saluan", 97_134, {"STT": 1}),
        _m("sangir", 553_853, SUL), _m("mongondow", 304_292, SUL), _m("talaud", 97_314, SUL),
        _m("kaili", 770_088, SUL), _m("pamona", 186_163, SUL), _m("buol", 119_713, SUL),
        _m("tomini", 93_879, SUL), _m("lauje", 72_371, SUL),
        _m("bajau", 241_836, {"STT": .5, "SULUT": .5}),
        _m("unnamed", None, {"Sulselbar": .34, "STT": .33, "SULUT": .33})],
    # Ambon, Saparua, Haruku and Banda people mostly speak Ambonese (or Banda) Malay
    "Suku Asal Maluku": [
        _m("trade", 442_585 + 89_674 + 31_052 + 23_247, {_MP: 1}),
        _m("kei", 213_826, EAST), _m("seram", 194_818, EAST), _m("ternate", 133_110, EAST),
        _m("tobelo", 115_946, EAST), _m("tanimbar", 110_597, EAST), _m("galela", 102_456, EAST),
        _m("tidore", 87_524, EAST), _m("sula", 84_858, EAST), _m("buru", 57_521, EAST),
        _m("geser_gorom", 33_598, EAST), _m("aru", 30_942, EAST), _m("loloda", 28_132, EAST),
        _m("kisar", 27_963, EAST), _m("babar", 27_450, EAST), _m("tabaru", 23_704, EAST),
        _m("unnamed", 90_960 + 19_387, EAST),                 # Makian (two languages), Patani
        _m("unnamed", None, EAST)],
    "Suku Asal Papua": [
        _m("dani", 650_898, PAP), _m("ekari", 316_357, PAP), _m("biak", 204_415, PAP),
        _m("yali", 133_812, PAP), _m("asmat", 132_991, PAP), _m("yapen", 99_305, PAP),
        _m("nduga", 99_239, PAP), _m("arfak", 73_828, PAP), _m("moni", 63_309, PAP),
        _m("maybrat", 52_654, PAP), _m("ketengban", 42_025, PAP), _m("marind", 37_558, PAP),
        _m("sentani", 30_661, PAP), _m("ngalum", 29_186, PAP), _m("kamoro", 28_645, PAP),
        _m("hupla", 27_353, PAP), _m("waropen", 27_073, PAP), _m("baham", 24_521, PAP),
        _m("citak", 22_970, PAP), _m("damal", 22_479, PAP), _m("moi", 21_923, PAP),
        _m("yaqay", 21_121, PAP), _m("unnamed", None, PAP)],
}


def _ipf(seed, rows, cols, iters=2000, tol=1e-9):
    """Rake `seed` to row sums `rows` and column sums `cols` (cols scaled to sum(rows)); ends on
    the rows. Columns with no seed anywhere stay zero."""
    x = seed.astype(float).copy()
    cols = cols * rows.sum() / cols.sum()
    for _ in range(iters):
        rs = x.sum(1)
        x *= np.divide(rows, rs, out=np.zeros_like(rows, dtype=float), where=rs > 0)[:, None]
        cs = x.sum(0)
        x *= np.divide(cols, cs, out=np.zeros_like(cols, dtype=float), where=cs > 0)[None, :]
        if np.abs(x.sum(1) - rows).max() < tol * max(1.0, rows.max()):
            break
    rs = x.sum(1)
    return x * np.divide(rows, rs, out=np.zeros_like(rows, dtype=float), where=rs > 0)[:, None]


def glottolog():
    return pd.read_csv(GLOTTOLOG, usecols=["ID", "Name", "Level", "Latitude", "Longitude",
                                           "Countries", "Family_ID"]).set_index("ID")


def dayak_points(g=None):
    """Every language Glottolog places in Indonesian Borneo that is not Malay, Banjar, Kutai,
    Bajau, Tidung, Paser or a migrant language: the Dayak languages, for placement only."""
    g = glottolog() if g is None else g
    m = g[(g["Level"] == "language") & g["Countries"].fillna("").str.contains("ID")
          & g["Longitude"].between(108.6, 119.2) & g["Latitude"].between(-4.2, 4.4)
          & (g["Family_ID"] == "aust1307")]
    bad = r"Malay|Banjar|Kutai|Bajau|Tidung|Paser|Pasir|Bugis|Buginese|Javanese|Berau|Sama"
    m = m[~m["Name"].str.contains(bad)]
    return list(m.index)


def codes(key, g=None):
    c = LANG[key][1]
    return dayak_points(g) if c is None else c


def nearness(keys, prov_codes, kernel_km):
    """Province x key: the population-weighted mean over a province's units of
    exp(-distance to the key's nearest Glottolog point / kernel). NaN for keys with no point."""
    u = pd.read_csv(UNITS, dtype={"unit": str, "prov": str})
    g = glottolog()
    out = pd.DataFrame(np.nan, index=prov_codes, columns=keys)
    pidx = u["prov"].map({p: i for i, p in enumerate(prov_codes)}).to_numpy()
    pop = u["pop"].to_numpy(dtype=float)
    ppop = np.bincount(pidx, weights=pop, minlength=len(prov_codes))
    for k in keys:
        cs = codes(k, g)
        if not cs:
            continue
        pts = g.loc[cs, ["Latitude", "Longitude"]].astype(float).to_numpy()
        best = np.full(len(u), np.inf)
        for lat, lon in pts:
            d = np.hypot((u["lon"].to_numpy() - lon) * 111.32 * np.cos(np.radians(lat)),
                         (u["lat"].to_numpy() - lat) * 110.57)
            best = np.minimum(best, d)
        out[k] = np.bincount(pidx, weights=pop * np.exp(-best / kernel_km),
                             minlength=len(prov_codes)) / np.where(ppop > 0, ppop, 1)
    return out


def largest_remainder(total, w):
    w = np.asarray(w, dtype=float)
    raw = w / w.sum() * total
    base = np.floor(raw).astype(np.int64)
    short = int(total - base.sum())
    base[np.argsort(-(raw - base), kind="mergesort")[:short]] += 1
    assert base.sum() == total and (base >= 0).all()
    return base


def share_out(E, rem, pop, l41, prov_codes):
    """E: province x 31 groups (L2.6, index = province names); rem: province remainder;
    pop: province population aged 5+ (L4.2 Total); l41: L4.1 {label: count}; prov_codes:
    province name -> BPS code. Returns (long DataFrame province, lang, count; info dict)."""
    provs = list(E.index)
    codes_ = [prov_codes[p] for p in provs]
    P = len(provs)

    # ---- stage 0: L2.6 groups -> NC members per province
    keys = sorted({m[0] for ms in GROUPS.values() for m in ms
                   if m[0] not in (None, "unnamed", "trade", "batak")})
    near = nearness(keys, codes_, KERNEL0_KM)
    members = []          # (group, lang, cols, ret, vector over provinces)
    for grp, ms in GROUPS.items():
        tot = E[grp].to_numpy(dtype=float)
        named = sum(m[1] for m in ms if m[1] is not None)
        resid = max(0.0, tot.sum() - named)
        ns = np.array([m[1] if m[1] is not None else resid for m in ms], dtype=float)
        if ns.sum() <= 0:
            raise SystemExit(f"id_shareout: {grp} has no members")
        seed = np.full((P, len(ms)), 1.0 / P)
        for j, m in enumerate(ms):
            if m[0] in near.columns and not near[m[0]].isna().all():
                k = near[m[0]].to_numpy()
                seed[:, j] = (1 - LAMBDA0) / P + LAMBDA0 * k / k.sum()
        if len(ms) == 1:
            F = tot[:, None]
        else:
            keep = ns > 0
            F = np.zeros((P, len(ms)))
            F[:, keep] = _ipf(seed[:, keep], tot, ns[keep])
        assert np.allclose(F.sum(1), tot), grp
        for j, m in enumerate(ms):
            members.append((grp, m[0], m[2], RET[m[3]], F[:, j]))

    # ---- stage 1: province x L4.1 column
    cols = list(COLS)
    target = np.array([sum(l41[x] for x in COLS[c]) for c in cols], dtype=float)
    assert target.sum() == rem.sum(), (target.sum(), rem.sum())
    S = np.zeros((P, len(cols)))
    for grp, lang, cs, ret, f in members:
        for c, sh in cs.items():
            S[:, cols.index(c)] += f * ret * sh
    S[:, cols.index("Isyarat")] = pop.reindex(provs).to_numpy(dtype=float) * (
        target[cols.index("Isyarat")] / pop.sum())
    rows = rem.reindex(provs).to_numpy(dtype=float)
    if ((S.sum(1) == 0) & (rows > 0)).any():
        raise SystemExit("id_shareout: a province with a remainder has no seed")
    X = _ipf(S, rows, target)
    col_err = np.abs(X.sum(0) - target).max()
    assert np.abs(X.sum(1) - rows).max() < 1e-6 and col_err < 1, col_err

    # ---- stage 2: within each province x column, languages by stage 0 x retention
    import id_papua
    PE, _ = id_papua.build()
    _add_papua_langs()
    pmix = id_papua.province_mix(PE)
    langs = [k for k in LANG]
    Y = np.zeros((P, len(langs)))
    li = {k: i for i, k in enumerate(langs)}
    for c in cols:
        ci = cols.index(c)
        if c == "Isyarat":
            Y[:, li["sign"]] += X[:, ci]
            continue
        if c == "Batak":
            tot = sum(BATAK_SPLIT.values())
            for k, v in BATAK_SPLIT.items():
                Y[:, li[k]] += X[:, ci] * v / tot
            continue
        if c == "Papua":
            # Indonesian New Guinea's languages by their indigenous people (id_papua.py): in
            # Papua and Papua Barat the province's own mix, elsewhere the two together
            for i, code in enumerate(codes_):
                w = pmix.get(code, pmix["other"])
                w = w[w > 0] / w[w > 0].sum()
                for gc, v in w.items():
                    Y[i, li[f"pap_{gc}"]] += X[i, ci] * v
            continue
        if c == "MP":
            for i, code in enumerate(codes_):
                Y[i, li[MP_NODE.get(code, "trade_malay")]] += X[i, ci]
            continue
        W = {}
        for grp, lang, cs, ret, f in members:
            if c in cs:
                W[lang] = W.get(lang, 0) + f * ret * cs[c]
        Wm = np.column_stack(list(W.values()))
        ws = Wm.sum(1)
        if ((ws == 0) & (X[:, ci] > 1e-9)).any():
            raise SystemExit(f"id_shareout: {c} has people but no member seed somewhere")
        share = np.divide(Wm, ws[:, None], out=np.zeros_like(Wm), where=ws[:, None] > 0)
        for j, lang in enumerate(W):
            Y[:, li[lang]] += X[:, ci] * share[:, j]
    assert np.allclose(Y.sum(1), rows)

    out = []
    for i, p in enumerate(provs):
        n = largest_remainder(int(rows[i]), Y[i])
        for j, k in enumerate(langs):
            if n[j] > 0:
                out.append(dict(province=p, lang=k, label=LANG[k][0], count=int(n[j])))
    df = pd.DataFrame(out)
    assert (df.groupby("province")["count"].sum().reindex(provs).to_numpy() == rows).all()
    info = dict(X=pd.DataFrame(X, index=provs, columns=cols), target=dict(zip(cols, target)),
                col_err=col_err)
    return df, info
