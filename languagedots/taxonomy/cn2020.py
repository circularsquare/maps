"""China 2020 census nationality (minzu) -> language nodes, with each group's retention share.

China's census asks nationality, not language. Under AGENT_BRIEF §2's ethnicity rule each
nationality is read as its language, but only for the share of it that speaks that language;
the rest is drawn on Chinese (`sinotibetan.sinitic`, "Chinese, language not named"), as Han are.

THE RETENTION SOURCE is one series, so every group is measured the same way: the National
Language Resource Monitoring and Research Center, Minority Languages (国家语言资源监测与研究民族
语言中心, Minzu University of China), "少数民族语言文字使用、发展和保护情况", one page per
nationality, 2024 (https://nmlr.muc.edu.cn/info/1119/2132.htm to 2682.htm). Each page gives:
  * SITE: "在调研点使用本民族语言的人数占...总人口平均比例" - the share speaking the language,
    averaged over the case-study survey sites. Sites are in the group's own areas, so this runs
    high for a group that is dispersed or shifting (She 89.2% at the sites, while the same page
    says about 1,000 people speak She at all).
  * often a NATIONAL speaker count ("约有N人使用X语", sometimes dated, sometimes from the
    Chinese Language Resources Protection Project, 语保工程), and the 2010 census population.
THE RULE: retention = the lower of SITE and (national count / 2010 population). A count below
the site share is the better national figure (the sites are heartlands); a count above it is
usually stale (counted when the group was smaller) or includes speakers of other nationalities.
Where the page gives no national count, SITE alone. Hui and Manchu, which the pages say use
Chinese ("回族主要转用汉语", "满族通用汉语汉文"), are 0. Every share and its numbers are in
RETENTION below and tabled in sources/cn.md.

GROUPS COVERING SEVERAL LANGUAGES go on a group node, unnamed, unless the page splits them: Yi on
Loloish, Miao on Hmongic, Tibetan on Tibetic, Dai on Kra-Dai (Tai Lue, Tai Nua, Tai Dam...),
Tajik on Iranian (Sarikoli and Wakhi), Jingpo on mm.txt's Kachin group (Jingpho and the
Burmish Zaiwa, Lashi, Lhao Vo, all inside it), Nu on Sino-Tibetan itself (Anong, Nusu, Zauzou),
Lhoba on Tani,
Yugur on `other` (Western Yugur is Turkic, Eastern Yugur Mongolic: no node holds both).
"""

CHINESE = "sinotibetan.sinitic"

# nationality column (chinaethnicity's key) -> its language node
NAMES = {
    "han": CHINESE,
    "mongol": "mongolic.mongolian",
    "hui": CHINESE,
    "tibetan": "sinotibetan.tibetic",
    "uyghur": "turkic.uyghur",
    "miao": "hmongmien.hmongic",
    "yi": "sinotibetan.loloish",
    "zhuang": "kradai.zhuang",
    "bouyei": "kradai.bouyei",
    "korean": "koreanic.korean",
    "manchu": CHINESE,
    "dong": "kradai.kam",
    "yao": "hmongmien",
    "bai": "sinotibetan.bai",
    "tujia": "sinotibetan.tujia",
    "hani": "sinotibetan.loloish.hani",
    "kazakh": "turkic.kazakh",
    "dai": "kradai",
    "li": "kradai.hlai",
    "lisu": "sinotibetan.loloish.lisu",
    "wa": "austroasiatic.wa",
    "she": CHINESE,
    "gaoshan": "austronesian.taiwan",
    "lahu": "sinotibetan.loloish.lahu",
    "sui": "kradai.sui",
    "dongxiang": "mongolic.santa",
    "naxi": "sinotibetan.naxi",
    # mm.txt's Kachin group holds Jingpho, Zaiwa, Lashi (lachik) and Lhaovo; was the Sino-Tibetan
    # root until 2026-10-06 (group-node audit, 5d7dac7e-aud)
    "jingpo": "sinotibetan.kachin",
    "kyrgyz": "turkic.kyrgyz",
    "tu": "mongolic.monguor",
    "daur": "mongolic.daur",
    "mulao": "kradai.mulam",
    "qiang": "sinotibetan.qiangic.qiang",
    "blang": "austroasiatic.blang",
    "salar": "turkic.salar",
    "maonan": "kradai.maonan",
    "gelao": "kradai.gelao",
    "xibe": "tungusic.xibe",
    "achang": "sinotibetan.burmish.achang",
    "pumi": "sinotibetan.qiangic.pumi",
    "tajik": "indoeuropean.iranian",
    "nu": "sinotibetan",
    "uzbek": "turkic.uzbek",
    "russian": "indoeuropean.slavic.east.russian",
    "evenki": "tungusic.evenki",
    "deang": "austroasiatic.palaung",
    "bonan": "mongolic.bonan",
    "yugur": "other",
    "gin": "austroasiatic.vietnamese",
    "tatar": "turkic.tatar",
    "derung": "sinotibetan.derung",
    "oroqen": "tungusic.oroqen",
    "hezhen": "tungusic.nanai",
    "monpa": "sinotibetan.eastbodish.monpa",
    "lhoba": "sinotibetan.tani",
    "jino": "sinotibetan.loloish.jino",
    # 未识别民族: peoples the state has not classified (Chuanqing, Gejia, Mang, Kemei, Deng...).
    # Their languages run from Chinese to Hmongic to Austroasiatic; nothing says which, so
    # `other`, language not named.
    "undetermined": "other",
    # 入籍: naturalised foreign citizens, any language
    "naturalised": "other",
}

# Nodes the splits below can return that are not NAMES values.
EXTRA_NODES = ["hmongmien.hmongic.bunu", "hmongmien.hmongic.she", "sinotibetan.qiangic",
               "sinotibetan.tshangla", "austronesian.tsat"]

# nationality -> (share speaking its language, basis). Everything not listed is 1.0 (Han,
# undetermined, naturalised, and Gaoshan, whose page has no survey: "无近10年个案调查数据").
# `site` = the page's survey-site average; `count / pop2010` = its national speaker count over
# the 2010 census population it quotes. The lower is taken (see the docstring).
RETENTION = {
    "mongol": (0.4581, "count 2,740,000 / pop2010 5,981,840 (site 85.25%)"),
    "hui": (0.0, "page: Hui have shifted to Chinese (site 42.7% is not national); Hainan below"),
    "tibetan": (0.795, "site 79.5% speak Tibetan; Sichuan's Qiangic speakers below"),
    "uyghur": (0.9705, "site 97.05%"),
    "miao": (0.7355, "site 73.55% (count 7,110,000 / 9,426,007 = 75.4%)"),
    "yi": (0.913, "site 91.3%"),
    "zhuang": (0.767, "site 76.7%"),
    "bouyei": (0.452, "site 45.2% (count 2,000,000 / 2,870,034 = 69.7%)"),
    "korean": (0.8415, "site 84.15% (count 1,920,000 exceeds pop2010 1,830,929)"),
    "manchu": (0.0, "page: Manchu use Chinese; about 500 speakers in 1982"),
    "dong": (0.4167, "count 1,200,000 / pop2010 2,879,974 (site 70.45%)"),
    "yao": (0.958, "site 95.8%, of which Bunu 268,000 / 2,796,003 = 9.59% (below)"),
    "bai": (0.6724, "count 1,300,000 / pop2010 1,933,510 (site 82.9%)"),
    "tujia": (0.0203, "count 170,000 (1980s) / pop2010 8,353,912 (site 19.9%)"),
    "hani": (0.887, "site 88.7%"),
    "kazakh": (0.927, "site 92.7%"),
    "dai": (0.947, "site 94.7%"),
    "li": (0.6835, "count 1,000,000 / pop2010 1,463,064 (site 95.9%)"),
    "lisu": (0.9575, "site 95.75%"),
    "wa": (0.8378, "count 360,000 / pop2010 429,709 (site 86%)"),
    "she": (0.0, "count about 1,000, all in Guangdong / pop2010 708,651 (site 89.2%); Guangdong below"),
    "lahu": (0.8231, "count 400,000 / pop2010 485,966 (site 94.6%)"),
    "sui": (1.0, "site 100%"),
    "dongxiang": (0.4023, "count 250,000 / pop2010 621,500 (site 71.9%)"),
    "naxi": (0.9975, "site 99.75%"),
    "jingpo": (0.975, "site 97.5%"),
    "kyrgyz": (1.0, "site 100%"),
    "tu": (0.648, "site 64.8%"),
    "daur": (0.629, "site 62.9% (count 115,000 / 131,992 = 87.1%)"),
    "mulao": (0.816, "site 81.6%"),
    "qiang": (0.1938, "count 60,000 / pop2010 309,576 (site 72.5%; page: 306,000 use Chinese)"),
    "blang": (0.994, "site 99.4%"),
    "salar": (0.8275, "site 82.75%"),
    "maonan": (0.941, "site 94.1%"),
    "gelao": (0.0116, "dialect counts 2,000+1,500+1,700+1,200 = 6,400 / pop2010 550,746 (site 59.85%)"),
    "xibe": (0.2599, "counts 2,700+35,000+3,000+8,800 = 49,500 / pop2010 190,481 (site 82.5%)"),
    "achang": (0.76, "site 76%"),
    "pumi": (0.6765, "site 67.65% (count 33,600 in 2000 / 42,861 = 78.4%)"),
    "tajik": (0.722, "site 72.2%"),
    "nu": (0.801, "site 80.1%"),
    "uzbek": (0.171, "site 17.1%"),
    "russian": (0.4815, "site 48.15% (count 9,000 / 15,393 = 58.5%)"),
    "evenki": (0.567, "site 56.7%"),
    "deang": (0.968, "site 96.8%"),
    "bonan": (0.4982, "count 10,000 / pop2010 20,074 (site 97.5%)"),
    "yugur": (0.705, "site 70.5%"),
    "gin": (0.214, "site 21.4% (count 10,000 in 1990 / 28,199 = 35.5%)"),
    "tatar": (0.13, "site 13%"),
    "derung": (0.7575, "site 75.75% (count 8,000 exceeds pop2010 6,930)"),
    "oroqen": (0.3686, "site 36.86%"),
    "hezhen": (0.0411, "count 220 (1982) / pop2010 5,354 (site 4.7%)"),
    "monpa": (0.1231, "Monpa count 1,300 / pop2010 10,561; Tshangla below"),
    "lhoba": (1.0, "site 100%"),
    "jino": (0.8945, "site 89.45%"),
}

# Further languages of a nationality the page counts on its own, as (node, share), nationally or
# in one province only (prov code). Each comes out of the Chinese remainder, never out of the
# main language.
SPLITS = {
    # Yao: Bunu, Hmongic, 268,000 speakers / 2,796,003; taken out of the 95.8% above
    ("yao", None): [("hmongmien.hmongic.bunu", 0.0959)],
    # Monpa: Tshangla (仓洛语) 7,000 / 10,561
    ("monpa", None): [("sinotibetan.tshangla", 0.6628)],
    # She: about 1,000 speakers, all in Guangdong, / Guangdong's 2020 She 42,080 (estimated)
    ("she", 44): [("hmongmien.hmongic.she", 0.0238)],
    # Hui: Tsat (回辉话), 6,000 speakers in Hainan (语保工程) / Hainan's 2020 Hui 17,089
    ("hui", 46): [("austronesian.tsat", 0.3511)],
    # Tibetan, Sichuan: Baima 10,000, Ergong 45,000, Ersu 20,000, Guiqiong 6,000, Gyalrong
    # 100,000, Lavrung 10,000, Minyak 10,000, Namuyi 5,000, Queyu 7,000, Shixing 1,800, Zhaba
    # 20,000 = 234,800 (the page's 1995-2008 counts) / Sichuan's 2020 Tibetans 1,604,629
    ("tibetan", 51): [("sinotibetan.qiangic", 0.1463)],
}

SPLIT_MAIN = {  # in these provinces the main language's share is scaled by (1 - the split)
    ("tibetan", 51): True,
}


# ---- Chinese dialect groups (Anita, 2026-10-05) ----
# Everyone the shares above put on CHINESE (Han, and the Hui, Manchu, She... who speak Chinese) is
# drawn on the county's main Sinitic group from the Language Atlas of China, via
# data/normalized/cn_dialect.csv (sources/cn_dialect.py). Approved proxy: each county is drawn
# whole as its main group, so minorities of other dialects inside a county vanish.
# The level: Mandarin whole (Anita, 2026-10-05: "we probably shouldn't split all the different
# Mandarins"; its eight groups, 区, all go on `sinitic.mandarin`), Min by its subgroups (the
# table's 片 under 闽 are the atlas's 闽南区, 闽东区...), the other groups whole. The table's finer
# 片 are not drawn.
SI = "sinotibetan.sinitic"
MANDARIN = ("Northeastern", "Beijing", "Jilu", "Jiaoliao", "Zhongyuan", "Lanyin", "Jianghuai",
            "Southwestern")
DIALECT = {   # the table's DiaGroup (and for Min and Yue its SubDiaGroup) -> node
    **{g: f"{SI}.mandarin" for g in MANDARIN},
    "Jin": f"{SI}.jin",
    "Wu": f"{SI}.wu",
    "Hui": f"{SI}.hui",
    "Gan": f"{SI}.gan",
    "Xiang": f"{SI}.xiang",
    "Hakka": f"{SI}.hakka",
    "Pinghua and Tuhua": f"{SI}.pinghua",
    # Min: one leaf per subgroup; Min Nan in the three Chaoshan prefectures is Teochew (below)
    ("Min", "Minnan"): f"{SI}.min_nan",
    ("Min", "Mindong"): f"{SI}.min_dong",
    ("Min", "Minbei"): f"{SI}.min_bei",
    ("Min", "Minzhong"): f"{SI}.min_zhong",
    ("Min", "Puxian"): f"{SI}.puxian",
    ("Min", "Shaojiang"): f"{SI}.shaojiang",
    ("Min", "Leizhou"): f"{SI}.leizhou",
    ("Min", "Qiongwen"): f"{SI}.hainanese",
    # Yue: 四邑 on Sze Yap, the node Hong Kong already draws Taishanese on; every other Yue
    # subgroup (广府, 高阳, 勾漏, 邕浔, 钦廉, 吴化, and 儋州话, which the table files under Yue) on
    # Cantonese, the name Yue as a whole goes by in English. Splitting Goulou or Gao-Yang off
    # would be the next step if Cantonese reads too broad.
    ("Yue", "Siyi"): f"{SI}.siyi",
    "Yue": f"{SI}.cantonese",
}
# Min Nan in Shantou (4405), Chaozhou (4451) and Jieyang (4452): the atlas's 潮汕片, Teochew,
# which Hong Kong and Singapore draw as its own node. Shanwei's Hailufeng Min stays on Min Nan.
TEOCHEW_PREF = {"4405", "4451", "4452"}
EXTRA_NODES += sorted({v for v in DIALECT.values()} | {f"{SI}.teochew"})


def dialect_node(unit, group, subgroup):
    if group == "Min" and subgroup == "Minnan" and unit[:4] in TEOCHEW_PREF:
        return f"{SI}.teochew"
    node = DIALECT.get((group, subgroup)) or DIALECT.get(group)
    if node is None:
        raise KeyError(f"no node for dialect {group}/{subgroup}")
    return node


def resolve(category):
    return NAMES.get(category)


def shares(category, prov):
    """[(node, share)] for one nationality in one province; shares sum to 1."""
    main = NAMES[category]
    r = RETENTION.get(category, (1.0, ""))[0]
    extra = SPLITS.get((category, prov), []) + SPLITS.get((category, None), [])
    if (category, prov) in SPLIT_MAIN:
        r = r * (1 - sum(s for _, s in extra))
    elif category == "yao":
        r = r - sum(s for _, s in extra)
    out = [(main, r)] + list(extra)
    rest = 1 - sum(s for _, s in out)
    if rest < -1e-9:
        raise ValueError(f"{category} in {prov}: shares add to {1 - rest:.4f}")
    if rest > 1e-9:
        out.append((CHINESE, rest))
    return [(n, s) for n, s in out if s > 0]
