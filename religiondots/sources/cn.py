"""China — 2000 census ethnicity by county, rescaled onto 2010 provincial totals.

Reads (or fetches) data/raw/cn/ and writes data/normalized/cn.csv.

**CHINA HAS NEVER ASKED ABOUT RELIGION IN A CENSUS, AND THIS FILE DOES NOT PRETEND
OTHERWISE.** What is counted here is NATIONALITY. spec §14.5 permits deriving a religion
from an ethnic category where that category was itself constituted religiously, at no finer
geography than the ethnicity is published at, and never where the group is religiously
mixed; `taxonomy/cn2000.py` is the argument about which of the 56 nationalities clear that
bar. Three groups do -- ten Muslim nationalities, four Tibetan Buddhist, and the Dai -- and
the drawn population is about 27M of 1.24 billion. Every row is tier `derived`.

THE SOURCE IS BETTER THAN sources.md §12 EXPECTED, AND IT IS FREE.
Harvard's `chinacensus` dataverse holds the *2000 Population Census Data Assembly* as 31
datasets, one per province, CC0, no auth, no bot wall. Table A0106 in each is "Population by
sex, nationality and by township" -- 179 columns, all 56 nationalities plus Han, unidentified
and naturalised, three columns each (total/male/female), with COUNTY SUBTOTAL ROWS already in
it (`V2 == 'Total'`). So county level needs no aggregation and the townships underneath are
free. 2,859 counties, 1,239,452,849 people.

**Township is deliberately NOT used.** The source reaches it; Anita's call of 2026-09-05 is
county, and spec §14.5's "no finer than the state publishes" is a ceiling rather than a
target. A township-level map of Uyghur settlement is a different object from a county one.

THE VINTAGE PROBLEM, AND WHY THE TOTALS ARE 2010 AND NOT 2020.
The structure is 2000 and there is no newer county-level ethnic table in the open. spec §3.4
says to take structure from the detailed source and totals from the recent one, so each
group's county figures are scaled to its 2010 provincial total. **2020 exists and is
unusable**: the NBS publishes the 2020 yearbook's table 1-4 as a 3 MB JPEG scan, while the
2010 edition's table 1-6 is HTML. Anita's call, 2026-09-05, after the observation that the
choice moves magnitude and not geography -- the shape is 2000's either way, and 2010 -> 2020
is close to a uniform per-group rescale (Uyghurs 10.07M -> 11.77M). Trusting OCR for a
country's magnitudes to buy a ~15% size correction was not worth it.

WHAT THE PROVINCE SUMS PROVE, AND THE ONE HOLE.
The 31 files sum to 1,239,452,849 against the published 2000 provincial sum of
1,242,612,226. The whole 3,159,377 difference is **Hainan**, whose file carries 14 of its 24
county-level units as name-only rows with no data at all. Every other province is exact to
the person, which is a stronger check than any tolerance would have been. Hainan's missing
units are Li and Miao autonomous counties plus the disputed island groups, and hold
essentially none of the drawn population -- Sanya, which holds the Utsul Muslims, is present.

Usage:
    python sources/cn.py --fetch    # 31 Dataverse tables + one NBS page, ~90 MB
    python sources/cn.py            # normalise from data/raw/cn/
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "cn")
OUT = os.path.join(ROOT, "data", "normalized", "cn.csv")

SOURCE_ID = "cn_census_2000_x_2010"
STRUCTURE_YEAR = 2000
TOTAL_YEAR = 2010
BASIS = "ethnicity_derived"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

DATAVERSE = "https://dataverse.harvard.edu/api"
NBS_2010 = "https://www.stats.gov.cn/sj/pcsj/rkpc/6rp/html/A0106a.htm"
NBS_2010_FILE = "nbs2010_A0106a.htm"

# Published 2000 census figures, for the checks at the end. Not tolerances -- the province
# sums hit these exactly, and anything else means the parse moved.
CENSUS_2000_PROVINCES = 1_242_612_226
HAINAN_2000 = 7_559_035
EXPECTED_COUNTIES = 2_859

# --- the 59 columns, in GB/T 3304 census order ------------------------------------
# Verified rather than assumed: Ningxia comes out 33.9% Hui, Xinjiang 45.2% Uyghur and
# 40.6% Han, Xizang 92.8% Tibetan, Guangxi 32.4% Zhuang, Han 91.6% nationally -- all of
# which are the published figures. A wrong order would put those on the wrong groups and
# be obvious immediately, which is why this check is worth more than a column header.
GROUPS = [
    "Total", "Han", "Mongol", "Hui", "Tibetan", "Uyghur", "Miao", "Yi", "Zhuang",
    "Bouyei", "Korean", "Manchu", "Dong", "Yao", "Bai", "Tujia", "Hani", "Kazakh",
    "Dai", "Li", "Lisu", "Va", "She", "Gaoshan", "Lahu", "Sui", "Dongxiang", "Naxi",
    "Jingpo", "Kyrgyz", "Tu", "Daur", "Mulao", "Qiang", "Blang", "Salar", "Maonan",
    "Gelao", "Xibe", "Achang", "Pumi", "Tajik", "Nu", "Uzbek", "Russian", "Evenk",
    "De'ang", "Bonan", "Yugur", "Gin", "Tatar", "Derung", "Oroqen", "Hezhen", "Monba",
    "Lhoba", "Jino", "Unidentified", "Naturalised",
]
N_GROUPS = len(GROUPS)
N_COLS = 2 + N_GROUPS * 3
assert N_GROUPS == 59 and N_COLS == 179

GB_PROVINCE = {
    "11": (110000, "Beijing"), "12": (120000, "Tianjin"), "13": (130000, "Hebei"),
    "14": (140000, "Shanxi"), "15": (150000, "Inner Mongolia"),
    "21": (210000, "Liaoning"), "22": (220000, "Jilin"), "23": (230000, "Heilongjiang"),
    "31": (310000, "Shanghai"), "32": (320000, "Jiangsu"), "33": (330000, "Zhejiang"),
    "34": (340000, "Anhui"), "35": (350000, "Fujian"), "36": (360000, "Jiangxi"),
    "37": (370000, "Shandong"), "41": (410000, "Henan"), "42": (420000, "Hubei"),
    "43": (430000, "Hunan"), "44": (440000, "Guangdong"), "45": (450000, "Guangxi"),
    "46": (460000, "Hainan"), "50": (500000, "Chongqing"), "51": (510000, "Sichuan"),
    "52": (520000, "Guizhou"), "53": (530000, "Yunnan"), "54": (540000, "Xizang"),
    "61": (610000, "Shaanxi"), "62": (620000, "Gansu"), "63": (630000, "Qinghai"),
    "64": (640000, "Ningxia"), "65": (650000, "Xinjiang"),
}


# ==================================================================================
# fetching
# ==================================================================================

def _curl(url, dest):
    """Fetch with curl rather than urllib.

    urllib sees a self-signed root on this machine and raises
    CERTIFICATE_VERIFY_FAILED where curl succeeds. Per sources.md §9h that is TLS
    interception locally, not a server condition, and turning verification off is the
    reflex that section warns against -- so the fetch goes through the client that has a
    working trust store instead.
    """
    r = subprocess.run(["curl", "-sSL", "-m", "180", url, "-o", dest,
                        "-w", "%{http_code}"], capture_output=True, text=True)
    return r.stdout.strip()


def fetch():
    os.makedirs(RAW, exist_ok=True)

    print("listing the Harvard `chinacensus` dataverse ...")
    with urllib.request.urlopen(f"{DATAVERSE}/dataverses/chinacensus/contents",
                                timeout=90) as r:
        contents = json.load(r)["data"]
    dois = [f"{c['protocol']}:{c['authority']}/{c['identifier']}" for c in contents]
    if len(dois) != 31:
        print(f"  !! expected 31 provincial datasets, found {len(dois)}")

    for i, doi in enumerate(dois, 1):
        url = f"{DATAVERSE}/datasets/:persistentId/?persistentId={doi}"
        with urllib.request.urlopen(url, timeout=120) as r:
            meta = json.load(r)["data"]["latestVersion"]
        hit = next((f["dataFile"] for f in meta["files"]
                    if re.fullmatch(r"J\d{2}A0106\.tab", f["dataFile"]["filename"])), None)
        if hit is None:
            print(f"  [{i:2d}] {doi}: no A0106 table")
            continue
        dest = os.path.join(RAW, hit["filename"])
        if os.path.exists(dest) and os.path.getsize(dest) == hit["filesize"]:
            print(f"  [{i:2d}] have {hit['filename']}")
            continue
        code = _curl(f"{DATAVERSE}/access/datafile/{hit['id']}", dest)
        print(f"  [{i:2d}] {code} {hit['filename']}  {hit['filesize']:,}")
        time.sleep(0.3)

    dest = os.path.join(RAW, NBS_2010_FILE)
    print(f"NBS 2010 table 1-6 -> {NBS_2010_FILE}: {_curl(NBS_2010, dest)}")
    print("\nBoundaries and the county index come from sources/cn_geo.py --fetch.")


# ==================================================================================
# parsing
# ==================================================================================

def _int(x):
    x = x.strip().replace(",", "")
    return int(x) if x.isdigit() else 0


def read_2000():
    """{province_code: [(county_name, [59 totals]), ...]} from the A0106 tables."""
    out = {}
    for fn in sorted(os.listdir(RAW)):
        m = re.fullmatch(r"J(\d{2})A0106\.tab", fn)
        if not m:
            continue
        code, _ = GB_PROVINCE[m.group(1)]
        rows = []
        with open(os.path.join(RAW, fn), encoding="utf-8", errors="replace") as fh:
            fh.readline()
            for line in fh:
                p = line.rstrip("\n").split("\t")
                # Zhejiang has one township row wrapped across two lines, which shows up
                # as a 2-column and a 178-column row. County subtotals are unaffected;
                # requiring the full width drops the debris without a special case.
                if len(p) < N_COLS or p[1].strip() != "Total":
                    continue
                rows.append((p[0].strip(), [_int(p[2 + 3 * i]) for i in range(N_GROUPS)]))
        out[code] = rows
    return out


def read_2010():
    """{province_code: [59 totals]} from the NBS 2010 table 1-6.

    The NBS page is a 178-column HTML table (one name column, then the same 59 groups x
    three) whose first data row is the national one. Provinces are matched by ORDER
    against GB_PROVINCE rather than by their Chinese names, and the national row is used
    as the check -- if the order were wrong the national column would not reproduce the
    published 2010 figures, which it does exactly.
    """
    path = os.path.join(RAW, NBS_2010_FILE)
    raw = open(path, "rb").read()
    html = raw.decode("gb18030", errors="replace")

    parsed = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S | re.I):
        cells = []
        for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S | re.I):
            t = re.sub(r"<[^>]+>", "", c).replace("&nbsp;", " ").replace("　", " ")
            cells.append(t.strip())
        if len(cells) == N_COLS - 1:
            vals = [_int(cells[1 + 3 * i]) for i in range(N_GROUPS)]
            if vals[0]:
                parsed.append((cells[0], vals))

    if not parsed:
        raise SystemExit(f"!! no data rows parsed from {path}")
    national = parsed[0][1]
    provinces = parsed[1:]
    order = [GB_PROVINCE[k][0] for k in sorted(GB_PROVINCE)]
    if len(provinces) != len(order):
        raise SystemExit(f"!! 2010 table has {len(provinces)} province rows, "
                         f"expected {len(order)}")
    return national, dict(zip(order, [v for _, v in provinces]))


# ==================================================================================
# county -> adcode
# ==================================================================================
#
# The census carries romanised names and no codes; DataV carries Chinese names and the
# GB/T 2260 adcode. Both lists are in code order within a province, which is what makes
# the residue small enough to hand-check: a mismatch is nearly always a rename, and the
# neighbours say which one.
#
# geoBoundaries was tried first and is UNUSABLE FOR CHINA -- CHN ADM2 has duplicated
# polygons, counties abolished in the 1980s (Xizang: 78 polygons for 73 counties), units
# filed under the wrong province, and systematically corrupted romanisation
# (`Erminxian` for Emin, `Duinongdeqingxian` for Duilongdeqing). It matched 59.9%.
# See sources/cn_geo.py.

# Transliteration rules the source follows and pinyin does not. Each fixes a class rather
# than a case, which is why they are here and not in OVERRIDES.
TRANSLIT_ANY = [("cangzu", "zangzu")]   # 藏族 — Tianzhu, Muli, the Tibetan autonomous counties
TRANSLIT_END = [("shen", "shi")]        # 什 — Kashgar (Kashi) and the rest of the -shen family

SUFFIX = re.compile(
    r"(zizhixianxian|zizhiqixian|zizhixian|zizhiqi|zizhizhou|zizhiqu|"
    r"linqu|kuangqu|tequ|xianxian|xian|shi|qu|qi|meng)$")

# Census county -> adcode, where no rule can get there. Every entry is a real
# administrative change between 2000 and now, and the great majority are one shape: a
# prefecture was upgraded to a city and its seat county became a district under a new
# name. Ordered by how much drawn population each carries.
OVERRIDES = {
    # ==============================================================================
    # THE 168 THAT WERE SILENTLY DROPPED UNTIL 2026-09-08 — spec §14.17
    # ==============================================================================
    # **These 168 counties are 67,805,092 people, 5.47% of China, and until this block
    # existed they were read out of the census volumes and then thrown away**, because the
    # romanised name matched no adcode. They are not a boundary problem: the geometry has
    # 2,848 polygons against the volumes' 2,859 counties, and every code that DID resolve
    # found a polygon. The people were in `data/raw/cn/` the whole time.
    #
    # WHY THIS WAS INVISIBLE, AND IT IS THE MORE USEFUL HALF. main() prints
    # `drawn population stranded: 0.26%`, which reads as fine — but it counts only the
    # RELIGIO-ETHNIC population, because when that check was written only spec §14.5's
    # minorities were drawn. §14.13 made everyone drawn on `unknown` and nobody updated the
    # check, so the number that mattered became the TOTAL stranded, twenty times larger.
    # **The check did not become wrong, it became irrelevant, and nothing said so.**
    # main() now reports both.
    #
    # THE SHORTFALL IS SYSTEMATICALLY URBAN, which is why it bit hardest exactly where the
    # map is most interesting: 142 of the missing units are 市辖区. Between 2000 and now
    # China converted a great many counties into districts and renamed the results, so the
    # provinces that urbanised fastest lost the most — Hainan 82.4% of its census total
    # drawn, Zhejiang 86.6%, Jiangxi 88.3%, Guangxi 88.4%, Sichuan 90.3% — against exactly
    # 100.0% in Xinjiang, Ningxia, Xizang, Qinghai and Beijing, which barely reorganised.
    #
    # HOW EACH ONE WAS RESOLVED, so the next person can check rather than trust. Both the
    # census file and DataV run in GB/T 2260 code order within a province, so an unresolved
    # county lies in a known interval between its resolved neighbours; the successor is the
    # modern unit in that interval. `free` was NOT used as a filter — a 2000 unit is often
    # not renamed but ABSORBED, so the target is frequently a code another census county
    # already claims, which sources/cn.py has always allowed and reports.
    #
    # -- the biggest single miss, and it is a romanisation typo in the source -------
    (440000, "ZHONGZHAN"): "442000",          # 中山市 — the volume writes ZHONGZHAN, not
                                              # ZHONGSHAN. 2.36M people, and the reason
                                              # Zhongshan drew ZERO dots between two of the
                                              # densest deltas on the map.
    # -- characters with a place-name reading the romaniser did not use -------------
    (440000, "FANYU"): "440113",              # 番禺区 — 番 is pān here, not fān
    (360000, "BOYANG"): "361128",             # 鄱阳县 — 鄱 is pó
    (330000, "LEQING"): "330382",             # 乐清市 — 乐 is yuè
    (130000, "LETING"): "130225",             # 乐亭县 — 乐 is lào
    (140000, "CHANGZI"): "140428",            # 长子县 — 长 is zhǎng
    (360000, "QIANSHAN"): "361124",           # 铅山县 — 铅 is yán
    (510000, "JIANWEI"): "511123",            # 犍为县 — 犍 is qián
    (440000, "DAPU"): "441422",               # 大埔县 — 埔 is bù
    (410000, "NANLE"): "410923",              # 南乐县 — 乐 is lè
    (320000, "XIXIA"): "320113",              # 南京栖霞区 — 栖 is qī
    (370000, "XIXIA"): "370686",              # 栖霞市, the same character one province over
    (620000, "MINLE"): "620722",              # 民乐县
    (610000, "WUBAO"): "610829",              # 吴堡县 — 堡 is bǔ
    (370000, "DONGA"): "371524",              # 东阿县
    (430000, "XIANGYIN"): "430624",           # 湘阴县
    (610000, "FUPING"): "610528",             # 富平县
    (360000, "JINXIAN"): "360124",            # 进贤县
    (420000, "GONGAN"): "421022",             # 公安县
    (130000, "JINGXING"): "130121",           # 井陉县 — blocked by 井陉矿区
    (130000, "JINGXIAN"): "131127",           # 景县
    (350000, "LICHENG"): "350502",            # 泉州鲤城区
    (350000, "MAWEI"): "350105",              # 福州马尾区
    (510000, "NAXI"): "510503",               # 泸州纳溪区
    (510000, "MIANZHU"): "510683",            # 绵竹市
    (420000, "JINGSHAN"): "420882",           # 京山市
    (420000, "SHENNONGJIALIN"): "429021",     # 神农架林区
    (360000, "GUANGFENG"): "361103",          # 广丰区
    (150000, "BAIYUNKUANGQU"): "150206",      # 白云鄂博矿区
    # -- suffix dropped by the romaniser -------------------------------------------
    (500000, "KAI"): "500154",                # 开县 -> 开州区 (2016)
    (510000, "DA"): "511703",                 # 达县 -> 达川区 (2013)
    (450000, "HENG"): "450127",               # 横县 -> 横州市 (2021)
    (510000, "AN"): "510705",                 # 安县 -> 安州区 (2016)
    (610000, "HU"): "610118",                 # 户县 -> 鄠邑区 (2016)
    (610000, "YAO"): "610204",                # 耀县 -> 耀州区 (2002)
    (610000, "BIN"): "610482",                # 彬县 -> 彬州市 (2018)
    (610000, "HUA"): "610503",                # 华县 -> 华州区 (2015)
    (410000, "SHANXIAN"): "411203",           # 陕县 -> 陕州区 (2016)
    (320000, "WUXIAN"): "320506",             # 吴县市 -> 吴中区 + 相城区 (2001); the larger
    (330000, "YINXIAN"): "330212",            # 鄞县 -> 鄞州区 (2002)
    (130000, "LUANXIAN"): "130284",           # 滦县 -> 滦州市 (2018)
    (130000, "RENXIAN"): "130505",            # 任县 -> 任泽区 (2020)
    (330000, "QUXIAN"): "330803",             # 衢县 -> 衢江区 (2001)
    # -- a county of the same name as its prefecture, absorbed into the seat --------
    (520000, "ZUNYI"): "520304",              # 遵义县 -> 播州区 (2016)
    (520000, "BIJIE"): "520502",              # 毕节市 -> 七星关区 (2011)
    (520000, "TONGREN"): "520602",            # 铜仁市 -> 碧江区 (2011)
    (450000, "LAIBIN"): "451302",             # 来宾县 -> 兴宾区 (2002)
    (450000, "HEZHOU"): "451102",             # 贺州市 -> 八步区 (2002)
    (450000, "BAISE"): "451002",              # 百色市 -> 右江区 (2002)
    (450000, "HECHI"): "451202",              # 河池市 -> 金城江区 (2002)
    (450000, "CHONGZUO"): "451402",           # 崇左县 -> 江州区 (2003)
    (610000, "BAOJI"): "610304",              # 宝鸡县 -> 陈仓区 (2003)
    (620000, "QINGYANG"): "621021",           # 庆阳县 -> 庆城县 (2002)
    (360000, "JIUJIANG"): "360404",           # 九江县 -> 柴桑区 (2017)
    (360000, "SHANGRAO"): "361104",           # 上饶县 -> 广信区 (2019)
    (420000, "YICHANG"): "420506",            # 宜昌县 -> 夷陵区 (2001)
    (430000, "ZHUZHOU"): "430212",            # 株洲县 -> 渌口区 (2018)
    (340000, "TONGLING"): "340706",           # 铜陵县 -> 义安区 (2015)
    (340000, "WUHU"): "340210",               # 芜湖县 -> 湾沚区 (2020)
    (330000, "SHAOXING"): "330603",           # 绍兴县 -> 柯桥区 (2013)
    (330000, "JINHUA"): "330703",             # 金华县 -> 金东区 (2000)
    (330000, "HUZHOU"): "330502",             # 湖州市区 -> 吴兴区 (2003)
    (510000, "YIBIN"): "511504",              # 宜宾县 -> 叙州区 (2018)
    (130000, "HANDAN"): "130402",             # 邯郸县 -> 邯山区 (2016)
    (130000, "XINGTAI"): "130503",            # 邢台县 -> 信都区 (2020)
    (140000, "DATONG"): "140215",             # 大同县 -> 云州区 (2018)
    (140000, "CHANGZHI"): "140404",           # 长治县 -> 上党区 (2018)
    (410000, "ANYANG JIAOQU"): "410506",      # 安阳市郊区 -> 龙安区 (2002)
    (360000, "XINGZI"): "360483",             # 星子县 -> 庐山市 (2016)
    (370000, "JIAONAN"): "370211",            # 胶南市 -> 黄岛区 (2012)
    (370000, "CHANGDAO"): "370614",           # 长岛县 -> 蓬莱区 (2020)
    (460000, "TONGSHEN"): "469001",           # 通什市 -> 五指山市 (2001)
    (310000, "NANHUI"): "310115",             # 南汇区 -> 浦东新区 (2009)
    (220000, "BADAOJIANG"): "220602",         # 八道江区 -> 浑江区 (2010)
    # -- 市辖区 and 郊区 dissolved into their city's modern districts ---------------
    # The generic ones — 郊区, 城区, 市中区 — are the largest class and the reason a
    # province-wide name lookup cannot place them: dozens of prefectures had one.
    (130000, "SHIJIAZHUANG JIAOQU"): "130108",   # -> 裕华区 (2001)
    (130000, "XINQU"): "130209",                 # 唐山市新区 -> 曹妃甸区 (2012)
    (130000, "TANGHAI"): "130209",               # 唐海县 -> 曹妃甸区 (2012), same target
    (140000, "DATONG NANJIAOQU"): "140214",      # 大同市南郊区 -> 云冈区 (2018)
    (140000, "CHANGZHI JIAOQU"): "140403",       # 长治市郊区 -> 潞州区 (2018)
    (230000, "JIAMUSHI JIAOQU"): "230811",       # 佳木斯市郊区
    (230000, "BEILIN"): "231202",                # 绥化市北林区
    (230000, "YONGHONG"): "230803",              # 佳木斯永红区 -> 向阳区 (2006)
    (320000, "WUXI JIAOQU"): "320211",           # 无锡市郊区 -> 滨湖区 (2001)
    (320000, "MASHAN"): "320211",                # 无锡马山区 -> 滨湖区 (2001)
    (320000, "CHONGAN"): "320213",               # 无锡崇安区 -> 梁溪区 (2015)
    (320000, "NANCHANG"): "320213",              # 无锡南长区 -> 梁溪区 (2015)
    (320000, "BEITANG"): "320213",               # 无锡北塘区 -> 梁溪区 (2015)
    (320000, "JIULI"): "320312",                 # 徐州九里区 -> 铜山区 (2010)
    (320000, "QISHUYAN"): "320402",              # 常州戚墅堰区 -> 天宁区 (2015)
    (320000, "CHANGZHOU JIAOQU"): "320411",      # 常州市郊区 -> 新北区 (2002)
    (320000, "CANGLANG"): "320508",              # 苏州沧浪区 -> 姑苏区 (2012)
    (320000, "PINGJIANG"): "320508",             # 苏州平江区 -> 姑苏区 (2012)
    (320000, "JINCHANG"): "320508",              # 苏州金阊区 -> 姑苏区 (2012)
    (320000, "GANGZHA"): "320602",               # 南通港闸区 -> 崇川区 (2020)
    (320000, "YUNTAI"): "320706",                # 连云港云台区 -> 海州区 (2001)
    (320000, "XINPU"): "320706",                 # 连云港新浦区 -> 海州区 (2014)
    (320000, "QINGPU"): "320812",                # 淮安清浦区 -> 清江浦区 (2016)
    (320000, "YANCHENG CHENGQU"): "320902",      # 盐城市城区 -> 亭湖区 (2004)
    (320000, "YANGZHOU JIAOQU"): "321003",       # 扬州市郊区 -> 邗江区 (2000)
    (330000, "HANGZHOU XIACHENG"): "330105",     # 杭州下城区 -> 拱墅区 (2021)
    (330000, "JIANGGAN"): "330102",              # 杭州江干区 -> 上城区 (2021)
    (330000, "JIANGDONG"): "330212",             # 宁波江东区 -> 鄞州区 (2016)
    (340000, "JINJIAZHUANG"): "340503",          # 马鞍山金家庄区 -> 花山区 (2012)
    (340000, "TONGGUANSHAN"): "340705",          # 铜陵铜官山区 -> 铜官区 (2015)
    (340000, "SHIZISHAN"): "340705",             # 铜陵狮子山区 -> 铜官区 (2015)
    (340000, "JUCHAO"): "340181",                # 巢湖居巢区 -> 巢湖市 (2011)
    (350000, "GULANGYU"): "350203",              # 厦门鼓浪屿区 -> 思明区 (2003)
    (350000, "XINGLIN"): "350211",               # 厦门杏林区 -> 集美区 (2003)
    (350000, "MEILIE"): "350403",                # 三明梅列区 -> 三元区 (2021); DataV keeps
                                                 # the old 三元 code 350403, not 350404
    (360000, "WANLI"): "360112",                 # 南昌湾里区 -> 新建区 (2019)
    (360000, "NANCHANG JIAOQU"): "360111",       # 南昌市郊区 -> 青山湖区 (2002)
    (370000, "SIFANG"): "370203",                # 青岛四方区 -> 市北区 (2012)
    (410000, "MANGSHANQU"): "410108",            # 郑州邙山区 -> 惠济区 (2004)
    (410000, "JILIQU"): "410306",                # 洛阳吉利区 -> 孟津区 (2021)
    (410000, "TIEXIQU"): "410505",               # 安阳铁西区 -> 殷都区 (2002)
    (410000, "HEBI JIAOQU"): "410611",           # 鹤壁市郊区 -> 淇滨区
    (410000, "BEIZHANQU"): "410704",             # 新乡北站区 -> 凤泉区 (2003)
    (410000, "XINXIANG JIAOQU"): "410711",       # 新乡市郊区 -> 牧野区 (2003)
    (420000, "SHIHUIYAO"): "420203",             # 黄石石灰窑区 -> 西塞山区 (2001)
    (430000, "JIANGDONG"): "430405",             # 衡阳江东区 -> 珠晖区 (2001)
    (430000, "CHENGNAN"): "430406",              # 衡阳城南区 -> 雁峰区 (2001)
    (430000, "CHENGBEI"): "430407",              # 衡阳城北区 -> 石鼓区 (2001)
    (430000, "HENGYANG JIAOQU"): "430408",       # 衡阳市郊区 -> 蒸湘区 (2001)
    (430000, "ZHISHAN"): "431102",               # 永州芝山区 -> 零陵区 (2005)
    (440000, "DONGSHAN"): "440104",              # 广州东山区 -> 越秀区 (2005)
    (440000, "FANGCUN"): "440103",               # 广州芳村区 -> 荔湾区 (2005)
    (440000, "BEIJIANG"): "440204",              # 韶关北江区 -> 浈江区 (2004)
    (440000, "DAHAO"): "440512",                 # 汕头达濠区 -> 濠江区 (2003)
    (440000, "HEPU"): "440512",                  # 汕头河浦区 -> 濠江区 (2003)
    (440000, "JINYUAN"): "440511",               # 汕头金园区 -> 金平区 (2003)
    (440000, "SHENGPING"): "440511",             # 汕头升平区 -> 金平区 (2003)
    (440000, "SHIWAN"): "440604",                # 佛山石湾区 -> 禅城区 (2002)
    (450000, "NANNING CHENGBEI"): "450102",      # 南宁城北区 -> 兴宁区 (2005)
    (450000, "NANNING YONGXIN"): "450107",       # 南宁永新区 -> 西乡塘区 (2005)
    (450000, "NANNING SHIJIAO"): "450107",       # 南宁市郊区 -> 西乡塘区 (2005)
    (450000, "LIUZHOUJIAOQU"): "450203",         # 柳州市郊区 -> 鱼峰区 (2002)
    (450000, "DIESHAN"): "450403",               # 梧州蝶山区 -> 万秀区 (2013)
    (450000, "WUZHOUJIAOQU"): "450405",          # 梧州市郊区 -> 长洲区 (2003)
    (460000, "ZHENDONG"): "460108",              # 海口振东区 -> 美兰区 (2002)
    (460000, "XINHUA"): "460106",                # 海口新华区 -> 龙华区 (2002)
    (500000, "WANSHENG"): "500110",              # 重庆万盛区 -> 綦江区 (2011)
    (500000, "SHUANGQIAO"): "500111",            # 重庆双桥区 -> 大足区 (2011)
    (510000, "YUANBA"): "510811",                # 广元元坝区 -> 昭化区 (2013)
    (510000, "ZIZHONG"): "511025",               # 资中县
    (520000, "XIAOHE"): "520111",                # 贵阳小河区 -> 花溪区 (2012)
    (520000, "WANSHANTE"): "520603",             # 万山特区 -> 万山区 (2011)
    (650000, "NANQUAN"): "650107",               # 乌鲁木齐南山矿区 -> 达坂城区 (2002)
    # **市中区 IS THE SHARPEST CASE OF THE GENERIC NAME** — Sichuan has three, in file
    # order 遂宁, 内江, 乐山, so this is a LIST and the nth occurrence takes the nth code.
    (510000, "SHIZHONG"): ["510903", "511002", "511102"],
    # -- Yichun's forestry districts, and the one group placed only APPROXIMATELY ---
    # 伊春 was 15 districts in 2000 and was reorganised into 4 districts + 4 counties in
    # 2019, so thirteen census units map onto seven modern ones. The pairings below follow
    # the 2019 reorganisation; where a 2000 district was split rather than absorbed whole,
    # the whole of it goes to the successor that took its seat. **Everyone lands inside
    # Yichun** — about 500,000 people — and the residual error is which of two adjacent
    # districts of one small city a dot sits in.
    (230000, "YICHUN"): "230717",             # 伊春区 -> 伊美区
    (230000, "MEIXI"): "230717",              # 美溪区 -> 伊美区
    (230000, "WUMAHE"): "230718",             # 乌马河区 -> 乌翠区
    (230000, "CUILUAN"): "230718",            # 翠峦区 -> 乌翠区
    (230000, "XILIN"): "230751",              # 西林区 -> 金林区
    (230000, "JINSHANTUN"): "230751",         # 金山屯区 -> 金林区
    (230000, "SHANGGANLING"): "230719",       # 上甘岭区 -> 友好区
    (230000, "XINQING"): "230723",            # 新青区 -> 汤旺县
    (230000, "TANGWANGHE"): "230723",         # 汤旺河区 -> 汤旺县
    (230000, "WUYILING"): "230723",           # 乌伊岭区 -> 汤旺县
    (230000, "WUYING"): "230724",             # 五营区 -> 丰林县
    (230000, "HONGXING"): "230724",           # 红星区 -> 丰林县
    (230000, "DAILING"): "230725",            # 带岭区 -> 大箐山县
    # 大兴安岭's three forestry districts have NO GB/T 2260 code of their own in DataV —
    # 松岭, 新林 and 呼中 are 林业局 areas the index does not carry. They go to the
    # adjacent county that administers them, which keeps 131,000 people inside the right
    # prefecture; this is the weakest placement in the block and is flagged as such.
    (230000, "SONGLING"): "232718",           # -> 加格达奇区 (adjacent)
    (230000, "XINLIN"): "232718",             # -> 加格达奇区 (adjacent)
    (230000, "HUZHONG"): "232722",            # -> 塔河县 (adjacent)
    # -- two names the source itself is byte-corrupted for -------------------------
    # The Hubei volume mis-encodes two rare characters — 硚 and 猇 — so the name that
    # arrives is mojibake and no romanisation rule can ever reach it. The adcode interval
    # identifies both without needing the name at all.
    (420000, "�~��"): "420104",   # 硚口区, 686,318 people
    (420000, "�Vͤ"): "420505",         # 猇亭区, 52,827 people
    # ==============================================================================
    # -- prefecture seats renamed on upgrade ---------------------------------------
    (640000, "GUYUAN"): "640402",             # 固原县 -> 固原市原州区
    (650000, "TULUFAN"): "650402",            # 吐鲁番市 -> 吐鲁番市高昌区
    (620000, "PINGLIANG"): "620802",          # 平凉市 -> 平凉市崆峒区
    (650000, "HAMI"): "650502",               # 哈密市 -> 哈密市伊州区
    (530000, "ZHAOTONG"): "530602",           # 昭通市 -> 昭通市昭阳区
    (540000, "RIKAZE"): "540202",             # 日喀则市 -> 日喀则市桑珠孜区
    (540000, "CHANGDU"): "540302",            # 昌都县 -> 昌都市卡若区
    (540000, "NAQU"): "540602",               # 那曲县 -> 那曲市色尼区
    (540000, "LINZHI"): "540402",             # 林芝县 -> 林芝市巴宜区
    (530000, "BAOSHAN"): "530502",            # 保山市 -> 保山市隆阳区
    (530000, "LINCANG"): "530902",            # 临沧县 -> 临沧市临翔区
    (620000, "QINCHENG"): "620502",           # 天水市秦城区 -> 秦州区
    (620000, "BEIDAO"): "620503",             # 天水市北道区 -> 麦积区
    (620000, "JIUQUAN"): "620902",            # 酒泉市 -> 酒泉市肃州区
    (620000, "WUWEI"): "620602",              # 武威市 -> 武威市凉州区
    (620000, "ZHANGYE"): "620702",            # 张掖市 -> 张掖市甘州区
    (620000, "DINGXI"): "621102",             # 定西县 -> 定西市安定区
    (640000, "ZHONGWEI"): "640502",           # 中卫县 -> 中卫市沙坡头区
    (530000, "QILIN"): "530302",              # 曲靖市麒麟区 (pinyin qulin/qilin)
    # -- renamed counties ----------------------------------------------------------
    # (530000, "ZHONGDIAN") was here with the WRONG code 533422 (德钦县) — corrected to
    # 533401 further down, 2026-09-08. Left as a comment because the mistake is instructive:
    # the entry's own comment named 香格里拉市 correctly and only the digits were wrong, so
    # nothing in the file read as suspicious.
    (650000, "MIQUAN"): "650109",             # 米泉市 -> 乌鲁木齐市米东区
    (650000, "DONGSHAN"): "650109",           # 乌鲁木齐东山区 -> 米东区 (merged)
    (620000, "ANXI"): "620922",               # 安西县 -> 瓜州县 (2006)
    (370000, "LING"): "371403",               # 陵县 -> 德州市陵城区
    (370000, "CANGSHAN"): "371324",           # 苍山县 -> 兰陵县 (2014)
    (520000, "PAN"): "520281",                # 盘县 -> 盘州市
    (410000, "KAIFENG"): "410212",            # 开封县 -> 开封市祥符区
    (510000, "BEICHUAN"): "510726",           # 北川县 -> 北川羌族自治县
    # (650000, "WEILI") was here with the WRONG code 652927 (乌什县) — corrected to 652823
    # further down, 2026-09-08. Same shape as ZHONGDIAN above.
    (540000, "LANGKAZI"): "540531",           # 浪卡子县 (DataV pinyin langqiazi)
    # -- word order differs from DataV's -------------------------------------------
    (620000, "JISHISHANDONGXIANGZUBAOANZUSALAZUZHZHIXIAN"): "622927",
    (530000, "LIJIANGNAXIZUZHZHIXIANXIAN"): "530721",   # -> 玉龙纳西族自治县 (2003)
    (530000, "PUERHANIZUYIZUZHZHIXIAN"): "530821",      # 普洱 -> 宁洱哈尼族彝族自治县
    # -- urban districts merged away -----------------------------------------------
    (640000, "YINCHUAN CHENGQU"): "640104",   # 银川城区 -> 兴庆区
    (640000, "YINCHUAN XINCHENGQU"): "640105",   # 新城区 -> 西夏区
    (640000, "YINCHUAN JIAOQU"): "640106",    # 郊区 -> 金凤区
    (640000, "SHIZUISHAN"): "640205",         # 石嘴山区 -> 惠农区
    (640000, "SHITANJING"): "640202",         # 石炭井区 -> 大武口区 (merged)
    (640000, "TAOLE"): "640221",              # 陶乐县 -> 平罗县 (merged, 2003)
    (110000, "XUANWU"): "110102",             # 宣武区 -> 西城区 (merged, 2010)
    (110000, "CHONGWEN"): "110101",           # 崇文区 -> 东城区 (merged, 2010)
    (120000, "TANGGU"): "120116",             # 塘沽 -> 滨海新区 (2009)
    (120000, "HANGU"): "120116",              # 汉沽 -> 滨海新区
    (120000, "DAGANG"): "120116",             # 大港 -> 滨海新区
    (310000, "NANSHI"): "310101",             # 南市区 -> 黄浦区 (2000)
    (310000, "LUWAN"): "310101",              # 卢湾区 -> 黄浦区 (2011)
    (310000, "ZHABEI"): "310106",             # 闸北区 -> 静安区 (2015)
    (410000, "DISTRICTS"): "410902",          # the translator left 濮阳市辖区 unromanised
    # -- urban districts renamed or absorbed, found via file-order neighbours ---------
    # A value that is a LIST is consumed in file order, for a name the province repeats:
    # Hebei has three 桥西区 and only two survive under that name.
    (140000, "CHANGZHI CHENGQU"): "140403",   # 长治市城区 -> 潞州区
    (140000, "DATONG CHENGQU"): "140213",     # 大同市城区 -> 平城区
    (610000, "BEILIN"): "610103",             # 西安市碑林区
    (410000, "XUCHANG"): "411003",            # 许昌县 -> 建安区
    (410000, "HUIXIAN"): "410782",            # 新乡市辉县市
    (410000, "MIYANG"): "411726",             # 驻马店市泌阳县
    (410000, "PINQIAOQU"): "411503",          # 信阳市平桥区
    (410000, "KAIFENG JIAOQU"): "410202",     # 开封市郊区 -> 龙亭区 (abolished 2005)
    (410000, "NANGUANQU"): "410205",          # 开封市南关区 -> 禹王台区
    (130000, "QIAOXI"): ["130104", "130503", "130703"],   # 石家庄 / 邢台(->信都区) / 张家口
    (130000, "XINSHI"): "130602",             # 保定市新市区 -> 竞秀区
    (130000, "NANSHI"): "130606",             # 保定市南市区 -> 莲池区
    (130000, "BEISHI"): "130606",             # 保定市北市区 -> 莲池区
    # ---- WRONG PREFECTURE ENTIRELY, FOUND BY tools/check_cn_prefecture.py ---------
    # **The worst errors in this file, and none of them was visible.** Each of these had
    # resolved to a real adcode in a DIFFERENT PREFECTURE — often a different corner of the
    # province — so the county's whole population was drawn hundreds of kilometres from
    # home while the true county drew nothing. ~3.4M people. Two were pre-existing
    # OVERRIDES whose comment named the right county and whose code named another.
    #
    # The check that finds them is one line of reasoning: both the census file and DataV
    # run in GB/T 2260 order, so a county's file NEIGHBOURS are its neighbours in code
    # space. A resolution landing in a prefecture that neither neighbour is in is almost
    # always the wrong same-named county. It flags 35, of which 25 are legitimate — a
    # county really did move prefecture (簡阳 to Chengdu, 无为 to Wuhu, 寿县 to Huainan,
    # 公主岭 to Changchun, 枞阳 to Tongling, 海原 to Zhongwei) or is provincially
    # administered (济源, 儋州, 石河子, 嘉峪关) — and these ten are real.
    (420000, "YUN"): "420304",                # 郧县 -> 郧阳区. Was landing on 云梦县 in
                                              # XIAOGAN, 400 km away. 584,315 people.
    (370000, "SHIZHONG"): "370402",           # 枣庄市中区. Was landing on 济南市中区.
    (370000, "JINING SHIZHONG"): "370811",    # 济宁市中区 -> 任城区 (2013). Was on 枣庄.
    (140000, "DATONG KUANGQU"): "140214",     # 大同矿区 -> 云冈区 (2018, with 南郊区).
                                              # Was landing on YANGQUAN's 矿区.
    (450000, "NANNING XINCHENG"): "450103",   # 南宁新城区 -> 青秀区 (2005). Was landing
                                              # on 忻城县 in LAIBIN.
    (440000, "FOSHAN CHENGQU"): "440604",     # 佛山城区 -> 禅城区 (2002, with 石湾区).
                                              # Was landing on SHANWEI's 城区.
    (340000, "ANQING JIAOQU"): "340811",      # 安庆市郊区 -> 宜秀区 (2005). Was landing
                                              # on TONGLING's 郊区.
    (360000, "LUSHAN"): "360402",             # 九江庐山区 -> 濂溪区 (2016). Was landing on
                                              # 庐山市, which is 星子县 and is XINGZI's.
    # -- and two OVERRIDES that were simply typed wrong; the comments were right ----
    (530000, "ZHONGDIAN"): "533401",          # 中甸县 -> 香格里拉市 (2001). The old entry
                                              # said 533422, which is 德钦县.
    (650000, "WEILI"): "652823",              # 尉犁县 (pinyin yuli). The old entry said
                                              # 652927, which is 乌什县 in AKSU.
    # ---- NAME COLLISIONS: ONE ROMANISATION, SEVERAL REAL COUNTIES -----------------
    # **Found 2026-09-08 and worth more than the 168 of §14.17, because these errors cross
    # PREFECTURES rather than districts.** Where the same romanisation names two or three
    # genuinely different counties, the resolver piled all of them onto one adcode: Hebei's
    # three Wei counties — 魏县 in Handan, 威县 in Xingtai, 蔚县 in Zhangjiakou, all
    # `WEIXIAN` — were drawn as one, putting 1.27M people up to 300 km from home while two
    # real counties drew nothing at all. ~2.48M people across the five entries below.
    #
    # A LIST TAKES THE nth OCCURRENCE IN FILE ORDER, which is the mechanism §12 built for
    # 伊宁市/伊宁县 and which is exactly right here: both the census file and DataV run in
    # GB/T 2260 order, so the nth `WEIXIAN` is the nth Wei county by adcode. Each was placed
    # from its file neighbours, not from the name — 魏县 sits between 馆陶县 and 曲周县,
    # which no romanisation could tell you.
    (130000, "WEIXIAN"): ["130434", "130533", "130726"],   # 魏县 / 威县 / 蔚县
    (130000, "QIAODONG"): ["130102", "130502", "130702"],  # 桥东区 x3: 石家庄 (-> 长安区,
                                                           # merged 2014), 邢台 (-> 襄都区),
                                                           # 张家口 (still 130702)
    (530000, "LUXI"): ["532527", "533103"],       # 泸西县 (红河) / 潞西市 (德宏 -> 芒市)
    (410000, "XINHUAQU"): ["410402", "410703"],   # 新华区 x2: 平顶山, and 新乡 -> 卫滨区
    (340000, "XIANGSHAN"): ["340504", "340603"],  # 马鞍山向山区 (-> 雨山区) / 淮北相山区
    # `XUANHUA` x2 is NOT here and is not a bug: 宣化区 and 宣化县 were two real counties in
    # 2000 and genuinely merged into one 宣化区 in 2016, so both landing on 130705 is right.
    (320000, "BAIXIA"): "320104",             # 南京市白下区 -> 秦淮区 (2013)
    (320000, "XIAGUAN"): "320106",            # 南京市下关区 -> 鼓楼区 (2013)
    (370000, "LANSHAN"): "371302",            # 临沂市兰山区 (not 日照市岚山区)
    (370000, "LAICHENG"): "370116",           # 莱芜市莱城区 -> 济南市莱芜区
    (230000, "TAIPING"): "230104",            # 哈尔滨市太平区 -> 道外区 (2004)
    (230000, "DONGLI"): "230110",             # 哈尔滨市动力区 -> 香坊区 (2004)
    (210000, "DONGLING"): "210112",           # 沈阳市东陵区 -> 浑南区
    (210000, "BEINING"): "210782",            # 北宁市 -> 北镇市
    (510000, "GUANGYUAN SHIZHONG"): "510802",  # 广元市市中区 -> 利州区
    (350000, "KAIYUAN"): "350203",            # 厦门市开元区 -> 思明区 (2003)
    (350000, "PUTIAN"): "350304",             # 莆田县 -> 荔城区 (abolished 2002)
    (520000, "LIUZHITE"): "520203",           # 六盘水市六枝特区
    (460000, "SANYA"): "460204",              # 三亚市 -> 天涯区, where the Utsul live
    # Anhui renamed the compass-point districts of both Hefei (2002) and Bengbu (2004),
    # so each of these names occurs twice in the file and the list is in code order.
    (340000, "DONGSHI"): ["340102", "340302"],    # 东市区 -> 瑶海区 / 龙子湖区
    (340000, "ZHONGSHI"): ["340103", "340303"],   # 中市区 -> 庐阳区 / 蚌山区
    (340000, "XISHI"): ["340104", "340304"],      # 西市区 -> 蜀山区 / 禹会区
    (340000, "HEFEI JIAOQU"): "340111",           # 合肥市郊区 -> 包河区
    (340000, "BANGBU JIAOQU"): "340311",          # 蚌埠市郊区 -> 淮上区
    (340000, "MATANG"): "340209",                 # 芜湖市马塘区 -> 弋江区 (DataV's code)
    (340000, "XINWU"): "340202",                  # 芜湖市新芜区 -> 镜湖区 (merged)
    # The rest, each confirmed from its file-order neighbours.
    (120000, "JIXIAN"): "120119",             # 蓟县 -> 蓟州区
    (320000, "DACHANG"): "320116",            # 南京市大厂区 -> 六合区 (2002)
    (320000, "JIANGPU"): "320111",            # 江浦县 -> 浦口区 (2002)
    (320000, "QINGHE"): "320812",             # 淮安市清河区 -> 清江浦区
    (210000, "XINCHENGZI"): "210113",         # 沈阳市新城子区 -> 沈北新区
    (210000, "TIEFA"): "211281",              # 铁法市 -> 调兵山市
    (140000, "WANBAILIN"): "140109",          # 太原市万柏林区
    (510000, "PI"): "510117",                 # 郫县 -> 郫都区
    (510000, "JINGYANG"): "510603",           # 德阳市旌阳区
    (330000, "XIUCHENG"): "330402",           # 嘉兴市秀城区 -> 南湖区
    (420000, "XIANGYANG"): "420607",          # 襄阳县 -> 襄州区
}


def _clean(s):
    return re.sub(r"[^a-z]", "", s.lower())


def _norm(name):
    """Census romanisation -> the keys it could be, comparable with pinyin.

    `zhzhi` is the source's own abbreviation of 自治 and appears in every autonomous
    county. The TRANSLIT rules are characters the source and pinyin disagree about, and
    they ADD a candidate rather than replace one: Shandong's 莘县 really is romanised
    SHEN, so rewriting every trailing -shen to -shi loses a county to fix Kashgar.
    """
    base = _clean(name).replace("zhzhi", "zizhi")
    out = [base]
    for a, b in TRANSLIT_ANY:
        out += [n.replace(a, b) for n in list(out) if a in n]
    for a, b in TRANSLIT_END:
        out += [n[: -len(a)] + b for n in list(out) if n.endswith(a)]
    seen, keys = set(), []
    for n in out:
        if n not in seen:
            seen.add(n)
            keys.append(n)
    return keys


def _stem(s):
    out = SUFFIX.sub("", s)
    return out if len(out) > 1 else s


def build_resolver(index_path):
    """(province_code, census_name) -> adcode, from DataV's county index."""
    if not os.path.exists(index_path):
        raise SystemExit(f"!! {index_path} missing — run `python sources/cn_geo.py --fetch`")
    from pypinyin import Style, lazy_pinyin, pinyin

    with open(index_path, encoding="utf-8") as fh:
        index = json.load(fh)

    # An OVERRIDES target that is not a real adcode drops its county silently — the join
    # succeeds, the scatter finds no polygon, and the people are simply gone. That is
    # spec 8.1's whole complaint, so a typo fails here instead of downstream. It has
    # already caught one: 340203 for Wuhu's Yijiang district, which DataV numbers 340209.
    have = {r["code"] for r in index}
    bad = sorted({c for v in OVERRIDES.values()
                  for c in (v if isinstance(v, list) else [v]) if c not in have})
    if bad:
        raise SystemExit(f"!! OVERRIDES targets that are not DataV adcodes: {bad}")

    def keys(name):
        het = [h[:3] for h in pinyin(name, style=Style.NORMAL, heteronym=True)]
        n = 1
        for h in het:
            n *= len(h)
        if n > 400:
            het = [[h[0]] for h in het]
        out = set()
        import itertools
        for combo in itertools.product(*het):
            k = _clean("".join(combo))
            out |= {k, _stem(k)}
        k = _clean("".join(lazy_pinyin(name, style=Style.NORMAL)))
        out |= {k, _stem(k)}
        return {k for k in out if k}

    # Two lookups, not one. Pooling the full pinyin and the stem makes a stem collision
    # poison a name that matched exactly: Henan's 固始县 is `GUSHI`, whose stem `gu` hits
    # half the province, and the county was being called ambiguous on the strength of a
    # key it never needed. Tiers are tried in order and the first non-empty one wins.
    lookup_full = defaultdict(lambda: defaultdict(list))
    lookup_stem = defaultdict(lambda: defaultdict(list))
    city_full = defaultdict(lambda: defaultdict(list))
    city_stem = defaultdict(lambda: defaultdict(list))
    cities = defaultdict(set)
    for r in index:
        p = int(r["prov_code"])
        cs = _clean("".join(lazy_pinyin(
            re.sub(r"(市|地区|自治州|盟)$", "", r["city"]), style=Style.NORMAL)))
        cities[p].add(cs)
        for k in keys(r["name"]):
            lookup_full[p][k].append(r["code"])
            lookup_stem[p][_stem(k)].append(r["code"])
            city_full[(p, cs)][k].append(r["code"])
            city_stem[(p, cs)][_stem(k)].append(r["code"])
    city_stems = {p: sorted(v, key=len, reverse=True) for p, v in cities.items()}

    def candidates(prov, name):
        """First non-empty tier of candidates, as a sorted adcode list.

        The city-scoped tiers are the important ones. The census disambiguates a district
        by prefixing its city -- `JINAN SHIZHONG` -- and Shandong has three 市中区, so a
        province-wide lookup on the stripped `shizhong` is ambiguous while the same
        lookup inside Jinan is exact. Scoping beats guessing, and it is available because
        DataV carries each county's parent city.
        """
        base = _norm(name)
        stripped, city = [], None
        for n in base:
            for cs in city_stems.get(prov, ()):
                if n.startswith(cs) and len(n) > len(cs) + 1:
                    stripped.append(n[len(cs):])
                    city = cs
                    break
        sstem = [_stem(n) for n in stripped]
        tiers = [
            (base, lookup_full, prov),                     # DAWUKOU -> 大武口区
            (base, lookup_stem, prov),                     # YINING -> 市 / 县, by order
            ([_stem(n) for n in base], lookup_stem, prov),
            (stripped, city_full, (prov, city)),           # JINAN SHIZHONG -> 370103
            (stripped, city_stem, (prov, city)),
            (sstem, city_stem, (prov, city)),
            (stripped, lookup_full, prov),
            (stripped, lookup_stem, prov),
        ]
        for ks, table, scope in tiers:
            if not ks or (isinstance(scope, tuple) and scope[1] is None):
                continue
            out = []
            for k in ks:
                for code in table[scope].get(k, ()):
                    if code not in out:
                        out.append(code)
            if out:
                return sorted(out)
        return []

    def resolve_province(prov, names):
        """Census county names IN FILE ORDER -> [(adcode | None, how), ...].

        Ambiguity here is not noise, it is a specific and very common shape: the census
        romanisation drops the administrative suffix, so Yining CITY and Yining COUNTY
        are both `YINING`, as are Hetian, Linxia and a few dozen more. Taking the first
        candidate would put a prefecture's whole urban Uyghur population in its rural
        county half the time.

        Both lists are in GB/T 2260 code order and 市 precedes 县 in it, so the nth
        occurrence of a repeated name is the nth candidate by adcode. That is the only
        thing resolving these, and it is why the file order is never sorted anywhere
        above this point.
        """
        out = [(None, "unresolved")] * len(names)
        cands = []
        nth = defaultdict(int)
        for i, nm in enumerate(names):
            if (prov, nm) in OVERRIDES:
                v = OVERRIDES[(prov, nm)]
                if isinstance(v, list):
                    k = nth[(prov, nm)]
                    nth[(prov, nm)] += 1
                    v = v[k] if k < len(v) else None
                out[i] = (v, "override" if v else "override-short")
                cands.append(None)
            else:
                cands.append(candidates(prov, nm))

        groups = defaultdict(list)
        for i, c in enumerate(cands):
            if c is None:
                continue
            if len(c) == 1:
                out[i] = (c[0], "name")
            elif c:
                groups[tuple(c)].append(i)

        for codes, rows in groups.items():
            if len(rows) == len(codes):
                for i, code in zip(rows, codes):   # file order vs adcode order
                    out[i] = (code, "ordered")
            else:
                for i in rows:
                    out[i] = (None, "ambiguous")
        return out

    return resolve_province, {int(r["prov_code"]): r for r in index}


# ==================================================================================
# main
# ==================================================================================

DRAWN_GROUPS = ["Hui", "Uyghur", "Kazakh", "Dongxiang", "Salar", "Kyrgyz", "Tajik",
                "Uzbek", "Bonan", "Tatar", "Tibetan", "Yugur", "Monba", "Pumi", "Dai"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch:
        fetch()
        return

    gi = {g: i for i, g in enumerate(GROUPS)}
    drawn_ix = [gi[g] for g in DRAWN_GROUPS]

    # ---- read ------------------------------------------------------------------
    by_prov = read_2000()
    n_counties = sum(len(v) for v in by_prov.values())
    file_total = sum(v[0] for rows in by_prov.values() for _, v in rows)
    print(f"2000 census: {len(by_prov)} provinces, {n_counties} counties, "
          f"{file_total:,} people")
    if n_counties != EXPECTED_COUNTIES:
        print(f"  !! expected {EXPECTED_COUNTIES} counties")
    shortfall = CENSUS_2000_PROVINCES - file_total
    hainan = sum(v[0] for _, v in by_prov.get(460000, []))
    print(f"  shortfall against the published provincial sum: {shortfall:,}")
    print(f"  Hainan in file {hainan:,} against its census {HAINAN_2000:,} "
          f"= {HAINAN_2000 - hainan:,}")
    if shortfall != HAINAN_2000 - hainan:
        print("  !! the shortfall is NO LONGER exactly Hainan — something else is missing")
    else:
        print("  the shortfall IS Hainan, exactly; every other province is complete")

    national_2010, prov_2010 = read_2010()
    print(f"\n2010 NBS table 1-6: {len(prov_2010)} provinces, "
          f"{national_2010[0]:,} people")
    for g, want in [("Hui", 10_586_087), ("Uyghur", 10_069_346),
                    ("Tibetan", 6_282_187), ("Mongol", 5_981_840)]:
        got = national_2010[gi[g]]
        print(f"  {g:9s} {got:>12,}  published {want:>12,}"
              f"{'' if got == want else '   !! MISMATCH'}")

    # ---- resolve counties to adcodes -------------------------------------------
    resolve_province, _ = build_resolver(
        os.path.join(RAW, "datav", "county_index.json"))
    resolved, unresolved, how = {}, [], defaultdict(int)
    for prov, rows in by_prov.items():
        for i, (code, kind) in enumerate(resolve_province(prov, [n for n, _ in rows])):
            name, vals = rows[i]
            how[kind] += 1
            if code is None:
                unresolved.append((sum(vals[j] for j in drawn_ix), prov, name, vals[0]))
            else:
                resolved[(prov, i)] = code

    drawn_total = sum(v[i] for rows in by_prov.values() for _, v in rows
                      for i in drawn_ix)
    drawn_lost = sum(u[0] for u in unresolved)
    print(f"\ncounty -> adcode: {len(resolved)}/{n_counties} resolved "
          f"({how['name']} by name, {how['ordered']} by code order, "
          f"{how['override']} by override)")
    print(f"  unresolved: {len(unresolved)} counties "
          f"({how['ambiguous']} of them because the name matched two adcodes)")
    print(f"  drawn population stranded: {drawn_lost:,} of {drawn_total:,} "
          f"= {drawn_lost / drawn_total:.2%}")
    # **AND THE SAME NUMBER FOR EVERYBODY, WHICH IS THE ONE THAT MATTERS NOW.** The line
    # above counts only spec §14.5's religio-ethnic groups, because when it was written
    # they were the only people drawn. §14.13 put the whole country on `unknown`, and from
    # that day the meaningful figure was the TOTAL stranded — which was 5.47% while this
    # check reported 0.26% and looked healthy. The check did not become wrong, it became
    # irrelevant, and nothing said so. spec §14.17.
    total_pop = sum(v[0] for rows in by_prov.values() for _, v in rows)
    lost_pop = sum(u[3] for u in unresolved)
    print(f"  TOTAL population stranded: {lost_pop:,} of {total_pop:,} "
          f"= {lost_pop / total_pop:.2%}"
          f"{'' if not lost_pop else '   !! these people are dropped, not redistributed'}")
    carrying = [u for u in sorted(unresolved, reverse=True) if u[0] > 0]
    print(f"  every unresolved county carrying drawn people ({len(carrying)}):")
    for d, prov, name, pop in carrying:
        print(f"    {d:>9,} drawn  pop {pop:>9,}  "
              f"{GB_PROVINCE[f'{prov//10000:02d}'][1]:15s} {name}")

    # Two adcodes taking more than one census county is expected -- Tianjin's three
    # coastal districts all became Binhai -- so it is reported, not treated as an error.
    per_code = defaultdict(list)
    for (prov, i), code in resolved.items():
        per_code[code].append(by_prov[prov][i][0])
    merged = {c: n for c, n in per_code.items() if len(n) > 1}
    print(f"  adcodes receiving more than one census county: {len(merged)}")
    for c, n in sorted(merged.items())[:8]:
        print(f"    {c}  <- {', '.join(n)}")

    # ---- spec 3.4 rescale ------------------------------------------------------
    # Denominator is the sum over ALL county rows in the file, not just the resolved
    # ones. Using the resolved subset would silently redistribute an unresolved county's
    # people into its neighbours, which is spec 8.1's Connecticut failure wearing a
    # different hat. The consequence is that the written rows sum to slightly LESS than
    # the 2010 provincial total, by exactly the stranded share, and that is the honest
    # arithmetic: dropped people are dropped, not spread.
    factor = {}
    suspicious = []
    for prov, rows in by_prov.items():
        base = [0] * N_GROUPS
        for _, v in rows:
            for i in range(N_GROUPS):
                base[i] += v[i]
        want = prov_2010.get(prov)
        if want is None:
            print(f"  !! no 2010 row for province {prov}")
            continue
        f = []
        for i in range(N_GROUPS):
            if base[i] == 0:
                f.append(0.0)
                if want[i] and i in drawn_ix:
                    suspicious.append((prov, GROUPS[i], "2000 zero, 2010 "
                                       f"{want[i]:,} — cannot place"))
            else:
                r = want[i] / base[i]
                f.append(r)
                if i in drawn_ix and want[i] > 5000 and not (0.5 <= r <= 2.0):
                    suspicious.append((prov, GROUPS[i], f"factor {r:.2f}"))
        factor[prov] = f

    print(f"\nspec 3.4 rescale: 2000 county structure x 2010 provincial totals")
    if suspicious:
        print(f"  !! {len(suspicious)} (province, group) factors outside 0.5–2.0 "
              f"or unplaceable:")
        for prov, g, why in suspicious[:15]:
            print(f"     {GB_PROVINCE[f'{prov//10000:02d}'][1]:15s} {g:10s} {why}")
    else:
        print("  every drawn factor inside 0.5–2.0")

    # ---- write -----------------------------------------------------------------
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    written = 0
    drawn_written = 0
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for prov, rows in sorted(by_prov.items()):
            f = factor.get(prov)
            if f is None:
                continue
            pname = GB_PROVINCE[f"{prov//10000:02d}"][1]
            for ri, (name, vals) in enumerate(rows):
                code = resolved.get((prov, ri))
                if code is None:
                    continue
                for i, g in enumerate(GROUPS):
                    if vals[i] == 0 and g != "Total":
                        continue
                    n = int(round(vals[i] * f[i]))
                    if n == 0 and g != "Total":
                        continue
                    note = (f"province={pname}; structure_year={STRUCTURE_YEAR}; "
                            f"total_year={TOTAL_YEAR}; census2000={vals[i]}; "
                            f"scale={f[i]:.4f}")
                    if g == "Total":
                        note += "; unit population, not a category"
                    w.writerow([code, "county", name, g, n, BASIS, TOTAL_YEAR,
                                SOURCE_ID, note])
                    written += 1
                    if i in drawn_ix:
                        drawn_written += n
    print(f"\nwrote {OUT}")
    print(f"  {written:,} rows over {len({c for c in resolved.values()})} adcodes")
    print(f"  drawn population written: {drawn_written:,}")
    print(f"  (2010 national totals for those groups: "
          f"{sum(national_2010[i] for i in drawn_ix):,})")


if __name__ == "__main__":
    main()
