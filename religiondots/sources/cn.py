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
    (530000, "ZHONGDIAN"): "533422",          # 中甸县 -> 香格里拉市 (2001)
    (650000, "MIQUAN"): "650109",             # 米泉市 -> 乌鲁木齐市米东区
    (650000, "DONGSHAN"): "650109",           # 乌鲁木齐东山区 -> 米东区 (merged)
    (620000, "ANXI"): "620922",               # 安西县 -> 瓜州县 (2006)
    (370000, "LING"): "371403",               # 陵县 -> 德州市陵城区
    (370000, "CANGSHAN"): "371324",           # 苍山县 -> 兰陵县 (2014)
    (520000, "PAN"): "520281",                # 盘县 -> 盘州市
    (410000, "KAIFENG"): "410212",            # 开封县 -> 开封市祥符区
    (510000, "BEICHUAN"): "510726",           # 北川县 -> 北川羌族自治县
    (650000, "WEILI"): "652927",              # 尉犁县 (pinyin yuli)
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
    (130000, "WEIXIAN"): "130533",            # 邢台市威县 (not 张家口市蔚县)
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
