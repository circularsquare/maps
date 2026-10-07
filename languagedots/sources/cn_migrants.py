"""China: residents registered in another province, by county and province of registration.

    python sources/cn_migrants.py --fetch     download the tables into data/raw/cn/migration/
    python sources/cn_migrants.py             parse -> data/normalized/cn_migrants.csv

Anita, 2026-10-06: "for china, is there internal migration data? would that be usable for mixing
dialects?" countries/cn.py draws each county's Chinese speakers on the county's own dialect group(s),
so Shenzhen, Dongguan or Shanghai drew as if every resident spoke the local dialect. This file gives,
for every county polygon, how many of its 2020 census residents are registered (hukou) in another
province, and which provinces; countries/cn.py moves that share of the county's Chinese speakers
onto the origin provinces' own dialect mix. Intra-province migrants stay on the local mix.

THE TABLES (2020 census, short form, every resident): table 1-3 "各地区分性别的户口登记地在外乡镇
街道的人口状况" (residents registered elsewhere: this county / another county of the province / another
province, "省外") and table 7-3 "按现住地、性别分的户口登记地在外省的人口" (residents registered in
another province, by that province). The national yearbook has both by province; provincial
yearbooks have them by county or prefecture. Like table 1-4 (chinaethnicity/fetch.py), the yearbook
index lists JPGs, and an .xls sits beside each under the same name.

GRAIN, finest first (the `grain` column):
  county     the province's own 1-3 or 7-3 by county; rows joined to polygons through
             chinaethnicity's leaves_2020.csv, the same table rows of the same yearbooks with their
             adcode splits (development zones folded as chinaethnicity folds them). Beijing,
             Jilin, Heilongjiang, Shanghai, Jiangsu, Fujian, Shandong, Henan, Hubei, Guangxi,
             Yunnan, Qinghai. Origins by county where 7-3 is by county; Jilin and Henan by
             prefecture; Guangxi by province.
  district   Shenzhen (深圳市人口普查年鉴-2020, a PDF, table 7-3 by district and origin) and
             Guangzhou (广州市人口普查年鉴-2020, xls, table 7-3 by district; volume 1 kept for its 1-3).
  prefecture Hebei, Liaoning, Hunan, Sichuan (their yearbooks stop at prefectures); Shanxi and
             13 Guangdong cities from each city's census bulletin (第七次全国人口普查公报, the
             "外省流入人口" figure, preliminary counts) read on tjgb.hongheiku.com; Zhejiang's 11
             cities from the provincial bureau's analysis (浙江省第七次人口普查系列分析之七, table
             7-4, in 万人). Inside a prefecture every county gets the prefecture's share.
  residual   Guangdong's six cities with no figure (Foshan, Zhuhai, Jiangmen, Shaoguan, Maoming,
             Shanwei): the province's 29,622,110 less every measured city. Maoming, Shaoguan and
             Shanwei, outside the delta, take the share of the nine measured non-delta cities;
             Foshan takes 3,000,000 (Yicai, from the 2020 county book: one of ten cities over 3 million); Zhuhai and Jiangmen split the rest by population.
  province   everywhere else (Tianjin, Inner Mongolia, Anhui, Jiangxi, Hainan, Chongqing,
             Guizhou, Tibet, Shaanxi, Gansu, Ningxia, Xinjiang): the national table's province
             figure, the same share in every county.
Origins: the finest 7-3 that covers the unit; Guangdong's non-district cities take Guangdong's
national row less Shenzhen's and Guangzhou's.

CHECKS: a county-grain province's rows sum to the national table's province figure (exactly, or
the difference is printed); Shanxi's 11 bulletins sum to Shanxi's national figure; every leaf of a
county-grain province is matched once; every unit's origins sum to its count.

Output: data/normalized/cn_migrants.csv  unit, origin (2-digit province code), count, grain.
Counts are people registered in another province, all nationalities; `base` is the unit's population
(cn.csv's, the 2020 census in every province since sources/cn_totals.py); countries/cn.py turns them
into a share of the county's census population.
"""
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import pandas as pd  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "cn" / "migration"
CE = HERE.parent / "chinaethnicity"
LEAVES = CE / "data" / "work" / "leaves_2020.csv"
COUNTIES = CE / "data" / "geo" / "counties.gpkg"
HUBEI_ZIP = CE / "data" / "raw" / "2020" / "hubei_2020.zip"
YUNNAN_DIR = CE / "data" / "raw" / "2020" / "yunnan_2020" / "excel"
NORM = HERE / "data" / "normalized" / "cn.csv"
OUT = HERE / "data" / "normalized" / "cn_migrants.csv"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
WB = "https://web.archive.org/web/{ts}if_/{url}"
NBS = "https://www.stats.gov.cn/sj/pcsj/rkpc/7rp/zk/html/"
GX = ("http://tjj.gxzf.gov.cn//tjsj/tjsj_ztsj/material/2020%E5%B9%B4%E5%B9%BF%E8%A5%BF%E4%BA%BA"
      "%E5%8F%A3%E6%99%AE%E6%9F%A5%E5%B9%B4%E9%89%B4/zk/html/")
FILES = {   # local name: url
    "national_103.xls": NBS + "A0103.xls",
    "national_703.xls": NBS + "A0703.xls",
    "beijing_703.xls": "https://nj.tjj.beijing.gov.cn/tjnj/rkpc-2020/e/zk/html/A0703.xls",
    "jilin_103.xls": "https://tjj.jl.gov.cn/tjsj/qwfb/jlsdqcqgrkpcnj/zk/html/A0103.xls",
    "jilin_703.xls": "https://tjj.jl.gov.cn/tjsj/qwfb/jlsdqcqgrkpcnj/zk/html/A0703.xls",
    "heilongjiang_703.xls": "https://tjj.hlj.gov.cn/tjjnianjian/2020rkpc/zk/html/A0703.xls",
    "shanghai_703.xls": "https://tjj.sh.gov.cn/tjnj/2020rktjnj/ANJ-7-03.xls",
    "jiangsu_703.xls": "https://tj.jiangsu.gov.cn/2020pcnj/zk/html/A0703.xls",
    "fujian_703.xls": "https://tjj.fujian.gov.cn/tongjinianjian/rk2020/html/a0703.xls",
    "shandong_703.xls": "http://tjj.shandong.gov.cn/pcsj/2020/zk/html/A0703.xls",
    "henan_103.xls": ("https://oss.henan.gov.cn/sbgt-wztipt/attachment/hntjj/hntj/lib/tjnj/"
                      "2020hnsrkpcnj/zk/html/A0103.xls"),
    "henan_703.xls": ("https://oss.henan.gov.cn/sbgt-wztipt/attachment/hntjj/hntj/lib/tjnj/"
                      "2020hnsrkpcnj/zk/html/A0703.xls"),
    "guangxi_103.xls": GX + "A1-03.xls",
    "qinghai_703.xls": WB.format(ts="20250216151054", url="http://tjj.qinghai.gov.cn/nj/rkpc/zk/html/A0703.xls"),
    "hebei_703.xls": "https://tjj.hebei.gov.cn/extra/col20/rkpc2020/zk/html/A0703.xls",
    "liaoning_703.xls": WB.format(ts="20250226002507",
                                  url="https://tjj.ln.gov.cn/tjj/tjxx/pcsj/people/pczl/zk/html/A0703.xls"),
    "hunan_103.xls": "http://222.240.193.190/2020rkpcnj/html/A1-03.xls",
    "hunan_703.xls": "http://222.240.193.190/2020rkpcnj/html/A7-3.xls",
    "sichuan_703.xls": "https://tjj.sc.gov.cn/scstjj/rkpcnew/2020/zk/html/A0703.xls",
    # 深圳市人口普查年鉴-2020 (one PDF, 795 pages; table 7-3 on PDF pages 614-661)
    "shenzhen_2020.pdf": "http://www.sz.gov.cn/attachment/1/1540/1540105/10688161.pdf",
    # 广州市人口普查年鉴-2020, volumes 1 (概要) and 7 (户口登记状况) as zips of xls
    "guangzhou_v1.zip": "http://tjj.gz.gov.cn/attachment/7/7209/7209547/8706327.zip",
    "guangzhou_v7.zip": "http://tjj.gz.gov.cn/attachment/7/7209/7209543/8706327.zip",
}
MAGIC = {".xls": b"\xd0\xcf\x11\xe0", ".pdf": b"%PDF", ".zip": b"PK\x03\x04"}

PROV = {"北京": "11", "天津": "12", "河北": "13", "山西": "14", "内蒙古": "15", "辽宁": "21",
        "吉林": "22", "黑龙江": "23", "上海": "31", "江苏": "32", "浙江": "33", "安徽": "34",
        "福建": "35", "江西": "36", "山东": "37", "河南": "41", "湖北": "42", "湖南": "43",
        "广东": "44", "广西": "45", "海南": "46", "重庆": "50", "四川": "51", "贵州": "52",
        "云南": "53", "西藏": "54", "陕西": "61", "甘肃": "62", "青海": "63", "宁夏": "64",
        "新疆": "65"}

# County-grain provinces: code -> (table giving the count, table giving origins, origin grain)
COUNTY = {
    "11": ("beijing_703.xls", "beijing_703.xls"),
    "22": ("jilin_103.xls", "jilin_703.xls"),        # origins by prefecture
    "23": ("heilongjiang_703.xls", "heilongjiang_703.xls"),
    "31": ("shanghai_703.xls", "shanghai_703.xls"),
    "32": ("jiangsu_703.xls", "jiangsu_703.xls"),
    "35": ("fujian_703.xls", "fujian_703.xls"),
    "37": ("shandong_703.xls", "shandong_703.xls"),
    "41": ("henan_103.xls", "henan_703.xls"),        # origins by prefecture
    "42": ("hubei:7-3", "hubei:7-3"),
    "45": ("guangxi_103.xls", None),                 # origins: Guangxi's national row
    "53": ("yunnan:a7-3", "yunnan:a7-3"),
    "63": ("qinghai_703.xls", "qinghai_703.xls"),
}
PREFECTURE = {"13": "hebei_703.xls", "21": "liaoning_703.xls", "43": "hunan_703.xls",
              "51": "sichuan_703.xls"}
# Liaoning prints the 沈抚新区 (between Shenyang and Fushun) as its own row; added to Fushun.
PREF_ALIAS = {"辽宁省沈抚新区管委会": "抚顺市", "湘西州": "湘西土家族苗族自治州"}

# City census bulletins (第七次全国人口普查公报, 流动人口 section, "外省流入人口为 N 人"),
# preliminary counts, read on tjgb.hongheiku.com (the bulletin's own text). city code -> (N, page)
BULLETIN = {   # city code: (外省流入人口, page id on tjgb.hongheiku.com/<id>.html)
    "441900": (6193503, 10915), "442000": (1930002, 10494), "441300": (1614182, 10418),   # 东莞 中山 惠州
    "440500": (415360, 10925), "441200": (286607, 11199), "441800": (227423, 10682),    # 汕头 肇庆 清远
    "445100": (175141, 10954), "441600": (151161, 11074), "440800": (136071, 11547),    # 潮州 河源 湛江
    "441700": (110274, 11537),   # 阳江, worded 跨省流动人口
    "445200": (106067, 10556), "441400": (88806, 11536), "445300": (68221, 10892),      # 揭阳 梅州 云浮
    "140100": (556261, 11800), "140200": (211695, 11870), "140300": (80161, 11720),
    "140400": (123889, 14520), "140500": (99472, 12024), "140600": (91080, 11980),
    "140700": (123946, 11751), "140800": (93972, 11796), "140900": (75379, 12189),
    "141000": (91226, 12209), "141100": (73437, 12297),
}
# Zhejiang, 浙江省第七次人口普查系列分析之七：流动人口 (tjj.zj.gov.cn, 2022-07-22), table 7-4,
# 2020 省外流入人口 by city, 万人
ZHEJIANG = {"330100": 320.50, "330200": 313.27, "330300": 229.43, "330400": 172.43,
            "330500": 76.23, "330600": 105.87, "330700": 218.57, "330800": 10.45,
            "330900": 23.20, "331000": 131.73, "331100": 16.96}
ZHEJIANG_URL = "https://tjj.zj.gov.cn/art/2022/7/22/art_1229129214_4956222.html"
GD_RESIDUAL_DELTA = ["440600", "440400", "440700"]       # Foshan, Zhuhai, Jiangmen
FOSHAN_FLOOR = 3_000_000
GD_RESIDUAL_OUTER = ["440900", "440200", "441500"]       # Maoming, Shaoguan, Shanwei
GD_NONDELTA_MEASURED = ["440500", "440800", "441400", "441600", "441700", "441800", "445100",
                        "445200", "445300"]
SZ = {"罗湖区": "440303", "福田区": "440304", "南山区": "440305", "宝安区": "440306",
      "龙岗区": "440307", "盐田区": "440308", "龙华区": "440309", "坪山区": "440310",
      "光明区": "440311", "大鹏新区": "440307"}   # 大鹏新区 is carved from 龙岗 (no polygon of its own)
# 深汕特别合作区 (7,557 people from other provinces) sits in Shanwei; left in Shanwei's residual.


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for fn, url in FILES.items():
        path = RAW / fn
        ext = path.suffix
        if path.exists() and path.read_bytes()[:4] == MAGIC[ext]:
            continue
        subprocess.run(["curl", "-s", "-k", "-L", "--max-time", "300", "-A", UA, "-o", str(path), url],
                       check=False)
        ok = path.exists() and path.read_bytes()[:4] == MAGIC[ext]
        print(f"  {fn}: {'ok' if ok else 'FAILED'}  {url}")


def clean(v):
    s = re.sub(r"[\s　]+", "", str(v))
    return re.sub(r"[①②③④⑤]", "", s) if s != "nan" else ""


def num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def read_table(src):
    """src: a file in RAW, or 'hubei:<prefix>' / 'yunnan:<prefix>' for chinaethnicity's copies."""
    if src.startswith("hubei:"):
        z = zipfile.ZipFile(HUBEI_ZIP)
        for i in z.infolist():
            n = i.filename.encode("cp437").decode("gbk", "replace")
            if "/短表/" in n and n.rsplit("/", 1)[-1].startswith(src[6:] + " "):
                return pd.read_excel(io.BytesIO(z.read(i)), header=None)
        raise SystemExit(f"{src}: not in the Hubei zip")
    if src.startswith("yunnan:"):
        for f in YUNNAN_DIR.iterdir():
            if f.name.startswith(src[7:] + "　") or f.name.startswith(src[7:] + " "):
                return pd.read_excel(f, header=None)
        raise SystemExit(f"{src}: not in the Yunnan folder")
    return pd.read_excel(RAW / src, header=None)


def parse_rows(df):
    """-> list of (name, {"total": n, "<prov code>": n, ...}) for a 1-3 or 7-3 sheet.
    1-3: "total" is the 省外 column. 7-3: "total" is the first number, then each province's 小计."""
    head = df.iloc[:10]
    cols, total_col = {}, None
    for r in range(len(head)):
        for c in range(1, df.shape[1]):
            h = clean(head.iat[r, c])
            if h in ("省外", "外省") and total_col is None:
                total_col = c
            for k, code in PROV.items():
                if h.startswith(k) and len(h) <= len(k) + 5 and c not in cols.values():
                    cols.setdefault(code, c)
    is73 = len(cols) >= 20
    first = None
    out = []
    for r in range(len(df)):
        name = clean(df.iat[r, 0])
        if not name or name in ("甲", "乙") or name.startswith(("1-3", "7-3", "地区", "现住地", "分组")):
            continue
        vals = [num(df.iat[r, c]) for c in range(1, df.shape[1])]
        if not any(v is not None for v in vals):
            continue
        if first is None:
            first = r
        if is73:
            d = {"total": next(v for v in vals if v is not None)}
            for code, c in cols.items():
                d[code] = num(df.iat[r, c]) or 0.0
        else:
            if total_col is None:
                raise SystemExit("1-3 sheet without a 省外 column")
            d = {"total": num(df.iat[r, total_col]) or 0.0}
        out.append((name, d))
    return out, is73


def walk(rows, leaves, prov):
    """Match a province table's rows to chinaethnicity's leaves (name, parent) in table order.
    -> {leaf index: values}, {prefecture name: values}"""
    lv = leaves[leaves["prov"] == prov]
    parents = set(lv["parent"].dropna())
    key = {(p if p == p else "", n): i for i, n, p in zip(lv.index, lv["name"], lv["parent"])}
    got, pref, cur, bad, zones = {}, {}, "", [], []
    for name, d in rows:
        if not got and not pref and (name in ("合计", "总计", "全省", "全市", "全国") or name in PROV
                                     or name.rstrip("省市") in PROV):
            continue                      # the province's own row, before any other
        if name == "市辖区" and not d["total"]:
            continue                      # an empty heading row (Henan, 三门峡)
        if name == "省直辖县级行政区划" and name not in parents:
            cur = ""
            continue
        if (cur, name) in key and key[(cur, name)] not in got:
            got[key[(cur, name)]] = d
        elif ("", name) in key and key[("", name)] not in got:     # province-direct (梅河口市)
            got[key[("", name)]] = d
        elif name in parents:
            cur = name
            pref[name] = d
        elif cur and re.search("开发区|新区|管委会|园区|示范区", name):
            zones.append((cur, name, d))     # a zone the nationality table did not print
        else:
            bad.append((cur, name))
    missing = sorted(set(lv.index) - set(got))
    if bad or missing:
        raise SystemExit(f"{prov}: rows matching no leaf {bad[:10]}; leaves no row reached "
                         f"{[(lv.at[i, 'parent'], lv.at[i, 'name']) for i in missing[:10]]}")
    return got, pref, zones


def split_codes(s):
    out = []
    for part in str(s).split(";"):
        code, _, w = part.partition(":")
        out.append((code, float(w) if w else None))
    if any(w is None for _, w in out):
        out = [(c, 1 / len(out)) for c, _ in out]
    return out


def sz_table():
    """Shenzhen yearbook table 7-3, district rows: {district: {prov code: n, "total": n}}."""
    import collections
    import fitz
    names = {k + ("省" if k not in ("北京", "天津", "上海", "重庆", "内蒙古", "广西", "西藏", "宁夏", "新疆") else ""): v
             for k, v in PROV.items()}
    full = {"北京市": "11", "天津市": "12", "上海市": "31", "重庆市": "50", "内蒙古自治区": "15",
            "广西壮族自治区": "45", "西藏自治区": "54", "宁夏回族自治区": "64", "新疆维吾尔自治区": "65"}
    names = {**{k: v for k, v in names.items() if k.endswith("省")}, **full, "合计": "total"}
    d = fitz.open(RAW / "shenzhen_2020.pdf")
    res = collections.defaultdict(dict)
    for pno in range(613, 661):
        w = d[pno].get_text("words")
        labs = sorted([x for x in w if x[4] in names and 90 < x[1] < 105], key=lambda x: x[0])
        rows = collections.defaultdict(list)
        for x in w:
            rows[round(x[1])].append(x)
        for ws in rows.values():
            nums = sorted([x for x in ws if re.fullmatch(r"\d+", x[4])], key=lambda x: x[0])
            nm = [x for x in ws if x[4] in SZ or x[4] == "深圳市"]
            if not nums or len(nm) != 1:
                continue
            if len(nums) != 3 * len(labs):
                raise SystemExit(f"Shenzhen p{pno}: {len(nums)} numbers for {len(labs)} groups")
            for gi, lab in enumerate(labs):
                res[nm[0][4]][names[lab[4]]] = float(nums[3 * gi][4])
    for k, v in res.items():
        if len(v) != 31 or abs(sum(x for c, x in v.items() if c != "total") - v["total"]) > 0:
            raise SystemExit(f"Shenzhen {k}: provinces do not sum to the total")
    return res


def gz_table():
    """Guangzhou yearbook 7-3, district rows (55 sheets, a block of rows by two province groups)."""
    import collections
    z = zipfile.ZipFile(RAW / "guangzhou_v7.zip")
    res = collections.defaultdict(dict)
    for i in z.infolist():
        n = i.filename.encode("cp437").decode("gbk", "replace")
        if not re.search(r"/07-03-\d+\.xls$", n):
            continue
        df = pd.read_excel(io.BytesIO(z.read(i)), header=None)
        cols = {}
        for r in range(6):
            for c in range(1, df.shape[1]):
                h = clean(df.iat[r, c])
                if h == "合计" and r < 5 and "total" not in cols and clean(df.iat[r + 1, c]) == "合计":
                    cols["total"] = c
                for k, code in PROV.items():
                    if h.startswith(k):
                        cols[code] = c
        for r in range(len(df)):
            name = clean(df.iat[r, 0])
            if name == "广州市" or (name.endswith("区") and len(name) <= 4):
                for code, c in cols.items():
                    res[name][code] = num(df.iat[r, c]) or 0.0
    for k, v in res.items():
        if len(v) != 31 or abs(sum(x for c, x in v.items() if c != "total") - v["total"]) > 0.5:
            raise SystemExit(f"Guangzhou {k}: {len(v)} columns, provinces do not sum to the total")
    return res


def main():
    import geopandas as gpd
    poly = gpd.read_file(COUNTIES, ignore_geometry=True)
    poly["adcode"] = poly["adcode"].astype(str)
    poly["city_code"] = poly["city_code"].astype(str)
    leaves = pd.read_csv(LEAVES, dtype=str)
    norm = pd.read_csv(NORM, dtype={"geo_id": str})
    pop = norm.groupby("geo_id")["count"].sum()
    est = norm.groupby("geo_id")["estimated"].max()
    # The population base: cn.csv's county totals. Since 2026-10-06 they are 2020 census figures
    # in all 31 provinces (the 15 estimated ones put onto Dong & Wang's county panel by
    # sources/cn_totals.py); before that the panel was read here for the estimated provinces.
    poly["pop"] = [pop.get(u, 0) for u in poly["adcode"]]
    n_est = int((est == 1).sum())
    city_of = dict(zip(poly["adcode"], poly["city_code"]))

    nat, _ = parse_rows(read_table("national_703.xls"))
    natp = {PROV[k]: d for k, d in ((n, d) for n, d in nat) if k in PROV}
    nat103, _ = parse_rows(read_table("national_103.xls"))
    nat_total = {PROV[n]: d["total"] for n, d in nat103 if n in PROV}
    for p, d in natp.items():
        if abs(d["total"] - nat_total[p]) > 0.5:
            raise SystemExit(f"national 7-3 and 1-3 disagree for {p}")

    def vec(d):
        v = {k: x for k, x in d.items() if k != "total" and x > 0}
        s = sum(v.values())
        return {k: x / s for k, x in v.items()} if s else {}

    rows = []          # unit, origin, count, grain

    def add(unit, count, origins, grain):
        for o, sh in origins.items():
            rows.append((unit, o, count * sh, grain))

    done = set()
    report = []
    # 1. county-grain provinces through the leaves
    for prov, (cnt_src, org_src) in COUNTY.items():
        crow, _ = parse_rows(read_table(cnt_src))
        got, _, zones = walk(crow, leaves, prov)
        if org_src and org_src != cnt_src:
            orow, _ = parse_rows(read_table(org_src))
            pref_names = set(leaves.loc[leaves["prov"] == prov, "parent"].dropna())
            pref_org = {n: vec(d) for n, d in orow[1:]}
            # a province-direct county (梅河口市, 济源市) has a row of its own in the prefecture table
            org = {}
            for i in got:
                par, nm = leaves.at[i, "parent"], leaves.at[i, "name"]
                hit = pref_org.get(par) if par == par else None
                org[i] = hit if hit is not None else pref_org[nm]
            ograin = "county, origins by prefecture"
        elif org_src:
            org = {i: vec(d) for i, d in got.items()}
            ograin = "county"
        else:
            org = {i: vec(natp[prov]) for i in got}
            ograin = "county, origins by province"
        s = 0
        for i, d in got.items():
            for code, w in split_codes(leaves.at[i, "adcodes"]):
                add(code, d["total"] * w, org[i], ograin)
                done.add(code)
            s += d["total"]
        for parent, name, d in zones:
            # spread over the prefecture's polygons by population, origins as the prefecture's
            cc = poly.loc[poly["city"] == parent, "city_code"].iloc[0]
            units = poly[poly["city_code"] == cc]
            pv = vec({k: sum(got[i].get(k, 0) for i in got if leaves.at[i, "parent"] == parent)
                      for k in natp[prov]})
            for u, pp in zip(units["adcode"], units["pop"]):
                add(u, d["total"] * pp / units["pop"].sum(), pv, ograin)
            s += d["total"]
            report.append(f"    {prov}: zone row {parent} {name} ({d['total']:,.0f}) spread over the prefecture")
        report.append(f"  {prov} county: {len(got)} rows, {s:,.0f} against the national {nat_total[prov]:,.0f}")
        if abs(s - nat_total[prov]) > 0.005 * nat_total[prov]:
            raise SystemExit(f"{prov}: county rows do not sum to the national figure")

    def by_prefecture(prov, totals, origins, grain):
        """totals: {city_code: n}; every county of the prefecture gets its share by population."""
        for cc, n in totals.items():
            units = poly[poly["city_code"] == cc]
            if units.empty or units["pop"].sum() == 0:
                raise SystemExit(f"{prov}: prefecture {cc} has no polygons")
            for u, p in zip(units["adcode"], units["pop"]):
                add(u, n * p / units["pop"].sum(), origins[cc] if isinstance(origins, dict) and cc in origins
                    else origins, grain)
                done.add(u)

    cityname = {}
    for cc, nm in zip(poly["city_code"], poly["city"]):
        cityname.setdefault(nm, cc)
    # 2. prefecture-grain yearbooks
    for prov, src in PREFECTURE.items():
        prow, _ = parse_rows(read_table(src))
        tot, org = {}, {}
        for n, d in prow[1:]:
            if prov == "13" and n in ("辛集市", "定州市"):
                continue          # inside the whole 石家庄市 and 保定市 rows (the ① rows leave them out)
            n = PREF_ALIAS.get(n, n)
            cc = cityname.get(n)
            if cc is None or not cc.startswith(prov):
                raise SystemExit(f"{prov}: prefecture row {n!r} matches no polygon city")
            if cc in tot and prov == "13":
                continue          # Hebei prints 石家庄市 and 石家庄市① (without 辛集): keep the first, whole
            tot[cc] = tot.get(cc, 0) + d["total"]
            o = org.setdefault(cc, {})
            for k, x in d.items():
                if k != "total":
                    o[k] = o.get(k, 0) + x
        s = sum(tot.values())
        report.append(f"  {prov} prefecture: {len(tot)} rows, {s:,.0f} against the national {nat_total[prov]:,.0f}")
        if abs(s - nat_total[prov]) > 0.005 * nat_total[prov]:
            raise SystemExit(f"{prov}: prefecture rows do not sum to the national figure")
        missing = set(poly.loc[poly["adcode"].str[:2] == prov, "city_code"]) - set(tot)
        if missing:
            raise SystemExit(f"{prov}: prefectures with no row {sorted(missing)}")
        by_prefecture(prov, tot, {cc: vec(o) for cc, o in org.items()}, "prefecture")

    # 3. Guangdong: Shenzhen and Guangzhou by district, the bulletins, the residual
    sz = sz_table()
    gz = gz_table()
    for name, d in sz.items():
        if name != "深圳市":
            add(SZ[name], d["total"], vec(d), "district")
            done.add(SZ[name])
    gz_units = poly[poly["city_code"] == "440100"]
    gz_code = dict(zip(gz_units["name"], gz_units["adcode"]))
    for name, d in gz.items():
        if name != "广州市":
            add(gz_code[name], d["total"], vec(d), "district")
            done.add(gz_code[name])
    if set(gz_code.values()) - done or set(SZ.values()) - done:
        raise SystemExit("Guangzhou or Shenzhen district with no row")
    gd_rest = {k: natp["44"].get(k, 0) - sz["深圳市"].get(k, 0) - gz["广州市"].get(k, 0)
               for k in natp["44"] if k != "total"}
    gd_vec = vec(gd_rest)
    gd_bul = {cc: n for cc, (n, _) in BULLETIN.items() if cc.startswith("44")}
    resid = nat_total["44"] - sz["深圳市"]["total"] - gz["广州市"]["total"] - sum(gd_bul.values())
    cpop = poly.groupby("city_code")["pop"].sum()
    outer_share = sum(gd_bul[c] for c in GD_NONDELTA_MEASURED) / cpop[GD_NONDELTA_MEASURED].sum()
    tot = dict(gd_bul)
    for cc in GD_RESIDUAL_OUTER:
        tot[cc] = cpop[cc] * outer_share
    left = resid - sum(tot[c] for c in GD_RESIDUAL_OUTER)
    for cc in GD_RESIDUAL_DELTA:
        tot[cc] = left * cpop[cc] / cpop[GD_RESIDUAL_DELTA].sum()
    # 《2020中国人口普查分县资料》 as reported by Yicai (2022-10, yicai.com/news/101583222.html):
    # Foshan is one of the ten cities over 3 million; Zhuhai and Jiangmen split what is left
    if tot["440600"] < FOSHAN_FLOOR:
        tot["440600"] = FOSHAN_FLOOR
        rest = left - FOSHAN_FLOOR
        for cc in ("440400", "440700"):
            tot[cc] = rest * cpop[cc] / cpop[["440400", "440700"]].sum()
    report.append(f"  44: Shenzhen {sz['深圳市']['total']:,.0f}, Guangzhou {gz['广州市']['total']:,.0f}, "
                  f"bulletins {sum(gd_bul.values()):,.0f}, residual {resid:,.0f} (outer cities at "
                  f"{outer_share:.1%}: " + ", ".join(f"{c} {tot[c]:,.0f}" for c in GD_RESIDUAL_OUTER + GD_RESIDUAL_DELTA) + ")")
    if left < 0:
        raise SystemExit("Guangdong residual below zero")
    missing = set(poly.loc[poly["adcode"].str[:2] == "44", "city_code"]) - set(tot) - {"440100", "440300"}
    if missing:
        raise SystemExit(f"Guangdong cities with no figure {sorted(missing)}")
    by_prefecture("44", tot, gd_vec, "prefecture (bulletin or residual)")

    # 4. Shanxi bulletins, Zhejiang's analysis
    sx = {cc: n for cc, (n, _) in BULLETIN.items() if cc.startswith("14")}
    report.append(f"  14 bulletins: {sum(sx.values()):,.0f} against the national {nat_total['14']:,.0f}")
    if abs(sum(sx.values()) - nat_total["14"]) > 0.005 * nat_total["14"]:
        raise SystemExit("Shanxi bulletins do not sum to the national figure")
    by_prefecture("14", sx, vec(natp["14"]), "prefecture (bulletin)")
    zj = {cc: v * 1e4 for cc, v in ZHEJIANG.items()}
    report.append(f"  33 analysis table: {sum(zj.values()):,.0f} against the national {nat_total['33']:,.0f}")
    k = nat_total["33"] / sum(zj.values())
    by_prefecture("33", {cc: v * k for cc, v in zj.items()}, vec(natp["33"]), "prefecture")

    # 5. everything else at the province's own share
    df = pd.DataFrame(rows, columns=["unit", "origin", "count", "grain"])
    for prov in sorted(set(poly["adcode"].str[:2])):
        units = poly[(poly["adcode"].str[:2] == prov) & ~poly["adcode"].isin(done) & (poly["pop"] > 0)]
        if units.empty:
            continue
        if prov in COUNTY or prov in PREFECTURE or prov in ("44", "14", "33"):
            raise SystemExit(f"{prov}: polygons with no migrant row {sorted(units['adcode'])[:10]}")
        share = nat_total[prov] / poly.loc[poly["adcode"].str[:2] == prov, "pop"].sum()
        for u, p in zip(units["adcode"], units["pop"]):
            add(u, p * share, vec(natp[prov]), "province")
        report.append(f"  {prov} province: {share:.1%} in every county")
    df = pd.DataFrame(rows, columns=["unit", "origin", "count", "grain"])
    df = df.groupby(["unit", "origin", "grain"], as_index=False)["count"].sum()
    df["base"] = df["unit"].map(dict(zip(poly["adcode"], poly["pop"])))
    report.append(f"  population base: cn.csv's county totals (2020 census; {n_est} counties in the "
                  f"estimated provinces on the county panel, sources/cn_totals.py)")
    if (df["origin"] == df["unit"].str[:2]).any():
        raise SystemExit("a unit has migrants from its own province")
    extra = set(df["unit"]) - set(poly["adcode"])
    if extra:
        raise SystemExit(f"units that are not polygons: {sorted(extra)}")
    share = df.groupby("unit")["count"].sum() / df.groupby("unit")["base"].first()
    hi = share[share > 0.75].sort_values(ascending=False)
    print("\n".join(report))
    print(f"  {df['unit'].nunique()} units, {df['count'].sum():,.0f} people registered in another "
          f"province (national {sum(nat_total.values()):,.0f})")
    print(f"  units over 75% from other provinces: " + ", ".join(f"{u} {v:.0%}" for u, v in hi.items()))
    df.to_csv(OUT, index=False, encoding="utf-8", float_format="%.2f")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    else:
        main()
