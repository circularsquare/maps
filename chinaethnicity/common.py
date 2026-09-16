"""Shared constants: paths, the 31 provinces, and the census's nationality columns in order."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RAW = os.path.join(DATA, "raw", "2020")
WORK = os.path.join(DATA, "work")
GEO = os.path.join(DATA, "geo")
PROC = os.path.join(DATA, "processed")

MAPS = os.path.dirname(HERE)
RELIGIONDOTS = os.path.join(MAPS, "religiondots")
ASPECT_ZIP = os.path.join(MAPS, "data", "asia1m", "china", "aspect_population_total_pop.zip")

# GB/T 2260 two-digit code -> (key, English name, name as the national table prints it)
PROVINCES = {
    "11": ("beijing", "Beijing", "北京"),
    "12": ("tianjin", "Tianjin", "天津"),
    "13": ("hebei", "Hebei", "河北"),
    "14": ("shanxi", "Shanxi", "山西"),
    "15": ("neimenggu", "Inner Mongolia", "内蒙古"),
    "21": ("liaoning", "Liaoning", "辽宁"),
    "22": ("jilin", "Jilin", "吉林"),
    "23": ("heilongjiang", "Heilongjiang", "黑龙江"),
    "31": ("shanghai", "Shanghai", "上海"),
    "32": ("jiangsu", "Jiangsu", "江苏"),
    "33": ("zhejiang", "Zhejiang", "浙江"),
    "34": ("anhui", "Anhui", "安徽"),
    "35": ("fujian", "Fujian", "福建"),
    "36": ("jiangxi", "Jiangxi", "江西"),
    "37": ("shandong", "Shandong", "山东"),
    "41": ("henan", "Henan", "河南"),
    "42": ("hubei", "Hubei", "湖北"),
    "43": ("hunan", "Hunan", "湖南"),
    "44": ("guangdong", "Guangdong", "广东"),
    "45": ("guangxi", "Guangxi", "广西"),
    "46": ("hainan", "Hainan", "海南"),
    "50": ("chongqing", "Chongqing", "重庆"),
    "51": ("sichuan", "Sichuan", "四川"),
    "52": ("guizhou", "Guizhou", "贵州"),
    "53": ("yunnan", "Yunnan", "云南"),
    "54": ("xizang", "Tibet", "西藏"),
    "61": ("shaanxi", "Shaanxi", "陕西"),
    "62": ("gansu", "Gansu", "甘肃"),
    "63": ("qinghai", "Qinghai", "青海"),
    "64": ("ningxia", "Ningxia", "宁夏"),
    "65": ("xinjiang", "Xinjiang", "新疆"),
}
CODE_OF = {v[0]: k for k, v in PROVINCES.items()}

# The census's own column order (GB 3304), after the unit total. Checked against every
# table's header by parse.py, so a table that ever reorders them fails loudly.
# (key, English, Chinese). The last two are not nationalities: people whose nationality was
# not determined, and foreign nationals naturalised as Chinese citizens.
GROUPS = [
    ("han", "Han", "汉族"),
    ("mongol", "Mongol", "蒙古族"),
    ("hui", "Hui", "回族"),
    ("tibetan", "Tibetan", "藏族"),
    ("uyghur", "Uyghur", "维吾尔族"),
    ("miao", "Miao", "苗族"),
    ("yi", "Yi", "彝族"),
    ("zhuang", "Zhuang", "壮族"),
    ("bouyei", "Bouyei", "布依族"),
    ("korean", "Korean", "朝鲜族"),
    ("manchu", "Manchu", "满族"),
    ("dong", "Dong", "侗族"),
    ("yao", "Yao", "瑶族"),
    ("bai", "Bai", "白族"),
    ("tujia", "Tujia", "土家族"),
    ("hani", "Hani", "哈尼族"),
    ("kazakh", "Kazakh", "哈萨克族"),
    ("dai", "Dai", "傣族"),
    ("li", "Li", "黎族"),
    ("lisu", "Lisu", "傈僳族"),
    ("wa", "Wa", "佤族"),
    ("she", "She", "畲族"),
    ("gaoshan", "Gaoshan", "高山族"),
    ("lahu", "Lahu", "拉祜族"),
    ("sui", "Sui", "水族"),
    ("dongxiang", "Dongxiang", "东乡族"),
    ("naxi", "Naxi", "纳西族"),
    ("jingpo", "Jingpo", "景颇族"),
    ("kyrgyz", "Kyrgyz", "柯尔克孜族"),
    ("tu", "Tu", "土族"),
    ("daur", "Daur", "达斡尔族"),
    ("mulao", "Mulao", "仫佬族"),
    ("qiang", "Qiang", "羌族"),
    ("blang", "Blang", "布朗族"),
    ("salar", "Salar", "撒拉族"),
    ("maonan", "Maonan", "毛南族"),
    ("gelao", "Gelao", "仡佬族"),
    ("xibe", "Xibe", "锡伯族"),
    ("achang", "Achang", "阿昌族"),
    ("pumi", "Pumi", "普米族"),
    ("tajik", "Tajik", "塔吉克族"),
    ("nu", "Nu", "怒族"),
    ("uzbek", "Uzbek", "乌孜别克族"),
    ("russian", "Russian", "俄罗斯族"),
    ("evenki", "Evenki", "鄂温克族"),
    ("deang", "De'ang", "德昂族"),
    ("bonan", "Bonan", "保安族"),
    ("yugur", "Yugur", "裕固族"),
    ("gin", "Gin", "京族"),
    ("tatar", "Tatar", "塔塔尔族"),
    ("derung", "Derung", "独龙族"),
    ("oroqen", "Oroqen", "鄂伦春族"),
    ("hezhen", "Hezhen", "赫哲族"),
    ("monpa", "Monpa", "门巴族"),
    ("lhoba", "Lhoba", "珞巴族"),
    ("jino", "Jino", "基诺族"),
    ("undetermined", "Not determined", "未定族称人口"),
    ("naturalised", "Naturalised", "入籍"),
]
KEYS = [g[0] for g in GROUPS]
NCAT = 1 + len(GROUPS)          # the unit total, then 58 columns


def norm(s):
    """Drop every kind of whitespace, including the full-width spaces tables pad with."""
    return "".join(str(s).split()) if s is not None else ""
