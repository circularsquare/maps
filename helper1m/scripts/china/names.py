"""Romanise Chinese admin-division names.

The township shapefile carries only Chinese names (省/市/县/乡). The viewer is
read by an English speaker, so every unit gets a Latin name: pinyin for the
proper-noun stem plus the English word for the division type.

    河南省          -> Henan Province
    南阳市          -> Nanyang City
    桐柏县          -> Tongbai County
    吴城镇          -> Wucheng Town
    陈州回族街道     -> Chenzhou Huizu Subdistrict

The Chinese original is kept alongside, so nothing here is load-bearing for
joins — it is display only.

Known limit: a character with two readings gets pypinyin's guess, and for place
names that guess is sometimes wrong — 长阳 comes out Zhangyang, not Changyang.
Fixing that means a hand list of place readings; it has not been worth it.
"""
from pypinyin import lazy_pinyin

# Division-type suffixes, longest first so 自治州 wins over 州.
SUFFIXES = [
    ("街道办事处", "Subdistrict"),
    ("特别行政区", "Special Administrative Region"),
    ("自治区", "Autonomous Region"),
    ("自治州", "Autonomous Prefecture"),
    ("自治县", "Autonomous County"),
    ("自治旗", "Autonomous Banner"),
    ("管委会", "Management Committee"),
    ("民族乡", "Ethnic Township"),
    ("民族镇", "Ethnic Town"),
    ("开发区", "Development Zone"),
    ("街道", "Subdistrict"),
    ("地区", "Prefecture"),
    ("新区", "New Area"),
    ("苏木", "Sumu"),
    ("嘎查", "Gacha"),
    ("农场", "Farm"),
    ("林场", "Forestry Station"),
    ("牧场", "Pasture"),
    ("林区", "Forest District"),
    ("省", "Province"),
    ("市", "City"),
    ("县", "County"),
    ("区", "District"),
    ("旗", "Banner"),
    ("盟", "League"),
    ("镇", "Town"),
    ("乡", "Township"),
]

# The four province-level cities are municipalities, not cities.
MUNICIPALITIES = {"北京市", "天津市", "上海市", "重庆市"}

# The 55 minority ethnonyms as they appear inside division names, without 族.
# Longest first, so 土家 is tried before 土 and 哈萨克 before anything shorter.
ETHNONYMS = sorted([
    "蒙古", "回", "藏", "维吾尔", "苗", "彝", "壮", "布依", "朝鲜", "满", "侗",
    "瑶", "白", "土家", "哈尼", "哈萨克", "傣", "黎", "傈僳", "佤", "畲", "高山",
    "拉祜", "水", "东乡", "纳西", "景颇", "柯尔克孜", "土", "达斡尔", "仫佬", "羌",
    "布朗", "撒拉", "毛南", "仡佬", "锡伯", "阿昌", "普米", "塔吉克", "怒",
    "乌孜别克", "俄罗斯", "鄂温克", "德昂", "保安", "裕固", "京", "塔塔尔", "独龙",
    "鄂伦春", "赫哲", "门巴", "珞巴", "基诺",
], key=len, reverse=True)

# Suffixes of units named for an ethnic group, where the 族 is sometimes dropped.
AUTONOMOUS = {"自治区", "自治州", "自治县", "自治旗", "民族乡", "民族镇"}


def _split_ethnonyms(stem, autonomous):
    """Break a stem into place-name and ethnonym pieces.

    An ethnonym followed by 族 is always its own piece: 陈州回族 is 陈州 / 回族,
    and 恩施土家族苗族 is 恩施 / 土家族 / 苗族. Autonomous units also drop the 族
    for some groups (新疆维吾尔, 伊犁哈萨克), so for those a trailing ethnonym of
    two or more characters is split off as well. Ordinary names are left alone,
    or 丽水 would lose its 水 to the Shui.
    """
    pieces = []
    start = 0
    for i, ch in enumerate(stem):
        if ch != "族":
            continue
        for e in ETHNONYMS:
            j = i - len(e)
            if j >= start and stem[j:i] == e:
                if j > start:
                    pieces.append(stem[start:j])
                pieces.append(stem[j:i + 1])
                start = i + 1
                break
    rest = stem[start:]
    if autonomous and rest:
        for e in ETHNONYMS:
            if len(e) >= 2 and len(rest) > len(e) and rest.endswith(e):
                pieces.append(rest[:-len(e)])
                rest = e
                break
    if rest:
        pieces.append(rest)
    return pieces


def _stem_pinyin(stem, autonomous=False):
    """Pinyin for a proper-noun stem, one capitalised word per piece.

    Chinese place names run their syllables together (南阳 -> Nanyang), but an
    ethnonym inside the name is conventionally its own word (回族 -> Huizu).
    """
    words = []
    for piece in _split_ethnonyms(stem, autonomous):
        syl = "".join(lazy_pinyin(piece))
        if syl:
            words.append(syl[0].upper() + syl[1:])
    return " ".join(words)


def romanise(name, level):
    """Latin name for one division. level is 1..4, used only for municipalities."""
    name = (name or "").strip()
    if not name:
        return ""
    if level == 1 and name in MUNICIPALITIES:
        return _stem_pinyin(name[:-1]) + " Municipality"
    for suffix, english in SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            stem = _stem_pinyin(name[: -len(suffix)], suffix in AUTONOMOUS)
            return (stem + " " + english).strip()
    return _stem_pinyin(name)
