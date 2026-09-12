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


def _stem_pinyin(stem):
    """Pinyin for a proper-noun stem: syllables joined, initial capital.

    Chinese place names run their syllables together (南阳 -> Nanyang), but an
    ethnonym inside the name is conventionally its own word (回族 -> Huizu), so
    split on those before joining.
    """
    if not stem:
        return ""
    parts = []
    rest = stem
    for marker in ("族",):
        idx = rest.find(marker)
        if idx > 0:
            parts.append(rest[: idx + 1])
            rest = rest[idx + 1 :]
    parts.append(rest)
    words = []
    for part in parts:
        if not part:
            continue
        syl = "".join(lazy_pinyin(part))
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
            return (_stem_pinyin(name[: -len(suffix)]) + " " + english).strip()
    return _stem_pinyin(name)
