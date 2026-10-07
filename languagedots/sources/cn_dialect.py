"""China: each county's Sinitic dialect group (Language Atlas of China) -> data/normalized/cn_dialect.csv.

    python sources/cn_dialect.py          (the two raw files are already in data/raw/cn/dialect/)

Anita, 2026-10-05: draw China's Han (everyone countries/cn.py puts on Chinese) on the county's main
Sinitic group, "it won't be accurate in terms of quantities but it'll help a lot in terms of
representation". Each county is drawn whole as its main group; minorities of other dialects in a
county vanish. Rows are `derived`.

THE TABLE: dan-qqq/Chinese_dialect_distance (GitHub), data/CH_dialect_county_compl.csv: one row per
county-level unit (2,855; modood's administrative codes, about 2018-2019), its 方言大区 / 方言区 /
方言片 (Mandarin or not / dialect group / subgroup), compiled by hand from the Language Atlas of
China, 2nd ed. (中国语言地图集, 2012) and the Chinese Dialect Dictionary (汉语方言大词典, 1991). One
classification per county: the table never marks a county as mixed. Counties whose main language is
not Sinitic (Tibetan, Mongolian, Uyghur...) carry that language instead. No licence file; it is a
compilation of the atlas's published classification (sources/cn.md §6).
    https://raw.githubusercontent.com/dan-qqq/Chinese_dialect_distance/master/data/CH_dialect_county_compl.csv

THE JOIN to chinaethnicity's 2,848 county polygons (the units of data/normalized/cn.csv):
  1. a code in both is joined directly;
  2. a code the table has and the polygons do not is followed through yescallop/areacodes'
     result.csv (县级以上行政区划代码历史, "新代码" = the codes it became) until it reaches polygon codes:
     renamed counties (滦县 -> 滦州市), merged districts (杭州 下城 -> 拱墅), split ones (三亚市 ->
     its four districts). A polygon that receives several old counties with different groups is
     split evenly between them (one share per old county; the record lists them);
  3. a polygon still without a Sinitic group (a county whose main language is not Sinitic, so its
     Han minority's dialect is not given) takes the Han-weighted main group of the other counties in
     its prefecture, failing that its province; Tibet, with no Sinitic county at all, takes
     Southwestern Mandarin (sources/cn.md §6).
Asserted: every polygon gets shares summing to 1; every table code is used or is a known non-unit
(Kinmen, which has no people).
"""
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "cn" / "dialect"
TABLE = RAW / "CH_dialect_county_compl.csv"
CODES = RAW / "areacodes_result.csv"     # yescallop/areacodes master result.csv
COUNTIES = HERE.parent / "chinaethnicity" / "data" / "geo" / "counties.gpkg"
NORM = HERE / "data" / "normalized" / "cn.csv"
OUT = HERE / "data" / "normalized" / "cn_dialect.csv"

SINITIC = {"Mandarin", "Non-Mandarin"}
TIBET_DEFAULT = ("Mandarin", "Southwestern", "Other")   # most of Tibet's Han are from Sichuan, Chongqing
# Alxa League (内蒙古 1529): the table files its three banners as Mongolian and Inner Mongolia's
# Han-weighted main group is Jin, but Alxa's Chinese is the Yinchuan side's, 兰银官话 银吴片.
FALLBACK_PREF = {"1529": ("Mandarin", "Lanyin", "Yinwu")}
NO_PEOPLE = {"350527"}                                  # Kinmen: in the table, no people in cn.csv
# Polygons carved out of a county after the table's codes (the history lists the parent as still in
# use, so no chain reaches them): they take the parent's row.
CARVED = {
    "330113": "330110",   # 临平区 from 余杭区 (2021)
    "330383": "330327",   # 龙港市 from 苍南县 (2019)
    "360113": "360112",   # 红谷滩区 from 新建区 and 东湖区 (2019); both Gan 昌都片
    "440309": "440306",   # 龙华区 from 宝安区 (2016)
    "440311": "440306",   # 光明区 from 宝安区 (2018)
    "440310": "440307",   # 坪山区 from 龙岗区 (2016)
    "460301": "460300",   # 西沙区 from 三沙市 (2020)
    "232718": "232721",   # 加格达奇区 (大兴安岭); every county of the prefecture is 东北官话 黑松片
    "659010": "654202",   # 胡杨河市 (2019), the XPCC 7th division's land between 乌苏 and 沙湾, both 兰银 北疆片
}
# Rows of the table that are wrong, corrected here (sources/cn.md §12):
# 陆河县 (Shanwei) is filed 粤/勾漏片 (Goulou is a Guangxi-Yulin group, 400 km away); the Language
# Atlas 2nd ed. sets up 客家话 海陆片 as 海丰, 陆丰 and "陆河县, split from 陆丰 in 1988" (Xiong and
# Zhang 2008, 方言 2008(2) p.103), and MCPDict's 陆河 point is 客家話－海陸片.
TABLE_FIX = {"441523": ("Non-Mandarin", "Hakka", "Hailu")}
# Polygons allowed no row: 茫崖市 and 大柴旦行委 (Haixi), carved from 行委 the table does not list;
# they take the fallback (prefecture, then province).
NO_ROW_OK = {"632803", "632825"}


def successors(codes):
    """old code -> list of codes it became, from areacodes' history (latest record of a code)."""
    h = pd.read_csv(codes, dtype=str, encoding="utf-8-sig").fillna("")
    # only changes after 2010: the table's codes are about 2018's, and a code reused since an
    # older change must not be sent down that old record
    names = dict(zip(h.sort_values("启用时间")["代码"], h.sort_values("启用时间")["名称"]))
    out = {}
    for r in h.sort_values("启用时间").itertuples(index=False):
        new = []
        for c in r[8].split(";"):
            if not c:
                continue
            yr = c.split("[")[1].rstrip("]") if "[" in c else r[7]
            if yr and int(yr) >= 2010:
                new.append(c.split("[")[0])
        if new:
            out[r[0]] = new
    return out, names


def main():
    import geopandas as gpd
    poly = gpd.read_file(COUNTIES, ignore_geometry=True)
    poly["adcode"] = poly["adcode"].astype(str)
    units = set(poly["adcode"])
    # chinaethnicity's polygons (DataV) carry a few codes of their own for districts renamed since
    # (孟津区 410306, officially 410308): a code the history leads to that is not a polygon is
    # matched by NAME among the polygons of the same prefecture
    by_name = {(r.adcode[:4], r.name): r.adcode for r in poly.itertuples()}
    t = pd.read_csv(TABLE, dtype={"AdCode": str})
    if t["AdCode"].duplicated().any():
        raise SystemExit("table: duplicate codes")
    t["cls"] = list(zip(t["SGroup"], t["DiaGroup"], t["SubDiaGroup"]))
    cls = dict(zip(t["AdCode"], t["cls"]))
    for code, c in TABLE_FIX.items():
        if code not in cls:
            raise SystemExit(f"TABLE_FIX {code}: not in the table")
        print(f"  table fix {code}: {cls[code]} -> {c}")
        cls[code] = c
    succ, names = successors(CODES)
    by_name_hits = []

    def resolve(code, depth=0):
        if code in units:
            return [code]
        if depth > 6:
            return []
        if code not in succ:
            hit = by_name.get((code[:4], names.get(code)))
            if hit:
                by_name_hits.append((code, names.get(code), hit))
                return [hit]
            return []
        got = []
        for c in succ[code]:
            got += resolve(c, depth + 1)
        return got

    # unit -> list of (old code, classification); one share per old code
    got, unused = {}, []
    for code, c in cls.items():
        tgt = sorted(set(resolve(code)))
        if not tgt:
            if code not in NO_PEOPLE:
                unused.append(code)
            continue
        for u in tgt:
            got.setdefault(u, []).append((code, c))
    if unused:
        raise SystemExit(f"table codes that reach no polygon: {unused}")
    for u, parent in CARVED.items():
        if u in got or u not in units:
            raise SystemExit(f"CARVED {u}: already reached, or not a polygon")
        got[u] = [(parent, cls[parent])]
    print(f"  matched by name in the prefecture: {sorted(set(by_name_hits))}")
    # the other direction: a polygon no table row reaches would silently take the fallback
    orphan = sorted(units - set(got) - NO_ROW_OK)
    if orphan:
        raise SystemExit(f"polygons no table row reaches: {orphan}")

    norm = pd.read_csv(NORM, dtype={"geo_id": str})
    han = norm[norm["source_category"] == "han"].groupby("geo_id")["count"].sum()
    rows, merged = [], []
    for u in sorted(units):
        lst = [(code, c) for code, c in got.get(u, []) if c[0] in SINITIC]
        if lst:
            basis = "direct" if [code for code, _ in lst] == [u] else "code history"
            if len({c for _, c in lst}) > 1:
                merged.append((u, lst))
            for code, c in lst:
                rows.append((u, *c, 1 / len(lst), basis, code))
    df = pd.DataFrame(rows, columns=["unit", "sgroup", "group", "subgroup", "share", "basis", "from_code"])

    # fallback for units with no Sinitic group: prefecture, then province, by Han population
    have = set(df["unit"])
    df["han"] = df["unit"].map(han).fillna(0) * df["share"]
    fb = []
    for u in sorted(units - have):
        if u[:4] in FALLBACK_PREF:
            fb.append((u, *FALLBACK_PREF[u[:4]], 1.0, "set by hand", ""))
            continue
        for level, key in (("prefecture", u[:4]), ("province", u[:2])):
            pool = df[df["unit"].str[:len(key)].eq(key) & df["basis"].isin(["direct", "code history"])]
            if len(pool):
                top = pool.groupby(["sgroup", "group", "subgroup"])["han"].sum().idxmax()
                fb.append((u, *top, 1.0, f"{level} main group", ""))
                break
        else:
            if u[:2] != "54":
                raise SystemExit(f"{u}: no Sinitic group in its prefecture or province")
            fb.append((u, *TIBET_DEFAULT, 1.0, "Tibet default", ""))
    df = pd.concat([df.drop(columns="han"),
                    pd.DataFrame(fb, columns=["unit", "sgroup", "group", "subgroup", "share", "basis", "from_code"])],
                   ignore_index=True)

    s = df.groupby("unit")["share"].sum()
    if set(s.index) != units or (s - 1).abs().max() > 1e-9:
        raise SystemExit("shares do not cover every polygon once")
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"  {len(units)} polygons: " + ", ".join(f"{k} {v}" for k, v in
                                                  df.groupby("basis")["unit"].nunique().items()))
    print(f"  {len(merged)} polygons split between old counties of different groups:")
    for u, lst in merged:
        print(f"    {u}: " + "; ".join(f"{code} {c[1]}/{c[2]}" for code, c in lst))
    nos = df[df["basis"].str.contains("main group|default")]
    print(f"  fallback polygons ({len(nos)}), Han {han.reindex(nos['unit']).sum():,.0f}")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
