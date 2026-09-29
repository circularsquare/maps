"""2020 and 2010 census figures for Xinjiang's counties, which the county panel lacks.

The panel (github.com/leiii/census) was assembled from local census bulletins, and
Xinjiang's counties published none, so all 106 of its Xinjiang rows are empty. The
national county book, 《中国人口普查分县资料—2020》, has them, but it is print-only
and the Excel copies sit behind resellers. hongheiku.com carries a per-county page
with a census table (7th / 6th / 5th census: resident, urban, male, female,
households) in the book's layout; its robots.txt allows us.

Checks before use:
  * Summed by prefecture, the 2020 figures must equal the official prefecture
    table in Xinjiang's own census bulletin No. 2 (xinjiang_prefectures_2020.csv)
    to the person. A prefecture that misses stops the script.
  * Hunan's pages were compared with the panel as a sample of the site outside
    Xinjiang (2026-09-29); see the README.

One missing county is derived rather than scraped: 和布克赛尔 has no census table
on its page, and is its prefecture's official total less its siblings.

2010 is kept as printed except for one swap: the site gives 伊宁市 166,261 and
奎屯市 515,082, which would have Yining growing 4.7x in a decade and Kuytun
shrinking by half; the other way round both are ordinary. Nothing official here
checks 2010 at county level, and fetch.py treats 2010 as context only.

Usage:
    python xinjiang_counties.py            # scrapes once into data/china, then matches
Writes scripts/china/xinjiang_counties.csv (county_code, name_cn, pop_2020, pop_2010).
"""
import csv
import html
import io
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data/china"
PANEL = DATA / "census_county_2010-2020_v1.csv"
CACHE = DATA / "hongheiku_xinjiang.csv"
PREFS = HERE / "xinjiang_prefectures_2020.csv"
OUT = HERE / "xinjiang_counties.csv"

BASE = "https://www.hongheiku.com/"
CATEGORY = "xianjirank/xjxjdiqu"

# XPCC cities that the bulletin counts inside the prefecture hosting them. Only
# 石河子, 阿拉尔, 图木舒克 and 五家渠 make up its "自治区直辖县级行政区划" row.
XPCC_HOST = {
    "北屯市": "阿勒泰地区", "铁门关市": "巴音郭楞蒙古自治州",
    "双河市": "博尔塔拉蒙古自治州", "可克达拉市": "伊犁哈萨克自治州直属县市",
    "昆玉市": "和田地区", "胡杨河市": "塔城地区",
}
DIRECT = {"石河子市", "阿拉尔市", "图木舒克市", "五家渠市"}

# The bulletin splits 伊犁哈萨克自治州 into its directly held counties and its two
# prefectures; the panel files all three under their own names already.
PANEL_TO_BULLETIN = {"伊犁哈萨克自治州": "伊犁哈萨克自治州直属县市"}

# Page names that are not the panel's county name.
ALIASES = {"高新技术开发区（新市区）": "新市区", "经济技术开发区（头屯河区）": "头屯河区"}

SWAP_2010 = ("伊宁市", "奎屯市")
DERIVED = "和布克赛尔蒙古自治县"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:  # a 404 past the last listing page is expected
            last = e
            time.sleep(2)
    print(f"  gave up on {url}: {last}")
    return ""


def scrape():
    ids, page = [], 1
    while True:
        h = get(BASE + "category/" + CATEGORY + ("" if page == 1 else f"/page/{page}"))
        new = [i for i in dict.fromkeys(re.findall(r"xjxjdiqu/(\d+)\.html", h))
               if i not in ids]
        if not new:
            break
        ids += new
        page += 1
        time.sleep(1)
    rows = []
    for i in ids:
        h = get(BASE + CATEGORY + f"/{i}.html")
        time.sleep(1)
        title = html.unescape((re.search(r"<title>(.*?)</title>", h, re.S) or [None, ""])[1])
        m = re.search(r"\[([^\]]+)\](.+?)(?:最新|人口)", title)
        rec = {"id": i, "pref": m.group(1) if m else "", "name": m.group(2) if m else title}
        hdr = re.search(r"<td>指标</td>((?:<td>[^<]*</td>)+)", h)
        val = re.search(r"<td>常住人口</td>((?:<td>[^<]*</td>)+)", h)
        if hdr and val:
            for c, v in zip(re.findall(r"<td>([^<]*)</td>", hdr.group(1)),
                            re.findall(r"<td>([^<]*)</td>", val.group(1))):
                if "第七次" in c:
                    rec["pop_2020"] = v
                elif "第六次" in c:
                    rec["pop_2010"] = v
        rows.append(rec)
        print(f"  {rec}")
    pd.DataFrame(rows).to_csv(CACHE, index=False, encoding="utf-8")
    print(f"scraped {len(rows)} pages into {CACHE.name}")


def main():
    if not CACHE.exists():
        scrape()
    hh = pd.read_csv(CACHE, dtype=str)
    hh["name"] = hh["name"].map(lambda s: ALIASES.get(s, s))
    hh = hh[hh["pop_2020"].notna()]

    panel = pd.read_csv(PANEL, dtype={"county_code": str})
    xj = panel[panel["province"] == "新疆维吾尔自治区"][["county_code", "city", "county"]]

    # A panel county matches the page whose name contains it; 焉耆县焉耆回族自治县
    # holds 焉耆回族自治县, 第八师石河子市 holds 石河子市.
    out = []
    for r in xj.itertuples():
        hit = hh[hh["name"].map(lambda n: r.county in n)]
        if len(hit) > 1:
            hit = hit[hit["name"] == r.county]
        if hit.empty:
            # Upgraded since 2020 (沙湾县 is now 沙湾市): try the name without its
            # county-level suffix, but only a single, whole-stem hit.
            stem = re.sub(r"(县|市|区)$", "", r.county)
            hit = hh[hh["name"].map(lambda n: re.sub(r"(县|市|区)$", "", n) == stem)]
        if len(hit) == 1:
            h = hit.iloc[0]
            out.append({"county_code": r.county_code, "city": r.city, "name_cn": r.county,
                        "pop_2020": int(h["pop_2020"]),
                        "pop_2010": int(h["pop_2010"]) if pd.notna(h["pop_2010"]) else None,
                        "page": h["id"]})
        else:
            out.append({"county_code": r.county_code, "city": r.city, "name_cn": r.county,
                        "pop_2020": None, "pop_2010": None, "page": ""})
    df = pd.DataFrame(out)

    a, b = (df["name_cn"] == SWAP_2010[0]), (df["name_cn"] == SWAP_2010[1])
    df.loc[a, "pop_2010"], df.loc[b, "pop_2010"] = \
        df.loc[b, "pop_2010"].values, df.loc[a, "pop_2010"].values

    df["bulletin"] = df["city"].map(lambda c: PANEL_TO_BULLETIN.get(c, c))
    df.loc[df["name_cn"].isin(XPCC_HOST), "bulletin"] = df["name_cn"].map(XPCC_HOST)
    df.loc[df["name_cn"].isin(DIRECT), "bulletin"] = "自治区直辖县级行政区划"
    official = pd.read_csv(PREFS, comment="#").set_index("name_cn")["pop_2020"]

    gap = df[df["pop_2020"].isna()]
    for r in gap.itertuples():
        if r.name_cn != DERIVED:
            raise SystemExit(f"no figure for {r.city} {r.name_cn}")
        rest = df[(df["bulletin"] == r.bulletin) & df["pop_2020"].notna()]["pop_2020"].sum()
        df.loc[r.Index, "pop_2020"] = official[r.bulletin] - rest
        df.loc[r.Index, "page"] = "derived"
        print(f"  {r.name_cn}: {official[r.bulletin] - rest:,.0f}, its prefecture's "
              f"official total less its siblings")

    got = df.groupby("bulletin")["pop_2020"].sum()
    bad = 0
    for name, want in official.items():
        have = got.get(name, 0)
        flag = "" if have == want else "  <-- MISMATCH"
        bad += bool(flag)
        print(f"  {name:<16} sum {have:>11,.0f}  bulletin {want:>11,}{flag}")
    if bad:
        raise SystemExit(f"{bad} prefectures do not match the bulletin; not writing")
    print(f"  all {len(official)} bulletin rows match; Xinjiang {df['pop_2020'].sum():,.0f}")

    ratio = df["pop_2010"] / df["pop_2020"]
    odd = df[(ratio < 0.6) | (ratio > 1.6)]
    for r in odd.itertuples():
        print(f"  2010 note: {r.name_cn} {r.pop_2010:,.0f} -> {r.pop_2020:,.0f}")

    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        fh.write("# Xinjiang county census figures, from hongheiku.com's county pages "
                 "(the panel has none). Built by xinjiang_counties.py.\n")
        fh.write("# 2020 sums to xinjiang_prefectures_2020.csv (the official bulletin) "
                 "exactly in every prefecture. 和布克赛尔 is derived; 伊宁市/奎屯市 2010 "
                 "swapped back.\n")
        df[["county_code", "name_cn", "pop_2020", "pop_2010", "page"]].astype(
            {"pop_2020": "int64", "pop_2010": "Int64"}).to_csv(fh, index=False)
    print(f"wrote {OUT.name}: {len(df)} counties")


if __name__ == "__main__":
    main()
