"""Latin names for the admin units the viewer names on hover.

DataV carries only Chinese names, so every unit needs romanising. Three sources, in order:

1. `data/work/names_wikidata.csv`, English labels pulled from Wikidata by adcode
   (property P442, "administrative division code of China"). This is the only source that
   knows the conventional exonyms: Lhasa, not Lasa; Hohhot, not Huhehaote; Kashgar, not
   Kashi. Built by `python names.py --fetch`; the file is optional and the build says so
   when it is missing.
2. `FIXES` below, for units Wikidata has no English label for or labels badly.
3. pypinyin, with the administrative suffix and any nationality in the name translated:
   长阳土家族自治县 -> "Changyang Tujia Autonomous County". This is right for Han China
   and plain pinyin everywhere else.

Suffixes are kept rather than dropped, because dropping them merges real places: Xinjiang
has both a Yining City and a Yining County, and the 2000 census romanisation, which drops
them, cannot tell you which row is which.

Usage:
    python names.py --fetch      # refresh data/work/names_wikidata.csv (one SPARQL query)
    python names.py              # romanise every county and print a sample
"""
import argparse
import csv
import os
import re
import subprocess
import sys

from common import GROUPS, WORK

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WIKIDATA = os.path.join(WORK, "names_wikidata.csv")
# rdfs:label rather than the label service, and the length filter inside the query: the
# obvious form (every P442, labels via SERVICE wikibase:label) times out at 60 s.
QUERY = ('SELECT ?code ?itemLabel WHERE { ?item wdt:P442 ?code . '
         'FILTER(STRLEN(REPLACE(?code, " ", "")) IN (2, 4, 6)) '
         '?item rdfs:label ?itemLabel . FILTER(LANG(?itemLabel) = "en") }')
UA = "anitamaps/1.0 (chinaethnicity county names)"

# Longest first: 自治县 must be tried before 县, 地区 before 区.
SUFFIXES = [
    ("特别行政区", "Special Administrative Region"),
    ("市辖区", "Urban Districts"),
    ("自治区", "Autonomous Region"),
    ("自治州", "Autonomous Prefecture"),
    ("自治县", "Autonomous County"),
    ("自治旗", "Autonomous Banner"),
    ("林区", "Forest District"),
    ("矿区", "Mining District"),
    ("新区", "New Area"),
    ("特区", "Special District"),
    ("群岛", "Islands"),
    ("地区", "Prefecture"),
    ("区", "District"),
    ("县", "County"),
    ("市", "City"),
    ("旗", "Banner"),
    ("盟", "League"),
    ("州", "Prefecture"),
]

# 族 forms are safe to strip anywhere in a name; the bare forms only from a unit that is
# already "autonomous something", where the word before the suffix is always the nationality.
# (仡佬 or 水 or 京 on their own would otherwise eat a syllable of an ordinary place name.)
_ETH = {cn: en for _, en, cn in GROUPS}
ETHNONYMS = [(cn, en) for cn, en in _ETH.items()] + [("各族", "Multi-ethnic")]
BARE = [(cn[:-1], en) for cn, en in _ETH.items()
        if len(cn) >= 3 and cn[:-1] in {"蒙古", "维吾尔", "哈萨克", "柯尔克孜", "俄罗斯"}]
BARE += [("藏", "Tibetan"), ("回", "Hui")]

# Units Wikidata has no usable English label for and pypinyin gets wrong. Add here rather
# than editing the CSV, which `--fetch` overwrites. Most of what Wikidata misses is a county
# promoted to a district since its item was written, and plain pinyin is right for those.
FIXES = {
    "350625": "Changtai District",       # 长泰: chang, not the zhang pypinyin picks
    "540422": "Mainling County",         # 米林, Tibetan
    "540530": "Cona County",             # 错那, Tibetan
    "632825": "Haixi, direct-administered",   # 海西...自治州直辖, a suffix with no English
    "540102": "Chengguan District",      # Wikidata calls 城关区 "Lhasa District", which
                                         # collides with Lhasa the prefecture above it
}


def _pinyin(stem):
    from pypinyin import lazy_pinyin
    syl = [s for s in lazy_pinyin(stem) if s]
    if not syl:
        return stem
    out = syl[0]
    for s in syl[1:]:
        # Pinyin's own rule: a syllable starting with a vowel needs an apostrophe after
        # another syllable, or Xi'an reads as Xian.
        out += ("'" if s[0] in "aoe" else "") + s
    return out[0].upper() + out[1:]


def romanise(name):
    """Chinese unit name -> pinyin with the suffix and any nationality in English."""
    stem, suffix_en = name, ""
    for cn, en in SUFFIXES:
        if name.endswith(cn) and len(name) > len(cn):
            stem, suffix_en = name[:-len(cn)], en
            break
    eth = []
    while True:
        for cn, en in ETHNONYMS:
            if stem.endswith(cn) and len(stem) > len(cn):
                eth.insert(0, en)
                stem = stem[:-len(cn)]
                break
        else:
            break
    if suffix_en.startswith("Autonomous"):
        while True:
            for cn, en in BARE:
                if stem.endswith(cn) and len(stem) > len(cn):
                    eth.insert(0, en)
                    stem = stem[:-len(cn)]
                    break
            else:
                break
    parts = [_pinyin(stem)]
    if eth:
        parts.append(" and ".join(eth))
    if suffix_en:
        parts.append(suffix_en)
    return " ".join(parts)


def _wikidata():
    """adcode -> English label, for the 6-digit codes, or {} if the file is not there."""
    if not os.path.exists(WIKIDATA):
        return {}
    out = {}
    with open(WIKIDATA, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            out[r["code"]] = r["en"]
    return out


def resolve(pairs):
    """[(code, Chinese name), ...] -> {code: English}. FIXES, then Wikidata, then pypinyin.

    Codes may be of any level: Wikidata files a prefecture as "54 01" and this reads it as
    540100, so the same table names a county and the city it sits in.
    """
    wd = _wikidata()
    names, from_wd = {}, 0
    for code, chinese in pairs:
        code = str(code)
        if code in FIXES:
            names[code] = FIXES[code]
        elif code in wd:
            names[code] = wd[code]
            from_wd += 1
        else:
            names[code] = romanise(chinese)
    return names, from_wd


def fetch():
    os.makedirs(WORK, exist_ok=True)
    tmp = WIKIDATA + ".json"
    # curl, not urllib: Python 3.9's certificate bundle rejects several of the hosts this
    # project fetches from, so the whole build shells out (fetch.py's note).
    cmd = ["curl", "-sS", "-G", "https://query.wikidata.org/sparql",
           "--data-urlencode", "query=" + QUERY, "--data-urlencode", "format=json",
           "-H", "User-Agent: " + UA, "--max-time", "300", "-o", tmp]
    subprocess.run(cmd, check=True)
    import json
    with open(tmp, encoding="utf-8") as fh:
        rows = json.load(fh)["results"]["bindings"]
    keep = {}
    for r in rows:
        # Wikidata writes the code in GB/T 2260's spaced form and drops the trailing zeros:
        # "54 25 21" is a county, "54 25" the prefecture around it, "54" the province.
        code = r["code"]["value"].replace(" ", "")
        label = r["itemLabel"]["value"].strip()
        if not re.fullmatch(r"\d{2}|\d{4}|\d{6}", code):
            continue
        code = code.ljust(6, "0")
        if code not in keep or len(label) < len(keep[code]):
            keep[code] = label
    with open(WIKIDATA, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["code", "en"])
        for code in sorted(keep):
            w.writerow([code, keep[code]])
    os.remove(tmp)
    print(f"wrote {WIKIDATA}: {len(keep):,} county-level codes from {len(rows):,} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="refresh the Wikidata labels")
    args = ap.parse_args()
    if args.fetch:
        fetch()
        return
    import json
    from common import RELIGIONDOTS
    src = os.path.join(RELIGIONDOTS, "data", "raw", "cn", "datav", "county_index.json")
    with open(src, encoding="utf-8") as fh:
        index = json.load(fh)
    pairs = dict([(str(r["city_code"]), r["city"]) for r in index]
                 + [(str(r["code"]), r["name"]) for r in index])
    names, from_wd = resolve(pairs.items())
    print(f"{len(names):,} units, {from_wd:,} named by Wikidata, "
          f"{len(names) - from_wd:,} by pypinyin")
    for row in index[::173]:
        print(f"  {row['code']}  {row['name']:<12s} {names[str(row['code'])]}")


if __name__ == "__main__":
    main()
