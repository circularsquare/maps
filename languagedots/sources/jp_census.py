"""Japan, 2020 census: Japanese nationals and foreign residents by nationality, read as languages.
-> data/normalized/jp.csv, taxonomy/tree.d/jp.txt (the borrowed-node block)

    python sources/jp_census.py

Nobody in Japan is asked a language: the 2020 census asks nationality, not language or ethnicity.
Built under Anita's 2026-10-05 ruling for countries with no language question (AGENT_BRIEF.md
§2): the national language, plus (a) regional languages from regional surveys and (b) immigrant
languages proxied by nationality, the way Saudi Arabia was done (sources/sa.md). EVERY ROW IS
`derived`. sources/jp.md is the record; the steps:

  1. Census 2020, table 44-1 (人口等基本集計 第44-1表, e-Stat statInfId 000032142708): population
     by nationality (Japanese, 12 foreign groups and "other", nationality not stated) for all
     1,896 municipalities and wards. Checked: units sum to their prefecture and to Japan in
     every column, wards to their designated city, the nationality columns to each unit's total.
  2. Nationality not stated (2,202,484 people, mostly non-response in big cities) is spread over
     each unit's known nationalities pro rata, so every unit keeps its census total.
  3. Two census groups hold several nationalities. "China" is split into China and Taiwan, and
     "Other" into its ~180 nationalities, by the Immigration Services Agency's resident
     foreigners by prefecture and nationality, December 2020 (在留外国人統計 第4表, e-Stat
     statInfId 000032104295): each municipality takes its prefecture's mix.
  4. Each nationality is drawn on its country's language or, for the larger multilingual origins
     already on this map, on that country's own drawn mix (HOME_MIX, as Saudi Arabia: languages
     of 1%+ kept and scaled to 100%). Brazil Portuguese, Peru and Bolivia Spanish (the migrants
     are Nikkei families from the cities).
  5. Retention: a share of each nationality is drawn as Japanese. Aichi Prefecture's foreign
     residents survey (外国人県民アンケート調査, March 2022, question 25, table 25-2): parents of
     children under 18, "always speak Japanese with the children", by nationality, among those
     who answered. Nationalities with 40+ respondents take their own row (Korea 80.6%, the
     Philippines 45.3%, China 29.5%, Brazil 12.3%, Vietnam 9.1%); everyone else the survey's
     total (29.5%).
  6. Ryukyuan languages. Okinawa: the prefecture's しまくとぅば県民意識調査 2023 and 2024 (18+,
     n = 1,028 and 1,043), "mainly use shimakutuba" plus half of "use it as much as standard
     Japanese", by survey region, the two years pooled by respondents; applied to each
     municipality's Japanese nationals 18+ (census table 3-3, 5-year bands; 18-19 taken as 2/5
     of the 15-19 band); nobody under 18. Each municipality on its traditional language
     (OKINAWA below). The Amami Islands (Kagoshima): no survey; Ethnologue's speaker figures
     (as quoted on Wikipedia, dated 2004) per language, spread over its municipalities by
     Japanese population; Kikai's figure (13,000) exceeds the island's population, so Kikai
     takes the other Amami languages' speakers-to-population ratio.
  7. Ainu is not drawn: the Hokkaido Ainu living conditions survey 2023 found 0.8% of 472
     respondents could hold a conversation in Ainu, all 60+, which is under one dot.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "taxonomy"), str(ROOT)]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RAW = ROOT / "data" / "raw" / "jp"
NORM = ROOT / "data" / "normalized"
OUT = NORM / "jp.csv"
FRAGMENT = ROOT / "taxonomy" / "tree.d" / "jp.txt"
CENSUS = RAW / "census2020_b44_01.xlsx"
AGES_OKINAWA = RAW / "census2020_b03_03_okinawa.xlsx"
ISA = RAW / "isa_2020_12_04_pref_nationality.xlsx"
TOTAL_2020 = 126_146_099

JAPANESE = "japonic.japanese"
RY = "japonic.ryukyuan"
OKINAWAN, KUNIGAMI = f"{RY}.okinawan", f"{RY}.kunigami"
MIYAKO, YAEYAMA, YONAGUNI = f"{RY}.miyako", f"{RY}.yaeyama", f"{RY}.yonaguni"
AMAMI_N, AMAMI_S = f"{RY}.amami_oshima_north", f"{RY}.amami_oshima_south"
KIKAI, TOKUNOSHIMA = f"{RY}.kikai", f"{RY}.tokunoshima"
OKINOERABU, YORON = f"{RY}.okinoerabu", f"{RY}.yoron"
OWN_NODES = [  # (id, label, colour or None): the fragment's hand-made part
    ("japonic", "Japonic", "0.86 0.09 20"),
    (JAPANESE, "Japanese", "0.86 0.09 20"),
    (RY, "Ryukyuan", "0.62 0.15 30"),
    (AMAMI_N, "Northern Amami-Oshima", "0.58 0.16 55"),
    (AMAMI_S, "Southern Amami-Oshima", "0.70 0.13 75"),
    (KIKAI, "Kikai", "0.50 0.14 40"),
    (TOKUNOSHIMA, "Tokunoshima", "0.66 0.15 20"),
    (OKINOERABU, "Okinoerabu", "0.55 0.17 0"),
    (YORON, "Yoron", "0.72 0.12 50"),
    (KUNIGAMI, "Kunigami", "0.55 0.16 30"),
    (OKINAWAN, "Central Okinawan", "0.62 0.19 10"),
    (MIYAKO, "Miyako", "0.56 0.17 350"),
    (YAEYAMA, "Yaeyama", "0.64 0.15 45"),
    (YONAGUNI, "Yonaguni", "0.50 0.16 15"),
]

# ---- census 44-1 columns -----------------------------------------------------------------
CATS = ["101", "102", "103", "104", "105", "106", "107", "108", "109", "110", "111", "112",
        "113"]
CAT_NAME = {"101": "Korea", "102": "China", "103": "Philippines", "104": "Thailand",
            "105": "Indonesia", "106": "Vietnam", "107": "India", "108": "Nepal",
            "109": "United Kingdom", "110": "United States", "111": "Brazil", "112": "Peru",
            "113": "Other"}
# ISO2 codes each census group holds (113 "Other" = everything else in the ISA table)
CAT_ISO = {"101": ["KR", "KP"], "102": ["CN", "TW"], "103": ["PH"], "104": ["TH"],
           "105": ["ID"], "106": ["VN"], "107": ["IN"], "108": ["NP"], "109": ["GB"],
           "110": ["US"], "111": ["BR"], "112": ["PE"]}

# ---- retention: Aichi foreign residents survey, March 2022, Q25, table 25-2 ----------------
# (always Japanese with the children, respondents who answered = N - not stated)
AICHI_Q25 = {"101": (50, 63 - 1), "102": (54, 189 - 6), "103": (53, 120 - 3),
             "106": (5, 59 - 4), "111": (25, 209 - 6)}
AICHI_ALL = (213, 746 - 23)
AICHI_MIN_N = 40

# ---- Okinawa: しまくとぅば県民意識調査, 2023 (R5) and 2024 (R6), Q "how much do you use
# shimakutuba": n, % mainly, % as much as standard Japanese, by survey region ---------------
SHIMA = {  # region: [(n, mainly, equally) for R5, R6]
    "north": [(148, 3.4, 14.2), (145, 3.4, 15.9)],
    "central": [(264, 4.2, 12.5), (300, 2.0, 12.0)],
    "south": [(432, 3.7, 7.6), (422, 2.6, 12.6)],
    "miyako": [(35, 2.9, 28.6), (30, 20.0, 23.3)],
    "yaeyama": [(68, 4.4, 10.3), (62, 0.0, 8.1)],
    "islands": [(66, 4.5, 22.7), (67, 4.5, 16.4)],
}
# municipality -> (survey region, language). Regions are the prefecture's 圏域; "islands" is the
# survey's その他の離島, taken as the remote islands outside Miyako and Yaeyama. Languages:
# Kunigami (Northern Okinawan, Glottolog kuni1268) north of the Onna-Kin line and on Ie, Iheya
# and Izena; Central Okinawan (cent2126) south of it and on Kume, Kerama, Aguni and Tonaki;
# Miyako (miya1259) on Miyako and Tarama; Yaeyama (yaey1239) on Ishigaki and Taketomi; Yonaguni
# (yona1241). The Daito islands, settled from Hachijo and Okinawa in 1900, are left Japanese.
OKINAWA = {
    "名護市": ("north", KUNIGAMI), "国頭村": ("north", KUNIGAMI), "大宜味村": ("north", KUNIGAMI),
    "東村": ("north", KUNIGAMI), "今帰仁村": ("north", KUNIGAMI), "本部町": ("north", KUNIGAMI),
    "恩納村": ("north", OKINAWAN), "宜野座村": ("north", OKINAWAN), "金武町": ("north", OKINAWAN),
    "伊江村": ("islands", KUNIGAMI), "伊平屋村": ("islands", KUNIGAMI),
    "伊是名村": ("islands", KUNIGAMI),
    "宜野湾市": ("central", OKINAWAN), "沖縄市": ("central", OKINAWAN),
    "浦添市": ("central", OKINAWAN), "うるま市": ("central", OKINAWAN),
    "読谷村": ("central", OKINAWAN), "嘉手納町": ("central", OKINAWAN),
    "北谷町": ("central", OKINAWAN), "北中城村": ("central", OKINAWAN),
    "中城村": ("central", OKINAWAN), "西原町": ("central", OKINAWAN),
    "那覇市": ("south", OKINAWAN), "糸満市": ("south", OKINAWAN), "豊見城市": ("south", OKINAWAN),
    "南城市": ("south", OKINAWAN), "与那原町": ("south", OKINAWAN),
    "南風原町": ("south", OKINAWAN), "八重瀬町": ("south", OKINAWAN),
    "久米島町": ("islands", OKINAWAN), "渡嘉敷村": ("islands", OKINAWAN),
    "座間味村": ("islands", OKINAWAN), "粟国村": ("islands", OKINAWAN),
    "渡名喜村": ("islands", OKINAWAN),
    "宮古島市": ("miyako", MIYAKO), "多良間村": ("miyako", MIYAKO),
    "石垣市": ("yaeyama", YAEYAMA), "竹富町": ("yaeyama", YAEYAMA),
    "与那国町": ("yaeyama", YONAGUNI),
}
OKINAWA_JAPANESE = {"南大東村", "北大東村"}

# ---- Amami Islands (Kagoshima): Ethnologue speaker figures, dated 2004 (18th ed., 2015, as
# quoted by Wikipedia's articles on each language), and the municipalities each is spoken in --
AMAMI = {
    AMAMI_N: (10_000, ["奄美市", "龍郷町", "大和村", "宇検村"]),
    AMAMI_S: (1_800, ["瀬戸内町"]),
    TOKUNOSHIMA: (5_100, ["徳之島町", "天城町", "伊仙町"]),
    OKINOERABU: (3_200, ["和泊町", "知名町"]),
    YORON: (950, ["与論町"]),
    KIKAI: (None, ["喜界町"]),   # Ethnologue's 13,000 exceeds the island's population
}

# ---- nationality -> language -------------------------------------------------------------
# origins drawn at their own country's mix on this map (multilingual, 5,000+ residents)
HOME_MIX = ["CN", "TW", "VN", "PH", "NP", "ID", "TH", "IN", "MM", "LK", "PK", "BD", "KH", "US",
            "MY", "CA", "AU", "RU", "TR"]
OVERRIDE = {"BR": "Portuguese", "PE": "Spanish", "BO": "Spanish", "FR": "French",
            "KR": "Korean", "KP": "Korean"}
ISA_NAMES = {  # ISA names babel's Japanese territory names do not match
    "米国": "US", "ミャンマー": "MM", "朝鮮": "KP", "英国": "GB", "南アフリカ共和国": "ZA",
    "無国籍": "XX", "コンゴ民主共和国": "CD", "パレスチナ": "PS", "南スーダン共和国": "SS",
    "ソロモン": "SB", "コンゴ共和国": "CG", "マーシャル": "MH", "ドミニカ": "DM",
    "中央アフリカ": "CF", "コソボ共和国": "XK", "セントクリストファー・ネービス": "KN",
    "セントビンセント": "VC", "セルビア・モンテネグロ": "RS", "ミクロネシア": "FM"}
CONTINENTS = {"総数", "アジア", "ヨーロッパ", "アフリカ", "北米", "南米", "オセアニア"}
MIN_SHARE = 0.01


def _num(s):
    return pd.to_numeric(s.replace("-", 0), errors="raise").astype("int64")


def read_census():
    d = pd.read_excel(CENSUS, header=None, skiprows=10, dtype=str)
    d = d[d[0] == "0_総数"].copy()
    d["level"] = d[1].astype(str)
    d["code"] = d[3].str[:5]
    d["name"] = d[3].str[6:]
    d["pref"] = d[2].str[:2]
    cols = {4: "total", 5: "foreign", 19: "japanese", 20: "unknown"}
    cols.update({6 + i: c for i, c in enumerate(CATS)})
    for k, v in cols.items():
        d[v] = _num(d[k])
    num = list(cols.values())
    nat = d[d["code"] == "00000"].iloc[0]
    if int(nat["total"]) != TOTAL_2020:
        raise SystemExit(f"national total {nat['total']:,} != {TOTAL_2020:,}")
    units = d[d["level"].isin(["0", "2", "3"])].copy()
    if len(units) != 1896:
        raise SystemExit(f"{len(units)} units, expected 1,896")
    # checks: units -> Japan, units -> prefectures, wards -> designated cities, columns
    for c in num:
        if int(units[c].sum()) != int(nat[c]):
            raise SystemExit(f"units' {c} sum to {units[c].sum():,}, Japan prints {nat[c]:,}")
    prefs = d[(d["level"] == "a") & (d["code"] != "00000")].set_index("pref")
    bad = (units.groupby("pref")[num].sum() != prefs[num]).any(axis=1)
    if bad.any():
        raise SystemExit(f"prefectures not matching their units: {list(bad[bad].index)}")
    cities = sorted(d.loc[d["level"] == "1", "code"])
    for _, city in d[d["level"] == "1"].iterrows():
        c = int(city["code"])
        nxt = [int(x) for x in cities if int(x) > c and x[:2] == city["code"][:2]]
        hi = min(nxt + [c + 100])
        w = units[units["code"].astype(int).between(c + 1, hi - 1) & units["code"].isin(
            d.loc[d["level"] == "0", "code"])]
        if int(w["total"].sum()) != int(city["total"]):
            raise SystemExit(f"{city['name']}: wards sum to {w['total'].sum():,}")
    if (units["foreign"] != units[CATS].sum(axis=1)).any() or \
            (units["total"] != units[["foreign", "japanese", "unknown"]].sum(axis=1)).any():
        raise SystemExit("nationality columns do not add up to unit totals")
    return units[["code", "name", "pref", *num]].reset_index(drop=True)


def read_isa():
    """prefecture (2-digit) x ISO2 resident foreigners, December 2020."""
    from babel import Locale
    inv = {v: k for k, v in Locale("ja").territories.items() if len(k) == 2}
    d = pd.read_excel(ISA, header=None)
    names = [str(x) for x in d.iloc[3, 1:]]
    body = d.iloc[5:52, :].reset_index(drop=True)
    if len(body) != 47 or str(body.iat[0, 0]) != "北海道" or str(body.iat[46, 0]) != "沖縄県":
        raise SystemExit("ISA table: prefecture rows not where expected")
    out, nomatch = {}, []
    for j, n in enumerate(names, start=1):
        if n in CONTINENTS:
            continue
        iso = ISA_NAMES.get(n) or inv.get(n)
        if iso is None:
            nomatch.append(n)
            continue
        out[iso] = out.get(iso, 0) + pd.to_numeric(body[j]).to_numpy()
    if nomatch:
        raise SystemExit(f"ISA nationalities with no ISO code: {nomatch}")
    df = pd.DataFrame(out, index=[f"{i:02d}" for i in range(1, 48)])
    total = int(pd.to_numeric(d.iat[4, 1]))
    unknown_pref = int(pd.to_numeric(d.iat[52, 1]))
    if int(df.to_numpy().sum()) + unknown_pref != total:
        raise SystemExit(f"ISA nationalities sum to {df.to_numpy().sum():,} + {unknown_pref:,}, "
                         f"total {total:,}")
    return df


def mix_from(df, col="node"):
    s = df.groupby(col)["count"].sum()
    s = s[s > 0]
    s = s / s.sum()
    s = s[s >= MIN_SHARE]
    return (s / s.sum()).to_dict()


_HOME = {}


def home_mix(cc):
    if cc not in _HOME:
        from countries import load_one
        _HOME[cc] = mix_from(load_one(cc.lower())["counts"]())
    return _HOME[cc]


def origin_mix(iso):
    import fr2023
    from fr_build import COUNTRY_LANG
    if iso == "XX":
        return {"other": 1.0}
    # the shared origin table (sources/origin_mix.py, 2026-10-05): HOME_MIX and OVERRIDE above
    # are kept for the record (OVERRIDE had no figure behind it; Brazil's home mix is
    # Portuguese anyway)
    from origin_mix import mix
    try:
        return mix(iso, "jp")
    except KeyError:
        return {"other": 1.0}
    v = COUNTRY_LANG.get({"GB": "UK", "GR": "EL"}.get(iso, iso))
    if v is None:
        return {"other": 1.0}
    items = [(v, 1.0)] if isinstance(v, str) else list(v.items())
    return {fr2023.NAMES[lab]: s for lab, s in items}


def retention():
    tot = AICHI_ALL[0] / AICHI_ALL[1]
    r = {}
    for c in CATS:
        a = AICHI_Q25.get(c)
        r[c] = a[0] / a[1] if a and a[1] >= AICHI_MIN_N else tot
    return r


def okinawa_adults():
    """{municipality name: Japanese nationals 18+}, census 2020 table 3-3."""
    d = pd.read_excel(AGES_OKINAWA, header=None, skiprows=10, dtype=str)
    d = d[(d[3] == "1_うち日本人") & (d[4] == "0_総数")].copy()
    d["name"] = d[2].str[6:]
    d["band"] = d[5].str[:2]
    d["n"] = _num(d[6])
    out = {}
    for name, g in d.groupby("name"):
        b = dict(zip(g["band"], g["n"]))
        bands = [b[f"{i:02d}"] for i in range(1, 22)]
        unk = b["22"]
        if sum(bands) + unk != b["00"]:
            raise SystemExit(f"{name}: age bands do not sum to the total")
        # 18+ = 20+ and 2/5 of 15-19; age not stated spread pro rata
        adults = (sum(bands[4:]) + 0.4 * bands[3]) * b["00"] / (b["00"] - unk)
        out[name] = adults
    if len(out) != 42:
        raise SystemExit(f"table 3-3: {len(out)} areas, expected Okinawa and its 41 municipalities")
    return out


def shima_rate():
    out = {}
    for reg, waves in SHIMA.items():
        n = sum(w[0] for w in waves)
        out[reg] = sum(w[0] * (w[1] + w[2] / 2) for w in waves) / n / 100
    return out


def ryukyuan(units, f):
    """{(code, node): speakers} for Okinawa and the Amami Islands."""
    out = {}
    ok = units[units["pref"] == "47"].set_index("name")
    names = set(ok.index)
    if names != set(OKINAWA) | OKINAWA_JAPANESE:
        raise SystemExit(f"Okinawa municipalities differ: {sorted(names ^ (set(OKINAWA) | OKINAWA_JAPANESE))}")
    adults = okinawa_adults()
    rate = shima_rate()
    for name, (reg, node) in OKINAWA.items():
        code = ok.at[name, "code"]
        out[(code, node)] = rate[reg] * adults[name] * f[code]
    ka = units[units["pref"] == "46"].set_index("name")
    jp_pop = {}
    for node, (n, munis) in AMAMI.items():
        missing = [m for m in munis if m not in ka.index]
        if missing:
            raise SystemExit(f"Kagoshima has no {missing}")
        jp_pop[node] = {ka.at[m, "code"]: ka.at[m, "japanese"] * f[ka.at[m, "code"]] for m in munis}
    known = [k for k, (n, _) in AMAMI.items() if n]
    ratio = sum(AMAMI[k][0] for k in known) / sum(sum(jp_pop[k].values()) for k in known)
    for node, (n, munis) in AMAMI.items():
        pops = jp_pop[node]
        n = n if n else ratio * sum(pops.values())
        for code, p in pops.items():
            out[(code, node)] = n * p / sum(pops.values())
    return out, rate, ratio


def round_within_rows(m):
    out = np.zeros(m.shape, dtype="int64")
    for i in range(m.shape[0]):
        row = m.iloc[i].to_numpy(dtype=float)
        target = int(round(row.sum()))
        base = np.floor(row).astype("int64")
        short = target - int(base.sum())
        if short:
            base[np.argsort(-(row - base))[:short]] += 1
        out[i] = base
    return pd.DataFrame(out, index=m.index, columns=m.columns)


def write_fragment(nodes):
    """tree.d/jp.txt: Japan's own nodes, then every other node it uses that tree.txt lacks."""
    tdir = ROOT / "taxonomy"
    base = {ln.split("|")[0].strip() for ln in (tdir / "tree.txt").read_text(encoding="utf-8")
            .splitlines() if ln.strip() and not ln.startswith("#")}
    known = {}
    for p in sorted((tdir / "tree.d").glob("*.txt")):
        if p.name == "jp.txt":
            continue
        for ln in p.read_text(encoding="utf-8").splitlines():
            if ln.strip() and not ln.startswith("#") and "|" in ln:
                parts = [x.strip() for x in ln.split("|")]
                known.setdefault(parts[0], parts[1])
    own = {n for n, _, _ in OWN_NODES}
    need = set()
    for n in nodes:
        parts = n.split(".")
        for i in range(1, len(parts) + 1):
            need.add(".".join(parts[:i]))
    borrowed = sorted(need - base - own)
    lost = [n for n in borrowed if n not in known]
    if lost:
        raise SystemExit(f"nodes defined nowhere: {lost}")
    head = [
        "# Japan (taxonomy/jp2020.py). No language question; the 2020 census by nationality, each",
        "# nationality read as a language or a home mix, and the Ryukyuan languages from surveys",
        "# (sources/jp_census.py, sources/jp.md).",
        "#",
        "# Ryukyuan: a group under Japonic (Glottolog splits Northern and Southern Ryukyuan; one",
        "# group is the one readers know). Languages as Glottolog: Kikai kika1239, Northern and",
        "# Southern Amami-Oshima nort2935 sout2954, Tokunoshima and Okinoerabu (both under",
        "# okin1245), Yoron yoro1243, Kunigami kuni1268, Central Okinawan cent2126, Miyako",
        "# miya1259, Yaeyama yaey1239, Yonaguni yona1241. Colours: Japanese stays the family's",
        "# pale shade; the Ryukyuan languages are saturated reds and oranges, neighbours on the",
        "# island chain pushed apart in hue and lightness.",
    ]
    lines = head + [" | ".join(x for x in (n, lab, col) if x) for n, lab, col in OWN_NODES]
    lines += ["#", "# Borrowed nodes, repeated without colour (AGENT_BRIEF §3), written by",
              "# sources/jp_census.py: every node the home mixes and single-language origins bring",
              "# in that tree.txt lacks."]
    lines += [f"{n} | {known[n]}" for n in borrowed]
    FRAGMENT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {FRAGMENT}: {len(OWN_NODES)} own nodes, {len(borrowed)} borrowed")


def main():
    units = read_census()
    isa = read_isa()
    known_cols = ["japanese", *CATS]
    units["f"] = units["total"] / units[known_cols].sum(axis=1)
    f = dict(zip(units["code"], units["f"]))
    print(f"  census: {len(units):,} units, {units['total'].sum():,} people, "
          f"{units['foreign'].sum():,} foreign, {units['unknown'].sum():,} nationality not stated")

    # census group vs ISA nationally (the census counts fewer; ratios printed as a check on
    # which ISA nationalities the census's "China" and "Other" hold)
    other_iso = [c for c in isa.columns if c not in {i for v in CAT_ISO.values() for i in v}]
    for c, isos in CAT_ISO.items():
        print(f"    {CAT_NAME[c]:<15} census {units[c].sum():>9,}  ISA {isa[isos].to_numpy().sum():>9,}"
              f"  ratio {units[c].sum() / isa[isos].to_numpy().sum():.2f}")
    print(f"    {'Other':<15} census {units['113'].sum():>9,}  ISA {isa[other_iso].to_numpy().sum():>9,}"
          f"  ratio {units['113'].sum() / isa[other_iso].to_numpy().sum():.2f}  "
          f"(with Taiwan in Other instead: "
          f"{units['113'].sum() / (isa[other_iso].to_numpy().sum() + isa['TW'].sum()):.2f}, China "
          f"{units['102'].sum() / isa['CN'].sum():.2f})")
    CAT_ISO["113"] = other_iso

    # nationality-group mixes, per prefecture
    ret = retention()
    print("  retention (Japanese share): " + ", ".join(f"{CAT_NAME[c]} {r:.1%}" for c, r in ret.items()))
    origin_cache = {}

    def group_mix(cat, pref):
        isos = CAT_ISO[cat]
        w = isa.loc[pref, isos].astype(float)
        if w.sum() == 0:
            w = isa[isos].sum().astype(float)
        w = w / w.sum()
        m = {JAPANESE: ret[cat]}
        for iso, s in w.items():
            if s == 0:
                continue
            if iso not in origin_cache:
                origin_cache[iso] = origin_mix(iso)
            for n, v in origin_cache[iso].items():
                m[n] = m.get(n, 0.0) + (1 - ret[cat]) * s * v
        return m

    mixes = {(c, p): group_mix(c, p) for c in CATS for p in sorted(units["pref"].unique())}
    for k, m in mixes.items():
        if abs(sum(m.values()) - 1) > 1e-9:
            raise SystemExit(f"{k}: mix sums to {sum(m.values())}")

    ry, rate, ratio = ryukyuan(units, f)
    print("  Okinawa shimakutuba rates (mainly + half equally, R5+R6): "
          + ", ".join(f"{k} {v:.1%}" for k, v in rate.items()))
    print(f"  Amami speakers per Japanese resident (Kikai's ratio): {ratio:.1%}")

    recs = {}
    for r in units.itertuples():
        row = {}
        for c in CATS:
            n = units.at[r.Index, c]
            if n == 0:
                continue
            for node, s in mixes[(c, r.pref)].items():
                key = ("foreign", node) if node != JAPANESE else ("foreign_retained", JAPANESE)
                row[key] = row.get(key, 0.0) + n * r.f * s
        for (code, node), v in ry.items():
            if code == r.code:
                row[("ryukyuan", node)] = v
        jp = r.total - sum(row.values())
        if jp < 0:
            raise SystemExit(f"{r.name}: negative Japanese remainder")
        row[("japanese", JAPANESE)] = row.get(("japanese", JAPANESE), 0.0) + jp
        recs[r.code] = row
    keys = sorted({k for row in recs.values() for k in row})
    m = pd.DataFrame([[recs[c].get(k, 0.0) for k in keys] for c in units["code"]],
                     index=units["code"].to_numpy(), columns=range(len(keys)))
    mc = round_within_rows(m)
    if (mc.sum(axis=1).to_numpy() != units["total"].to_numpy()).any():
        raise SystemExit("rounded units do not sum to the census")
    name = dict(zip(units["code"], units["name"]))
    long = mc.reset_index(names="geo_id").melt(id_vars="geo_id", var_name="k", value_name="count")
    long = long[long["count"] > 0]
    long["origin"] = long["k"].map(lambda i: keys[i][0])
    long["source_category"] = long["k"].map(lambda i: keys[i][1])
    long["geo_level"] = "municipality"
    long["geo_name"] = long["geo_id"].map(name)
    long["tier"] = "derived"
    long["year"] = 2020
    out = long[["geo_id", "geo_level", "geo_name", "origin", "source_category", "count", "tier",
                "year"]]
    if int(out["count"].sum()) != TOTAL_2020:
        raise SystemExit("output does not sum to the census")
    out.to_csv(OUT, index=False, encoding="utf-8")
    write_fragment(sorted(out["source_category"].unique()))

    # ---- report ----
    t = out.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(f"\nwrote {OUT}: {len(out):,} rows, {out['geo_id'].nunique():,} units, "
          f"{int(t.sum()):,} people, {len(t)} nodes")
    for n, c in t.head(30).items():
        print(f"    {n:<50} {c:>11,}  {c / TOTAL_2020:7.3%}")
    o = out.groupby("origin")["count"].sum()
    print("  by origin: " + ", ".join(f"{k} {v:,}" for k, v in o.items()))
    ryu = out[out["origin"] == "ryukyuan"].groupby("source_category")["count"].sum()
    print("  Ryukyuan: " + ", ".join(f"{k.split('.')[-1]} {v:,}" for k, v in ryu.items()))


if __name__ == "__main__":
    main()
