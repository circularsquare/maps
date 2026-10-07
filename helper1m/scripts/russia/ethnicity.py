"""Ethnic composition pies for Russia, from the 2021 census by settlement.

Usage: C:\\Python39\\python.exe helper1m/scripts/russia/ethnicity.py

Reads the "To Be Precise" (tochno.st) table of every settlement and municipality
with its count for each of the census's 194 nationalities (CC BY; built on the
settlement database Seva Bashirov published from the census), and writes
countries/russia/composition.json for both helper1m levels.

The census's own municipal rows are the counts used: they are complete, where
settlements of 10 people or fewer have their nationality blanked. Those rows are
on 2021 municipalities, and many have since merged, been renamed or renumbered,
so each is carried onto helper1m's polygons through its settlements, placed by
their coordinates (the table's "current" codes are a year newer than the
polygons). A municipality whose settlements lie in one polygon goes there whole;
one split between several gives each its own settlements' counts, and the
remainder (blanked villages, people outside any settlement) by population.

People could name two nationalities in 2021 and the table counts both, so a
unit's groups can add to slightly more than its population. "Not stated" is the
census's people with no nationality on their form (mostly counted from
administrative records, not interviewed) plus those who said they had none.
"""
import colorsys
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HELPER = Path(__file__).resolve().parents[2]
RAW = HELPER / "data" / "russia" / "raw" / "tochno" / "data_allsettlements_anon_156_v20260925.parquet"
UNIT_MAP = HELPER / "data" / "russia" / "boundaries" / "unit_map.csv"
BOUNDS = HELPER / "data" / "russia" / "boundaries" / "adm2.gpkg"
COUNTRY = HELPER / "countries" / "russia"
OUT = COUNTRY / "composition.json"
COLORS = Path(__file__).with_name("ethnicity_colors.csv")

NAT_MIN = 50_000        # people nationally for a group to get its own colour
SHARE_MIN = 0.10        # ... or this share of some municipality
UNIT_MIN = 1_000        # ... of at least this many people
SPLIT_MIN = 0.05        # a polygon needs this share of a municipality's people to take a part
EXCLUDE_REGIONS = ("35", "67")   # Crimea, Sevastopol: not drawn, see fetch.py INCLUDE_CRIMEA

UPPER = "Муниципалитет верхнего уровня"
ADMIN = "Административный район"      # Moscow's okrugs, St Petersburg's districts
SETTLEMENT = "Населенный пункт"
NAT_FIRST, NAT_LAST = 24, 218          # the 194 nationality columns
NOT_STATED = ["Лица, в переписных листах которых национальная принадлежность не указана",
              "Нет национальной принадлежности*"]
OTHER_ANSWER = "Указавшие другие ответы о национальной принадлежности (не перечисленные выше)"

# English names for every group that can pass the thresholds, keyed by the
# census column's name before any bracket.
EN = {
    "Русские": "Russians", "Татары": "Tatars", "Чеченцы": "Chechens", "Башкиры": "Bashkirs",
    "Чуваши": "Chuvash", "Аварцы": "Avars", "Армяне": "Armenians", "Украинцы": "Ukrainians",
    "Даргинцы": "Dargins", "Казахи": "Kazakhs", "Кумыки": "Kumyks", "Кабардинцы": "Kabardians",
    "Ингуши": "Ingush", "Лезгины": "Lezgins", "Мордва": "Mordvins", "Осетины": "Ossetians",
    "Якуты": "Yakuts (Sakha)", "Азербайджанцы": "Azerbaijanis", "Буряты": "Buryats",
    "Марийцы": "Mari", "Удмурты": "Udmurts", "Таджики": "Tajiks", "Узбеки": "Uzbeks",
    "Тувинцы": "Tuvans", "Крымские татары": "Crimean Tatars", "Карачаевцы": "Karachays",
    "Белорусы": "Belarusians", "Немцы": "Germans", "Калмыки": "Kalmyks", "Лакцы": "Laks",
    "Цыгане": "Roma", "Табасараны": "Tabasarans", "Коми": "Komi", "Киргизы": "Kyrgyz",
    "Балкарцы": "Balkars", "Турки": "Turks", "Черкесы": "Circassians", "Грузины": "Georgians",
    "Адыгейцы": "Adyghe", "Ногайцы": "Nogais", "Корейцы": "Koreans", "Евреи": "Jews",
    "Алтайцы": "Altai", "Молдаване": "Moldovans", "Хакасы": "Khakas", "Мордва-эрзя": "Erzya",
    "Коми-пермяки": "Komi-Permyaks", "Греки": "Greeks", "Казаки": "Cossacks",
    "Ненцы": "Nenets", "Абазины": "Abazins", "Туркмены": "Turkmens", "Эвенки": "Evenks",
    "Агулы": "Aguls", "Рутульцы": "Rutuls", "Карелы": "Karelians", "Ханты": "Khanty",
    "Кряшены": "Kryashens", "Курды": "Kurds", "Эвены": "Evens", "Андийцы": "Andi",
    "Чукчи": "Chukchi", "Дидойцы": "Tsez (Didoi)", "Горные марийцы": "Hill Mari",
    "Цахуры": "Tsakhurs", "Манси": "Mansi", "Мордва-мокша": "Moksha", "Нанайцы": "Nanai",
    "Долганы": "Dolgans", "Коряки": "Koryaks", "Каратинцы": "Karata",
    "Тувинцы-тоджинцы": "Todzha Tuvans", "Бежтинцы": "Bezhta", "Нагайбаки": "Nagaibaks",
    "Ахвахцы": "Akhvakh", "Коми-ижемцы": "Izhma Komi", "Сойоты": "Soyots",
    "Селькупы": "Selkups", "Дунгане": "Dungans", "Теленгиты": "Telengits",
    "Ительмены": "Itelmens", "Ульчи": "Ulchi", "Бесермяне": "Besermyan",
    "Эскимосы": "Yupik (Eskimos)", "Юкагиры": "Yukaghirs", "Тубалары": "Tubalars",
    "Шорцы": "Shors", "Сибирские татары": "Siberian Tatars", "Китайцы": "Chinese",
    "Езиды": "Yazidis", "Поляки": "Poles", "Саамы": "Sami", "Вепсы": "Veps",
    "Арабы": "Arabs", "Литовцы": "Lithuanians", "Болгары": "Bulgarians",
}

# Language family, for the colour band. Anything unlisted is "other".
FAMILY = {
    "slavic": ["Русские", "Украинцы", "Белорусы", "Казаки", "Поляки", "Болгары"],
    "turkic": ["Татары", "Башкиры", "Чуваши", "Казахи", "Кумыки", "Якуты", "Азербайджанцы",
               "Узбеки", "Тувинцы", "Крымские татары", "Карачаевцы", "Киргизы", "Балкарцы",
               "Турки", "Ногайцы", "Алтайцы", "Хакасы", "Туркмены", "Кряшены", "Долганы",
               "Тувинцы-тоджинцы", "Нагайбаки", "Сойоты", "Теленгиты", "Тубалары", "Шорцы",
               "Сибирские татары"],
    "uralic": ["Мордва", "Марийцы", "Удмурты", "Коми", "Мордва-эрзя", "Коми-пермяки", "Ненцы",
               "Карелы", "Ханты", "Горные марийцы", "Манси", "Мордва-мокша", "Коми-ижемцы",
               "Селькупы", "Бесермяне", "Саамы", "Вепсы"],
    "caucasian": ["Чеченцы", "Аварцы", "Даргинцы", "Кабардинцы", "Ингуши", "Лезгины", "Лакцы",
                  "Табасараны", "Черкесы", "Адыгейцы", "Абазины", "Агулы", "Рутульцы",
                  "Андийцы", "Дидойцы", "Цахуры", "Каратинцы", "Бежтинцы", "Ахвахцы",
                  "Грузины"],
    "mongolic": ["Буряты", "Калмыки"],
    "siberian": ["Эвенки", "Эвены", "Чукчи", "Нанайцы", "Коряки", "Ительмены", "Ульчи",
                 "Эскимосы", "Юкагиры"],
    "indo_european": ["Армяне", "Осетины", "Таджики", "Немцы", "Цыгане", "Молдаване", "Греки",
                      "Курды", "Езиды", "Евреи", "Литовцы"],
}
FAMILY_OF = {k: fam for fam, ks in FAMILY.items() for k in ks}
FAMILY_HUES = {"slavic": (200, 225), "turkic": (0, 35), "uralic": (95, 150),
               "caucasian": (270, 330), "mongolic": (45, 60), "siberian": (170, 190),
               "indo_european": (20, 45), "other": (60, 90)}
HAND = {"Русские": "#b4c0cc", "not_stated": "#f0f0f0", "other": "#9a9a9a"}


def log(msg):
    print(msg, flush=True)


def short(col):
    return col.split(" (")[0].strip()


def auto_color(fam, i):
    lo, hi = FAMILY_HUES[fam]
    h = (lo + ((i * 0.618034) % 1) * (hi - lo)) / 360
    light = (0.45, 0.60, 0.35)[i % 3]
    sat = (0.60, 0.50, 0.70)[(i // 3) % 3]
    r, g, b = colorsys.hls_to_rgb(h, light, sat)
    return "#{:02x}{:02x}{:02x}".format(round(r * 255), round(g * 255), round(b * 255))


def palette(groups):
    """Colours from ethnicity_colors.csv; groups it lacks are appended. Hers to hand-edit."""
    have = {}
    if COLORS.exists():
        with COLORS.open(encoding="utf-8", newline="") as fh:
            have = {r["key"]: r["color"] for r in csv.DictReader(fh)}
    new, count = [], {}
    for g in groups:
        if g["key"] in have:
            continue
        fam = g["family"]
        c = HAND.get(g["key"]) or auto_color(fam, count.get(fam, 0))
        count[fam] = count.get(fam, 0) + 1
        have[g["key"]] = c
        new.append({"key": g["key"], "en": g["en"], "family": fam, "color": c})
    if new:
        exists = COLORS.exists()
        with COLORS.open("a", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["key", "en", "family", "color"])
            if not exists:
                w.writeheader()
            w.writerows(new)
        log(f"added {len(new)} rows to {COLORS.name}")
    return have


def main():
    df = pd.read_parquet(RAW)
    cols = list(df.columns)
    nat_cols = cols[NAT_FIRST:NAT_LAST]
    df["oktmo"] = df["oktmo"].astype(str)
    df["oktmo_new"] = df["oktmo_new"].astype(str)
    df = df[~df["oktmo"].str[:2].isin(EXCLUDE_REGIONS)]

    src = df[df["object_level"].isin([UPPER, ADMIN])].reset_index(drop=True)
    sett = df[df["object_level"] == SETTLEMENT].reset_index(drop=True)
    log(f"{len(src):,} 2021 municipalities, {len(sett):,} settlements, "
        f"{src['population'].sum():,} people")

    # --- groups
    S = src[nat_cols].clip(lower=0).to_numpy(float)
    share = S / np.maximum(src["population"].to_numpy(float), 1)[:, None]
    big = (src["population"] >= UNIT_MIN).to_numpy()
    passes = (S.sum(0) >= NAT_MIN) | (share[big].max(0) >= SHARE_MIN)
    keep = [j for j in np.argsort(-S.sum(0)) if passes[j]]
    groups = []
    for j in keep:
        k = short(nat_cols[j])
        if k not in EN:
            sys.exit(f"no English name for {nat_cols[j]!r}; add it to EN")
        groups.append({"key": k, "en": EN[k], "family": FAMILY_OF.get(k, "other"),
                       "title": nat_cols[j]})
    other_i = len(groups)
    groups.append({"key": "other", "en": "Other", "family": "other",
                   "title": "Every nationality without its own colour, and answers the "
                            "census lists under no nationality"})
    ns_i = len(groups)
    groups.append({"key": "not_stated", "en": "Not stated", "family": "other",
                   "title": "No nationality on the census form (mostly people counted from "
                            "administrative records), or answered none"})
    col_to_g = np.full(len(nat_cols), other_i)
    for gi, j in enumerate(keep):
        col_to_g[j] = gi
    n_g = len(groups)
    log(f"{n_g - 2} groups with their own colour, plus Other and Not stated")

    def vectors(rows):
        M = rows[nat_cols].clip(lower=0).to_numpy(float)
        G = np.zeros((len(rows), n_g))
        np.add.at(G.T, col_to_g, M.T)
        G[:, other_i] += rows[OTHER_ANSWER].clip(lower=0).to_numpy(float)
        G[:, ns_i] += rows[NOT_STATED].clip(lower=0).sum(1).to_numpy(float)
        return G

    G_src = vectors(src)
    G_set = vectors(sett)

    # --- targets: helper1m's polygons. The table's "current" codes are a year
    # newer than the Rosstat table helper1m is drawn on, and Altai and Krasnoyarsk
    # renumbered in between, so settlements are placed by their coordinates.
    import geopandas as gpd
    umap = pd.read_csv(UNIT_MAP, dtype=str)
    poly_of = dict(zip(umap["oktmo"], umap["code"]))
    poly_of["40360000"] = "40280000"   # Kronstadt: the census row carries its municipal code
    adm2 = gpd.read_file(BOUNDS)[["code", "geometry"]].to_crs("EPSG:4326")
    has_xy = sett["latitude"].notna()
    pts = gpd.GeoDataFrame(sett.loc[has_xy, []],
                           geometry=gpd.points_from_xy(sett.loc[has_xy, "longitude"],
                                                       sett.loc[has_xy, "latitude"]),
                           crs="EPSG:4326")
    hit = gpd.sjoin(pts, adm2, how="left", predicate="within")
    hit = hit[~hit.index.duplicated()]
    sett["tgt"] = hit["code"].reindex(sett.index)
    sett["src"] = sett["oktmo"].str[:5] + "000"
    miss = sett["tgt"].isna()
    log(f"settlements in no polygon: {miss.sum():,} ({sett.loc[miss, 'population'].sum():,} "
        f"people; {(~has_xy).sum():,} of them have no coordinates)")

    acc = {}   # polygon code -> [counts, sum w*x, sum w*y, sum w]

    def put(code, v, pts):
        a = acc.setdefault(code, [np.zeros(n_g), 0.0, 0.0, 0.0])
        a[0] += v
        for x, y, w in pts:
            a[1] += w * x
            a[2] += w * y
            a[3] += w

    by_src = {k: g for k, g in sett.groupby("src")}
    whole, split, direct, lost, splits = 0, 0, 0, [], []
    for i, row in src.iterrows():
        s = by_src.get(row["oktmo"])
        if s is None or s["population"].sum() == 0:
            code = poly_of.get(row["oktmo"])
            if code is None:
                lost.append((row["oktmo"], row["object_name"], row["population"]))
                continue
            put(code, G_src[i], [])
            direct += 1
            continue
        s = s[s["tgt"].notna()]
        pts = lambda part: [(r.longitude, r.latitude, r.population) for r in part.itertuples()
                            if pd.notna(r.latitude) and r.population > 0]
        if s.empty:
            code = poly_of.get(row["oktmo"])
            if code is None:
                lost.append((row["oktmo"], row["object_name"], row["population"]))
                continue
            put(code, G_src[i], [])
            direct += 1
            continue
        # A geocoded village a few hundred metres over a simplified border is
        # noise, not a split: only a polygon holding SPLIT_MIN of the
        # municipality's people takes a share; strays go to the biggest.
        pop_t = s.groupby("tgt")["population"].sum()
        real = pop_t[pop_t >= SPLIT_MIN * pop_t.sum()]
        if len(real) == 1:
            put(real.idxmax(), G_src[i], pts(s))
            whole += 1
            continue
        split += 1
        s = s.copy()
        s.loc[~s["tgt"].isin(real.index), "tgt"] = real.idxmax()
        pop_t = s.groupby("tgt")["population"].sum()
        part_sum = G_set[s.index].sum(0)
        resid = np.maximum(G_src[i] - part_sum, 0)
        for code, p in pop_t.items():
            part = s[s["tgt"] == code]
            put(code, G_set[part.index].sum(0) + resid * p / pop_t.sum(), pts(part))
        splits.append((row["object_name"], {c: int(p) for c, p in pop_t.items()}))
    log(f"2021 municipalities: {whole:,} whole into one polygon, {split:,} split by their "
        f"settlements, {direct:,} by code alone")
    for o, n, p in lost:
        log(f"  !! {o} {n}: {p:,} people on no polygon")
    for n, parts in splits:
        log(f"  split {n}: {parts}")

    # Pie position: where the people are, else the polygon's middle.
    import shapely.geometry
    polys, parent = {}, {}
    for f in (COUNTRY / "adm2").glob("*.geojson"):
        for feat in json.loads(f.read_text(encoding="utf-8"))["features"]:
            p = feat["properties"]
            polys[p["code"]] = feat["geometry"]
            parent[p["code"]] = p["parent_code"]
    nopie = sorted(set(polys) - set(acc))
    log(f"level 2: {len(acc):,} of {len(polys):,} polygons have a pie"
        + (f"; none for {nopie}" if nopie else ""))

    def finish(a, geom=None):
        v, sx, sy, sw = a
        if sw <= 0:
            pt = shapely.geometry.shape(geom).representative_point()
            x, y = pt.x, pt.y
        else:
            x, y = sx / sw, sy / sw
        return v, x, y

    lv2 = {c: finish(a, polys.get(c)) for c, a in acc.items() if c in polys}
    lv1 = {}
    for c, (v, x, y) in lv2.items():
        b = lv1.setdefault(parent[c], [np.zeros(n_g), 0.0, 0.0, 0.0])
        w = v.sum()
        b[0] += v
        b[1] += w * x
        b[2] += w * y
        b[3] += w
    lv1 = {c: (b[0], b[1] / b[3], b[2] / b[3]) for c, b in lv1.items()}

    out = {}
    for lvl, units in (("1", lv1), ("2", lv2)):
        uo = {}
        for code, (v, x, y) in units.items():
            v = np.rint(v).astype(np.int64)
            g = np.flatnonzero(v > 0)
            g = g[np.argsort(-v[g], kind="stable")]
            uo[code] = {"t": int(v.sum()), "x": round(float(x), 4), "y": round(float(y), 4),
                        "g": [int(k) for k in g], "k": [int(v[k]) for k in g]}
        out[lvl] = uo
        log(f"  level {lvl}: {len(uo):,} units, {sum(u['t'] for u in uo.values()):,} answers")

    # Check: each subject's answers against its 2021 census population as
    # helper1m has it. Over 0 is people who gave two nationalities.
    pops = pd.read_csv(HELPER / "data" / "russia" / "population.csv", dtype={"code": str})
    census = pops[(pops["level"] == 1) & (pops["year"] == 2021)].set_index("code")["pop"]
    ratio = sorted((u["t"] / census[c] - 1, c) for c, u in out["1"].items())
    log(f"  subject answers / 2021 census population: {ratio[0][0]:+.2%} ({ratio[0][1]}) "
        f"to {ratio[-1][0]:+.2%} ({ratio[-1][1]}); median {ratio[len(ratio) // 2][0]:+.2%}")

    colors = palette(groups)
    doc = {
        "label": "Nationality",
        "year": 2021,
        "levels": out,
        "groups": [{"key": g["key"], "en": g["en"], "title": g["title"],
                    "color": colors[g["key"]]} for g in groups],
        "source": "2021 census (VPN-2020) nationality by settlement and municipality, from "
                  "\"Settlements of Russia: population, ethnic composition, and geographic "
                  "coordinates\", Rosstat, processed by To Be Precise (tochno.st), CC BY. "
                  "Both answers of people who gave two are counted.",
    }
    with OUT.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, separators=(",", ":"))
    log(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")

    nat = np.zeros(n_g)
    for u in out["1"].values():
        nat[u["g"]] += u["k"]
    for k in np.argsort(-nat)[:15]:
        log(f"    {groups[k]['en']:<24} {nat[k]:>13,.0f}  {nat[k] / nat.sum():6.2%}")


if __name__ == "__main__":
    main()
