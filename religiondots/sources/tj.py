"""Tajikistan: no published source places anyone's religion, so the map is an ETHNICITY MODEL on the
2020 census's regional populations. Nationality by region from the 2010 census; Islam for every
nationality of Muslim heritage; the Slavic and other minorities by their own religion.

Reads data/geo/tj/tj_lookup.csv (`sources/tj_geo.py`), the 2010 census's Volume III, and
Kazakhstan's 2021 census religion by nationality (`sources/kz_model.py`); writes
data/normalized/tj.csv. `sources/tj.md` is the record; the rulings are `ask/RULINGS.md` 2026-09-15
(priority holes) and 2026-09-16 (a country no published source places is drawn on the best figure
there is, method disclosed).

## WHAT ASKS, AND WHAT IT SAYS

The 2020 census asked religion (Form 2, question 7) and has published nothing from it at any level
(`sources/tj.md` §1-2; Volume III is still an empty heading on `stat.tj` on 2026-10-03). LiTS III
returns 99.5% Muslim and no Orthodox at all; the Central Asia Barometer does not ask in Tajikistan
(§3-4). Pew Research Center's 2020 figure (Muslim 98.91%, Christian 1.00%, unaffiliated 0.07%)
comes from its Survey of the World's Muslims, 2011-12 (Pew 2025, Appendix A, p.24), a sample of
about 1,500 adults: its 97,515 Christians rest on about fifteen respondents.

## THE MODEL

For each of the five regions: the 2020 census's permanent population (`tj_geo.py`). Inside it,
the nationalities that are not of Muslim heritage are placed and given a religion; everyone else is
drawn on Islam.

  group                       2010 count  placed by                        religion
  Russians                        34,838  2010 Russians by region           Kazakhstan 2021's Russians
  Tatars                           6,495  2010 Tatars by region             Kazakhstan 2021's Tatars
  Ukrainians                       1,090  2010 Russians by region           Kazakhstan 2021's Ukrainians
  Belarusians                        104  "                                 Kazakhstan 2021's Belarusians
  Germans                            446  "                                 Kazakhstan 2021's Germans
  Armenians                          434  "                                 Armenian Apostolic
  Georgians                           92  "                                 Georgian Orthodox
  Jews (both rows)                    36  "                                 Jewish
  other non-Muslim heritage        2,946  "                                 not known

Volume III prints nationality by region only for Tajiks, Uzbeks, Russians, Kyrgyz, Turkmens, Tatars
and Kazakhs; the other 85 nationalities are national rows. So the small European and other groups
are placed where the Russians are (55% of them in Dushanbe in 2010), and the region's own `other`
column is asserted to hold them.

**2010 to 2020.** The 2020 census has published nationality only as shares on the Agency's own
slides (UNECE workshop, September 2023, `unece2023_WS10RizoevENG.pdf`, slide 27): Tajik 86.1%,
Uzbek 11.3%, Kyrgyz 0.4%, Russian 0.3%, other 1.9%. Russians are taken at 0.3% of 9,657,005, which
is 28,971 against 34,838 in 2010, and every non-Muslim-heritage group above is scaled by the same
factor (0.832) on its 2010 regional distribution. The rounding of 0.3% alone spans 24,000 to 34,000.

**Russians, Tatars, Ukrainians, Belarusians and Germans take Kazakhstan's census coefficients**, with
refusals and the small answers (Catholic, Protestant, Judaism, Buddhism, other) taken out and the rest
renormalised, as `sources/az.py` does: Russians 92.7% Orthodox, 2.1% Muslim, 5.1% non-believers.
Germans in Kazakhstan are 73.7% Orthodox in that table (89.1% once refusals and the 3.2% Catholics
are taken out), and are taken as printed. Armenians,
Georgians and Jews are religio-ethnic (spec §14.5). Koreans, Ossetians, Moldovans, the Chinese and
everyone else outside the Muslim-heritage list are drawn on `unknown`.

## WHAT IS NOT DRAWN

Pew's 97,515 Christians, against the model's 28,000 or so: no source places the difference, and it
rests on a handful of survey respondents. The Ismailis of Gorno-Badakhshan: the region's Muslims stay
on bare `islam`, like every other Muslim here (no sect is drawn anywhere in the country), and whether
to draw them as Ismaili is ask 051 (§14; `sources/tj.md` §8).

Usage:
    python sources/tj.py            rebuild data/normalized/tj.csv and print the witnesses
"""

import io
import os
import re
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "6")

import pandas as pd

RAW = os.path.join(ROOT, "data", "raw", "tj")
LOOKUP = os.path.join(ROOT, "data", "geo", "tj", "tj_lookup.csv")
VOL3 = os.path.join(RAW, "census2010_vol3.pdf")
VOL3_URL = ("https://web.archive.org/web/20131014054442id_/http://www.stat.tj/ru/img/"
            "526b8592e834fcaaccec26a22965ea2b_1355501132.pdf")
PEW = os.path.join(ROOT, "data", "raw", "estimates", "pew.zip")
OUT = os.path.join(ROOT, "data", "normalized", "tj.csv")

TOTAL_2010 = 7_564_502
TOTAL_2020 = 9_657_005
RUSSIAN_SHARE_2020 = 0.003      # the Agency's slide 27, rounded to one decimal of a percent

# Volume III, "Национальный состав населения Республики Таджикистан" (pp.7-11), the 2010 column,
# every row. Asserted to sum to the census total.
NAT2010 = {
    "Таджики": 6373834, "Узбеки": 926344, "Русские": 34838, "Татары": 6495, "Кыргызы": 60715,
    "Украинцы": 1090, "Немцы": 446, "Туркмены": 15171, "Корейцы": 634, "Казахи": 595, "Евреи": 34,
    "Осетины": 396, "Белорусы": 104, "Татары крымские": 18, "Татары сибирские": 17, "Башкиры": 143,
    "Армяне": 434, "Мордва": 42, "Евреи среднеазиатские": 2, "Азербайджанцы": 371, "Чуваши": 47,
    "Афганцы": 3675, "Цыгане": 2334, "Лакцы": 2, "Болгары": 19, "Грузины": 92, "Молдаване": 157,
    "Турки (османы)": 1360, "Поляки": 23, "Удмурты": 12, "Марийцы": 13, "Греки": 28, "Уйгуры": 276,
    "Литовцы": 11, "Персы (иране)": 473, "Даргинцы": 6, "Латыши": 9, "Лезгины": 13, "Арабы": 4184,
    "Кабардинцы": 8, "Аварцы": 13, "Караимы": 2, "Каракалпаки": 4, "Буряты": 6, "Коми": 1,
    "Эстонцы": 10, "Чеченцы": 20, "Кумыки": 5, "Ингуши": 11, "Черкесы": 5, "Хакасы": 4, "Финны": 5,
    "Коми-пермяки": 2, "Табасараны": 6, "Китайцы": 801, "Курды": 7, "Карачаевцы": 2, "Абхазы": 4,
    "Болкарцы": 2, "Абазины": 5, "Австрийцы": 9, "Американцы": 62, "Румыны": 4, "Англичане": 104,
    "Ненцы": 1, "Вьетнамцы": 3, "Голландцы": 6, "Испанцы": 7, "Карелы": 166, "Словаки": 2,
    "Французы": 7, "Итальянцы": 2, "Японцы": 2, "Дунгане": 1, "Коряки": 4, "Венгры": 1, "Агулы": 1,
    "Тофалары": 2, "Чуванцы": 4, "Ногайцы": 1, "Минги": 268, "Дурмены": 7608, "Лакайцы": 65555,
    "Конграты": 38078, "Катаганы": 7601, "Юзы": 3798, "Барлосы": 5271, "Семизы": 47, "Кесамиры": 156,
    "Народы Индии и Пакистана": 262, "Другие национальности": 15, "Национальность не указана": 74,
}
# Nationalities of Muslim heritage: drawn on Islam with the majority. The Uzbek tribal names the
# census lists apart (Lakai, Kongrat, Durmen, Katagan, Yuz, Barlos, Ming, Semiz, Kesamir) are here.
MUSLIM_HERITAGE = {
    "Таджики", "Узбеки", "Кыргызы", "Туркмены", "Казахи", "Татары крымские", "Татары сибирские",
    "Башкиры", "Азербайджанцы", "Афганцы", "Цыгане", "Лакцы", "Турки (османы)", "Уйгуры",
    "Персы (иране)", "Даргинцы", "Лезгины", "Арабы", "Кабардинцы", "Аварцы", "Каракалпаки",
    "Чеченцы", "Кумыки", "Ингуши", "Черкесы", "Табасараны", "Курды", "Карачаевцы", "Болкарцы",
    "Абазины", "Дунгане", "Агулы", "Ногайцы", "Минги", "Дурмены", "Лакайцы", "Конграты", "Катаганы",
    "Юзы", "Барлосы", "Семизы", "Кесамиры",
}
# The rest: nationality -> religion rule. Russians and Tatars are placed on their own 2010 regional
# counts; every other group on the Russians'.
RULE = {"Русские": "kz:Орыстар", "Татары": "kz:Татарлар", "Украинцы": "kz:Украиндар",
        "Белорусы": "kz:Белорустар", "Немцы": "kz:Немістер", "Армяне": "Armenian Apostolic",
        "Грузины": "Georgian Orthodox", "Евреи": "Jewish", "Евреи среднеазиатские": "Jewish"}
UNKNOWN = "Other nationality, religion not known"
KZ_KEEP = {"Православие": "Orthodox (Kazakhstan's coefficient)",
           "Ислам": "Muslim",
           "Неверующие": "Non-believer (Kazakhstan's coefficient)"}

# Volume III's regional tables (pp.108-115), in print order, and the label that opens each block.
REGION_LABELS = [("TJ-GB", r"Горно\s*-\s*Бадахшанская\s+Автономная\s+область"),
                 ("TJ-SU", r"Согдийская\s+область"),
                 ("TJ-KT", r"Хатлонская\s+область"),
                 ("TJ-DU", r"г\.\s*Душанбе"),
                 ("TJ-RA", r"Города\s+и\s+районы\s+республиканского\s+подчинения")]
REGION_GROUPS = ("Таджики", "Узбеки", "Русские", "Кыргызы", "Туркмены", "Татары", "Казахи")

# Measured 2026-10-03 and asserted, so note_public cannot drift from the data.
NOTE = dict(people=9_657_005, muslims=9_622_434, orthodox=29_641, christians=30_078,
            unaffiliated=2_000, unknown=2_464, jews=29, russians=28_971,
            dushanbe_non_muslim_pct=1.94, pew_christians=97_515)


def read_regions():
    """{unit: {nationality: 2010 count}} for the seven groups Volume III prints by region."""
    import fitz

    if not os.path.exists(VOL3):
        import urllib.request
        req = urllib.request.Request(VOL3_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = r.read()
        if not data.startswith(b"%PDF"):
            raise SystemExit(f"{VOL3_URL} is not a PDF")
        open(VOL3, "wb").write(data)
    doc = fitz.open(VOL3)
    if doc.page_count != 537:
        raise SystemExit(f"{VOL3}: {doc.page_count} pages, expected 537 (truncated?)")
    text = "\n".join(doc[i].get_text() for i in range(107, 115))       # printed pp.108-115
    marks = [(0, "national")]
    for unit, pat in REGION_LABELS:
        m = re.search(pat, text)
        if not m:
            raise SystemExit(f"Volume III p.108-115: no block for {unit}")
        marks.append((m.start(), unit))
    if [u for _p, u in marks] != ["national"] + [u for u, _ in REGION_LABELS] or \
            sorted(p for p, _u in marks) != [p for p, _u in marks]:
        raise SystemExit("Volume III regional blocks are not in print order")
    out = {u: {} for _p, u in marks}
    for m in re.finditer(r"\n(" + "|".join(REGION_GROUPS) + r")\s*\n[^\d]*?Оба пола\s*\n\s*(\d+)",
                         text):
        unit = [u for p, u in marks if p <= m.start()][-1]
        if m.group(1) in out[unit]:
            raise SystemExit(f"{unit}: {m.group(1)} read twice")
        out[unit][m.group(1)] = int(m.group(2))
    for u, d in out.items():
        if set(d) != set(REGION_GROUPS):
            raise SystemExit(f"{u}: read {sorted(d)}, expected the seven groups")
    for g in REGION_GROUPS:
        s = sum(out[u][g] for u, _p in REGION_LABELS)
        if s != out["national"][g] or s != NAT2010[g]:
            raise SystemExit(f"{g}: regions sum to {s:,}, national row {out['national'][g]:,}, "
                             f"p.7 {NAT2010[g]:,}")
    print("  witness: Volume III's seven regional groups sum to its national rows and to p.7")
    return {u: out[u] for u, _p in REGION_LABELS}


def kz_coefficients(nats):
    import kz_model

    coef = kz_model.read_coefficients()["total"]
    out = {}
    for nat in nats:
        row = coef[nat]
        kept = {KZ_KEEP[k]: float(row[k]) for k in KZ_KEEP}
        tot = sum(kept.values())
        out[nat] = {k: v / tot for k, v in kept.items()}
        print(f"  Kazakhstan 2021, {nat}: {int(row['total']):,}; kept {tot / float(row['total']):.1%}; "
              + ", ".join(f"{k} {v:.2%}" for k, v in out[nat].items()))
    return out


def main():
    lut = pd.read_csv(LOOKUP)
    if len(lut) != 5 or int(lut["pop"].sum()) != TOTAL_2020 or int(lut["pop2010"].sum()) != TOTAL_2010:
        raise SystemExit(f"{LOOKUP} is not the five regions at 9,657,005 / 7,564,502; re-run tj_geo.py")
    pop = dict(zip(lut["geo_id"], lut["pop"]))
    pop10 = dict(zip(lut["geo_id"], lut["pop2010"]))
    name = dict(zip(lut["geo_id"], lut["name"]))

    if sum(NAT2010.values()) != TOTAL_2010:
        raise SystemExit(f"Volume III p.7-11 transcription sums to {sum(NAT2010.values()):,}, "
                         f"not {TOTAL_2010:,}")
    print(f"  witness: Volume III's {len(NAT2010)} national rows sum to the census total")
    if not MUSLIM_HERITAGE <= set(NAT2010) or not set(RULE) <= set(NAT2010):
        raise SystemExit("a classified nationality is not in the national table")
    reg = read_regions()
    for u in pop:
        named = sum(reg[u].values())
        if named > pop10[u]:
            raise SystemExit(f"{u}: the seven groups exceed the 2010 population")

    f = round(RUSSIAN_SHARE_2020 * TOTAL_2020) / NAT2010["Русские"]
    print(f"  2010 -> 2020: Russians {NAT2010['Русские']:,} -> {round(RUSSIAN_SHARE_2020 * TOTAL_2020):,} "
          f"(0.3% of the 2020 census); factor {f:.4f} for every non-Muslim-heritage group")

    ru10 = {u: reg[u]["Русские"] for u in pop}
    ru_sum = sum(ru10.values())
    groups = {}                     # nationality -> {unit: 2010 count}
    for nat, n in NAT2010.items():
        if nat in MUSLIM_HERITAGE:
            continue
        if nat in ("Русские", "Татары"):
            groups[nat] = {u: reg[u][nat] for u in pop}
        else:
            groups[nat] = {u: n * ru10[u] / ru_sum for u in pop}
    # each region's `other` column must hold the groups placed on the Russians' distribution
    for u in pop:
        other10 = pop10[u] - sum(reg[u].values())
        spread = sum(v[u] for k, v in groups.items() if k not in ("Русские", "Татары"))
        if spread > other10:
            raise SystemExit(f"{name[u]}: {spread:,.0f} spread on the Russians' distribution, but "
                             f"only {other10:,} outside the seven groups in 2010")

    kz = kz_coefficients(sorted({r[3:] for r in RULE.values() if r.startswith("kz:")}))
    rows = []
    placed = {u: 0.0 for u in pop}
    for nat, by_u in groups.items():
        rule = RULE.get(nat, UNKNOWN)
        shares = kz[rule[3:]] if rule.startswith("kz:") else {rule: 1.0}
        for u, n10 in by_u.items():
            n = n10 * f
            if n <= 0:
                continue
            placed[u] += n
            for cat, s in shares.items():
                rows.append((u, cat, n * s))
    for u in pop:
        rows.append((u, "Muslim", pop[u] - placed[u]))
    df = pd.DataFrame(rows, columns=["geo_id", "source_category", "count"])
    df = df.groupby(["geo_id", "source_category"], as_index=False)["count"].sum()

    out = []
    for u, g in df.groupby("geo_id"):
        fl = g["count"].astype(float)
        base = fl.astype(int)
        short = int(round(pop[u] - base.sum()))
        order = (fl - base).sort_values(ascending=False).index[:short]
        base.loc[order] += 1
        g = g.assign(count=base)
        if int(g["count"].sum()) != pop[u]:
            raise SystemExit(f"{u}: rounded to {int(g['count'].sum())}, not {pop[u]}")
        out.append(g)
    df = pd.concat(out)
    df = df[df["count"] > 0]
    df["geo_level"] = "unit"
    df["geo_name"] = df["geo_id"].map(name)
    df["basis"] = "model"
    df["year"] = 2020
    df["source_id"] = "tj_census2020_ethnicity_model"
    df["note"] = ("2020 census permanent population; minorities placed on the 2010 census's "
                  "nationality by region; religion within group per sources/tj.py")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df[["geo_id", "geo_level", "geo_name", "source_category", "count", "basis", "year", "source_id",
        "note"]].to_csv(OUT, index=False, encoding="utf-8")

    tot = df.groupby("source_category")["count"].sum().sort_values(ascending=False)
    people = int(df["count"].sum())
    print(f"\nwrote {OUT}: {df['geo_id'].nunique()} units, {people:,} people")
    for k, v in tot.items():
        print(f"    {k:<45} {v:>10,}  {v / people:.3%}")
    print("\n  not on Islam, by region:")
    nm = df[df["source_category"] != "Muslim"].groupby("geo_id")["count"].sum()
    for u in sorted(pop, key=lambda x: -nm.get(x, 0)):
        print(f"    {name[u]:<38} {int(nm.get(u, 0)):>7,}  {nm.get(u, 0) / pop[u]:.2%} of {pop[u]:,}")

    with zipfile.ZipFile(PEW) as z:
        nm_ = [n for n in z.namelist() if n.endswith("(unrounded counts).csv")][0]
        t = pd.read_csv(io.BytesIO(z.read(nm_)), thousands=",")
    p = t[(t["Country"] == "Tajikistan") & (t["Year"] == 2020)].iloc[0]
    print(f"\n  Pew 2020 (Survey of the World's Muslims 2011-12): {int(p['Population']):,}; Muslims "
          f"{p['Muslims'] / p['Population']:.2%}, Christians {int(p['Christians']):,}, unaffiliated "
          f"{int(p['Religiously_unaffiliated']):,}")
    christ = ("Orthodox (Kazakhstan's coefficient)", "Armenian Apostolic", "Georgian Orthodox")
    got = dict(people=people, muslims=int(tot["Muslim"]),
               orthodox=int(tot.get("Orthodox (Kazakhstan's coefficient)", 0)),
               christians=int(sum(tot.get(k, 0) for k in christ)),
               unaffiliated=int(tot.get("Non-believer (Kazakhstan's coefficient)", 0)),
               unknown=int(tot.get(UNKNOWN, 0)), jews=int(tot.get("Jewish", 0)),
               russians=round(RUSSIAN_SHARE_2020 * TOTAL_2020),
               dushanbe_non_muslim_pct=float(round(100 * nm.get("TJ-DU", 0) / pop["TJ-DU"], 2)),
               pew_christians=int(p["Christians"]))
    print(f"  note_public's figures: {got}")
    if got != NOTE:
        raise SystemExit(f"note_public's figures are {NOTE}; the build gives {got}")


if __name__ == "__main__":
    main()
