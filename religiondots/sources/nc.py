"""New Caledonia: no census or survey asks religion, so the levels are Pew Research Center's 2020
estimate (from the World Religion Database) and the geography is a model: J.-M. Kohler's count of
church members by community and commune at the start of 1978 (ORSTOM 1979), applied to each
commune's communities in the 2019 census.

Reads, from data/raw/nc/ (`--fetch` downloads them):

  * ISEE, `rp-structure-communautes.xls`, sheet `commune`: the 2019 census's population by
    commune and community (European, Kanak, Wallisian and Futunian, several communities, other
    and not declared), 33 communes, with the provincial and national rows;
  * ISEE, `rp-population-communautes.xls`, sheet `provinces`: the 2019 census by province and
    every community ISEE prints (Tahitian, Indonesian, Vietnamese, Ni-Vanuatu included), and sheet
    `NC`, the national row;
  * J.-M. Kohler, *Religions et dynamique sociale en Nouvelle-Calédonie, Fascicule II : Effectifs
    et pratique religieuse* (ORSTOM Nouméa, 1979; IRD Horizon `divers18-07/17982.pdf`), a scan:
    Tableau 1 (p.16, church by community, territory), Tableau 2 (p.23, Catholics and Protestants
    by commune and community; Melanesians by commune of ORIGIN), Tableaux 10 and 11 (pp.30-31,
    Melanesians of the Grande Terre and of each Loyalty island). Transcribed here and pinned
    against the tables' own printed totals; the text layer is OCR and is not read;
  * data/raw/estimates/pew.zip (Pew Research Center 2020).

Writes data/normalized/nc.csv. `sources/nc.md` is the record.

## WHY THIS ROUTE

France's statistics law keeps religion off the census: the 1996, 2009 and 2019 forms do not ask
(sources.md §scout-2026-09-14-asia-oceania), and Kohler wrote in 1979 that the last
administrative censuses had not recorded it either (p.8). No survey that asks has been found
(sources/nc.md §1). Pew's figure for New Caledonia is the World Religion Database's (Pew 2012,
Appendix B), and gives no Catholic or Protestant split and no geography. A single national mix
would draw Lifou, whose Melanesians were 87% Protestant in Kohler's count, at the territory's
Catholic majority. Kohler is the only source that places either church, and it does so by
community and commune; the 2019 census counts the communities by commune. So spec §14.12's
ethnicity model, with Kohler's counts as the coefficients.

## THE CONSTRUCTION

  1. Seeds per commune and community, from Kohler (counts of members, start of 1978):
     Europeans, Wallisians and Futunians, Tahitians, Indonesians, Vietnamese, Ni-Vanuatu and
     `Autres` at their Tableau 1 territory mix; Kanak at their commune's Tableau 2 mix of
     Catholics and Protestants (by origin), with Tableau 1's Melanesian minorities and `Divers`
     at the territory rate. The census's commune table prints Tahitians, Indonesians, Vietnamese
     and Ni-Vanuatu only inside "other and not declared"; each commune's cell is split at its
     province's proportions (the province sheet). Several communities, and other or undeclared
     people who are none of the four, take the commune's mix of everyone else.
  2. Kanak who live outside their commune of origin. Kohler counts urban Melanesians at their
     village of origin (p.9), so Nouméa and Dumbéa have none. Each origin commune's Kohler count
     is scaled by the Kanak population's growth (2019 census Kanak over Kohler's Melanesian
     Catholics and Protestants); where a commune holds fewer Kanak in 2019 than that, the
     difference is taken as having left, at the commune's mix; where it holds more, the extra
     people take the mix of everyone who left. The two pools are equal by construction.
     Kouaoua (1995, from Canala) and Poum (1977, from Koumac) share their parent's origin row.
  3. National levels from Pew 2020: Christian 85.10, unaffiliated 10.52, Muslim 2.76, other
     0.95, Buddhist 0.62, Jewish 0.04 (percent). Pew's `other` is split by the World Religion
     Database's own parts (ARDA, 2025, read 2026-10-03): Baha'i 0.37 of 0.96, the rest (ethnic
     and new religionists) with the Jews to `Autres religions`.
  4. Iterative proportional fitting of the seeds to those national levels and to each commune's
     2019 population, by group: Christians as one group (then split inside each commune at the
     seeds' Catholic, Protestant and minority-church shares), Kohler's `Divers` for no religion,
     his Muslims, his Baha'is; Buddhists seeded flat on the "other and not declared" column,
     `Autres religions` flat on everyone, since nothing places either.

Every row is `modelled` (spec §7b).

Usage:
    python sources/nc.py --fetch    download the ISEE workbooks and Kohler's fascicle
    python sources/nc.py            rebuild data/normalized/nc.csv and print the checks
"""

import io
import os
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np
import pandas as pd

from afrobarometer import round_within_rows

RAW = os.path.join(ROOT, "data", "raw", "nc")
STRUCT = os.path.join(RAW, "rp-structure-communautes.xls")
STRUCT_URL = "https://www.isee.nc/sites/default/files/2025-11/rp-structure-communautes.xls"
POPCOM = os.path.join(RAW, "rp-population-communautes.xls")
POPCOM_URL = "https://www.isee.nc/sites/default/files/2025-11/rp-population-communautes.xls"
KOHLER = os.path.join(RAW, "kohler1979_fasc2.pdf")
KOHLER_URL = "https://horizon.documentation.ird.fr/exl-doc/pleins_textes/divers18-07/17982.pdf"
PEW = os.path.join(ROOT, "data", "raw", "estimates", "pew.zip")
OUT = os.path.join(ROOT, "data", "normalized", "nc.csv")

# ---- Kohler 1979, Tableau 1 (p.16): members by church and community, start of 1978. Major
# churches rounded to 50, minorities to 5 (the table's footnotes). Columns in the table's order.
T1_COLS = ["Mélan. cal.", "Mélan. hébr.", "Wallis.", "Tahit.", "Européens", "Indon.", "Vietn.",
           "Autres"]
T1 = {
    "Catholiques":   [28_500, 150, 10_000, 2_100, 46_700, 300, 1_800, 1_450],
    "Protestants":   [28_900, 400, 0, 3_200, 1_000, 100, 0, 400],
    "Musulmans":     [0, 0, 0, 0, 100, 4_000, 0, 150],
    "A. de Dieu":    [200, 100, 10, 60, 120, 130, 15, 45],
    "Adventistes":   [80, 150, 5, 125, 130, 0, 0, 160],
    "T. de Jéhovah": [50, 5, 20, 80, 380, 20, 40, 25],
    "Mormons":       [25, 0, 5, 375, 100, 0, 10, 15],
    "Baha'is":       [235, 0, 60, 0, 15, 0, 0, 10],
    "Sanitos":       [0, 0, 0, 225, 25, 10, 0, 0],
    "Divers":        [410, 295, 0, 235, 2_930, 540, 135, 145],
}
T1_COL_TOTALS = [58_400, 1_100, 10_100, 6_400, 51_500, 5_100, 2_000, 2_400]
T1_ROW_TOTALS = {"Catholiques": 91_000, "Protestants": 34_000, "Musulmans": 4_250,
                 "A. de Dieu": 680, "Adventistes": 650, "T. de Jéhovah": 620, "Mormons": 530,
                 "Baha'is": 320, "Sanitos": 260, "Divers": 4_690}
T1_TOTAL = 137_000
T1_ROUNDING = 250   # row totals are printed rounded; the cells sum to within this of them

# ---- Kohler 1979, Tableau 2 (p.23): Melanesian Catholics (C) and Protestants (P) by commune of
# ORIGIN ("c'est le critère d'origine, et non celui de résidence"). Blank cells are 0; Païta's P
# cell prints a dot. Nouméa, Dumbéa and Farino have no Melanesian row entries.
T2_MEL = {
    "Nouméa": (0, 0), "Dumbéa": (0, 0), "Mont-Dore": (991, 0), "Bélep": (993, 0),
    "Boulouparis": (466, 0), "Bourail": (746, 368), "Canala": (1_542, 1_011), "Farino": (0, 0),
    "Hienghène": (1_492, 567), "Houaïlou": (1_073, 1_940), "Kaala-Gomen": (390, 683),
    "Koné": (875, 1_068), "Koumac": (451, 914), "La Foa": (383, 11), "Moindou": (309, 68),
    "Ouégoa": (1_385, 43), "Païta": (828, 0), "Ile des Pins": (1_439, 0),
    "Poindimié": (1_965, 777), "Ponérihouen": (949, 1_547), "Pouébo": (2_143, 116),
    "Pouembout": (86, 0), "Poya": (794, 192), "Sarraméa": (327, 79), "Thio": (965, 0),
    "Touho": (897, 824), "Voh": (93, 950), "Yaté": (1_287, 0), "Ouvéa": (2_953, 1_879),
    "Lifou": (1_607, 10_971), "Maré": (1_078, 4_918),
}
T2_MEL_TOTALS = (28_507, 28_926)
# Tableau 11 (p.31), Loyalty Melanesians by island, rounded to 10: (C, P, minorities, total).
T11 = {"Ouvéa": (2_950, 1_880, 20, 4_850), "Lifou": (1_610, 10_970, 10, 12_590),
       "Maré": (1_080, 4_920, 80, 6_080)}
# Tableau 10 (p.30), rounded to 50: Grande Terre and Loyalty Melanesians (C, P).
T10 = {"Grande Terre": (22_850, 11_150), "Iles Loyauté": (5_650, 17_750)}
# Communes created after 1978, counted in their parent's Tableau 2 row.
ORIGIN_OF = {"Kouaoua": "Canala", "Poum": "Koumac"}

# ---- 2019 census communes: the ISEE name, the commune code (data.gouv.nc), the province.
COMMUNES = {
    "Bélep": ("98801", "Nord"), "Boulouparis": ("98802", "Sud"), "Bourail": ("98803", "Sud"),
    "Canala": ("98804", "Nord"), "Dumbéa": ("98805", "Sud"), "Farino": ("98806", "Sud"),
    "Hienghène": ("98807", "Nord"), "Houaïlou": ("98808", "Nord"),
    "Ile des Pins (L')": ("98809", "Sud"), "Kaala-Gomen": ("98810", "Nord"),
    "Koné": ("98811", "Nord"), "Kouaoua": ("98833", "Nord"), "Koumac": ("98812", "Nord"),
    "La Foa": ("98813", "Sud"), "Lifou": ("98814", "Iles"), "Maré": ("98815", "Iles"),
    "Moindou": ("98816", "Sud"), "Mont-Dore (Le)": ("98817", "Sud"), "Nouméa": ("98818", "Sud"),
    "Ouégoa": ("98819", "Nord"), "Ouvéa": ("98820", "Iles"), "Païta": ("98821", "Sud"),
    "Poindimié": ("98822", "Nord"), "Ponérihouen": ("98823", "Nord"),
    "Pouébo": ("98824", "Nord"), "Pouembout": ("98825", "Nord"), "Poum": ("98826", "Nord"),
    "Poya": ("98827", "Nord"), "Sarraméa": ("98828", "Sud"), "Thio": ("98829", "Sud"),
    "Touho": ("98830", "Nord"), "Voh": ("98831", "Nord"), "Yaté": ("98832", "Sud"),
}
SHORT = {"Ile des Pins (L')": "Ile des Pins", "Mont-Dore (Le)": "Mont-Dore"}
COM_COLS = ["Européenne", "Kanak", "Wallisienne et Futunienne", "Plusieurs communautés *",
            "Autre * et non déclarée", "Total"]
PROVINCE_ROWS = {"Iles": "Province Iles Loyauté", "Nord": "Province Nord", "Sud": "Province Sud"}
NATIONAL_2019 = 271_407
NAMED_OTHER = ["Tahitienne", "Indonésienne", "Vietnamienne", "Ni-Vanuatu"]
# Poya straddles the Nord/Sud line; ISEE's 2019 commune sheet prints it once and its provincial
# rows count its southern part in Sud. Measured: the communes sum to each province row except
# for that part.
POYA_SUD_TOL = 400

# ---- Pew 2020, asserted.
PEW_2020 = {"Christians": 85.102859, "Muslims": 2.760157, "Religiously_unaffiliated": 10.523974,
            "Buddhists": 0.624888, "Hindus": 0.0, "Jews": 0.035027, "Other_religions": 0.953095}
# World Religion Database parts of `other` (ARDA national profile u=162c, 2025, read 2026-10-03).
WRD_OTHER = {"Baha'is": 0.37, "Ethnic religionists": 0.18, "New religionists": 0.41}
WRD_CHRISTIAN = {"Catholics": 51.04, "Independents": 9.52, "Protestants": 15.32,
                 "Unaffiliated Christians": 9.23}

# Drawn categories (taxonomy/nc2020.py maps each to a node).
CHRISTIAN = ["Catholiques", "Protestants (Églises évangéliques océaniennes)",
             "Protestants (autres)", "Assemblées de Dieu", "Adventistes", "Témoins de Jéhovah",
             "Mormons et Sanitos"]
OTHERS = ["Sans religion", "Musulmans", "Baha'is", "Bouddhistes", "Autres religions"]
CATS = CHRISTIAN + OTHERS
YEAR = 2020
SOURCE_ID = "nc_pew2020_kohler1978_census2019"
IPF_TOL = 1e-9
# Measured 2026-10-03 and asserted, so note_public cannot drift from the data.
NOTE = dict(census=271_407, catholic=59.1, protestant=24.6, loyalty_protestant=71.1,
            noumea_none=14.1)
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for url, dst, minsize in ((STRUCT_URL, STRUCT, 100_000), (POPCOM_URL, POPCOM, 30_000),
                              (KOHLER_URL, KOHLER, 1_000_000)):
        if os.path.exists(dst) and os.path.getsize(dst) > minsize:
            continue
        print("GET", url)
        r = requests.get(url, timeout=600, headers={"User-Agent": UA})
        r.raise_for_status()
        with open(dst + ".part", "wb") as fh:
            fh.write(r.content)
        os.replace(dst + ".part", dst)


# ---------------------------------------------------------------- Kohler
def check_kohler():
    cols = np.array(list(T1.values()))
    if list(cols.sum(axis=0)) != T1_COL_TOTALS:
        raise SystemExit(f"Tableau 1 columns sum to {list(cols.sum(axis=0))}, printed "
                         f"{T1_COL_TOTALS}")
    for k, v in T1.items():
        if abs(sum(v) - T1_ROW_TOTALS[k]) > T1_ROUNDING:
            raise SystemExit(f"Tableau 1 row {k} sums to {sum(v)}, printed {T1_ROW_TOTALS[k]}")
    if abs(cols.sum() - T1_TOTAL) > T1_ROUNDING:
        raise SystemExit(f"Tableau 1 sums to {cols.sum()}, printed {T1_TOTAL}")
    c = sum(v[0] for v in T2_MEL.values())
    p = sum(v[1] for v in T2_MEL.values())
    if (c, p) != T2_MEL_TOTALS:
        raise SystemExit(f"Tableau 2 Melanesian columns sum to {(c, p)}, printed {T2_MEL_TOTALS}")
    if abs(c - T1["Catholiques"][0]) > 50 or abs(p - T1["Protestants"][0]) > 50:
        raise SystemExit("Tableau 2's Melanesian totals do not round to Tableau 1's")
    for isl, (tc, tp, _, _) in T11.items():
        ec, ep = T2_MEL[isl]
        if abs(ec - tc) > 10 or abs(ep - tp) > 10:
            raise SystemExit(f"{isl}: Tableau 2 {(ec, ep)} against Tableau 11 {(tc, tp)}")
    loy = [sum(T2_MEL[i][k] for i in T11) for k in (0, 1)]
    gt = [c - loy[0], p - loy[1]]
    for name, got in (("Iles Loyauté", loy), ("Grande Terre", gt)):
        want = T10[name]
        if any(abs(g - w) > 50 for g, w in zip(got, want)):
            raise SystemExit(f"{name}: Tableau 2 gives {got}, Tableau 10 {want}")
    print(f"  Kohler 1979 Tableau 1 read back: {cols.sum():,} members in 8 communities, every "
          f"column equal to its printed total; Tableau 2 Melanesians {c:,} Catholic, {p:,} "
          f"Protestant, agreeing with Tableaux 1, 10 and 11")
    cath = cols[0].sum() / cols.sum()
    prot = cols[1].sum() / cols.sum()
    print(f"      territory, start of 1978: Catholic {100 * cath:.1f}%, Protestant "
          f"{100 * prot:.1f}%, Muslim {100 * cols[2].sum() / cols.sum():.1f}%, Divers (with "
          f"the atheists) {100 * cols[-1].sum() / cols.sum():.1f}%")


def t1_vector(col):
    """Tableau 1 column as a seed over CATS (Baha'is kept for their own group)."""
    j = T1_COLS.index(col)
    v = {k: T1[k][j] for k in T1}
    oceanian = col in ("Mélan. cal.", "Mélan. hébr.", "Tahit.")
    return pd.Series({
        "Catholiques": v["Catholiques"],
        "Protestants (Églises évangéliques océaniennes)": v["Protestants"] if oceanian else 0,
        "Protestants (autres)": 0 if oceanian else v["Protestants"],
        "Assemblées de Dieu": v["A. de Dieu"],
        "Adventistes": v["Adventistes"],
        "Témoins de Jéhovah": v["T. de Jéhovah"],
        "Mormons et Sanitos": v["Mormons"] + v["Sanitos"],
        "Sans religion": v["Divers"],
        "Musulmans": v["Musulmans"],
        "Baha'is": v["Baha'is"],
        "Bouddhistes": 0,
        "Autres religions": 0,
    }, dtype=float)


# ---------------------------------------------------------------- census
def _int(x):
    if isinstance(x, str):
        return None if x.strip() == "ss" else int(x.replace(" ", ""))
    return int(x)


def read_communes():
    raw = pd.read_excel(STRUCT, sheet_name="commune", header=None)
    head = raw.index[raw[0].astype(str).str.strip() == "En 2019"]
    if len(head) != 1:
        raise SystemExit("the 2019 block of the commune sheet was not found")
    h = head[0]
    if [str(x).strip() for x in raw.iloc[h, 1:7]] != COM_COLS:
        raise SystemExit(f"2019 header reads {list(raw.iloc[h, 1:7])}")
    rows, extra = {}, {}
    for i in range(h + 1, h + 60):
        name = str(raw.iloc[i, 0]).strip()
        if name in ("nan", "Unité : habitant"):
            break
        vals = [_int(x) for x in raw.iloc[i, 1:7]]
        (rows if name in COMMUNES else extra)[name] = vals
    if set(rows) != set(COMMUNES):
        raise SystemExit(f"2019 communes: missing {set(COMMUNES) - set(rows)}")
    df = pd.DataFrame(rows, index=COM_COLS).T
    # secret cells: fitted to each row's residual and each column's shortfall against the
    # national row (a small IPF over the `ss` cells only), then rounded inside each row
    nat = extra["Nouvelle-Calédonie"]
    cols = COM_COLS[:-1]
    known = df[cols].astype(float)
    mask = known.isna()
    row_t = df["Total"].astype(float) - known.fillna(0).sum(axis=1)
    col_t = pd.Series([nat[COM_COLS.index(c)] for c in cols], index=cols) - known.fillna(0).sum()
    if (row_t[mask.any(axis=1)] < 0).any() or (col_t[mask.any(axis=0)] < 0).any():
        raise SystemExit("secret cells would be negative")
    x = mask.astype(float)
    for _ in range(2000):
        x = x.mul((row_t / x.sum(axis=1)).where(x.sum(axis=1) > 0, 0), axis=0)
        cs = x.sum(axis=0)
        x = x.mul((col_t / cs).where(cs > 0, 0), axis=1)
    rows_ss = mask.any(axis=1)
    fill = round_within_rows(x[rows_ss].mul(row_t[rows_ss] / x[rows_ss].sum(axis=1), axis=0))
    for name in fill.index:
        for c in cols:
            if mask.loc[name, c]:
                df.loc[name, c] = int(fill.loc[name, c])
    filled = int(row_t[rows_ss].sum())
    df = df.astype(int)
    if (df[COM_COLS[:-1]].sum(axis=1) != df["Total"]).any():
        raise SystemExit("a commune's communities do not sum to its total")
    if int(df["Total"].sum()) != NATIONAL_2019 or nat[-1] != NATIONAL_2019:
        raise SystemExit(f"communes sum to {df['Total'].sum():,}, national row {nat[-1]:,}")
    gap = [abs(int(df[c].sum()) - nat[COM_COLS.index(c)]) for c in COM_COLS[:-1]]
    if max(gap) > 20:
        raise SystemExit(f"after filling secret cells the columns miss the national row by {gap}")
    df["province"] = [COMMUNES[n][1] for n in df.index]
    for p, label in PROVINCE_ROWS.items():
        got = int(df.loc[df["province"] == p, "Total"].sum())
        want = extra[label][-1]
        if abs(got - want) > (POYA_SUD_TOL if p != "Iles" else 0):
            raise SystemExit(f"{p}: communes {got:,}, province row {want:,}")
    print(f"  ISEE 2019 commune sheet read back: 33 communes, {NATIONAL_2019:,} people; "
          f"{filled} people in secret cells filled from row residuals; provinces within "
          f"{POYA_SUD_TOL} of their rows (Poya's southern part)")
    return df


def read_provinces():
    raw = pd.read_excel(POPCOM, sheet_name="provinces", header=None)
    yr = raw.index[raw[0].astype(str).str.strip() == "Communauté d'appartenance"][0]
    years = raw.iloc[yr].tolist()
    c0 = years.index(2019)
    if [str(x) for x in raw.iloc[yr + 1, c0:c0 + 4]] != ["Iles", "Nord", "Sud", "Total"]:
        raise SystemExit("province sheet: the 2019 columns moved")
    out = {}
    for i in range(yr + 2, yr + 20):
        name = str(raw.iloc[i, 0]).strip()
        if name == "nan":
            break
        out[name] = [int(x) for x in raw.iloc[i, c0:c0 + 4]]
    t = pd.DataFrame(out, index=["Iles", "Nord", "Sud", "Total"]).T
    if int(t.loc["Total", "Total"]) != NATIONAL_2019:
        raise SystemExit("province sheet: 2019 total")
    nc = pd.read_excel(POPCOM, sheet_name="NC", header=None)
    r = nc.index[nc[0].astype(str).str.strip() == "Communauté d'appartenance"][0]
    j = nc.iloc[r].tolist().index(2019.0)
    for g in NAMED_OTHER:
        k = nc.index[nc[0].astype(str).str.strip() == g][0]
        if int(nc.iloc[k, j]) != int(t.loc[g, "Total"]):
            raise SystemExit(f"{g}: province sheet {t.loc[g, 'Total']}, NC sheet {nc.iloc[k, j]}")
    print("  ISEE province sheet, 2019: "
          + ", ".join(f"{g} {int(t.loc[g, 'Total']):,}" for g in NAMED_OTHER)
          + " (equal to the NC sheet)")
    return t


# ---------------------------------------------------------------- model
def kanak_seeds(com):
    """Per-commune Kanak counts split into Catholics and Protestants (step 2 of the docstring)."""
    k = com["Kanak"]
    groups = {}
    for name in com.index:
        o = ORIGIN_OF.get(name, SHORT.get(name, name))
        groups.setdefault(o, []).append(name)
    origin = {o: T2_MEL[o] for o in groups}
    if set(origin) != set(T2_MEL):
        raise SystemExit(f"origin rows unused: {set(T2_MEL) - set(origin)}")
    tot = sum(c + p for c, p in origin.values())
    g = int(k.sum()) / tot
    stay, leave, extra = {}, np.zeros(2), {}
    for o, names in groups.items():
        c, p = origin[o]
        expect = g * (c + p)
        have = int(k[names].sum())
        mix = np.array([c, p], dtype=float) / (c + p) if c + p else np.array([0.0, 0.0])
        s = min(have, expect)
        stay[o] = s * mix
        leave += max(0.0, expect - have) * mix
        extra[o] = have - s
    pool = leave / leave.sum()
    if abs(leave.sum() - sum(extra.values())) > 1e-6 * tot * g:
        raise SystemExit("the Kanak who left and the Kanak who arrived do not balance")
    out = {}
    for o, names in groups.items():
        cp = stay[o] + extra[o] * pool
        share = k[names] / max(int(k[names].sum()), 1)
        for n in names:
            out[n] = cp * float(share[n])
    print(f"  Kanak growth since Kohler: {g:.3f} (2019 census Kanak {int(k.sum()):,} over "
          f"{tot:,} Melanesian Catholics and Protestants); {leave.sum():,.0f} live outside "
          f"their origin's expected count, {100 * pool[1]:.1f}% Protestant among them")
    return pd.DataFrame(out, index=["C", "P"]).T


def seeds(com, prov):
    mel = t1_vector("Mélan. cal.")
    mel_min = mel.drop(["Catholiques", "Protestants (Églises évangéliques océaniennes)"])
    mel_min_share = mel_min / T1_COL_TOTALS[0]
    vec = {c: t1_vector(c) / T1_COL_TOTALS[T1_COLS.index(c)] for c in T1_COLS}
    named = {"Tahitienne": "Tahit.", "Indonésienne": "Indon.", "Vietnamienne": "Vietn.",
             "Ni-Vanuatu": "Mélan. hébr."}
    kan = kanak_seeds(com)
    rows = {}
    for name, r in com.iterrows():
        p = r["province"]
        # province proportions of the four named communities inside "other and not declared"
        o_tot = (prov.loc["Total", p] - prov.loc["Kanak", p] - prov.loc["Européenne", p]
                 - prov.loc["Wallisienne, Futunienne", p] - prov.loc["dont plusieurs communautés", p])
        s = pd.Series(0.0, index=CATS)
        s += r["Européenne"] * vec["Européens"]
        s += r["Wallisienne et Futunienne"] * vec["Wallis."]
        kc, kp = kan.loc[name, "C"], kan.loc[name, "P"]
        kn = r["Kanak"]
        s["Catholiques"] += kc * (1 - mel_min_share.sum())
        s["Protestants (Églises évangéliques océaniennes)"] += kp * (1 - mel_min_share.sum())
        s[mel_min.index] += kn * mel_min_share
        known = 0.0
        for g, col in named.items():
            n = r["Autre * et non déclarée"] * prov.loc[g, p] / o_tot
            s += n * vec[col]
            known += n
        # several communities, and other or undeclared people outside the four: the commune's mix
        rest = r["Plusieurs communautés *"] + r["Autre * et non déclarée"] - known
        if rest < 0:
            raise SystemExit(f"{name}: negative remainder")
        s += rest * s / s.sum()
        rows[name] = s
    m = pd.DataFrame(rows).T
    if not np.allclose(m.sum(axis=1), com["Total"]):
        raise SystemExit("seed rows do not sum to the commune populations")
    return m


def ipf(m, com):
    """Fit the group totals to Pew's national levels and each commune's population."""
    pop = com["Total"].astype(float)
    total = float(pop.sum())
    other = PEW_2020["Other_religions"]
    wsum = sum(WRD_OTHER.values())
    target = {
        "christian": PEW_2020["Christians"],
        "Sans religion": PEW_2020["Religiously_unaffiliated"],
        "Musulmans": PEW_2020["Muslims"],
        "Baha'is": other * WRD_OTHER["Baha'is"] / wsum,
        "Bouddhistes": PEW_2020["Buddhists"],
        "Autres religions": other * (wsum - WRD_OTHER["Baha'is"]) / wsum + PEW_2020["Jews"],
    }
    if abs(sum(target.values()) + PEW_2020["Hindus"] - 100) > 1e-4:
        raise SystemExit(f"Pew's shares sum to {sum(target.values())}")
    target = pd.Series({k: v / 100 * total for k, v in target.items()})
    g = pd.DataFrame({"christian": m[CHRISTIAN].sum(axis=1)})
    for c in OTHERS:
        g[c] = m[c]
    # nothing places these two, so they are seeded flat
    g["Bouddhistes"] = com["Autre * et non déclarée"].astype(float)
    g["Autres religions"] = pop
    g = g[target.index]
    for it in range(5000):
        g = g.mul(target / g.sum(axis=0), axis=1)
        g = g.mul(pop / g.sum(axis=1), axis=0)
        if (g.sum(axis=0) - target).abs().max() < IPF_TOL * total:
            break
    else:
        raise SystemExit("IPF did not converge")
    print(f"  IPF converged in {it + 1} rounds; national targets (Pew 2020 on the 2019 count): "
          + ", ".join(f"{k} {v:,.0f}" for k, v in target.items()))
    within = m[CHRISTIAN].div(m[CHRISTIAN].sum(axis=1), axis=0)
    out = within.mul(g["christian"], axis=0)
    for c in OTHERS:
        out[c] = g[c]
    return out[CATS]


def pew_row():
    with zipfile.ZipFile(PEW) as z:
        name = [n for n in z.namelist() if n.endswith("(percentages).csv")][0]
        t = pd.read_csv(io.BytesIO(z.read(name)))
    r = t[(t["Country"] == "New Caledonia") & (t["Year"] == 2020)].iloc[0]
    got = {k: float(r[k]) for k in PEW_2020}
    if any(abs(got[k] - PEW_2020[k]) > 1e-5 for k in PEW_2020):
        raise SystemExit(f"Pew 2020 New Caledonia row reads {got}, pinned {PEW_2020}")
    print("  Pew 2020 read back: " + ", ".join(f"{k} {v:.2f}" for k, v in got.items()))


def witnesses(counts, com, seed):
    nat = counts.sum()
    tot = nat.sum()
    chr_ = nat[CHRISTIAN].sum()
    prot = nat["Protestants (Églises évangéliques océaniennes)"] + nat["Protestants (autres)"]
    print("\n  witnesses")
    k_chr = sum(sum(T1[k]) for k in ("Catholiques", "Protestants", "A. de Dieu", "Adventistes",
                                     "T. de Jéhovah", "Mormons", "Sanitos"))
    print(f"      Catholics as a share of Christians: drawn {100 * nat['Catholiques'] / chr_:.1f}%; "
          f"WRD 2025 {100 * WRD_CHRISTIAN['Catholics'] / sum(WRD_CHRISTIAN.values()):.1f}% "
          f"(with 22% of its Christians Independent or unaffiliated); Kohler 1978 "
          f"{100 * sum(T1['Catholiques']) / k_chr:.1f}%")
    print(f"      Protestants: drawn {100 * prot / tot:.1f}% of everyone; WRD 2025 "
          f"{WRD_CHRISTIAN['Protestants']:.1f}% plus Independents {WRD_CHRISTIAN['Independents']:.1f}%")
    s0 = seed.sum()
    print(f"      the seed before fitting: Christian {100 * s0[CHRISTIAN].sum() / s0.sum():.1f}%, "
          f"Divers {100 * s0['Sans religion'] / s0.sum():.1f}%, Muslim "
          f"{100 * s0['Musulmans'] / s0.sum():.1f}% (Kohler's 1978 mix on 2019 communities)")
    loy = counts.loc[[n for n in counts.index if COMMUNES[n][1] == "Iles"]]
    for n, r in loy.iterrows():
        c, p = r["Catholiques"], r["Protestants (Églises évangéliques océaniennes)"]
        kc, kp = T2_MEL[n]
        print(f"      {n:<8} drawn Catholic {100 * c / (c + p):4.1f}% of Catholics and "
              f"Protestants; Kohler's Melanesians of origin {100 * kc / (kc + kp):4.1f}%")
    sh = counts.div(counts.sum(axis=1), axis=0)
    print("\n  per commune (percent): Catholic, Protestant, no religion, Muslim")
    for n in sh.sort_values("Catholiques").index:
        pr = sh.loc[n, "Protestants (Églises évangéliques océaniennes)"] + sh.loc[n, "Protestants (autres)"]
        print(f"      {n:<20} {int(com.loc[n, 'Total']):>7,}  {100 * sh.loc[n, 'Catholiques']:5.1f} "
              f"{100 * pr:5.1f} {100 * sh.loc[n, 'Sans religion']:5.1f} "
              f"{100 * sh.loc[n, 'Musulmans']:5.1f}")


def main():
    if "--fetch" in sys.argv or not all(os.path.exists(p) for p in (STRUCT, POPCOM, KOHLER)):
        fetch()
    check_kohler()
    pew_row()
    com = read_communes()
    prov = read_provinces()
    seed = seeds(com, prov)
    fit = ipf(seed, com)
    counts = round_within_rows(fit)
    if not (counts.sum(axis=1) == com["Total"].reindex(counts.index)).all():
        raise SystemExit("a commune's rounded counts do not sum to its population")
    witnesses(counts, com, seed)

    out = counts.stack().rename("count").reset_index()
    out.columns = ["geo_name", "source_category", "count"]
    out["geo_id"] = out["geo_name"].map(lambda n: COMMUNES[n][0])
    out["geo_level"] = "commune"
    out["basis"] = "estimate"
    out["year"] = YEAR
    out["source_id"] = SOURCE_ID
    out["note"] = ("Pew 2020 national levels; geography and the Christian split from Kohler's "
                   "1978 church counts by community and commune on the 2019 census's "
                   "communities by commune (modelled)")
    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count", "basis", "year",
            "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    by = out.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(f"\nwrote {OUT} ({len(out)} rows, {int(by.sum()):,} people, 33 communes)")
    for c, k in by.items():
        print(f"      {c:<48} {k:>8,}  {100 * k / by.sum():5.2f}%")

    loy = counts.loc[[n for n in counts.index if COMMUNES[n][1] == "Iles"]].sum()
    got = dict(
        census=int(by.sum()),
        catholic=float(round(100 * by["Catholiques"] / by.sum(), 1)),
        protestant=round(100 * (by["Protestants (Églises évangéliques océaniennes)"]
                                + by["Protestants (autres)"]) / by.sum(), 1),
        loyalty_protestant=round(100 * loy["Protestants (Églises évangéliques océaniennes)"]
                                 / loy.sum(), 1),
        noumea_none=round(100 * counts.loc["Nouméa", "Sans religion"]
                          / counts.loc["Nouméa"].sum(), 1),
    )
    print(f"  note_public's figures: {got}")
    if NOTE and got != NOTE:
        raise SystemExit(f"note_public's figures are {NOTE}; the build gives {got}")


if __name__ == "__main__":
    main()
