"""Japan: JGSS 2021-2024's national figures spread over the 47 prefectures. Every row `modelled`.

    python sources/jp_alloc.py                      Christians on NHK 1996's pattern (default)
    python sources/jp_alloc.py --christians roll    Christians on the Agency roll's pattern

Writes `data/normalized/jp_prefecture_allocated.csv`, one row per (prefecture, JGSS code), in
people. `countries.py`'s `_jp_counts` reads it. Needs `sources/jp.py` (the national table),
`sources/jp_checks.py` (the measured 1996 chart) and `sources/jp_grid.py` (population) first.

**ANITA, 2026-09-14, ask 014:** *"if nothing is more recent, 1996 is better than nothing. we can
use it for allocating"*, and a block pattern *"can come from different survey"*. So three sources,
each deciding one thing and nothing else:

    HOW MANY, nationally       JGSS 2021H-2024N pooled, 10,612 respondents (sources/jp.py).
                               Every code's national total is exactly its share of them times
                               the 2024 population, and nothing below moves a national figure.
    HOW RELIGIOUS each block   JGSS-2015, believe or family religion, by the survey's six
                               sampling blocks (Iwai, Pew 2017; sources/jp.md §2). Relative
                               only: each block's 2015 share over the 2015 national share,
                               rescaled so the blocks land on 2021-2024's national figure.
    WHERE, inside that         NHK 全国県民意識調査 1996, as measured off 社会実情データ図録 7770
                               (sources/jp_checks.py): the share naming each school or religion
                               in every prefecture, about 600 answers each.

**THE CONSTRUCTION.**

1. Non-response (JGSS 9999 and 8700, 3.60%) is laid on every prefecture at the national rate
   and is not drawn.
2. Each prefecture's share naming any religion starts at NHK 1996's. Inside each JGSS block,
   every prefecture's log-odds moves by the same amount until the block hits its target, so
   inside Chubu Toyama stays far above Shizuoka and only the block's level changes.
3. The religious people are split among the religions by iterative proportional fitting. Rows
   are each prefecture's religious total from step 2; columns are the JGSS national totals;
   the seed is NHK 1996's share for the answer a JGSS code falls under, times population.
   Several codes share one NHK answer and so share its pattern: Tendai and Shingon; Jodo-shu
   and Jodo Shinshu; Buddhist with no school takes the five Buddhist answers summed.
4. A code with no NHK answer of its own (other new religions, Islam, combinations, ancestor
   veneration), or whose answer does not differ between prefectures beyond sampling noise, is
   seeded flat on each prefecture's religious population. The noise test is a chi-square over
   the 47 prefectures on the measured share times the achieved sample; p >= 0.001 is flat.

**A MEASURED ZERO IS READ AS 0.15.** The chart cannot tell a segment under about 0.3 points from a
zero-height one (sources/jp_checks.py), so a 0.0 is taken at the middle of what it could be.
Left at zero, the fitting would keep it at zero, and Fukushima and Saga would have no Christians.

**CHRISTIANS DEFAULT TO NHK 1996, NOT THE ROLL** (sources/jp.md §8). The roll's believers per
Christian body are 886 in Tokyo and 578 in Kanagawa against a median of 76, because national
bodies file their whole membership at their registered address; on the roll's pattern Tokyo
would be Japan's most Christian prefecture at 4.7% and Saitama next door 0.4%. `--christians
roll` is here so the other version can be looked at.

**WHAT THIS CANNOT DO.** It cannot see anything that changed between 1996 and 2021-2024 below the
block level, and it cannot separate what the 1996 survey asked together. A 2% group rests on about
twelve answers per prefecture, so its prefecture detail is largely sampling noise that passed the
chi-square because a few prefectures differ strongly.
"""

import csv
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]], scipy below
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "jp")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "taxonomy"))

from jp_grid import PREFS, unit_of, prefecture_population   # noqa: E402
from jp2024 import resolve                                  # noqa: E402

JP_CSV = os.path.join(ROOT, "data", "normalized", "jp.csv")
NHK = os.path.join(RAW, "nhk1996_7770_measured.csv")
ROLL = os.path.join(RAW, "aca_r7_table2_2_prefecture.xls")
OUT = os.path.join(ROOT, "data", "normalized", "jp_prefecture_allocated.csv")

N_RESP = 10_612
NONRESPONSE = {9999, 8700}

# The chart's nine answers, as sources/jp_checks.py names them.
ANSWERS = ["tendai_shingon", "jodo_shinshu", "zen", "nichiren", "soka_gakkai",
           "rissho_koseikai", "other_buddhist", "shinto", "christian"]
BUDDHIST = ["tendai_shingon", "jodo_shinshu", "zen", "nichiren", "other_buddhist"]

# node -> the NHK answer whose prefecture pattern it takes. Anything not here is flat.
PATTERN = {
    "buddhism.mahayana.tendai": "tendai_shingon",
    "buddhism.mahayana.shingon": "tendai_shingon",
    "buddhism.mahayana.pureland": "jodo_shinshu",
    "buddhism.mahayana.zen": "zen",
    "buddhism.mahayana.nichiren": "nichiren",
    "buddhism.mahayana": "buddhist_any",
    "shinto": "shinto",
}
# Two new religions the 1996 card named; the rest of eastasiannew.japanese is flat.
CODE_PATTERN = {2905: "soka_gakkai", 2817: "rissho_koseikai"}

# Achieved sample per prefecture. sources.md §11an summed the book's regional n for the seven it
# could read; the other forty are taken at 600, which is what those seven average to.
N_1996 = {"北海道": 596, "青森": 535, "岩手": 603, "宮城": 671, "秋田": 698, "山形": 567, "福島": 601}
N_DEFAULT = 600
ZERO_READ = 0.15
P_FLAT = 1e-3

# JGSS's six sampling blocks and JGSS-2015's believe-or-family-religion share in each (Iwai 2017).
# Mie is in Kinki, as sources/jp.md §2 aggregated NHK 1996 for the comparison.
BLOCKS = {
    "Hokkaido/Tohoku": (25.9, ["北海道", "青森", "岩手", "宮城", "秋田", "山形", "福島"]),
    "Kanto": (24.7, ["茨城", "栃木", "群馬", "埼玉", "千葉", "東京", "神奈川"]),
    "Chubu": (34.3, ["新潟", "富山", "石川", "福井", "山梨", "長野", "岐阜", "静岡", "愛知"]),
    "Kinki": (31.8, ["三重", "滋賀", "京都", "大阪", "兵庫", "奈良", "和歌山"]),
    "Chugoku/Shikoku": (36.0, ["鳥取", "島根", "岡山", "広島", "山口", "徳島", "香川", "愛媛", "高知"]),
    "Kyushu": (36.9, ["福岡", "佐賀", "長崎", "熊本", "大分", "宮崎", "鹿児島", "沖縄"]),
}

EN = {"北海道": "Hokkaido", "青森": "Aomori", "岩手": "Iwate", "宮城": "Miyagi", "秋田": "Akita",
      "山形": "Yamagata", "福島": "Fukushima", "茨城": "Ibaraki", "栃木": "Tochigi", "群馬": "Gunma",
      "埼玉": "Saitama", "千葉": "Chiba", "東京": "Tokyo", "神奈川": "Kanagawa", "新潟": "Niigata",
      "富山": "Toyama", "石川": "Ishikawa", "福井": "Fukui", "山梨": "Yamanashi", "長野": "Nagano",
      "岐阜": "Gifu", "静岡": "Shizuoka", "愛知": "Aichi", "三重": "Mie", "滋賀": "Shiga",
      "京都": "Kyoto", "大阪": "Osaka", "兵庫": "Hyogo", "奈良": "Nara", "和歌山": "Wakayama",
      "鳥取": "Tottori", "島根": "Shimane", "岡山": "Okayama", "広島": "Hiroshima", "山口": "Yamaguchi",
      "徳島": "Tokushima", "香川": "Kagawa", "愛媛": "Ehime", "高知": "Kochi", "福岡": "Fukuoka",
      "佐賀": "Saga", "長崎": "Nagasaki", "熊本": "Kumamoto", "大分": "Oita", "宮崎": "Miyazaki",
      "鹿児島": "Kagoshima", "沖縄": "Okinawa"}


def _logit(p):
    return np.log(p / (1 - p))


def _expit(x):
    return 1 / (1 + np.exp(-x))


def _spearman(x, y):
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])


def pattern_of(code, node):
    if code in NONRESPONSE or node is None:
        return None
    if code in CODE_PATTERN:
        return CODE_PATTERN[code]
    if node == "unaffiliated":
        return "none"
    if node.startswith("christianity") or node == "unification":
        return "christian"
    return PATTERN.get(node, "flat")


def jgss():
    df = pd.read_csv(JP_CSV)
    df["code"] = df["source_category"].str.extract(r"^(\d{4})")[0].astype(int)
    if int(df["count"].sum()) != N_RESP:
        raise SystemExit(f"jp.csv holds {df['count'].sum():,} respondents, expected {N_RESP:,}")
    df["node"] = df["source_category"].map(resolve)
    lost = df[df["node"].isna() & ~df["code"].isin(NONRESPONSE)]
    if len(lost):
        raise SystemExit(f"codes with no node: {list(lost['source_category'])}")
    df["pattern"] = [pattern_of(c, n) for c, n in zip(df["code"], df["node"])]
    return df


def nhk():
    t = pd.read_csv(NHK).set_index("unit")
    if list(t.index) != ["全国"] + PREFS:
        raise SystemExit("the measured chart's rows are not the nation plus JIS order")
    t = t.loc[PREFS].copy()
    t["buddhist_any"] = t[BUDDHIST].sum(axis=1)
    return t


def noise_test(t):
    from scipy.stats import chi2_contingency
    n = np.array([N_1996.get(p, N_DEFAULT) for p in PREFS], float)
    out = {}
    for g in ANSWERS + ["buddhist_any"]:
        yes = t[g].to_numpy() / 100 * n
        chi2, p, dof, _ = chi2_contingency(np.vstack([yes, n - yes]))
        out[g] = (chi2, dof, p)
    return out


def roll_share(pop):
    import xlrd
    sh = xlrd.open_workbook(ROLL).sheet_by_name("キリスト教系宗教団体")
    bel = {}
    for r in range(sh.nrows):
        v = sh.row_values(r)
        if str(v[1]).strip() in PREFS:
            bel[str(v[1]).strip()] = float(v[20])
    if len(bel) != 47:
        raise SystemExit("roll: the Christian sheet does not list 47 prefectures")
    return np.array([100 * bel[p] / pop[p] for p in PREFS])


def allocate(christians):
    df = jgss()
    t = nhk()
    pop = prefecture_population()
    popv = np.array([pop[p] for p in PREFS], float)
    total = popv.sum()
    idx = {p: i for i, p in enumerate(PREFS)}

    # --- the noise test ------------------------------------------------------------------
    tests = noise_test(t)
    print("NHK 1996 answers, chi-square across 47 prefectures (n about 600 each):")
    flat = set()
    for g, (chi2, dof, p) in tests.items():
        verdict = "pattern" if p < P_FLAT else "FLAT"
        if p >= P_FLAT:
            flat.add(g)
        print(f"    {g:<16} national {100 * (t[g] * popv).sum() / total / 100:6.2f}%  "
              f"chi2 {chi2:8.1f} on {dof}  p {p:9.2e}  {verdict}")

    patterns = {}
    for g in set(df["pattern"].dropna()) - {"none", "flat"}:
        if g == "christian" and christians == "roll":
            patterns[g] = roll_share(pop)
        elif g not in flat:
            patterns[g] = np.maximum(t[g].to_numpy(), ZERO_READ)
    print(f"  drawn with a prefecture pattern: {sorted(patterns)}; Christians from "
          f"{'the Agency roll' if christians == 'roll' else 'NHK 1996'}")

    # --- 1. non-response ------------------------------------------------------------------
    nr_resp = int(df.loc[df["code"].isin(NONRESPONSE), "count"].sum())
    drawn = popv * (1 - nr_resp / N_RESP)

    # --- 2. religious share per prefecture, block level from JGSS-2015 --------------------
    if sorted(p for _, ps in BLOCKS.values() for p in ps) != sorted(PREFS):
        raise SystemExit("the six blocks do not partition the 47 prefectures")
    none_resp = int(df.loc[df["node"] == "unaffiliated", "count"].sum())
    r_nat = 1 - none_resp / (N_RESP - nr_resp)
    f = t["total_measured"].to_numpy() / 100
    drawn_b = {b: drawn[[idx[p] for p in ps]].sum() for b, (_, ps) in BLOCKS.items()}
    k = r_nat * drawn.sum() / sum(drawn_b[b] * s / 100 for b, (s, _) in BLOCKS.items())
    r = np.empty(47)
    print(f"\nshare naming a religion: JGSS 2021-2024 national {100 * r_nat:.2f}% of answers")
    nhk_b, jgss_b = [], []
    for b, (s, ps) in BLOCKS.items():
        ii = [idx[p] for p in ps]
        target = s / 100 * k
        lo, hi = -10.0, 10.0
        for _ in range(200):
            mid = (lo + hi) / 2
            got = (drawn[ii] * _expit(_logit(f[ii]) + mid)).sum() / drawn_b[b]
            lo, hi = (mid, hi) if got < target else (lo, mid)
        r[ii] = _expit(_logit(f[ii]) + mid)
        nhk_b.append((drawn[ii] * f[ii]).sum() / drawn_b[b])
        jgss_b.append(s)
        print(f"    {b:<16} JGSS-2015 {s:4.1f}%  NHK 1996 {100 * nhk_b[-1]:4.1f}%  -> drawn "
              f"{100 * r[ii] @ drawn[ii] / drawn_b[b]:5.2f}% (log-odds shift {mid:+.3f})")
    print(f"  blocks, Spearman JGSS-2015 vs NHK 1996: {_spearman(np.array(jgss_b), np.array(nhk_b)):+.3f}")
    relig = drawn * r

    # --- 3. religious codes by iterative proportional fitting ------------------------------
    rel = df[df["pattern"].notna() & (df["pattern"] != "none")].reset_index(drop=True)
    T = rel["count"].to_numpy(float) / N_RESP * total
    if abs(T.sum() - relig.sum()) > 1e-6 * total:
        raise SystemExit("religious rows and columns do not share a total")
    seed = np.empty((47, len(rel)))
    for j, g in enumerate(rel["pattern"]):
        seed[:, j] = patterns[g] * popv if g in patterns else relig
    M = seed * (T / seed.sum(axis=0))
    for it in range(20000):
        M *= (relig / M.sum(axis=1))[:, None]
        M *= T / M.sum(axis=0)
        err = max(np.abs(M.sum(axis=1) / relig - 1).max(), np.abs(M.sum(axis=0) / T - 1).max())
        if err < 1e-12:
            break
    else:
        raise SystemExit(f"fitting did not converge, error {err:.2e}")
    print(f"\nfitting converged after {it + 1} passes ({len(rel)} codes x 47 prefectures)")

    # --- rows --------------------------------------------------------------------------------
    rows = []
    for i, p in enumerate(PREFS):
        for _, d in df.iterrows():
            c = int(d["code"])
            if c in NONRESPONSE:
                n, g = popv[i] * d["count"] / N_RESP, "national rate"
            elif d["pattern"] == "none":
                n, g = drawn[i] * (1 - r[i]), "none"
            else:
                j = int(rel.index[rel["code"] == c][0])
                n, g = M[i, j], (d["pattern"] if d["pattern"] in patterns else "flat")
            rows.append(dict(geo_id=unit_of(p), geo_level="prefecture", geo_name=p,
                             source_category=d["source_category"], count=round(float(n), 3),
                             basis="self_id", year=2024, tier="modelled", pattern=g,
                             source_id=f"jgss_2021h_2024n_x_nhk1996_{christians}"))
    out = pd.DataFrame(rows)

    # --- checks ------------------------------------------------------------------------------
    by_p = out.groupby("geo_name")["count"].sum().reindex(PREFS).to_numpy()
    if np.abs(by_p / popv - 1).max() > 1e-6:
        raise SystemExit("a prefecture's rows do not sum to its population")
    nat = out.groupby("source_category")["count"].sum()
    want = df.set_index("source_category")["count"] / N_RESP * total
    worst = (nat.reindex(want.index) / want - 1).abs().max()
    if worst > 1e-5:
        raise SystemExit(f"a code's national total moved by {worst:.2e}")
    print(f"every prefecture sums to its 2024 population; every code's national total is "
          f"its JGSS share (worst {worst:.1e})")

    # --- what it drew ------------------------------------------------------------------------
    out["node"] = out["source_category"].map(resolve)
    nodes = out.dropna(subset=["node"]).groupby(["node", "geo_name"])["count"].sum().unstack()
    nodes = nodes[PREFS]
    share = 100 * nodes / popv
    print("\nnode, national share, then highest and lowest prefectures:")
    for node in share.sum(axis=1).sort_values(ascending=False).index:
        s = share.loc[node]
        natp = 100 * nodes.loc[node].sum() / total
        if natp < 0.05:
            continue
        hi = ", ".join(f"{EN[p]} {s[p]:.1f}" for p in s.sort_values(ascending=False).index[:3])
        lo = ", ".join(f"{EN[p]} {s[p]:.2f}" for p in s.sort_values().index[:2])
        print(f"    {node:<30} {natp:5.2f}%   high {hi};  low {lo}")
    chr_ = share.loc[[n for n in share.index if n.startswith("christianity") or n == "unification"]].sum()
    print("  all Christian nodes: " + ", ".join(f"{EN[p]} {chr_[p]:.2f}%" for p in
                                               ["東京", "長崎", "神奈川", "沖縄", "埼玉", "千葉", "大阪"]))
    return out


def main():
    christians = "nhk1996"
    if "--christians" in sys.argv:
        christians = sys.argv[sys.argv.index("--christians") + 1]
        if christians not in ("nhk1996", "roll"):
            raise SystemExit("--christians takes nhk1996 or roll")
    out = allocate(christians)
    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count", "basis", "year",
            "tier", "pattern", "source_id"]
    tmp = OUT + ".part"
    out[cols].to_csv(tmp, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    os.replace(tmp, OUT)
    print(f"\nwrote {OUT}: {len(out):,} rows")


if __name__ == "__main__":
    main()
