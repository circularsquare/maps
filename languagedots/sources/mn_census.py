"""Mongolia, 2020 Population and Housing Census: ethnic group by aimag, read as language.

    python sources/mn_census.py        -> data/normalized/mn.csv  (+ checks printed)

No language question was asked in 2020 (questionnaire Q6 is ethnicity; the only language item is
literacy "in any language"). Everything is read from the English national report, already
downloaded by religiondots (read-only):
    ../religiondots/data/raw/mn/Census2020_Main_report_Eng.pdf
  p210  appendix table 1.1  resident population by aimag (21 aimags + Ulaanbaatar)
  p230  appendix table 4.1  resident population by ethnic group, national COUNTS
  p223-226 appendix table 3.6a  each ethnic group's distribution across aimags, % (one decimal)
  p59   table 3.7           foreign citizens by aimag (counts); table 3.6 their citizenship, %

Method. count(aimag, group) = national count(group) x the group's aimag share (3.6a), then
raked (iterative proportional fitting) so that every aimag sums to its Mongolian citizens
(table 1.1 minus table 3.7's foreign citizens) and every group to its table 4.1 count. 3.6a
prints one decimal, so a group's aimag cell is good to about 0.05% of the group; the rake moves
nothing by much (printed below). Table 3.6 (the row-percentage twin) is the check.

Foreign citizens (22,418) are split by the national citizenship shares of table 3.6 (China,
Russia, Korea, USA, other) in every aimag: tier `modelled`, since no table crosses the two.
The 37 stateless people are not drawn.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import fitz  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rdlink import RD  # noqa: E402

PDF = RD / "data" / "raw" / "mn" / "Census2020_Main_report_Eng.pdf"
OUT = ROOT / "data" / "normalized" / "mn.csv"

# Aimag name as the report prints it -> COD-AB pcode (= the census's own aimag code; religiondots'
# sources/mn_geo.py asserts that). Ulaanbaatar is one unit here: no table gives ethnicity by düüreg.
CODE = {
    "Arkhangai": "MN65", "Bayan-Ulgii": "MN83", "Bayankhongor": "MN64", "Bulgan": "MN63",
    "Govi-Altai": "MN82", "Dornogovi": "MN44", "Dornod": "MN21", "Dundgovi": "MN48",
    "Zavkhan": "MN81", "Uvurkhangai": "MN62", "Umnugovi": "MN46", "Sukhbaatar": "MN22",
    "Selenge": "MN43", "Tuv": "MN41", "Uvs": "MN85", "Khovd": "MN84", "Khuvsgul": "MN67",
    "Khentii": "MN23", "Darkhan-Uul": "MN45", "Orkhon": "MN61", "Govisumber": "MN42",
    "Ulaanbaatar": "MN11",
}
FOREIGN_NAME = {"Gobi-Altai": "Govi-Altai", "Dornogobi": "Dornogovi", "Dundgobi": "Dundgovi",
                "Umnugobi": "Umnugovi", "Gobisumber": "Govisumber"}

# Table 3.6a's columns, in print order, page by page.
COLS_223A = ["Khalkh", "Kazakh", "Durvud", "Buriad", "Bayad", "Dariganga", "Uriankhai",
             "Zakhchin"]
COLS_223B = ["Darkhad", "Torguud", "Uuld", "Khoton", "Myangad", "Barga", "Uzemchin", "Kharchin",
             "Khotgoid"]
COLS_226 = ["Eljigen", "Tsaatan (Dukha)", "Sartuul", "Tuva", "Uzbek (Chantuu)", "Khamnigan",
            "Khoshuud", "Other Mongolian ethnic groups", "Other foreign /Mongolian citizen/"]
# Table 4.1's groups that 3.6a folds into "Other ethnic groups / Mongolian" (143 people).
OTHER_MONGOL = ["Tsakhar", "Khorchin", "Khalimag", "Tumed", "Sunud", "Tuved", "Balba", "Other"]

# 2020 foreign citizens by citizenship, table 3.6 (national only).
FOREIGN_SHARE = {"Foreign citizen: China": 41.7, "Foreign citizen: Russia": 11.6,
                 "Foreign citizen: Korea": 9.9, "Foreign citizen: USA": 10.6,
                 "Foreign citizen: other": 26.2}


def rows(page):
    """Words grouped into printed lines -> list of token lists."""
    by = {}
    for w in PDF_DOC[page - 1].get_text("words"):
        by.setdefault((w[5], w[6]), []).append(w)
    lines = {}
    for ws in by.values():
        lines.setdefault(round(ws[0][1] / 3), []).extend(ws)
    return [[x[4] for x in sorted(v, key=lambda x: x[0])] for _, v in sorted(lines.items())]


def pct_table(page, cols):
    """Read one block of 3.6a: aimag -> {col: pct}. Takes the first block of 22 aimag rows."""
    out = {}
    for t in rows(page):
        if t and t[0] in CODE and t[0] not in out:
            vals = [float(v) for v in t[1:]]
            if len(vals) != len(cols):
                raise SystemExit(f"p{page} {t[0]}: {len(vals)} values for {len(cols)} columns")
            out[t[0]] = dict(zip(cols, vals))
    if set(out) != set(CODE):
        raise SystemExit(f"p{page}: aimags missing {sorted(set(CODE) - set(out))}")
    return out


def blocks(page):
    """Every block of 22 aimag rows on a page, in order: [{aimag: [values]}, ...]."""
    out = []
    for t in rows(page):
        if t and t[0] in CODE:
            if not out or t[0] in out[-1]:
                out.append({})
            out[-1][t[0]] = [float(v) for v in t[1:]]
    return out


def tables_3_6():
    """(row %, column %) per aimag and group. Table 3.6 is each aimag's split by group (row %),
    3.6a each group's split by aimag (column %). Both print one decimal.
      p222 block 0: 3.6 row %, citizens-total then 8 groups
      p223 block 0: 3.6a col %, the aimag's share of all citizens then 8 groups;
           block 1: 3.6 row %, 9 groups
      p224 block 0: 3.6a col %, 9 groups
      p225 block 0: 3.6 row %, 9 groups
      p226 block 0: 3.6a col %, 9 groups"""
    row, col = {a: {} for a in CODE}, {a: {} for a in CODE}
    spec = [(222, 0, row, COLS_223A, 1), (223, 0, col, COLS_223A, 1), (223, 1, row, COLS_223B, 0),
            (224, 0, col, COLS_223B, 0), (225, 0, row, COLS_226, 0), (226, 0, col, COLS_226, 0)]
    for page, b, dest, cols, skip in spec:
        blk = blocks(page)[b]
        if set(blk) != set(CODE):
            raise SystemExit(f"p{page} block {b}: aimags missing {sorted(set(CODE) - set(blk))}")
        for a, v in blk.items():
            if len(v) - skip != len(cols):
                raise SystemExit(f"p{page} {a}: {len(v)} values for {len(cols)} columns")
            dest[a].update(zip(cols, v[skip:]))
    return row, col


def split_counts(tokens, n):
    """Split space-grouped integers into n numbers: a number is a 1-3 digit head followed by
    zero or more 3-digit groups. Brute force over the split points, keeping the unique split
    where every number is well formed. Ambiguous splits raise."""
    sols = []

    def rec(i, acc):
        if len(acc) == n:
            if i == len(tokens):
                sols.append(list(acc))
            return
        for j in range(i + 1, len(tokens) + 1):
            head, tail = tokens[i], tokens[i + 1:j]
            if not (1 <= len(head) <= 3 and head.isdigit()) or head.startswith("0") and head != "0":
                break
            if any(len(x) != 3 or not x.isdigit() for x in tail):
                break
            acc.append(int("".join(tokens[i:j])))
            rec(j, acc)
            acc.pop()
    rec(0, [])
    return sols


def table_1_1():
    """Aimag totals: the first numeric column of p210's first block. The row has 8 numbers."""
    first, second = {}, {}
    for t in rows(210):
        if t and t[0] in CODE:
            (second if t[0] in first else first)[t[0]] = t[1:]
    out = {}
    for a in first:
        # the total must equal the sum of the 15 age groups printed across the two blocks
        hits = {s[0] for s in split_counts(first[a], 8) for s2 in split_counts(second[a], 8)
                if s[0] == sum(s[1:]) + sum(s2)}
        if len(hits) != 1:
            raise SystemExit(f"p210 {a}: ambiguous total {sorted(hits)}")
        out[a] = hits.pop()
    return out


def table_4_1():
    """National counts by ethnic group: name (may be several words), Total, Male, Female."""
    out = {}
    for t in rows(230):
        nums = []
        while t and t[-1].isdigit():
            nums.insert(0, t.pop())
        name = " ".join(t)
        if not nums or not name or name.startswith("TABLES"):
            continue
        good = [s for s in split_counts(nums, 3) if s[0] == s[1] + s[2]]
        if len(good) != 1:
            raise SystemExit(f"p230 {name}: {good}")
        out[name] = good[0][0]
    return out


def table_3_7():
    out = {}
    for t in rows(59):
        if t and (t[0] in CODE or t[0] in FOREIGN_NAME):
            # 2010 n, 2010 %, 2020 n, 2020 %
            nums = t[1:]
            name = FOREIGN_NAME.get(t[0], t[0])
            # the 2020 count sits between the 2010 % and the 2020 % (both have a dot)
            dots = [k for k, x in enumerate(nums) if "." in x]
            out[name] = int("".join(nums[dots[0] + 1:dots[1]]))
    return out


def ipf(m, rows_t, cols_t, iters=200):
    m = m.copy()
    for _ in range(iters):
        m *= (rows_t / m.sum(1))[:, None]
        m *= (cols_t / np.where(m.sum(0) == 0, 1, m.sum(0)))[None, :]
    return m


def main():
    global PDF_DOC
    PDF_DOC = fitz.open(PDF)
    pop = table_1_1()
    assert sum(pop.values()) == 3_197_020, sum(pop.values())
    nat = table_4_1()
    assert nat["TOTAL"] == 3_197_020 and nat["Total Mongolian citizens-Total"] == 3_174_565
    foreign = table_3_7()
    assert sum(foreign.values()) == 22_418, sum(foreign.values())

    row, col = tables_3_6()

    nat["Other Mongolian ethnic groups"] = sum(nat[g] for g in OTHER_MONGOL)
    groups = COLS_223A + COLS_223B + COLS_226
    named = sum(nat[g] for g in groups)
    assert named == 3_174_565, (named, 3_174_565 - named)

    aimags = list(CODE)
    citizens = np.array([pop[a] - foreign[a] for a in aimags], dtype=float)
    gt = np.array([nat[g] for g in groups], dtype=float)
    for g in groups:
        s = sum(col[a][g] for a in aimags)
        if abs(s - 100) > 1.2:
            raise SystemExit(f"3.6a {g}: aimag shares sum to {s}")
    for i, a in enumerate(aimags):
        s = sum(row[a][g] for g in groups)
        if abs(s - 100) > 1.3:
            raise SystemExit(f"3.6 {a}: group shares sum to {s}")

    # Each cell is printed twice, to one decimal: as a share of the aimag (3.6) and as a share of
    # the group (3.6a). Seed each cell from whichever is finer there: 0.05% of the aimag's
    # citizens against 0.05% of the group's national count.
    seed = np.zeros((len(aimags), len(groups)))
    lo = np.zeros_like(seed)
    hi = np.zeros_like(seed)
    n_row = 0
    for i, a in enumerate(aimags):
        for j, g in enumerate(groups):
            r, c = row[a][g], col[a][g]
            r_est, c_est = citizens[i] * r / 100, gt[j] * c / 100
            if citizens[i] <= gt[j]:
                seed[i, j] = r_est
                n_row += 1
            else:
                seed[i, j] = c_est
            # the cell must sit inside both tables' rounding intervals (with 0.01 slack)
            lo[i, j] = max(citizens[i] * max(r - 0.06, 0), gt[j] * max(c - 0.06, 0)) / 100
            hi[i, j] = min(citizens[i] * (r + 0.06), gt[j] * (c + 0.06)) / 100
    print(f"seeded {n_row} cells from 3.6 (row %), {seed.size - n_row} from 3.6a (column %)")
    seed += 1e-6
    fit = ipf(seed, citizens, gt)
    move = np.abs(fit - seed).sum() / 2
    print(f"rake moved {move:,.0f} people of {gt.sum():,.0f} ({move / gt.sum():.2%})")
    bad = [(aimags[i], groups[j], round(fit[i, j]), round(lo[i, j]), round(hi[i, j]))
           for i in range(len(aimags)) for j in range(len(groups))
           if not (lo[i, j] - 2 <= fit[i, j] <= hi[i, j] + 2)]
    print(f"cells outside both tables' rounding intervals: {len(bad)} of {seed.size}")
    miss = max([max(b[3] - b[2], b[2] - b[4]) for b in bad] or [0])
    print(f"   worst miss {miss} people (a rake to two margins cannot hit every interval; "
          f"the misses are a few dozen people each)")
    if miss > 200:
        raise SystemExit("a cell is far outside the printed tables")

    out = []
    for i, a in enumerate(aimags):
        for j, g in enumerate(groups):
            c = int(round(fit[i, j]))
            if c > 0:
                out.append((CODE[a], "aimag", a, g, c, "derived",
                            "2020 PHC tables 4.1 x 3.6a, raked to table 1.1"))
        for g, p in FOREIGN_SHARE.items():
            c = int(round(foreign[a] * p / 100))
            if c > 0:
                out.append((CODE[a], "aimag", a, g, c, "modelled",
                            "table 3.7 foreign citizens x table 3.6 national citizenship shares"))
    df = pd.DataFrame(out, columns=["geo_id", "geo_level", "geo_name", "source_category",
                                    "count", "tier", "note"])
    df["source_id"] = "mn_phc_2020"
    df["year"] = 2020
    tot = df["count"].sum()
    print(f"{len(df)} rows, {df['geo_id'].nunique()} units, {tot:,} people "
          f"(census 3,197,020 less 37 stateless = 3,196,983)")
    assert abs(tot - 3_196_983) < 60, tot
    for a in aimags:
        s = df.loc[df["geo_name"] == a, "count"].sum()
        # rounding of ~30 cells, plus the 37 stateless people, whose aimag is not printed
        assert abs(s - pop[a]) <= 40, (a, s, pop[a])
    df.to_csv(OUT, index=False)
    print(df.groupby("source_category")["count"].sum().sort_values(ascending=False).to_string())
    print("wrote", OUT)


if __name__ == "__main__":
    main()
