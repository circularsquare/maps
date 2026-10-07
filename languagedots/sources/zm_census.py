"""Zambia 2022 census, the widely spoken language of communication -> data/normalized/zm.csv.

    python sources/zm_census.py [--fetch]

SOURCE. ZamStats, "2022 Census of Population and Housing, Series C1: Language Descriptive
Tables" (May 2026, 330 pp.). One question, one answer per person: the "widely spoken language of
communication". The universe is the de facto population in households, 18,292,402 (the religion
volume's 18,340,343 less the 47,941 in institutions; checked below per constituency).

TWO KINDS OF TABLE, AT TWO GRAINS.
  C1.0-C1.10  ~85 languages in 14 sub-groups (`Language Group A (Northern)`...), for Zambia and
              each of the ten provinces, rural and urban.
  C2.0/.3/.4  nine "major language groups" (Bemba, Tonga, North Western, Western, Nyanja,
              Tumbuka, Mambwe, English, Other Languages) by province, district, constituency and
              ward: all, rural, urban.

THE NINE GROUPS ARE UNIONS OF C1's LANGUAGES, BUT NOT OF ITS SUB-GROUPS. Found by matching the
national figures and asserted for every province and residence (check 4): Kunda and Chikunda
(C1 Group A) and Fungwe (Group L) count in the Nyanja group; Totela and Subiya (Group K) and Nkoya
and Mashasha (Group H) in the Western group; Lukolwe and Lushangi (Group H) and Mbowe (Group C2)
in North Western; Wina (Group C2) in Mambwe. `Other Languages` is babies not yet able to speak,
people unable to speak, sign language, the foreign languages, `Other African` and `Other
Language` together.

THE SPREAD. For constituency c in province p, residence r (rural, urban), group g, language l:

    count(c, l) = sum over r of  C2_r(c, g) * C1_r(p, l) / C1_r(p, g)

so each constituency's measured rural and urban group counts are split by its province's rural
or urban mix of the group's languages. Every province x language x residence re-aggregates to
C1 (check 6). Every row is `derived`.

SUPPRESSED CELLS (`*`, small counts). In C1, a cell is first recovered from its own row (total =
rural + urban), then from its sub-group's total, the residual shared equally among the starred
cells left, and where the sub-group total is starred too, from the table's residual. In C2 a
constituency's cell is recovered from the other residence's table and the total table, then from
the row total. What is shared rather than recovered is counted and printed (about 2,200 people in
C1). Two printing slips are handled by name: Central labels `Other Language` "English", and
Western prints its Sign Language group blank (filled as Zambia less the other provinces).

CHECKS (all must pass):
  1. C1: every province's languages sum to its total, and the provinces to Zambia, per residence
  2. C1: every province table prints the same languages as C1.0
  3. C2: wards sum to their constituency where no ward cell is starred; constituencies to their
     district and province; the rural and urban tables to the total table
  4. C2 province rows equal the C1 province tables summed into the nine groups (rural, urban)
  5. constituencies join one-to-one to religiondots' 156 COD-AB constituencies, and each
     constituency's language total is the religion volume's de facto count less its people in
     institutions (the difference is >= 0 and sums to 47,941)
  6. the spread re-aggregates to C1 for every province, residence and language
"""
import argparse
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "zm"
PDF = RAW / "Languages-Descriptive-Tables.pdf"
URL = "https://www.zamstats.gov.zm/wp-content/uploads/2026/05/Languages-Descriptive-Tables.pdf"
OUT = HERE / "data" / "normalized" / "zm.csv"
sys.path.insert(0, str(HERE))
from rdlink import RD, RD_GEO  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}

PROVINCES = ["Central", "Copperbelt", "Eastern", "Luapula", "Lusaka", "Muchinga", "Northern",
             "North-Western", "Southern", "Western"]
NATIONAL = 18_292_402
INSTITUTIONS = 47_941           # Summary Report Part 2; religion de facto 18,340,343 less this
N_CONSTITUENCIES = 156
N_WARDS = 1_858
GROUPS = ["Bemba Group", "Tonga Group", "North Western Group", "Western Group", "Nyanja group",
          "Tumbuka Group", "Mambwe Group", "English Group", "Other Languages"]

# C1 sub-group (by its letter) -> C2 group, with the languages that cross over. Folded labels.
SUBGROUP = {"A": "Bemba Group", "B": "North Western Group", "C1": "Western Group",
            "C2": "Western Group", "D": "North Western Group", "E": "North Western Group",
            "F": "Mambwe Group", "G": "Mambwe Group", "H": "Western Group",
            "I": "Nyanja group", "J": "Nyanja group", "K": "Tonga Group", "L": "Tumbuka Group",
            "Official Language": "English Group"}
CROSS = {"kunda": "Nyanja group", "chikunda": "Nyanja group", "fungwe": "Nyanja group",
         "totela": "Western Group", "subiya": "Western Group",
         "lukolwe(mbwela)": "North Western Group", "lushangi": "North Western Group",
         "mbowe": "North Western Group", "wina": "Mambwe Group"}

# The join to religiondots' COD-AB constituencies: religiondots/sources/zm.py's aliases.
FOLD_ALIASES = {"ikelenge": "ikelengi"}
CONSTITUENCY_ALIASES = {
    ("shibuyunji", "mwembeshi"): "mwembezhi",
    ("petauke", "petaukecentral"): "petauke",
    ("kalumbila", "solweziwest"): "kalumbila",
    ("mushindamo", "solwezieast"): "mushindamo",
    ("solwezi", "solwezicentral"): "solwezi",
    # this volume's own two: the electoral names of the sole constituency of each district
    ("mpongwe", "mpongwecentral"): "mpongwe",
    ("shangombo", "sinjembela"): "shangombo",
}
SOLE_CONSTITUENCY = {"kalumbila", "mushindamo", "solwezi", "mpongwe", "shangombo"}

VAL = re.compile(r"^(?:[\d,]+|\*|-)$")


def nn(x):
    """Not starred: a figure, as opposed to None/NaN."""
    return x is not None and not pd.isna(x)


def fold(s):
    k ="".join(ch for ch in str(s).lower() if ch.isalnum() or ch in "()")
    return FOLD_ALIASES.get(k, k)


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    data = urllib.request.urlopen(urllib.request.Request(URL, headers=UA), timeout=300).read()
    if data[:5] != b"%PDF-" or b"%%EOF" not in data[-2048:]:
        raise SystemExit(f"{PDF.name}: not a complete PDF ({len(data):,} bytes)")
    PDF.write_bytes(data)
    print(f"wrote {PDF} ({len(data):,} bytes)")


def val(s):
    if s == "*":
        return None          # suppressed
    if s == "-":
        return 0
    return int(s.replace(",", ""))


def table_pages(doc):
    """{'1.0': [page idx...], ..., '2.4': [...]} from each page's TABLE heading."""
    out = defaultdict(list)
    for i in range(doc.page_count):
        m = re.search(r"TABLE\s+C\.?\s?(\d+\.\d+)\s*:", doc[i].get_text())
        if m and i >= 5:
            out[m.group(1)].append(i)
    return out


def body(doc, pages, first_header, last_header, nth=1):
    """The lines of a run of pages with each page's repeated header cut off: everything up to
    the nth `last_header` line after `first_header`."""
    out = []
    for i in pages:
        ls = [x.strip() for x in doc[i].get_text().split("\n")]
        ls = [x for x in ls if x]
        b = ls.index(first_header)
        for _ in range(nth):
            b = ls.index(last_header, b + 1)
        out += ls[b + 1:]
    return out


def records(ls, n):
    recs = []
    for x in ls:
        if VAL.match(x):
            if not recs:
                raise SystemExit(f"value {x!r} before any label")
            recs[-1][1].append(val(x))
        elif recs and not recs[-1][1] and recs[-1][0].count("(") > recs[-1][0].count(")"):
            recs[-1][0] += " " + x       # a label wrapped onto two lines: "(Not" / "Applicable)"
        else:
            recs.append([re.sub(r"\s+", " ", x), []])
    for lab, v in recs:
        if len(v) not in (0, n):
            raise SystemExit(f"{lab!r}: {len(v)} values, expected {n} or 0")
    return recs


# ---------------------------------------------------------------- C1: languages by province
def parse_c1(doc, pages, stats, template=None):
    """DataFrame [label, key, sub, group, T, R, U] for one province (or Zambia); first row total."""
    # the header block is Languages, Zambia, Rural, Urban, (Total, Male, Female) x3
    recs = records(body(doc, pages, "Languages", "Female", nth=3), 9)
    rows, sub, total = [], None, None
    for i, (lab, v) in enumerate(recs):
        # a sub-group heading is the line before its `Total` row; a few print dashes beside
        # them (Northern's `Foreign Languages`), so it is not "a label with no values"
        if i + 1 < len(recs) and recs[i + 1][0] == "Total" and lab != "Total" or not v:
            if v and any(x for x in v):
                raise SystemExit(f"sub-group heading {lab!r} carries figures {v}")
            sub = lab
            continue
        T, R, U = v[0], v[3], v[6]
        if sub is None:
            total = (T, R, U)
            continue
        rows.append(dict(sub=sub, label=lab, T=T, R=R, U=U))
    df = pd.DataFrame(rows)
    df["key"] = df["label"].map(fold)
    tot = df[df["label"] == "Total"].copy()
    df = df[df["label"] != "Total"].copy()
    if template is not None:
        # A province omits a one-language group's member row where it prints the group total
        # (Central's `Other Language`), and prints Western's `Sign Language` group with no
        # figures at all. Align to C1.0's rows: a missing row takes its one-member group's
        # total, else 0. Anything else missing or extra fails.
        # Central (C1.1) labels the one row of `Other Language Groups` "English", a slip: the
        # group is `Other Language` everywhere else, and Central's English is in its own place.
        slip = (df["sub"].map(fold) == "otherlanguagegroups") & (df["key"] == "english")
        if slip.sum() > 1:
            raise SystemExit("more than one `English` under Other Language Groups")
        if slip.any():
            df.loc[slip, ["label", "key"]] = ["Other Language", "otherlanguage"]
            stats["c1 relabelled"] += 1
        dup = df[df["key"].duplicated(keep=False)]
        if len(dup):
            raise SystemExit(f"C1 duplicate rows:\n{dup.to_string()}")
        extra = set(df["key"]) - set(template["key"])
        if extra:
            raise SystemExit(f"C1 rows not in C1.0: {sorted(extra)}")
        have = set(df["key"])
        subtot = {fold(s): t for s, t in zip(tot["sub"], tot["T"])}
        add = []
        for r in template.itertuples():
            if r.key in have:
                continue
            members = (template["sub"] == r.sub).sum()
            st = tot[tot["sub"].map(fold) == fold(r.sub)]
            if members == 1 and len(st) == 1 and nn(st.iloc[0]["T"]):
                s = st.iloc[0]
                add.append(dict(sub=r.sub, label=r.label, key=r.key, T=s["T"], R=s["R"], U=s["U"]))
                stats["c1 row from group total"] += 1
            else:      # filled from Zambia less the other provinces, in main()
                add.append(dict(sub=r.sub, label=r.label, key=r.key, T=0, R=0, U=0,
                                absent=True))
                stats["c1 row absent"] += 1
        if add:
            df = pd.concat([df, pd.DataFrame(add)])
        df = df.set_index("key").loc[list(template["key"])].reset_index()
        del subtot
    if "absent" not in df.columns:
        df["absent"] = False
    df["absent"] = df["absent"].eq(True)
    df = df.reset_index(drop=True)
    tot = tot.reset_index(drop=True)
    left = fill_c1(df, tot, stats)
    df["group"] = [CROSS.get(k) or group_of(s) for k, s in zip(df["key"], df["sub"])]
    return df, total, left


def group_of(sub):
    if sub.startswith("Language Group"):
        letter = sub.split()[2]
        return SUBGROUP[letter]
    return SUBGROUP.get(sub, "Other Languages")


def fill_c1(df, tot, stats):
    """Recover starred cells: from the row (T = R + U), then the sub-group's residual, then (a
    sub-group whose own total is starred) the whole table's residual, shared equally."""
    for frame in (tot, df):
        for i in frame.index:
            T, R, U = frame.at[i, "T"], frame.at[i, "R"], frame.at[i, "U"]
            hit = True
            if nn(T) and nn(R) and not nn(U):
                frame.at[i, "U"] = T - R
            elif nn(T) and nn(U) and not nn(R):
                frame.at[i, "R"] = T - U
            elif nn(R) and nn(U) and not nn(T):
                frame.at[i, "T"] = R + U
            else:
                hit = False
            if hit and frame is df:
                stats["c1 row"] += 1
    subtot = dict(zip(tot["sub"], zip(tot["T"], tot["R"], tot["U"])))
    left = {c: [] for c in "TRU"}
    for sub, g in df.groupby("sub"):
        for j, col in enumerate("TRU"):
            star = g.index[g[col].isna()]
            if not len(star):
                continue
            st = subtot.get(sub, (None,) * 3)[j]
            if not nn(st):      # the sub-group total itself is starred
                left[col] += list(star)
                continue
            resid = st - g[col].dropna().sum()
            if resid < 0:
                print(tot.to_string(), "\n", g.to_string())
                raise SystemExit(f"C1 {sub} {col}: named cells exceed the total by {-resid}")
            for i in star:
                df.at[i, col] = resid / len(star)
            stats["c1 shared"] += len(star)
            stats["c1 shared people"] += resid
    return left


def fill_left(df, total, left, stats):
    """The cells of sub-groups whose own total is starred: the table's residual, shared."""
    for col in "RU":
        if not left[col]:
            continue
        if total is None:
            raise SystemExit("starred sub-group total in C1.0")
        resid = total[1 if col == "R" else 2] - df[col].dropna().sum()
        if resid < 0:
            raise SystemExit(f"C1 {col}: named cells exceed the table total by {-resid}")
        for i in left[col]:
            df.at[i, col] = resid / len(left[col])
        stats["c1 table residual"] += len(left[col])
        stats["c1 table residual people"] += resid
    for i in left["T"]:
        df.at[i, "T"] = df.at[i, "R"] + df.at[i, "U"]
    for col in "TRU":
        df[col] = df[col].astype(float)


# ---------------------------------------------------------------- C2: groups by area
def parse_c2(doc, pages):
    recs = records(body(doc, pages, "Province, District, Constituency and Ward", "Languages"), 10)
    rows, prov, dist, con = [], None, None, None
    for lab, v in recs:
        if not v:
            raise SystemExit(f"C2 label with no values: {lab!r}")
        if lab == "Zambia":
            level, name = "zambia", "Zambia"
        else:
            level, name = lab.split(" ", 1)
            level = level.lower()
            if level == "zambia":           # the rural and urban tables print "Zambia Total"
                name = "Zambia"
        if level == "province":
            prov, dist, con = name, None, None
        elif level == "district":
            dist, con = name, None
        elif level == "constituency":
            con = name
        elif level != "ward" and level != "zambia":
            raise SystemExit(f"C2 row of unknown level: {lab!r}")
        rows.append(dict(level=level, name=name, province=prov, district=dist,
                         constituency=con, total=v[0], **dict(zip(GROUPS, v[1:]))))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch or not PDF.exists():
        fetch()

    ok = True

    def report(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    print("Zambia, 2022 census, widely spoken language of communication (ZamStats Series C1)\n")
    doc = fitz.open(PDF)
    tp = table_pages(doc)
    stats = defaultdict(float)

    # ---- C1
    c1, lefts = {}, {}
    for k in range(11):
        df, total, left = parse_c1(doc, tp[f"1.{k}"], stats,
                                   template=c1["Zambia"][0] if k else None)
        title = doc[tp[f"1.{k}"][0]].get_text()
        if k == 0:
            name = "Zambia"
        else:
            m = re.search(r"RURAL/URBAN,\s+([A-Z\- ]+?) PROVINCE", title)
            name = m.group(1).title().replace("North-Western", "North-Western")
        c1[name] = (df, total)
        lefts[name] = left
    names = list(c1)
    # Western prints its `Sign Language` group with no figures at all: take Zambia less the
    # other nine provinces, per residence (284 people, 239 rural)
    nat = c1["Zambia"][0]
    for n in PROVINCES:
        df = c1[n][0]
        for i in df.index[df["absent"] == True]:  # noqa: E712
            for col in "TRU":
                others = sum(c1[m][0].at[i, col] for m in PROVINCES if m != n)
                df.at[i, col] = nat.at[i, col] - others
            print(f"     {n} `{df.at[i, 'label']}` printed blank: Zambia less the other "
                  f"provinces = {df.at[i, 'T']:,.0f} ({df.at[i, 'R']:,.0f} rural)")
    for n in names:
        fill_left(c1[n][0], c1[n][1], lefts[n], stats)
        if c1[n][0][list("TRU")].isna().any().any():
            raise SystemExit(f"C1 {n}: starred cells left unfilled")
    report(names[1:] == PROVINCES, f"C1 tables for Zambia and {len(names) - 1} provinces "
                                   f"({', '.join(names[1:])})")
    nat, nat_total = c1["Zambia"]
    report(nat_total[0] == NATIONAL, f"Zambia total {nat_total[0]:,} (expected {NATIONAL:,})")

    # 1-2
    bad, worst = [], 0
    for name, (df, total) in c1.items():
        for j, col in enumerate("TRU"):
            off = df[col].sum() - total[j]
            worst = max(worst, abs(off))
            if abs(off) > 10:          # a starred sub-group total was set to 5 a cell
                bad.append(f"{name} {col} {df[col].sum():,.0f} vs {total[j]:,}")
    report(not bad, f"every C1 table's languages sum to its total within 10, per residence "
                    f"(largest {worst:,.0f}) {bad[:4]}")
    keys0 = list(nat["key"])
    diff = [n for n in PROVINCES if list(c1[n][0]["key"]) != keys0]
    for n in diff:
        ks = list(c1[n][0]["key"])
        print(f"     {n}: missing {[k for k in keys0 if k not in ks]}, "
              f"extra {[k for k in ks if k not in keys0]}")
    report(not diff, f"every province prints C1.0's {len(keys0)} rows, in order {diff}")
    prov_sum = sum(c1[n][0][["T", "R", "U"]].to_numpy() for n in PROVINCES)
    gap = pd.DataFrame(prov_sum - nat[["T", "R", "U"]].to_numpy(), columns=list("TRU"),
                       index=nat["label"])
    for lab in gap.index[(gap.abs() > 20).any(axis=1)]:
        print(f"     {lab}: Zambia {nat.loc[nat['label'] == lab, ['T', 'R', 'U']].values}")
        for n in PROVINCES:
            df = c1[n][0]
            print(f"       {n}: {df.loc[df['label'] == lab, ['T', 'R', 'U']].values}")
    d = gap.abs().to_numpy().max()
    report(d <= 50, f"provinces sum to Zambia on every language and residence "
                    f"(largest gap {d:,.0f}, starred cells)")
    print(f"     C1 starred cells: {stats['c1 row']:.0f} recovered from their row, "
          f"{stats['c1 shared']:.0f} shared from the sub-group residual "
          f"({stats['c1 shared people']:,.0f} people), {stats['c1 table residual']:.0f} shared "
          f"from the table's residual ({stats['c1 table residual people']:,.0f} people; their "
          f"sub-group total starred too); {stats['c1 row from group total']:.0f} absent rows "
          f"taken from a one-language group's total, {stats['c1 row absent']:.0f} absent "
          "rows filled from Zambia less the other provinces")

    # ---- C2
    c2 = {r: parse_c2(doc, tp[t]) for r, t in (("T", "2.0"), ("R", "2.3"), ("U", "2.4"))}
    for r, df in c2.items():
        print(f"     C2 {r}: {len(df):,} rows, {dict(df['level'].value_counts())}")
    ws = c2["T"][c2["T"]["level"] == "ward"]
    report(len(ws) == N_WARDS, f"{len(ws):,} wards (expected {N_WARDS:,})")
    cols = ["total"] + GROUPS
    same = all((c2[r][["level", "name", "province", "district", "constituency"]]
                .equals(c2["T"][["level", "name", "province", "district", "constituency"]]))
               for r in "RU")
    if not same:
        for r in "RU":
            a = c2["T"][["level", "name"]].apply(tuple, axis=1)
            b = c2[r][["level", "name"]].apply(tuple, axis=1)
            dd = [(x, y) for x, y in zip(a, b) if x != y]
            print(f"     {r}: {len(dd)} differ, e.g. {dd[:4]}")
    report(same, "the total, rural and urban C2 tables list the same areas in the same order")

    # 3. nesting, on cells with no star
    for r, df in c2.items():
        bad, n = [], 0
        for child, parent, key in (("ward", "constituency", ["province", "district", "constituency"]),
                                   ("constituency", "district", ["province", "district"]),
                                   ("district", "province", ["province"])):
            ch = df[df["level"] == child]
            pa = df[df["level"] == parent].set_index(key)
            for k, g in ch.groupby(key, sort=False):
                for c in cols:
                    if g[c].isna().any() or not nn(pa.loc[k, c]):
                        continue
                    n += 1
                    if g[c].sum() != pa.loc[k, c]:
                        bad.append((child, k, c, g[c].sum(), pa.loc[k, c]))
        z = df[df["level"] == "zambia"].iloc[0]
        pr = df[df["level"] == "province"]
        bad += [("province", "Zambia", c, pr[c].sum(), z[c]) for c in cols if pr[c].sum() != z[c]]
        report(not bad, f"C2 {r}: wards, constituencies, districts and provinces nest "
                        f"({n:,} unstarred sums; {len(bad)} differ {bad[:3]})")
    t, rr, uu = (c2[r][cols].astype(float) for r in "TRU")
    full = t.notna() & rr.notna() & uu.notna()
    nbad = int(((t - rr - uu).abs() > 0)[full].sum().sum())
    report(nbad == 0, f"C2: rural + urban = total on every unstarred cell ({nbad} differ)")

    # 4. the nine groups are unions of C1's languages, per province and residence
    bad, worst = [], 0
    for p in PROVINCES:
        g1 = c1[p][0].groupby("group")[["R", "U"]].sum()
        for r in "RU":
            row = c2[r][(c2[r]["level"] == "province") & (c2[r]["name"].map(fold) == fold(p))]
            if len(row) != 1:
                bad.append((p, r, "no C2 row"))
                continue
            for g in GROUPS:
                v = row.iloc[0][g]
                if not nn(v):
                    continue
                off = g1.loc[g, r] - v
                worst = max(worst, abs(off))
                if abs(off) > 15:
                    bad.append((p, r, g, round(g1.loc[g, r]), v))
    report(not bad, f"C2's province rows are C1's languages summed into the nine groups, rural "
                    f"and urban (largest gap {worst:,.0f}, starred C1 cells) {bad[:4]}")

    # ---- constituencies: fill starred cells, rural and urban
    con = {r: c2[r][c2[r]["level"] == "constituency"].reset_index(drop=True) for r in "TRU"}
    filled = defaultdict(int)
    for _ in range(2):
        for i in range(len(con["T"])):
            for c in cols:
                T, R, U = (con[r].at[i, c] for r in "TRU")
                if nn(T) and nn(R) and not nn(U):
                    con["U"].at[i, c] = T - R; filled["from the other residence"] += 1
                elif nn(T) and nn(U) and not nn(R):
                    con["R"].at[i, c] = T - U; filled["from the other residence"] += 1
                elif nn(R) and nn(U) and not nn(T):
                    con["T"].at[i, c] = R + U
        for r in "RU":
            df = con[r]
            for i in df.index:
                star = [g for g in GROUPS if not nn(df.at[i, g])]
                if star and nn(df.at[i, "total"]):
                    resid = df.at[i, "total"] - sum(df.at[i, g] for g in GROUPS if g not in star)
                    for g in star:
                        df.at[i, g] = resid / len(star)
                    filled["from the row total" if len(star) == 1 else "row residual shared"] += len(star)
    left = sum(int(con[r][GROUPS].isna().sum().sum()) for r in "RU")
    report(left == 0, f"constituency cells starred: {dict(filled)}, {left} left")

    # 5. join to religiondots' COD-AB constituencies
    lut = pd.read_csv(RD_GEO / "zm" / "zm_lookup.csv", dtype=str)
    lut["key"] = [(fold(d), fold(n)) for d, n in zip(lut["district"], lut["name"])]
    ct = con["T"]
    ct["key"] = [(fold(d), CONSTITUENCY_ALIASES.get((fold(d), fold(n)), fold(n)))
                 for d, n in zip(ct["district"], ct["constituency"])]
    for d in SOLE_CONSTITUENCY:
        if (lut["key"].str[0] == d).sum() != 1 or (ct["key"].str[0] == d).sum() != 1:
            report(False, f"{d}: aliased as a district's sole constituency, but it is not")
    k2u = dict(zip(lut["key"], lut["unit"]))
    ct["unit"] = ct["key"].map(k2u)
    miss = ct.loc[ct["unit"].isna(), ["district", "constituency"]].values.tolist()
    report(len(ct) == N_CONSTITUENCIES and not miss and ct["unit"].is_unique
           and set(ct["unit"]) == set(lut["unit"]),
           f"{len(ct)} constituencies join one-to-one to religiondots' {len(lut)} "
           f"({len(CONSTITUENCY_ALIASES)} by an explicit alias; unmatched {miss[:5]})")
    rel = pd.read_csv(RD / "data" / "normalized" / "zm.csv", dtype={"geo_id": str})
    rel = rel.groupby("geo_id")["count"].sum().round()
    d = rel.reindex(ct["unit"]).to_numpy() - ct["total"].to_numpy()
    report((d >= -1).all() and abs(d.sum() - INSTITUTIONS) <= len(d),
           f"religion de facto less language total, per constituency: min {d.min():,.0f}, "
           f"median {pd.Series(d).median():,.0f}, max {d.max():,.0f}, sum {d.sum():,.0f} "
           f"(the {INSTITUTIONS:,} in institutions)")
    big = ct.assign(d=d).nlargest(3, "d")[["constituency", "d"]].values.tolist()
    print(f"     largest: {big}")

    # ---- the spread
    out, fallback = [], 0
    for r in "RU":
        cr = con[r].copy()
        cr["unit"] = ct["unit"].to_numpy()
        for p in PROVINCES:
            L = c1[p][0]
            nat_l = c1["Zambia"][0]
            for g in GROUPS:
                lg = L[L["group"] == g]
                shares = lg[r] / lg[r].sum() if lg[r].sum() > 0 else None
                if shares is None:
                    ng = nat_l[nat_l["group"] == g]
                    shares = ng[r] / ng[r].sum()
                    shares.index = lg.index
                sub = cr[cr["province"].map(fold) == fold(p)]
                for unit, dist, cname, v in zip(sub["unit"], sub["district"],
                                                sub["constituency"], sub[g]):
                    if v <= 0:
                        continue
                    if lg[r].sum() <= 0:
                        fallback += v
                    for li, s in shares.items():
                        if s > 0:
                            out.append((unit, p, dist, cname, r, L.at[li, "label"], g, v * s))
    res = pd.DataFrame(out, columns=["geo_id", "province", "district", "constituency",
                                     "residence", "source_category", "ward_group", "count"])
    print(f"     {fallback:,.0f} people in a group their province's C1 has none of for that "
          "residence; split by Zambia's mix")
    report(abs(res["count"].sum() - NATIONAL) < 1,
           f"spread total {res['count'].sum():,.1f} (Zambia {NATIONAL:,})")

    # 6. re-aggregation
    worst, n = 0, 0
    for p in PROVINCES:
        L = c1[p][0].set_index("label")
        agg = res[res["province"] == p].groupby(["residence", "source_category"])["count"].sum()
        for r in "RU":
            g2 = con[r][con[r]["province"].map(fold) == fold(p)][GROUPS].sum()
            for g in GROUPS:
                lg = L[L["group"] == g][r]
                if lg.sum() <= 0:
                    continue
                expect = lg / lg.sum() * g2[g]
                got = agg.loc[r].reindex(expect.index).fillna(0)
                worst = max(worst, (got - expect).abs().max())
                n += len(expect)
    report(worst < 0.01, f"the spread re-aggregates to C1's province mix x C2's constituency "
                         f"groups, {n:,} province x residence x language cells "
                         f"(largest gap {worst:.4f})")

    if not ok:
        raise SystemExit("\nreconciliation FAILED; nothing written")
    res["geo_level"] = "constituency"
    res = res[["geo_id", "geo_level", "province", "district", "constituency", "residence",
               "ward_group", "source_category", "count"]]
    res["count"] = res["count"].round(3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT}: {len(res):,} rows, {res['geo_id'].nunique()} constituencies, "
          f"{res['count'].sum():,.0f} people, {res['source_category'].nunique()} categories")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
