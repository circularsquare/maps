"""Malawi: 2018 PHC tribe by district, read as home language with Afrobarometer's retention shares.

    python sources/mw_census.py     -> data/normalized/mw.csv

THE CENSUS. Malawi's 2018 PHC asked no language question (its literacy item says "in any
language"). It asked tribe, and the Main Report's Table E4 ("Malawian Population by Tribe,
Region and District", pp. 132-133) prints 12 named tribes plus Other for Malawians on the 28
districts and four cities. The PDF is religiondots' download of the same report
(data/raw/mw/mw_phc2018_main_report.pdf, read-only), and religiondots' 32 district units are
reused: the four cities are peers of their districts there and here.

THE TRIBES ARE NOT THE LANGUAGES PEOPLE SPEAK. Afrobarometer rounds 5, 7, 8 and 9 (2012-2022,
about 6,000 Malawians who named both a tribe and a home language; sources/mw_afro.py) say 97%
of Chewa speak Chichewa at home, but most Lomwe and Ngoni speak Chichewa too, half the Yao
and Sena do, and the Mang'anja mostly call their speech Chichewa. So each district's count of
a tribe is shared across the home languages its members name, in the survey's shares for
that tribe.

LINGUA FRANCAS (Anita's ruling on ask 018, 2026-10-05): Chichewa (for non-Chewa) and English
take their shares from R7's mother-tongue question alone (LF, LF_ROUNDS below); R7's answer
in the extract is that question since then. Chichewa fell from 70% to 50% with it.

ROUNDS 4 AND 6 ARE LEFT OUT. They record the tribe's own language far more often than the
other four: Lomwe answering Chilomwe 77% (R4, 2008) and 67% (R6, 2014) against 3%, 29%, 9%
and 5% in R5, R7, R8, R9; Ngoni answering Chingoni 65% and 52% against 6-12%. Nationally
they put Chichewa at 47% and 51% where the other rounds give 66-74%. The 1998 census, the
last to ask the household's language, gave Chichewa about 70% and Chilomwe under 3%, which
sides with R5, R7-R9. Pooled, R4 and R6 would have drawn 1.5 million Chilomwe speakers.

THE SHARES ARE LOCAL WHERE THE SURVEY CAN BE. P(language | tribe) is estimated nationally,
then by region (three regions), then by district (R7 and R9 name a district), each level
shrunk towards the one above with a prior worth K weighted respondents. A district with few
respondents of a tribe takes its region's shares.

CHINGONI MEANS TWO THINGS. Glottolog lists Ngoni (Nyanja) ngon1270, a dialect of Nyanja, and
Ngoni (Tumbuka) ngon1272, a dialect of Tumbuka: the Ngoni of the Central and Southern regions
speak a Nyanja variety, those of Mzimba a Tumbuka one (the Nguni language itself is near
gone). The answer "Chingoni" is drawn on the first in the Central and Southern regions and on
the second in the Northern region (AGENT_BRIEF §3: a label whose meaning depends on place).

Every row is `modelled`: the census counts the people and the tribe, the survey supplies the
language. The record is sources/mw.md.
"""
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RD = HERE.parent / "religiondots"
PDF = RD / "data" / "raw" / "mw" / "mw_phc2018_main_report.pdf"
AB = HERE / "data" / "raw" / "mw" / "ab_mw_language.csv"
OUT = HERE / "data" / "normalized" / "mw.csv"

SOURCE_ID = "nso_phc2018_tribe_x_afrobarometer_r5_r7_r9"
YEAR = "2018 (census), 2012-2022 (survey)"
ROUNDS = {5, 7, 8, 9}            # R4 and R6 left out: see the docstring
K = 30
# Anita's ruling on ask 018 (2026-10-05): lingua francas at R7's mother-tongue question (Q2A,
# R7's "lang" in the extract since then). Chichewa (for every tribe but the Chewa, whose own
# language it is) and English take their shares from R7 alone (LF_ROUNDS), national -> region
# -> district as below, the national share shrunk to the R7 rate of all non-Chewa with K0;
# every other answer from ROUNDS among the non-lingua-franca answers, scaled to what is left.
LF = ["Chichewa", "English"]
LF_ROUNDS = {7}
K0 = 10
MALAWIANS = 17_506_538          # Table E4's Malawi row

TRIBES = ["Chewa", "Tumbuka", "Lomwe", "Tonga", "Yao", "Sena", "Nkhonde", "Ngoni", "Lambya",
          "Sukwa", "Mang'anja", "Nyanja", "Other"]
REGIONS = ["Northern", "Central", "Southern"]
# religiondots' order, which gives its MW<region><nn> unit ids (religiondots sources/mw.py).
DISTRICTS = {
    "Northern": ["Chitipa", "Karonga", "Nkhata Bay", "Rumphi", "Mzimba", "Likoma",
                 "Mzuzu City"],
    "Central": ["Kasungu", "Nkhotakota", "Ntchisi", "Dowa", "Salima", "Lilongwe",
                "Mchinji", "Dedza", "Ntcheu", "Lilongwe City"],
    "Southern": ["Mangochi", "Machinga", "Zomba", "Chiradzulu", "Blantyre", "Mwanza",
                 "Thyolo", "Mulanje", "Phalombe", "Chikwawa", "Nsanje", "Balaka", "Neno",
                 "Zomba City", "Blantyre City"],
}
CODE = {d: f"MW{ri + 1}{di + 1:02d}" for ri, r in enumerate(REGIONS)
        for di, d in enumerate(DISTRICTS[r])}
REGION_OF = {d: r for r in REGIONS for d in DISTRICTS[r]}

# Afrobarometer tribe -> census tribe. R6 labels Chewa "Chewu" (936 respondents, 98% Chichewa,
# and no "Chewa" code in that round). Tribes the census has no row for go to Other. None: not
# a tribe (don't know, refused, national identity only), left out of the shares.
ETH = {"Chewa": "Chewa", "Chewu": "Chewa", "Tumbuka": "Tumbuka", "Lomwe": "Lomwe",
       "Tonga": "Tonga", "Yao": "Yao", "Sena": "Sena", "Nkhonde": "Nkhonde",
       "Ngonde": "Nkhonde", "Ngoni": "Ngoni", "Lambya": "Lambya", "Sukwa": "Sukwa",
       "Mang'anja": "Mang'anja", "Nyanja": "Nyanja", "Other": "Other", "Others": "Other",
       "Senga": "Other", "Ndali": "Other", "Khokhola": "Other", "Wiza": "Other",
       "Chikunda": "Other"}
ETH_VERBATIM = {"NYANJA": "Nyanja", "NKHONDE": "Nkhonde", "NGONDE": "Nkhonde"}

# Afrobarometer home-language answer -> the answer drawn (taxonomy/mw2018.py maps these).
# Round 4 prints bare names, later rounds the Chi- forms; both are one answer.
LANG = {"Chichewa": "Chichewa", "Chewa": "Chichewa", "Chitumbuka": "Chitumbuka",
        "Tumbuka": "Chitumbuka", "Chiyao": "Chiyao", "Yao": "Chiyao", "Chilomwe": "Chilomwe",
        "Lomwe": "Chilomwe", "Chingoni": "Chingoni", "Ngoni": "Chingoni", "Chisena": "Chisena",
        "Sena": "Chisena", "Chimang'anja": "Chimang'anja", "Mang'anja": "Chimang'anja",
        "Chitonga": "Chitonga", "Tonga": "Chitonga", "Chinkhonde": "Chinkhonde",
        "Nkhonde": "Chinkhonde", "Chilambya": "Chilambya", "Lambya": "Chilambya",
        "Chisenga": "Chisenga", "Senga": "Chisenga", "Chindali": "Chindali", "Ndali": "Chindali",
        "Chinyanja": "Chinyanja", "Chisukwa": "Chisukwa", "Sukwa": "Chisukwa",
        "Khokhola": "Chikhokhola", "Wiza": "Chiwiza", "Nyika": "Chinyika",
        "Chinyakyusa": "Chinyakyusa", "Namwanga": "Chinamwanga", "Kiswahili": "Swahili",
        "English": "English", "Portuguese": "Portuguese", "Other": "Other", "Others": "Other"}
# "Other" with a verbatim: the verbatim's answer where it names one.
VERBATIM = [(r"NDALI", "Chindali"), (r"NYAKYUSA", "Chinyakyusa"), (r"KHOKHOLA", "Chikhokhola"),
            (r"NYIKA", "Chinyika"), (r"MAMBWE", "Chimambwe"), (r"NYUNG", "Chinyungwe"),
            (r"WANDYA", "Chiwandya"), (r"LAMBYA", "Chilambya"), (r"NAMWANGA", "Chinamwanga"),
            (r"SWAHILI", "Swahili"), (r"NYANJA", "Chinyanja"), (r"KUNDA", "Chikunda"),
            (r"NDEBELE", "Sindebele"), (r"WIZA", "Chiwiza")]
NOT_LANG = {"Don't know", "Missing", "Refused"}

DIST_FIX = {"Chikhwawa": "Chikwawa", "Nkatabay": "Nkhata Bay", "Nkhatabay": "Nkhata Bay"}


def say(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        raise SystemExit(msg)


# ---------------------------------------------------------------- census

def read_e4():
    """Table E4, pages 132-133 of the report (0-based 142-143): an area name, then 14 figures
    (Total and the 13 tribes) in header order. Text comes out in that order; figures sometimes
    share a line, so the page is read as one token stream."""
    import fitz
    doc = fitz.open(PDF)
    rows = {}
    for pno in (142, 143):
        text = doc[pno].get_text()
        say("Table E4: Malawian Population byTribe, Region and District" in text,
            f"page {pno + 1} is Table E4")
        body = text.split("Other", 1)[1].split("Series E.", 1)[0]
        toks = re.findall(r"[A-Za-z][A-Za-z ]*[A-Za-z]|[\d,]+", body)
        i = 0
        while i < len(toks):
            name = toks[i]
            if name[0].isdigit() or not all(t[0].isdigit() for t in toks[i + 1:i + 15]):
                raise SystemExit(f"Table E4 p{pno + 1}: {name!r} is not a name and 14 figures")
            vals = [int(t.replace(",", "")) for t in toks[i + 1:i + 15]]
            rows[name] = vals
            i += 15
    df = pd.DataFrame(rows, index=["Total"] + TRIBES).T
    say(len(df) == 1 + 3 + 32, f"{len(df)} areas: Malawi, 3 regions, 32 districts")
    say((df[TRIBES].sum(1) == df["Total"]).all(), "every area's tribes sum to its total")
    for r in REGIONS:
        say((df.loc[DISTRICTS[r]].sum() == df.loc[r]).all(),
            f"{r}'s {len(DISTRICTS[r])} districts sum to the region row on all 14 columns")
    say((df.loc[REGIONS].sum() == df.loc["Malawi"]).all(), "regions sum to Malawi")
    say(df.loc["Malawi", "Total"] == MALAWIANS, f"Malawi = {MALAWIANS:,}")
    return df


# ---------------------------------------------------------------- survey

def survey():
    a = pd.read_csv(AB, keep_default_na=False)
    a = a[a["round"].isin(ROUNDS)].copy()
    for c in ("eth", "lang"):     # the merged files print Mang'anja's apostrophe three ways
        a[c] = a[c].str.replace("�", "'").str.replace("’", "'")
    ev = a["eth_verbatim"].str.upper().str.strip()
    a["tribe"] = a["eth"].map(ETH)
    a.loc[ev.isin(ETH_VERBATIM), "tribe"] = ev.map(ETH_VERBATIM)
    a["answer"] = a["lang"].map(LANG)
    vb = a["verbatim"].str.upper()
    for pat, ans in VERBATIM:
        hit = a["lang"].isin(["Other", "Others"]) & vb.str.contains(pat)
        a.loc[hit, "answer"] = ans
    unmapped = sorted(set(a.loc[a["answer"].isna() & ~a["lang"].isin(NOT_LANG), "lang"]))
    say(not unmapped, f"every language answer is mapped (unmapped {unmapped})")
    a = a[a["tribe"].notna() & a["answer"].notna()].copy()
    a["region"] = a["region"].str[0].map({"N": "Northern", "C": "Central", "S": "Southern"})
    say(a["region"].notna().all(), "every respondent has a region")
    a["district"] = a["district"].replace(DIST_FIX)
    has = a["district"] != ""
    bad = sorted(set(a.loc[has & ~a["district"].isin(CODE), "district"]))
    say(not bad, f"every district name is a census district ({bad})")
    wrong = a[has & (a["district"].map(REGION_OF) != a["region"])]
    say(len(wrong) == 0, f"district and region agree for all {has.sum():,} respondents")
    # Two Sena respondents in the lower Shire answered "Chisenga": a one-letter slip for Chisena,
    # since Senga is a Tumbuka variety of the far north and of Zambia. One respondent each
    # carried Chisenga to 1-3% of Nsanje and Chikwawa before this.
    a.loc[(a["answer"] == "Chisenga") & (a["tribe"] == "Sena"), "answer"] = "Chisena"
    # Chingoni is a Tumbuka variety in the north, a Nyanja one elsewhere (see docstring)
    a.loc[(a["answer"] == "Chingoni") & (a["region"] == "Northern"), "answer"] = \
        "Chingoni (northern)"
    print(f"  {len(a):,} respondents with a tribe and a language; {has.sum():,} name a district")
    return a


def shrink(counts, prior, k=K):
    idx = prior.index.union(counts.index)
    c = counts.reindex(idx, fill_value=0.0)
    p = prior.reindex(idx, fill_value=0.0)
    return (c + k * p) / (c.sum() + k)


REST = "_rest"


def lf_of(t):
    return [x for x in LF if not (t == "Chewa" and x == "Chichewa")]


def lf_counts(s, L):
    """Weighted counts over the lingua francas L and REST (everything else)."""
    x = s["answer"].where(s["answer"].isin(L), REST)
    return s["w"].groupby(x).sum()


def lf_shares(a):
    """{(district, tribe): Series over lf_of(tribe) + REST}, from LF_ROUNDS."""
    r7 = a[a["round"].isin(LF_ROUNDS)]
    base = r7[r7["tribe"] != "Chewa"]
    out = {}
    for t in TRIBES:
        L = lf_of(t)
        g = lf_counts(base, L)
        s = r7[r7["tribe"] == t]
        nat = shrink(lf_counts(s, L), g / g.sum(), K0)
        for r in REGIONS:
            sr = s[s["region"] == r]
            reg = shrink(lf_counts(sr, L), nat)
            for d in DISTRICTS[r]:
                out[(d, t)] = shrink(lf_counts(sr[sr["district"] == d], L), reg)
    return out


def shares(a):
    """{(district, tribe): Series answer -> share}: lingua francas from lf_shares, every other
    answer from all ROUNDS among the non-lingua-franca answers."""
    lfs = lf_shares(a)
    rest = shares_rest(a)
    out = {}
    for (d, t), p in rest.items():
        A = lfs[(d, t)]
        q = pd.concat([A.drop(REST), A[REST] * p]).groupby(level=0).sum()
        out[(d, t)] = q[q > 0] / q[q > 0].sum()
    return out


def shares_rest(a):
    """{(district, tribe): Series answer -> share} over the answers that are not the tribe's
    lingua francas."""
    out = {}
    for t in TRIBES:
        s = a[(a["tribe"] == t) & ~a["answer"].isin(lf_of(t))]
        nat = s.groupby("answer")["w"].sum()
        say(len(s) >= 3, f"{t}: {len(s):,} respondents not naming a lingua franca")
        nat = nat / nat.sum()
        for r in REGIONS:
            sr = s[s["region"] == r]
            reg = shrink(sr.groupby("answer")["w"].sum(), nat)
            # Chingoni's variant follows the region: re-point the prior's other-region form
            other = "Chingoni" if r == "Northern" else "Chingoni (northern)"
            this = "Chingoni (northern)" if r == "Northern" else "Chingoni"
            if other in reg.index and reg[other] > 0:
                reg[this] = reg.get(this, 0.0) + reg[other]
                reg = reg.drop(other)
            for d in DISTRICTS[r]:
                sd = sr[sr["district"] == d]
                p = shrink(sd.groupby("answer")["w"].sum(), reg)
                out[(d, t)] = p[p > 0] / p[p > 0].sum()
    return out


def main():
    e4 = read_e4()
    a = survey()
    sh = shares(a)
    rows = []
    for d, code in CODE.items():
        for t in TRIBES:
            n = e4.loc[d, t]
            if n == 0:
                continue
            for ans, p in sh[(d, t)].items():
                rows.append((code, d, t, ans, n * p))
    df = pd.DataFrame(rows, columns=["geo_id", "geo_name", "tribe", "source_category", "count"])
    per = df.groupby("geo_id")["count"].sum()
    tot = pd.Series({CODE[d]: e4.loc[d, "Total"] for d in CODE})
    say((per - tot).abs().max() < 0.01, "every district's languages sum to its census total")
    say(abs(df["count"].sum() - MALAWIANS) < 1, f"drawn total {df['count'].sum():,.0f}")
    out = (df.groupby(["geo_id", "geo_name", "source_category"], as_index=False)["count"].sum())
    out["count"] = out["count"].round(1)
    out.insert(1, "geo_level", "district")
    out["tier"] = "modelled"
    out["source_id"] = SOURCE_ID
    out["year"] = YEAR
    out["note"] = ""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(out):,} rows)")

    # the retention check, as the record quotes it
    nat = df.groupby(["tribe", "source_category"])["count"].sum().unstack(fill_value=0)
    print("\nnational, by tribe (census count shared by the survey's shares), % of tribe:")
    pct = (nat.div(nat.sum(1), axis=0) * 100).round(1)
    for t in TRIBES:
        top = pct.loc[t][pct.loc[t] >= 1].sort_values(ascending=False)
        print(f"  {t:10s} n={(a['tribe'] == t).sum():5d}  " +
              ", ".join(f"{k} {v}" for k, v in top.items()))
    print("\nnational by language:")
    lang = df.groupby("source_category")["count"].sum().sort_values(ascending=False)
    for k, v in lang.items():
        print(f"  {k:22s} {v:12,.0f}  {100 * v / MALAWIANS:5.2f}%")


if __name__ == "__main__":
    main()
