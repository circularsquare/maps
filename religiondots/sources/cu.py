"""Cuba: no census has asked religion since at least 1899; NORC's 2016 survey of 840 Cuban adults
is the one measurement with microdata, and its national shares are drawn in every province.

Reads data/raw/cu/norc_cuba_2017.dta (NORC at the University of Chicago, *A Rare Look Inside
Cuban Society: A New Survey of Cuban Public Opinion*, public use file, September 2017),
data/geo/cu/cu_lookup.csv (`sources/cu_geo.py`, ONEI's 2024 count per province) and
data/raw/estimates/pew.zip; writes data/normalized/cu.csv. `sources/cu.md` is the record;
`ask/RULINGS.md` 2026-09-15 and 2026-09-16 (draw a country no census asks on the best survey or
compiler figure, with the method said) the rulings.

## NOBODY COUNTS RELIGION

Every census form from 1899 to 2012 has been read and none asks (sources.md §11ap,
§scout-2026-09-15-negatives; 1970 and 1981 unread). The two polls fielded inside Cuba publish
national figures only (Bendixen & Amandi 2015 has no microdata).

## NORC 2016

In-person interviews, 840 adults 18 and over, a national random-route sample stratified by three
regions (west, centre, east) and settlement size; main fieldwork 3 October to 26 November 2016,
with April's pilot interviews kept; weighted (`finalwt`) to the 2012 census by age, sex and urban
or rural settlement. Areas of eastern Cuba holding about 15% of the population were left out after
Hurricane Matthew. Question Z10, "What is your religion, if any?". The public file merges the
card's Evangelical, Protestant and Christian (other) into one code, and has no region variable:
only `sector`, urban or rural.

So the shares are national and are drawn at one mix in every province. The urban-rural split is
tested and not drawn: across the seven answers chi-square p 0.14 (82 rural interviews), and the
one answer that leans (Santería, 19.2% urban against 9.8% rural, weighted permutation p 0.039)
does not survive seven tests (`urban_test`). Answers of don't know or refused (5) are left out of
the base.

Usage:
    python sources/cu.py            rebuild data/normalized/cu.csv and print the checks
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
from scipy import stats

from afrobarometer import round_within_rows

PUF = os.path.join(ROOT, "data", "raw", "cu", "norc_cuba_2017.dta")
PUF_ZIP_URL = ("https://www.norc.org/content/dam/norc-org/documents/standard-projects-pdf/"
               "NORC%20Cuba%20Public%20Use%20Files%20and%20Codebook.zip")
LOOKUP = os.path.join(ROOT, "data", "geo", "cu", "cu_lookup.csv")
PEW = os.path.join(ROOT, "data", "raw", "estimates", "pew.zip")
OUT = os.path.join(ROOT, "data", "normalized", "cu.csv")

# `religion` codes in the public file (its value labels, read 2026-10-03).
CATS = {
    1: "Catholic",
    2: "Santeria or Order of Osha",
    3: "Christian, not Catholic (Evangelical, Protestant or other Christian)",
    4: "Believe in god but do not belong to a particular religion",
    5: "Atheist",
    6: "None of the above",
    7: "Other",
}
DROPPED = {77: "Don't know", 99: "Refused"}
# The published topline (Cuba Topline_FINAL.pdf, Z10, percent of all 840 with DK and refused in the
# base): code 3 is its Evangelical 1, Protestant *, Christian (other) 6.
TOPLINE = {1: 28, 2: 17, 3: 7, 4: 22, 5: 1, 6: 24, 7: 1}
N_ALL = 840
NATIONAL_2024 = 9_748_007
YEAR = 2016
SOURCE_ID = "cu_norc2016_national_mix"
# Measured 2026-10-03 and asserted, so note_public cannot drift from the data.
NOTE = dict(n=835, catholic=28.3, santeria=16.9, christian=6.5, believer=21.8, atheist=1.1,
            none=24.4, other=1.0, pew_christian=60.7, pew_other=17.4, pew_unaff=21.6)


def load():
    df = pd.read_stata(PUF, convert_categoricals=False)
    if len(df) != N_ALL:
        raise SystemExit(f"{PUF} has {len(df)} rows, expected {N_ALL}")
    with pd.io.stata.StataReader(PUF) as r:
        labels = r.value_labels()
    lab = next(v for v in labels.values() if any("Santeria" in str(x) for x in v.values()))
    for code, name in CATS.items():
        got = str(lab.get(code, ""))
        if code != 3 and name.split(" (")[0].split(",")[0].lower()[:8] not in got.lower():
            raise SystemExit(f"religion code {code} is labelled {got!r}, expected {name!r}")
    if "Christian" not in str(lab.get(3, "")):
        raise SystemExit(f"religion code 3 is labelled {lab.get(3)!r}")
    df["religion"] = df["religion"].fillna(-1).astype(int)
    left = df[~df["religion"].isin(CATS)]
    print(f"  NORC 2016: {N_ALL} interviews; left out of the base: {len(left)} "
          f"(codes {sorted(left['religion'].unique())}: don't know, refused or blank)")
    return df[df["religion"].isin(CATS)].copy()


def urban_test(df, n_perm=5000):
    tab = pd.crosstab(df["religion"], df["sector"])
    chi, p, dof, _ = stats.chi2_contingency(tab.values)
    print(f"\n  urban-rural test: {int(tab[1].sum())} urban, {int(tab[2].sum())} rural interviews; "
          f"chi-square {chi:.2f} on {dof} df, p {p:.3f}")
    rng = np.random.default_rng(1)
    w, s, rel = df["finalwt"].to_numpy(), df["sector"].to_numpy(), df["religion"].to_numpy()

    def diff(sv):
        return {c: w[(rel == c) & (sv == 1)].sum() / w[sv == 1].sum()
                - w[(rel == c) & (sv == 2)].sum() / w[sv == 2].sum() for c in CATS}

    obs = diff(s)
    perm = [diff(rng.permutation(s)) for _ in range(n_perm)]
    for c in CATS:
        d = np.array([x[c] for x in perm])
        pc = float(np.mean(np.abs(d) >= abs(obs[c])))
        print(f"      {CATS[c][:40]:<40} urban minus rural {obs[c]:+.3f}, permutation p {pc:.3f}, "
              f"Bonferroni over {len(CATS)} {min(1.0, pc * len(CATS)):.3f}")
    return p


def pew_cuba():
    with zipfile.ZipFile(PEW) as z:
        name = [n for n in z.namelist() if n.endswith("(percentages).csv")][0]
        t = pd.read_csv(io.BytesIO(z.read(name)))
    r = t[(t["Country"] == "Cuba") & (t["Year"] == 2020)].iloc[0]
    print("  Pew Research Center 2020, everyone living in Cuba: " + ", ".join(
        f"{c} {float(r[c]):.1f}%" for c in ("Christians", "Religiously_unaffiliated", "Other_religions",
                                            "Hindus", "Muslims", "Buddhists", "Jews")))
    return (round(float(r["Christians"]), 1), round(float(r["Other_religions"]), 1),
            round(float(r["Religiously_unaffiliated"]), 1))


def main():
    df = load()
    w = df.groupby("religion")["finalwt"].sum()
    share = (w / w.sum()).reindex(list(CATS))
    print("  weighted national shares (adults 18+), n per answer:")
    for c in CATS:
        n = int((df["religion"] == c).sum())
        print(f"      {100 * share[c]:6.2f}%  n={n:>3}  {CATS[c]}")
    # the published topline, whose base also holds the 5 don't know or refused
    allw = pd.read_stata(PUF, convert_categoricals=False)["finalwt"].sum()
    for c, pct in TOPLINE.items():
        got = 100 * w[c] / allw
        if abs(got - pct) > 1.0:
            raise SystemExit(f"code {c}: {got:.2f}% of all 840 against the topline's {pct}%")
    print("  every answer within a point of the published topline (Z10)")

    p = urban_test(df)
    if p < 0.05:
        raise SystemExit("religion differs by urban and rural at 5%; the one-mix construction needs revisiting")

    pew = pew_cuba()

    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    if len(lut) != 16 or int(lut["pop"].sum()) != NATIONAL_2024:
        raise SystemExit(f"{LOOKUP}: {len(lut)} provinces, {int(lut['pop'].sum()):,}; re-run cu_geo.py")
    pop = lut.set_index("geo_id")["pop"]
    m = pd.DataFrame({CATS[c]: pop * share[c] for c in CATS})
    counts = round_within_rows(m)
    if not (counts.sum(axis=1) == pop.reindex(counts.index)).all():
        raise SystemExit("a province's rounded counts do not sum to its ONEI count")
    out = counts.stack().rename("count").reset_index()
    out.columns = ["geo_id", "source_category", "count"]
    out["geo_level"] = "province"
    out["geo_name"] = out["geo_id"].map(dict(zip(lut["geo_id"], lut["name"])))
    out["basis"] = "self_id"
    out["year"] = YEAR
    out["source_id"] = SOURCE_ID
    out["note"] = ("NORC 2016 national shares (835 adults, weighted), the same mix in every province, "
                   "on ONEI's 2024 count")
    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count", "basis", "year",
            "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {int(out['count'].sum()):,} people, 16 provinces)")

    r1 = lambda c: round(100 * float(share[c]), 1)       # noqa: E731
    got = dict(n=len(df), catholic=r1(1), santeria=r1(2), christian=r1(3), believer=r1(4),
               atheist=r1(5), none=r1(6), other=r1(7), pew_christian=pew[0], pew_other=pew[1],
               pew_unaff=pew[2])
    print(f"  note_public's figures: {got}")
    if got != NOTE:
        raise SystemExit(f"note_public's figures are {NOTE}; the build gives {got}")


if __name__ == "__main__":
    main()
