"""Gilgit-Baltistan and Azad Jammu & Kashmir, the two areas PBS's Table 11 leaves out
-> data/normalized/pk_north.csv (modelled rows, one per district and language).

    python sources/pk_north.py [--fetch]

The record is sources/pk.md. In short:

GILGIT-BALTISTAN (10 census districts, 1,709,030 people). The 2023 census did ask mother tongue
there, but the only published result is GB-wide: *GB at a Glance 2025* (P&DD Statistical &
Research Cell, citing "Census 2023, Pakistan Bureau of Statistics"), p.9: Shina 50.21, Balti
29.94, Pushto 0.86, Kohistani 0.86, Urdu 0.38, Others 17.74 percent. The district split comes
from the GB MICS 2016-17 microdata, mother tongue of the household head, weighted shares of
persons per district (sources/pk_mics.py -> data/normalized/pk_mics_gb2016.csv, since
2026-10-09). Until then it was the same survey's unweighted household counts as printed in Shah
Zaman, "Treading the Sacred Linguistic Landscape of Gilgit-Baltistan", Pamir Times, 2023-12-23;
that table is kept below (MICS17) as the check pk_mics.py runs against the microdata. The census
"Others" is divided among Burushaski, Khowar, Wakhi and a remainder by the GB MICS 2024-25
report, Table SR.3.1 (6,929 households, language of household head: Shina 48.0, Balti 29.2,
Burushaski 12.3, Khowar 5.2, Wakhi 1.0, Other 4.2), turned from households into persons with
2016-17's persons per household for each language. The district x language table is then raked
(IPF) to the census 2023 district populations (GB at a Glance p.4) and to those GB totals.

AZAD JAMMU & KASHMIR (10 districts, 4,333,467 people). AJK Statistical Year Book 2025, Table
15.31 "Languages Spoken in AJ&K", percent by district (source: Kashmir Liberation Cell,
Muzaffarabad), printed rounded to whole percents, times each district's 2023 census population
(the same yearbook's Table 15.24, as religiondots' pk.csv carries it). An estimate, not a census
count: every row is `modelled`.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD  # noqa: E402

RAW = HERE / "data" / "raw" / "pk_north"
OUT = HERE / "data" / "normalized" / "pk_north.csv"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
FILES = {
    "gb_at_glance_2025.pdf": "https://pnd.gog.pk/storage/downloads/AiRIlDEcscWPC1s58oXIgpjlVAS7jd-"
                             "metaR0IgQVQgR2xhbmNlIDIwMjUuMS5wZGY=-.pdf",
    "gb_mics_2024_25_sfr.pdf": "https://pnd.gog.pk/storage/downloads/oSrtpZkKNFTPMVipYa8VTI94BZmR2g-"
                               "metaR0IgTUlDUyAyMDI0LTI1IFN1cnZleSBGaW5kaW5ncyBSZXBvcnQucGRm-.pdf",
    "pamirtimes_2023-12-23_linguistic_landscape.html":
        "https://pamirtimes.net/2023/12/23/treading-the-sacred-linguistic-landscape-of-gilgit-baltistan/",
}
AJK_YEARBOOK = RD / "data" / "raw" / "pk2023" / "ajk_statistical_year_book_2025.pdf"

# ---------------------------------------------------------------- Gilgit-Baltistan

# GB at a Glance 2025 p.4, census 2023, (male, female) by district. The GB row is printed as
# 891,558 male and 817,472 female; the districts must add to it.
GB_POP = {
    "Astore": (58837, 52735), "Diamer": (174458, 162865), "Ghanche": (83216, 74603),
    "Ghizer": (100714, 99353), "Gilgit": (175006, 149539), "Hunza": (33694, 31803),
    "Kharmang": (32764, 28540), "Nagar": (44813, 42597), "Shigar": (43756, 40852),
    "Skardu": (144300, 134585),
}
GB_TOTAL = (891558, 817472)

# GB at a Glance 2025 p.9, "MOTHER TONGUE", census 2023, percent of GB.
GB_CENSUS = {"Shina": 50.21, "Balti": 29.94, "Pushto": 0.86, "Kohistani": 0.86, "Urdu": 0.38,
             "Others": 17.74}

# GB MICS 2024-25, Table SR.3.1 (report p.39), weighted percent of households by language of
# household head. Used only to divide the census's "Others".
MICS25 = {"Shina": 48.0, "Balti": 29.2, "Brushaski": 12.3, "Khowar": 5.2, "Wakhi": 1.0,
          "Other": 4.2}

# GB MICS 2016-17 sampled households by language of household head and district, as printed in
# the Pamir Times article (second table). No longer drawn (the weighted microdata is, through
# pk_mics_gb2016.csv); kept because sources/pk_mics.py asserts the microdata reproduces it. Columns: Balti, Shina, Burushaski, Khowar, Wakhi,
# Other languages; then the printed row total. MICS 2016-17's "Sikardu" is pre-2019 Skardu
# (with Rondu), as the census's is.
MICS17_COLS = ["Balti", "Shina", "Burushaski", "Khowar", "Wakhi", "Other"]
MICS17 = {
    "Astore": ([0, 602, 0, 0, 0, 3], 605),
    "Skardu": ([496, 131, 2, 1, 0, 16], 646),
    "Diamer": ([1, 498, 1, 0, 0, 85], 585),
    "Ghanche": ([707, 0, 0, 0, 0, 1], 708),
    "Ghizer": ([1, 244, 117, 177, 6, 46], 591),
    "Gilgit": ([7, 461, 80, 17, 8, 38], 611),
    "Hunza": ([0, 64, 389, 1, 140, 0], 594),
    "Kharmang": ([495, 77, 0, 0, 0, 29], 601),
    "Nagar": ([1, 148, 471, 0, 0, 0], 620),
    "Shigar": ([648, 3, 0, 0, 0, 1], 652),
}
MICS17_COLTOTAL = [2356, 2228, 1060, 196, 154, 219]      # printed "Total" row, 6,213
MICS17_DIVISIONS = {"Baltistan": (["Skardu", "Ghanche", "Kharmang", "Shigar"], 2607),
                    "Diamer": (["Astore", "Diamer"], 1190),
                    "Gilgit": (["Ghizer", "Gilgit", "Hunza", "Nagar"], 2416)}

# the census's Kohistani is placed in Diamer alone: it is the only GB district on the Indus
# Kohistan border, and Kohistani Shina is spoken in Darel and Tangir (now inside Diamer's
# census district). Pashto and Urdu follow each district's MICS "Other" households.
KOHISTANI_DISTRICTS = {"Diamer"}

# ---------------------------------------------------------------- Azad Kashmir

# AJK Statistical Year Book 2025, Table 15.31 (pdf p.226, printed p.190), percent. The table's
# columns are Kashmiri, Gojri, Pahari, Shina, Others, but several cells carry their own label:
# the Pahari cells name the local variety (Dhundi-Khairali, Chibali, Punchi, Pahari Pothwari,
# Mirpuri), and Bhimber's Shina and Others cells hold "30 Dogri" and "35 Punjabi". Each entry is
# (label as printed, percent); the variety names are kept in `note`.
AJK = {
    "muzaffarabad": [("Kashmiri", 15), ("Gojri", 35), ("Pahari", 50)],
    "neelum": [("Kashmiri", 20), ("Gojri", 10), ("Pahari", 63), ("Shina", 5),
               ("Kundal Shahi", 2)],
    "jhelum-valley": [("Kashmiri", 15), ("Gojri", 35), ("Pahari", 50)],
    "bagh": [("Kashmiri", 2), ("Gojri", 3), ("Pahari (Dhundi-Khairali)", 95)],
    "haveli": [("Kashmiri", 5), ("Gojri", 30), ("Pahari (Chibali)", 65)],
    "poonch": [("Gojri", 6), ("Pahari (Punchi)", 94)],
    "sudhnoti": [("Pahari (Punchi)", 95), ("Others", 5)],
    "kotli": [("Gojri", 35), ("Pahari (Pahari Pothwari)", 63), ("Others", 2)],
    "mirpur": [("Gojri", 10), ("Pahari (Mirpuri)", 85), ("Others", 2)],
    "bhimber": [("Gojri", 5), ("Pahari (Mirpuri)", 30), ("Dogri", 30), ("Punjabi", 35)],
}
AJK_TOTAL = 4333467


def fetch():
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        p = RAW / name
        if p.exists() and p.stat().st_size > 10_000:
            continue
        r = requests.get(url, headers=UA, timeout=600)
        r.raise_for_status()
        p.write_bytes(r.content)
        print(f"  fetched {name} ({len(r.content):,} bytes)")


def check_sources():
    """The figures above are transcribed; check what can be checked against the files."""
    import fitz
    t = fitz.open(RAW / "gb_at_glance_2025.pdf")[3].get_text().replace(",", "")
    for d, (m, f) in GB_POP.items():
        if str(m) not in t or str(f) not in t:
            raise SystemExit(f"GB at a Glance p.4: {d} {m}/{f} not found in the page text")
    mics = " ".join(fitz.open(RAW / "gb_mics_2024_25_sfr.pdf")[56].get_text().split())
    for k, v in MICS25.items():
        if f"{k} {v:.1f}" not in mics:
            raise SystemExit(f"MICS 2024-25 Table SR.3.1: '{k} {v:.1f}' not in page text")
    html = (RAW / "pamirtimes_2023-12-23_linguistic_landscape.html").read_text(encoding="utf-8")
    import re
    cells = [c.strip() for c in re.sub(r"<[^>]+>", "|", html).split("|") if c.strip()]
    start = next(i for i, c in enumerate(cells) if c.startswith("Districts/"))  # second table
    for d, (row, tot) in MICS17.items():
        name = "Sikardu" if d == "Skardu" else d
        i = cells.index(name, start)
        got = [int(x) for x in cells[i + 1:i + 8]]
        if got != row + [tot]:
            raise SystemExit(f"Pamir Times table, {d}: page has {got}, transcribed {row + [tot]}")
    aj = fitz.open(AJK_YEARBOOK)[225].get_text()
    for lab in ("Table:15.31", "Dhundi-Khairali", "Chibali", "Punchi", "Pahari Pothwari",
                "Mirpuri", "Dogri", "Punjabi", "Kundal Shahi", "Kashmir Liberation Cell"):
        if lab not in aj:
            raise SystemExit(f"AJK yearbook p.226: '{lab}' not in page text")
    print("  transcriptions agree with the source files")


def ipf(seed, rows, cols, iters=500):
    m = seed.copy()
    for _ in range(iters):
        m = m.mul(rows / m.sum(axis=1), axis=0)
        m = m.mul(cols / m.sum(axis=0), axis=1)
    return m


def gilgit_baltistan():
    # transcription checks
    for d, (row, tot) in MICS17.items():
        assert sum(row) == tot, d
    assert [sum(MICS17[d][0][j] for d in MICS17) for j in range(6)] == MICS17_COLTOTAL
    for div, (ds, tot) in MICS17_DIVISIONS.items():
        assert sum(MICS17[d][1] for d in ds) == tot, div
    assert tuple(map(sum, zip(*GB_POP.values()))) == GB_TOTAL
    assert abs(sum(GB_CENSUS.values()) - 100) < 0.011
    assert abs(sum(MICS25.values()) - 100) < 0.11

    pop = pd.Series({d: m + f for d, (m, f) in GB_POP.items()})
    total = pop.sum()

    # MICS 2016-17 microdata (sources/pk_mics.py): weighted shares per district
    mp = HERE / "data" / "normalized" / "pk_mics_gb2016.csv"
    if not mp.exists():
        raise SystemExit(f"{mp} missing: run sources/pk_mics.py first")
    m17 = pd.read_csv(mp)
    if set(m17["district"]) != set(pop.index):
        raise SystemExit(f"pk_mics_gb2016.csv districts {sorted(set(m17['district']))}")
    share17 = m17.pivot(index="district", columns="label", values="persons_share").fillna(0)
    # persons per household by language, GB-wide, 2016-17 (district shares weighted by 2023
    # population): turns 2024-25's household shares into persons' shares (Wakhi households are
    # small, Burushaski ones a little small)
    hp = m17.assign(p=m17.persons_share * pop.reindex(m17.district).values,
                    h=m17.households_share * pop.reindex(m17.district).values)
    hp = hp.groupby("label")[["p", "h"]].sum()
    ratio = (hp.p / hp.p.sum()) / (hp.h / hp.h.sum())
    ratio["Other"] = ((hp.p["Other"] + hp.p["Urdu"]) / hp.p.sum()) / \
                     ((hp.h["Other"] + hp.h["Urdu"]) / hp.h.sum())   # 2024-25 has Urdu in Other
    m25 = {"Shina": MICS25["Shina"], "Balti": MICS25["Balti"], "Burushaski": MICS25["Brushaski"],
           "Khowar": MICS25["Khowar"], "Wakhi": MICS25["Wakhi"], "Other": MICS25["Other"]}
    m25p = {k: v * ratio[k] for k, v in m25.items()}
    m25p = {k: v * 100 / sum(m25p.values()) for k, v in m25p.items()}
    print("MICS 2024-25 households -> persons (%): " + ", ".join(
        f"{k} {m25[k]:.1f} -> {m25p[k]:.1f}" for k in m25))

    # GB-wide targets: the census's own categories, and its Others divided by MICS 2024-25
    pku = GB_CENSUS["Pushto"] + GB_CENSUS["Kohistani"] + GB_CENSUS["Urdu"]
    rest = m25p["Other"] - pku            # MICS "Other" also holds what the census names apart
    if rest <= 0:
        raise SystemExit("MICS 'Other' smaller than the census's Pashto+Kohistani+Urdu")
    others = {"Burushaski": m25p["Burushaski"], "Khowar": m25p["Khowar"],
              "Wakhi": m25p["Wakhi"], "Other": rest}
    osum = sum(others.values())
    target = {"Shina": GB_CENSUS["Shina"], "Balti": GB_CENSUS["Balti"],
              "Pashto": GB_CENSUS["Pushto"], "Kohistani": GB_CENSUS["Kohistani"],
              "Urdu": GB_CENSUS["Urdu"]}
    target.update({k: GB_CENSUS["Others"] * v / osum for k, v in others.items()})
    cols = pd.Series(target) / 100 * total
    cols = cols * total / cols.sum()

    # seed: MICS 2016-17 weighted district shares x 2023 population; "Other" split four ways,
    # Urdu its own answer plus its part of "Other"
    seed = pd.DataFrame(0.0, index=pop.index, columns=cols.index)
    for d in pop.index:
        share = share17.loc[d]
        for k in ("Shina", "Balti", "Burushaski", "Khowar", "Wakhi"):
            seed.loc[d, k] = share.get(k, 0) * pop[d]
        oth = share.get("Other", 0) * pop[d]
        seed.loc[d, "Other"] = oth * rest / m25p["Other"]
        seed.loc[d, "Pashto"] = oth * GB_CENSUS["Pushto"] / m25p["Other"]
        seed.loc[d, "Urdu"] = share.get("Urdu", 0) * pop[d] + oth * GB_CENSUS["Urdu"] / m25p["Other"]
        if d in KOHISTANI_DISTRICTS:
            seed.loc[d, "Kohistani"] = oth * GB_CENSUS["Kohistani"] / m25p["Other"]
    fit = ipf(seed, pop.astype(float), cols)
    err_r = (fit.sum(axis=1) - pop).abs().max()
    err_c = (fit.sum(axis=0) - cols).abs().max()
    if err_r > 1 or err_c > 1:
        raise SystemExit(f"IPF did not converge: rows off {err_r:.1f}, columns off {err_c:.1f}")

    print(f"\nGilgit-Baltistan, {total:,} people; raked shares per district (%):")
    print((fit.div(pop, axis=0) * 100).round(1).to_string())
    raw = seed.sum(axis=0) / seed.values.sum() * 100
    print("GB-wide, seed (MICS 2016-17 weighted shares x 2023 population) against target "
          "(census 2023):")
    for k in cols.index:
        print(f"  {k:11s} seed {raw[k]:5.2f}%  target {cols[k] / total * 100:5.2f}%")

    rows = []
    for d in fit.index:
        # integer counts that keep each district's census total exactly (largest remainder)
        v = fit.loc[d]
        fl = v.astype(int)
        short = int(pop[d] - fl.sum())
        for k in (v - fl).sort_values(ascending=False).index[:short]:
            fl[k] += 1
        for k, c in fl.items():
            if c > 0:
                rows.append(dict(geo_id=f"PK23-gilgit-baltistan/{d.lower()}-district",
                                 geo_level="district", geo_name=d.upper(), source_category=k,
                                 count=int(c), year=2023, source_id="pk_gb_census2023_mics_ipf",
                                 note="modelled: MICS 2016-17 weighted district shares raked to "
                                      "census 2023 district population and GB mother-tongue "
                                      "totals"))
    df = pd.DataFrame(rows)
    assert df.groupby("geo_id")["count"].sum().sum() == total
    return df


def azad_kashmir():
    n = pd.read_csv(RD / "data" / "normalized" / "pk.csv", dtype={"geo_id": str},
                    low_memory=False)
    n = n[(n["geo_level"] == "district") & n["geo_id"].str.startswith("PK23-azad-jammu-and-kashmir/")]
    pop = n.groupby("geo_id")["count"].sum()
    if len(pop) != 10 or int(pop.sum()) != AJK_TOTAL:
        raise SystemExit(f"religiondots pk.csv AJK: {len(pop)} districts, {int(pop.sum()):,}")
    rows = []
    for d, cells in AJK.items():
        gid = f"PK23-azad-jammu-and-kashmir/{d}-district"
        if gid not in pop.index:
            raise SystemExit(f"{gid} not in religiondots' AJK districts {sorted(pop.index)}")
        s = sum(p for _, p in cells)
        # every row prints 100 except Mirpur, 10 + 85 + 2 = 97; its shares are scaled up
        if s != (97 if d == "mirpur" else 100):
            raise SystemExit(f"Table 15.31, {d}: {s}%")
        P = int(pop[gid])
        counts = [int(P * p / s) for _, p in cells]
        # largest remainder, so each district keeps its census total
        rem = sorted(range(len(cells)), key=lambda i: -(P * cells[i][1] / s - counts[i]))
        for i in rem[:P - sum(counts)]:
            counts[i] += 1
        for (lab, p), c in zip(cells, counts):
            rows.append(dict(geo_id=gid, geo_level="district", geo_name=d.upper(),
                             source_category=lab, count=c, year=2023,
                             source_id="pk_ajk_yearbook2025_t15_31",
                             note=f"modelled: {p}% (Table 15.31) x census 2023 population {P}"))
    df = pd.DataFrame(rows)
    assert df["count"].sum() == AJK_TOTAL
    print(f"\nAzad Kashmir, {AJK_TOTAL:,} people:")
    print(df.groupby("source_category")["count"].sum().sort_values(ascending=False).to_string())
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    check_sources()
    df = pd.concat([gilgit_baltistan(), azad_kashmir()], ignore_index=True)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT} ({len(df)} rows, {df['count'].sum():,} people)")


if __name__ == "__main__":
    main()
