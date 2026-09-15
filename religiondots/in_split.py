"""India: split the 2011 census `Muslim` column into Sunni and Shi'a by Pew's 2021 regions.

Pew's volunteered Ahmadiyya share was drawn on `islam.ahmadiyya` and folded back into `islam` the
same day, Anita 2026-09-14; see FOLDED below.

Writes data/normalized/in_split.csv: every sub-district's `Muslim` count, replaced by the branch
rows Pew's survey names plus the census's own `Muslim` for the rest, summing to the census figure
to the person. countries.py::_in_counts swaps these in for the allocated file's `Muslim` rows.

WHY THIS EXISTS
---------------
India draws 172.2M Muslims, and until 2026-09-14 every one of them sat on the bare `islam` node:
the census clubs every sect into `Muslim`, and its Annexure's 573 Shia are write-ins, not a count
(sources/in.md §4). Anita, 2026-09-14, on Pew's *Religion in India: Tolerance and Segregation*
(2021): *"i feel like the india data would help. we leave the no sect and i dont know as
unspecified. i think in general even if we have a lot of unspecified, thats fine and expected."*
That ruling is spec §2.7a.

THE SOURCE
----------
Pew Research Center, topline `PF_06.29.21_India_topline.pdf`, printed p. 23, item QSECT *"Are
you ...?"*, asked of Muslims (N=3,336), by six regions. Fieldwork 17 Nov 2019 to 23 Mar 2020,
face to face, RTI International. % of Muslims, as printed:

    region      N      Sunni  Shi'a  other  no sect  Ahmadiyya(vol.)  DK/ref
    India       3,336  55     6      2      14       1                22
    Northeast   512    32     5      4      20       0                38
    North       655    80     7      1      6        0                6
    Central     202    79     5      1      2        0                12
    East        1,016  43     2      1      18       0                35
    West        579    50     11     4      17       1                17
    South       372    38     6      7      24       4                22

`check_source()` re-reads those 56 numbers off the PDF, so a transcription slip fails the build.

**THE ROWS DO NOT ALL SUM TO 100, AND NOTHING HERE ROUNDS THEM.** Northeast, Central and East sum
to 99 and South to 101; Pew's own `Total` column prints 100 for each, so it is independent
rounding of each cell. The three named shares are carried exactly as printed, and the census
Muslims they do not account for stay on `islam`. The rounding therefore lands on the unspecified
remainder, which is the one cell that is already a sum of three printed cells.

THE REGIONS
-----------
Page numbers below are the full report's PDF pages (the printed folio is one lower).

Pew's regions are the zonal councils (report p. 16, and p. 224 footnote 26). The map on p. 16
draws them and the text lists four of them:

    North      Chandigarh, Delhi, Haryana, Himachal Pradesh, Jammu and Kashmir, Ladakh, Punjab,
               Rajasthan (p. 123)
    Central    Chhattisgarh, Madhya Pradesh, Uttar Pradesh, Uttarakhand (p. 16 map only)
    East       Bihar, Jharkhand, Odisha, West Bengal (p. 33)
    Northeast  Arunachal Pradesh, Assam, Manipur, Meghalaya, Mizoram, Nagaland, Sikkim, Tripura
               (p. 16 map only)
    West       Goa, Gujarat, Maharashtra (p. 43); the p. 16 map also draws Dadra and Nagar Haveli
               and Daman and Diu inside it
    South      Andhra Pradesh, Karnataka, Kerala, Tamil Nadu, Telangana, Puducherry (p. 49)

The 2011 census has Andhra Pradesh undivided, and both halves are South, so nothing is lost.

WHAT PEW DID NOT SURVEY STAYS ON `islam`
---------------------------------------
Anita's line is *whether anything measured the place* (queue.md, Ecuador's Galápagos), not how
thin the sample is. The grain here is the state or union territory, which is Pew's stratum
(p. 224), and Pew's own p. 16 note names every one where no interview took place:

  * Kashmir Valley: *"Fieldwork could not be conducted in the Kashmir Valley due to security
    concerns"*, and p. 227 drops *"Kashmir districts due to continued shutdown of the Kashmir
    Valley"* after sampling. Pew names no districts. Read here as the 2011 census's Kashmir
    division, the ten districts in KASHMIR_VALLEY. Jammu division was surveyed (p. 222: *"Jammu &
    Kashmir Feb. 20 - March 8, 2020"*, and p. 229 moves the Valley's 480 interviews to Jammu among
    other places).
  * Manipur and Sikkim: stopped by COVID-19, no interviews (p. 16, p. 229).
  * Ladakh, Chandigarh, Dadra and Nagar Haveli, Daman and Diu: *"No locations ... were selected
    for inclusion in the survey"* (p. 16). In the frame, and still nothing measured them. Ladakh
    is the 2011 districts of Leh (Ladakh) and Kargil, and it is also hatched as not surveyed on
    the p. 16 map.
  * Lakshadweep, Andaman and Nicobar Islands: outside the sample design (p. 224, footnote 27).

p. 25 also mentions *"a few districts elsewhere"* dropped for security. Pew does not name them,
so nothing can be done about them and they get their region's shares.

THE METHOD, AND ITS COST
------------------------
Per sub-district: census Muslims x each named share, split by largest remainder over (Sunni,
Shi'a, Ahmadiyya, unspecified), with the Ahmadiyya part then added to unspecified, so the parts sum
to the measured count exactly. **The composition
is uniform inside a region**, because six regions is the finest geography Pew publishes. So the
known concentrations cannot show (spec §3.10's allocation cost): Lucknow's Shia get Central's 5%
like everywhere else in Uttar Pradesh, Hyderabad's get the South's 6%, and the Bohras of Gujarat
and Maharashtra are inside the West's 11%. Kargil, where Muslims are mostly Shia, would have got
the North's 80% Sunni, and does not only because nothing measured Ladakh.

TIER: `derived`, NOT `modelled`, AND THE ROLL-UP IS THE REASON
--------------------------------------------------------------
Türkiye's rows are `modelled` because nobody counted religion there at any level. India's
Muslims WERE counted, at this sub-district, by the census; only which branch is from the survey.
That is spec §7a-i's Israel case (*"what is inferred is only which branch, never that the people
exist"*) and uk_split.py's England, where a survey's national mix divides a counted `Christian`
column and every row is `derived`. It also decides what the viewer does: index.html rolls only
`derived` dots up to their column and removes `modelled` ones outright, so `modelled` here would
make `inferred dots: not shown` delete about 100 million counted Muslims from India instead of
redrawing them on `islam`. `parent_column=Muslim` and in2011.COLUMNS carry the roll.

The unspecified remainder is written back as the census's own `Muslim` category, still
`measured`, just smaller, exactly as uk_split.py's REMAINDER is. So are the unsurveyed units.

Usage:
    python in_split.py             build data/normalized/in_split.csv
    python in_split.py --dry-run   report the numbers, write nothing
"""

import os
import re
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
NORM = os.path.join(HERE, "data", "normalized")
SRC_ALLOC = os.path.join(NORM, "in_subdistrict_allocated.csv")
SRC_CENSUS = os.path.join(NORM, "in.csv")
TOPLINE = os.path.join(HERE, "data", "raw", "in", "pew_2021", "PF_06.29.21_India_topline.pdf")
OUT = os.path.join(NORM, "in_split.csv")

COL = "Muslim"

# QSECT, topline p. 23. (N, [Sunni, Shi'a, some other sect, no sect in particular,
# Ahmadiyya (volunteered), DK/refused]) in the order the topline prints the rows.
QSECT = {
    "India":     (3336, [55, 6, 2, 14, 1, 22]),
    "Northeast": (512,  [32, 5, 4, 20, 0, 38]),
    "North":     (655,  [80, 7, 1, 6, 0, 6]),
    "Central":   (202,  [79, 5, 1, 2, 0, 12]),
    "East":      (1016, [43, 2, 1, 18, 0, 35]),
    "West":      (579,  [50, 11, 4, 17, 1, 17]),
    "South":     (372,  [38, 6, 7, 24, 4, 22]),
}
# Which printed cells name a branch that is drawn, and the category string each is written as.
# Some other sect, no sect in particular and DK/refused stay on `islam` (spec §2.7a).
NAMED = {0: "Muslim: Sunni (Pew 2021)", 1: "Muslim: Shi'a (Pew 2021)"}
# FOLDED BACK INTO `islam`, Anita 2026-09-14: "lets move ahmadiya back for now". Cell 4 is the
# volunteered Ahmadiyya code. South's 4% is about fifteen respondents and became 1.17M people,
# against a national figure of about 150,000, and East reads 0 where Odisha has organised Ahmadi
# villages (sources/branches.md, "India's Ahmadis"). It is still split out as its own part and
# then added to the remainder, so Sunni and Shi'a come out to the person as they did when it was
# drawn.
FOLDED = {4: "Ahmadiyya (volunteered)"}
REMAINDER = COL

# 2011 census state code -> (the census's own spelling, Pew region or None). Names asserted
# against in.csv so a code typo cannot silently move a state ([[reference_name_join_wrong_neighbour]]).
STATES = {
    "01": ("JAMMU & KASHMIR", "North"),        # minus KASHMIR_VALLEY and LADAKH below
    "02": ("HIMACHAL PRADESH", "North"),
    "03": ("PUNJAB", "North"),
    "04": ("CHANDIGARH", None),
    "05": ("UTTARAKHAND", "Central"),
    "06": ("HARYANA", "North"),
    "07": ("NCT OF DELHI", "North"),
    "08": ("RAJASTHAN", "North"),
    "09": ("UTTAR PRADESH", "Central"),
    "10": ("BIHAR", "East"),
    "11": ("SIKKIM", None),
    "12": ("ARUNACHAL PRADESH", "Northeast"),
    "13": ("NAGALAND", "Northeast"),
    "14": ("MANIPUR", None),
    "15": ("MIZORAM", "Northeast"),
    "16": ("TRIPURA", "Northeast"),
    "17": ("MEGHALAYA", "Northeast"),
    "18": ("ASSAM", "Northeast"),
    "19": ("WEST BENGAL", "East"),
    "20": ("JHARKHAND", "East"),
    "21": ("ODISHA", "East"),
    "22": ("CHHATTISGARH", "Central"),
    "23": ("MADHYA PRADESH", "Central"),
    "24": ("GUJARAT", "West"),
    "25": ("DAMAN & DIU", None),
    "26": ("DADRA & NAGAR HAVELI", None),
    "27": ("MAHARASHTRA", "West"),
    "28": ("ANDHRA PRADESH", "South"),         # includes Telangana in 2011; both are South
    "29": ("KARNATAKA", "South"),
    "30": ("GOA", "West"),
    "31": ("LAKSHADWEEP", None),
    "32": ("KERALA", "South"),
    "33": ("TAMIL NADU", "South"),
    "34": ("PUDUCHERRY", "South"),
    "35": ("ANDAMAN & NICOBAR ISLANDS", None),
}
NOT_SURVEYED_STATE = {
    "04": "no location selected (Pew p. 16)",
    "11": "no interviews, COVID-19 (Pew p. 16)",
    "14": "no interviews, COVID-19 (Pew p. 16)",
    "25": "no location selected (Pew p. 16)",
    "26": "no location selected (Pew p. 16)",
    "31": "outside the sample design (Pew p. 224)",
    "35": "outside the sample design (Pew p. 224)",
}
# The 2011 census's Kashmir division. Pew says "Kashmir districts" and names none.
KASHMIR_VALLEY = {
    "01001": "Kupwara", "01002": "Badgam", "01008": "Baramula", "01009": "Bandipore",
    "01010": "Srinagar", "01011": "Ganderbal", "01012": "Pulwama", "01013": "Shupiyan",
    "01014": "Anantnag", "01015": "Kulgam",
}
LADAKH = {"01003": "Leh(Ladakh)", "01004": "Kargil"}
NOT_SURVEYED_DISTRICT = {
    **{d: "Kashmir Valley, not surveyed for security (Pew p. 16)" for d in KASHMIR_VALLEY},
    **{d: "Ladakh, no location selected (Pew p. 16)" for d in LADAKH},
}

NOTE_NAMED = ("level=leaf; derivation=survey_share; structure=pew_india_2021_qsect; "
              "structure_geo=pew_region:{region}; share={share}%; parent_column=Muslim")
NOTE_REST = ("level=leaf; cat=Muslim; derivation=exact_single_child; branch not named "
             "(some other sect, no sect in particular, DK/refused, and Ahmadiyya, folded back "
             "2026-09-14; Pew 2021 {region}); "
             "parent_column=Muslim")
NOTE_UNSPLIT = ("level=leaf; cat=Muslim; derivation=exact_single_child; branch not split, "
                "{why}; parent_column=Muslim")
OUT_COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count", "basis",
               "year", "source_id", "tier", "note"]


def check_source():
    """Re-read QSECT's 56 numbers off the topline PDF and compare them with QSECT above."""
    if not os.path.exists(TOPLINE):
        raise SystemExit(f"missing {TOPLINE}; see sources/in.md §8 for the URL")
    import fitz
    doc = fitz.open(TOPLINE)
    if doc.page_count < 23:
        raise SystemExit(f"topline has {doc.page_count} pages, expected at least 23")
    text = doc[22].get_text()
    i, j = text.find("QSECT"), text.find("ASK IF CHRISTIAN")
    if i < 0 or j < i:
        raise SystemExit("QSECT block not found on topline p. 23")
    block = text[i:j]
    k = block.rfind("South")
    nums = [int(x) for x in re.findall(r"\b\d+\b", block[k:])]
    expected = []
    for n, shares in QSECT.values():
        expected += shares + [100, n]
    if nums != expected:
        raise SystemExit(f"QSECT on the PDF does not match the table here:\n  pdf  {nums}\n"
                         f"  here {expected}")
    for region, (n, shares) in QSECT.items():
        if not 99 <= sum(shares) <= 101:
            raise SystemExit(f"{region}: printed shares sum to {sum(shares)}")
    print(f"  OK QSECT re-read off the topline PDF, all {len(nums)} numbers match")


def _check_names():
    census = pd.read_csv(SRC_CENSUS, dtype={"geo_id": str}, low_memory=False,
                         usecols=["geo_id", "geo_level", "geo_name", "source_category"])
    st = census[(census["geo_level"] == "state") & (census["source_category"] == COL)]
    got = dict(zip(st["geo_id"], st["geo_name"]))
    want = {k: v[0] for k, v in STATES.items()}
    if got != want:
        diff = {k: (got.get(k), want.get(k)) for k in set(got) | set(want)
                if got.get(k) != want.get(k)}
        raise SystemExit(f"state codes disagree with in.csv: {diff}")
    di = census[(census["geo_level"] == "district") & (census["source_category"] == COL)]
    dnames = dict(zip(di["geo_id"], di["geo_name"]))
    for code, name in {**KASHMIR_VALLEY, **LADAKH}.items():
        if dnames.get(code) != name:
            raise SystemExit(f"district {code}: in.csv says {dnames.get(code)!r}, "
                             f"this file says {name!r}")
    for code in NOT_SURVEYED_STATE:
        if STATES[code][1] is not None:
            raise SystemExit(f"{code} is not surveyed and still has a region")
    for code, (_, region) in STATES.items():
        if region is None and code not in NOT_SURVEYED_STATE:
            raise SystemExit(f"{code} has no region and no recorded reason")
    print("  OK 35 state codes and 12 unsurveyed district codes match in.csv by name")


def _largest_remainder(counts_by_share, total):
    """Integer split of `total` by integer percentages, summing to `total` exactly."""
    exact = [total * s / 100.0 for s in counts_by_share]
    base = [int(x) for x in exact]
    short = total - sum(base)
    order = sorted(range(len(exact)), key=lambda i: -(exact[i] - base[i]))
    for i in order[:short]:
        base[i] += 1
    return base


def build():
    check_source()
    _check_names()

    df = pd.read_csv(SRC_ALLOC, dtype={"geo_id": str}, low_memory=False)
    mus = df[df["source_category"] == COL].copy()
    if mus.empty:
        raise SystemExit(f"no `{COL}` rows in {SRC_ALLOC}")
    if (mus["tier"] != "measured").any():
        raise SystemExit("some `Muslim` rows are not measured; the allocation changed shape")
    if mus["geo_id"].duplicated().any():
        raise SystemExit("a sub-district carries two `Muslim` rows")
    if ((mus["count"] - mus["count"].round()).abs() > 1e-9).any():
        raise SystemExit("a census `Muslim` count is not a whole number")
    mus["count"] = mus["count"].round().astype("int64")
    mus = mus[mus["count"] > 0]
    unknown = sorted(set(mus["geo_id"].str[:2]) - set(STATES))
    if unknown:
        raise SystemExit(f"state codes with no entry in STATES: {unknown}")

    rows, per_region = [], {}
    for r in mus.itertuples(index=False):
        state, district = r.geo_id[:2], r.geo_id[:5]
        region = STATES[state][1]
        why = NOT_SURVEYED_DISTRICT.get(district) or NOT_SURVEYED_STATE.get(state)
        base = dict(geo_id=r.geo_id, geo_level=r.geo_level, geo_name=r.geo_name,
                    basis=r.basis, year=r.year, source_id=r.source_id)
        if why is not None:
            rows.append(dict(base, source_category=REMAINDER, count=int(r.count),
                             tier="measured", note=NOTE_UNSPLIT.format(why=why)))
            per_region.setdefault("not surveyed", {}).setdefault(REMAINDER, 0)
            per_region["not surveyed"][REMAINDER] += int(r.count)
            continue
        shares = QSECT[region][1]
        named = [shares[i] for i in NAMED]
        folded = [shares[i] for i in FOLDED]
        # FOLDED cells are split as parts of their own and then summed into the remainder, which
        # keeps every named part identical to the split that drew them (see FOLDED).
        parts = _largest_remainder(named + folded + [100 - sum(named) - sum(folded)],
                                   int(r.count))
        rest = sum(parts[len(NAMED):])
        tally = per_region.setdefault(region, {})
        for (idx, label), n in zip(NAMED.items(), parts[:len(NAMED)]):
            tally[label] = tally.get(label, 0) + n
            if n > 0:
                rows.append(dict(base, source_category=label, count=n, tier="derived",
                                 note=NOTE_NAMED.format(region=region, share=shares[idx])))
        tally[REMAINDER] = tally.get(REMAINDER, 0) + rest
        if rest > 0:
            rows.append(dict(base, source_category=REMAINDER, count=rest,
                             tier="measured", note=NOTE_REST.format(region=region)))

    out = pd.DataFrame(rows, columns=OUT_COLUMNS)
    _check(out, mus, per_region)
    return out


def _check(out, mus, per_region):
    per_unit = out.groupby("geo_id")["count"].sum()
    measured = mus.set_index("geo_id")["count"]
    diff = (per_unit.reindex(measured.index).fillna(0) - measured).abs()
    if diff.max() != 0 or set(per_unit.index) != set(measured.index):
        raise SystemExit(f"in_split does not conserve sub-district Muslims "
                         f"(worst difference {diff.max():,.0f})")
    total = int(measured.sum())
    print(f"  OK every one of {len(measured):,} sub-districts' rows sum to its census "
          f"`Muslim` count, {total:,} in all")

    print("\n  region        Muslims       Sunni      Shi'a  unspecified")
    labels = list(NAMED.values())
    for region in list(QSECT)[1:] + ["not surveyed"]:
        t = per_region.get(region, {})
        m = sum(t.values())
        print(f"  {region:<12}{m:>10,}" + "".join(f"{t.get(l, 0):>11,}" for l in labels)
              + f"{t.get(REMAINDER, 0):>13,}")
    nat = out.groupby("source_category")["count"].sum()
    surveyed = total - per_region.get("not surveyed", {}).get(REMAINDER, 0)
    print(f"  {'India':<12}{total:>10,}" + "".join(f"{int(nat.get(l, 0)):>11,}" for l in labels)
          + f"{int(nat.get(REMAINDER, 0)):>13,}")

    # A relationship, not an identity: Pew weights its national row by adult Muslims in the
    # surveyed population, and this weights by every census Muslim in the surveyed units.
    print("\n  census-weighted share of surveyed Muslims against Pew's own national row:")
    for idx, label in NAMED.items():
        got = 100.0 * int(nat.get(label, 0)) / surveyed
        print(f"      {label:<30} {got:5.1f}%   (Pew national {QSECT['India'][1][idx]}%)")
    ns = per_region.get("not surveyed", {}).get(REMAINDER, 0)
    print(f"  left undivided where Pew did not survey: {ns:,} ({100.0 * ns / total:.2f}% of "
          f"India's Muslims)")


def main():
    out = build()
    if "--dry-run" in sys.argv:
        print("  --dry-run: nothing written")
        return
    tmp = OUT + ".part"
    out.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, OUT)
    print(f"  wrote {OUT} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
