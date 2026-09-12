"""South Africa — religion and Christian denomination, Community Survey 2016, by LOCAL
MUNICIPALITY, from the person microdata.

Writes data/normalized/za.csv: 213 municipalities x 27 categories.

**THIS COUNTRY WAS DRAWN AT NINE PROVINCES UNTIL 2026-09-09 AND THIS FILE IS THE REPLACEMENT.**
The province build read the nine published CS 2016 provincial profiles, which are the finest
open tabulation of this variable anywhere; `sources/za_profiles.py` is that parser, kept.
6.1 million people per unit was the coarsest counting geography on the map. Anita registered
with DataFirst and downloaded catalogue 611, so the same survey is now read per person:
**3,328,867 person records, 213 municipalities, ~261,000 people per unit, the same 24
published categories.** ask/answered/002-za is the ruling.

THE RELEASE IS STILL THE SURVEY AND NOT THE 2022 CENSUS, DELIBERATELY. Stats SA's Census 2022
publishes religion for the nine provinces over eleven categories with `Christianity` as one
undivided cell holding 83.6% of the country. CS 2016 splits Christianity fourteen ways, which
on the country that is the historic home of the African Independent Churches is the whole
reason to draw it. §3.1 forbids mixing the two and they are not interchangeable even where
their category lists match verbatim: CS 2016 puts `No religious affiliation/belief` at 10.9%
of answers and Census 2022 at 2.9%. See sources/za.md §3.

THE SURVEY IS DESIGNED FOR THIS TIER AND SAYS SO IN ITS OWN WORDS. Report 03-01-07 §1.2:
*"This household-based survey is one of the few available data sources providing data at
municipal level."* §1.2.3: *"it became clear that there is an increased demand for data at
municipal level."* And §1.2.2 is the sampling fact that matters more than either: *"The sample
design for CS 2016 was a stratified single-stage sample design. At enumeration area (EA)
level, all in-scope EAs were included in the sample and a sample of dwelling units was taken
within each EA (i.e. there was no subsampling of EAs)."* Every EA in the country is in the
sample, so no municipality is represented by a neighbour's households. The smallest unweighted
municipal sample is 529 people (Prince Albert) against a median of 7,890; the effective sample
sizes after weighting are printed by this script and are in sources/za.md §2.2.

**THE NINE PUBLISHED PROVINCIAL PROFILES ARE THE RECONCILIATION AND THIS SCRIPT WILL NOT
WRITE WITHOUT THEM.** Same survey, same 24 categories, tabulated independently by Stats SA
and parsed by different code. Every published province cell must reproduce from the microdata
to within TOLERANCE people. It does, on 215 of 216 cells; the exception is North West's
`Christian: Other`, which is a defect in Report 03-01-11 that the microdata settles —
see below.

TWO THINGS THE MICRODATA RESOLVED THAT THE PROVINCE BUILD COULD ONLY DESCRIBE:

  * **North West's table 2.10b.** Report 03-01-11's fourteen rows sum to 90.1% of its own
    printed total, leaving 336,482 people in no denomination row, and its `Other` cell reads
    `21 873` -- character for character the `Do not know` figure in that table's own footnote.
    The province build had two stories for that and neither closed, so it parked the 336,482
    on bare `christianity` rather than guessing. **The microdata puts North West's `Other` at
    358,355, which is 21,873 + 336,482 exactly**, so the row was mis-set and the second story
    (an excluded `Not applicable` universe) is wrong. Those people are Christians of another
    denomination and are now drawn as such.
  * **The residual.** `Christian: Denomination not reported` was the arithmetic gap between
    table 2.10a's Christianity cell and table 2.10b's rows, 567,039 people. It is now the
    survey's own answer: `Do not know` (227,585) plus `Unspecified` (2,976) on the
    denomination question, 230,561 people, measured per person. The other 336,478 were North
    West's defect.

AND THE HOLE IS NOW A COLUMN RATHER THAN A FOOTNOTE. 707,296 people answered `Do not know`
(704,358) or nothing at all (2,938) to the religion question. The province build could only
read those out of eight printed footnotes and infer Western Cape's by difference; here they
are two rows per municipality. They are EXCLUDED in taxonomy/za2016.py rather than dropped
before the file, so `tools/gap_share.py` computes the country's gap from the normalised file
instead of it being authored. §3.5 is why they are not drawn and not spread.

FOUR THINGS THIS READER HAS TO GET RIGHT:

  * **THE VINTAGE.** Every record carries geography twice, 2011 and 2016, and the two
    demarcations are different: `MN_CODE_2011` has 234 municipalities, `MN_CODE_2016` has 213
    after the August 2016 boundary reform. The 2016 set is used, because OCHA's COD-AB ADM3
    is the 213 and joins to it 213/213 on the MDB code with nothing left over on either side.
    The 2011 set is 10% finer in unit count and there is no boundary file on disk for it.
  * **THE LABEL SET IS NOT THE VARIABLE NAME.** Stata truncates label-set names to eight
    characters, so the file carries `MN_CODE` (234 labels, the 2011 demarcation) and
    `MN_COD_A` (213, the 2016 one) and neither is named after the variable it belongs to.
    Take the wrong one and 213 codes still resolve, to the wrong municipalities, and every
    national total still reconciles. The set is chosen by its size and then asserted against
    the label text, which differs: the 2016 set writes `WC011 : Matzikama` with a space before
    the colon and the 2011 set writes `WC011: Matzikama` without one.
  * **`Christianity` IS ASKED ONLY OF CHRISTIANS.** Code 88 `Not applicable` is 12,229,937
    people and is exactly the non-Christian population; it must not be emitted, or every
    non-Christian is counted twice. Asserted both ways.
  * **BOTH ANSWER SETS HAVE A ROW CALLED `Other`** and they mean different things. Every
    category is emitted prefixed `Religion: ` / `Christian: `, which is `sources/sz.py`'s
    convention; a bare join on the label would move 1,482,210 people of other faiths into
    Christianity. `countries.py` asserts the prefixes are still there.

THE CODEBOOK'S LABELS ARE FOLDED ONTO THE PUBLISHED ONES, not used raw. The microdata writes
`Buddism`, `Jehovahs Witness`, `Traditional african religion (e.g. ancestral; tribal; animis`
and `Just a christian/non-denominational`; the published tables and therefore
`taxonomy/za2016.py` write `Buddhism`, `Jehovah's Witness`, `Traditional African religion` and
`Just a Christian/non-denominational`. `za_profiles.CANON` is the single fold, shared with the
PDF parser, and every code is asserted to land on a canonical label -- so a re-release that
renames a category fails the build instead of silently emitting a category nothing maps.

Usage:
    python sources/za.py            rebuild from data/raw/za/cs-2016-person.dta
    python sources/za.py --fetch    same; the .dta is account-walled and cannot be fetched
"""

import csv
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import za_profiles as prof

RAW = os.path.join(ROOT, "data", "raw", "za")
DTA = os.path.join(RAW, "cs-2016-person.dta")
OUT = os.path.join(ROOT, "data", "normalized", "za.csv")

SOURCE_ID = "za_cs2016_micro"
YEAR = 2016
BASIS = "self_id"

EXPECTED_RECORDS = 3_328_867
EXPECTED_UNITS = 213
EXPECTED_POP = 55_653_654          # CS 2016's own weighted total, table 2.1

# Every published province cell must reproduce from the microdata to this many people. The
# published figures are weighted sums rounded by Stats SA and re-rounded here, so a cell can
# differ by a person or two and that is arithmetic, not a parse. Measured: the largest
# disagreement on the 215 sound cells is 2.
TOLERANCE = 2

# The one cell that does NOT reconcile, and by exactly how much. Report 03-01-11's `Other`
# row was mis-set to the `Do not know` figure from its own footnote; see the module docstring
# and sources/za.md §4.1. Asserted as an equality, so a reissued report fails the build.
NW_OTHER_DEFECT = ("North West", "Christian: Other", prof.NW_SHORTFALL)

# Rounding 213 x 27 float cells independently drifts each province total off the published
# figure by a few people. Measured: the worst province x category drift is 6. A wider drift
# means the weights or the label fold have changed, not the rounding.
MAX_ROUNDING_DRIFT = 12

RELIGION_VAR = "ReligionBelief"
CHRISTIAN_VAR = "Christianity"
# DC_MDB_C_2016 is carried only so it can reach the `note` column, where `sources/za_geo.py`
# reads it back as the join's independent evidence: the district each municipality belongs to
# according to Stats SA, checked against the district COD's polygon belongs to. A code join
# that had paired two municipalities the wrong way round would have to have paired them
# inside the same district to survive that.
COLS = ["PR_CODE_2016", "MN_CODE_2016", "DC_MDB_C_2016", RELIGION_VAR, CHRISTIAN_VAR,
        "pers_pstrwgt"]

# The two ReligionBelief codes that are not an answer. Emitted, EXCLUDED in the mapping, and
# read by tools/gap_share.py -- §3.5, and see the module docstring.
REL_NOT_ANSWERED = {12: "Do not know", 99: "Unspecified"}
# The two Christianity codes that are a Christian whose denomination was not established.
CHR_NOT_REPORTED = {15: "Do not know", 99: "Unspecified"}
CHR_NOT_APPLICABLE = 88
NOT_REPORTED = "Denomination not reported"

PROVINCE_CODES = {1: "Western Cape", 2: "Eastern Cape", 3: "Northern Cape", 4: "Free State",
                  5: "KwaZulu-Natal", 6: "North West", 7: "Gauteng", 8: "Mpumalanga",
                  9: "Limpopo"}

# The Stats SA municipality labels are not all usable as names, and both exceptions are named
# rather than repaired by a rule, because a rule loose enough to fix either is loose enough to
# rename a municipality that is merely spelt differently
# ([[reference_name_join_wrong_neighbour]]). `sources/za_geo.py` asserts that these two are
# the ONLY two of 213 whose Stats SA and COD names disagree.
#
#   LIM345 -- the microdata label is the literal word `New`. The municipality was created in
#     the August 2016 demarcation out of Thulamela and Makhado and had no name when the file
#     was coded; it is Collins Chabane. COD-AB names it, so COD's name is used.
#   NC067 -- the label reads `Kh+ói-Ma`, which is not an encoding artefact: those are the
#     bytes in the .dta. It is Khâi-Ma, in the Namakwa district, and COD spells it correctly.
NAME_OVERRIDE = {
    "LIM345": "Collins Chabane",
    "NC067": "Khâi-Ma",
}
# mdb code -> the label Stats SA actually prints, filled in as the file is read. The two
# overridden ones are recorded in the CSV's own `note` column so a reader of
# data/normalized/za.csv can see that a name was changed and by whom.
RAW_LABEL = {}


def _label_sets(reader):
    """Pick the 2016 municipality label set out of the file's own value labels.

    STATA TRUNCATES LABEL-SET NAMES TO EIGHT CHARACTERS, so this file carries `MN_CODE` and
    `MN_COD_A` and neither says which of MN_CODE_2011 / MN_CODE_2016 it belongs to. Picking
    the wrong one is silent: 213 codes still resolve, to different municipalities, and every
    national and provincial total still reconciles because the codes are a subset either way.
    So the set is picked by SIZE and then confirmed on the label TEXT, which differs -- the
    2016 set writes `WC011 : Matzikama` and the 2011 set writes `WC011: Matzikama`.
    """
    vls = reader.value_labels()
    cands = {k: v for k, v in vls.items() if k.startswith("MN_CO")}
    if len(cands) != 2:
        raise SystemExit(f"expected two MN_ label sets, found {sorted(cands)}")
    by_size = sorted(cands.items(), key=lambda kv: len(kv[1]))
    mn16, mn11 = by_size[0], by_size[1]
    if len(mn16[1]) != EXPECTED_UNITS or len(mn11[1]) != 234:
        raise SystemExit(f"municipality label sets are {len(mn16[1])} and {len(mn11[1])}, "
                         f"expected {EXPECTED_UNITS} (2016) and 234 (2011)")
    spaced = sum(1 for v in mn16[1].values() if " :" in v)
    tight = sum(1 for v in mn11[1].values() if " :" not in v)
    if spaced != len(mn16[1]) or tight != len(mn11[1]):
        raise SystemExit(
            f"the two municipality label sets can no longer be told apart on their text: "
            f"{mn16[0]} has {spaced}/{len(mn16[1])} written `CODE : Name` and {mn11[0]} has "
            f"{tight}/{len(mn11[1])} written `CODE: Name`. Do NOT guess -- the wrong set "
            "resolves silently to the wrong municipalities.")
    print(f"  municipality labels: {mn16[0]} ({len(mn16[1])}, 2016 demarcation), "
          f"{mn11[0]} ({len(mn11[1])}, 2011) not used")
    return mn16[1], vls[RELIGION_VAR.upper()[:8]], vls[CHRISTIAN_VAR.upper()[:8]]


def _canon(label, allowed):
    """Fold a codebook label onto the published canonical one, or refuse."""
    c = prof.CANON.get(prof.key(label))
    if c is None or c not in allowed:
        raise SystemExit(f"CS 2016 codebook label {label!r} folds to "
                         f"{prof.key(label)!r}, which is not one of the published "
                         f"categories. Stats SA has renamed a category; fix "
                         "sources/za_profiles.py's CANON, do not widen this check.")
    return c


def read_microdata():
    import pandas as pd

    if not os.path.exists(DTA):
        raise SystemExit(
            f"missing {DTA}.\nCS 2016 catalogue 611 is behind a free DataFirst account plus "
            "a signed confidentiality declaration and cannot be fetched from a script; see "
            "ask/answered/002-za.")

    with pd.io.stata.StataReader(DTA) as r:
        mn_lbl, rel_lbl, chr_lbl = _label_sets(r)

    # Canonicalise the two answer sets up front, so a renamed category fails before a single
    # record is read rather than after twenty minutes of aggregation.
    rel_canon, chr_canon = {}, {}
    for code, lab in rel_lbl.items():
        code = int(code)
        if code in REL_NOT_ANSWERED:
            continue
        rel_canon[code] = _canon(lab, set(prof.RELIGIONS))
    for code, lab in chr_lbl.items():
        code = int(code)
        if code in CHR_NOT_REPORTED or code == CHR_NOT_APPLICABLE:
            continue
        chr_canon[code] = _canon(lab, set(prof.DENOMINATIONS))
    if sorted(rel_canon.values()) != sorted(prof.RELIGIONS):
        raise SystemExit(f"ReligionBelief carries {sorted(rel_canon.values())}, expected the "
                         f"{len(prof.RELIGIONS)} published categories")
    if sorted(chr_canon.values()) != sorted(prof.DENOMINATIONS):
        raise SystemExit(f"Christianity carries {sorted(chr_canon.values())}, expected the "
                         f"{len(prof.DENOMINATIONS)} published denominations")
    print(f"  {len(rel_canon)} religion categories and {len(chr_canon)} denominations fold "
          "onto the published labels")

    parts, n = [], 0
    for chunk in pd.read_stata(DTA, columns=COLS, convert_categoricals=False,
                               chunksize=400_000):
        n += len(chunk)
        chunk["w2"] = chunk["pers_pstrwgt"] ** 2
        parts.append(chunk.groupby(
            ["PR_CODE_2016", "DC_MDB_C_2016", "MN_CODE_2016", RELIGION_VAR, CHRISTIAN_VAR],
            dropna=False)[["pers_pstrwgt", "w2"]].agg(["sum", "size"]))
        print(f"    read {n:,}")
    if n != EXPECTED_RECORDS:
        raise SystemExit(f"{n:,} person records, expected {EXPECTED_RECORDS:,} -- this is "
                         "not DataFirst catalogue 611's person file")

    agg = pd.concat(parts).groupby(level=[0, 1, 2, 3, 4]).sum().reset_index()
    agg.columns = ["pr", "dc", "mn", "rel", "chr", "w", "n", "w2", "_n2"]
    agg = agg[["pr", "dc", "mn", "rel", "chr", "w", "w2", "n"]]
    return agg, mn_lbl, rel_canon, chr_canon


def main():
    import numpy as np
    import pandas as pd

    print("South Africa - Community Survey 2016 person microdata, DataFirst catalogue 611")
    print(f"  {DTA} ({os.path.getsize(DTA):,} bytes)\n")

    agg, mn_lbl, rel_canon, chr_canon = read_microdata()
    agg["prov"] = agg["pr"].map(PROVINCE_CODES)
    if agg["prov"].isna().any():
        raise SystemExit(f"unknown province codes {sorted(set(agg.loc[agg['prov'].isna(), 'pr']))}")
    agg["mdb"] = agg["mn"].map(lambda k: mn_lbl[k].split(":")[0].strip())
    agg["mname"] = agg["mn"].map(lambda k: mn_lbl[k].split(":", 1)[1].strip())
    RAW_LABEL.update(dict(zip(agg["mdb"], agg["mname"])))
    missing = sorted(set(NAME_OVERRIDE) - set(RAW_LABEL))
    if missing:
        raise SystemExit(f"NAME_OVERRIDE names municipalities that are not in the file: "
                         f"{missing} -- the demarcation or the codes have changed")
    agg["mname"] = [NAME_OVERRIDE.get(c, nm) for c, nm in zip(agg["mdb"], agg["mname"])]

    total = float(agg["w"].sum())
    print(f"\n  {int(agg['n'].sum()):,} records, {total:,.0f} weighted people, "
          f"{agg['mdb'].nunique()} municipalities")
    if agg["mdb"].nunique() != EXPECTED_UNITS:
        raise SystemExit(f"{agg['mdb'].nunique()} municipalities, expected {EXPECTED_UNITS}")
    if abs(total - EXPECTED_POP) > 2:
        raise SystemExit(f"weighted total {total:,.0f}, expected CS 2016's published "
                         f"{EXPECTED_POP:,} (Report 03-01-07 table 2.1)")

    # A municipality must sit in exactly one province, or the geography is not nested and the
    # reconciliation below is comparing sums of different things.
    span = agg.groupby("mdb")[["prov", "dc"]].nunique()
    bad_span = span[(span["prov"] > 1) | (span["dc"] > 1)]
    if len(bad_span):
        raise SystemExit("municipalities in more than one province or district: "
                         f"{list(bad_span.index)}")
    district = dict(zip(agg["mdb"], agg["dc"]))

    # `Christianity` is asked only of Christians. Code 88 is `Not applicable` and must be
    # exactly the non-Christian population; emitting it would double-count 12.2M people.
    na = float(agg.loc[agg["chr"] == CHR_NOT_APPLICABLE, "w"].sum())
    non_christian = total - float(agg.loc[agg["rel"] == 1, "w"].sum())
    if abs(na - non_christian) > 2:
        raise SystemExit(f"Christianity=`Not applicable` is {na:,.0f} people but "
                         f"{non_christian:,.0f} are not Christian. The denomination question "
                         "is no longer nested inside the religion question and nothing here "
                         "is safe.")
    stray = agg[(agg["rel"] != 1) & (agg["chr"] != CHR_NOT_APPLICABLE)]
    if len(stray):
        raise SystemExit(f"{len(stray)} cells give a denomination to a non-Christian")
    print(f"  Christianity=Not applicable is {na:,.0f}, the non-Christian population exactly")

    # ---- long form: (municipality, category) -> weight ----
    def _slice(mask, cat):
        sel = agg[mask].groupby(["mdb", "mname", "prov"])[["w", "n"]].sum().reset_index()
        sel["cat"] = cat
        return sel

    rows = []
    for code, canon in rel_canon.items():
        if canon == "Christianity":
            continue                      # replaced by the fourteen denominations
        rows.append(_slice(agg["rel"] == code, f"Religion: {canon}"))
    for code, lab in REL_NOT_ANSWERED.items():
        rows.append(_slice(agg["rel"] == code, f"Religion: {lab}"))
    for code, canon in chr_canon.items():
        rows.append(_slice(agg["chr"] == code, f"Christian: {canon}"))
    rows.append(_slice(agg["chr"].isin(CHR_NOT_REPORTED), f"Christian: {NOT_REPORTED}"))

    long = pd.concat(rows, ignore_index=True)
    long = long.groupby(["mdb", "mname", "prov", "cat"], as_index=False)[["w", "n"]].sum()

    # ---- THE RECONCILIATION: every published province cell, from the PDFs ----
    print("\n  reconciling against the nine published provincial profiles "
          "(sources/za_profiles.py):")
    pub = prof.read_all(verbose=False)
    byprov = long.groupby(["prov", "cat"])["w"].sum()
    bad, worst = [], 0.0
    for pname, rec in pub.items():
        for canon, n in rec["a"].items():
            if canon == "Christianity":
                continue
            got = byprov.get((pname, f"Religion: {canon}"), 0.0)
            d = got - n
            worst = max(worst, abs(d))
            if abs(d) > TOLERANCE:
                bad.append((pname, f"Religion: {canon}", n, got))
        for canon, n in rec["b"].items():
            got = byprov.get((pname, f"Christian: {canon}"), 0.0)
            d = got - n
            if (pname, f"Christian: {canon}") == NW_OTHER_DEFECT[:2]:
                if round(d) != NW_OTHER_DEFECT[2]:
                    raise SystemExit(
                        f"North West's `Christian: Other` is {got:,.0f} in the microdata "
                        f"against {n:,} printed, a difference of {d:,.0f}. The build expects "
                        f"exactly {NW_OTHER_DEFECT[2]:,}, which is Report 03-01-11's own "
                        "shortfall. Either Stats SA has reissued the report or the "
                        "microdata has been revised; read sources/za.md §4.1.")
                continue
            worst = max(worst, abs(d))
            if abs(d) > TOLERANCE:
                bad.append((pname, f"Christian: {canon}", n, got))
    if bad:
        for pname, cat, n, got in bad[:25]:
            print(f"    {pname:15s} {cat:56s} published {n:>10,}  microdata {got:>12,.1f}")
        raise SystemExit(f"{len(bad)} published province cells do not reproduce from the "
                         "microdata. The build is comparing two different universes; do not "
                         "widen TOLERANCE to make this pass.")
    print(f"    216 province x category cells, 215 agree to within {worst:.1f} "
          f"people; the 216th is")
    print(f"    North West's `Christian: Other`, {NW_OTHER_DEFECT[2]:,} higher than Report "
          "03-01-11 prints,")
    print("    which is that report's own shortfall to the person -- the row was mis-set, "
          "and\n    the 336,482 are drawn as Christians of another denomination rather than "
          "as a residual.")

    # ---- rounding ----
    long["count"] = np.rint(long["w"]).astype(np.int64)
    drift = (long.groupby(["prov", "cat"])["count"].sum()
             - long.groupby(["prov", "cat"])["w"].sum()).abs().max()
    if drift > MAX_ROUNDING_DRIFT:
        raise SystemExit(f"rounding the municipal cells moves a province total by {drift:.0f} "
                         f"people, over the measured {MAX_ROUNDING_DRIFT}")
    print(f"\n  rounding 213 x {long['cat'].nunique()} cells moves the worst province total "
          f"by {drift:.1f} people")

    # ---- per-municipality sample sizes, for the note column and for sources/za.md ----
    per = agg.groupby("mdb").agg(w=("w", "sum"), w2=("w2", "sum"), n=("n", "sum"))
    per["neff"] = per["w"] ** 2 / per["w2"]
    print(f"  unweighted records per municipality: min {int(per['n'].min()):,} "
          f"median {int(per['n'].median()):,} max {int(per['n'].max()):,}")
    print(f"  Kish effective n per municipality:   min {per['neff'].min():,.0f} "
          f"median {per['neff'].median():,.0f} max {per['neff'].max():,.0f}")
    print(f"  people per municipality:             min {per['w'].min():,.0f} "
          f"median {per['w'].median():,.0f} max {per['w'].max():,.0f} "
          f"mean {per['w'].mean():,.0f}")

    # ---- how thin does it get, and where does that matter ----
    # THE COST OF DRAWING A SURVEY AT A FINE TIER, stated rather than assumed. The units are
    # sound (§1.2.2: every EA is in the sample), but a small category in a small municipality
    # can rest on a handful of records, and a share computed from four households carries a
    # weight of several thousand people. This is the number to look at before quoting any
    # municipal superlative, which is why `cell_n` goes into the file.
    drawn_cells = long[~long["cat"].isin(
        [f"Religion: {v}" for v in REL_NOT_ANSWERED.values()])]
    nz = drawn_cells[drawn_cells["w"] > 0]
    thin = nz[nz["n"] < 10]
    print(f"\n  {len(nz):,} non-empty (municipality, category) cells; "
          f"{len(thin):,} rest on fewer than 10 records "
          f"({100.0 * thin['w'].sum() / nz['w'].sum():.3f}% of the people drawn)")
    heavy = nz.sort_values("w", ascending=False)
    heavy = heavy[heavy["n"] <= 5].head(5)
    if len(heavy):
        print("  the heaviest cells resting on five records or fewer, which is where a "
              "municipal\n    share can be an artefact of one household's weight:")
        for r in heavy.itertuples(index=False):
            print(f"    {r.mname:26s} {r.cat:46s} {r.w:>9,.0f} people from "
                  f"{int(r.n)} record(s)")

    # ---- write ----
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    exemplars = {}
    for rec in pub.values():
        for canon, label in rec["exemplars"].items():
            import re as _re
            m = _re.search(r"\((?:e\.?g\.?|eg)[^)]*\)?", label, flags=_re.I)
            if m and canon not in exemplars:
                exemplars[canon] = "; " + " ".join(m.group(0).split())

    long = long.sort_values(["mdb", "cat"])
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_id", "geo_level", "geo_name", "source_category", "count",
                    "basis", "year", "source_id", "note"])
        for r in long.itertuples(index=False):
            canon = r.cat.split(": ", 1)[1]
            # `cell_n` is the number of unweighted CS 2016 person records behind THIS cell,
            # and it is in the file because this is a survey drawn at a fine tier. A share
            # computed from a municipality with 691 records is a real measurement and a wide
            # one, and nothing else in data/normalized/za.csv would let a reader tell a
            # 40-record cell from a 40,000-record one. `unit_n` is the whole municipality's.
            note = (f"level=municipality; province={r.prov}; district={district[r.mdb]}; "
                    f"CS 2016 person microdata (DataFirst 611), "
                    f"unit_n={int(per.loc[r.mdb, 'n']):,} records, cell_n={int(r.n):,}"
                    + exemplars.get(canon, ""))
            if r.mdb in NAME_OVERRIDE:
                note += (f"; Stats SA's own label for this municipality is "
                         f"'{RAW_LABEL[r.mdb]}', see sources/za.py NAME_OVERRIDE")
            w.writerow([r.mdb, "municipality", r.mname, r.cat, r.count, BASIS, YEAR,
                        SOURCE_ID, note])

    drawn = long.loc[~long["cat"].isin(
        [f"Religion: {v}" for v in REL_NOT_ANSWERED.values()]), "count"].sum()
    gap = long.loc[long["cat"].isin(
        [f"Religion: {v}" for v in REL_NOT_ANSWERED.values()]), "count"].sum()
    print(f"\nwrote {OUT}")
    print(f"  {len(long):,} rows, {long['mdb'].nunique()} municipalities, "
          f"{long['cat'].nunique()} categories")
    print(f"  {drawn:,} people drawn, {gap:,} not ({100.0 * gap / (drawn + gap):.3f}%, "
          "the two rows taxonomy/za2016.py excludes)")

    nat = long.groupby("cat")["count"].sum().sort_values(ascending=False)
    print(f"\n  national, {drawn:,} answers:")
    for cat, n in nat.items():
        if cat.split(": ", 1)[1] in REL_NOT_ANSWERED.values():
            print(f"    {cat:56s} {n:>11,}  {100.0 * n / (drawn + gap):5.2f}% of the "
                  "survey population, NOT DRAWN")
        else:
            print(f"    {cat:56s} {n:>11,}  {100.0 * n / drawn:5.2f}%")


if __name__ == "__main__":
    main()
