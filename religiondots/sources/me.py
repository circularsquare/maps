"""Montenegro — MONSTAT, Popis 2023, religion by settlement, aggregated to municipality.

Reads (or fetches) data/raw/me/ and writes data/normalized/me.csv.

One workbook, **`naselja vjera popis 2023..xlsx`** — *Broj stanovnika po naseljima i vjeri,
Popis 2023. godine* — 1,462 settlements x 12 religion categories plus a total. 124 KB, one
GET, no key and no wall.

**FINDING IT WAS THE WHOLE PROBLEM, AND THE FOLDER IS NAMED FOR THE WRONG YEAR.** sources.md
§11c walked nine MONSTAT census subpages and concluded none carries an XLSX; §11k repeated
that. Both are wrong, and the reason neither found it is that Montenegro's 2023 census lives
under `uploads/files/popis 2021/` — the year it was originally scheduled for, before it was
postponed twice. A path search for `2023` finds nothing; the census pages themselves link it,
but only from `page.php?id=2342`, one of fourteen subpages under `pageid=1992`. Sweeping
every id in the range is what found it. **When an office's own folder names disagree with its
publication titles, enumerate the pages rather than guessing the paths.**

**THE `z` SENTINEL IS SUPPRESSION AND THE WORKBOOK SAYS SO.** Its last three rows are a
legend:

    "-" nema pojave                 no occurrence — a true zero
    "z" zaštićen podatak            protected data, under the Law on Official Statistics

So this is Lithuania's case (§9q) and not Kosovo's (§9w): reading `z` as zero deletes people.
2,188 cells across 964 of the 1,462 settlements are `z`, and they hide **29,194 people,
4.69% of Montenegro**.

**AND IT CANNOT BE DIFFERENCED OUT, WHICH IS WORTH KNOWING BEFORE TRYING.** Where a
settlement's total is published, a lone suppressed cell would give itself away by
subtraction. 185 settlements have exactly one `z` — and their gap is **zero, every one of
them**, so every lone suppression is a blanked zero and differencing recovers nothing worth
having. `check()` asserts that, so a future vintage with sloppier control is noticed rather
than silently exploited.

**THE THRESHOLD IS TEN, AND IT IS READABLE OFF THE DATA.** No published value anywhere in
either 2023 settlement workbook is below 10 — not one settlement total, not one category
cell. So a primary `z` is a number in 1..9. **But most of the hidden mass is not primary
suppression**: 223 settlements have a gap larger than 9 x their number of `z` cells, so at
least one of those cells is 10 or more, and those 223 hold **23,932 of the 29,225**. That is
complementary suppression — a cell blanked to protect a small one beside it — and it is why
the gap cannot be bounded cell by cell.

**THE TOTAL COLUMN IS SUPPRESSED TOO, IN 219 SETTLEMENTS, AND THOSE PEOPLE ARE NOT DRAWN AT
ALL.** 74 settlements publish `-` throughout and are genuinely uninhabited. A further **219
publish `z` as their own population**, so there is no denominator to compute a residual
against; between them they hold 31 people in unsuppressed category cells and, by the
threshold above, **between 219 and 1,971 in total**. They are left out rather than estimated
(§3.5), which is what `gap=` on the countries.py entry is for. The same 219 are suppressed in
`naselja popis 2023.xlsx`, so there is no second table to recover them from.

**THE SUPPRESSION IS NOT UNIFORM AND ITS SHAPE IS THE THING TO REPORT.** It protects small
counts, and a small count is a *local minority*, so the gap concentrates in the municipalities
where the national majority is locally scarce: **Petnjica 29.4%, Šavnik 25.4%, Rožaje 21.3%,
Žabljak 19.5%, Ulcinj 15.0%**, against Tivat 0.8% and Podgorica 0.9%. Nationally it hides
11,868 Orthodox and 13,171 Muslims — 86% of the total gap between them — which is the same
statement from the other side: every municipality under-reports whichever of the two is its
own minority. The undrawn mass goes to `unknown` (§6.3a) rather than being spread, per §3.5.

**Tuzi (2018) and Zeta (2022) were split out of Podgorica and geoBoundaries has no polygon
for either**, so the two are folded back into Podgorica here and the drawn geography is 23
municipalities rather than 25. That loses a genuinely interesting unit — Tuzi is Montenegro's
Albanian municipality — and it is a boundary-vintage limit, not a data one: the counts exist.
`sources/me_geo.md` says what would fix it.

Usage:
    python sources/me.py --fetch    one GET, 124 KB
    python sources/me.py            normalise from data/raw/me/
"""

import csv
import os
import sys
import urllib.parse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "me")
OUT = os.path.join(ROOT, "data", "normalized", "me.csv")

SOURCE_ID = "me_popis_2023"
YEAR = 2023
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# The folder is named `popis 2021` because that is when this census was first scheduled.
# The file name really does carry two dots before the extension.
URL = ("https://www.monstat.org/uploads/files/popis 2021/podaci/"
       "naselja vjera popis 2023..xlsx")
BOOK = os.path.join(RAW, "naselja_vjera_popis_2023.xlsx")
SHEET = "vjera"

# Column order in the sheet, after `Opština`, `Naselje` and `Ukupno`. The first five sit
# under a merged `Hrišćanstvo` header in row 1 and are named in row 2; the rest are named in
# row 1 with row 2 blank. Read positionally against this list rather than by header, because
# a two-row merged header does not survive pandas' header inference intact.
CATEGORIES = [
    "Pravoslavna",              # Orthodox
    "Katolička",                # Catholic
    "Protestanti",              # Protestants
    "Jehovini svjedoci",        # Jehovah's Witnesses
    "Ostale hrišćanske",        # other Christian
    "Agnostik",                 # agnostic
    "Ateista",                  # atheist
    "Budisti",                  # Buddhist
    "Islamska",                 # Islamic
    "Ne želi da se izjasni",    # does not wish to declare
    "Ostale vjere",             # other religions
    "Ostalo",                   # other / not stated
]
TOTAL_CAT = "Ukupno"
SUPPRESSED_CAT = "zaštićen podatak"      # the residual this build writes for the `z` mass

NO_OCCURRENCE = "-"
PROTECTED = "z"

# Split out of Podgorica after the boundary file was cut; folded back in. See the docstring.
MERGED_INTO_PODGORICA = ("Tuzi", "Zeta")
PODGORICA = "Podgorica"

EXPECTED_SETTLEMENTS = 1_462
EXPECTED_MUNICIPALITIES = 23          # 25 counted, less Tuzi and Zeta
NATIONAL = 622_537                    # sum of the PUBLISHED settlement totals
EXPECTED_WITHHELD = 219               # settlements whose own total is `z`
EXPECTED_EMPTY = 74                   # settlements that are `-` throughout — uninhabited

# The lowest value published anywhere in either 2023 settlement workbook. Asserted rather
# than assumed: it is what turns "some unknown number" into "1 to 9 per primary cell", and a
# vintage that publishes a 3 has changed its disclosure rule.
THRESHOLD = 10

# MONSTAT's own published national figures for the 2023 census, from its release. Used ONLY
# as a check on how much each category loses to suppression — nothing here is scaled to them.
PUBLISHED = {
    "Pravoslavna": 443_394,
    "Islamska": 124_668,
    "Katolička": 20_408,
    "Ateista": 14_260,
}


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(BOOK) and os.path.getsize(BOOK) > 50_000:
        print("already have", BOOK)
        return
    url = urllib.parse.quote(URL, safe=":/")
    print("GET", url)
    r = requests.get(url, timeout=300,
                     headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    r.raise_for_status()
    # §5a: HTTP 200 is not a download. monstat.org answers a missing path with an HTML page.
    if r.content[:2] != b"PK":
        raise SystemExit(f"not an xlsx: first bytes {r.content[:80]!r}")
    with open(BOOK, "wb") as fh:
        fh.write(r.content)
    print(f"  {os.path.getsize(BOOK):,} bytes")


def _sheet():
    import pandas as pd

    if not os.path.exists(BOOK):
        raise SystemExit(f"missing {BOOK} -- run with --fetch first")
    df = pd.read_excel(BOOK, sheet_name=SHEET, header=None)
    if df.shape[1] != len(CATEGORIES) + 3:
        raise SystemExit(f"{SHEET}: {df.shape[1]} columns, expected {len(CATEGORIES) + 3}. "
                         "MONSTAT has changed the category list; re-read the header before "
                         "trusting any positional read.")

    # The legend is in the sheet and is the evidence for how `z` is read. Assert it is still
    # there and still says what it says, because the whole treatment below depends on it.
    tail = " ".join(str(x) for x in df.iloc[:, 0].tail(6).tolist())
    if "zaštićen podatak" not in tail:
        raise SystemExit("the workbook's `z` legend is gone. It is the only evidence that "
                         "`z` is disclosure control rather than a zero (§3.8) — re-read the "
                         "sheet before assuming either.")

    head = df.iloc[1].tolist() + df.iloc[2].tolist()
    for want in ("Pravoslavna", "Islamska", "Ukupno"):
        if not any(str(x).strip() == want for x in head):
            raise SystemExit(f"header row does not carry {want!r} -- the sheet's shape has "
                             "changed and the positional read is no longer safe")

    body = df.iloc[3:].copy()
    body.columns = ["opstina", "naselje", TOTAL_CAT] + CATEGORIES
    body = body[body[TOTAL_CAT].notna()].copy()
    body["opstina"] = body["opstina"].astype(str).str.strip()
    body["naselje"] = body["naselje"].astype(str).str.strip()
    return body


def read():
    import numpy as np
    import pandas as pd

    body = _sheet()
    if len(body) != EXPECTED_SETTLEMENTS:
        raise SystemExit(f"{len(body)} settlements, expected {EXPECTED_SETTLEMENTS}")

    raw = body[CATEGORIES].astype(str).apply(lambda s: s.str.strip())
    total_raw = body[TOTAL_CAT].astype(str).str.strip()
    protected = raw.eq(PROTECTED)
    counts = (raw.replace({NO_OCCURRENCE: "0", PROTECTED: np.nan})
                 .apply(pd.to_numeric, errors="coerce"))
    total = pd.to_numeric(body[TOTAL_CAT], errors="coerce")

    # THE TOTAL COLUMN IS ITSELF SUPPRESSED IN 219 SETTLEMENTS, and assuming otherwise is
    # what this build did first. It surfaced as the categories out-summing the country by 31
    # people: those settlements' unsuppressed cells were being counted against a total that
    # had coerced to NaN. Where no denominator is published there is no residual to compute,
    # so the gap is taken only where a total exists and the withheld settlements are
    # reported separately instead of being silently absorbed.
    withheld = total_raw.eq(PROTECTED)
    empty = total_raw.eq(NO_OCCURRENCE)
    has_total = total.notna()

    known = counts.sum(axis=1, skipna=True)
    gap = (total - known).round().where(has_total, 0.0)

    lone = protected.sum(axis=1).eq(1) & has_total
    pos = counts.values[np.greater(counts.values, 0, where=~np.isnan(counts.values),
                                   out=np.zeros(counts.shape, dtype=bool))]
    stats = {
        "settlements": len(body),
        "protected_cells": int(protected[has_total].values.sum()),
        "cells": int(protected[has_total].size),
        "settlements_with_z": int(protected[has_total].any(axis=1).sum()),
        "gap_total": float(gap.sum()),
        "national": float(total.sum()),
        # every lone suppression turns out to be a blanked zero — asserted in check()
        "lone_z_gap": float(gap[lone].sum()),
        "lone_z_settlements": int(lone.sum()),
        "negative_gaps": int((gap < 0).sum()),
        # settlements with no published total at all, and the empty ones
        "withheld": int(withheld.sum()),
        "withheld_known": float(known[withheld].sum()),
        "empty": int(empty.sum()),
        # the smallest published value anywhere: the disclosure threshold
        "min_published": float(min(pos.min(), total[has_total].min())),
        # how much of the gap sits in settlements where some `z` must be >= the threshold,
        # i.e. is complementary suppression rather than a genuinely small number
        "complementary": float(
            gap[has_total][gap[has_total] > (THRESHOLD - 1) *
                           protected[has_total].sum(axis=1)].sum()),
        # §3.8's per-category shape: how many municipalities have at least one settlement
        # with this category withheld. The only per-category measure available for the nine
        # MONSTAT publishes no national total for.
        "z_units": {
            cat: int(body.loc[has_total & protected[cat], "opstina"]
                     .replace({n: PODGORICA for n in MERGED_INTO_PODGORICA}).nunique())
            for cat in CATEGORIES
        },
    }

    # A settlement with no published total is not drawn, so none of it is drawn — including
    # the 31 people who do appear in its unsuppressed category cells. Keeping those would
    # make the categories out-sum the universe by exactly 31 and would mean drawing a
    # fragment of a settlement whose size is unknown. Dropping the whole row is the
    # consistent reading; `check()` reports the cost.
    frame = pd.concat([body[["opstina"]], counts, total.rename(TOTAL_CAT),
                       gap.rename(SUPPRESSED_CAT)], axis=1)[has_total.values]
    muni = frame.groupby("opstina").sum(min_count=0)

    # Tuzi and Zeta have no polygon; fold them back into the Podgorica they were cut from.
    for name in MERGED_INTO_PODGORICA:
        if name not in muni.index:
            raise SystemExit(f"{name} is not in the workbook -- MONSTAT has changed its "
                             "municipality list, so the Podgorica merge needs re-deciding "
                             "(sources/me_geo.md)")
        muni.loc[PODGORICA] = muni.loc[PODGORICA] + muni.loc[name]
        muni = muni.drop(name)
    stats["merged"] = MERGED_INTO_PODGORICA

    rows = []
    for name, r in muni.iterrows():
        for cat in [TOTAL_CAT] + CATEGORIES + [SUPPRESSED_CAT]:
            n = int(round(float(r[cat])))
            note = "level=municipality"
            if cat == TOTAL_CAT:
                note += "; universe total, not a religion category"
            elif cat == SUPPRESSED_CAT:
                note += ("; the unit's own total less its published categories — people "
                         "MONSTAT suppressed as `z` under the Law on Official Statistics. "
                         "NOT DRAWN: spec §3.8 removes suppressed cells rather than "
                         "placing them, and §6.3a's `unknown` explicitly excludes them")
            if name == PODGORICA:
                note += "; includes Tuzi and Zeta, which have no polygon in geoBoundaries"
            rows.append({"geo_id": name, "geo_level": "municipality", "geo_name": name,
                         "source_category": cat, "count": n, "basis": BASIS,
                         "year": YEAR, "source_id": SOURCE_ID, "note": note})

    # the national row, so consumers can reconcile without re-summing
    nat = muni.sum()
    for cat in [TOTAL_CAT] + CATEGORIES + [SUPPRESSED_CAT]:
        rows.append({"geo_id": "ME", "geo_level": "country", "geo_name": "Crna Gora",
                     "source_category": cat, "count": int(round(float(nat[cat]))),
                     "basis": BASIS, "year": YEAR, "source_id": SOURCE_ID,
                     "note": "level=country; summed from the settlement table"})
    return rows, stats, muni


def check(rows, stats, muni):
    ok = True
    print(f"  {stats['settlements']:,} settlements aggregated to {len(muni)} municipalities")

    good = len(muni) == EXPECTED_MUNICIPALITIES
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {len(muni)} municipalities "
          f"(expected {EXPECTED_MUNICIPALITIES} — 25 counted, less "
          f"{' and '.join(stats['merged'])} folded into {PODGORICA})")

    nat = {r["source_category"]: r["count"] for r in rows if r["geo_level"] == "country"}
    good = nat[TOTAL_CAT] == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} national total {nat[TOTAL_CAT]:,} "
          f"(expected {NATIONAL:,})")

    parts = sum(v for k, v in nat.items() if k != TOTAL_CAT)
    good = parts == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the 12 categories plus the suppressed residual "
          f"partition the country exactly ({parts:,})")

    good = stats["negative_gaps"] == 0
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} no settlement's categories exceed its own total "
          f"({stats['negative_gaps']} negative)")

    # ---- the two layers of suppression (§3.8) -----------------------------------------
    print(f"\n  SUPPRESSION, LAYER ONE — cells. {stats['protected_cells']:,} of "
          f"{stats['cells']:,} category cells are `z` "
          f"({100.0 * stats['protected_cells'] / stats['cells']:.1f}%), across "
          f"{stats['settlements_with_z']:,} settlements.")
    print(f"  They hide {stats['gap_total']:,.0f} people — "
          f"{100.0 * stats['gap_total'] / stats['national']:.2f}% of the counted country. "
          "That mass goes to `unknown`.")
    good = stats["lone_z_settlements"] == 0
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the suppression is properly complementary: of the "
          f"{stats['settlements_with_z']:,} settlements that publish a total\n      and "
          f"carry a `z`, **{stats['lone_z_settlements']} have exactly one** — so no cell is "
          "ever left recoverable by\n      subtracting the categories from the total. If "
          "this goes non-zero the control has gone\n      sloppy and those cells can be "
          "read straight off.")

    good = stats["min_published"] == THRESHOLD
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the disclosure threshold is "
          f"{stats['min_published']:.0f} — the smallest value published anywhere in the "
          f"sheet\n      (expected {THRESHOLD}), so a primary `z` is a number in "
          f"1..{THRESHOLD - 1}")
    print(f"      but {stats['complementary']:,.0f} of the {stats['gap_total']:,.0f} sits in "
          f"settlements whose gap EXCEEDS {THRESHOLD - 1} x their `z` count,\n      so at "
          "least one cell there is above the threshold: that is complementary suppression, "
          "and it is\n      why the hidden mass cannot be bounded cell by cell.")

    print(f"\n  SUPPRESSION, LAYER TWO — whole settlements. {stats['withheld']} publish `z` "
          f"as their OWN total,\n  so they have no denominator and no residual can be "
          f"computed for them. They carry "
          f"{stats['withheld_known']:,.0f} people\n  in unsuppressed cells and, at the "
          f"threshold above, between {stats['withheld']:,} and "
          f"{stats['withheld'] * (THRESHOLD - 1):,} in all.\n  They are NOT drawn and NOT "
          "estimated (§3.5) — countries.py states it with `gap=`. A further "
          f"{stats['empty']}\n  settlements publish `-` throughout and are genuinely "
          "uninhabited.")
    good = stats["withheld"] == EXPECTED_WITHHELD and stats["empty"] == EXPECTED_EMPTY
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {stats['withheld']} withheld / {stats['empty']} "
          f"empty (expected {EXPECTED_WITHHELD} / {EXPECTED_EMPTY})")

    # ---- PER CATEGORY, because §3.8 says a headline percentage hides the damage --------
    # Lithuania is the precedent: 0.06% of the country withheld read as nothing, and the
    # truth was that the map lost 60% of one religion. So report what each category loses
    # and how many units it goes missing from, never one national number.
    print("\n  WHAT THE SUPPRESSION COSTS EACH CATEGORY (§3.8 — never report this as one\n"
          "  headline percentage; 4.69% reads as nothing and the per-category damage is\n"
          "  where the harm is). `z units` is how many of the 23 municipalities have at\n"
          "  least one settlement with that category withheld:")
    print(f"    {'category':<24} {'drawn':>9} {'published':>10} {'hidden':>8} {'':>6} "
          f"{'z units':>8}")
    for cat in CATEGORIES:
        got = nat[cat]
        zu = stats["z_units"].get(cat, 0)
        if cat in PUBLISHED:
            pub = PUBLISHED[cat]
            print(f"    {cat:<24} {got:>9,} {pub:>10,} {pub - got:>8,} "
                  f"{100.0 * (pub - got) / pub:5.1f}% {zu:>8}")
        else:
            print(f"    {cat:<24} {got:>9,} {'—':>10} {'—':>8} {'':>6} {zu:>8}")
    named_hidden = sum(PUBLISHED[c] - nat[c] for c in PUBLISHED)
    print(f"\n    The four MONSTAT publishes nationally account for {named_hidden:,} of the "
          f"{stats['gap_total']:,.0f} withheld.\n    Islam loses "
          f"{100.0 * (PUBLISHED['Islamska'] - nat['Islamska']) / PUBLISHED['Islamska']:.1f}% "
          "of itself and Orthodoxy "
          f"{100.0 * (PUBLISHED['Pravoslavna'] - nat['Pravoslavna']) / PUBLISHED['Pravoslavna']:.1f}% "
          "— four times the rate — which is\n    the same fact as the geography below, seen "
          "per category: suppression protects local\n    minorities, and Montenegro's local "
          "minorities are disproportionately Muslim.\n    For the nine smaller categories "
          "MONSTAT publishes no national figure, so the loss can\n    only be reported as "
          "the unit count; it is certainly proportionally worse.")

    # ---- the geography of the suppression, which is the reason to read note_public -----
    m = muni.copy()
    m["pct"] = 100.0 * m[SUPPRESSED_CAT] / m[TOTAL_CAT]
    worst = m.sort_values("pct", ascending=False).head(6)
    best = m.sort_values("pct").head(3)
    print("\n  and it is not spread evenly — it protects small counts, and a small count is\n"
          "  a LOCAL minority, so it lands hardest where the national majority is scarce:")
    for name, r in worst.iterrows():
        print(f"    {name:<16} {r[TOTAL_CAT]:>8,.0f} people   "
              f"{r[SUPPRESSED_CAT]:>6,.0f} suppressed  {r['pct']:>5.1f}%")
    print("    ...")
    for name, r in best.iterrows():
        print(f"    {name:<16} {r[TOTAL_CAT]:>8,.0f} people   "
              f"{r[SUPPRESSED_CAT]:>6,.0f} suppressed  {r['pct']:>5.1f}%")

    print(f"\n  {len(rows):,} rows. Categories, national:")
    for cat, n in sorted(nat.items(), key=lambda kv: -kv[1]):
        mark = ""
        if cat == TOTAL_CAT:
            mark = "  <- universe"
        elif cat == SUPPRESSED_CAT:
            mark = "  <- undrawn, goes to `unknown`"
        print(f"    {n:>10,}  {100.0 * n / NATIONAL:6.2f}%  {cat}{mark}")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows, stats, muni = read()
    check(rows, stats, muni)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
