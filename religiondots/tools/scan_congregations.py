"""What one congregation is worth, and whether the taxonomy can say — §4.4's ratio question.

155 of the US Religion Census's 373 bodies report congregations and no membership, holding
26,999 congregations. spec.md §4.3 says they draw nothing until §4.4 converts them, and §4.4
says the conversion "needs a defensible per-family ratio rather than one national average" and
that "the ratio's spread within a branch is the thing to check before trusting it".

This checks it. Run it with no arguments; it reads the raw ASARB workbook and the hand-mapped
`taxonomy/usrc_groups.csv` and prints five things:

  1. the spread of adherents per congregation across all counted bodies
  2. whether that ratio tracks body size (it does)
  3. the spread within each branch, which is what a per-family ratio would be read off
  4. every counted body outside Christianity, which is the comparator set for the bodies that
     have no Christian branch to borrow from
  5. what each congregations-only body would be estimated at, and from what

WHAT IT FOUND, 2026-09-03, so the next reader does not have to re-derive it:

  * The ratio runs 18 to 3,188 with a median of 141. One national average is indefensible, as
    §4.4 already said.
  * The spread is not noise, it is structure, and the largest cut is not a Christian family
    boundary — it is Christianity itself. Christian bodies run a median of 134 per
    congregation. Non-Christian bodies run 541, and the largest are Hindu temples at 1,920 and
    mosques at 1,607. A congregation is not one unit of measurement: a Baptist church of 130
    and a temple serving a metropolitan area's Hindus are both "one congregation".
  * So a fallback that borrows across that boundary is not a weak estimate, it is a wrong one,
    and it is wrong in a fixed direction. The American Sikh Council's 307 gurdwaras come to
    43,000 people on the global median, 166,000 on the non-Christian median, and 590,000 at the
    Hindu-temple ratio. Outside estimates of the US Sikh population are in the last range.
  * Stopping at the root still carries almost all of it: 144 of the 155 bodies, 25,777 of the
    26,999 congregations and about 6.2M people find three counted bodies inside their own
    religion. The largest single item, the United Pentecostal Church International at 4,549
    congregations, sits next to three counted Oneness Pentecostal bodies. What is left is 11
    bodies and 1,222 congregations, and every one of them is a religion the US Religion Census
    counted nobody in — Sikh, Jain, Zoroastrian, Shinto, Daoist and six more.
  * Two cautions that the branch median cannot see. Congregation size rises with body size
    (median ratio 95 in the smallest decile, 272 in the largest), and a branch can hold two
    kinds of thing: Chabad's 952 storefront centres would take Judaism's 789-per-synagogue and
    become 751,000 people, which is not what a Chabad house is.
"""
import pandas as pd

from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
NEED = 3                    # counted bodies a branch must hold before its median is usable


def load():
    paths = pd.read_csv(HERE / "taxonomy" / "usrc_groups.csv", dtype={"Group Code": str})
    det = pd.read_excel(HERE / "data" / "raw" / "2020_USRC_Group_Detail.xlsx",
                        sheet_name="2020 Group by County",
                        dtype={"FIPS": str, "Group Code": str})
    det["code"] = det["Group Code"].str.strip().str.zfill(3)
    paths["code"] = paths["Group Code"].str.strip().str.zfill(3)
    det = det.merge(paths[["code", "path"]], on="code", how="left")
    det = det[det.path.notna() & (det.path != "UNMAPPED")]
    nat = det.groupby(["code", "Group Name", "path"]).agg(
        adh=("Adherents", "sum"), cong=("Congregations", "sum"),
        counties=("FIPS", "nunique")).reset_index()
    counted = nat[(nat.adh > 0) & (nat.cong > 0)].copy()
    counted["ratio"] = counted.adh / counted.cong
    return counted, nat[~nat.code.isin(counted.code)].copy()


def comparator(counted, path, cross_root=False):
    """Deepest branch holding NEED counted bodies -> (branch, n, median ratio).

    The search runs up to and including the L1 root and then stops. Widening past the root is
    the finding above: a body whose own religion has no counted member does not get
    Christianity's congregation size, because that is a different unit of measurement, not a
    coarser one. `cross_root=True` allows it anyway, for seeing what it would have said.
    """
    parts = path.split(".")
    for i in range(len(parts), 0, -1):
        pre = ".".join(parts[:i])
        sib = counted[counted.path.str.startswith(pre + ".") | (counted.path == pre)]
        if len(sib) >= NEED:
            return pre, len(sib), sib.ratio.median()
    if cross_root:
        return "(all)", len(counted), counted.ratio.median()
    return None, 0, float("nan")


def main():
    counted, unc = load()
    counted["l1"] = counted.path.str.split(".").str[0]
    unc["l1"] = unc.path.str.split(".").str[0]
    money = lambda v: f"{v:12,.0f}"                                        # noqa: E731

    print(f"{len(counted)} counted bodies, {len(unc)} congregations-only "
          f"({int(unc.cong.sum()):,} congregations)\n")

    print("1. ADHERENTS PER CONGREGATION, all counted bodies")
    print(counted.ratio.describe(percentiles=[.05, .25, .5, .75, .95])
          .to_string(float_format=lambda v: f"{v:10.1f}"))

    print("\n2. DOES IT TRACK BODY SIZE - counted bodies by congregation decile")
    counted["dec"] = pd.qcut(counted.cong, 10, labels=False, duplicates="drop")
    print(counted.groupby("dec").agg(n=("ratio", "size"), cong_lo=("cong", "min"),
                                     cong_hi=("cong", "max"), median=("ratio", "median"))
          .to_string(float_format=lambda v: f"{v:10.0f}"))

    print("\n3. SPREAD WITHIN A BRANCH - L2 branches holding at least 3 counted bodies")
    counted["l2"] = counted.path.str.split(".").str[:2].str.join(".")
    g = counted.groupby("l2").agg(n=("ratio", "size"), lo=("ratio", "min"),
                                  p25=("ratio", lambda s: s.quantile(.25)),
                                  median=("ratio", "median"),
                                  p75=("ratio", lambda s: s.quantile(.75)),
                                  hi=("ratio", "max"))
    g["hi/lo"] = g.hi / g.lo
    print(g[g.n >= NEED].sort_values("n", ascending=False)
          .to_string(float_format=lambda v: f"{v:9.1f}"))

    print("\n4. EVERY COUNTED BODY OUTSIDE CHRISTIANITY - the whole comparator set for a")
    print("   body whose own religion has to answer for it")
    nc = counted[counted.l1 != "christianity"].sort_values("ratio", ascending=False)
    print(nc[["Group Name", "path", "cong", "adh", "ratio"]]
          .to_string(index=False, float_format=money))
    print(f"   median {nc.ratio.median():,.0f} | pooled {nc.adh.sum() / nc.cong.sum():,.0f} | "
          f"Christian median {counted[counted.l1 == 'christianity'].ratio.median():,.0f}")

    print("\n5. WHAT EACH CONGREGATIONS-ONLY BODY WOULD BE ESTIMATED AT")
    rows = []
    for r in unc.sort_values("cong", ascending=False).itertuples():
        br, n, med = comparator(counted, r.path, cross_root=False)
        rows.append(dict(name=r._2[:36], path=r.path, cong=int(r.cong), branch=br or "-",
                         n=n, ratio=med, people=r.cong * med))
    out = pd.DataFrame(rows)
    ok, none = out[out.n > 0], out[out.n == 0]
    print(f"\n   {len(ok)} bodies have a comparator inside their own religion "
          f"({int(ok.cong.sum()):,} congregations, {ok.people.sum():,.0f} people):")
    print(ok.head(25).to_string(index=False, float_format=money))
    if len(ok) > 25:
        print(f"   ... and {len(ok) - 25} smaller")
    print(f"\n   {len(none)} have none, and these are the ones that need a decision "
          f"({int(none.cong.sum()):,} congregations):")
    print(none[["name", "path", "cong"]].to_string(index=False))


if __name__ == "__main__":
    main()
