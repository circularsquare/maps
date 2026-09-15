# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _vc_counts():
    """SVG Statistical Office 2012 census at enumeration district: 16 nodes on 221 units.

    ONE level, no allocation, nothing modelled — every drawn row is `measured` and may ring.
    vc.csv also carries the country and 13 census divisions, which are the same people twice
    more; only `ed` is read.

    **THE FINEST GEOGRAPHY IN THE PROJECT PER HEAD** — 221 units over 109,188 people, a
    median of 415 people and 0.66 km2 each — and the smallest country on the map.

    **THE RECONCILIATION IS CROSS-TABLE AND IS STRONGER THAN A WITHIN-TABLE ONE.** This file
    publishes no religion total; the 18 religion cells are checked against `ETH_TPOP`, the
    separately tabulated ethnicity universe, and they agree TO THE PERSON on all 235 rows at
    every level (sources/vc.py). Two independent questions, one answer.

    **RINGS DO MORE WORK HERE THAN ANYWHERE ELSE.** Six categories are under 400 people —
    Presbyterian 294, Salvation Army 287, Mormon 207, Muslim 111, Hindu 89, Traditional 74 —
    and at 1:1,000 none of them draws a dot. Section 4.3's presence marks are what put them on
    the map at all, and this country is the argument for having built them.

    95.3% of the census is drawn; the 4.67% not stated is spec 3.5.
    """
    from vc2012 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "vc.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "ed"].copy()
    if df["geo_id"].nunique() != 221:
        raise SystemExit(f"{df['geo_id'].nunique()} enumeration districts, expected 221 -- "
                         "re-run sources/vc.py")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]) - {"Not stated"})
    if unmapped:
        raise SystemExit(f"vc.csv has unmapped source categories: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "measured"
    # Two categories share `other.vc`, so collapse before returning.
    df = (df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum())
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "vc": dict(
        name="Saint Vincent and the Grenadines",
        source="Census 2012 (Saint Vincent and the Grenadines Statistical Office)",
        basis="self-identification",
        view=[-61.60, 12.50, -61.05, 13.42],
        gap="4.7% for whom no religion was recorded",
        gap_share=0.04666,
        note_public=(
            "**The most finely counted country on this map.** 221 enumeration districts for "
            "109,188 people — a median of **415 people per unit**, over an area smaller than "
            "the Isle of Wight — with **18 religions named**. Nothing else here counts this "
            "few people at a time, and it is why religions of a hundred people are visible "
            "at all. "
            "**Six categories are under 400 people**: Presbyterian 294, Salvation Army 287, "
            "Mormon 207, Muslim 111, Hindu 89, Traditional 74. At one dot per thousand "
            "people none of them draws a dot, so they appear as presence rings — the mark "
            "that says *this religion is here* without claiming how many. "
            "**Saint Vincent is 27.6% Pentecostal**, the highest share of any country drawn "
            "here, with Anglicans at 13.9% and Adventists at 11.6%. The Anglican and "
            "Methodist inheritance is the British colonial church; the Pentecostal majority "
            "arrived in the twentieth century and overtook it. "
            "**Only 7.5% report no religion — against 21.4% in Jamaica**, 160 km away and "
            "drawn from the same series. That is the sharpest irreligion contrast between "
            "neighbours anywhere on this map. "
            "**Rastafari is 1.08% here and 1.08% in Jamaica** — two independently designed "
            "censuses arriving at the same share — but it is spread quite differently: "
            "present in **186 of the 221 districts**, where the other small religions sit in "
            "twenty or thirty. "
            "**And the census's `Traditional` cell is not what it looks like.** 74 people, "
            "and the obvious reading is the Kalinago — the largest surviving indigenous "
            "community in the eastern Caribbean, at Sandy Bay in the north. It is not them: "
            "every one of the eight most-Kalinago districts returns zero Traditional. What "
            "it is instead, this census cannot say, so those 74 people are drawn in the "
            "residual rather than assigned to a religion nobody can verify."),
        how="census, 2012",
        grain="enumeration districts, 415 people on average",
        counts=_vc_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "vc" / "vc_eds.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        note="**THE COMPANION 11j ASKED FOR, AND THE CROSS-TABLE CHECK IS THE REASON TO "
             "TRUST IT.** 11j said *\"build it as a companion to Jamaica or not at all\"*; "
             "Jamaica landed 2026-09-06 (9ab) and this followed. **This file publishes NO "
             "religion total** — there is no `RLG_RTOTL` the way Jamaica has one — so the 18 "
             "religion cells are checked against `ETH_TPOP`, the separately tabulated "
             "ethnicity universe sitting in the same sheet. They agree **to the person on "
             "all 235 rows at every level**, and the 13 divisions and 221 districts each sum "
             "to the national row on all 18 categories with zero discrepancy. Two "
             "independently tabulated questions agreeing exactly is a better check than a "
             "column summing to its own neighbour. It also settles the universe: religion is "
             "asked of everybody, so no share needs a 15+ caveat. "
             "**KONTUR WAS BUILT, MEASURED AND REMOVED, AND THE REASON GENERALISES.** Every "
             "country since Kenya has been placed on Kontur's 400 m grid, so it was the "
             "default here too. A Kontur r8 hex is ~0.16 km2 and the median enumeration "
             "district is **0.66 km2**, with 43 units smaller than a single hex: the country "
             "holds **509 hexes**, **78 of 219 populated districts get none at all**, and the "
             "per-district Kontur/census ratio runs p10 0.00, median 0.45, p90 2.68. A "
             "weighting absent for a third of units and scattering over an order of "
             "magnitude for the rest is noise, not a weighting. Placement is 8.2's uniform "
             "within-unit instead, which a 0.66 km2 unit does not need improving on. **The "
             "rule: a population grid must be finer than the counting tier to be worth "
             "anything, and Kontur r8 stops paying at roughly 1 km2 per unit** — every "
             "earlier customer was far above that line, so the floor had never been reached. "
             "What it costs is bounded and named: the largest district is 44 km2 of "
             "uninhabited Soufriere massif, and uniform scatter puts one or two dots up a "
             "volcano. "
             "**The boundaries came out of a coastal engineering report.** The statistical "
             "office does not publish them; USCB digitised them from *Figure 3* of the "
             "Georgetown Coastal Defense environmental assessment. 384.6 km2 against the "
             "country's 389, and the exact `GEO_MATCH` join — 221/221, sixth country in the "
             "series, sixth exact join — is what vouches for it. "
             "**`Traditional` is not the indigenous population, and that was tested rather "
             "than assumed** (12's Philippines co-location technique, applied to the "
             "ethnicity columns in the same sheet): r = -0.03 against the Indigenous share, "
             "and zero Traditional in all eight of the most-indigenous districts. Filed in "
             "the residual with `afrodiasporic` recorded as the node that would be wanted if "
             "it were ever resolved.",
    ),
}
