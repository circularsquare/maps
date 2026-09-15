# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _ba_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "ba_grid_400m.gpkg", "sources/ba_geo.py")


def _ba_counts():
    """BHAS Popis 2013 at municipality: 5 nodes on 142 units.

    ONE level, no allocation, nothing modelled — every drawn row is `measured` and may ring.

    **THE PARTITION IS EXACT IN BOTH DIRECTIONS**, which is the whole reconciliation and is
    better than most sources here manage: the eight categories sum to each municipality's own
    total, and the 142 municipalities sum to BHAS's published national row category by
    category, both with a discrepancy of zero. Nothing is suppressed, rounded or prorated.
    sources/ba.py asserts it rather than reporting it.

    **98.9% of the country is drawn** — 3,491,871 of 3,531,159. What is not is the two
    non-response cells —
    `Nisu se izjasnili` (32,700 active refusals) and `Bez odgovora` (6,588 with no answer
    recorded) — which BHAS publishes apart and taxonomy/ba2013.py keeps apart, per §3.5.
    """
    from ba2013 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ba.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "municipality"].copy()
    if df["geo_id"].nunique() != 142:
        raise SystemExit(f"{df['geo_id'].nunique()} municipalities, expected 142 -- re-run "
                         "sources/ba.py")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "ba": dict(
        name="Bosnia and Herzegovina",
        source="Census 2013 (Agency for Statistics of Bosnia and Herzegovina)",
        basis="self-identification",
        view=[15.6, 42.5, 19.7, 45.35],
        gap="the 1.1% who declined the question or gave no answer at all",
        gap_share=0.01113,
        note_public=(
            "**Three religions and almost nothing else — 96.6% of the country is Muslim, "
            "Orthodox or Catholic**, and the census offers no fourth religious box. That "
            "is not a thin form so much as an accurate one: in Bosnia religion and "
            "nationality are near-substitutes, Bosniak with Muslim, Serb with Orthodox, "
            "Croat with Catholic, and a question about either returns much the same "
            "answer. So this map is legible at a glance and tells you almost nothing "
            "about belief or practice. "
            "**At 50.7% Muslim it is the second most Muslim country in Europe** after "
            "Kosovo, and the only one on this map where three large religions meet. "
            "**The striking thing is how completely separated they are.** Most "
            "municipalities are not mixed at all: Bužim is 99.7% Muslim, Posušje 99.8% "
            "Catholic, Ribnik 99.5% Orthodox, and 84 of the 142 units are over 90% one "
            "religion. **That geography is not ancient — it was made between 1992 and "
            "1995**, and this census is the first since. Srebrenica, 73% Muslim in 1991, "
            "returns 55% Muslim and 45% Orthodox here. "
            "**The exceptions are worth finding.** Brčko in the north is the only "
            "genuinely three-way unit — 44% Muslim, 35% Orthodox, 21% Catholic — and it "
            "is the district that was placed under international arbitration precisely "
            "because neither entity could be given it. Mostar is split 50/46 Catholic and "
            "Muslim across the Neretva, and the central Bosnian valley towns — Vitez, "
            "Busovača, Kiseljak, Novi Travnik, Jajce — each sit near half and half. "
            "**Irreligion is a Sarajevo phenomenon and it is small.** 1.1% nationally, "
            "the lowest of any European country here, but 8.6% in Centar Sarajevo and "
            "7.1% in Novo Sarajevo. There is no 'no religion' box at all — the only "
            "irreligious answers are the positions *atheist* and *agnostic* — so that "
            "1.1% is a floor rather than a measurement. "
            "**The 2013 results are disputed and the dispute is about who counts as "
            "resident, not about religion.** Republika Srpska's statistical institute "
            "rejected BHAS's treatment of people living abroad and publishes lower "
            "figures for its own entity. The numbers here are BHAS's, which are the ones "
            "the state, Eurostat and the EU use."),
        how="census, 2013",
        grain="municipalities, 25,000 people on average",
        counts=_ba_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ba" / "ba_grid_400m.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ba_place_weight,
        note="**§11c FILED THIS COUNTRY AS 'REACHABLE, NOT QUICK' AND BOTH HALVES WERE "
             "TRUE WITHOUT EITHER MATTERING.** It recorded `popis.gov.ba` as a React SPA "
             "with no data endpoint and the results as 12-19 MB PDF books, then stopped. "
             "§12's grep-the-bundle rule was run here and returned **zero `/api` routes in "
             "all 125 KB** — the correct negative, not a failed search, because there is "
             "nothing behind the SPA to serve: the books are static files under "
             "`/popis2013/doc/`. The directory listing 403s while the files themselves are "
             "open, so the one thing that looked like a wall was a statement about a "
             "directory. **A 403 on a directory is not a 403 on its contents**, and the "
             "file was one guessed path away for a year. "
             "**The reconciliation is as good as any source here has had.** Table 4.3 is "
             "twelve printed pages; its eight categories partition every municipality "
             "exactly, and the 142 municipalities sum to the published national row "
             "category by category with a gap of **zero** — no suppression, no rounding, "
             "single people published. "
             "**The parse trap that would have shipped quietly is the neighbouring "
             "table.** 4.1 (entity) and 4.2 (canton) sit immediately before 4.3, use "
             "identical headers and an identical three-row-per-unit structure, and a "
             "generous page window swallows all three — summing to **5,583,946 against a "
             "country of 3,531,159** while still passing a per-row partition check, "
             "because each of those rows is internally consistent. Only the national total "
             "catches it. The window is pinned and the caption asserted. "
             "**And geoBoundaries BIH ADM3 is dirty in four ways, with 142 units against "
             "the census's 142** — the coincidence that makes §9s's missing-unit trap look "
             "like a clean join. One polygon is named **`Republika Srpska`** and is "
             "actually Višegrad; **`Novi Grad` appears twice** and the two are 130 km "
             "apart, so a name join is wrong about Novi Grad Sarajevo's 118,553 people; "
             "`Kupres` and `Kupres (BiH)` are the RS and Federation halves with the "
             "labels reading backwards; and `Kupra na Uni` is a typo for Krupa na Uni. "
             "All four are repaired in `sources/ba_geo.py` **before** any matching, each "
             "with a geometric assertion beside it, so the join itself stays a plain name "
             "join that either works completely or fails completely — 142/142, both ways. "
             "**The Kontur check had to have its band measured rather than inherited** "
             "(§9u): the counts are 2013 and the surface 2023, and BiH emigrated heavily "
             "in between, so the ratio sits at a median of **0.94x** rather than near 1. "
             "What the check tests is that the ratio holds *together* — 141 of 142 units "
             "inside median/3 to median×3, with only Usora outside.",
    ),
}
