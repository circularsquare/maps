# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _sv_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    El Salvador is Guatemala's problem inverted. Nothing here is EMPTY -- 14 departments over
    21,041 km2 is 1,503 km2 a unit, the tightest ratio of any LAPOP country here -- but
    **San Salvador is 1.7 million people in 886 km2, a quarter of the country on 4% of its
    land**, and the metropolitan area spills over the La Libertad line rather than stopping
    at it. An equal share per polygon puts a quarter of the country's colour on the Volcan de
    San Salvador and the Cordillera del Balsamo (sources/sv_grid.py).
    """
    return _kontur_place_weight(place, "sv_hexes.gpkg", "sources/sv_grid.py")


def _sv_counts():
    """LAPOP AmericasBarometer, waves 2010-2023 pooled, at departamento: 11 categories, 14
    units, and EVERY ROW IS `modelled` IN §7.

    The second country drawn from the AmericasBarometer, after Guatemala. sources.md §11x
    closed El Salvador by sweeping ONEC/BCR's WordPress library -- 777 media items, one census
    file, zero religion tables -- and the UNSD oracle has no Salvadoran row at all. Nothing
    here reopens that. sources.md §11ad assesses the source; sources/sv.md is this country's
    record; sources/lapop.py holds the construction that both countries share.

    THE BEST-SAMPLED OF THE NINE, AND IT SHOWS. 9,063 respondents over 14 departments is 647
    apiece against Guatemala's 405, so the split-half runs on ~4,500 a side and comes back
    much stronger: Catholic +0.88 and Evangelical +0.82 where Guatemala managed +0.57 and
    +0.50. **THREE categories carry their own department geography here rather than two** --
    `Ninguna (creyente)` passes at +0.74 where Guatemala's failed at +0.21 -- so El Salvador
    is the first country in this set whose no-religion geography is a measurement rather than
    a national rate spread flat.

    `Protestante Tradicional` MISSES THE BAR BY 0.02, at +0.52 against +0.54 on 14 units, and
    is drawn at the national rate. That is the closest call in either country and it was left
    alone deliberately: the bar is 1.96/sqrt(n-1), and moving it because a value landed just
    underneath is fitting the test to the answer.

    THE JOIN IS ON NAME AND THE CODE JOIN IS A TRAP -- the opposite of Guatemala, which joins
    on the code. LAPOP's `prov` is 300 plus the official west-to-east department number;
    **COD's SV pcodes are ALPHABETICAL**. Two of fourteen coincide and twelve are wrong, and a
    permutation preserves every total, so San Salvador's 1.7 million would have been drawn in
    La Paz with every check still passing. sources/sv_geo.py joins on the name and asserts
    that the code join still mispairs, so nobody restores it.
    """
    from sv2023 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "sv.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])

    lut = pd.read_csv(HERE / "data" / "geo" / "sv" / "sv_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"sv.csv departments with no polygon: {missing} -- re-run "
                         "sources/sv_geo.py, the lookup is stale")
    if df["unit"].nunique() != 14:
        raise SystemExit(f"{df['unit'].nunique()} departments, expected 14")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"sv.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    # EVERY row, without exception -- there is no measured tier in this country (§7).
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "sv": dict(
        name="El Salvador",
        source="AmericasBarometer, six rounds 2010 to 2023 (LAPOP Lab, Vanderbilt "
               "University), against OCHA COD-PS 2024 department populations",
        basis="self-identification, adults 18 and over",
        view=[-90.3, 13.0, -87.5, 14.6],
        note_public=(
            "**El Salvador's census does not ask about religion, so this is a survey "
            "standing where a census would be.** Its statistics office publishes 777 files "
            "and not one of them carries the question, and the UN's register of census "
            "religion tabulations has no Salvadoran row at all. The map is drawn from the "
            "LAPOP AmericasBarometer: **9,063 people** across six rounds between 2010 and "
            "2023, pooled, applied to the 2024 population estimates. The dots are drawn "
            "desaturated to say that. "
            "**It is the best-measured country of the nine this survey could serve**, at 647 "
            "interviews per department against Guatemala's 405, and the difference shows in "
            "what the map is willing to claim. Four of the eleven answers are drawn where "
            "the survey found them rather than two: splitting the six rounds in half and "
            "re-ranking the 14 departments returns **+0.88** for Catholic, +0.82 for "
            "Evangelical and **+0.74** for the people who believe in a higher being and "
            "belong to no religion. In Guatemala that last one came back at +0.21 and had to "
            "be spread flat. "
            "**So this is the first country here whose no-religion geography is a "
            "measurement.** It is **18.3% of Usulután** and 17.2% of Morazán against 6.2% of "
            "La Paz, and those two are the eastern departments that sent the most people "
            "abroad during and after the civil war. "
            "**Santa Ana is the only department where Evangelicals outnumber Catholics**, at "
            "37.6% against 32.7%. The other end is San Vicente at **76.6% Catholic** and "
            "11.0% Evangelical, so the Catholic share runs better than two to one across a "
            "country of 21,000 square kilometres. "
            "**A fourth answer joined them in September 2026, without any new data.** "
            "`Protestante Tradicional`, 8.0% of El Salvador and **506,196** people, re-ranks "
            "the departments at +0.52. That used to be counted as a miss, against a bar of "
            "+0.54 set by a formula that turned out to be a one in fifty test rather than the "
            "one in twenty it was meant to be; against the real one in twenty bar of +0.46 it "
            "passes, and it is now drawn where the survey found it. It is 11.7% of Santa Ana "
            "and 3.7% of Morazán. Correcting that bar moved this answer and one in Costa "
            "Rica, and nothing else in the five countries drawn from this survey. "
            "**The level is a fourteen-year average.** Salvadoran Catholicism runs 51.5% in "
            "2010 to 42.1% in 2023 across the pooled rounds, so this map is several points "
            "more Catholic than the last round alone would draw. Pooling is what buys the "
            "department detail; a single round is 1,500 people over 14 departments."),
        how="survey, six rounds 2010 to 2023 pooled",
        grain="departments, 450,000 people on average",
        counts=_sv_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "sv" / "sv_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_sv_place_weight,
        note="THE COUNTRY IS A SURVEY ON A PROJECTION AND EVERY ROW IS `modelled` (§7b), the "
             "same construction as Guatemala and sources/kz.py: LAPOP's `q3c` gives a "
             "department share, OCHA COD-PS 2024 gives the people it applies to, and no "
             "magnitude is invented. sources/lapop.py holds the shared half so that nine "
             "countries cannot drift apart in the places that matter. "
             "THE CODE JOIN IS A TRAP HERE AND IT IS GUATEMALA'S INVERTED. LAPOP's `prov` is "
             "300 plus the official west-to-east department number; COD's `SV` pcodes are "
             "ALPHABETICAL. Two of fourteen coincide -- Ahuachapan at 01 and La Libertad at "
             "05, which is exactly enough for a spot check to pass -- and the other twelve "
             "are wrong. A permutation preserves every total, so San Salvador's 1.7 million "
             "would have been drawn in La Paz with every reconciliation still passing. "
             "sources/sv_geo.py joins on the NAME, all 14 matching with no aliases, and "
             "asserts that the code join still mispairs so that nobody restores it. "
             "THE HELD-OUT CHECK IS A PERMUTATION TEST, AND THE ONE IT REPLACED HAD NO "
             "POWER. LAPOP's weighted department distribution tracks COD-PS's population "
             "distribution at r=+0.968, and **none of 20,000 random pairings of the same 14 "
             "units reaches that** (best random +0.964, which is why the permutation and not "
             "the correlation is the test). An earlier version also asserted on department "
             "mean adult age; that comparison was then measured and its between-unit variance "
             "is BELOW its sampling variance in both countries (F=0.36 here, 0.88 in "
             "Guatemala), so it cannot discriminate a good decode from a permuted one. It is "
             "still printed, with its F beside it, and it decides nothing. "
             "THREE CATEGORIES CARRY THEIR OWN GEOGRAPHY AND EIGHT DO NOT, decided by the "
             "split-half rather than by size (§14.16). `Protestante Tradicional` at 7.97% "
             "misses the +0.54 bar by 0.02 and `Religiones Orientales` at 1.42% by more; both "
             "are drawn at the national rate inside each department's residual, so the "
             "partition stays closed and only the claim to know where those people are is "
             "withdrawn. "
             "UNLIKE GUATEMALA, THE TRADITIONAL-RELIGION CELL IS NOT CALLED A FLOOR. It is "
             "0.03% here, and El Salvador's 2007 census counted 0.2% indigenous after the "
             "1932 matanza made Nahua-Pipil identity dangerous to state, so §11ad's Suriname "
             "finding -- a card with no local option in a country with a large indigenous "
             "population -- does not apply in its second half. taxonomy/sv2023.py has it.",
    ),
}
