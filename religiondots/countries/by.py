# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _by_place_weight(place):
    """countries.py hook. `place` is the Kontur 400 m hex layer, keyed to the seven units.

    Seven units over 207,600 km² is about 29,700 km² apiece, and each oblast's people are a
    regional capital and a thinning countryside; an equal share per polygon would put most of
    the colour on forest and Polesian marsh. sources/by_grid.py.
    """
    return _kontur_place_weight(place, "by_hexes.gpkg", "sources/by_grid.py")


def _by_counts():
    """EBRD Life in Transition Survey III, 2015-16, on Belstat's 1 January 2026 populations,
    with Catholics placed by the Catholic Church's diocesan counts. 8 answers, 7 units, EVERY
    ROW `modelled` (§7b).

    No Belarusian census asks about religion (2019 Form 2N has no item). The survey's split-half
    passes no usable category at seven units, and its Catholic geography is set aside on a
    witness: its Poles rank -0.21 against the 2019 census's. sources/by.py has the construction
    and the assertions; sources/by.md the record.
    """
    from by2016 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "by.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "by" / "by_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"by.csv units with no polygon: {missing}; re-run sources/by_geo.py")
    if df["unit"].nunique() != 7:
        raise SystemExit(f"{df['unit'].nunique()} Belarus units, expected 7")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"by.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "by": dict(
        name="Belarus",
        source="Life in Transition Survey III, 2015 to 2016 (European Bank for Reconstruction "
               "and Development), with Catholics placed by the Catholic Church's diocesan counts "
               "(Annuario Pontificio), against Belstat's 1 January 2026 populations",
        basis="self-identification, adults 18 and over",
        view=[23.1, 51.2, 32.8, 56.2],
        note_public=(
            "**No Belarusian census asks about religion, so this is a survey applied to the "
            "population.** The 2019 census form had 25 questions, nationality and language among "
            "them, and none on religion. The EBRD's Life in Transition Survey interviewed "
            "**1,504 adults** across all six oblasts and Minsk city in 2015 and 2016, and its "
            "national shares are applied to Belstat's 1 January 2026 population of 9,056,080. The "
            "dots are drawn desaturated to say that. "
            "**The survey can say how many, but not where.** Splitting its 75 interview clusters "
            "in half at random 400 times and re-ranking the seven units gives no answer a stable "
            "order: +0.11 for Orthodox, +0.21 for no religion and +0.37 for Catholic, where an "
            "answer with no geography at all reaches +0.43 to +0.50 on this many units. So "
            "Orthodox Christians, the non-religious and every smaller answer are drawn at the "
            "same national rate inside each unit, once Catholics are placed. "
            "**Catholics are placed by the Catholic Church's own diocesan counts, because the "
            "survey puts them in the wrong oblasts.** Its Catholic answers are most common in "
            "Gomel (18%) and Brest (14%), with Grodno, the historically Catholic west, third. "
            "Its ethnicity question goes wrong in the same places: it has Poles at 8.2% of Gomel "
            "against 0.19% in the 2019 census, and 3.9% of Grodno against 21.7%, and most of "
            "Gomel's Catholic answers came from five interviewers. The survey's national "
            "Catholic share, **9.2%**, is kept, and it is divided by the Church's count of "
            "Catholics in each diocese: Grodno 548,125, Minsk-Mohilev 652,300, Vitebsk 167,516 "
            "and Pinsk 54,140. That makes Grodno oblast **32.9%** Catholic and Vitebsk 9.3%. The "
            "Pinsk diocese covers Brest and Gomel oblasts, and Minsk-Mohilev covers Minsk city, "
            "Minsk oblast and Mogilev, so those units get their diocese's share, 1.2% and 8.7%; "
            "nothing here measures the difference between them. The Church's own total is 1.42 "
            "million, 15.7% of the country, a count of members rather than of people who call "
            "themselves Catholic, and it is used only for where. "
            "**The Buddhist figure is a ceiling.** Thirteen people answered Buddhist, which "
            "comes to **92,068** Belarusians in a country with a handful of registered Buddhist "
            "communities. The same card gave the same kind of answer in Kyrgyzstan, where it "
            "looked like a slip between neighbouring codes, and it is left at the national rate "
            "as the most Buddhism could be."),
        how="survey, one round of 1,504 interviews in 2015 and 2016; Catholics placed by the "
            "Catholic Church's diocesan counts",
        grain="oblasts and Minsk city, 1.3 million people on average",
        gap="0.25% of the survey, who would not say",
        gap_share=0.00253,
        counts=_by_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "by" / "by_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_by_place_weight,
        note="A SURVEY ON A REGISTER, EVERY ROW `modelled` (§7b), with a church's structure for one "
             "answer (§3.5a's shape). LiTS III `q922` gives the national composition; its "
             "split-half (lits.stability, 400 PSU halves against a shuffled-label null) passes "
             "only BUDDHIST, 13 respondents, overridden to the national rate as a keying artefact "
             "(by.py::OVERRIDE). CATHOLIC's survey geography is set aside on a witness, not on "
             "the test: LiTS's q923 Poles rank -0.21 against the 2019 census's Poles by unit, "
             "Gomel 8.17% against 0.19%, and Gomel's 41 Catholics sit with five interviewers "
             "(by.py::ethnicity_witness asserts the rank stays under +0.5). The survey's 9.21% "
             "is split by the Annuario Pontificio's Catholics per diocese, uniform within a "
             "diocese, and floored against the census's Poles in every unit (by.py::poles_floor). "
             "COD-AB's Minsk City is 87 km² against the city's 353.64 and misses five of nine "
             "city PSUs; by_geo.py replaces it with OSM relation 59195 plus COD's own polygon "
             "and asserts every city PSU inside it. Population is Belstat's 1 January 2026 "
             "table, not COD-PS.",
    ),
}
