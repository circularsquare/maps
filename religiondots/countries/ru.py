# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


class _RuHexWeighter:
    """Split a federal subject's dots across its 3km hexes by hex POPULATION.

    The most consequential placement weight on this map, because Russia has the worst
    within-unit problem on it: the counts are at federal subject, a mean of 1.8 million
    people over a mean of 200,000 km², and Sakha alone is 3.08 million km² with a million
    people living along four rivers. spec §8.2's usual trick — an equal share over a layer
    an agency built to a population target — has nothing to work with here, so the weight
    is a measured population surface (Kontur H3 r6, sources/ru_geo.py).

    It is a POPULATION weight and not a religion one. Nothing in Russia measures where the
    Old Believers or the Sunni live inside a subject, so every node in a subject is spread
    identically and a Buddhist dot in Buryatia sits where Buryatia's people are, not where
    its Buddhists are. Germany is the only country here that does better, because destatis
    publishes religion on its grid; Russia publishes religion for 79 polygons and nothing
    else at all.
    """

    def __init__(self, place):
        self.pop = place["pop"].to_numpy(dtype=float)
        self.n_pop = 0
        self.n_uniform = 0

    def weights(self, node, idx, count, plain=False):
        pop = self.pop[idx]
        if pop.sum() > 0:
            self.n_pop += 1
            return pop
        self.n_uniform += 1
        return None

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on 3km hex population, "
                f"{self.n_uniform:,} on equal shares where a subject's hexes sum to zero "
                f"(sources/ru_geo.py)")


def _ru_place_weight(place):
    """countries.py hook. `place` is the 3km hex layer scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! ru_grid_3km.gpkg has no `pop` column — run sources/ru_geo.py; "
              "placing on equal shares, which for Russia is very wrong (§8.2)")
        return None
    return _RuHexWeighter(place)


def _ru_counts():
    """Sreda Arena 2012 at federal subject: 18 answers on 79 units.

    ONE level, no allocation, and none is possible — Arena published one cross-tabulation
    and there is nothing finer or coarser to reconcile it against. ru.csv also carries a
    `country` level, which is Arena's own national column and the same people again.

    THE COARSEST GEOGRAPHY ON THIS MAP BY A WIDE MARGIN. 79 units for 142.6 million people
    is 1.8 million each; North Macedonia, the previous worst, is 23,000. Every dot inside a
    subject is drawn from the same distribution because that is the only thing measured,
    and sources/ru_geo.py places them on a 3km population surface so that at least they sit
    where Russians actually live.

    AND THE ONLY COUNTRY DRAWN FROM A SURVEY AS ITS PRIMARY SOURCE. Russia's census has not
    asked about religion since 1937 and the 2021 census does not either, so there is no
    census route. Arena is 56,900 respondents, about 720 per subject: the large categories
    are solid and the small ones are sampling noise wearing a map. `tier` is `modelled` on
    every row for that reason, which is what spec §7 is for — nothing here is a count of
    anybody, and the Old Believers at 0.32% are the clearest case (see ru2012.py).

    ARENA PUBLISHES SHARES, NOT PEOPLE. sources/ru.py multiplies them by the 2021 census
    population per subject — §3.4's "structure from the detailed source, totals from the
    recent one", the same rule Brazil follows. Applying 2012 shares to 2021 populations is
    deliberate rather than lazy: the Muslim republics grew and the Russian oblasts shrank
    over those nine years, and this carries that shift instead of freezing it.

    ARENA COVERS 79 OF 83, AND THE OTHER FOUR ARE FILLED FROM CENSUS ETHNICITY. It has no
    Chechnya, no Ingushetia, no Nenets and no Chukotka, and the first two are the most
    Muslim republics in the country — so the hole sat exactly where Islam is densest and
    made Russia read as less Muslim than it is. `ru_fill.py` estimates those four from the
    2021 census ethnic composition through a relationship fitted on Arena's own 79 measured
    subjects (islam = 0.724 x muslim ethnic share, R² = 0.964), and this adapter reads the
    filled file. Their rows carry `source_id = ru_ethnic_fill_2021` and are separable at
    any point; every Russian row is `modelled` either way. See ru_fill.py for why Chechnya
    comes out at 84.8% rather than the 98.5% a naive ethnic fill would give.
    """
    from ru2012 import resolve

    path = HERE / "data" / "normalized" / "ru_filled.csv"
    if not path.exists():                       # the fill is a separate step, like br_rescale
        print("  !! ru_filled.csv missing — run `python ru_fill.py`; drawing Arena's 79 "
              "subjects only, so Chechnya and Ingushetia will be blank")
        path = HERE / "data" / "normalized" / "ru.csv"
    df = pd.read_csv(path, dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "subject"].copy()

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "modelled"
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "ru": dict(
        name="Russia",
        source="Sreda «Arena» Atlas of Religions and Nationalities 2012",
        basis="self-identification, sample survey",
        # REQUIRED here, not cosmetic. Filling Chukotka put dots east of 180°, so the
        # measured bbox is [-175.3 … 179.5] — which does not mean "Russia is wide", it
        # means "Russia wraps", and fitting it frames the entire globe. This stops at the
        # antimeridian and gives up the Chukotkan sliver beyond it: 47,000 people who are
        # still drawn and still reachable by panning. The US `view` exists for the same
        # class of reason (Alaska and Hawaii); this is the antimeridian version.
        view=[19.0, 41.0, 180.0, 78.0],
        gap="5.4% of the survey, who found the question difficult to answer",
        gap_share=0.05443,
        note_public=(
            "**The only country here that is not drawn from a census.** Russia has not "
            "asked about religion in a census since 1937 and the 2021 census does not ask "
            "either, so this map is a survey: Sreda's Arena project, 56,900 respondents "
            "across 79 of the then 83 federal subjects, in the summer of 2012. That is "
            "about 720 people per region, so read the large shares and distrust the small "
            "ones — the Old Believers at 0.32% are roughly 180 respondents spread over 79 "
            "regions, and where they appear on this map is close to noise even though the "
            "national figure is real. "
            "**It is also the coarsest map here by a wide margin**, at 79 units for 142.6 "
            "million people, about 1.8 million each. Every dot inside a region is drawn "
            "from the same mixture, because a region is all the survey measures; the dots "
            "are placed on a 3km population grid so they at least sit where Russians "
            "actually live, which in Sakha means four river valleys in an area the size of "
            "India. "
            "**What makes it worth drawing anyway is the question.** Arena offered "
            "seventeen answers and split things almost nothing else does: the Russian "
            "Orthodox Church from Orthodoxy outside it and from the Old Believers, and "
            "Sunni from Shia from Muslims who decline both. It is the only source on this "
            "map that asks a Muslim which branch, and the answer turns out to be regional "
            "rather than doctrinal — **Dagestan says Sunni (48.6%) and Tatarstan and "
            "Bashkortostan say neither** (31.5% and 38.3% answered 'I profess Islam, but "
            "am neither Sunni nor Shia', against 1.4% and 0.4% Sunni). Nationally that "
            "'neither' answer is 4.7% of Russia against Sunni's 1.7%, so what looks like "
            "a map of Islamic branches is substantially a map of how the question was "
            "taken. "
            "**Two answers about belief rather than belonging are 38% of the country "
            "between them.** 24.9% say they believe in God but profess no particular "
            "religion — 44% in Karelia, 41% in Komi and Amur — and 12.9% say they do not "
            "believe in God, which peaks in the Far East and southern Siberia: Primorsky "
            "34.8%, Altai Krai 27.4%, Sakha 25.6%. Arena never offers 'no religion' as an "
            "option at all, so the people who would tick that box elsewhere are split "
            "between those two here. "
            "**The Orthodox core is the Black Earth, not Moscow.** The Russian Orthodox "
            "Church is 41.3% nationally and runs 78.4% in Tambov, 71.3% in Lipetsk and "
            "69.3% in Nizhny Novgorod, against 52.8% in Moscow and 26.6% in Primorsky. "
            "Two regions are outright majority something else — **Tuva 61.8% Buddhist** and "
            "**Dagestan 82.6% Muslim** across its three Islamic answers — and in Kalmykia "
            "the largest single answer is Buddhism at 37.6%, which is true of nowhere else "
            "in Europe. North Ossetia is the strangest column in the table: 49.2% Russian "
            "Orthodox and 29.4% practising the traditional Ossetian religion at once. "
            "**Four regions were never surveyed and are estimated, not measured.** Arena "
            "covers 79 of 83: it has no Chechnya, no Ingushetia, no Nenets and no "
            "Chukotka, and the first two are the most Muslim republics in Russia. Those "
            "four — 2.1 million people, 1.5% of the country — are filled in from the 2021 "
            "census's ethnic composition, using the relationship between ethnicity and "
            "Arena's own answers measured across the other 79 regions. That relationship "
            "is strong but it is not one-to-one: **only about seven in ten ethnically "
            "Muslim Russians actually give an Islam answer**, the rest saying they believe "
            "in God without a religion, or not at all. So Chechnya is drawn at 85% Muslim "
            "rather than the 98% its ethnic make-up alone would suggest. Treat those four "
            "regions as an informed estimate and the other 79 as a survey. A further 5.4% "
            "of the country answered 'difficult to say', which is itself regional — 18% in "
            "Magadan and 16% in Sakhalin against 0.4% in North Ossetia. "
            "**And it is fourteen years old**, in a period when this is one of the things "
            "about Russia most likely to have moved. No successor survey exists."),
        how="survey, 56,900 people, 2012; no census asks",
        grain="federal subjects, 1.7m people on average",
        counts=_ru_counts,
        # Counts are on the federal subject; the Kontur hexes carry no subject code, so
        # `units` + `unit_key` puts scatter.py on the spatial-join path and sources/ru_geo.py
        # has already assigned and clipped every hex. `place_unit` reads the column it wrote.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ru" / "ru_grid_3km.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ru_place_weight,
        note="Arena publishes SHARES, not people; sources/ru.py multiplies them by the 2021 "
             "census population per subject (spec §3.4, Brazil's rule). Every row is tier "
             "`modelled` — a 720-per-region survey is not a count of anybody. Placement is "
             "Kontur's 3km H3 population grid, 107,240 populated hexes covering 3.9M km² of "
             "a 16.4M km² country, which is the single most useful thing the layer does "
             "here; its own totals reproduce the census to 0.984x nationally and stay "
             "inside a factor of two in all 79 subjects (sources/ru_geo.py). Boundaries are "
             "geoBoundaries ADM1, which carries 83 subjects and no Crimea or Sevastopol — "
             "the same composition Arena surveyed, so the two agree about what Russia is "
             "without any editing.",
    ),
}
