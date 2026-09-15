# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _in_counts():
    """Census of India 2011 at sub-district: 91 categories on 5,988 units.

    ONE level. in.csv carries nation, state, district and subdistrict — the same 1.21
    billion people counted four times — and the allocated file adds a fifth reading of the
    finest one. Only `in_subdistrict_allocated.csv` is read here; it already contains the
    six census religions untouched plus the `Other religions and persuasions` bucket split
    into its 83 named religions, so reading in.csv as well would double the country.

    WHAT IS DERIVED AND WHAT IS NOT, because India's ratio is unusually good. The six
    religions — 99.34% of the population — are published at the sub-district and are
    `measured`. Only the 7,937,734 people in `Other religions and persuasions` are split
    from state-level structure, so India is 0.66% estimated against Australia's much larger
    share. And within that 0.66%, 245 of the (state, column) pairs have a single named
    religion and are therefore exact rather than allocated.

    THE ALLOCATION IS WITHIN EACH STATE, which is what makes it defensible at all. India is
    the first source where `allocate.py --within` was needed and the reason is visible in
    one line: Sanamahi is 100% Manipur, Niam Khasi 100% Meghalaya, Donyi-Polo 98% Arunachal
    Pradesh. Pooling the states into one national composition — which is what every earlier
    country does — would have put Manipuri and Arunachali religions into every sub-district
    in India in proportion to its `Other` count. Allocated within states, each religion
    reproduces its published state distribution exactly.

    The Annexure's 47 write-in sects are in in.csv and resolve to None here on purpose;
    see taxonomy/in2011.py for why a table that names 573 Shia Muslims is not a sect
    breakdown.
    """
    import in2011
    from in2011 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "in_subdistrict_allocated.csv",
                     dtype={"geo_id": str}, low_memory=False)
    # THE `Muslim` ROWS ARE REPLACED, NOT SUPPLEMENTED (in_split.py, 2026-09-14), the way
    # uk_split.csv replaces England's `Christian`. in_split.csv holds every sub-district's
    # Muslims again: Sunni and Shi'a rows from Pew's regional shares (`derived`,
    # rolling back to `islam` through in2011.COLUMNS) plus a smaller census `Muslim` for the
    # rest, summing to the census figure. Adding them without dropping these would double
    # India's Muslims, and the check below is what makes that impossible to do quietly.
    replaced = df.loc[df["source_category"] == "Muslim", "count"].sum()
    df = df[df["source_category"] != "Muslim"]
    split = pd.read_csv(HERE / "data" / "normalized" / "in_split.csv",
                        dtype={"geo_id": str}, low_memory=False)
    if abs(split["count"].sum() - replaced) > 0.5:
        raise SystemExit(f"in_split.csv draws {split['count'].sum():,.0f} Muslims against the "
                         f"{replaced:,.0f} it replaces; re-run in_split.py")
    df = pd.concat([df, split], ignore_index=True)
    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["unit"] = df["geo_id"]
    df["congregations"] = 0
    # spec §3.10: an allocated count may never ring, because a ring asserts presence and
    # allocation only spreads a total. India currently draws no rings at all — every node
    # reaches a dot somewhere — so this changes nothing today, and it is set anyway
    # because scatter.py defaults a missing `may_ring` to True. Without it, a future dot
    # value or a shrunken category would let an allocated Adivasi religion claim presence
    # in a sub-district that may have none of it.
    df["may_ring"] = df["tier"] == "measured"
    _add_roll(df, in2011.COLUMNS)
    return df[["unit", "node", "count", "congregations", "may_ring", "tier", "roll"]]


class _InSettlementWeighter:
    """India's dots land on villages and towns, weighted by their own 2011 population.

    spec §8.2a is the section that says India cannot do what every other country does:
    there is no statistical layer between the sub-district and the settlement, so an equal
    share per polygon would weight a hamlet like a city. sources/in_place.py builds the
    layer anyway and does the weighting there, where the census totals are in hand — a
    village's `t_pop2011` from SHRUG, a town's population from C-01 itself, and an
    area-proportional share of whatever a unit's total does not account for.

    So there is nothing per-node here and there cannot be: **India publishes no religion at
    any geography finer than the sub-district.** This is a population weight, exactly as
    Serbia's and Kenya's are. A Muslim dot and a Hindu dot in Malappuram are spread the same
    way, and a cluster means "this sub-district, drawn where its people actually live" and
    never a neighbourhood reading. Germany (§8.2b) is the only country on the map that can
    say more, because destatis publishes the religion on the grid.
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
        return (f"{self.n_pop:,} (unit, node) rows placed on village and town population, "
                f"{self.n_uniform:,} on equal shares where a unit's settlements sum to zero "
                f"(sources/in_place.py)")


def _in_place_weight(place):
    """countries.py hook. `place` is the settlement layer scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! in_places.gpkg has no `pop` column — run sources/in_place.py; "
              "placing on equal shares (§8.2)")
        return None
    return _InSettlementWeighter(place)


ENTRY = {
    "in": dict(
        name="India",
        source="Census of India 2011, table C-01 and its Appendix (ORGI); Muslim branches "
               "from Pew Research Center, Religion in India (2021)",
        basis="self-identification, reported by the head of household",
        view=[67.5, 6.5, 97.8, 36.0],
        gap="0.2%, whose religion the head of household did not state; Muslim branches only "
            "at Pew's six regions, and none where Pew did not survey",
        gap_share=0.002368,
        note_public=(
            "One in six people on earth, and the oldest source on this map by a decade: "
            "the 2021 census has never been held, so 2011 is not the best Indian figure "
            "but the only one. The census offers six boxes — Hindu, Muslim, Christian, "
            "Sikh, Buddhist, Jain — and clubs every sect into them, so the census cannot "
            "say more: its Syro-Malabar Catholics and its Digambar Jains are all inside a "
            "single category and no census table separates them. What the census does do, and almost uniquely, is write down what the "
            "7.9 million people who refused all six boxes actually said. Those answers are "
            "83 named religions, nearly all Adivasi: Sarna of the Chotanagpur sacred "
            "groves, five million strong and still campaigning for a box of its own; the "
            "Gondi religion of the central highlands; Donyi-Polo, the Sun-and-Moon faith "
            "the Tani peoples of Arunachal organised in the 1970s against the missions; "
            "Sanamahi, revived in Manipur against an 18th-century conversion; Niam Khasi "
            "and Niamtre in a Meghalaya that is three-quarters Christian. Those 83 are "
            "published only by state, so their placement within a state is derived. The "
            "six large religions are not — they are counted on all 5,988 sub-districts. "
            "India has no 'no religion' box at all, so the blank space on this map where "
            "irreligion would be is a property of the question and not of the country. "
            "**India's Muslims are divided using a survey, not the census.** Pew Research "
            "Center asked 3,336 Muslims in 2019 and 2020 whether they were Sunni or Shi'a and "
            "published the answers for six regions, so every sub-district in a region gets "
            "the same shares: **80%** Sunni in the North against **32%** in the Northeast, "
            "where 38% did not know or would not say. Those who named no branch stay as plain "
            "Muslims, **42%** of India's. The Shia of Lucknow and Hyderabad and the Bohras of "
            "Gujarat are spread across their whole region, and Kargil's Shia are not shown at "
            "all. Pew interviewed nobody in the Kashmir Valley, Ladakh, Manipur, Sikkim or five "
            "small union territories, so their Muslims are not divided."),
        how="census, 2011, answered by the head of household; Muslim branches from a 2019 to "
            "2020 survey",
        fill="from Pew's 2019 to 2020 survey regions for Muslim branches, and from the same "
             "census at state level for the other religions",
        grain="sub-districts, 200,000 people on average",
        counts=_in_counts,
        # THE COUNT LAYER IS STILL THE SUB-DISTRICT AND NOTHING HERE CHANGES THAT. India's
        # religion figures are published on 5,988 sub-districts and are read from exactly
        # those; the median one still holds about 204,000 people and is still the coarsest
        # count unit on the map. What changed is only where inside one a dot may land.
        #
        # §8.2a is the section that said India could not have a placement layer: its finer
        # geography is 645,828 villages and 4,135 towns, natural settlements running from
        # ten people to two million rather than units built to a population target, so an
        # equal share per polygon would weight a hamlet like a city. The answer is not to
        # share equally but to WEIGHT BY THE SETTLEMENT'S OWN POPULATION, which SHRUG and
        # C-01 between them publish for very nearly all of them — and `place_weight`, built
        # for the US in §8.4, is the hook that takes it.
        #
        # sources/in_place.py does the joining and the weighting; its docstring carries the
        # traps, of which the sharp one is that 3,892 six-digit codes name both a village
        # and a town. Every unit's weights sum to its census total, because each unit also
        # carries its own outline holding whatever its settlements do not account for —
        # Assam publishes no village populations at all, so that fallback is the whole of
        # its rural placement and is precisely §8.2a's behaviour. Nowhere is worse than
        # before; most places are a great deal better.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "in" / "in_places.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_in_place_weight,
        note="ORGI is self_id but answered by the head of household, not per person, "
             "which is why `religion not stated` is only 0.24% (spec §3.1). The 0.66% in "
             "`Other religions and persuasions` is allocated within each state, not "
             "pooled nationally (allocate.py --within; spec §3.10). The `Muslim` column is "
             "split into Sunni and Shi'a by Pew 2021's six regions (in_split.py, "
             "sources/in.md §8); those rows are derived and roll back to islam.",
    ),
}
