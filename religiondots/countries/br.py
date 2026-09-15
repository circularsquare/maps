# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


class _BrSetorWeighter:
    """Split a município's dots across its census setores by setor POPULATION.

    Mechanically the same as `_PhBarangayWeighter` above; kept separate because the reason
    it is needed differs, and because a weighter is the place a country's placement
    argument lives. If a third country needs one, these two should become one class.

    §8.2 assumes a placement unit is small enough that spreading dots evenly inside it is
    harmless. **São Paulo is one polygon holding 11.5M people**, so for Brazil that
    assumption fails at exactly the places a reader looks first — the dots would be as
    dense in the Serra da Cantareira as on Avenida Paulista. Setores fix it: ~452,000 of
    them, nesting inside the município by code.

    AND EQUAL SHARES PER SETOR IS NOT ENOUGH, which is where this differs from the US.
    American tracts are built to ~4,000 people each, so an equal split is already a
    population weighting and `place_weight` is unnecessary. Brazilian setores are built to
    roughly 300 households in cities and fewer in the country, and rural ones cover
    enormous areas — an equal split would systematically pull Brazil's dots into the
    countryside. The weight is the setor's own 2022 resident population (`v0001`).

    It is a POPULATION weight, not a religion one. Nothing measures where a given church's
    members live inside a município, so a Catholic dot and an Assembleia de Deus dot are
    spread identically, and every município's total is exactly IBGE's either way. §14.4
    permits precisely this and no more: refine placement, never invent magnitude. Read a
    cluster as "this município, drawn where its people are", never as a neighbourhood.
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
        return (f"{self.n_pop:,} (unit, node) rows placed on setor population, "
                f"{self.n_uniform:,} on equal shares where a município's setores sum to "
                f"zero (sources/br_setores.py)")


def _br_place_weight(place):
    """countries.py hook. `place` is the setor layer scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! br_setores_2022.gpkg has no `pop` column — run sources/br_setores.py; "
              "placing on equal shares (§8.2)")
        return None
    return _BrSetorWeighter(place)


def _br_counts():
    """IBGE Censo 2022 totals on Censo 2010 structure, at município — spec §3.4, BUILT
    2026-09-04.

    `br_rescale.py` writes the file this reads. Until then Brazil was drawn at 2010 as it
    stood, because 2022 publishes NINE categories and lumps 47.4M evangelicals into one
    (sources/br.md §1); §3.4 has described the fix since the project began and it is now
    done. Every município's rescaled rows sum to that município's own 2022 total exactly.

    TWO THINGS CHANGED FOR THE READER, and note_public carries both.

    The totals are 2022, so Brazil is no longer fifteen years stale: Catholics 64.6% ->
    56.7%, evangelicals 22.2% -> 26.9%, Umbanda e Candomblé 588,810 -> 1,849,835.

    The universe is now **people aged 10 or over**, because that is who the 2022 religion
    question was put to. The drawn population falls from 190.8M to 176.3M. It is NOT scaled
    back up to the whole population, for Chile's reason (§14.4): the source publishes an
    exact partition of its own universe, and inventing the rest would be a larger claim
    than anything else on this map makes.

    TIER IS NOT UNIFORM HERE, which is unusual. The three 2022 categories that map to a
    single 2010 leaf — Católica Apostólica Romana, Espírita, Tradições indígenas — pass
    through untouched and are `measured`, 58.7% of the drawn people. The rest is `derived`:
    a 2022 magnitude wearing a 2010 shape, and §3.10 forbids it from ringing.
    """
    import br2010
    from br2010 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "br_municipio_rescaled.csv",
                     dtype={"geo_id": str}, low_memory=False)

    # The rescaled file is already leaves-only — br_rescale.py derives them the same way,
    # from IBGE's own parent chain — so there is no nesting left to collapse here.
    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    df["may_ring"] = df["tier"] == "measured"
    # §7a-i-1. Brazil's "column" is the 2022 GROUP the município total was published for —
    # `group22=` in the note rather than `parent_column=`, but the same claim: IBGE counted
    # that many evangelicals in that município, and only their denomination is 2010's.
    _add_roll(df, br2010.COLUMNS, key="group22")
    return df.rename(columns={"geo_id": "unit"})[
        ["unit", "node", "count", "congregations", "may_ring", "tier", "roll"]]


ENTRY = {
    "br": dict(
        name="Brazil",
        source="Censo Demográfico 2022 totals on 2010 denominations (IBGE)",
        basis="self-identification, census sample, people aged 10 or over",
        # The dot bbox runs to 28.8°W — Martim Vaz, in the Atlantic, which belongs to the
        # município of Vitória, as Fernando de Noronha belongs to Pernambuco. Real data, but
        # fitting it puts 6° of empty ocean beside the country. Mainland only.
        view=[-74.2, -34.0, -34.2, 5.5],
        note_public=(
            "**Two censuses, and each supplies half of this map.** How many people are in "
            "each município and each broad group is the 2022 census; which denomination "
            "they are drawn as is the 2010 one. IBGE published only nine categories for "
            "2022 and withheld the evangelical breakdown over data quality, saying it may "
            "never appear — so 2010 is not merely the older list, it is the only municipal "
            "denominational data Brazil has. Each município's 2022 evangelical total is "
            "split by that same município's 2010 mix, which is why Assembleia de Deus and "
            "Congregação Cristã can be drawn apart at all. "
            "**The change between the two is the point.** Catholics fell from 64.6% to "
            "56.7% and evangelicals rose from 22.2% to 26.9%, and Umbanda and Candomblé "
            "more than tripled, from 589,000 to 1,850,000 — the largest proportional move "
            "between the censuses, and whether that is growth, reduced stigma or a changed "
            "question is not something the numbers can say. "
            "**Two things to hold.** The 2022 question was asked only of people **aged 10 "
            "or over**, so this draws 176.3 million of Brazil's 203 million and nothing "
            "here scales it up to children. And about 41% of these dots carry a 2022 "
            "magnitude on a 2010 shape: where a denomination has grown or shrunk unevenly "
            "inside a município since 2010, this map cannot see it. The Catholic, Spiritist "
            "and indigenous-tradition dots are the exception — those three came through 2022 "
            "untouched, and they are most of the country. "
            "**Where the dots sit inside a município is a population estimate, not a "
            "measurement.** Religion is published per município and São Paulo is a single "
            "one holding 11.5 million people, so the dots are spread across Brazil's "
            "452,000 census setores in proportion to how many people live in each. That "
            "puts them on the streets rather than in the forest, but nothing measures which "
            "setor a given church's members are in — a Catholic dot and an Assembleia de "
            "Deus dot are spread the same way. Read a cluster as this município drawn where "
            "its people are, never as a neighbourhood."),
        how="census, 2022 totals with 2010 denominations",
        fill="from the 2010 census",
        grain="municípios, 32,000 people on average",
        counts=_br_counts,
        # PLACEMENT IS SETORES, COUNTS ARE MUNICÍPIOS — added 2026-09-05. The counts are
        # published per município and nothing finer exists, but a município is far too big
        # to spread dots evenly inside (São Paulo is one polygon, 11.5M people), so the
        # dots are placed across the ~452,000 census setores weighted by setor population.
        # No `units`/`unit_key` spatial join is needed: a setor's code STARTS WITH its
        # município's, so the assignment is a string slice (sources/br_setores.py).
        #
        # 2022 vintage throughout (§8.1), changed 2026-09-04 with the §3.4 rescale.
        # br_municipios_2022.gpkg is still built and is the right layer for anything that
        # wants municipal outlines; it is no longer what places the dots.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "br" / "br_setores_2022.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_br_place_weight,
        note="PLACEMENT IS SETORES (2026-09-05, spec §8.2d): the counts are municipal and "
             "cannot be finer, but a município is far too large to spread dots evenly "
             "inside — São Paulo alone is one polygon holding 11.5M people — so the dots "
             "sit in 466,996 census setores weighted by setor population "
             "(sources/br_setores.md). Same 176,291 dots as before, across 150,088 polygons "
             "instead of 5,565, with no fallbacks. A population weight, not a religion one. "
             "§3.4 built 2026-09-04: 2022 municipal totals split by 2010 municipal shares "
             "(br_rescale.py). 58.7% of the drawn people are `measured` — the three 2022 "
             "categories that map to one 2010 leaf — and 41.3% `derived`, which may never "
             "ring. Both censuses are sample tabulations and municipal figures do not sum "
             "to IBGE's national ones, by construction (sources/br.md §4).",
    ),
}
