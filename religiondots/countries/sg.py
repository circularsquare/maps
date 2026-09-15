# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


class _SgSubzoneWeighter:
    """Split a planning area's dots across its URA subzones by CENSUS RESIDENT POPULATION.

    Not Kontur, and Singapore is the one country here where that is a real improvement
    rather than a preference. The weight comes from *Resident Population by Planning
    Area/Subzone of Residence* — the same 2020 census, the same office and the same
    universe-defining word, `residents`, as the religion table it is weighting.

    A modelled surface would be wrong in a knowable direction here. Kontur counts everyone
    physically present, and about 1.64 million people in Singapore are non-residents the
    census does not ask about religion at all, many of them in worker dormitories at places
    like Tuas and Sungei Kadut, where the RESIDENT population is 70 and 750 people. Weighting residents by
    a surface that is largely non-residents would push dots into exactly the places this
    census did not count.

    It is a POPULATION weight and not a religion one: SingStat publishes religion at
    planning area and nothing finer, so inside Bedok a Muslim dot and a Buddhist dot are
    spread the same way. Read a cluster as "this planning area, drawn where its residents
    live", never as a neighbourhood reading. sources/sg_geo.py builds the layer.
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
        return (f"{self.n_pop:,} (unit, node) rows placed on census 2020 resident population "
                f"per subzone, {self.n_uniform:,} on equal shares where a planning area's "
                "subzones sum to zero (sources/sg_geo.py)")


def _sg_place_weight(place):
    """countries.py hook. `place` is the 332-subzone layer scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! sg_subzones.gpkg has no `pop` column — run sources/sg_geo.py; "
              "placing on equal shares (§8.2)")
        return None
    return _SgSubzoneWeighter(place)


def _sg_counts():
    """SingStat COP 2020: 9 categories on 31 planning-area units, 3.46M people.

    ONE level, no allocation, nothing modelled — SingStat publishes these nine categories at
    this geography and the map draws exactly that, so every row is `measured` and may ring.
    There is no COLUMNS dict because there is nothing derived to roll up (spec §7a-i-1), and
    tools/check_rollup.py should never name `sg`.

    THE 31st UNIT IS A REAL ROW AND NOT A LEFTOVER. The table names 30 planning areas and
    puts the other 25 in one `Others` cell of 25,756 people; sources/sg_geo.py gives that
    cell the union of exactly those 25 areas as its placement geometry. So it is `measured`
    like the rest — the count IS measured, at a unit that happens to be disjoint — and
    nobody is dropped. What it costs is that Rochor, which contains both Little India and
    Kampong Glam, shares one mixture with Tuas and the Southern Islands.

    `Total` resolves to None and is the universe row; the nine categories partition it.
    """
    from sg2020 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "sg.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "planning_area"].copy()
    if df["geo_id"].nunique() != 31:
        raise SystemExit(f"sg: {df['geo_id'].nunique()} units, expected 31 "
                         "(30 named planning areas + Others)")

    df["unit"] = df["geo_id"].astype(str)
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]) - {"Total"})
    if unmapped:
        raise SystemExit(f"sg.csv has unmapped source categories: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "sg": dict(
        name="Singapore",
        source="Census of Population 2020 (Department of Statistics Singapore)",
        basis="self-identification, residents aged 15 or over",
        note_public=(
            "**Singapore counts the religion of three and a half million of the five and a "
            "half million people on the island.** The census asks residents, meaning "
            "citizens and permanent residents, aged 15 and over. That leaves out 585,117 "
            "resident children and **1,641,590 non-residents**, the work permit and "
            "employment pass holders, dependants and students who are 29% of everybody "
            "here. Nobody asks them and no table anywhere gives their religion, so the "
            "dormitories out at Tuas and the domestic workers living in flats across the "
            "island are simply not in these numbers. It is close to a third of the country "
            "missing, and it is the first thing to know about this map. "
            "**The map is flatter than almost any other country here, and that is policy "
            "rather than chance.** Since 1989 Singapore has capped the share of each ethnic "
            "group in every public housing block and neighbourhood, and about eight in ten "
            "residents live in that housing. Malays are **98.8%** Muslim by this same "
            "census, so a rule about ethnicity is in practice a rule about religion: the "
            "most Muslim planning area, Woodlands at 28.1%, is under twice the national "
            "15.6%. Nothing on this map's Muslim geography looks like that anywhere else. "
            "**What the quota does not touch is where the differences show up instead.** "
            "Christians are drawn from all three main ethnic groups, so the housing rule "
            "never constrained them, and Christianity has by far the widest range of "
            "anything here: **44.1% of Bukit Timah against 11.5% of Woodlands**, twelve "
            "kilometres apart. Bukit Timah, Tanglin and River Valley are the private "
            "housing districts, and they are also where no religion runs near 30% against "
            "a national 20%, and where Taoism falls to between 2 and 5% against 8.8%. The "
            "strongest "
            "religious geography Singapore has left is not the ethnic one; it is the line "
            "between public and private housing. "
            "**Taoism here is broader than the word, and the census says so itself.** "
            "SingStat footnotes the category as including Chinese traditional beliefs, so "
            "the 303,960 people in it are ancestor veneration, the deity temples and the "
            "seventh month as much as the Daoist canon. This map files them as Chinese "
            "religions for that reason. Hong Kong, drawn from a survey that offered Taoism "
            "with no such gloss, is filed as Daoism, so the two are deliberately different "
            "colours and the difference is in the questionnaires rather than in the two "
            "populations. "
            "**Christianity is split only two ways, and the smaller half is the named "
            "one.** 242,681 Catholics and 411,674 everyone else, so Catholics are 37% of "
            "Singapore's Christians; the Methodists, Anglicans, Presbyterians, Brethren, "
            "Baptists and the large independent churches all sit in one undivided cell "
            "because the census offers no box to separate them. So this map can show you "
            "where the Catholics are and cannot show you that the Protestants outnumber "
            "them. "
            "**Thirty planning areas are named and the other twenty-five share a single "
            "row between them.** Those twenty-five hold 25,756 residents in total and "
            "several have nobody living in them at all, so grouping them is reasonable. It "
            "costs one real thing: Rochor is in there, and Rochor contains both Little "
            "India and Kampong Glam. The two districts a visitor would go to looking for "
            "Hindu and Muslim Singapore are drawn with one averaged mixture spread over an "
            "area that also takes in Tuas and the Southern Islands. "
            "**Where a dot sits inside a planning area comes from the same census.** "
            "SingStat publishes resident population for all 332 subzones, so dots follow "
            "where residents actually live rather than an even wash across a polygon. That "
            "matters more here than it sounds: the global population grid this map uses "
            "elsewhere counts everybody present, and in Singapore that would have put "
            "resident dots in the worker dormitories."),
        how="census, 2020, ages 15 and over",
        grain=("30 planning areas plus one row holding the other 25; 112,000 people on "
               "average"),
        gap_share=0.39,
        gap=("non-residents and residents under 15, together 39% of the people in "
             "Singapore"),
        counts=_sg_counts,
        # The subzones carry the planning area's name and there is no separate unit layer,
        # which is Hong Kong's and Tonga's wiring. sources/sg_geo.py labels them.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "sg" / "sg_subzones.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_sg_place_weight,
        note="9 categories on 31 units, 3,459,094 people, every row `measured` and nothing "
             "allocated: SingStat publishes religion at planning area and this draws exactly "
             "that. Table CT/17592 via data.gov.sg, Open Data Licence, no key. "
             "**The universe is residents aged 15+ and is NOT scaled up**, which is Chile's "
             "rule (sources/cl.md 3) applied twice over: 585,117 resident children and "
             "1,641,590 non-residents are outside it, so the map draws 60.8% of the people "
             "in Singapore. Religiosity varies sharply by age here (24.2% of 15-24s report "
             "no religion against 15.2% of over-55s), so a flat scale-up would be an "
             "assertion about children the census declined to make; the non-residents are "
             "not tabulated for religion by anybody. "
             "**The 31st unit is `Others`, a real measured row** covering the 25 planning "
             "areas the table does not name, 25,756 people, drawn on the union of exactly "
             "those 25 polygons. Rochor is in it, so Little India and Kampong Glam share a "
             "mixture with Tuas. "
             "**Placement is the census's own subzone populations, not Kontur** "
             "(sources/sg_geo.py) and Singapore is the one country where that is strictly "
             "better rather than a preference: same census, same office, same `residents` "
             "universe. A footprint-derived grid counts the 1.64M non-residents and would "
             "put resident dots in the Tuas dormitories. Cost: the weight is all ages "
             "against a 15+ count, so the 15+ share runs 0.76 to 0.89 across planning areas "
             "and moves dots inside a unit by a couple of per cent, never between units. "
             "**Two mapping calls are argued in taxonomy/sg2020.py's REVIEW**: `Taoism`, "
             "which the source footnotes as including Chinese Traditional Beliefs, goes to "
             "`chinesefolk` and not `daoism` (Hong Kong's survey answer goes the other way, "
             "on purpose); and `Other Christians`, 411,674 people, goes to the `christianity` "
             "ROOT rather than `christianity.protestant`, which is lk2024.py's call on the "
             "identical Catholic/not-Catholic binary.",
    ),
}
