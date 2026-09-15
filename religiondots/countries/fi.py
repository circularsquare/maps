# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _fi_counts():
    """Finland at NUTS 3: 19 maakunnat, from two halves of one census table.

    Finland's population register records religious-community membership for everybody and
    Statistics Finland publishes it with NO GEOGRAPHY AT ALL. All twelve databases at
    pxdata.stat.fi were walked in full on 2026-09-08 — 5,634 nodes, nine search terms — and
    `vaerak/11rx.px` is the only religion table there is: 25 named communities crossed with
    age, sex and year and nothing else. So the register is not the source here, and the two
    populations are Greece's (§9z) and Italy's (§9bp):

      * **Finnish citizens, 5.25M.** ESS rounds 5 to 11 pooled and restricted to
        `ctzcntr = Yes` — 12,741 people over 19 maakunnat, a median of 455 each, and
        `region` is NUTS 3 in every one of the seven rounds rather than dropping a level the
        way Italy's does.
      * **Foreign residents, 271k.** Eurostat's 2021 census table `cens_21ctz_r3`, 200 named
        citizenships at NUTS 3, crossed with Pew's composition for each origin country.

    **Both come out of the same census table**, which publishes `NAT` and `FOR` next to the
    named citizenships, so the halves partition the country by construction. 99.6% is drawn.

    **Nothing here is authored.** Unlike Greece there is no minority the survey cannot see
    that a published figure could put back: the one Finland has, the Conservative Laestadian
    revival, is inside the Lutheran church and inside the Lutheran answer, and no source
    states its geography. taxonomy/fi2024.py says so rather than estimating it.
    """
    from fi2024 import resolve

    cit = pd.read_csv(HERE / "data" / "normalized" / "fi.csv", dtype={"geo_id": str},
                      keep_default_na=False, na_values=[""])
    cit = cit[cit["geo_level"] == "nuts3"].copy()
    cit["node"] = cit["source_category"].map(resolve)
    unmapped = sorted(set(cit.loc[cit["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"fi.csv has unmapped source categories: {unmapped}")

    ext = pd.read_csv(HERE / "data" / "normalized" / "fi_foreign.csv",
                      dtype={"geo_id": str})
    ext = ext[ext["geo_level"] == "nuts3"]

    df = pd.concat([cit[["geo_id", "node", "count"]], ext[["geo_id", "node", "count"]]],
                   ignore_index=True)
    df["congregations"] = 0
    # A survey is not a count of anybody and a nationality model is not either, so nothing
    # here is `measured` and §7 desaturates all of it.
    df["tier"] = "modelled"
    return df.rename(columns={"geo_id": "unit"})[
        ["unit", "node", "count", "congregations", "tier"]]


def _fi_place_weight(place):
    """Finland's 310 municipalities, weighted by municipal population.

    The same weighter Greece uses. It matters less here than in Greece or France because the
    counting units are already 291,000 people each rather than 750,000 or 3.1 million, but it
    is still what stops Lappi being one flat blob the size of Portugal.

    **It is a population weight and not a religion one, and in Finland that costs one
    nameable thing.** Helsinki-Uusimaa is drawn as a single composition over 1.69M people, so
    its 3.12% Muslim share spreads in proportion to where anyone lives; the real geography,
    eastern Helsinki and Vantaa against the western commuter belt, is invisible and no source
    on this map can supply it. Statistics Finland does publish foreign citizens by
    municipality (`vaerak/11rq`), which would let Finland use the Italy weighter and place
    the foreign half where foreign residents actually are; that is a named improvement rather
    than a thing this build does. See sources/fi.md §5.
    """
    if "pop" not in place.columns:
        print("  !! fi_lau.gpkg has no `pop` column — run sources/fi_geo.py")
        return None
    return _GrLauWeighter(place)


ENTRY = {
    "fi": dict(
        name="Finland",
        name_in="Finland",
        source="ESS rounds 5-11 (citizens) + Eurostat census 2021 x Pew 2020 (residents)",
        basis="self-identification, sample survey (citizens); nationality-derived (residents)",
        note_public=(
            "**Finland counts religion twice and gets two different answers, and this map "
            "draws the second one.** The population register records membership of a "
            "registered religious community for every resident, because that is a matter of "
            "civil records, and at the end of 2024 it put **62.24%** of the country in the "
            "Evangelical Lutheran Church. The European Social Survey asks people whether "
            "they consider themselves as belonging to any particular religion, and gets "
            "**45.00%**. Neither number is a mistake. The difference is the people who have "
            "never filed the form to resign from the church and also do not describe "
            "themselves as belonging to it, and there are something like a million of them. "
            "**The register would have made a better map and Statistics Finland does not "
            "publish one.** Its religion table names 25 religious communities and has no "
            "geography at all, not even a province; every one of the twelve statistical "
            "databases at "
            "pxdata.stat.fi was searched, 5,634 tables, and there is no second table. "
            "Germany is drawn from exactly this kind of register down to the municipality; "
            "Finland publishes the same quantity and withholds the map, so the survey is "
            "what is left. "
            "**So this is 12,741 Finnish citizens interviewed between 2010 and 2023, plus a "
            "census count of the 271,000 foreign residents.** The two halves come out of the "
            "same Eurostat census table, which is why they add up to the country rather than "
            "having to be reconciled, and 99.6% of Finland is drawn. **No religion is the "
            "largest single answer at 46.94%**, just ahead of the Lutheran 45.00%, and those "
            "two together are 92% of everybody. "
            "**The north-south reading most people expect is not the one the data gives.** "
            "Measured as a share of each region's own drawn population, the most Lutheran "
            "maakunta is Etela-Pohjanmaa on the west coast at **59.54%** and the least is "
            "Helsinki-Uusimaa at **36.75%**; the most unaffiliated is Lappi at **53.06%** and "
            "the least is Etela-Pohjanmaa at **32.89%**. So the axis is the Ostrobothnian "
            "revivalist belt against the capital and the far north, rather than a simple "
            "gradient up the country. "
            "**North Karelia is Orthodox and the survey finds it without being told.** "
            "Pohjois-Karjala comes out **6.22%** Orthodox against 1.86% nationally, which is "
            "the highest of the nineteen regions and is the part of Karelia that stayed "
            "Finnish in 1944. That is a real historical geography recovered from about 400 "
            "respondents, and it is the strongest evidence here that the instrument is "
            "working. "
            "**On Islam the register is the weaker of the two instruments, not the "
            "stronger.** It records 0.48% of Finland as Muslim, because it can only see "
            "people who have joined a registered Islamic congregation and most Finnish "
            "Muslims never do. This map gets **1.67%**, from the survey for citizens and from "
            "the census's own count of Iraqi, Afghan, Syrian and Somali residents for the "
            "rest, and puts the highest share in Helsinki-Uusimaa at **3.12%**. The same "
            "thing happens to Orthodoxy, at 1.86% here against the register's 1.03%, because "
            "a Russian or Estonian resident who is Orthodox need not belong to a Finnish "
            "parish. "
            "**What this cannot do.** The regions are large and the sample inside them is "
            "not: a median of 455 respondents per maakunta, 3,406 in Helsinki-Uusimaa and 64 "
            "in Aland, so read the Lutheran and unaffiliated shares and treat anything below "
            "one percent as 'some, here' rather than as a number. The survey offers no "
            "atheist or agnostic option, so everyone reporting no religion lands in one "
            "category and Finland puts nothing on the secular node. And Conservative "
            "Laestadianism, the revival movement inside the national church whose strongholds "
            "are Pohjois-Pohjanmaa and Lappi, cannot be drawn at all: its members "
            "are members of the national church and answer the same way as everyone else in "
            "it, and estimating its geography from the movement's own figures would be "
            "inventing one."),
        how="survey, 12,741 people; foreign residents by nationality",
        grain="regions, 291,000 people on average",
        counts=_fi_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "fi" / "fi_lau.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_fi_place_weight,
        note="**§11k closed Finland on reachability and the closure was right about the "
             "register and wrong about the country.** Its finding was that StatFin's "
             "`vaerak/11rx.px` has no geography, and that four Nordic "
             "states fail the same way. That is confirmed here in the strongest available "
             "form: all twelve top-level databases at `pxdata.stat.fi` were walked in full "
             "on 2026-09-08, **5,634 nodes against nine search terms** including `kirkko`, "
             "`seurakun`, `luteril` and `ortodoks`, and the only other matches in the whole "
             "system are parish payroll and a leisure survey's 'read religious or devotional "
             "books'. `StatFin_Passiivi`, the archive of discontinued tables, has nothing "
             "either. **What the sweep did not test is the tier**, and Finland is in every "
             "ESS round with `region` at NUTS 3, which nobody had looked at because the "
             "register closure made the country look finished. §11ai named the deferred ESS "
             "route for Norway, Sweden, Denmark and five others and did not name Finland, "
             "for that reason. "
             "**`11rx.px` is the id and `statfin_vaerak_pxt_11rx.px` is not.** The second is "
             "what the PxWeb web UI puts in its own URL and it returns 400 from the API; the "
             "listing's `id` field is the only thing that works. Worth knowing before "
             "concluding a Finnish table is gone. "
             "**The denomination variable is `rlgdnafi`. `rlgdnfi` does not exist in any "
             "round.** §11ai's warning was that `a`-suffixed revisions silently drop rounds "
             "when you pool on the bare name; here the bare name drops everything and raises "
             "`E201VariableNotFound`, which is the loud version of the same trap. It is worth "
             "chasing rather than falling back to `rlgdnm`: the harmonised variable has seven "
             "families and would put 45% of Finland on `Protestant`, where `rlgdnafi` names "
             "the Evangelical Lutheran Church, the Orthodox, Pentecostals, the Free Church, "
             "Adventists, Jehovah's Witnesses and Mormons separately. "
             "**ESS's `table.path` is a list of INDICES into `values`, not a list of code "
             "values.** Reading it as codes returns zero for every category whose code is not "
             "also a valid index, which for Finland is everything from `10` up: Islam, both "
             "Other Christian buckets and both non-Christian ones. The map would have come "
             "out with no Muslims and no error. gr.py indexes correctly and never said why. "
             "**Three NUTS vintages in seven rounds, and the recode is asserted against ESS's "
             "own labels rather than trusted.** Round 5 is NUTS 2006, rounds 6 to 10 are NUTS "
             "2013 and round 11 is NUTS 2021, so `Kainuu` is `FI134`, `FI1D4` and `FI1D8` in "
             "the same pooled file. A recode table is a name join wearing a code's clothes, "
             "so every pair must carry the same ESS label on both sides. **The check earned "
             "itself immediately**, refusing `FI181 Uusimaa -> FI1B1 Helsinki-Uusimaa`: "
             "Ita-Uusimaa was abolished into Uusimaa in 2011 and NUTS renamed the enlarged "
             "region, so two of the sixteen pairs are genuine and are exempted by name with "
             "the reason. Territorially `FI181 + FI182 = FI1B1` and pooling loses nobody. "
             "**The boundaries cost nothing.** The GISCO LAU 2021 bundle on disk since Poland "
             "carries all 310 Finnish municipalities with their NUTS 3 code and their "
             "population, so the join is 310 of 310 in both directions with no download at "
             "all. Portugal was the other one.",
    ),
}
