# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _gr_place_weight(place):
    if "pop" not in place.columns:
        print("  !! gr_lau.gpkg has no `pop` column — run sources/gr_geo.py")
        return None
    return _GrLauWeighter(place)


def _gr_counts():
    """Greece at NUTS 2: 48 nodes on 14 units, from two halves of one census.

    Greece has not asked about religion since 1951. The two populations are Spain's (§9y):

      * **Greek citizens, 9.72M.** ESS rounds 5, 10 and 11 pooled and restricted to
        `ctzcntr = Yes` — 7,885 respondents over 13 regions, about 600 each, which is the
        same order as Russia's federal subjects.
      * **Foreign residents, 759k.** Eurostat's 2021 census table `cens_21ctz_r3`, 200 named
        citizenships at NUTS 3, crossed with Pew's composition for each origin country.

    **Both come out of the same census table**, which publishes `NAT` and `FOR` next to the
    named citizenships, so the halves partition the country by construction rather than by
    reconciliation. 99.8% of Greece is drawn.

    **Two cells are authored and both are in taxonomy/gr2024.py.** The Muslim minority of
    Western Thrace is split out of its region's citizen population, because ESS reaches
    essentially none of it; and Mount Athos, an extra-regio unit of 1,744 monks that no sample
    will ever contain, is drawn Orthodox.
    """
    from gr2024 import resolve

    cit = pd.read_csv(HERE / "data" / "normalized" / "gr.csv", dtype={"geo_id": str},
                      keep_default_na=False, na_values=[""])
    cit = cit[cit["geo_level"] == "nuts2"].copy()
    cit["node"] = cit["source_category"].map(resolve)
    unmapped = sorted(set(cit.loc[cit["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"gr.csv has unmapped source categories: {unmapped}")

    ext = pd.read_csv(HERE / "data" / "normalized" / "gr_foreign.csv",
                      dtype={"geo_id": str})
    ext = ext[ext["geo_level"] == "nuts2"]

    df = pd.concat([cit[["geo_id", "node", "count"]], ext[["geo_id", "node", "count"]]],
                   ignore_index=True)
    df["congregations"] = 0
    # A survey is not a count of anybody, a nationality model is not either, and the two
    # authored cells are assertions — so nothing here is `measured` and §7 desaturates it all.
    df["tier"] = "modelled"
    return df.rename(columns={"geo_id": "unit"})[
        ["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "gr": dict(
        name="Greece",
        name_in="Greece",
        source="ESS rounds 5/10/11 (citizens) + Eurostat census 2021 x Pew 2020 (residents)",
        basis="self-identification, sample survey (citizens); nationality-derived (residents)",
        view=[19.2, 34.7, 28.4, 41.8],
        note_public=(
            "**Greece has not asked about religion in a census since 1951**, and this is the "
            "first map of it that is not simply a national number painted flat. It is built "
            "the way Spain is, from two populations that between them are the whole country: "
            "**9.7 million Greek citizens**, drawn from three pooled rounds of the European "
            "Social Survey, and **759,000 foreign residents**, drawn from the 2021 census's "
            "own count of who lives in each region and where they are from. "
            "**Greece is 84% Greek Orthodox and 7% of people say they belong to no religion**, "
            "which is the second largest answer and is concentrated in Attiki, Peloponnisos "
            "and Thessalia rather than spread evenly. Dytiki Makedonia is the most Orthodox "
            "region at 95%. "
            "**The Muslim minority of Western Thrace is the one thing a survey cannot see, "
            "and it is put back by hand.** 100,000 to 120,000 people in Rodopi, Xanthi and "
            "Evros descend from the minority recognised by the 1923 Treaty of Lausanne. They "
            "are Greek citizens, they are Turkish- and Pomak-speaking, and a Greek-language "
            "national sample of 2,700 people finds almost none of them — nine in 2010 and "
            "none at all in 2020 or 2023. So the published figure is used instead, and it "
            "makes **Anatoliki Makedonia-Thraki 22% Muslim**, against 2-6% everywhere else. "
            "Without it this map would have said the historically Muslim region of Greece was "
            "its least Muslim one. "
            "**The rest of Greece's Islam is an immigration and it is everywhere.** Albanians "
            "are half of all foreign residents; after them come Pakistanis, Bangladeshis, "
            "Afghans, Egyptians and Syrians, and the result is that **every one of the "
            "thirteen regions is at least 2% Muslim** — highest in the Dodecanese and the "
            "Ionian islands, where foreign residents are more than a tenth of the population, "
            "and in Attiki, which holds Greece's largest Muslim population in absolute terms. "
            "**The Catholics of the Cyclades survive in the data.** Notio Aigaio comes out "
            "10% Catholic, against 1% nationally — Syros, Tinos and Naxos, where Latin-rite "
            "communities have been continuous since the Venetian period. That share rests on "
            "a few dozen respondents and should be read as 'a lot, here' rather than as a "
            "number. "
            "**And Mount Athos is drawn as itself.** The monastic republic is a separate "
            "statistical region of 1,744 people that no survey will ever sample; only "
            "Orthodox monks may live there, and it is drawn accordingly rather than from "
            "mainland Macedonia's mixture. "
            "**What the sources cannot do.** The survey offers eight denominations and no "
            "atheist or agnostic option, so everyone who reports no religion lands in one "
            "category and Greece has nothing on the `secular` node at all. The immigrant half "
            "counts people as their country of origin's religion, which is an upper bound: it "
            "cannot see anyone who stopped practising, or converted, after arriving — and for "
            "Albanians, who are documented to have adopted Orthodox identity in Greece in "
            "large numbers, that limitation is doing real work."),
        how="survey, 7,885 people; foreign residents by nationality",
        grain="regions, 750,000 people on average",
        counts=_gr_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "gr" / "gr_lau.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_gr_place_weight,
        note="**Two halves of one census table, which is why they partition exactly.** "
             "Eurostat's `cens_21ctz_r3` — 2021 census, population by citizenship at NUTS 3, "
             "221 citizenships, keyless JSON — publishes `NAT` and `FOR` alongside its named "
             "countries, so the denominator of the citizen half and the numerator of the "
             "foreign half are rows of the same file. 10,482,482 people against ELSTAT's own "
             "census total of 10,482,487, and the 200 named citizenships cover 99.84% of the "
             "foreign population. **It is EU-wide and is the right starting point for any "
             "future EU country; Spain's foreign half should be moved onto it.** "
             "**The ESS API is open and nobody says so.** `ess.sikt.no` is an SPA returning "
             "the same 1,070-byte shell for every path, and its `/env.js` names the backend "
             "in one line: `api.nsd.no/graphql`, which answers ANONYMOUS queries including "
             "server-side cross-tabulation. A country's religion-by-region table is one "
             "request and no microdata moves — which matters, because the portal's own "
             "download flow does require an account. Three gotchas, all in sources/gr.py: "
             "`breakVariables` takes variable NAMES not the UUIDs the search returns; its "
             "type is `[String!]!` and anything looser is rejected; and `region` across all "
             "30 countries trips `E204TooManyCategoriesInVariable`, which `byVariables: "
             "[\"cntry\"]` avoids where every documented form of `subsetJson` returns a bare "
             "422. "
             "**`ctzcntr` is what stops the halves double-counting.** Unlike Spain's CIS, ESS "
             "does sample non-citizens — and badly, and worse over time: 203 of 2,713 "
             "respondents in round 5, 83 of 2,800 in round 10, 87 of 2,757 in round 11, "
             "against a true 7.2%. Restricting to citizens removes the overlap and the "
             "undercount at once. Greece is in rounds 1, 2, 4, 5, 10 and 11 and the first "
             "three have no `region` variable at all, so three rounds pool to 7,885 citizen "
             "respondents. "
             "**Two traps in reading the sources, both of which fail silently.** ESS labels "
             "the Greek regions in Latin in round 5 and in Greek in rounds 10-11, so pooling "
             "on labels rather than codes splits all thirteen regions in two; and the "
             "Eurostat geo dimension holds every NUTS level in one column, so a prefix filter "
             "counts the same people four times and reported Greece at 41.9 million. "
             "**The cross-check is the reason to believe it.** Muslims come out at **5.08% "
             "of Greece** against Pew's own independent 2020 country estimate of **5.12%** — "
             "from Eurostat counts, Pew origin compositions and one minority figure, none of "
             "which is Pew's Greece row. That is also what settles the Albanian coefficient: "
             "Albania is 49% of Greece's foreign residents, Pew puts it at 59% Muslim, the "
             "literature says Albanians in Greece are more Orthodox than that, and excluding "
             "them gives 2.31% — less than half. The undocumented adjustment would have been "
             "the error. "
             "**Boundaries were free and the join was one zero-pad.** GISCO's LAU bundle "
             "ships both the 6,137 Greek polygons and the workbook mapping each to its NUTS "
             "3; Excel had stripped the leading zero from every LAU code in regions 01-09, "
             "which is 644 of them and reads as 'the workbook is missing rows'.",
    ),
}
