# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _hk_place_weight(place):
    """countries.py hook. `place` is the Kontur hex layer scatter.py has read.

    Three quarters of Hong Kong is country park, steep hillside and reservoir, so weighting
    by area would scatter Islands district across 176 km2 of which the people occupy a few.
    Kontur undercounts the high-rise districts and overcounts the rural ones, which moves
    dots within a district and never between them (sources/hk_grid.py).
    """
    return _kontur_place_weight(place, "hk_hexes.gpkg", "sources/hk_grid.py")


def _hk_counts():
    """Hong Kong: census ethnicity at 18 districts, plus one survey for the territory.

    THE SECOND COUNTRY ON THIS MAP WHOSE CENSUS HAS NEVER ASKED ABOUT RELIGION, and it is
    next door to the first. The 2021 Population Census publishes its 46 topics and religion
    is not among them; Hong Kong is absent from UNSD table 28. So this is built the way
    China is, in the same two layers and in the same order, and `taxonomy/hk2021.py` argues
    every category.

      * **the ethnic derivation, at the 18 District Council districts.** Three nationalities
        carry a fractional share of one religion each: Indonesians and Pakistanis to `islam`,
        Filipinos to `christianity.catholic.latin`, the remainder of each to `unknown`, both
        halves `modelled`. This is spec §14.5 as §14.9 amended it.
      * **the survey, for the whole territory, carved out of what is left.** The Hong Kong
        Political Culture Survey 2021 (Cai and Hung, China Quarterly 259, Table 1; 3,744
        respondents aged 16+, 72 clusters) asks which religion a person belongs to, so this
        is §3.1's `self_id`, the same basis as the mainland's CGSS layer.

    THE COEFFICIENTS ARE THIS MAP'S OWN COUNTRIES AND NOT A NEW SOURCE. Indonesia, Pakistan
    and the Philippines are each drawn here from their own censuses, so Hong Kong's
    Indonesians get the Muslim share of Indonesia as this project already draws it: 87.51%,
    96.47% and 78.88%. §14.5 wants a coefficient documented rather than fitted, and this is
    the strongest form of documented available -- it cannot drift away from the rest of the
    map, because it IS the rest of the map.

    ONLY THE `unknown` RESIDUAL IS CARVED, which is `_cn_counts`'s arithmetic exactly and for
    the same reason: applying the survey's shares to the whole population would count an
    Indonesian domestic worker as Muslim once from the census and again from the survey. The
    cost runs the other way and is small, because the derived population is 4.1% of Hong Kong.

    ISLAM IS NOT TAKEN FROM THE SURVEY, and that is the one substantive choice here. The
    survey has 89 Muslim respondents in a 72-cluster design and no usable geography; the
    census counts 142,065 Indonesians and 24,385 Pakistanis exactly and says which district
    each lives in. The two agree on the size to within 16% -- 147,845 against 176,431 -- which
    is what makes either believable, and only one of them can put them in Yau Tsim Mong and
    Yuen Long. China refuses CGSS's Islam for the same reason.

    THE GOVERNMENT'S OWN FIGURES ARE DELIBERATELY NOT DRAWN. gov.hk's *Hong Kong: The Facts*
    gives over a million Buddhists, over a million Taoists, 1,040,000 Protestants and 300,000
    Muslims, and every one is the religious body's own claim. The same office said 480,000
    Protestants in July 2022 and 1,040,000 in January 2026, while the churches' own eighth
    Hong Kong Church Survey counted 197,935 weekly worshippers, down 26% in five years -- the
    figure doubled while the only measurement fell by a quarter. Its Muslim and Hindu numbers
    are each about twice what the census and the survey independently agree on. sources/hk.md.

    65.83% OF HONG KONG NAMED NO RELIGION AND IT IS NOT DRAWN AS IRRELIGION. The same table
    says why: 2,097 of those 2,462 respondents -- 56.07% of the whole sample -- report
    practising folk religion anyway. Until 2026-09-15 all of it stayed `unknown`, because §3.1
    forbade mixing practice with the naming layer. Spec §3.13 relaxed that for `chinesefolk` in
    China, Taiwan and Hong Kong: a religious altar at home among people who name no religion
    counts. The survey has no altar item, so Pew 2023's 21% of the unaffiliated is carved from
    the `Chinese` row's residual below (hk2021.ALTAR_FOLK_SHARE, ALTAR_FOLK_ROWS), and 52.3% of
    Hong Kong stays grey.
    """
    from hk2021 import shares, SURVEY_NODES

    df = pd.read_csv(HERE / "data" / "normalized" / "hk.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[(df["geo_level"] == "district") & (df["count"] > 0)]

    parts = []
    for cat, sub in df.groupby("source_category", sort=False):
        for node, share, tier in shares(cat):
            if share <= 0:
                continue
            parts.append(pd.DataFrame({
                "unit": sub["geo_id"].to_numpy(),
                "cat": cat,
                "node": node,
                "count": sub["count"].to_numpy(dtype=float) * share,
                "tier": tier,
            }))
    out = pd.concat(parts, ignore_index=True)

    # ---- the survey layer, carved out of `unknown` only ------------------------------
    sv = pd.read_csv(HERE / "data" / "normalized" / "hk_survey.csv")
    sv = sv[sv["category"].isin(SURVEY_NODES)]
    total_share = float(sv["share"].sum())
    if not 0.0 < total_share < 1.0:
        raise ValueError(f"hk_survey shares sum to {total_share}, which cannot be carved")

    unk = out["node"] == "unknown"
    scale = pd.Series(1.0, index=out.index)
    scale[unk] = 1.0 - total_share
    carved = [out.assign(count=out["count"] * scale)]
    for _, r in sv.iterrows():
        part = out.loc[unk].copy()
        part["count"] = part["count"] * float(r["share"])
        part["node"] = SURVEY_NODES[r["category"]]
        part["tier"] = "modelled"      # the coefficient is the survey's, not the census's
        carved.append(part)
    out = pd.concat(carved, ignore_index=True)

    # ---- spec §3.13: a religious home altar among the people left grey --------------------
    # Anita, 2026-09-15. After the carve the `unknown` residual stands for the people who named
    # no religion, and Pew finds 21% of Hong Kong's unaffiliated keep an altar at home; that
    # share of the residual is drawn on `chinesefolk`, territory-wide like the survey layer.
    # ONLY THE `Chinese` ROW'S RESIDUAL (ALTAR_FOLK_ROWS), 2026-09-15, session cb8b206e-folkfix:
    # applied to every row it drew about 44,800 non-Chinese residents as Chinese folk religion,
    # 10,400 of them the Indians and Nepalese the note says are left grey. sources/hk.md §10.
    from hk2021 import ALTAR_FOLK_NODE, ALTAR_FOLK_SHARE, ALTAR_FOLK_ROWS
    unk = (out["node"] == "unknown") & out["cat"].isin(ALTAR_FOLK_ROWS)
    folk = out.loc[unk].copy()
    folk["count"] = folk["count"] * ALTAR_FOLK_SHARE
    folk["node"] = ALTAR_FOLK_NODE
    folk["tier"] = "modelled"
    out.loc[unk, "count"] = out.loc[unk, "count"] * (1.0 - ALTAR_FOLK_SHARE)
    out = pd.concat([out, folk], ignore_index=True)

    out = out.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()
    out = out[out["count"] > 0]
    out["congregations"] = 0
    return out[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "hk": dict(
        name="Hong Kong",
        source=("2021 Population Census ethnicity by district (C&SD); religion shares from "
                "the Hong Kong Political Culture Survey 2021; home altars from Pew Research "
                "Center's 2023 survey of East Asian societies"),
        basis=("ethnicity, derived for three migrant nationalities; plus self-identified "
               "religion from one territory-wide survey; folk religion also counts a home altar "
               "among Chinese residents who name no religion, from a second survey"),
        view=[113.80, 22.13, 114.52, 22.58],
        note_public=(
            "**Hong Kong has never asked anybody here what their religion is either.** Its "
            "2021 census publishes the 46 things it asked about and religion is not one of "
            "them, so like mainland China next door there is no census answer to draw, and "
            "this map is built the same way in the same two layers. "
            "**The figures you will find everywhere else are not used here, and it is worth "
            "saying why.** The government's own fact sheet gives over a million Buddhists, "
            "over a million Taoists, 1,040,000 Protestants, 390,000 Catholics, 300,000 "
            "Muslims and 100,000 Hindus. Every one of those is supplied by the religious "
            "body itself, and the set does not survive the only outside check there is. The "
            "same office put Protestants at 480,000 in 2022 and at 1,040,000 in 2026, while "
            "over the same years Hong Kong's own churches counted themselves and found "
            "**197,935 people at worship, down by a quarter in five years**. The figure "
            "doubled while the only measurement of it fell. "
            "**What is drawn instead is what people said about themselves.** A survey of "
            "3,744 Hong Kong residents in 2021 asked which religion they belong to: 13.7% "
            "said Buddhism, 9.1% Protestant Christianity, 4.3% Catholicism, 4.0% Taoism, "
            "2.4% Islam, and about two thirds said none. Those shares are for the whole "
            "territory, so **Buddhism and Christianity do not vary between districts on this "
            "map**. The survey cannot say whether Sha Tin differs from Wan Chai, and rather "
            "than invent a difference this map draws none. "
            "**The one thing that does vary by district is drawn from the census, and it is "
            "the migrant communities.** Hong Kong counts 142,065 Indonesians, 201,291 "
            "Filipinos and 24,385 Pakistanis and says which district each lives in. "
            "Indonesia is 87% Muslim, Pakistan 96%, the Philippines 79% Roman Catholic, all "
            "three taken from those countries' own censuses as this map already draws them, "
            "so those shares are carried across. That is where Hong Kong's Muslim dots come "
            "from. "
            "**And the geography that produces is not the one you would expect.** The Muslim "
            "share runs from 3.0% in Wan Chai down to 1.5% in Kwun Tong, and the Catholic "
            "share from 11.3% to 5.0%, with both highest in the wealthiest districts on Hong "
            "Kong Island. There is no enclave here. Most of the people involved are live-in "
            "domestic workers, so what this layer draws is not where a community settled but "
            "where the households that employ them are. "
            "**Two independent sources agree about the size of that community, and the "
            "official figure does not.** The census's ethnic counts give about 148,000 "
            "Muslims and the survey about 176,000; the government's sheet says 300,000. For "
            "Hindus the census's upper bound is about 72,000 and the survey says 44,000, "
            "against an official 100,000. When a census and a survey with nothing in common "
            "agree with each other and disagree with a third number, the third number is the "
            "one to leave out. "
            "**About one in eight people in Hong Kong is drawn as Chinese folk religion, from "
            "home altars.** Mainland China and Taiwan are drawn on one rule: people who name folk "
            "religion, and people who name no religion but keep a religious shrine or altar "
            "at home. This survey has no altar question, so Pew's 2023 survey supplies it: "
            "**21%** of Hong Kong people with no religion say there is an altar in their "
            "home, and that share of the grey among Chinese residents is drawn as folk "
            "religion, **13.1%** of Hong Kong and the same in every district. Nobody outside "
            "the Chinese population is counted this way. Pew's question is narrower than the one "
            "asked in China and Taiwan (in Taiwan the two get 49% and 71%), so read it as a "
            "floor. "
            "**About half of Hong Kong is still grey, and it is not drawn as irreligion.** "
            "The same survey asked the people who said they had no religion what they "
            "actually do, and **56% of everyone surveyed practises folk religion** in some "
            "way: grave-tending at Ching Ming, incense at Wong Tai Sin, a fortune stick, a "
            "date chosen for a wedding. Most of them keep no altar, and they are counted and "
            "placed with nothing claimed about them. "
            "**The Indians and Nepalese are counted and left grey on purpose.** They are the "
            "obvious next candidates, 42,569 and 29,701 people, but Hong Kong's Indian "
            "community is disproportionately Sindhi Hindu and Punjabi Sikh rather than a "
            "cross-section of India, and its Nepalese are the families of Gurkha soldiers, "
            "recruited from hill peoples far more Buddhist than Nepal's average. A national "
            "share laid over a community selected on exactly that axis is a guess wearing a "
            "citation. The Hindus and Sikhs drawn here come from the survey instead, where "
            "they rest on 22 and 2 respondents and should be read as *this community exists "
            "and is small*, not as a count. "
            "**Where the dots sit inside a district is modelled and Hong Kong is the hardest "
            "place on earth for it.** Placement uses a global population grid built from "
            "building footprints, which reads a 40-storey housing estate much as it reads a "
            "village. It undercounts Wong Tai Sin and Sham Shui Po and overcounts the rural "
            "north, so within a district the dots drift a little away from the tower estates. "
            "It never changes how many dots a district gets, only which street they are on."),
        how="no census question; ethnicity for three migrant nationalities, two surveys for "
            "everyone else",
        fill="from the 2021 census's ethnicity table",
        grain=("18 District Council districts, 412,000 people on average; the survey layer is "
               "territory-wide, so most religions do not vary between districts"),
        gap=("a religion for 52% of these dots; with no census question they say only that "
             "somebody was counted"),
        counts=_hk_counts,
        # The hexes carry the district letter and there is no separate unit layer, which is
        # China's and Tonga's wiring. sources/hk_geo.py labels them.
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "hk" / "hk_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_hk_place_weight,
        note="**NO CENSUS HERE HAS EVER ASKED ABOUT RELIGION**: the 2021 Population Census "
             "lists its 46 topics and religion is absent, and Hong Kong does not appear in "
             "UNSD table 28. Built 2026-09-08 as China's sibling, spec §14.24, in the same "
             "two layers: §14.5's ethnic derivation at 18 District Council districts, and a "
             "`self_id` survey for the territory carved out of the `unknown` residual, which "
             "is `_cn_counts`'s arithmetic exactly. Nothing is `measured`. "
             "**The three coefficients are this map's own countries and not a new source**: "
             "Indonesia 87.51% `islam`, Pakistan 96.47% `islam`, the Philippines 78.88% "
             "`christianity.catholic.latin`, each recomputable from that country's entry "
             "here. §14.5 wants a coefficient documented rather than fitted and this is the "
             "strongest form available, because it cannot drift away from the rest of the map. "
             "**Islam is taken from the census and NOT from the survey**, which is China's "
             "rule for the same category: 89 Muslim respondents in a 72-cluster design carry "
             "no geography, while the census counts 142,065 Indonesians and 24,385 Pakistanis "
             "by district. The two agree on the magnitude to within 16%. "
             "**gov.hk's fact sheet is refused and the refusal is evidenced.** Its figures are "
             "each the religious body's own claim; it says 1,040,000 Protestants in January "
             "2026 where the same office said 480,000 in July 2022 and where the 2024 Hong "
             "Kong Church Survey counted 255,091 congregants and 197,935 weekly worshippers, "
             "down 26% in five years. Its Muslim and Hindu figures are each about twice what "
             "the census and the survey independently agree on. "
             "**Indian and Nepalese are refused as §14.12 cases**: a national coefficient "
             "over a migration stream selected on exactly the axis it would need to be stable "
             "on; taxonomy/hk2021.py has the argument, and Thai is the close call in REVIEW. "
             "The survey's `Hinduism` and `Sikhism` rest on 22 and 2 respondents and are drawn "
             "at territory grain with no geographic claim, which is Guatemala's rule (§9bi): "
             "nobody is deleted, only the claim to know where they are. "
             "Table 8.1 of the census's *Thematic Report: Ethnic Minorities* is parsed from "
             "the PDF and checked two ways, neither a tolerance: its Filipino and Indonesian "
             "columns must reproduce DC_21C.CSV's exact counts (worst 0.05 pp over 36 "
             "comparisons), and its South Asian subtotal must equal the weighted mean of its "
             "own four South Asian columns (worst 0.07 pp over 18 districts). "
             "Placement is Kontur's 2023 grid, 1,294 hexes; it correlates with the census at "
             "r = 0.87 over the 18 districts with a 0.60-1.61 band, which is a modelling "
             "difference and not a join failure. It undercounts the high-rise districts and "
             "overcounts the rural ones, and it moves dots within a district and never "
             "between them. **1,125 people are not drawn**: the marine population, which the "
             "census counts and assigns to no district (spec §3.5). "
             "**THE KNOWN WEAKNESS IS SPATIAL AND IT IS THE THING TO FIX NEXT.** 31.8% of "
             "every district is the territory-wide survey and does not vary at all; the "
             "derived layer varies only between 1.51% and 3.00% Muslim; so the 18 districts "
             "do almost no work and this reads as a national pie chart with a "
             "population-weighted scatter. §3.9b removed the granularity floor so it is drawn "
             "rather than skipped, but coarse is a fact to state and not a resting place. "
             "Routes in order: a survey with district-level religion (HKPSSD or the Asian "
             "Barometer, both gated, so §11b's order applies), the 2024 Hong Kong Church "
             "Survey's district tables, the Catholic diocese's parish statistics, and only "
             "alongside those the 452 constituency areas on data.gov.hk. sources/hk.md §8.",
    ),
}
