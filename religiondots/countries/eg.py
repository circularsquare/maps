# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _eg_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Egypt is the extreme case of §8.2's emptiness argument and the country the grid exists
    for. **About 96% of Egypt is uninhabited desert** and effectively everyone lives on the
    Nile valley and delta, a green ribbon a few kilometres wide for a thousand kilometres. The
    governorates contain that ribbon and then run east and west into nothing: New Valley is
    429,151 km2, 43% of Egypt's land and 0.25% of its people. Drawn flat, the Coptic
    concentration in Minya, Asyut and Sohag would be painted mostly onto the Eastern Desert
    (sources/eg_grid.py).
    """
    return _kontur_place_weight(place, "eg_hexes.gpkg", "sources/eg_grid.py")


def _eg_counts():
    """Arab Barometer waves III, IV, V and VII pooled, at governorate: 2 categories, 24 of 27;
    the Christians split Coptic Orthodox / other from the Global Flourishing Study (below).

    THE COUNTRY IS A SURVEY ON CAPMAS'S OWN PROJECTION AND EVERY ROW IS `modelled` (§7b).
    `Q1012` gives a governorate share; CAPMAS's api/GovernoratePopulation gives the number of
    people it applies to, pinned to 2026-01-01. No magnitude is invented: every person drawn
    is a person CAPMAS counts in that governorate, and the survey only decides the column
    (§14.4 rule 1, the construction sources/kz.py uses).

    DRAWING EGYPT AT GOVERNORATE IS ANITA'S CALL, 2026-09-08, on ask/answered/001-eg. §14.4
    rule 2 is about exactly this case -- a persecuted minority mapped finer than its own state
    publishes -- and Egypt collected religion in 1986, 1996, 2006 and 2017 and published it
    from 1986 only. Her ruling was to draw at governorate, on the ground that governorates are
    pretty big. She also asked whether anything finer was possible and it is not: the survey
    cuts by governorate and carries nothing below it, and the microdata extract CAPMAS
    deposited for the 2017 census has thirteen variables and religion is not among them. So
    governorate is simultaneously the ruling and the ceiling.

    THREE GOVERNORATES HAVE NO RESPONDENTS IN ANY WAVE: New Valley, North Sinai and South
    Sinai. Filling them at the national rate would assert that the Western Desert oases and the
    Sinai look like Egypt's average, which nothing supports. Two of them were measured by the
    census, though, and are drawn from it (scout 2026-10-03, sources/eg_census.py, on Anita's
    "whether anything measured the place" line of 2026-09-08): South Sinai at the 1996 census's
    5.03% Christian, from the governorate table Arab-West Report copied in the CAPMAS library,
    and New Valley at the 1976 census's 1.8%, from Chitam (1986) Table 4.1. North Sinai has no
    census figure anyone has printed and stays empty: 475,331 people, 0.44% of Egypt; `gap=`.

    BOTH ANSWERS CARRY THEIR OWN GEOGRAPHY AND THERE IS NO TAIL, which is unique here. The
    card has two boxes that anybody in Egypt chose, both clear the 1% eligibility floor, and
    both clear the split-half bar (+0.52 weighted against +0.42 on the 23 governorates present
    in both wave halves). So the governorate shares are a closed partition and nothing is
    spread at a national rate.

    THE WEIGHTED AND UNWEIGHTED READINGS DISAGREE ABOUT WHICH GOVERNORATE IS TOP, and the file
    draws the weighted one. Unweighted, Sohag is 16.9% and Minya 16.1%; weighted, which is
    what sources/eg.py applies, Minya is 16.41% and Sohag 13.90%. The weights are the survey's
    own and they are what makes the national reading agree with the census (5.93% weighted
    against 6.68% unweighted, and the 1986 census published 5.7%), so they are used. The three
    top governorates are within each other's 95% intervals either way, and note_public says to
    read them as a group.

    THE UNIVERSE IS ADULTS AND THE DOTS ARE EVERYBODY. The Arab Barometer interviews people
    aged 18 and over in every wave; the shares are applied to the whole population, which
    assumes Egypt's children are distributed like its adults. Drawing only the adults would
    leave a third of a young country blank, and §6.12 is about how badly a blank reads on a
    dot map.
    """
    from eg2022 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "eg.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    unsampled = {"EG32", "EG34", "EG35"}
    if unsampled & set(df["geo_id"]):
        raise SystemExit(f"{sorted(unsampled & set(df['geo_id']))} is in eg.csv and must not "
                         "be -- the Arab Barometer never sampled it; re-run sources/eg.py")
    # New Valley (1976 census) and South Sinai (1996 census), the two frontier governorates a
    # census measured; scout 2026-10-03, sources/eg_census.py. North Sinai stays empty.
    census = pd.read_csv(HERE / "data" / "normalized" / "eg_census.csv",
                         dtype={"geo_id": str}, keep_default_na=False, na_values=[""])
    if set(census["geo_id"]) != {"EG32", "EG35"}:
        raise SystemExit(f"eg_census.csv has {sorted(set(census['geo_id']))}, expected EG32 "
                         "and EG35; re-run sources/eg_census.py")
    df = pd.concat([df, census], ignore_index=True)

    lut = pd.read_csv(HERE / "data" / "geo" / "eg" / "eg_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"eg.csv governorates with no polygon: {missing} -- re-run "
                         "sources/eg_geo.py, the lookup is stale")
    # 26 of Egypt's 27. North Sinai keeps its polygon and its hexes and draws no religion;
    # see the docstring and `gap=`.
    if df["unit"].nunique() != 26 or "EG34" in set(df["unit"]):
        raise SystemExit(f"{df['unit'].nunique()} governorates, expected 26 without EG34")

    # THE COPTS (2026-10-04, `fafd1067-copts`, sources/eg_churches.py). Every governorate's
    # Christians, survey and census rows alike, split at one national share: 86.49% to the
    # Coptic Orthodox Church (Global Flourishing Study 2023, `REL3` Orthodox) and the rest left
    # on `christianity`. One share everywhere, so this labels the Christians already drawn and
    # moves nobody: 131 GFS Christians cannot say where the share differs. Rows stay `modelled`
    # like the parent; `derived` would claim a counted parent to roll back to, and there is none.
    ch = pd.read_csv(HERE / "data" / "normalized" / "eg_churches.csv")
    share = dict(zip(ch["source_category"], ch["share_of_christians"]))
    if len(share) != 2 or abs(sum(share.values()) - 1) > 1e-6:
        raise SystemExit(f"eg_churches.csv is not a two-way split of Christians: {share}; "
                         "re-run sources/eg_churches.py")
    coptic_key = next(k for k in share if "Orthodox" in k)
    rest_key = next(k for k in share if k != coptic_key)
    is_c = df["source_category"] == "Christian"
    c = df[is_c]
    cop = c.assign(source_category=coptic_key,
                   count=(c["count"] * share[coptic_key]).round().astype(int))
    rest = c.assign(source_category=rest_key, count=c["count"] - cop["count"].to_numpy())
    df = pd.concat([df[~is_c], cop, rest], ignore_index=True)

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"eg.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    # EVERY row, without exception -- there is no measured tier in this country (§7).
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "eg": dict(
        name="Egypt",
        source="Arab Barometer, four waves 2013 to 2022 (Arab Barometer, Princeton "
               "University), against CAPMAS's own governorate population estimates for "
               "January 2026; South Sinai from the 1996 census (governorate table copied in "
               "the CAPMAS library, Arab-West Report paper 52) and New Valley from the 1976 "
               "census (E. J. Chitam, The Coptic Community in Egypt, Durham 1986, Table 4.1); "
               "the Coptic Orthodox share of Christians from the Global Flourishing Study, 2023",
        basis="self-identification, adults 18 and over; census, all residents, in two "
              "governorates",
        note_public=(
            "**Egypt's last four censuses asked about religion, and only the 1986 answer was "
            "published.** So this map is a survey standing where a census would "
            "be. It is drawn from the Arab Barometer: **6,840 people** interviewed across "
            "four rounds between 2013 and 2022, pooled, with each governorate's answers "
            "applied to CAPMAS's own estimate of that governorate's population. Nobody "
            "counted this, so the dots disappear when inferred dots are turned off. "
            "**Christians are 6.01% of the country as drawn, and Upper Egypt is where they "
            "are.** Minya is **16.41%**, Sohag 13.90% and Asyut 13.58%, against **1.06% of "
            "Sharqia** and 1.83% of Beheira in the Delta. Those top three rest on about 350 "
            "interviews each and sit inside one another's margins, so read them as a group "
            "and not as an order; what the survey does pin down is that the middle Nile is "
            "several times more Christian than the Delta is. "
            "**The reason this map was drawn at all is that the survey agrees with the last "
            "census that published.** The 1986 census, the last one whose religion table was "
            "released, put Christians at 5.7% of Egypt, and this map's national figure is "
            "**6.01%**. Cairo is drawn at **8.19%** against the 8.57% recorded for Cairo in "
            "1996, a governorate figure that reached the academic literature and that CAPMAS "
            "has never published itself. Those two are the only Egyptian numbers there are "
            "to check a survey against, and it clears both. How many Christians Egypt has is "
            "an old argument in which the Coptic Orthodox Church's own count is the higher "
            "one, and this map lands on the census side of it. "
            "**Three governorates rest on almost nothing.** Matrouh has **16 interviews**, "
            "the Red Sea 28 and Port Said 44, against a median of 271. The Red Sea is drawn "
            "as entirely Muslim because none of its 28 respondents answered Christian, which "
            "on that sample is consistent with anything up to about 11%; it is not a finding "
            "that no Copts live in Hurghada; the 1996 census counted **10.5%** Christians "
            "in the governorate. "
            "**The survey interviewed nobody in New Valley, North Sinai or South Sinai**, in "
            "any of the four rounds. Two of them are drawn from the census instead. South "
            "Sinai takes the 1996 census's **5.03%** Christian, from the governorate table "
            "Arab-West Report copied in the statistics agency's library, and New Valley the "
            "1976 census's **1.8%**, from E. J. Chitam's 1986 study of the Coptic community; "
            "both are shares from decades ago laid on today's population. North Sinai has "
            "no census figure anyone has printed and is not on the map: **475,331 people**, "
            "0.44% of Egypt. "
            "**No irreligion is drawn, and that is a fact about the questionnaire rather "
            "than about Egypt.** Two of the 6,840 answered atheist, on a card that only one "
            "of the four rounds carried; two rounds offered no such box at all and the "
            "fourth offered a differently worded one. In a country where saying it to a "
            "stranger with a clipboard carries a real risk, two is a floor of unknown depth, "
            "and nothing published says what the right number would be. The card is shallow "
            "the other way too: it has one Muslim box, so the Sunni share drawn here comes "
            "from elsewhere (last paragraph). "
            "**86.5% of Egypt's Christians are drawn as Coptic Orthodox.** That is the share "
            "of the 131 Christians in the Global Flourishing Study's 2023 Egyptian sample who "
            "named the Orthodox church as the one they most identify with, give or take about "
            "7 points. The other 13.5% stay plain Christian: Catholics, Protestants, and the "
            "one in thirteen who named no church. The same share is used in every governorate, "
            "because a sample that size cannot say where it differs, so the Coptic dots follow "
            "the Christian ones exactly. The Arab Barometer's 2018 round put the Orthodox lower, "
            "at 72%, but it also put Catholics at 16% of Christians, three times what the "
            "Catholic Church itself counts in Egypt; the 2023 figure sits much closer to the "
            "churches' own numbers."),
        how="survey, four rounds 2013 to 2022 pooled; two frontier governorates from old "
            "censuses; Copts from a 2023 survey",
        grain="governorates, 4.2 million people on average",
        gap="North Sinai, 0.44% of Egypt, where the survey interviewed nobody",
        gap_share=0.0044,
        counts=_eg_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "eg" / "eg_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_eg_place_weight,
        note="DRAWN AT GOVERNORATE ON ANITA'S CALL, 2026-09-08 (ask/answered/001-eg). Egypt "
             "collected religion in the 1986, 1996, 2006 and 2017 censuses and published it "
             "from 1986 only, which is spec §14.4 rule 2's central example: a persecuted "
             "minority mapped finer than its own state publishes. Her ruling was to draw at "
             "governorate. Nothing finer is available anyway -- the survey cuts by "
             "governorate and carries nothing below it, and the microdata extract CAPMAS "
             "deposited for the 2017 census has thirteen variables (governorate, station, "
             "marital status, sex, age, work status, education) and religion is not one of "
             "them. "
             "THE POPULATION IS CAPMAS'S OWN AND NOT COD-PS. HDX's cod-ps-egy is the 2012 "
             "COMPAS estimate, 81,395,541 people, and Egypt passed 100 million in 2020; "
             "drawing on it would put a quarter of the country nowhere and unevenly. "
             "sources/eg_geo.py pulls CAPMAS's api/GovernoratePopulation instead, pinned to "
             "2026-01-01 so the build is reproducible. That endpoint is on port 8080 of "
             "capmas.gov.eg and was found by grepping the React bundle -- §11d and §11af both "
             "concluded CAPMAS had no API, having probed the 443 host, which returns the same "
             "1,421-byte shell for every path including nonsense ones. "
             "SIXTY WAVE III RESPONDENTS ARE DROPPED. Their governorate label is `The West "
             "Bank`, on a code that appears on no non-Egyptian row. It is almost certainly "
             "Gharbia -- wave III's codes reproduce Egypt's own governorate numbering, "
             "position 10 in that numbering is Gharbia, Gharbia is otherwise absent from wave "
             "III alone, and al-Gharbiyya means 'the western', which a translator renders as "
             "the West Bank the same way it renders al-Buhayra as `The Lake` two rows above. "
             "It is not mapped anyway: sources/eg.py's BOGUS has the argument, and it is that "
             "0.9% of the pool is not worth risking a silent permutation for. "
             "COD-AB'S LUXOR IS THE CITY, NOT THE GOVERNORATE. Its Arabic name is literally "
             "'City of Luxor' and its polygon is 596 km2 against CAPMAS's 5,428, with Esna "
             "and Armant falling inside COD's Qena. Luxor's 1,459,385 dots are therefore "
             "packed into the city and the rest of its territory takes Qena's colours. The "
             "counts are right and the placement inside two adjacent governorates is not; "
             "both read 8 to 9% Christian, so the visible cost is small.",
    ),
}
