# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _kg_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    KYRGYZSTAN NEEDS THIS MORE THAN ALMOST ANY COUNTRY HERE. Nine units over 199,951 km2 is
    22,217 km2 apiece and the country is 94% mountain: Naryn oblast is 45,200 km2 of Tien
    Shan pasture holding 316,182 people strung along the Naryn river and the Torugart road,
    and Issyk-Kul's people are a ring around a lake that is itself 6,236 km2 of water. An
    equal share of dots per polygon puts most of the country's colour on ice and rock.
    sources/kg_grid.py.
    """
    return _kontur_place_weight(place, "kg_hexes.gpkg", "sources/kg_grid.py")


def _kg_counts():
    """EBRD Life in Transition Survey III, 2015-16, at oblast: 8 categories, 9 units, and
    EVERY ROW IS `modelled` IN §7.

    NO KYRGYZ CENSUS HAS EVER ASKED THE QUESTION, and that was established against the
    QUESTIONNAIRE rather than against the publications, which is §9cd's rule. Book I of the
    2022 census prints Form 2 in full as an annex: seventeen questions, nationality at 7 and
    native language at 10.1, and no religion item on any of the five forms. The office's own
    download catalogue was harvested across all 22 subject pages (806 entries, zero religion)
    and its bare-integer `dynamic` route swept as well (546 named files, zero); data.gov.kg
    was paged out in full (1,410 CKAN packages, religion only as a register of BUILDINGS).
    sources/kg.md has the record and sources.md §9co the write-up.

    THE SPLIT-HALF BAR IS A PERMUTATION HERE AND NOT `1.96/sqrt(n-1)`, AND IT CHANGES THE
    ANSWER. LiTS III is one wave, so the split is on PSUs and the statistic is the median over
    400 random halves; `1.96/sqrt(n-1)` is the standard error of ONE correlation, so on nine
    units it sits at +0.693 while the shuffled-label null's own 95th percentile is +0.450.
    Islam's median is +0.548: the fixed bar would have rejected a category the data separates
    from chance at p=0.020. sources/lits.py has the argument.

    THE POPULATION CHECK CANNOT SEPARATE ISSYK-KUL FROM BATKEN AND SAYS SO. Observed r =
    +0.9866 against all 362,879 other orderings of the nine units; exactly one beats it, the
    Issyk-Kul/Batken swap, at +0.9870. Those two are 0.69pp apart on a share the survey
    measures to +/-1.33pp, so no correlation on it can tell them apart. `lits.held_out`
    forgives a beating ordering only when every unit it moves goes to one inside 1.96 standard
    errors, and names the pair on every run. The join is pinned by two other witnesses that
    are not correlations: the office's own SOATE code IS COD's pcode (417NN -> KGNN), and the
    Russian oblast names agree independently of the codes.

    THE POPULATION IS THE OFFICE'S OWN AND NOT COD-PS, whose Kyrgyz file is dated 2018 and
    totals 6,140,200 against the National Statistical Committee's 7,404,329 at 1 January 2026.
    That is a 20.6% gap and eight years of real growth, not a projection disagreement; §9bn
    took the office's own over a COD error of 3.4%. COD-PS is still read for its age bands,
    which are used as a ratio for the §3.5 lean and for nothing else.
    """
    from kg2016 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "kg.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])

    lut = pd.read_csv(HERE / "data" / "geo" / "kg" / "kg_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"kg.csv oblasts with no polygon: {missing} -- re-run "
                         "sources/kg_geo.py, the lookup is stale")
    if df["unit"].nunique() != 9:
        raise SystemExit(f"{df['unit'].nunique()} oblasts, expected 9")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"kg.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    # EVERY row, without exception -- there is no measured tier in this country (§7).
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "kg": dict(
        name="Kyrgyzstan",
        source="Life in Transition Survey III, 2015 to 2016 (European Bank for "
               "Reconstruction and Development), against the National Statistical "
               "Committee's 1 January 2026 oblast populations",
        basis="self-identification, adults 18 and over",
        view=[69.0, 39.0, 80.5, 43.4],
        note_public=(
            "**No Kyrgyz census has ever asked about religion, so this is a survey standing "
            "where a census would be.** The 2022 census printed its own questionnaire in the "
            "back of the results volume, and it runs seventeen questions, with nationality "
            "at 7 and native language at 10, and no religion item anywhere on any of the "
            "five forms. The statistics committee's catalogue of **806** downloadable tables has "
            "none either, and the national open data portal's **1,410** datasets carry "
            "religion only as a quarterly register of religious buildings. The map is drawn "
            "from the EBRD Life in Transition Survey: **1,500 people** interviewed across all "
            "nine oblasts in 2015 and 2016, applied to the committee's 1 January 2026 "
            "population of 7,404,329. The dots are drawn desaturated to say that. "
            "**Two of the eight answers are drawn where the survey found them.** Islam and "
            "Orthodoxy are **96.1%** of the country between them. Splitting the 75 "
            "interview clusters in half at random four hundred times and re-ranking the nine "
            "oblasts returns +0.55 for Islam and +0.75 for Orthodoxy, and shuffling which "
            "oblast each cluster belongs to shows that an answer with no geography at all "
            "gets to +0.45 on this many units. The other six go at the national rate inside "
            "each oblast's remainder, so those people are still drawn and only the claim to "
            "know where they live is withdrawn. "
            "**Orthodoxy is Bishkek, the Chui valley and almost nowhere else.** Of "
            "Kyrgyzstan's 509,117 Orthodox Christians, **331,243** are in the capital, where "
            "they are 24.4% of the population against 6.9% nationally; Chui, the oblast that "
            "wraps Bishkek along the Kazakh border, is 10.1%. In Osh and Batken oblasts not "
            "one of the 460 people interviewed answered Orthodox, so the map draws none "
            "there, and zero out of that many still leaves room for about one percent. "
            "**The Buddhist figure is a ceiling rather than a measurement.** Sixteen people "
            "ticked it, and six of them were in Osh oblast and four in Batken, the two most "
            "rural and most uniformly Muslim parts of the country, while Bishkek returned "
            "one. Nothing about Kyrgyzstan makes that a geography, and the stability test "
            "ranks it worst of any answer here. Its **66,738** people are left in because "
            "dropping them would only move them somewhere else, but the number should be "
            "read as the largest Buddhism could be. "
            "**Talas is sixty interviews.** It is the thinnest oblast, and after Islam and "
            "Orthodoxy are placed, **8.8%** of it is left for the other six answers against "
            "4.0% nationally. Most of that gap is sixty people rather than Talas. "
            "**Everyone interviewed was an adult, and that tilts the whole map slightly.** A "
            "third of Kyrgyzstan is under 18, and the share runs from 31.3% in Bishkek to "
            "41.0% in Talas, so the places with the most Orthodox Christians are the places "
            "with the fewest children. Across the nine oblasts the under-18 share correlates "
            "**-0.85** with the Orthodox share drawn here, and leaving any single oblast out "
            "keeps it between -0.74 and -0.96. Applying an adult composition to everybody "
            "therefore draws Kyrgyzstan a little more Orthodox and a little less Muslim than "
            "it is. "
            "**And the survey has no answer for the mazar.** Kyrgyz shrine pilgrimage, spring "
            "and grave veneration, and the healers who work at them are practised widely and "
            "have no box on this card; the people who take part answer Muslim, which is also "
            "how most of them would describe themselves. Nothing here counts that separately "
            "and the 0.2% other cell is not where it is hiding."),
        how="survey, one round of 1,500 interviews in 2015 and 2016",
        grain="oblasts and republican cities, 823,000 people on average",
        counts=_kg_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "kg" / "kg_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_kg_place_weight,
        note="THE COUNTRY IS A SURVEY ON A REGISTER AND EVERY ROW IS `modelled` (§7b), the "
             "same construction as Guatemala and El Salvador: LiTS III's `q922` gives an "
             "oblast share, the National Statistical Committee's own 1 January 2026 resident "
             "population gives the people it applies to, and no magnitude is invented. "
             "sources/lits.py holds the shared half, because Tajikistan is the next queue row "
             "on this file. "
             "THE CENSUS WAS RULED OUT ON THE QUESTIONNAIRE AND THE OFFICE ON AN ENUMERATION, "
             "not on a search (§9cd, §11aj). Form 2 of the 2022 census is printed in Book I "
             "and has seventeen questions with no religion item; the office's 22 subject "
             "pages were harvested for all 806 of their downloads and its bare-integer "
             "`dynamic` route swept to 1,400 for the 546 that return a named file; "
             "data.gov.kg was paged out for all 1,410 CKAN packages. Zero religion tables of "
             "people in any of them. No stat.gov.kg page carries an iframe and there is no "
             "Qlik, Power BI, Tableau or ArcGIS anywhere on the domain, which is where "
             "Kazakhstan's figures turned out to be. `census.stat.gov.kg` resolves and drops "
             "every port tried, and nothing is known about what is behind it. "
             "THE PERMUTATION BAR IS BUILT RATHER THAN ASSUMED, AND IT CHANGES THE ANSWER. "
             "The split-half statistic is the median over 400 random PSU halves, which cannot "
             "be compared with 1.96/sqrt(n-1) because that is the error of ONE correlation. "
             "On nine units the fixed bar is +0.693 and Islam's own shuffled-label null puts "
             "its 95th percentile at +0.450, so the fixed bar would have thrown out Islam at "
             "+0.548, p=0.020. This is [[reference_check_needs_power]] pointing the other "
             "way: the fix is to measure the null, never to move a bar so something passes. "
             "THERE IS NO SINGLE NEW BAR. SEVEN TESTABLE CATEGORIES GET SEVEN NULLS AND TWO "
             "OF THEM ARE STRICTER THAN THE FIXED BAR THEY REPLACED: MUSLIM +0.450, ORTHODOX "
             "+0.468, BUDDHIST +0.460, ATHEISTIC/AGNOSTIC/NONE +0.524, OTHER CHRISTIAN "
             "+0.550, then OTHER at +0.750 and JEWISH at +1.000 against the fixed +0.693. "
             "That is the answer to `is this just weaker', and it is the strongest form of it: "
             "the permutation is looser exactly where a category is dense enough for a "
             "median-of-400 to be stable and stricter exactly where it is not, which is what "
             "a calibrated null does and what a moved bar cannot do. OTHER fails at p=0.34 "
             "because its own null sits at +0.750, not because +0.450 was applied to it and "
             "missed. THE LEVEL WAS NOT TOUCHED, ONLY THE INSTRUMENT: `alpha` is "
             "`lits.stability`'s inherited 0.05 default, the same 95% that 1.96/sqrt(n-1) "
             "already encoded, so nobody picked a significance level here. And the size is "
             "measured rather than assumed: the second read took 200 datasets with the "
             "geography destroyed by that same shuffle, ran each through `lits.stability` as "
             "if it were observed data with its own null, and the 1,400 p-values reject at "
             "0.046 against a nominal 0.05, with MUSLIM at 0.040 and ORTHODOX at 0.055. That "
             "calibration is a one-off review run and not something the build prints; its "
             "method and its per-category table are in sources/kg.md §9.1 and §9.2. "
             "THE HELD-OUT CHECK NAMES THE PAIR IT CANNOT SEPARATE. r=+0.9866 against all "
             "362,879 other orderings of the nine units, and the single ordering that beats "
             "it swaps Issyk-Kul and Batken, which are 0.69pp apart on a share measured to "
             "+/-1.33pp. lits.held_out forgives only orderings that move units inside 1.96 "
             "standard errors of each other, prints them, and stops the build on anything "
             "else. THE WITNESS THAT PINS THE DECODE IS THE ABBREVIATION AND NOT THE "
             "ARITHMETIC. The SOATE-to-pcode identity and the Russian names pin the OFFICE to "
             "COD; the join the held-out check actually tests is LiTS's own region label to "
             "the pcode, which is hand-written in kg_geo.LITS_REGION, and §9.5 showed the "
             "swapped pairing passing the held-out check MORE cleanly than the true one. So "
             "kg_geo.lits_decode_witness requires each LiTS label to abbreviate exactly one "
             "of COD's nine Russian names token for token (`И` begins `Иссык`, `КУЛЬСКАЯ` "
             "equals `Кульская`), and exactly one of the 362,880 pairings satisfies it. No "
             "correlation and no population figure is involved. "
             "THE BUDDHIST CELL IS AN ARTEFACT AND THE STABILITY TEST FOUND IT. Six of "
             "sixteen respondents are in Osh oblast and four in Batken, against one in "
             "Bishkek; median split-half -0.151 with 68% of halves negative, the worst here. "
             "Spread at the national rate and called a ceiling rather than dropped. "
             "taxonomy/kg2016.py has it, along with why Islam is drawn on the bare family "
             "node in a country that is Hanafi Sunni throughout: the card asks one Muslim "
             "question and §14.3 forbids putting 6.6 million people on a node this source "
             "cannot see.",
    ),
}
