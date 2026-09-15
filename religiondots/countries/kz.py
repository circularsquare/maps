# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _kz_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    17 regions over 2.7 MILLION km2 -- the largest units on this map by area by a long way.
    Karaganda region alone is 428,000 km2, bigger than Germany and Poland together, at a
    national density of 7 people per km2, and the population sits in a ring round the edge
    plus Astana and Almaty. An equal share would scatter Karaganda's dots evenly across the
    Betpak-Dala desert (sources/kz_grid.py).
    """
    return _kontur_place_weight(place, "kz_hexes.gpkg", "sources/kz_grid.py")


def _kz_counts():
    """COUNTED: 9 drawn nodes on 17 regions, every one of them BNS's own cross-tabulation.

    THIS WAS A MODELLED COUNTRY FOR ONE DAY AND IS NOT ONE NOW (2026-09-08). sources.md
    §11u established four ways that no BNS PUBLICATION cuts religion by region, and all four
    checks were correct. None of them was about the census dashboard, which is served by a
    Qlik Sense engine at qap.stat.gov.kz whose data model is the census microdata itself --
    35,195,612 person rows carrying `Вероисповедание` beside `Область`. The engine is open to
    anonymous callers and cross-tabulates on request, so religion x oblast is a query rather
    than a document. sources/kz.py has the route and the four margin checks that prove the
    pull is the census: every national religion total reproduces the published volume to the
    person, and the urban and rural margins are the volume's own constants.

    THE MODEL IS KEPT, IN sources/kz_model.py, BECAUSE IT CAN NOW BE SCORED. It is the only
    ethnicity->religion model on this map with a truth to check against, and it misplaces
    1,439,367 people, 7.50% of the country. Islam and Orthodoxy come back within 4.5%, so the
    north-south pattern it drew was real; `Отказались указать` is 27.8% misplaced and
    `Неверующие` 22.9%, so its refusal layer was an artefact. Spec §14.25 is written from
    that, and it is the reason to distrust the same construction elsewhere.

    THE 11.01% WHO REFUSED TO STATE ARE DRAWN, on `unknown` (Anita, 2026-09-07). The census
    form decides it: Question 11 offers seven options and the sixth is `Отказываюсь указать`
    -- "I decline to state", printed, numbered, chosen. That is an answer, not the derived
    residual §3.5 and tt2011.py are written about, and drawing it redistributes nobody.
    Kazakhstan is 100.00% drawn. taxonomy/kz2021.py has the argument.

    AND IT IS THE LAYER THE MEASURED DATA CHANGES MOST. Modelled, refusal came out almost
    flat, every region between 9.35% and 12.19%; counted, it runs from 1.19% in East
    Kazakhstan to 22.07% in Mangystau. note_public says so, and says that nothing BNS
    publishes explains the spread.
    """
    from kz2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "kz.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])

    lut = pd.read_csv(HERE / "data" / "geo" / "kz" / "kz_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"kz.csv regions with no polygon: {missing} -- re-run "
                         "sources/kz_geo.py, the lookup is stale")
    if df["unit"].nunique() != 17:
        raise SystemExit(f"{df['unit'].nunique()} regions, expected 17")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    # No `tier` column: every row is BNS's own count of that religion in that oblast, which
    # is `measured` by default (§7). The column used to be pinned to `modelled` for every
    # row in the country and was dropped when the counts arrived.
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "kz": dict(
        name="Kazakhstan",
        source="National Population Census 2021 (Bureau of National Statistics), religion by "
               "region from the census dashboard's own data service",
        basis="self-identification, whole enumerated population",
        view=[46.0, 40.5, 87.5, 55.5],
        # No `gap` line: every one of the seven answers Kazakhstan's Question 11 offers is on
        # the tree, refusal included, so 100.00% of the census is drawn. It had one until
        # 2026-09-07 (see taxonomy/kz2021.py's note on `Отказались указать`).
        note_public=(
            "**Kazakhstan counts religion, and these are the counts.** The 2021 census asked "
            "everyone, and BNS printed the answer for the country as a whole in a volume "
            "whose 542 pages do not name a single oblast. The regional figures drawn here "
            "come from the same census by a different door: the census dashboard is backed "
            "by the 19,186,015 individual records themselves and will cut religion by "
            "region on request. Every national total here reproduces the printed volume to "
            "the person. "
            "**The north and the south are two different countries on this question.** "
            "North Kazakhstan region is **55.2%** Christian and **38.7%** Muslim; Turkistan, "
            "on the Uzbek border, is **1.6%** Christian and **92.4%** Muslim. Those two are "
            "the most and the least Christian of the seventeen regions, and the line between "
            "them is the Slavic settlement of the northern steppe, first under the Empire "
            "and again under the Virgin Lands campaign of the 1950s. Islam is highest in "
            "Kyzylorda at **96.2%** and lowest in Kostanay at **37.0%**. "
            "**A ninth of the country declined to answer, and they are on the map.** "
            "2,112,653 people, **11.0%**, chose *I decline to state*, which is option 6 of "
            "the seven the census form offers, printed and numbered, not a blank somebody "
            "left. So it is an answer and it is drawn as one, in its own colour; nobody is "
            "quietly shared out among the religions. "
            "**Where people decline varies more than anything else here, and nothing "
            "published explains it.** East Kazakhstan is **1.2%** and Mangystau **22.1%**, "
            "which is an eighteenfold spread across the same form and the same nine "
            "options. It does not follow the north-south line, it barely moves between town "
            "and country, and it does not follow ethnicity: Mangystau is one of the most "
            "Kazakh regions in the country and Kazakhs decline at close to the national "
            "rate. A range that wide between neighbouring administrative units can be "
            "fieldwork practice as easily as reticence, so read this layer as a real "
            "pattern with an unknown cause rather than as a map of how guarded a region is. "
            "**Some part of that refusal is probably about the law.** Kazakhstan requires "
            "religious groups to register, refuses registration to Jehovah's Witnesses and "
            "to Ahmadi Muslims, and prosecutes unregistered worship. The **9,419** "
            "Protestants here are the country's entire Protestant count, house churches "
            "included; read that number, and the Muslim one, as floors. "
            "**Do not read the Russian border.** Kazakhstan looks vastly less secular than "
            "Russia 200 km away, **1.9%** non-believers in North Kazakhstan against **52%** "
            "of Omsk reporting no religious institution, and almost all of that cliff is "
            "the questionnaire. Russia's survey offers *believes in God, professes no "
            "religion*, which **24.9%** of Russians choose and which Kazakhstan's census "
            "does not offer at all. A Russian in Petropavl who believes vaguely and attends "
            "nothing has to pick something else, and **85.3%** of Kazakhstan's Russians are "
            "recorded Orthodox against **43%** of Russia's population. Russia's largest "
            "non-institutional answer has not vanished at the border; it is inside "
            "Kazakhstan's Orthodox count, and the 11% who declined to state are the rest of "
            "it. "
            "**Kazakhstan splits Christianity three ways, which most censuses here do "
            "not**, into Orthodox, Catholic and Protestant. **99.1%** of Kazakhstani "
            "Christians are Orthodox. The Catholics sit where the deportations left them, "
            "**0.9%** of North Kazakhstan region and **0.5%** of Akmola against almost none "
            "in the south: they are mostly the descendants of Germans and Poles moved to "
            "the steppe in the 1930s and 40s, which is why Karaganda has a cathedral. And "
            "**82%** of the country's Buddhists are Koreans, deported from the Soviet Far "
            "East in 1937."),
        how="census, 2021, whole enumerated population",
        grain="17 regions, 1.1m people on average",
        counts=_kz_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "kz" / "kz_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_kz_place_weight,
        note="COUNTED, and it was `modelled` for one day. sources.md §11u established four "
             "ways that no BNS PUBLICATION cuts religion by region, and every one of those "
             "checks is still correct: not one of the 542 pages of the 2021 religion volume "
             "names an oblast, while the LANGUAGE chapter of the same volume has a regional "
             "cut. All four were about documents. §9cd found the figures somewhere no "
             "document check reaches. "
             "THE SOURCE IS THE CENSUS DASHBOARD'S ENGINE, NOT A FILE. "
             "stat.gov.kz/ru/instuments/dashboards/28424/ embeds sheets from a Qlik Sense "
             "server at qap.stat.gov.kz, and a Qlik app ships its DATA MODEL rather than "
             "its charts. This one's model is the census: 35,195,612 person rows (2009 plus "
             "2021, to the person) carrying `Вероисповедание` beside `Область`, `КАТО "
             "РАЙОН`, `Тип местности` and `Национальность`. The JSON API is open to "
             "anonymous callers. sources/kz.py holds the route. "
             "FOUR MARGINS PROVE THE PULL IS THE CENSUS: all nine national religion totals "
             "reproduce volume ch.12 to the person; they sum to 19,186,015 with no "
             "residual; the urban and rural margins are the volume's own 11,741,342 and "
             "7,444,673; and the two censuses' row counts are their published populations. "
             "THE JOIN IS ON POPULATION, NOT NAMES. The engine labels oblasts in caps and "
             "carries no KATO code for them; names pair sixteen of seventeen and fail on "
             "the capital, which the census enumerated as Nur-Sultan and the dashboard "
             "calls Astana. kz.py asserts the seventeen populations distinct, joins on "
             "them, then checks the names and allows exactly that one disagreement. "
             "THE MODEL IS KEPT AND SCORED (sources/kz_model.py, spec §14.25). It "
             "misplaces 1,439,367 people, 7.50% of the country: Islam 4.4% and Orthodoxy "
             "4.5%, so its north-south pattern was real, but `Отказались указать` 27.8% and "
             "`Неверующие` 22.9%. Its near-flat refusal layer was an artefact; counted, "
             "refusal runs 1.19% (East Kazakhstan) to 22.07% (Mangystau). "
             "11.01% IS DRAWN, on `unknown` — `Отказались указать` is printed option six of "
             "seven on Question 11, so it is a chosen answer and not Trinidad's derived "
             "residual (§3.5, tt2011.py). Kazakhstan is 100.00% drawn. "
             "17 REGIONS IS THE 2021 VINTAGE AND THE BOUNDARY FILE IS 2023. COD-AB ships 20 "
             "ADM1 polygons because Kazakhstan created Abay, Jetisu and Ulytau in 2022; "
             "kz_geo.py dissolves the three pairs back. No polygon is cut and the reform "
             "split no rayon, so the ADM2 count (218, which the census also publishes) "
             "checks it, and Kontur's per-region band checks it again. "
             "218 RAYONS ARE NOW AVAILABLE AS COUNTS AND ARE STILL NOT TAKEN. The engine "
             "returns religion × `КАТО РАЙОН` for 218 units summing to 19,186,015, and the "
             "app carries the rayon boundaries too (table `карта_район`, 190 rows with a "
             "`Район.Line` geometry), so the old objection — 218 fuzzy transliteration "
             "matches, and 218 units of pure inference — is gone on both counts. It was "
             "left for a session that can build the finer geography properly rather than "
             "bolted onto this one; queue.md carries the row and data/raw/kz/ has the pull. "
             "THAT ROW NEEDS AN ASK FIRST. Going to 218 is a §14 resolution question that 17 "
             "never raised: Kazakhstan prosecutes unregistered worship and refuses "
             "registration to Jehovah's Witnesses and Ahmadi Muslims, and at 218 the 9,419 "
             "Protestants and the refusal cell stop being a wash over a million-person unit. "
             "Counted rather than modelled makes that sharper, not safer.",
    ),
}
