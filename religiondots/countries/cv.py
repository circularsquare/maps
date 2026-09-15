# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _cv_place_weight(place):
    """countries.py hook. `place` is the Kontur 400 m hex layer scatter.py has read.

    22 concelhos over 4,033 km2 is 183 km2 a unit, so Cabo Verde is not drawn for §8.2's
    emptiness reason at the national scale. It is drawn for it inside the units. Santa
    Catarina do Fogo IS the caldera of Pico do Fogo; Porto Novo is 558 km2 of Santo Antao
    with its people in the port and one ribeira; Boa Vista and Maio are dune. An equal-area
    spread would put a fifth of the country's dots on lava.
    """
    return _kontur_place_weight(place, "cv_hexes.gpkg", "sources/cv_grid.py")


def _cv_counts():
    """INE RGPH-2021 at concelho: 14 nodes on 22 units.

    ONE level, no allocation, nothing modelled -- every row is `measured` and may ring.

    **THE CONCELHO IS THE FINEST TIER THAT HAS RELIGION AND THAT WAS ESTABLISHED RATHER
    THAN ASSUMED.** INE's census area publishes four things for 2021: the 22 `Quadros por
    Concelho` workbooks, which carry it; the general and thematic volumes, which are
    national; and `Agregados e População por Zonas e Lugares`, nine island workbooks that go
    down to the locality and carry households and population and no religion at all. The 32
    freguesias have boundaries in COD-AB and no religion table anywhere.

    **THE UNIVERSE IS THE POPULATION AGED 15 AND OVER**, 352,494 of 491,233, so 28.24% of
    Cabo Verde was never asked. That is `gap` rather than a category, and it is the reason
    the country's dots are 351,183 people and not half a million.
    """
    from cv2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "cv.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "concelho"].copy()
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 22:
        raise SystemExit(f"{df['geo_id'].nunique()} concelhos, expected 22 -- re-run "
                         "sources/cv.py")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna() & (df["count"] > 0), "source_category"]
                      .unique())
    if set(unmapped) - {"Não sabe / Não respondeu"}:
        raise SystemExit(f"cv.csv categories with no node: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df.groupby(["unit", "node"], as_index=False)[["count", "congregations"]].sum()


ENTRY = {
    "cv": dict(
        name="Cabo Verde",
        source="V Recenseamento Geral da População e Habitação 2021, Cabo Verde em Números "
               "(INE)",
        basis="self-identification, population aged 15 and over",
        note_public=(
            "**Cabo Verde's census names nine Christian churches one by one, and asks "
            "about a tenth thing that is barely a religion anywhere else.** Racionalismo "
            "Cristão is **6,129** people, **1.74%** "
            "of everyone the census asked, and the fourth largest named religion in the "
            "country after the Catholics, the Adventists and the Nazarenes. Christian "
            "Rationalism is a spiritualist doctrine founded in Santos, in Brazil, in 1910; "
            "it reached the islands the next year, carried back by a Cape Verdean who had "
            "been a medium in Rio. **3,988** of the 6,129 are in São Vicente alone, "
            "**6.86%** of the adults there, and after that come Tarrafal de São Nicolau at "
            "4.41%, Boa Vista at 3.54%, Paul at 3.17% and Sal at 2.38%. In São Salvador do "
            "Mundo and Santa Catarina do Fogo it draws nobody at all. That is the northern "
            "and eastern islands and not Santiago, which is the island list the movement's "
            "own centres give for their own history. "
            "**Catholicism is 72.49% of the country and the concelhos run from 97.53% to "
            "42.98%.** The top of that range is rural interior Santiago, São Salvador do "
            "Mundo and São Lourenço dos Órgãos, where no religion is 0.77% and 1.20%. The "
            "bottom is two very different places. São Vicente, the northern port island, "
            "is **46.36%** Catholic and **38.20%** no religion, by far the most secular "
            "concelho in Cabo Verde. Santa Catarina do Fogo, **3,204** adults inside the "
            "caldera of the volcano, is **42.98%** Catholic because the missions got there "
            "instead: 13.42% Adventist, 11.52% New Apostolic and 7.65% Latter-day Saint, "
            "each of them the highest figure in the country. "
            "**The Church of the Nazarene is the Protestant church here, and it is on the "
            "emigration islands.** It came from New England in 1901 with returning "
            "sailors, and it is 8.10% of Brava and 3.90% of Mosteiros against 0.07% in "
            "São Lourenço dos Órgãos. INE gives it one box labelled *Igreja do Nazareno / "
            "Protestante*, with no separate Protestant option anywhere on the form. "
            "**Islam is 1.31%, and where it is says what it is.** Boa Vista is 6.63% and "
            "Sal is 4.37%, the two islands the resorts were built on, then Praia at 1.88%; "
            "six concelhos are under a fifth of one per cent. This is West African "
            "labour migration of the last twenty years rather than a Cape Verdean "
            "community. At the other end of the table, **23** people gave their religion "
            "as Jewish, across seven concelhos and never more than seven in one, which is "
            "the smallest answer the Cape Verdean census printed. "
            "**Everyone here is aged 15 or over.** The religion question was not put to "
            "children, so the map draws the **352,494** adults the census asked and the "
            "not drawn row carries the rest. That cut is not spread evenly: the concelhos "
            "with the most children are the rural ones on Santiago and Fogo and the one "
            "with the fewest is São Vicente, so an adults-only map tilts a little towards "
            "São Vicente's no-religion and Racionalismo Cristão shares and away from rural "
            "Santiago's Catholics and Fogo's Adventists. Nothing here corrects for that."),
        how="census, 2021, ages 15 and over",
        grain="concelhos, 16,000 people on average",
        gap=("28.5% of Cabo Verde: the 28.2% who are under 15 and were never asked the "
             "religion question, and the 0.3% who were asked and did not answer"),
        gap_share=0.28510,
        counts=_cv_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "cv" / "cv_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_cv_place_weight,
        note="THE QUEUE PRICED THIS COUNTRY AS NATIONAL-ONLY AND IT IS 22 UNITS. §11w had "
             "Cabo Verde at 'open office, not chased past the catch-all', off the UNSD "
             "oracle, whose rows are national and urban/rural. INE publishes one workbook "
             "per municipality, CABO VERDE EM NÚMEROS, twenty-two of them plus a national "
             "one, filed under the census area's `Quadros por Concelho 2021` rather than "
             "under its publications, and every one carries the religion table. "
             "ine.cv IS AN ANGULAR APP AND ITS PAGES CARRY NO LINKS; the workbooks are "
             "reached through the API its own bundle names, bdmi.ine.cv/site_deploy_api "
             "([[reference_spa_hidden_apis]]). sources/cv.md §1 has the four calls. "
             "THE RECONCILIATION IS THE TWENTY-TWO AGAINST THE TWENTY-THIRD: summing the "
             "municipal workbooks reproduces all fifteen of the national workbook's "
             "figures and its 352,494 total to the person, and the UNSD Demographic "
             "Yearbook, an independent transcription INE forwarded to New York, "
             "reproduces them again. The oracle's sixteenth category, `Unknown`, is "
             "138,739 people and is exactly the under-15 population; it is an age cut and "
             "not a non-response. "
             "THREE PAIRS OF CONCELHOS SHARE A NAME AND THE SHEET TITLES USE THE SHORT "
             "FORM. Ribeira Grande is on Santo Antão and Ribeira Grande de Santiago is "
             "not; Santa Catarina is on Santiago and Santa Catarina do Fogo is not; "
             "Tarrafal is on Santiago and Tarrafal de São Nicolau is not. A name join "
             "would pair each of the three with a coin flip and every total would still "
             "reconcile, so sources/cv_geo.py asserts the pairing three ways and tests "
             "each swap explicitly ([[reference_name_join_wrong_neighbour]]). "
             "THE DOTS ARE SPREAD ACROSS 2,156 KONTUR 400 M HEXAGONS weighted by hex "
             "population (sources/cv_grid.py), which also carries the magnitude check on "
             "that join: r = 0.96 over 22 units against a best of 0.66 in 2,000 random "
             "pairings. 123 hexes fall just seaward of the shoreline and are snapped to "
             "the nearest concelho rather than dropped, because on these islands the loss "
             "is seaward and dropping them walks the dots uphill onto empty volcano "
             "([[reference_archipelago_grid_snap]]). "
             "FINER RELIGION DOES NOT EXIST. INE's 2021 census area publishes the 22 "
             "concelho workbooks, national general and thematic volumes, and nine "
             "`Agregados e População por Zonas e Lugares` island workbooks that reach the "
             "locality with households and population and no religion. The 32 freguesias "
             "have COD-AB boundaries and no religion table.",
    ),
}
