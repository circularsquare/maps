# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _tl_place_weight(place):
    """countries.py hook. `place` is the Kontur 400 m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "tl_hexes.gpkg", "sources/tl_grid.py")


def _tl_counts():
    """INETL 2022 census religion at municipality: 8 nodes on 14 units.

    ONE level, no allocation, nothing modelled, and the join is an identity: `sources/tl.py`
    writes COD-AB p-codes into `geo_id` and `sources/tl_geo.py` builds the polygons on the
    same p-codes.

    **THIRTEEN OF THE FOURTEEN UNITS ARE `measured` AND ATAURO IS `derived`.** The 2022
    census tabulates fourteen municipalities and publishes a religion table for none of them;
    what exists is a `<Municipality> em Números 2022` volume per municipality, and thirteen
    were written. Atauro, which became a municipality in time for the census, has no volume,
    so its row here is main report table 4.07 minus the thirteen. That is exact arithmetic on
    two published tables rather than a spread or a model, and every category comes out
    non-negative with Islam, Buddhism and indigenous religion at exactly zero. But no
    publication anywhere prints it, so it draws desaturated, which is the honest signal:
    a reader who wants to check Atauro against a source cannot.
    """
    from tl2022 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "tl.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "municipality"].copy()
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 14:
        raise SystemExit(f"{df['geo_id'].nunique()} municipalities, expected 14 -- re-run "
                         "sources/tl.py")
    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["tier"] = df["note"].str.contains("residual=yes").map(
        {True: "derived", False: "measured"})
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    g = df.groupby(["unit", "node"], as_index=False).agg(
        count=("count", "sum"),
        congregations=("congregations", "sum"),
        tier=("tier", lambda s: "derived" if (s == "derived").any() else "measured"))
    return g[["unit", "node", "count", "tier", "congregations"]]


ENTRY = {
    "tl": dict(
        name="Timor-Leste",
        source="Population and Housing Census 2022, the thirteen municipal em Números 2022 "
               "volumes and main report table 4.07 (INETL)",
        basis="self-identification, population aged 3 and over in private households",
        note_public=(
            "**Timor-Leste is 97.5% Catholic, the highest Catholic share of any country on "
            "this map**, ahead of Paraguay at 90.5% and Poland at 90.1%. Twelve of the "
            "fourteen municipalities are above 96% and Covalima reaches **99.8%**, so "
            "almost the whole country is one colour and what is worth looking at is the "
            "one place it is not. "
            "**That place is Atauro.** The island north of Dili was a posto of the capital "
            "until 2022 and is now a municipality of its own, with 9,622 people in the "
            "census's religion universe. It is **55.4%** Protestant against **44.5%** "
            "Catholic, and it is the only unit in the country where Catholicism is not the "
            "answer of at least nine people in ten. Everywhere else the Protestants are "
            "thin and scattered: Aileu is second at **7.9%** and nothing else reaches 4%. "
            "**Atauro's numbers are the only ones here that nobody published.** The census "
            "asked about religion and printed the answer for the nation alone, so what "
            "carries the municipalities is a separate series, one `em Números` volume "
            "each, written by the municipal statistics services and issued three years "
            "after the count. Thirteen volumes exist and Atauro, the fourteenth, has "
            "none, so its row is the national table minus the other thirteen. That is "
            "subtraction between two published tables rather than a model, and the three "
            "categories the island plainly has none of come out at exactly zero without "
            "being told to, but it is drawn desaturated because no source states it. "
            "**Below Christianity everything is very small, and two of the cells should be "
            "read carefully.** Islam is 3,202 people, 0.26%, with **1,964** of them in "
            "Dili and, less obviously, **363** at the far eastern end of the island in "
            "Lautém. Indigenous religion is 240 people, 0.019%, and the census's own report "
            "notes that the count was just over 900 in 2015 and that this kind of religion "
            "looks close to disappearing. Read that as a floor rather than a measurement of "
            "practice: the box counts people who gave the ancestral tradition instead of a "
            "church, not the far larger number who keep both. "
            "**No religion was a new box in 2022** and 797 people ticked it, 0.064%. Only "
            "the Philippines at 0.040%, Kiribati at 0.046% and Myanmar at 0.060% are lower "
            "among the countries here that record any such answer at all; the many that "
            "read as zero mostly do so because their source never asked. "
            "**Both of the holes in this map were tested for lean and neither moves "
            "anything.** The 239 people who gave no answer sit a little more in Dili than "
            "elsewhere, correlating **+0.78** with the Hindu share across the fourteen "
            "municipalities, which is a statement about the capital rather than about "
            "religion. The much larger hole, the people the question never reached, runs "
            "from **6.5%** of Dili to **7.7%** of Aileu, about a point of spread in a "
            "figure that is mostly the under-three population."),
        how="census, 2022",
        grain="municipalities, 89,000 people on average",
        fill="from the national census table, after the other thirteen municipalities",
        gap="children under 3 and people not in private households, who were never asked, "
            "and the 239 who gave no answer; 7.0% of the country between them",
        gap_share=0.06951,
        counts=_tl_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "tl" / "tl_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_tl_place_weight,
        note="THE MAIN REPORT PUBLISHES RELIGION NATIONALLY AND NOWHERE ELSE, AND THE "
             "GEOGRAPHY IS IN A DIFFERENT PUBLICATION SERIES ENTIRELY. Table 4.07 of the "
             "2022 census main report is religion by age and sex for the whole country; "
             "none of the 24 basic tables and none of the eight thematic reports crosses it "
             "with geography. What does is `<Municipality> em Números`, an annual volume "
             "written by each of the thirteen Serviços de Estatística Municipais, whose "
             "2022 edition prints 2010, 2015 and 2022 side by side under Proteção Social. "
             "This is Botswana's and Rwanda's shape again (§9bu, §9cd): the national report "
             "stops at the nation and a per-unit booklet series outside it carries the "
             "fine geography. "
             "THE VOLUMES ARE TYPED BY HAND AND FOUR CELLS OF THEIR 2015 COLUMN ARE WRONG. "
             "Every volume reprints 2015, which was published centrally as table 11 of "
             "Census 2015 Volume 2, so sources/tl.py checks all 252 cells of it, male, "
             "female and total. Ainaro prints the municipality's whole population as its "
             "Catholic total and Manatuto's `Seluk` total is four people over, in both "
             "cases with the male and female figures beside them exactly right; Oecusse "
             "mistypes a female cell and carries the error into its total, which no "
             "internal check can see. So a 2022 row whose sexes do not sum to its own total "
             "is refused, and the 2022 column is reconciled against table 4.07 category by "
             "category rather than trusted. "
             "ATAURO IS THE FOURTEENTH MUNICIPALITY AND IT FALLS OUT OF THAT "
             "RECONCILIATION. It became a municipality in time for the 2022 census, which "
             "lists it separately at 10,295 people, and no volume was ever written for it. "
             "The thirteen sum to 1,239,083 against table 4.07's 1,248,705; the 9,622 left "
             "over is 0.935 of Atauro's census population where the country's religion "
             "universe is 0.931 of its own, and 5,332 of them are Protestant, which is the "
             "island's known signature. Islam, Buddhism and indigenous religion all come "
             "out at exactly zero. It is drawn `derived`. "
             "COD-AB IS ONE MUNICIPALITY BEHIND AND THE FIX IS A TIER DOWN, NOT ANOTHER "
             "FILE. The OCHA bundle is valid_on 2020-09-11 with 13 ADM1 polygons, but "
             "Atauro is already in it as ADM2 TL0604 at 139.91 km2 against the census's "
             "140.55, so sources/tl_geo.py subtracts that polygon from Dili's ADM1 and adds "
             "it as the fourteenth unit. All fourteen come within 1.8% of the census's own "
             "area column. Nothing joins on a name anywhere in this country: every unit is "
             "a p-code on both sides. "
             "THE UNSD DEMOGRAPHIC YEARBOOK IS THE OUTSIDE WITNESS AND IT AGREES TO THE "
             "PERSON. Table 28 carries Timor-Leste for 2004, 2015 and 2022; all seven of "
             "its 2015 figures reproduce from Volume 2 table 11's thirteen municipalities "
             "and all five of its named 2022 figures from main report table 4.07, and both "
             "are asserted in sources/tl.py. Ask it BY NAME: oracle.py matches UNSD's own "
             "country string, so `oracle.py tl` comes back as a miss that reads like an "
             "absence (the fm review, 2026-09-08). Its 2022 row is NOT usable as a source, "
             "though: it names five categories and puts everything else in one `Other` cell "
             "of 95,328 against a total of 1,341,737, which is the resident population "
             "rather than the religion universe, so that residual is mostly the "
             "under-threes and drawing it would put 7.1% of the country into an "
             "unclassified cell that does not exist. "
             "FINER RELIGION DOES NOT EXIST, and it was looked for in four places: the "
             "2015 census Volume 4 suco tables (twelve indicators, no religion), the 452 "
             "Sensu Fó Fila Fali suco reports launched in July 2024 (never put online), "
             "the REDATAM population dashboard INETL links from its own census page "
             "(20.6.104.113, which answers on no port at all), and the 2021, 2023 and 2024 "
             "editions of all thirteen volumes. Only Viqueque ever prints religion below "
             "the municipality, and only for 2010 and 2015; Covalima's `Posto "
             "Administrativo` table on the same page as its religion table is Bolsa da Mãe "
             "recipients, and the two are told apart on column count.",
    ),
}
