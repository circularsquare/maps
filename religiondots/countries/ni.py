# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _ni_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Nicaragua needs this for both of §8.2's reasons at once. EMPTINESS: the two Caribbean
    autonomous regions are 46% of the land and 12% of the people, and Waspám alone is
    9,341 km² — larger than eleven whole departments. WATER: Lake Cocibolca is 8,264 km²
    and Xolotlán 1,042, and the municipal boundaries run out into both.

    And the two compound, because the emptiest units are the ones carrying the category
    Nicaragua is drawn for: Prinzapolka, Puerto Cabezas and Waspám are 43-53% Moravian and
    are among the largest municipios in the country. An equal share per polygon would paint
    the Moravian coast across uninhabited rainforest (sources/ni_grid.py).
    """
    return _kontur_place_weight(place, "ni_hexes.gpkg", "sources/ni_grid.py")


def _ni_counts():
    """INIDE 2005 census variable P13 at municipio: 8 categories on 153 municipios.

    ONE level, no allocation, nothing modelled — every row is `measured` and may ring.

    THE COUNTS COME OUT OF INIDE'S OWN REDATAM SERVER, NOT OUT OF A PUBLICATION. INIDE
    prints religion by DEPARTMENT (17 units, Volume I CUADRO 12) and serves it by
    municipality and by comarca; Volume IV's 546 pages of municipal tables carry no religion
    table at all. sources/ni.py runs the query and checks its nine national figures against
    the 2006 printed volume, which is a genuinely independent witness (sources.md §11x).

    THE COMARCA TABLE EXISTS AND IS NOT DRAWN. `OF COMR05, PERS05.P13` returns 2,579 units
    at 1,759 people each and reconciles to the same total. There are no comarca boundaries
    published anywhere — OCHA's COD-AB stops at municipio and geoBoundaries 404s on NIC
    ADM3 — so the geography is what limits this country, not the counts.

    THE UNIVERSE IS AGE 5 AND OVER. 4,537,200 of a 5,142,098 census population; the 604,898
    under-fives were never asked and are in `gap=` rather than drawn as a §3.5 undercount.

    THE JOIN IS ON NAME AND MUST STAY THAT WAY. COD's pcode is `NI` + a code in INIDE's own
    format and 145 of 153 agree, which makes a code join look right and be wrong: INIDE's
    9105 is Waspám and COD's NI9105 is Mulukukú. See sources/ni_geo.py, which refuses to run
    if that stops being true.
    """
    from ni2005 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ni.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "municipio"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "ni" / "ni_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"ni.csv municipios with no polygon: {missing} -- re-run "
                         "sources/ni_geo.py, the lookup is stale")
    if df["unit"].nunique() != 153:
        raise SystemExit(f"{df['unit'].nunique()} municipios, expected 153")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "ni": dict(
        name="Nicaragua",
        source="VIII Censo de Población y IV de Vivienda 2005, variable P13 (INIDE), "
               "tabulated at municipio from INIDE's own Redatam server",
        basis="self-identification, population aged 5 and over",
        view=[-87.7, 10.7, -82.7, 15.1],
        note_public=(
            "**Nicaragua is the only census on this map that names the Moravian Church**, "
            "and it names it while naming no Anglicans, no Baptists, no Adventists and no "
            "Latter-day Saints — which is a statement about standing rather than size. On "
            "the Caribbean coast the Moravians are the historic church: they arrived in "
            "1849, ran the schools and clinics of the Miskito and Creole coast under the "
            "British protectorate and after it, and their congregations are the institution "
            "the two autonomous regions are organised around. "
            "**So the map has a coastline on it.** 73,902 Moravians are 1.63% of Nicaragua "
            "and **53.3% of Prinzapolka, 50.9% of Puerto Cabezas and 43.6% of Waspám** — "
            "the first place on this map where the Moravians are anybody's plurality. Nine "
            "municipalities are over 10%, they are all on the Caribbean, and between them "
            "they hold 94% of every Moravian in the country on 5% of its people. Forty-one "
            "municipalities have none at all. "
            "**The other coastal category is the one the census did not name.** `Otra` runs "
            "at 1.63% nationally and **44.1% on Corn Island**, 21.4% in Laguna de Perlas "
            "and 17.5% in Bluefields, against 0.02% in the far north-west — a spread of "
            "more than two thousand to one, and every unit above 8% is on the Caribbean. "
            "The Anglican church of the Mosquito Coast and the Jamaican Baptist mission "
            "that worked the same Creole towns from the 1840s are the obvious contents and "
            "neither has a box on this form. The map does not split the cell. "
            "**No religion is 15.7% and it is not an urban figure.** It peaks in the "
            "northern mountains and on the agricultural frontier — Santa María 39.8%, Murra "
            "36.1%, Wiwilí de Jinotega 32.9% — and bottoms out on the Moravian coast, where "
            "Waspám is 0.41% and Puerto Cabezas 1.22%. That is the opposite of the usual "
            "shape, and the Pacific cities sit in the middle of the range rather than at "
            "the top. "
            "**The Catholic heartland is the cattle country of the centre and north.** San "
            "Francisco de Cuapa is 93.5% — the highest in Nicaragua, and Cuapa is the "
            "village of the 1980 Marian apparitions and a national pilgrimage site — with "
            "Camoapa at 91.7% and a block of Nueva Segovia and Madriz municipalities behind "
            "them. The evangelical map is almost its negative: Waslala 37.3%, Murra 36.5%, "
            "Paiwas 33.8%, all of it the interior frontier rather than the cities."),
        how="census, 2005, ages 5 and over",
        grain="municipios, 30,000 people on average",
        gap_share=0.118,
        gap="under-fives, 11.8% of the country, who were not asked the religion question",
        counts=_ni_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ni" / "ni_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ni_place_weight,
        note="THE SOURCE IS A QUERY, NOT A PUBLICATION, AND THAT IS WHAT MAKES THE COUNTRY "
             "DRAWABLE AT THIS RESOLUTION. INIDE runs an open, unauthenticated Redatam "
             "webserver over the 2005 census microdata (redatam.inide.gob.ni). What INIDE "
             "PRINTS is religion by department — 17 units, Volume I CUADRO 12 — and Volume "
             "IV's 546 pages of municipal tables contain no religion table at all. What it "
             "SERVES is the same variable at 153 municipios and at 2,579 comarcas. That is "
             "a 9x resolution gap between an office's printed and served output on the same "
             "website, and it is why sources.md §11x adds the rule: ask whether the office "
             "runs a Redatam instance before reading its PDFs. "
             "THE COMARCA TABLE IS REAL, FREE, AND UNPLOTTABLE. 2,579 units at 1,759 people "
             "each would be among the finest geographies on this map. No comarca boundaries "
             "are published: OCHA's cod-ab-nic says in its own words that it is "
             "'structured into 2 levels' and geoBoundaries 404s on NIC ADM3. For once the "
             "BOUNDARIES are the ceiling and not the counts. "
             "THE UNIVERSE IS AGE 5 AND OVER — 4,537,200 of a 5,142,098 census population. "
             "The 604,898 under-fives were never asked, which is a different thing from "
             "being missed, so they are in `gap=` and not drawn as a §3.5 undercount. "
             "Within the universe the eight categories are an exact partition: they sum to "
             "the municipio total on all 153 rows, there is no `no especificado` cell, and "
             "100% of the table is drawn. "
             "THE READ IS CHECKED AGAINST THE PRINTED VOLUME, WHICH SHARES NO CODE PATH "
             "WITH IT. All nine of CUADRO 12's national figures, typeset in 2006, are "
             "reproduced exactly by a 2026 query against the microdata. Nothing else here "
             "would catch a column landing in the wrong place, because every internal "
             "identity reconciles whichever order the columns are read in. "
             "THE JOIN IS ON NAME AND THE CODE JOIN IS A TRAP. COD's adm2_pcode is 'NI' + a "
             "code in INIDE's own four-digit format and 145 of 153 match, which is exactly "
             "§12's shape-2 failure: ten municipalities were renumbered between 2005 and "
             "COD's 2023 vintage, and five of them collide rather than going missing. INIDE "
             "9105 is Waspám; COD NI9105 is Mulukukú — so a code join would move a 43.6% "
             "Moravian border municipality inland and every total would still reconcile. "
             "sources/ni_geo.py joins on name, confirms it with the department prefix on "
             "all 153, and then checks that the most Moravian municipios really are the "
             "easternmost using COD's own centroid longitudes — a witness that uses neither "
             "name nor code. It refuses to run if the code join ever stops being wrong. "
             "The dots are spread across 47,270 Kontur 400m hexagons weighted by hex "
             "population (sources/ni_grid.py). AND THE VINTAGE GAP THERE IS EIGHTEEN YEARS, "
             "THE LARGEST ON THIS MAP: the counts are 2005 and the placement grid is 2023. "
             "It moves dots within a unit and never between units, so no count is affected, "
             "but in the eastern frontier municipios the dots land in settlements that had "
             "barely begun when the census was taken — Prinzapolka is both the most Moravian "
             "municipality and the second most re-settled. It also means the Kontur/census "
             "ratio band cannot be the discriminating check here, unlike Zimbabwe's; the "
             "correlation is (r=0.948 on 153 units, best of 0.296 over 2,000 shuffles), and "
             "sources/ni_grid.py says which one is carrying it.",
    ),
}
