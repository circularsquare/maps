# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _pe_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Peru needs this for both of §8.2's reasons at once, and harder than most. EMPTINESS: the
    Amazonian districts run to thousands of km2 at well under one person per km2, and Loreto
    alone is 29% of the land and 3% of the people. THE DESERT is the same problem inverted --
    the coastal districts are rainless waste with everybody in an irrigated valley a few km
    wide. And the two compound on the altiplano, where the most Adventist districts are large,
    high and mostly empty (sources/pe_grid.py).
    """
    return _kontur_place_weight(place, "pe_hexes.gpkg", "sources/pe_grid.py")


def _pe_counts():
    """INEI 2017 census variable C5P26 at district: 8 categories on 1,873 drawn units.

    ONE level, no allocation, nothing modelled -- every row is `measured` and may ring.

    THE COUNTS COME OUT OF INEI'S OWN REDATAM SERVER, NOT OUT OF A PUBLICATION, AND THAT IS
    WHAT MAKES THE COUNTRY DEEP. What INEI PUBLISHES is four categories; what it SERVES is
    eight, at 1,874 districts, unauthenticated. The UNSD oracle reports four because four is
    what INEI forwarded to it -- and the published `otra religión` of 1,115,872 is EXACTLY
    the five extra columns added together, which sources/pe.py asserts. sources.md §11y.

    THE UNIVERSE IS AGE 12 AND OVER. 23,196,391 of a 29,381,884 census population; the
    6,185,493 under-twelves were never asked and are in `gap=` rather than drawn as a §3.5
    undercount. Within the universe the eight categories are an exact partition on all 1,874
    districts -- there is no `no especificado` cell and 100% of the table is drawn.

    THE JOIN IS ON CODE, WHICH REVERSES NICARAGUA DELIBERATELY. COD's adm3_pcode is `PE` +
    the six-digit ubigeo the census tabulates on; 1,872 of 1,874 codes are present and 1,870
    of those agree on the district name outright. A NAME join would be the risky one here,
    because Peru has many districts sharing a name across provinces. sources/pe_geo.py has
    the argument and three witnesses.

    AND TWO DISTRICTS SHARE ONE POLYGON. COD carries a single `Mazamari - Pangoa` polygon
    where the census has two districts, so pe_lookup.csv sends both to PE120699 and they are
    summed here. 62,229 people, 0.27% of the universe, drawn at half Peru's usual resolution
    and not separable on the map.
    """
    from pe2017 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "pe.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "distrito"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "pe" / "pe_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"pe.csv districts with no polygon: {missing} -- re-run "
                         "sources/pe_geo.py, the lookup is stale")
    if df["unit"].nunique() != 1873:
        raise SystemExit(f"{df['unit'].nunique()} units, expected 1873")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    # Mazamari and Pangoa are two census districts on one polygon; sum them there.
    df = df.groupby(["unit", "node"], as_index=False)["count"].sum()
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "pe": dict(
        name="Peru",
        source="Censos Nacionales 2017: XII de Población, VII de Vivienda y III de "
               "Comunidades Indígenas, variable C5P26 (INEI), tabulated at district from "
               "INEI's own Redatam server",
        basis="self-identification, population aged 12 and over",
        view=[-81.5, -18.5, -68.5, 0.1],
        note_public=(
            "**Peru's census names eight religions, and its own published tables name "
            "four.** INEI printed Catholic, Evangelical, *other religion* and none — and "
            "that *other religion* of 1,115,872 people turns out to be five separate "
            "census answers added together: Adventists, Jehovah's Witnesses, "
            "Latter-day Saints, plain 'Christian', and a genuine remainder. The eight are "
            "all there in the microdata, at 1,874 districts, and this map draws them. "
            "**The Adventists are the reason to look.** 353,430 people, 1.5% of Peru — and "
            "not spread thinly. They are **two regions**. The first is the Aymara "
            "altiplano around Lake Titicaca, where the Adventist mission at Platería opened "
            "in 1898 and ran the schools: San Antón is **23.4%** Adventist, Crucero 22.1%, "
            "Amantaní 20.9%, Huacullani 17.5%. The second is 800 km north in the Alto Mayo "
            "colonisation frontier — Yantaló 17.8%, Omia 17.1%, San Fernando 16.8%. "
            "Between the two, 399 districts have no Adventist at all. **And the two regions "
            "are two peoples.** Adventists are **6.6% of Aymara Peru, four times the "
            "national rate**, and 4.3% of Amazonian indigenous Peru — while Quechua Peru, "
            "four times the size of Aymara Peru and largely the same highlands, sits at the "
            "national average. The altiplano cluster is Aymara rather than merely southern. "
            "**And Peru is the first census on this map to print a box for the Latter-day "
            "Saints.** 113,659 people, named by the state rather than counted by the church "
            "or hidden in an 'other' bucket. They are a southern coastal population — "
            "Pacocha 2.2%, Islay 2.1%, Mollendo 1.9% — and absent from 914 districts. "
            "**Evangelicals are 14.1% and they are the periphery, not the cities.** Elías "
            "Soplín Vargas in San Martín is 77.9%; Uchuraccay and Anchihuay in Ayacucho are "
            "66.9% and 65.3%; El Cenepa and Río Santiago — the Awajún and Wampís districts "
            "of Amazonas — are 65.1% and 59.7%. They are above zero in every one of the "
            "1,874 districts, which nothing else here manages. "
            "**No religion is 5.1%, and reading it as secularity would be wrong.** It peaks "
            "in indigenous Amazonia: Puerto Bermúdez 37.7%, Awajún 30.5%, Raymondi 26.9% — "
            "Asháninka and Awajún country. **The census asks a second question that says "
            "what is happening there.** Among the 210,612 people who answer that by custom "
            "and ancestry they are *native or indigenous of the Amazon*, 'none' runs at "
            "**18.7%, three and a half times the national rate** — so the association is "
            "real. But it is the third answer, not the first: **41.5% are evangelical**, "
            "nearly three times the national rate, 34.1% Catholic, and about **81% give a "
            "Christian answer of some kind**. The story in Amazonia is evangelical "
            "conversion, with a large minority the form has no box for. "
            "**The smallest category has the sharpest geography.** `Otra` is 0.41% "
            "nationally and 20.7% in Yavarí, 19.2% in Tournavista, 18.9% in San Pablo — "
            "Amazon river and colonisation districts on the Brazilian and Colombian "
            "frontier, a spread of fifty to one. **It is not the indigenous Amazonians, "
            "and that is measured rather than assumed** — `Otra` runs at 0.40% among them "
            "against 0.41% nationally, which is no elevation at all. The Israelitas del "
            "Nuevo Pacto Universal, a Peruvian church founded in 1968 whose settlement "
            "colonies are in exactly these frontier districts and whose members are Andean "
            "migrants rather than Amazonian peoples, are the strongest candidate. The "
            "highest shares of all are among the **Tusán (5.5%) and Nikkei (3.5%)**, Peru's "
            "Chinese and Japanese populations. The map does not split the cell."),
        how="census, 2017, ages 12 and over",
        grain="districts, 12,378 people on average",
        gap_share=0.211,
        gap="under-twelves, 21.1% of the country, who were not asked the religion question",
        counts=_pe_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "pe" / "pe_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_pe_place_weight,
        note="THE SOURCE IS A QUERY, NOT A PUBLICATION, AND HERE THAT DOUBLES THE CATEGORY "
             "COUNT RATHER THAN THE RESOLUTION. INEI runs an open, unauthenticated Redatam "
             "webserver over the 2017 census microdata (censos2017.inei.gob.pe/bininei). "
             "sources.md §11t DECLINED Peru on the UNSD oracle's report of four categories; "
             "§11y reopened it because four is what INEI FORWARDED to UNSD, not what the "
             "census holds. The variable is Poblacio.C5P26 and it returns EIGHT named "
             "columns: Católica, Evangélica, Otra, Ninguna, Cristiano, Adventista, Testigo "
             "de Jehová, Mormones. That is the limit of §11w's 'ask the oracle' rule — the "
             "oracle ranks what was reported, and a country can be deeper than its return. "
             "THE PUBLISHED FOUR-CATEGORY TABLE IS WHERE THE OTHER FIVE WENT, AND THAT IS "
             "ALSO THE INDEPENDENT CHECK. INEI's own release prints Católica 17,635,339, "
             "Evangélica 3,264,819, otra religión 1,115,872 and Ninguna 1,180,361; the "
             "query reproduces the first, second and fourth exactly, and its `otra "
             "religión` is EXACTLY Otra + Cristiano + Adventista + Testigo de Jehová + "
             "Mormones. Every internal identity here reconciles whichever order the columns "
             "are read in — the three geographies agree to the person — so the published "
             "figures are the only thing that would catch a column landing in the wrong "
             "place. sources/pe.py asserts all four. "
             "THE UNIVERSE IS AGE 12 AND OVER — 23,196,391 of a 29,381,884 census "
             "population. The 6,185,493 under-twelves were never asked, which is a "
             "different thing from being missed, so they are in `gap=` and not drawn as a "
             "§3.5 undercount. Within the universe the eight categories are an exact "
             "partition on all 1,874 districts, there is no `no especificado` cell, and "
             "100% of the table is drawn. "
             "THE JOIN IS ON CODE, WHICH REVERSES NICARAGUA ON PURPOSE. COD's adm3_pcode is "
             "'PE' + the six-digit ubigeo the census tabulates on; 1,872 of 1,874 codes are "
             "present and 1,870 agree on the name outright, the two exceptions being "
             "spelling (Hualla/Huaya, San Pedro de Laraos/Laraos), each confirmed by "
             "reading the whole province's list out of both sources. A NAME join would be "
             "the risky one here, because Peru has many districts sharing a name across "
             "provinces and it would have to be disambiguated by the code. "
             "AND TWO DISTRICTS SHARE ONE POLYGON, WHICH IS THE WHOLE 1,874-VS-1,873 GAP. "
             "§11y read COD's ADM3 count as a vintage difference; it is not. COD carries a "
             "single 'Mazamari - Pangoa' polygon in Satipo, Junín where the census has two "
             "districts and no separate polygon for either. Both census codes go to "
             "PE120699 and are summed there: 62,229 people, 0.27% of the universe, drawn at "
             "half Peru's usual resolution and not separable on the map. Nothing is "
             "dropped. "
             "THE WITNESS THAT USES NEITHER NAME NOR CODE IS SPATIAL SMOOTHNESS, AND ITS "
             "FIRST VERSION WAS WRONG. It asserted that the most Adventist districts are "
             "the Puno altiplano, on the history of the 1898 Platería mission — and it "
             "fired, because the prior was wrong and not the join: Yantaló, Omia and San "
             "Fernando are the Alto Mayo, Peru's other Adventist region, 800 km north. So "
             "the check is now the property that made the naive version tempting, stated "
             "without naming anywhere: religion shares are spatially smooth, and a permuted "
             "join would destroy that while leaving every name, code and total intact. "
             "Catholic r=0.754, Evangelical r=0.750, Adventist r=0.690 against the eight "
             "nearest neighbours, with a best of 0.12 over 200 random re-pairings of the "
             "same shares. "
             "The dots are spread across 258,279 Kontur 400m hexagons weighted by hex "
             "population (sources/pe_grid.py), and THAT correlation is the strongest check "
             "the join gets, because a modelled 2023 grid shares no lineage with either "
             "INEI's counts or OCHA's boundaries: r=0.959 on 1,871 units against a best of "
             "0.067 over 500 shuffles. The vintage gap is six years, the SMALLEST on this "
             "map. The ratio band is not the check here and the reason is size rather than "
             "vintage: Kontur models population from building footprints, which on a "
             "district of 237 people is noise, so the spread narrows monotonically with "
             "district size (the largest 365 districts sit inside a factor of 5, the "
             "smallest 133 reach 12.6). Two districts in Bongará, Amazonas — Chisquilla and "
             "Recta, 403 people between them — are too small to contain a single hex "
             "CENTROID and fall back to their own polygon as one uniform placement cell, "
             "because scatter.py drops a unit with no placement polygon and would have lost "
             "them silently. "
             "FINALLY, THE `Ninguna` SENTENCE IN note_public IS CHECKED RATHER THAN "
             "ASSERTED, AND THE CHECK CHANGED IT. C5P25 — self-identified ethnicity — is "
             "asked of the same 23,196,391 people, so the two cross exactly with no "
             "modelling; sources/pe_ethnicity.py runs it and NOTHING IT TOUCHES IS DRAWN. "
             "The first version of the note said `Ninguna` peaks in Amazonia because the "
             "form has no box for Amazonian indigenous religion. Measured: `Ninguna` is "
             "18.74% among Amazonian indigenous respondents against 5.09% nationally, 3.68x "
             "— so the association is real and large — but 41.54% are Evangélica (2.95x) "
             "and about 81% give a Christian answer, so 'no box' was the third fact and the "
             "note now leads with evangelical conversion. THE SAME QUERY REFUTED A CANDIDATE "
             "branches.py HAD LISTED for other.pe: `Otra` is 0.40% among Amazonian "
             "indigenous against 0.41% nationally — 0.98x, district correlation r=0.07 — so "
             "whatever is in that cell, it is not them, which strengthens the Israelitas "
             "reading and leaves Tusán (5.51%) and Nikkei (3.50%) as its most concentrated "
             "groups. AND IT FOUND SOMETHING NOBODY LOOKED FOR: the two Adventist regions "
             "are two peoples. Adventists are 6.58% of Aymara Peru (4.32x) and 4.34% of "
             "Amazonian indigenous Peru, while Quechua Peru — four times larger and largely "
             "the same highlands — is at 1.60%, the national average.",
    ),
}
