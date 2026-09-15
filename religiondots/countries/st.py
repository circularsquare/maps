# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _st_place_weight(place):
    """countries.py hook. `place` is the Kontur 400 m hex layer scatter.py has read.

    Seven districts over 1,001 km2 is 143 km2 a unit, but the areas run from 16.5 km2 for
    Agua-Grande to 267 km2 for Caue and the empty ones are empty for a reason: Caue is the
    southern massif and the Obo national park at 28 people per km2, and Lemba is the
    western slope. An equal-area spread would put dots up Pico de Sao Tome.
    """
    return _kontur_place_weight(place, "st_hexes.gpkg", "sources/st_grid.py")


def _st_counts():
    """INE IV RGPH-2012 at district: 11 nodes on 7 units.

    ONE level, no allocation, nothing modelled -- every row is `measured` and may ring.

    **THE DISTRICT IS NOT THE FINEST TIER THAT HAS RELIGION, AND IT IS STILL THE RIGHT ONE.**
    INE's 2016 `Publicacao dos Resultados sobre Localidades` prints religion for every
    locality in the country, hundreds of them, in Tabela 3 -- but folded to SEVEN columns,
    with `Outras religioes` swallowing Mana, the Universal Church, the Witnesses, Deus e
    Amor and the World Messianic Church, and `Nao tem` swallowing both non-response rows.
    The seven district reports carry all THIRTEEN. Dots are 1,000 people, so a locality of
    200 never earns one and the finer geography would buy nothing visible while costing six
    of the eleven nodes. sources/st.py's check() reconciles the two tables against each
    other, exactly, so the coarser one is a witness rather than a road not taken.

    **THE UNIVERSE IS EVERYBODY**, all 178,739 people of every age, unlike Cabo Verde's
    15-and-over. The only hole is the 1,756 who did not declare or did not know.
    """
    from st2012 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "st.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "district"].copy()
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 7:
        raise SystemExit(f"{df['geo_id'].nunique()} districts, expected 7 -- re-run "
                         "sources/st.py")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna() & (df["count"] > 0), "source_category"]
                      .unique())
    if set(unmapped) - {"Não declarou", "Não sabe"}:
        raise SystemExit(f"st.csv categories with no node: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df.groupby(["unit", "node"], as_index=False)[["count", "congregations"]].sum()


ENTRY = {
    "st": dict(
        name="São Tomé and Príncipe",
        source="IV Recenseamento Geral da População e da Habitação 2012, Resultados "
               "Distritais (INE)",
        basis="self-identification, whole resident population",
        note_public=(
            "**São Tomé's census asks which of nine churches you belong to, and offers "
            "nothing else.** The 2012 form names the Roman Catholics, the Adventists, the "
            "Assembly of God, the New Apostolic Church, Maná, the Universal Church of the "
            "Kingdom of God, the Jehovah's Witnesses, Deus é Amor and the World Messianic "
            "Church, and then one box for anything else. There is no Protestant option, no "
            "Islam and no traditional religion, which is why **5.03%** of the country ends "
            "up in that last box, more people than any single church except the Catholics. "
            "Catholicism is **55.71%**, and the seven districts run from **68.31%** in "
            "Cantagalo down to **38.22%** in Caué. "
            "**No religion is 21.22%, and its two ends are an island apart.** Lembá, on the "
            "north-western coast of São Tomé, is **33.61%**, then Mé-Zóchi at 27.66% and "
            "Caué at 27.19%. The Região Autónoma do Príncipe, the smaller island about 150 "
            "km to the north-east, is **4.60%**, a seventh of Lembá's figure inside a "
            "country of 179,000 people. Príncipe is the most Adventist district too, 9.63% "
            "against a national 4.05%. When the 2024 census offered *sem religião* and "
            "*ateu* as separate boxes for the first time, thirteen of every fourteen people "
            "who took one took the absence rather than the label. "
            "**Caué is 6,031 people and it is where the newer churches have taken the most "
            "ground.** It is the southern district, the old plantation country below the "
            "forest, and the emptiest in the country. It is also the least Catholic, and "
            "the reason is not that it is secular: the Assembly of God is **10.08%** there "
            "against 3.35% nationally, the New Apostolic Church **8.41%** against 2.90%, "
            "and the Universal Church of the Kingdom of God **5.50%** against 2.00%. All "
            "three are at their highest in Caué. "
            "**Maná is 4,191 people, 2.34%, and no other census on this map counts it.** "
            "The Igreja Maná was founded in Lisbon in 1984 by Jorge Tadeu and runs its own "
            "television and radio in São Tomé. It is at its strongest in the capital, "
            "3.01% of Água-Grande and 2.79% of Cantagalo against "
            "0.62% in Lembá; the New Apostolic Church is its mirror image, 0.89% in the "
            "capital and its best figures in Caué and Príncipe. And **688** people, 0.38%, "
            "gave the Igreja Messiânica Mundial, the Japanese religion founded at Atami in "
            "1935 and carried to the Portuguese-speaking world from Brazil. That is a "
            "larger share of São Tomé than it is of Brazil, where it is 0.05%. "
            "**1,756 people, 0.98%, did not declare a religion or did not know**, and are "
            "not drawn. They are commonest in Caué at 2.01% and rarest in Príncipe at "
            "0.60%, but leaving them out moves no share on this map by more than six tenths "
            "of a point. The date is the larger caveat. São Tomé counted itself again in "
            "2024 and published religion by district for that year as well, but **26.9%** "
            "of the 2024 answers are recorded as undeclared, against 0.98% in 2012, so this "
            "map draws the older census."),
        how="census, 2012, all ages",
        grain="districts, 26,000 people on average",
        gap=("1.0% of São Tomé and Príncipe: the 0.7% who did not know their religion and "
             "the 0.3% who did not declare one"),
        gap_share=0.00982,
        counts=_st_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "st" / "st_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_st_place_weight,
        note="THE OFFICE PUBLISHES A WHOLE CENSUS REPORT PER DISTRICT AND EVERY ONE CARRIES "
             "THE RELIGION TABLE. §11w priced São Tomé off the UNSD oracle, whose rows are "
             "national and urban/rural, and closed the row as 'not chased'. www.ine.st has "
             "a folder `Dados Distritais e Nacional Recenseamento 2012` holding eight PDFs, "
             "one per district plus a national one, each reprinting Quadros 1-26 for that "
             "district alone. Same shape as Cabo Verde's municipal workbooks a day earlier: "
             "ask whether the office publishes a volume per unit of the tier below. "
             "THE ROUTE IS AN OPEN AUTOINDEX. ine.st is Joomla with Phoca Download, but "
             "LiteSpeed serves /phocadownload/userupload/Documentos/ with directory listing "
             "left on, so the whole 223-file tree walks recursively. No plugin id sweep and "
             "no REST base needed; /phocadownload/ itself is NOT an index and the sweep has "
             "to start one level down. "
             "THE PARSE CLOSES FOUR WAYS. Each district's thirteen categories sum to its "
             "own printed total; the seven districts sum to the national report in every "
             "category; INE's 2016 locality publication, typeset separately and folded to "
             "seven columns, reproduces all of it (its `Outras religiões` is exactly the "
             "five categories it swallows and its `Não tem` exactly the three); and the "
             "UNSD Demographic Yearbook, INE's own return to New York, reproduces all "
             "thirteen national figures to the person. "
             "2024 EXISTS AND IS NOT DRAWN. The V RGPH of November 2024 was published in "
             "July 2025 and tabulates religion by district in sixteen categories, adding "
             "Islam and a separate Ateu row. 56,200 of its 209,161 people, 26.9%, are `ND`, "
             "against 1,756 non-responses in 2012, and unlike Cabo Verde the residual "
             "cannot be shown to be an age cut: it is 1.026 to 1.037 times each district's "
             "under-10 population, close but never exact, and the report never says who was "
             "asked. Its labels have also been round-tripped through machine translation, "
             "so `Messiânica Mundial` prints as `Copa do Mundo`, `Quadro` as `Pintura` and "
             "`idade de união` as `idade sindical`. It is used as a witness and not as a "
             "source. See sources/st.md §4. "
             "THE JOIN HAS NO TWINS, which is worth saying because the two before it did. "
             "Seven distinct district names, six folding to COD-AB's spelling and the "
             "seventh being COD's English gloss for the autonomous region. sources/st_geo.py "
             "confirms it on the areas the 2024 census prints per district and on the growth "
             "between the two censuses; the area witness pins five of the seven outright and "
             "CANNOT separate Cantagalo from Mé-Zóchi, which the file asserts rather than "
             "glosses. COD-AB calls this tier ADM1 and COD-PS calls it ADM2 with a different "
             "pcode scheme, so the two COD files do not join to each other at all. "
             "THE DOTS ARE SPREAD ACROSS 546 KONTUR 400 M HEXAGONS weighted by hex "
             "population (sources/st_grid.py), r = 0.99 over 7 units. 49 hexes carrying "
             "7.4% of the grid's people fall just outside the shoreline and are snapped to "
             "the nearest district rather than dropped, which matters more here than "
             "anywhere: every district but Mé-Zóchi is coastal and dropping them walks the "
             "dots uphill into the Obô forest ([[reference_archipelago_grid_snap]]). "
             "THE LOCALITY TABLE IS FINER AND SHALLOWER. INE's 2016 `Publicação dos "
             "Resultados sobre Localidades` has religion for every locality in the country, "
             "in seven columns rather than thirteen, and localities average a few hundred "
             "people against a 1,000-person dot. Drawing on it would cost six of the eleven "
             "nodes and buy nothing visible, so it is the check and not the source.",
    ),
}
