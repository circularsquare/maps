# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


class _AoHexWeighter(_KeHexWeighter):
    """Split a municipality's dots across Kontur 400 m hexagons by hex POPULATION.

    Angola is Kenya's problem with more units and a wider spread. The 326 municipalities
    average 3,800 km² and run from Rangel's 3 km² to over 25,000 in Rivungo; the eastern
    and southern quarter of the country — Moxico, Moxico Leste, Cuando, Cubango — is a
    third of its area and under 4% of its people, while roughly a quarter of Angolans live
    inside greater Luanda. An equal share per polygon would wash the Kalahari sand-veld in
    evenly spaced dots and leave the Ovimbundu plateau, where the people actually are,
    paler than the desert beside it.

    It is also the coastal case: the municipal boundaries run to the shoreline rather than
    to a generalised coast, and the western halves of Namibe and Cunene are the Namib.
    A population grid has no hexes there, so §8.2c's problem does not arise.

    Same caveat as Kenya's and Malawi's: it is a POPULATION weight and not a religion one.
    Nothing measures where a municipality's Tocoists sit inside it, so a Tocoist dot and a
    Catholic dot are spread identically. Read it as "religion by municipality, drawn where
    Angolans live".
    """

    def summary(self):
        return (f"{self.n_pop:,} (unit, node) rows placed on Kontur hex population, "
                f"{self.n_uniform:,} on equal shares where a municipality's hexes sum to "
                f"zero (sources/ao_grid.py)")


def _ao_place_weight(place):
    """countries.py hook. `place` is the hex layer scatter.py has read."""
    if "pop" not in place.columns:
        print("  !! ao_hexes.gpkg has no `pop` column — run sources/ao_grid.py; "
              "placing on equal shares (§8.2)")
        return None
    return _AoHexWeighter(place)


def _ao_counts():
    """INE 2024 RGPH at municipality: 20 drawn bodies on 326 municipalities.

    ONE level, no allocation, nothing modelled — INE publishes these categories at this
    geography and the map draws exactly that, so every row is `measured` and may ring.

    326 UNITS FOR 34.5M PEOPLE IS ~106,000 EACH, and the tier is one the country has only
    had since Lei 14/24 of 5 September 2024. The census was collected on the old division
    of 18 provinces and 164 municipalities and retabulated onto the new one of 21 and 326
    before publication, so these figures exist at no other geography and no pre-2024
    boundary file matches them; sources/ao_geo.py has where the polygons came from and why
    OCHA's could not be used.

    TWO PROVINCES ARE DRAWN ON ELEVEN BODIES INSTEAD OF TWENTY-ONE, and it is the source
    and not the parse. The page carrying the second half of the religion table is BLANK in
    the published Uíge and Moxico Leste volumes — no text, no image, no drawing — so
    2,277,708 people, 6.6% of the country, have Catholic, Protestant, Tocoist,
    Kimbanguist, Josafat, Bom Deus, Islamic, Animist, Judaic, Universal and New Apostolic
    counts and no Methodist, Baptist, Adventist, Evangelical, Pentecostal, Jehovah's
    Witness, Mensagem, no-religion or other-religion counts. Nothing is inferred for them:
    their municipalities are drawn on what was printed, which reaches 62.9% of Uíge's 2+
    population and 68.5% of Moxico Leste's.

    THE UNIVERSE IS THE POPULATION AGED 2 AND OVER, 34,492,888 of the census's
    36,175,745. Infants were not asked, so the missing 1.68 million are not a refusal;
    the shares are of those asked and `basis` says so.
    """
    from ao2024 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ao.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "municipality"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "ao" / "ao_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"ao.csv municipalities with no polygon: {missing} -- re-run "
                         "sources/ao_geo.py, the lookup is stale")
    if df["unit"].nunique() != 326:
        raise SystemExit(f"{df['unit'].nunique()} municipalities, expected 326")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0

    # UÍGE AND MOXICO LESTE'S TEN UNPRINTED BODIES ARE DERIVED AND SAY SO. sources/ao.py
    # `fill()` writes `tier=derived` into the note of every row it invents; the amount each
    # municipality receives is its own measured remainder, and only the SPLIT of that
    # amount between the ten is assumed. So they may not ring (allocation spreads a total,
    # it cannot establish presence) and they carry NO `roll`: neither province measures any
    # ancestor of `christianity.methodist` at the municipality, so under `inferred dots:
    # not shown` these correctly disappear rather than rolling up to something nobody
    # counted. taxonomy/ao2024.py's COLUMNS note says the same thing from the other end.
    df["tier"] = df["note"].str.contains("tier=derived", na=False).map(
        {True: "derived", False: "measured"})
    df["may_ring"] = df["tier"] == "measured"
    # `NOWHERE` and not None: an absent roll tells rollup.py to walk the tree for a measured
    # ancestor, and `measured` is a whole-country set, so the walk would find
    # `christianity.pentecostal` — measured in the nineteen provinces that printed
    # `Mensagem dos ultimos Tempos` and in neither of these two — and keep 116,174 people on
    # screen under `inferred dots: not shown`.
    from rollup import NOWHERE
    df["roll"] = df["tier"].map({"derived": NOWHERE, "measured": None})
    return df[["unit", "node", "count", "congregations", "may_ring", "tier", "roll"]]


ENTRY = {
    "ao": dict(
        name="Angola",
        source="Recenseamento Geral da População e Habitação 2024, Quadro 7 of the 21 "
               "provincial volumes (INE)",
        basis="self-identification, population aged 2 and over",
        note_public=(
            "**Angola's census names four African-founded churches one by one**, which no "
            "other census on this map does. The Tocoístas, the Kimbanguistas, Bom Deus and "
            "Josafat are 1.2 million people between them, and three of the four have a "
            "geography you can date. "
            "**Simão Toco was born at Sadi Zulumongo in Maquela do Zombo, and the census "
            "can still find it.** He founded the Igreja do Nosso Senhor Jesus Cristo no "
            "Mundo in Léopoldville in 1949 and spent most of the next twenty-five years "
            "deported or confined, first to the Azores and then to southern Angola. His "
            "church is **6.2%** of Uíge province against 1.0% nationally, reaching 22.9% "
            "in Nsosso and **11.1%** in Maquela do Zombo itself. Luanda holds more "
            "Tocoístas outright, which is the twentieth-century migration rather than the "
            "origin. "
            "**Kimbanguism arrives from the other side of the river.** Simon Kimbangu "
            "preached at Nkamba in the Belgian Congo in 1921, was arrested that September "
            "and died in a Katangan prison thirty years later; his church is **34.7%** of "
            "Lufíco and 28.1% of Nóqui, both of them on the Congo opposite Matadi, and it "
            "falls away inland as distance would predict. The border is a colonial line "
            "through one people, and the census draws how little it counts. "
            "**The Catholic south and the Protestant east are the mission map rather than "
            "a modern division.** Catholicism is 70.6% in Cunene, 69.5% in Benguela and "
            "63.8% in Huíla, above 90% in several southern municipalities, against 35.8% "
            "in Luanda and under a fifth across the Lunda east; the bare answer "
            "Protestante is 21.3% in Cuando and 20.8% in Moxico, where the Portuguese "
            "missions reached last. The Methodists are one block and almost nothing else: "
            "Bengo is 17.2% against 1.7% nationally and Quicunzo is **75.3%**, the highest "
            "single-church figure in the country, on the field the Methodist Episcopal "
            "mission opened from Luanda in 1885. "
            "**No religion is 12.1% and it is not an urban answer.** The high units are "
            "Iona (60.6%), Virei (53.3%) and Curoca (48.6%) in Namibe and Cunene, which is "
            "transhumant herding country and some of the least missionised ground in "
            "Angola; Luanda province is 14.9%, seventh of the nineteen provinces that "
            "published the cell. Beside it the animist box holds **0.13%** of the country, "
            "which is not a credible count of traditional practice here, so the two cells "
            "are worth reading together rather than apart. "
            "**Two provinces had half a table, and the missing half is filled in from the "
            "province row.** The page carrying the second eleven bodies is blank in the "
            "published Uíge and Moxico Leste volumes, so their municipalities show only "
            "what was printed plus a remainder. Both margins of that remainder are "
            "published: each municipality's own unexplained total, and the province's "
            "figure for each missing body in the national volume. Uíge's two sides agree "
            "to **0.16%** and Moxico Leste's agree exactly, so the amount each municipality "
            "receives is measured; only the split of it between the ten is assumed, and it "
            "is the same split in every municipality of a province. Those dots are marked "
            "inferred and disappear under `inferred dots: not shown`."),
        how="census, 2024, ages 2 and over",
        grain="municipalities, 106,000 people on average",
        fill="from the same census at province level, for the ten bodies Uíge's and Moxico "
             "Leste's volumes did not print",
        gap=("the 2.3% who did not answer or did not know; and children under 2, who were not "
             "asked the religion question"),
        gap_share=0.02301,
        counts=_ao_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ao" / "ao_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ao_place_weight,
        note="THE COUNTRY WAS REDRAWN BETWEEN THE FIELDWORK AND THE PUBLICATION, and that "
             "is the whole reason this map exists at 326 units. The census was collected "
             "on Lei 18/16's division of 18 provinces, 164 municipalities and 562 "
             "communes, and Lei 14/24 of 5 September 2024 replaced it with 21 provinces, "
             "326 municipalities and 378 communes while the data was being processed. INE "
             "retabulated onto the new division before publishing, so these figures exist "
             "at no other geography. "
             "NO STANDARD BOUNDARY SET HAS THAT TIER. OCHA's COD-AB is the 2018 vintage "
             "(18 / 161 / 539) and its commune layer cannot be dissolved into the new "
             "municipalities either, because its parentage is wrong where it matters: "
             "Luanda's `Belas` contains Viana and Kilamba Kiaxi, `Cazenga` contains "
             "Kikolo, and only 282 of the statute's 538 leaf units match one of its 539 "
             "communes by name. geoBoundaries represents 2006 and OpenStreetMap has "
             "admin_level 6 for Luanda's nine old municipalities and little else. The "
             "polygons are therefore a digitisation of Lei 14/24's own boundary text "
             "published as an ArcGIS feature service, and sources/ao_geo.py checks it "
             "three ways: against the statute's municipality list (326 of 326, province by "
             "province), against Quadro 9 of the census (per-province counts), and against "
             "Natural Earth for outline and area. "
             "THE UNIVERSE IS THE POPULATION AGED 2 AND OVER: 34,492,888 of the census's "
             "36,175,745. Infants were not asked, so the 1.68 million missing are not a "
             "refusal and not an undercount of any body, and the shares are of those "
             "asked. A further 767,283 people, 2.22%, answered `Não sabe/Não respondeu` "
             "and are a §3.5 residual, off the tree. "
             "TWO PROVINCES HAVE HALF A TABLE AND IT IS THE SOURCE, NOT THE PARSE. Page "
             "112 of the Uíge volume and page 86 of the Moxico Leste volume are blank "
             "pages carrying no text, no image and no drawing, and the words `Metodista` "
             "and `Adventista` appear nowhere in either volume outside its list of tables. "
             "The eleven bodies they do print cover 61.3% of Uíge's 2+ population and "
             "47.2% of Moxico Leste's. "
             "SO THE OTHER TEN ARE FILLED IN, BECAUSE OMITTING THEM MAKES THE STRONGER "
             "FALSE CLAIM. Drawn on the printed eleven alone, Uíge shows 39% short on "
             "people, its Catholic share reads 55% against a true 33.8%, and the province "
             "appears to hold no Evangelicals (347,084 of them, its second largest body) "
             "and nobody with no religion. The fill uses two PUBLISHED margins: each "
             "municipality's own unexplained remainder, off its provincial table, and the "
             "province's total for each missing body, off Quadro 7 of the national volume. "
             "They agree without being made to, which is the evidence that they are the "
             "same quantity: Moxico Leste's remainders sum to 202,180 against the national "
             "volume's 202,180 exactly, and Uíge's to 733,662 against 734,842, a ratio of "
             "0.9984. WHAT IS ASSUMED IS THE MIX AND NOTHING ELSE. With two margins and "
             "nothing inside them the maximum-entropy fill is the outer product, so the "
             "AMOUNT each municipality receives is measured and varies (Lucunga's "
             "remainder is 29.3% of its people, the city of Uíge's 44.2%) while the SPLIT "
             "of that amount between Methodist and Adventist is identical everywhere in a "
             "province. Those 320 rows are `derived`: they never ring, they carry no roll "
             "because neither province measures any ancestor of them at the municipality, "
             "and they vanish under `inferred dots: not shown`. INE has already issued an "
             "errata for Luanda and Icolo e Bengo, so a corrected volume is worth checking "
             "for; as of 2026-09-08 the publications listing has none, and no microdata. "
             "AND THE PROVINCIAL VOLUMES REVISE THE NATIONAL ONE. The national report of "
             "20 November 2025 and the provincial reports of January and February 2026 "
             "disagree: Luanda hands 144,745 people to Icolo e Bengo, which is the "
             "published errata; Cabinda gains 7,050 from Uíge; and inside the revised "
             "provinces people move into `Sem religião` and out of nearly everything else, "
             "which raises the national no-religion figure by about 311,000. The "
             "provincial volumes are the later word and are what is drawn; sources/ao.py "
             "reports both and reconciles each province against its own volume rather than "
             "against the national one. "
             "The dots are spread across 213,689 Kontur 400m hexagons weighted by hex "
             "population (sources/ao_grid.py). Angola needs that more than most: the "
             "municipalities run from 3 km² in Rangel to over 25,000 in Rivungo, the "
             "eastern and southern quarter of the country holds under 4% of its people, "
             "and the municipal boundaries run to the shoreline across the Namib.",
    ),
}
