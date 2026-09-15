# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _fj_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Fiji's units are ARCHIPELAGOS, not areas. Fifteen provinces over 330 islands: Lau is
    sixty-odd islands scattered across 500 km of ocean holding 10,683 people between them,
    and Cakaudrove is half of Vanua Levu plus Taveuni plus Rabi and Kioa. An equal share per
    polygon puts dots in open sea and on uninhabited islets, and weights a copra island like
    a Suva suburb (sources/fj_grid.py).
    """
    return _kontur_place_weight(place, "fj_hexes.gpkg", "sources/fj_grid.py")


def _fj_counts():
    """FBoS 2007 census Table P01-3 at province: 23 drawn categories on 15 provinces.

    ONE level, no allocation, nothing modelled -- every row is `measured` and may ring.

    `Christian` AND `Total` ARE BOTH NESTED UNIVERSES AND NEITHER IS DRAWN. The eighteen
    named denominations sum to `Christian` exactly on all fifteen provinces, so drawing the
    parent as well would double 543,588 people, 65% of the country. taxonomy/fj2007.py has
    them in EXCLUDED and sources/fj.py asserts the identity that makes it necessary.

    THE TIER IS PROVINCES BECAUSE THE CATEGORIES ARE THE POINT. SPC's PopGIS serves the same
    census at 86 tikina -- 5.7x finer -- with all eighteen Christian bodies collapsed into
    one column. That would delete Methodist, which is 34.7% of Fiji and its largest body.
    sources/fj.md §3 argues it; §9k's trade-off, decided the other way from Peru's because
    here the two really are exclusive.

    NOT STATED IS 895 PEOPLE AND IS NOT A PRINTED ROW. The table states a total of 837,271
    and prints six top-level rows summing to 836,376; the difference is carried as a §3.5
    residual and not drawn, so Fiji is 99.89% drawn.
    """
    from fj2007 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "fj.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "province"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "fj" / "fj_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"fj.csv provinces with no polygon: {missing} -- re-run "
                         "sources/fj_geo.py, the lookup is stale")
    if df["unit"].nunique() != 15:
        raise SystemExit(f"{df['unit'].nunique()} provinces, expected 15")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.groupby(["unit", "node"], as_index=False)["count"].sum()
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "fj": dict(
        name="Fiji",
        source="2007 Census of Population and Housing, Table P01-3 (Fiji Bureau of "
               "Statistics)",
        basis="self-identification, whole population",
        view=[176.7, -21.2, 180.7, -12.2],
        gap=("895 people, 0.1%, the difference between the table's own total and the rows "
             "printed under it"),
        gap_share=0.001069,
        note_public=(
            "**Fiji is the most religiously plural country in the Pacific, and the only one "
            "on this map that needs more than a Christian palette.** 64.9% Christian, "
            "**27.7% Hindu, 6.3% Muslim** — and 2,548 Sikhs, who get a printed line of their "
            "own. That is the indenture system: from 1879 the colonial government brought "
            "60,000 labourers from India to the cane districts, and their descendants are "
            "about a third of the country. "
            "**The two halves are not mixed together, they are different islands.** "
            "Methodist is **83.7% of Lau and 81.7% of Kadavu** — the outer eastern islands, "
            "where the Methodist mission arrived in the 1830s and where the chiefly system "
            "and the church grew together. Hindu is **44.3% of Macuata and 39.7% of Ba** — "
            "the sugar belt of northern Vanua Levu and western Viti Levu, which is where the "
            "cane is and where the indentured labourers were sent. You can read the "
            "plantation economy off the map. "
            "**Methodism here is bigger than in any country that invented it.** 290,555 "
            "people, **34.7% of Fiji** — the largest single body in the country and the "
            "highest Methodist share on this map anywhere. "
            "**And the census names eighteen Christian bodies**, which is why Fiji is drawn "
            "at provinces rather than at the finer tier that exists: Assembly of God is "
            "5.7% and the third-largest church, ahead of the Anglicans, Presbyterians, "
            "Baptists, Adventists and Salvationists combined; Seventh Day Adventist is 3.9%; "
            "and two of the eighteen are Fijian foundations rather than imported missions — "
            "**Christian Mission Fellowship**, started in Suva in 1990 and now planting "
            "churches in a hundred countries, and All Nations Christian Fellowship. "
            "**No religion is 0.51%**, one of the lowest figures on this map."),
        how="census, 2007, whole population",
        grain="provinces, 56,000 people on average",
        counts=_fj_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "fj" / "fj_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_fj_place_weight,
        note="THE TIER IS A TRADE-OFF AND IT WAS DECIDED THE OTHER WAY FROM PERU. Fiji "
             "publishes religion twice. FBoS's own Table P01-3 gives 15 provinces and 23 "
             "categories, eighteen of them named Christian bodies. SPC's PopGIS "
             "(fiji.popgis.spc.int) serves the SAME 2007 census at 86 tikina -- 5.7x finer, "
             "about 9,700 people a unit -- with all eighteen collapsed into one `Christians` "
             "column. Taking the finer geography would delete Methodist, which is 34.7% of "
             "Fiji and its largest body, and leave the most plural country in the Pacific "
             "looking like every other Pacific census here. So the categories won. §9k's "
             "trade-off; Peru had both and Fiji genuinely does not. sources/fj.md §3. "
             "POPGIS IS NOT WASTED: IT IS THE INDEPENDENT WITNESS, and shares no code path "
             "with a PDF typeset in 2008. It reproduces the national total, Hindu and Muslim "
             "EXACTLY and names the same 15 provinces. It differs on the Christian/no-"
             "religion boundary by 1,929 people, 0.23% of Fiji, which is a coding difference "
             "in a residual rather than a disagreement about the census. "
             "THE PRINTED TABLE DOES NOT PRINT ITS OWN RESIDUAL, AND POPGIS IS WHAT "
             "IDENTIFIES IT. Six top-level rows sum to 836,376 against a stated total of "
             "837,271; the 895 difference is unaccounted on the page. PopGIS's `other "
             "religion` equals the printed Sikh + Other religion + exactly those 895 -- "
             "which is where a residual goes, not where a religion goes. Carried as `Not "
             "stated` and not drawn (§3.5); Fiji is 99.89% drawn. "
             "`Christian` IS A NESTED UNIVERSE AND MUST NEVER BE DRAWN: the eighteen "
             "denominations sum to it exactly on all fifteen provinces, so mapping the "
             "parent too would double 543,588 people. It is EXCLUDED beside `Total`. "
             "THE JOIN IS FREE AND FOR ONCE THAT IS NOT A TRAP. COD-AB Fiji is built from "
             "FBoS's own POPGIS and carries FBOS_PID, the office's province id -- the same "
             "identifier the census tabulates on and the same one SPC's PopGIS returns. "
             "15 units, ids 1-15, names agreeing outright. "
             "FIJI IS THE FIRST COUNTRY HERE THAT STRADDLES THE ANTIMERIDIAN, AND IT BROKE "
             "TWO THINGS THAT LOOK NOTHING ALIKE. First, reprojecting the provinces to "
             "EPSG:4326 tears Cakaudrove, Lau and Macuata into 360-degree polygons -- the "
             "file opens, the count is right, the names are right, and a point-in-polygon "
             "join against it is nonsense. Second, and worse because it is upstream, NINE OF "
             "KONTUR'S OWN HEXES are stored torn across the 3857 plane, so their centroids "
             "compute to longitude ~0: this extract put six of them in the Atlantic, the "
             "Sahara and the Indian Ocean at Fiji's latitude. Projecting into a Pacific CRS "
             "does not fix either, because pyproj does not wrap longitude -- it relocates "
             "the problem. sources/fj_grid.py repairs the hexes in the tiling CRS, then does "
             "every join in degrees with negative longitudes shifted +360, and asserts the "
             "country comes out about 5 degrees wide. Seven repaired cells that would tear "
             "again on output are dropped: 385 people, 0.044% of the placement WEIGHTS and "
             "no count at all. "
             "The dots are spread across 9,945 Kontur 400m hexagons weighted by hex "
             "population, with a sixteen-year vintage gap (counts 2007, grid 2023), second "
             "only to Nicaragua's. THAT CORRELATION IS THE ONLY CHECK ON THE JOIN WITH ANY "
             "POWER and sources/fj_geo.py says so rather than inventing a weak one: fifteen "
             "units over 500 km of ocean cannot calibrate a neighbour test, so Peru's "
             "spatial-smoothness witness is deliberately NOT used here. r=0.9896 on 15 "
             "provinces against a best of 0.8884 over 2,000 random pairings. "
             "Six percent of Kontur's people fall outside every province, which looks "
             "alarming and is not: 98% of them are within 500 m of a boundary -- coastal "
             "cells just seaward of a detailed island coastline on a 400 m grid, which is "
             "what an archipelago costs.",
    ),
}
