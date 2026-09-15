# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _cl_counts():
    """INE Censo 2024 at comuna: 13 categories on 346 comunas, people aged 15 or over.

    ONE level, no allocation, nothing modelled. INE publishes these categories at this
    geography and the map draws exactly that, so every row is `measured` and may ring.

    THE UNIVERSE IS 15+ AND IT IS NOT SCALED UP TO THE WHOLE POPULATION — the call is
    argued at length in sources/cl.py. In one line: Chile's own data shows the share
    professing a religion running 96.0% at 65+ down to 63.9% at 15-29, so handing under-15s
    the 75.1% adult average would overstate religion among children by six to eleven points,
    and §14.4 forbids inventing a magnitude when the source publishes an exact one. §3.5a
    scales Pew onto American children only because the alternative there is drawing half the
    country as nothing at all; Chile has no such hole. Drawn: 81.8% of the population.

    ANTÁRTICA IS DROPPED. Comuna 12202, 60 people aged 15+, has no polygon in COD's admin3
    and would in any case put a dot near the South Pole. It draws no dot at 1:1,000 either
    way. sources/cl_geo.py names it; this is where it leaves the data.
    """
    from cl2024 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "cl.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "comuna"].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "cl" / "cl_lookup.csv",
                      dtype={"geo_id": str})
    drawn = set(lut[lut["drawn"] == 1]["geo_id"])
    missing = sorted(set(df["geo_id"]) - drawn)
    if missing != ["12202"]:
        raise SystemExit(f"expected only Antártica to lack a polygon, got {missing} -- "
                         "re-run sources/cl_geo.py, the lookup is stale")
    df = df[df["geo_id"].isin(drawn)]

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


def _cl_place_weight(place):
    """countries.py hook. `place` is the Kontur 400m hex layer scatter.py has read.

    Added 2026-09-14 after Anita found Chile's dots "a bit more artificial than i'd expect".
    Until then each comuna's dots were spread evenly over its polygon, and the median comuna
    is 630 km2: Antofagasta's 400,000 people painted across 30,000 km2 of the Atacama,
    Natales's across 49,000 km2 of icefield, and every rural comuna of the central valley an
    even speckle with no town in it. A population weight inside the comuna, not a religion
    one; no count moves (sources/cl_grid.py, sources/cl.md §7).
    """
    return _kontur_place_weight(place, "cl_hexes.gpkg", "sources/cl_grid.py")


ENTRY = {
    "cl": dict(
        name="Chile",
        source="Censo de Población y Vivienda 2024 (INE)",
        basis="self-identification, people aged 15 or over",
        # Chile's own dots run to Easter Island at 109°W, which would fit the country into a
        # sliver of ocean. Continental Chile only; Rapa Nui is drawn, just not framed.
        view=[-76.5, -56.0, -66.0, -17.3],
        gap="0.6% of the 15 and overs, who did not answer question 31",
        gap_share=0.005755,
        note_public=(
            "Chile asked about religion in 2024 for the first time since 2002 — the 2017 "
            "census was abbreviated and left it out — and wrote the question with the "
            "national religious-affairs office rather than inheriting it, so it names more "
            "bodies than most censuses this size: Jehovah's Witnesses, the Latter-day "
            "Saints, the Orthodox and the Bahá'í each have their own answer. "
            "**What the 22-year gap shows is a country changing fast.** Catholics are 53.7% "
            "of adults, against 70.0% in 2002 and 76.9% in 1992; people reporting no "
            "religion are 25.7%, against 8.3% in 2002. That is not evenly spread across "
            "ages — 96% of over-65s profess a religion and 64% of 15-29s do — so much of it "
            "is one generation replacing another. "
            "**And the map of it is as much economic as regional.** Evangelical and "
            "Protestant Chile is 16.2% nationally but is concentrated hard in the coal and "
            "forestry towns of the Biobío coast, where it is the majority faith — Los Álamos "
            "62%, Curanilahue 62%, Lota 61% — and stays high through La Araucanía. The least "
            "religious places are the wealthy eastern comunas of Santiago: Providencia is "
            "43% no-religion and Ñuñoa 40%. The most Catholic are rural Maule and the "
            "islands of Chiloé, around 80%. The small named groups sit in eastern Santiago "
            "almost by definition — a third of Chile's Jews are in Las Condes, Lo Barnechea "
            "and Vitacura — except the Muslims, whose largest single community is in "
            "Iquique. "
            "**Two things to hold while reading it.** The question was put only to people "
            "**aged 15 or over**, and nothing here scales that up to children, so this map "
            "draws 81.8% of Chile and its shares are shares of adults. And 'Evangélica o "
            "protestante' is a single cell covering 2.5 million people: Chilean "
            "Protestantism is overwhelmingly Pentecostal, and the two largest Pentecostal "
            "churches are the biggest religious bodies in the country after the Catholic "
            "Church, but no table separates them, so they are drawn as one colour."),
        how="census, 2024, ages 15 and over",
        grain="comunas, 44,000 people on average",
        counts=_cl_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "cl" / "cl_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_cl_place_weight,
        note="346 comunas is INE's ceiling for religion and the table is exact — no "
             "suppression, no blanks, every comuna's categories summing to its own 15+ "
             "total. The 15+ universe is NOT scaled up to the whole population; "
             "sources/cl.py argues why, and the short version is that Chile's religiosity "
             "gradient by age is 32 points wide, so a flat scale-up would be measurably "
             "wrong rather than merely uncertain. Antártica (60 people) has no polygon and "
             "is dropped. COD mislabels Pozo Almonte as 'Tocopilla', so the names here come "
             "from INE (sources/cl_geo.py).",
    ),
}
