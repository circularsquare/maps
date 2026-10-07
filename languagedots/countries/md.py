# Moldova. RPL 2024 mother tongue by UAT (sources/md_census.py), on Kontur hexes keyed to
# religiondots' 901 UAT polygons (sources/md_geo.py). The record is sources/md.md.
# Transnistria and Bender, which the 2024 census did not enumerate, are extra units PMR-* from the
# Transnistrian authorities' 2015 census of nationality (sources/md_pmr.py; sources/md.md, "Transnistria"),
# kept inside Moldova as religiondots keeps them.
from _shared import *  # noqa: F401,F403

_PMR = 408_575      # Transnistria and Bender, nationality stated (2015)


def _counts():
    import md2015_pmr
    import md2024
    df = pd.read_csv(NORM / "md.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "uat"].copy()
    if df["geo_id"].nunique() != 901:
        raise SystemExit(f"md.csv: {df['geo_id'].nunique()} UATs, expected 901")
    df["unit"] = df["geo_id"]          # the CUATM code, the polygons' own key
    df["node"] = df["source_category"].map(md2024.resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["tier"] = "measured"
    pmr = pd.read_csv(NORM / "md_pmr.csv")
    if abs(pmr["count"].sum() - _PMR) > 1:
        raise SystemExit(f"md_pmr.csv sums to {pmr['count'].sum():,.0f}, expected {_PMR:,}; "
                         "re-run sources/md_pmr.py")
    pmr["unit"] = pmr["geo_id"]
    pmr["node"] = pmr["source_category"].map(md2015_pmr.resolve)
    df = pd.concat([df[["unit", "node", "count", "tier"]], pmr[["unit", "node", "count", "tier"]]])
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Moldova",
    source="Recensământul Populaţiei şi al Locuinţelor 2024, table 5.15, mother tongue by town "
           "and commune (Biroul Naţional de Statistică); Transnistria's 2015 census, nationality "
           "by city and raion",
    how=("census, 2024, mother tongue; Transnistria and Bender from their own 2015 census of "
         "nationality, read as language"),
    parts=[
        dict(covers="Transnistria and Bender",
             source="2015 census of Transnistria, nationality by raion, read as language with "
                    "Moldova's 2024 shares",
             people=_PMR),
        dict(covers="The rest of Moldova", source="2024 census, mother tongue", rest=True),
    ],
    grain=("901 towns, communes and Chişinău sectors, 2,700 people on average; Transnistria's 6 "
           "raions and 2 cities"),
    gap=("1,582 people who gave no mother tongue, and 66,432 in Transnistria (14%) who gave no "
         "nationality"),
    view=[26.6, 45.45, 30.2, 48.5],
    # Natural Earth breakaway areas this entry now draws: not_drawn.py stops hatching them whole
    drawn_named=("Transnistria",),
    counts=_counts,
    mappings=["md2024", "md2015_pmr"],
    place=GEO / "md" / "md_plus_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Moldovan and Romanian are one language, and the census asked them apart: 1.16 million "
        "people gave Moldovan as their mother tongue and 766,000 gave Romanian. The choice says "
        "more about identity than about speech, and both are drawn as the census printed them. "
        "Mother tongue leans towards identity here too: 280,000 gave Russian, while about "
        "370,000 people aged 3 and over usually speak it. "
        "The 2024 census was not taken in Transnistria or Bender. They are drawn from the "
        "Transnistrian authorities' 2015 census of nationality, each nationality in the "
        "language shares Moldova's 2024 census found for it across the river; the 14% who gave "
        "no nationality are not drawn. Russian dominates more on the left bank, so this "
        "probably understates it."),
)
