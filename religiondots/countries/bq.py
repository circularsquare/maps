# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


ENTRY = {
    "bq": dict(
        name="Caribbean Netherlands",
        name_in="the Caribbean Netherlands",
        source="Omnibus survey 2021, table 82868NED (Statistics Netherlands)",
        basis="self-identification, survey of people aged 15 and over",
        view=[-68.45, 12.00, -62.90, 17.70],
        gap_share=0.00779,                      # 216/27,726 of the island populations drawn on
        gap="0.8%, in answers Statistics Netherlands withheld as too uncertain to publish",
        note_public=(
            "These are survey shares, not a census count. Statistics Netherlands asked people "
            "aged 15 and over in 2021 and published a share for each island, applied here to "
            "each island's population on 1 January 2022. Bonaire is **60.3%** Catholic, while "
            "on Sint Eustatius Methodists (**24.8%**) and Adventists (18.9%) outnumber the "
            "Catholics (23.3%). Kontur's population grid has no cells on these islands, so "
            "dots are spread evenly over each one."),
        how="a survey, 2021, people aged 15 and over",
        grain="3 islands, 9,200 people on average",
        counts=lambda: _terr_counts("bq", "bq2021", 3, tier="modelled"),
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "bq" / "bq_islands.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        note="Survey shares (CBS Omnibus 2021, provisional) times CBS's 1 January 2022 island "
             "populations; the withheld remainder is excluded. Polygons are Natural Earth's "
             "map unit NLY split into its three parts; country_shapes.py reads the same unit "
             "through FROM_UNITS.",
    ),
}
