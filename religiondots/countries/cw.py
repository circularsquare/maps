# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


ENTRY = {
    "cw": dict(
        name="Curaçao",
        source="Census 2023, Table D-5 (Central Bureau of Statistics Curaçao)",
        basis="self-identification",
        view=[-69.20, 12.02, -68.72, 12.41],
        gap_share=0.02623,                      # 4,088/155,826, exact; "rows only" as for nr
        gap="2.6%, whose religion was not reported",
        note_public=(
            "Curaçao is **68.2%** Roman Catholic, and 8.7% said they have no religion. The "
            "census publishes religion only for the whole island, so the dots follow where "
            "people live and say nothing about where any religion is concentrated."),
        how="a census question, 2023",
        grain="the country as one unit, 155,800 people",
        counts=lambda: _terr_counts("cw", "cw2023", 1),
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "cw" / "cw_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_micro_place_weight("cw", "sources/terr.py"),
        note="156,000 people is above §11aa's ~100k line for one polygon, and it is one "
             "polygon anyway, because neither the 2011 nor the 2023 census tabulates religion "
             "by district; the alternative is not drawing it (sources/terr.md §4). Not in UNSD "
             "table 28.",
    ),
}
