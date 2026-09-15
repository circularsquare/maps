# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


ENTRY = {
    "fk": dict(
        name="Falkland Islands",
        name_in="the Falkland Islands",
        source="Census 2016, Table 6 (Falkland Islands Government)",
        basis="self-identification",
        view=[-61.5, -52.5, -57.5, -50.9],
        gap_share=0.06004,                      # 192/3,198, exact; "rows only" as for nr
        gap="6.0%, whose religion was not specified",
        note_public=(
            "The 2016 census printed Christians as one group, so no church is drawn "
            "separately: **57.1%** said Christian and **35.4%** no religion. With 3,198 people "
            "the islands draw only a few dots, and they are in Stanley."),
        how="a census question, 2016",
        grain="3 locations, 1,100 people on average",
        counts=lambda: _terr_counts("fk", "fk2016", 3),
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "fk" / "fk_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_micro_place_weight("fk", "sources/terr.py"),
        note="Newer than UNSD's 2006 row; the 2021 census asked no religion. Units are "
             "Stanley, Camp and the Mount Pleasant Complex. Kontur models Camp at 7.7x its "
             "381 people, which moves nothing at this scale: Camp draws no dot.",
    ),
}
