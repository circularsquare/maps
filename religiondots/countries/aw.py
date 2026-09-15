# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


ENTRY = {
    "aw": dict(
        name="Aruba",
        source="Fifth Population and Housing Census 2010, Table P-A.5 (Central Bureau of "
               "Statistics Aruba)",
        basis="self-identification",
        view=[-70.10, 12.39, -69.84, 12.65],
        gap_share=0.00507,                      # 515/101,484; "rows only" as for nr
        gap="0.5%, whose religion was not reported",
        note_public=(
            "Three quarters of Aruba is Roman Catholic (**75.3%**). The 2010 form named eight "
            "religions and gave a line for anything else, and the census printed the "
            "write-ins as one number, so **11.7%** is drawn as other religion."),
        how="a census question, 2010",
        grain="the country as one unit, 101,500 people",
        counts=lambda: _terr_counts("aw", "aw2010", 1),
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "aw" / "aw_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_micro_place_weight("aw", "sources/terr.py"),
        note="From the census report, not UNSD's row, because UNSD labels the census's `No "
             "religion` as `Pagan`; the questionnaire (report p.270) has no such box. The "
             "ten rows sum to one person under the printed total, in the census and in UNSD.",
    ),
}
