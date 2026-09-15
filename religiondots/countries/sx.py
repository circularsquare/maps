# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


ENTRY = {
    "sx": dict(
        name="Sint Maarten",
        source="Census 2011 (UN Demographic Yearbook, table 28)",
        basis="self-identification",
        view=[-63.16, 17.99, -62.99, 18.08],
        gap_share=0.02368,                      # 796/33,609, exact; "rows only" as for nr
        gap="2.4%, whose religion was not reported",
        note_public=(
            "Roman Catholics are **33.1%** and Pentecostals 14.7%, and Hindus are **5.2%**. "
            "The census printed Islam and Judaism as one row, and Buddhism and Sikhism as "
            "another, so both are drawn with other religions."),
        how="a census question, 2011",
        grain="the country as one unit, 33,600 people",
        counts=lambda: _micro_counts("sx", "sx2011"),
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "sx" / "sx_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_micro_place_weight("sx"),
        note="UNSD's row equals the census report's Table B-10 to the person. The Dutch side "
             "only; the French Collectivity of Saint Martin publishes no religion (§11ap).",
    ),
}
