# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


ENTRY = {
    "ai": dict(
        name="Anguilla",
        source="Census 2001 (UN Demographic Yearbook, table 28)",
        basis="self-identification",
        view=[-63.45, 18.14, -62.90, 18.62],
        gap_share=0.00341,                      # 39/11,430, exact; "rows only" as for nr
        gap="0.3%, who did not specify a religion",
        note_public=(
            "Anglicans (**29.0%**) and Methodists (**23.9%**) together are more than half of "
            "Anguilla."),
        how="a census question, 2001",
        grain="the country as one unit, 11,400 people",
        counts=lambda: _micro_counts("ai", "ai2001"),
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ai" / "ai_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_micro_place_weight("ai"),
        note="UNSD's 2001 row, exact; §11ap found no later religion table.",
    ),
}
