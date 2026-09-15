# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


ENTRY = {
    # ---- THE SMALL TERRITORIES (sources/terr.py, sources/bq.py, sources/micro.py) -------
    # 2026-09-14, the buildable rows of sources.md §11ap. The microstate tier's reasoning
    # applies: at 1 dot = 1,000 people a national or island table carries no claim about
    # placement. sources/terr.md is the record for all eight.
    "vg": dict(
        name="British Virgin Islands",
        name_in="the British Virgin Islands",
        source="2010 Population and Housing Census Report, Table 77 (Government of the Virgin "
               "Islands)",
        basis="self-identification",
        view=[-64.85, 18.30, -64.25, 18.78],
        # gap_share.py reports 2.43% as "rows only" and declines to write it; Table 77 closes
        # to the person, so 683/28,054 is exact (the nr precedent).
        gap_share=0.02435,
        gap="2.4%, who did not state a religion",
        note_public=(
            "The 2010 census counted religion on each island. Methodists are the largest "
            "church at **17.6%**, ahead of the Church of God at 10.4% and the Anglicans at "
            "9.5%. Tortola holds 84% of the people and Virgin Gorda 14%, so nearly every dot is "
            "on those two; on Virgin Gorda the Church of God leads at **16.6%**, and on Jost "
            "Van Dyke 42% of its 298 people are Methodist."),
        how="a census question, 2010",
        grain="4 islands, 7,000 people on average",
        counts=lambda: _terr_counts("vg", "vg2010", 4, fold={
            "VG-COOPER-ISLAND": "VG-TORTOLA", "VG-GREAT-CAMANOE-ISLAND": "VG-TORTOLA",
            "VG-YACHTS": "VG-TORTOLA"}),
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "vg" / "vg_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_micro_place_weight("vg", "sources/terr.py"),
        note="Finer than UNSD's national row, and more exact: UNSD prints Muslim 255 where "
             "Table 77 prints 266, which is the whole of the oracle's failure to partition. "
             "Cooper Island (26), Great Camanoe (6) and people on yachts (18) carry no Kontur "
             "hex and are drawn with Tortola.",
    ),
}
