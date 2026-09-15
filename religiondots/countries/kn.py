# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


ENTRY = {
    "kn": dict(
        name="Saint Kitts and Nevis",
        source="Population by Religious Belief, 2011 (Department of Statistics, St Kitts and "
               "Nevis)",
        basis="self-identification",
        view=[-62.90, 17.08, -62.52, 17.43],
        gap_share=0.00119,                      # 56/47,195, exact; "rows only" as for nr
        gap="0.1%, who did not state a religion",
        note_public=(
            "Anglicans (**16.6%**) and Methodists (**15.8%**) are the two largest churches, "
            "followed by Pentecostals at 10.8% and people with no religion at 8.8%."),
        how="a census question, 2011",
        grain="the country as one unit, 47,200 people",
        counts=lambda: _terr_counts("kn", "kn2011", 1),
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "kn" / "kn_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_micro_place_weight("kn", "sources/terr.py"),
        note="Not in UNSD table 28. The office's web table, transcribed and compared cell by "
             "cell with the saved HTML. Its total of 47,195 is 797 above a census count "
             "quoted elsewhere (§11ap); not resolved.",
    ),
}
