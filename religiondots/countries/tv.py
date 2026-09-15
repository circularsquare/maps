# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


ENTRY = {
    "tv": dict(
        name="Tuvalu",
        source="Census 2017 (UN Demographic Yearbook, table 28)",
        basis="self-identification",
        view=[175.9, -10.9, 179.95, -5.5],
        gap_share=0.001,
        gap="0.1%, who declined to answer",
        note_public=(
            "**85.9% of Tuvalu belongs to one church, which is the largest share any single "
            "church holds in any country on this map.** Ekalesia Kelisiano Tuvalu is the "
            "established church under the constitution, brought by Samoan missionaries of "
            "the London Missionary Society from 1861. **9,023 people.** "
            "**The tail is far more specific than the country's size would suggest.** A "
            "census of 10,507 people counts the Brethren, the Adventists, the Baha'is, the "
            "Assembly of God, the Witnesses, the Latter-day Saints and **53 Catholics** "
            "separately. At one dot per thousand almost none of that can draw a dot, so "
            "most of it appears as a ring instead: a mark that says the religion is present "
            "without claiming a place for it. "
            "**Irreligion is 0.25%, the lowest of any country drawn here.** Twenty-six "
            "people said they had no religion."),
        how="a census question, 2017",
        grain="the country as one unit, 10,500 people",
        counts=lambda: _micro_counts("tv", "tv2017"),
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "tv" / "tv_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_micro_place_weight("tv"),
        note="`stats.gov.tv` answers (queue.md) and was never needed. The nine atolls are "
             "one unit here; Kontur places about half the dots on Funafuti, which is where "
             "about half of Tuvalu lives.",
    ),
}
