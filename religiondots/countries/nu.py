# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


ENTRY = {
    "nu": dict(
        name="Niue",
        source="Census 2017 (UN Demographic Yearbook, table 28)",
        basis="self-identification",
        view=[-170.10, -19.20, -169.72, -18.90],
        gap_share=0.051,
        gap="5.1%, who did not state a religion",
        note_public=(
            "**Niue is the smallest country on this map: 1,591 people, and it draws no dots "
            "at all.** Every religion here has fewer than a thousand followers, so at one "
            "dot per thousand people not one of them reaches a whole dot and the entire "
            "country is drawn as rings. A ring says a religion is present without claiming "
            "a place for it, which is the honest mark at this size; the counts below are "
            "the actual census figures. "
            "**Ekalesia Niue holds 61.7% of it**, 981 people. It is a London Missionary "
            "Society church planted from 1846 through Samoan and Rarotongan teachers, and "
            "the sibling of the national churches of Tuvalu and the Cook Islands. "
            "**The Latter-day Saints are 8.7%, the highest share of any country drawn "
            "here**, just ahead of the Roman Catholics at 8.4%."),
        how="a census question, 2017",
        grain="the country as one unit, 1,600 people",
        counts=lambda: _micro_counts("nu", "nu2017"),
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "nu" / "nu_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_micro_place_weight("nu"),
        note="The smallest population any country here is built from. Kontur's 2023 model "
             "puts 2,479 people on Niue against a 2017 census of 1,591, a ratio of 1.56 and "
             "the widest in this tier; on a country this small the model is coarse and it "
             "only decides which of two dots goes where.",
    ),
}
